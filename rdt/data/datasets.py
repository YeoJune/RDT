"""Data loading and preprocessing for RDT with multi-dataset support"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
import random
import numpy as np
from typing import List, Dict, Tuple, Iterator, Union, Optional
from tqdm import tqdm
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RDTDatasetBase:
    """Base class with shared functionality for RDT datasets"""
    
    def __init__(self, tokenizer_name='bert-base-uncased', max_seq_length=512,
                 total_steps=10, max_chain_length=5, visible_loss_ratio=0.15,
                 bert_masking_enabled=True, mask_prob=0.8, random_prob=0.1, keep_prob=0.1):
        """Initialize common parameters"""
        self.max_seq_length = max_seq_length
        self.total_steps = total_steps
        self.max_chain_length = max_chain_length
        self.visible_loss_ratio = visible_loss_ratio
        
        # BERT-style masking
        self.bert_masking_enabled = bert_masking_enabled
        self.mask_prob = mask_prob
        self.random_prob = random_prob
        self.keep_prob = keep_prob
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.vocab_size = self.tokenizer.vocab_size
    
    def _apply_bert_masking(self, input_ids: torch.Tensor, masked_indices: torch.Tensor, 
                           original_tokens: torch.Tensor) -> torch.Tensor:
        """Apply BERT-style masking: 80% [MASK], 10% random, 10% keep"""
        if len(masked_indices) == 0 or not self.bert_masking_enabled:
            input_ids[masked_indices] = self.mask_token_id
            return input_ids
        
        rand = torch.rand(len(masked_indices))
        
        # 80%: [MASK]
        mask_indices = masked_indices[rand < self.mask_prob]
        input_ids[mask_indices] = self.mask_token_id
        
        # 10%: random token
        random_indices = masked_indices[(rand >= self.mask_prob) & (rand < self.mask_prob + self.random_prob)]
        if len(random_indices) > 0:
            random_tokens = torch.randint(self.tokenizer.cls_token_id + 1, self.vocab_size, (len(random_indices),))
            input_ids[random_indices] = random_tokens
        
        # 10%: keep original
        return input_ids
    
    def _process_text(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process tokenized text into training sample with chain"""
        seq_len = len(tokens)
        restore_order = torch.randperm(seq_len)
        
        chain_length = random.randint(1, self.max_chain_length)
        max_start = self.total_steps
        min_start = chain_length
        start_step = random.randint(min_start, max_start)
        
        # Generate input (s_0)
        step_0 = start_step
        num_visible_0 = int(seq_len * (1 - step_0 / self.total_steps))
        visible_indices_0 = restore_order[:num_visible_0]
        masked_indices_0 = restore_order[num_visible_0:]
        
        input_ids = tokens.clone()
        if len(masked_indices_0) > 0:
            input_ids = self._apply_bert_masking(input_ids, masked_indices_0, tokens)
        
        # Generate targets (s_1, s_2, ..., s_L) and gate_targets
        targets = []
        gate_targets = []
        gate_targets.append(step_0 / self.total_steps * 20)
        
        for offset in range(1, chain_length + 1):
            step = start_step - offset
            num_visible = int(seq_len * (1 - step / self.total_steps))
            
            visible_indices = restore_order[:num_visible]
            masked_indices = restore_order[num_visible:]
            
            target_ids = tokens.clone()
            if len(masked_indices) > 0:
                target_ids[masked_indices] = self.mask_token_id
            
            targets.append(target_ids)
            gate_targets.append(step / self.total_steps * 20)
        
        # Generate loss masks
        loss_masks = []
        for i in range(chain_length):
            current_step = start_step - i
            next_step = start_step - (i + 1)
            
            current_visible = int(seq_len * (1 - current_step / self.total_steps))
            next_visible = int(seq_len * (1 - next_step / self.total_steps))
            
            delta_indices = restore_order[current_visible:next_visible]
            
            mask_bool = torch.zeros(seq_len, dtype=torch.bool)
            if len(delta_indices) > 0:
                mask_bool[delta_indices] = True
            
            if current_visible > 0 and self.visible_loss_ratio > 0:
                num_maint = max(1, int(current_visible * self.visible_loss_ratio))
                perm = torch.randperm(current_visible)[:num_maint]
                maint_indices = restore_order[perm]
                mask_bool[maint_indices] = True
            
            loss_masks.append(mask_bool)
        
        return {
            'input': input_ids,
            'targets': torch.stack(targets),
            'loss_masks': torch.stack(loss_masks),
            'gate_targets': torch.tensor(gate_targets, dtype=torch.float),
            'chain_length': chain_length
        }


class DatasetLoaderMixin:
    """Mixin for loading single or multiple datasets"""
    
    @staticmethod
    def _load_single_dataset(dataset_name: str, split: str, streaming: bool):
        """Load a single dataset by name"""
        dataset_name_lower = dataset_name.lower()
        
        if 'bookcorpus' in dataset_name_lower:
            dataset = load_dataset('rojagtap/bookcorpus', split='train', streaming=streaming)
        elif 'wikipedia' in dataset_name_lower:
            dataset = load_dataset('wikimedia/wikipedia', '20231101.en', split='train', streaming=streaming)
        else:
            if 'wikitext-2' in dataset_name_lower:
                config = 'wikitext-2-raw-v1'
            elif 'wikitext-103' in dataset_name_lower:
                config = 'wikitext-103-raw-v1'
            else:
                config = dataset_name
            
            dataset = load_dataset('wikitext', config, split=split, streaming=streaming)
        
        return dataset
    
    @staticmethod
    def _needs_split_filtering(dataset_name: str) -> bool:
        """Check if dataset needs manual split filtering"""
        dataset_name_lower = dataset_name.lower()
        return 'bookcorpus' in dataset_name_lower or 'wikipedia' in dataset_name_lower
    
    @staticmethod
    def _apply_split_filter(idx: int, split: str) -> bool:
        """Apply modulo-based split filtering"""
        mod = idx % 1000
        
        if split == 'train':
            return mod < 990
        elif split == 'validation':
            return 990 <= mod < 995
        elif split == 'test':
            return 995 <= mod < 1000
        else:
            return True


class StreamingTextDataset(IterableDataset, RDTDatasetBase, DatasetLoaderMixin):
    """Streaming dataset for on-the-fly loading with multi-dataset support"""
    
    def __init__(self, dataset_name: Union[str, List[str]] = 'wikitext-2', 
                 split='train', dataset_probabilities: Optional[List[float]] = None,
                 max_val_samples: int = 5000, max_test_samples: int = 10000, **kwargs):
        super().__init__(**kwargs)
        
        if isinstance(dataset_name, str):
            self.dataset_names = [dataset_name]
        else:
            self.dataset_names = dataset_name
        
        self.split = split
        self.is_multi_dataset = len(self.dataset_names) > 1
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        
        self.needs_filtering = [self._needs_split_filtering(name) for name in self.dataset_names]
        self.any_needs_filtering = any(self.needs_filtering)
        
        print(f"Loading dataset(s) in streaming mode (split={split})...")
        if split == 'validation':
            print(f"  Validation limited to {max_val_samples} samples")
        elif split == 'test':
            print(f"  Test limited to {max_test_samples} samples")
        
        if self.is_multi_dataset:
            print(f"  Mixed {len(self.dataset_names)} datasets:")
            datasets = []
            
            for name in self.dataset_names:
                ds = self._load_single_dataset(name, split, streaming=True)
                datasets.append(ds)
            
            if dataset_probabilities is None:
                # Calculate exact probabilities from validation set
                print("  Calculating probabilities from validation set sizes...")
                
                val_sizes = []
                for name in self.dataset_names:
                    val_dataset = self._load_single_dataset(name, 'validation', streaming=False)
                    
                    if self._needs_split_filtering(name):
                        val_dataset = self._split_dataset_for_size_calc(val_dataset, 'validation')
                    
                    val_size = self._tokenize_for_size_calc(val_dataset, name)
                    val_sizes.append(val_size)
                    
                    print(f"    - {name}: {val_size} validation samples")
                
                total_val = sum(val_sizes)
                self.probabilities = [size / total_val for size in val_sizes]
                
                print("  Calculated probabilities (from validation ratio):")
                for name, prob, size in zip(self.dataset_names, self.probabilities, val_sizes):
                    print(f"    - {name}: {prob*100:.1f}% ({size} val samples)")
            else:
                if len(dataset_probabilities) != len(self.dataset_names):
                    raise ValueError(f"Number of probabilities ({len(dataset_probabilities)}) must match "
                                   f"number of datasets ({len(self.dataset_names)})")
                prob_sum = sum(dataset_probabilities)
                self.probabilities = [p / prob_sum for p in dataset_probabilities]
                
                for name, prob in zip(self.dataset_names, self.probabilities):
                    print(f"    - {name}: {prob*100:.1f}%")
            
            datasets_with_index = []
            for dataset_idx, ds in enumerate(datasets):
                def add_dataset_index(example, idx=dataset_idx):
                    example['__dataset_index__'] = idx
                    return example
                
                ds_with_index = ds.map(add_dataset_index)
                datasets_with_index.append(ds_with_index)
            
            self.dataset = interleave_datasets(
                datasets_with_index,
                probabilities=self.probabilities,
                seed=42,
                stopping_strategy='all_exhausted'
            )
        else:
            self.dataset = self._load_single_dataset(self.dataset_names[0], split, streaming=True)
            print(f"  Loaded: {self.dataset_names[0]}")
        
        print("Dataset(s) loaded in streaming mode")
    
    def _split_dataset_for_size_calc(self, dataset, split: str):
        """Split dataset for size calculation"""
        total_size = len(dataset)
        train_size = int(0.99 * total_size)
        val_size = int(0.005 * total_size)
        test_size = total_size - train_size - val_size
        
        splits = dataset.train_test_split(test_size=val_size + test_size, seed=42)
        train_data = splits['train']
        remaining = splits['test']
        
        val_test_splits = remaining.train_test_split(test_size=test_size, seed=42)
        val_data = val_test_splits['train']
        test_data = val_test_splits['test']
        
        if split == 'train':
            return train_data
        elif split == 'validation':
            return val_data
        elif split == 'test':
            return test_data
        else:
            raise ValueError(f"Invalid split: {split}")
    
    def _tokenize_for_size_calc(self, dataset, dataset_name: str) -> int:
        """Tokenize dataset FULLY to get exact sample count"""
        count = 0
        for item in tqdm(dataset, desc=f"Calculating size for {dataset_name}"):
            text = item['text'].strip()
            if len(text) == 0:
                continue
            
            encoded = self.tokenizer(
                text,
                max_length=self.max_seq_length,
                truncation=True,
                padding=False,
                return_overflowing_tokens=True,
                stride=0
            )
            
            for input_ids in encoded['input_ids']:
                if len(input_ids) >= 10:
                    count += 1
        
        return count
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over streaming dataset with multi-worker support"""
        worker_info = torch.utils.data.get_worker_info()
        dataset_counters = {i: 0 for i in range(len(self.dataset_names))}
        
        yielded_samples = 0
        max_samples = None
        if self.split == 'validation':
            max_samples = self.max_val_samples
            if worker_info is not None:
                max_samples = max_samples // worker_info.num_workers
        elif self.split == 'test':
            max_samples = self.max_test_samples
            if worker_info is not None:
                max_samples = max_samples // worker_info.num_workers
        
        for idx, item in enumerate(self.dataset):
            if worker_info is not None:
                if idx % worker_info.num_workers != worker_info.id:
                    continue
            
            if max_samples is not None and yielded_samples >= max_samples:
                return
            
            should_include = True
            if self.any_needs_filtering:
                if self.is_multi_dataset:
                    dataset_idx = item.get('__dataset_index__', 0)
                    
                    if self.needs_filtering[dataset_idx]:
                        sample_idx = dataset_counters[dataset_idx]
                        dataset_counters[dataset_idx] += 1
                        
                        if not self._apply_split_filter(sample_idx, self.split):
                            should_include = False
                else:
                    if self.needs_filtering[0]:
                        if not self._apply_split_filter(idx, self.split):
                            should_include = False
            
            if not should_include:
                continue
            
            text = item.get('text', '').strip()
            if len(text) == 0:
                continue
            
            encoded = self.tokenizer(
                text,
                max_length=self.max_seq_length,
                truncation=True,
                padding=False,
                return_overflowing_tokens=True,
                stride=0
            )
            
            for token_ids in encoded['input_ids']:
                tokens = torch.tensor(token_ids, dtype=torch.long)
                if len(tokens) < 10:
                    continue
                
                if max_samples is not None and yielded_samples >= max_samples:
                    return
                
                yielded_samples += 1
                yield self._process_text(tokens)


class WikiTextDataset(Dataset, RDTDatasetBase, DatasetLoaderMixin):
    """Map-style dataset for preloading data with multi-dataset support"""
    
    def __init__(self, dataset_name: Union[str, List[str]] = 'wikitext-2', 
                 split='train', samples_per_text=1, streaming=False,
                 dataset_probabilities: Optional[List[float]] = None,
                 max_val_samples: int = 5000, max_test_samples: int = 10000, **kwargs):
        super().__init__(**kwargs)
        
        if streaming:
            raise ValueError("Use StreamingTextDataset for streaming mode instead")
        
        self.split = split
        self.samples_per_text = samples_per_text
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        
        if isinstance(dataset_name, str):
            self.dataset_names = [dataset_name]
        else:
            self.dataset_names = dataset_name
        
        self.is_multi_dataset = len(self.dataset_names) > 1
        
        print(f"Loading dataset(s) (split={split})...")
        
        if self.is_multi_dataset:
            print(f"  Loading {len(self.dataset_names)} datasets:")
            all_tokenized = []
            dataset_sizes = []
            
            for name in self.dataset_names:
                dataset = self._load_single_dataset(name, split, streaming=False)
                
                if 'bookcorpus' in name.lower() or 'wikipedia' in name.lower():
                    dataset = self._split_dataset(dataset, split)
                
                tokenized = self._prepare_data_from_dataset(dataset, name)
                all_tokenized.extend(tokenized)
                dataset_sizes.append(len(tokenized))
                print(f"    - {name}: {len(tokenized)} sequences")
            
            total_size = sum(dataset_sizes)
            print(f"  Actual proportions:")
            for name, size in zip(self.dataset_names, dataset_sizes):
                proportion = size / total_size * 100
                print(f"    - {name}: {proportion:.1f}%")
            
            if dataset_probabilities is not None:
                print("  Warning: dataset_probabilities is ignored for map-style datasets.")
                print("  Samples are naturally proportional to dataset sizes.")
            
            self.tokenized_data = all_tokenized
            print(f"  Total: {len(self.tokenized_data)} sequences")
            
        else:
            name = self.dataset_names[0]
            dataset = self._load_single_dataset(name, split, streaming=False)
            
            if 'bookcorpus' in name.lower() or 'wikipedia' in name.lower():
                dataset = self._split_dataset(dataset, split)
            
            self.tokenized_data = self._prepare_data_from_dataset(dataset, name)
            print(f"  Loaded: {name} ({len(self.tokenized_data)} sequences)")
        
        print(f"Total samples (with samples_per_text={samples_per_text}): "
              f"{len(self.tokenized_data) * samples_per_text}")
    
    def _split_dataset(self, dataset, split: str):
        """Split datasets that only have train split"""
        total_size = len(dataset)
        train_size = int(0.99 * total_size)
        val_size = int(0.005 * total_size)
        test_size = total_size - train_size - val_size
        
        splits = dataset.train_test_split(test_size=val_size + test_size, seed=42)
        train_data = splits['train']
        remaining = splits['test']
        
        val_test_splits = remaining.train_test_split(test_size=test_size, seed=42)
        val_data = val_test_splits['train']
        test_data = val_test_splits['test']
        
        if split == 'train':
            return train_data
        elif split == 'validation':
            return val_data
        elif split == 'test':
            return test_data
        else:
            raise ValueError(f"Invalid split: {split}")
    
    def _prepare_data_from_dataset(self, dataset, dataset_name: str) -> List[torch.Tensor]:
        """Tokenize dataset and split into chunks"""
        tokenizer_name = self.tokenizer.name_or_path
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        max_samples = None
        if self.split == 'validation':
            max_samples = self.max_val_samples
        elif self.split == 'test':
            max_samples = self.max_test_samples
        
        tokenized = []
        for item in tqdm(dataset, desc=f"Tokenizing {dataset_name}"):
            if max_samples is not None and len(tokenized) >= max_samples:
                break
            
            text = item['text'].strip()
            if len(text) == 0:
                continue
            
            encoded = tokenizer(
                text,
                max_length=self.max_seq_length,
                truncation=True,
                padding=False,
                return_overflowing_tokens=True,
                stride=0
            )
            
            for input_ids in encoded['input_ids']:
                tokens = torch.tensor(input_ids, dtype=torch.long)
                if len(tokens) >= 10:
                    tokenized.append(tokens)
                    
                    if max_samples is not None and len(tokenized) >= max_samples:
                        break
        
        return tokenized
    
    def __len__(self):
        return len(self.tokenized_data) * self.samples_per_text
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text_idx = idx // self.samples_per_text
        tokens = self.tokenized_data[text_idx]
        return self._process_text(tokens)


def collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """Collate function for RDT batches"""
    max_seq_len = max(item['input'].size(0) for item in batch)
    max_chain_len = max(item['chain_length'] for item in batch)
    batch_size = len(batch)
    
    inputs = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
    targets = torch.full((batch_size, max_chain_len, max_seq_len), pad_token_id, dtype=torch.long)
    loss_masks = torch.zeros((batch_size, max_chain_len, max_seq_len), dtype=torch.bool)
    gate_targets = torch.zeros((batch_size, max_chain_len + 1), dtype=torch.float)
    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    chain_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, item in enumerate(batch):
        seq_len = item['input'].size(0)
        chain_len = item['chain_length']
        
        inputs[i, :seq_len] = item['input']
        targets[i, :chain_len, :seq_len] = item['targets']
        loss_masks[i, :chain_len, :seq_len] = item['loss_masks']
        gate_targets[i, :chain_len + 1] = item['gate_targets']
        attention_mask[i, :seq_len] = 1
        chain_lengths[i] = chain_len
    
    return {
        'input': inputs,
        'targets': targets,
        'loss_masks': loss_masks,
        'gate_targets': gate_targets,
        'attention_mask': attention_mask,
        'chain_lengths': chain_lengths
    }


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders for RDT training"""
    data_config = config['data']
    training_config = config['training']
    streaming = data_config.get('streaming', False)
    
    bert_config = training_config.get('bert_masking', {})
    bert_masking_enabled = bert_config.get('enabled', True)
    mask_prob = bert_config.get('mask_prob', 0.8)
    random_prob = bert_config.get('random_prob', 0.1)
    keep_prob = bert_config.get('keep_prob', 0.1)
    
    dataset_name = data_config['dataset_name']
    dataset_probabilities = data_config.get('dataset_probabilities', None)
    
    max_val_samples = data_config.get('max_val_samples', 5000)
    max_test_samples = data_config.get('max_test_samples', 10000)
    
    common_kwargs = {
        'tokenizer_name': data_config['tokenizer_name'],
        'max_seq_length': data_config['max_seq_length'],
        'total_steps': training_config['total_steps'],
        'max_chain_length': training_config['max_chain_length'],
        'visible_loss_ratio': training_config.get('visible_loss_ratio', 0.15),
        'bert_masking_enabled': bert_masking_enabled,
        'mask_prob': mask_prob,
        'random_prob': random_prob,
        'keep_prob': keep_prob,
        'dataset_probabilities': dataset_probabilities
    }
    
    if streaming:
        print("\nDataset mode: Train=Streaming, Val/Test=Map-style (standard)")
        
        train_dataset = StreamingTextDataset(
            dataset_name=dataset_name,
            split='train',
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
            **common_kwargs
        )
        
        val_dataset = WikiTextDataset(
            dataset_name=dataset_name,
            split='validation',
            samples_per_text=1,
            streaming=False,
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
            **common_kwargs
        )
        
        pad_token_id = train_dataset.tokenizer.pad_token_id or 0
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config['batch_size'],
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory'],
            collate_fn=lambda batch: collate_fn(batch, pad_token_id)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory'],
            collate_fn=lambda batch: collate_fn(batch, pad_token_id)
        )
        
    else:
        train_dataset = WikiTextDataset(
            dataset_name=dataset_name,
            split='train',
            samples_per_text=data_config.get('samples_per_text', 1),
            streaming=False,
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
            **common_kwargs
        )
        
        val_dataset = WikiTextDataset(
            dataset_name=dataset_name,
            split='validation',
            samples_per_text=data_config.get('samples_per_text', 1),
            streaming=False,
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
            **common_kwargs
        )
        
        pad_token_id = train_dataset.tokenizer.pad_token_id or 0
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config['batch_size'],
            shuffle=True,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory'],
            collate_fn=lambda batch: collate_fn(batch, pad_token_id)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory'],
            collate_fn=lambda batch: collate_fn(batch, pad_token_id)
        )
    
    return train_loader, val_loader


def create_mlm_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders for MLM training (BERT, RoBERTa, etc.)"""
    from .collators import get_collator
    
    data_config = config['data']
    training_config = config['training']
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(data_config['tokenizer_name'])
    
    # Load datasets
    dataset_name = data_config['dataset_name']
    if 'wikitext-2' in dataset_name.lower():
        dataset_config = 'wikitext-2-raw-v1'
    elif 'wikitext-103' in dataset_name.lower():
        dataset_config = 'wikitext-103-raw-v1'
    else:
        dataset_config = dataset_name
    
    print(f"Loading {dataset_name}...")
    train_dataset = load_dataset('wikitext', dataset_config, split='train')
    val_dataset = load_dataset('wikitext', dataset_config, split='validation')
    
    # Text chunking
    max_seq_length = data_config['max_seq_length']
    overlap = data_config.get('chunk_overlap', 0)
    min_length = data_config.get('min_text_length', 50)
    
    def chunk_and_tokenize(examples):
        """
        RDT 방식과 동일하게 Tokenizer의 기능을 사용하여 Chunking.
        - return_overflowing_tokens=True: 긴 텍스트를 자동으로 여러 시퀀스로 분할
        - stride: 겹치는 구간 설정
        - truncation: max_seq_length에 맞춰 자름 (특수 토큰 공간 확보 포함)
        """
        outputs = tokenizer(
            examples["text"],
            max_length=max_seq_length,
            truncation=True,
            stride=overlap,  # config의 chunk_overlap 적용
            return_overflowing_tokens=True,
            return_attention_mask=True,
            padding=False  # 패딩은 Collator에서 수행
        )
        
        # 'overflow_to_sample_mapping' 키는 map 함수에서 반환 시 
        # 샘플 개수 불일치 에러를 유발하므로 제거해야 함
        if "overflow_to_sample_mapping" in outputs:
            outputs.pop("overflow_to_sample_mapping")
            
        return outputs
    
    # Tokenize and chunk
    print("Tokenizing and chunking train data...")
    train_dataset = train_dataset.map(
        chunk_and_tokenize,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train"
    )
    train_dataset = train_dataset.filter(lambda x: len(x['input_ids']) > 0)
    
    print("Tokenizing and chunking validation data...")
    val_dataset = val_dataset.map(
        chunk_and_tokenize,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation"
    )
    val_dataset = val_dataset.filter(lambda x: len(x['input_ids']) > 0)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create collator
    collator = get_collator(
        model_type='mlm',
        tokenizer=tokenizer,
        mlm_probability=data_config.get('mlm_probability', 0.15)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        collate_fn=collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        collate_fn=collator
    )
    
    return train_loader, val_loader


def create_cmlm_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders for CMLM training.
    Returns original tokens without pre-masking (masking done on-the-fly).
    """
    from .collators import CMLMCollator
    
    data_config = config["data"]
    training_config = config["training"]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(data_config["tokenizer_name"])
    
    # Load datasets
    dataset_name = data_config["dataset_name"]
    if "wikitext-2" in dataset_name.lower():
        dataset_config = "wikitext-2-raw-v1"
    elif "wikitext-103" in dataset_name.lower():
        dataset_config = "wikitext-103-raw-v1"
    else:
        dataset_config = dataset_name
    
    print(f"Loading {dataset_name}...")
    train_dataset = load_dataset("wikitext", dataset_config, split="train")
    val_dataset = load_dataset("wikitext", dataset_config, split="validation")
    
    # Text chunking (same as MLM, but no masking)
    max_seq_length = data_config["max_seq_length"]
    overlap = data_config.get("chunk_overlap", 0)
    min_length = data_config.get("min_text_length", 50)
    
    def chunk_and_tokenize(examples):
        """
        RDT 방식과 동일하게 Tokenizer의 기능을 사용하여 Chunking.
        - return_overflowing_tokens=True: 긴 텍스트를 자동으로 여러 시퀀스로 분할
        - stride: 겹치는 구간 설정
        - truncation: max_seq_length에 맞춰 자름 (특수 토큰 공간 확보 포함)
        """
        outputs = tokenizer(
            examples["text"],
            max_length=max_seq_length,
            truncation=True,
            stride=overlap,  # config의 chunk_overlap 적용
            return_overflowing_tokens=True,
            return_attention_mask=True,
            padding=False  # 패딩은 Collator에서 수행
        )
        
        # 'overflow_to_sample_mapping' 키는 map 함수에서 반환 시 
        # 샘플 개수 불일치 에러를 유발하므로 제거해야 함
        if "overflow_to_sample_mapping" in outputs:
            outputs.pop("overflow_to_sample_mapping")
            
        return outputs
    
    # Tokenize and chunk
    print("Tokenizing and chunking train data...")
    train_dataset = train_dataset.map(
        chunk_and_tokenize,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train"
    )
    train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    
    print("Tokenizing and chunking validation data...")
    val_dataset = val_dataset.map(
        chunk_and_tokenize,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation"
    )
    val_dataset = val_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create collator (no masking, just padding)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collator = CMLMCollator(pad_token_id=pad_token_id)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True),
        collate_fn=collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True),
        collate_fn=collator
    )
    
    return train_loader, val_loader

def create_mdlm_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders for MDLM training.
    
    Similar to CMLM: returns original tokens without pre-masking.
    Masking is done on-the-fly using continuous_time_masking().
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_loader, val_loader: DataLoaders for training and validation
    """
    from .collators import MDLMCollator
    
    data_config = config["data"]
    training_config = config["training"]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(data_config["tokenizer_name"])
    
    # Load datasets
    dataset_name = data_config["dataset_name"]
    if "wikitext-2" in dataset_name.lower():
        dataset_config = "wikitext-2-raw-v1"
    elif "wikitext-103" in dataset_name.lower():
        dataset_config = "wikitext-103-raw-v1"
    else:
        dataset_config = dataset_name
    
    print(f"Loading {dataset_name}...")
    train_dataset = load_dataset("wikitext", dataset_config, split="train")
    val_dataset = load_dataset("wikitext", dataset_config, split="validation")
    
    # Text chunking (same as CMLM, no masking)
    max_seq_length = data_config["max_seq_length"]
    overlap = data_config.get("chunk_overlap", 0)
    min_length = data_config.get("min_text_length", 50)
    
    def chunk_and_tokenize(examples):
        """
        RDT 방식과 동일하게 Tokenizer의 기능을 사용하여 Chunking.
        - return_overflowing_tokens=True: 긴 텍스트를 자동으로 여러 시퀀스로 분할
        - stride: 겹치는 구간 설정
        - truncation: max_seq_length에 맞춰 자름 (특수 토큰 공간 확보 포함)
        """
        outputs = tokenizer(
            examples["text"],
            max_length=max_seq_length,
            truncation=True,
            stride=overlap,  # config의 chunk_overlap 적용
            return_overflowing_tokens=True,
            return_attention_mask=True,
            padding=False  # 패딩은 Collator에서 수행
        )
        
        # 'overflow_to_sample_mapping' 키는 map 함수에서 반환 시 
        # 샘플 개수 불일치 에러를 유발하므로 제거해야 함
        if "overflow_to_sample_mapping" in outputs:
            outputs.pop("overflow_to_sample_mapping")
            
        return outputs
    
    # Tokenize and chunk
    print("Tokenizing and chunking train data...")
    train_dataset = train_dataset.map(
        chunk_and_tokenize,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train"
    )
    train_dataset = train_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    
    print("Tokenizing and chunking validation data...")
    val_dataset = val_dataset.map(
        chunk_and_tokenize,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation"
    )
    val_dataset = val_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create collator (no masking, just padding - same as CMLM)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collator = MDLMCollator(pad_token_id=pad_token_id)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True),
        collate_fn=collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True),
        collate_fn=collator
    )
    
    return train_loader, val_loader