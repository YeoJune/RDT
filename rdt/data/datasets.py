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
        gate_targets.append(step_0 / self.total_steps * 20)  # s_0의 gate
        
        # s_1 ~ s_L 생성
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
            
            # Delta: 새로 드러나는 위치
            delta_indices = restore_order[current_visible:next_visible]
            
            mask_bool = torch.zeros(seq_len, dtype=torch.bool)
            if len(delta_indices) > 0:
                mask_bool[delta_indices] = True
            
            # Maintenance: 이미 드러난 것 중 일부
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
        """
        Load a single dataset by name.
        
        Note: For BookCorpus and Wikipedia, always loads 'train' split.
        Actual split filtering is done in __iter__ using modulo-based approach.
        """
        dataset_name_lower = dataset_name.lower()
        
        if 'bookcorpus' in dataset_name_lower:
            # BookCorpus only has train split - split filtering done in __iter__
            dataset = load_dataset('rojagtap/bookcorpus', split='train', streaming=streaming)
            
        elif 'wikipedia' in dataset_name_lower:
            # Wikipedia only has train split - split filtering done in __iter__
            dataset = load_dataset('wikimedia/wikipedia', '20231101.en', split='train', streaming=streaming)
            
        else:
            # WikiText datasets (have train/validation/test splits)
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
        """Check if dataset needs manual split filtering (BookCorpus, Wikipedia)"""
        dataset_name_lower = dataset_name.lower()
        return 'bookcorpus' in dataset_name_lower or 'wikipedia' in dataset_name_lower
    
    @staticmethod
    def _apply_split_filter(idx: int, split: str) -> bool:
        """
        Apply modulo-based split filtering for datasets without built-in splits.
        
        Split ratio: 99% train, 0.5% validation, 0.5% test
        Uses modulo 1000 for large datasets to ensure validation/test sets 
        are kept manageable in size.
        
        Returns:
            True if this sample should be included in the split, False otherwise
        """
        # Modulo-based split: 99% train, 0.5% val, 0.5% test
        mod = idx % 1000
        
        if split == 'train':
            return mod < 990  # Indices 0-989 (99%)
        elif split == 'validation':
            return 990 <= mod < 995  # Indices 990-994 (0.5%)
        elif split == 'test':
            return 995 <= mod < 1000  # Indices 995-999 (0.5%)
        else:
            return True


class StreamingTextDataset(IterableDataset, RDTDatasetBase, DatasetLoaderMixin):
    """Streaming dataset for on-the-fly loading with multi-dataset support"""
    
    def __init__(self, dataset_name: Union[str, List[str]] = 'wikitext-2', 
                 split='train', dataset_probabilities: Optional[List[float]] = None,
                 max_val_samples: int = 5000, max_test_samples: int = 10000, **kwargs):
        super().__init__(**kwargs)
        
        # Normalize to list
        if isinstance(dataset_name, str):
            self.dataset_names = [dataset_name]
        else:
            self.dataset_names = dataset_name
        
        self.split = split
        self.is_multi_dataset = len(self.dataset_names) > 1
        
        # Sample limits for validation/test (streaming datasets only)
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        
        # Track which datasets need split filtering
        self.needs_filtering = [self._needs_split_filtering(name) for name in self.dataset_names]
        self.any_needs_filtering = any(self.needs_filtering)
        
        print(f"Loading dataset(s) in streaming mode (split={split})...")
        if split == 'validation':
            print(f"  Validation limited to {max_val_samples} samples")
        elif split == 'test':
            print(f"  Test limited to {max_test_samples} samples")
        
        if self.is_multi_dataset:
            # Load multiple datasets and interleave
            print(f"  Mixed {len(self.dataset_names)} datasets:")
            datasets = []
            
            # Load datasets first
            for name in self.dataset_names:
                ds = self._load_single_dataset(name, split, streaming=True)
                datasets.append(ds)
            
            # Calculate probabilities based on dataset sizes if not provided
            if dataset_probabilities is None:
                # For streaming datasets, use interleave_datasets default behavior
                # which samples proportionally to dataset sizes
                self.probabilities = None
                print("  Using dataset-size proportional sampling (default)")
            else:
                # Normalize provided probabilities
                if len(dataset_probabilities) != len(self.dataset_names):
                    raise ValueError(f"Number of probabilities ({len(dataset_probabilities)}) must match "
                                   f"number of datasets ({len(self.dataset_names)})")
                prob_sum = sum(dataset_probabilities)
                self.probabilities = [p / prob_sum for p in dataset_probabilities]
                
                for name, prob in zip(self.dataset_names, self.probabilities):
                    print(f"    - {name}: {prob*100:.1f}%")
            
            # Interleave datasets
            # IMPORTANT: When interleaving, we lose track of which dataset each sample came from
            # So we need to add dataset index to each sample
            datasets_with_index = []
            for dataset_idx, ds in enumerate(datasets):
                # Add dataset index to each sample
                def add_dataset_index(example, idx=dataset_idx):
                    example['__dataset_index__'] = idx
                    return example
                
                ds_with_index = ds.map(add_dataset_index)
                datasets_with_index.append(ds_with_index)
            
            self.dataset = interleave_datasets(
                datasets_with_index,
                probabilities=self.probabilities,  # None = proportional to size
                seed=42,
                stopping_strategy='all_exhausted'
            )
        else:
            # Single dataset
            self.dataset = self._load_single_dataset(self.dataset_names[0], split, streaming=True)
            print(f"  Loaded: {self.dataset_names[0]}")
        
        print("Dataset(s) loaded in streaming mode")
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over streaming dataset with multi-worker support"""
        
        # Multi-worker support: prevent data duplication
        worker_info = torch.utils.data.get_worker_info()
        
        # Track indices per dataset for split filtering
        dataset_counters = {i: 0 for i in range(len(self.dataset_names))}
        
        # Track yielded samples for validation/test limits
        # Adjust max_samples for num_workers to get correct total
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
            # Multi-worker handling: each worker processes different subset
            if worker_info is not None:
                if idx % worker_info.num_workers != worker_info.id:
                    continue
            
            # For validation/test with max_samples, stop early
            if max_samples is not None and yielded_samples >= max_samples:
                return
            
            # For datasets without built-in splits, apply manual split filtering
            should_include = True
            if self.any_needs_filtering:
                if self.is_multi_dataset:
                    # Multi-dataset mode: check which dataset this sample came from
                    dataset_idx = item.get('__dataset_index__', 0)
                    
                    # Only filter if this specific dataset needs filtering
                    if self.needs_filtering[dataset_idx]:
                        # Get counter for this specific dataset
                        sample_idx = dataset_counters[dataset_idx]
                        dataset_counters[dataset_idx] += 1
                        
                        # Apply split filter
                        if not self._apply_split_filter(sample_idx, self.split):
                            should_include = False
                else:
                    # Single-dataset mode
                    if self.needs_filtering[0]:
                        if not self._apply_split_filter(idx, self.split):
                            should_include = False
            
            if not should_include:
                continue
            
            # Process text
            text = item.get('text', '').strip()
            if len(text) == 0:
                continue
            
            # Tokenize with chunking
            encoded = self.tokenizer(
                text,
                max_length=self.max_seq_length,
                truncation=True,
                padding=False,
                return_overflowing_tokens=True,
                stride=0
            )
            
            # Yield each chunk as a separate sample
            for token_ids in encoded['input_ids']:
                tokens = torch.tensor(token_ids, dtype=torch.long)
                if len(tokens) < 10:
                    continue
                
                # Check limit again before yielding
                if max_samples is not None and yielded_samples >= max_samples:
                    return
                
                yielded_samples += 1
                yield self._process_text(tokens)


class WikiTextDataset(Dataset, RDTDatasetBase, DatasetLoaderMixin):
    """Map-style dataset for preloading data with multi-dataset support"""
    
    def __init__(self, dataset_name: Union[str, List[str]] = 'wikitext-2', 
                 split='train', samples_per_text=1, streaming=False,
                 dataset_probabilities: Optional[List[float]] = None, **kwargs):
        super().__init__(**kwargs)
        
        if streaming:
            raise ValueError("Use StreamingTextDataset for streaming mode instead")
        
        self.split = split
        self.samples_per_text = samples_per_text
        
        # Normalize to list
        if isinstance(dataset_name, str):
            self.dataset_names = [dataset_name]
        else:
            self.dataset_names = dataset_name
        
        self.is_multi_dataset = len(self.dataset_names) > 1
        
        print(f"Loading dataset(s) (split={split})...")
        
        if self.is_multi_dataset:
            # Load and combine multiple datasets
            print(f"  Loading {len(self.dataset_names)} datasets:")
            all_tokenized = []
            dataset_sizes = []
            
            for name in self.dataset_names:
                dataset = self._load_single_dataset(name, split, streaming=False)
                
                # Handle datasets without built-in splits
                if 'bookcorpus' in name.lower() or 'wikipedia' in name.lower():
                    dataset = self._split_dataset(dataset, split)
                
                tokenized = self._prepare_data_from_dataset(dataset, name)
                all_tokenized.extend(tokenized)
                dataset_sizes.append(len(tokenized))
                print(f"    - {name}: {len(tokenized)} sequences")
            
            # Show actual proportions
            total_size = sum(dataset_sizes)
            print(f"  Actual proportions:")
            for name, size in zip(self.dataset_names, dataset_sizes):
                proportion = size / total_size * 100
                print(f"    - {name}: {proportion:.1f}%")
            
            # Note: For map-style, we just concatenate all data
            # The actual sampling will be uniform across all samples
            # (which naturally gives size-proportional sampling from each dataset)
            if dataset_probabilities is not None:
                print("  Warning: dataset_probabilities is ignored for map-style datasets.")
                print("  Samples are naturally proportional to dataset sizes.")
            
            self.tokenized_data = all_tokenized
            print(f"  Total: {len(self.tokenized_data)} sequences")
            
        else:
            # Single dataset
            name = self.dataset_names[0]
            dataset = self._load_single_dataset(name, split, streaming=False)
            
            # Handle datasets without built-in splits
            if 'bookcorpus' in name.lower() or 'wikipedia' in name.lower():
                dataset = self._split_dataset(dataset, split)
            
            self.tokenized_data = self._prepare_data_from_dataset(dataset, name)
            print(f"  Loaded: {name} ({len(self.tokenized_data)} sequences)")
        
        print(f"Total samples (with samples_per_text={samples_per_text}): "
              f"{len(self.tokenized_data) * samples_per_text}")
    
    def _split_dataset(self, dataset, split: str):
        """
        Split datasets that only have train split (BookCorpus, Wikipedia).
        Uses 99% train, 0.5% validation, 0.5% test split to match streaming behavior.
        """
        total_size = len(dataset)
        train_size = int(0.99 * total_size)
        val_size = int(0.005 * total_size)
        test_size = total_size - train_size - val_size
        
        # First split: train vs (val + test)
        splits = dataset.train_test_split(test_size=val_size + test_size, seed=42)
        train_data = splits['train']
        remaining = splits['test']
        
        # Second split: val vs test  
        val_test_splits = remaining.train_test_split(test_size=test_size, seed=42)
        val_data = val_test_splits['train']
        test_data = val_test_splits['test']
        
        # Return requested split
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
        
        tokenized = []
        for item in tqdm(dataset, desc=f"Tokenizing {dataset_name}"):
            text = item['text'].strip()
            if len(text) == 0:
                continue
            
            # Tokenize with chunking
            encoded = tokenizer(
                text,
                max_length=self.max_seq_length,
                truncation=True,
                padding=False,
                return_overflowing_tokens=True,
                stride=0
            )
            
            # Each chunk as a separate sample
            for input_ids in encoded['input_ids']:
                tokens = torch.tensor(input_ids, dtype=torch.long)
                if len(tokens) >= 10:
                    tokenized.append(tokens)
        
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
    
    # Initialize padded tensors
    inputs = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
    targets = torch.full((batch_size, max_chain_len, max_seq_len), pad_token_id, dtype=torch.long)
    loss_masks = torch.zeros((batch_size, max_chain_len, max_seq_len), dtype=torch.bool)
    gate_targets = torch.zeros((batch_size, max_chain_len + 1), dtype=torch.float)
    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    chain_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill tensors
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
    
    # BERT masking config
    bert_config = training_config.get('bert_masking', {})
    bert_masking_enabled = bert_config.get('enabled', True)
    mask_prob = bert_config.get('mask_prob', 0.8)
    random_prob = bert_config.get('random_prob', 0.1)
    keep_prob = bert_config.get('keep_prob', 0.1)
    
    # Dataset configuration
    dataset_name = data_config['dataset_name']
    dataset_probabilities = data_config.get('dataset_probabilities', None)
    
    # Sample limits for streaming validation/test
    max_val_samples = data_config.get('max_val_samples', 5000)
    max_test_samples = data_config.get('max_test_samples', 10000)
    
    # Common kwargs
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
        # STANDARD PRACTICE: Train streaming, Val/Test map-style
        print("\nDataset mode: Train=Streaming, Val/Test=Map-style (standard)")
        
        # Train: Streaming for memory efficiency
        train_dataset = StreamingTextDataset(
            dataset_name=dataset_name,
            split='train',
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
            **common_kwargs
        )
        
        # Val/Test: Map-style for speed and reproducibility
        val_dataset = WikiTextDataset(
            dataset_name=dataset_name,
            split='validation',
            samples_per_text=1,
            streaming=False,  # Force map-style
            **common_kwargs
        )
        
        # Get pad_token_id
        pad_token_id = train_dataset.tokenizer.pad_token_id or 0
        
        # Create dataloaders
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
            shuffle=False,  # No shuffle for validation
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory'],
            collate_fn=lambda batch: collate_fn(batch, pad_token_id)
        )
        
    else:
        # All map-style datasets
        train_dataset = WikiTextDataset(
            dataset_name=dataset_name,
            split='train',
            samples_per_text=data_config.get('samples_per_text', 1),
            streaming=False,
            **common_kwargs
        )
        
        val_dataset = WikiTextDataset(
            dataset_name=dataset_name,
            split='validation',
            samples_per_text=data_config.get('samples_per_text', 1),
            streaming=False,
            **common_kwargs
        )
        
        # Get pad_token_id
        pad_token_id = train_dataset.tokenizer.pad_token_id or 0
        
        # Create dataloaders (with shuffle for map-style)
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
        """Tokenize and chunk texts into multiple samples"""
        all_input_ids = []
        all_attention_masks = []
        
        for text in examples['text']:
            if len(text.strip()) < min_length:
                continue
            
            # Tokenize full text
            encoded = tokenizer(
                text,
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=True
            )
            
            input_ids = encoded['input_ids']
            
            # Chunk into multiple samples
            stride = max_seq_length - overlap
            for i in range(0, len(input_ids), stride):
                chunk = input_ids[i:i + max_seq_length]
                
                # Pad if needed
                if len(chunk) < max_seq_length:
                    padding_length = max_seq_length - len(chunk)
                    chunk = chunk + [tokenizer.pad_token_id] * padding_length
                    attention_mask = [1] * (max_seq_length - padding_length) + [0] * padding_length
                else:
                    attention_mask = [1] * max_seq_length
                
                all_input_ids.append(chunk)
                all_attention_masks.append(attention_mask)
        
        return {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_masks
        }
    
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