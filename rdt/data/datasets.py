"""
Data loading and preprocessing for RDT and Baselines.
- RDT: Uses GPU Preprocessing (returns raw input_ids, preprocessing in Trainer).
- Baselines (MLM, CMLM, MDLM): Uses standard Collators from collators.py.
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
import numpy as np
from typing import List, Dict, Tuple, Iterator, Union, Optional
from tqdm import tqdm
import os

# Tokenizer parallelism disabled to prevent deadlocks in DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        dataset_name_lower = dataset_name.lower()
        return 'bookcorpus' in dataset_name_lower or 'wikipedia' in dataset_name_lower
    
    @staticmethod
    def _apply_split_filter(idx: int, split: str) -> bool:
        mod = idx % 1000
        if split == 'train': return mod < 990
        elif split == 'validation': return 990 <= mod < 995
        elif split == 'test': return 995 <= mod < 1000
        return True


class StreamingTextDataset(IterableDataset, DatasetLoaderMixin):
    """
    Streaming dataset that yields raw token sequences.
    No masking or chain generation here.
    """
    
    def __init__(self, dataset_name: Union[str, List[str]], tokenizer_name: str,
                 max_seq_length: int = 512, split='train', 
                 dataset_probabilities: Optional[List[float]] = None,
                 max_val_samples: int = 5000, max_test_samples: int = 10000, **kwargs):
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length
        self.split = split
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        
        if isinstance(dataset_name, str):
            self.dataset_names = [dataset_name]
        else:
            self.dataset_names = dataset_name
            
        self.is_multi_dataset = len(self.dataset_names) > 1
        self.needs_filtering = [self._needs_split_filtering(name) for name in self.dataset_names]
        self.any_needs_filtering = any(self.needs_filtering)

        if self.is_multi_dataset:
            datasets = [self._load_single_dataset(name, split, streaming=True) for name in self.dataset_names]
            
            if dataset_probabilities is None:
                self.probabilities = [1.0 / len(datasets)] * len(datasets)
            else:
                prob_sum = sum(dataset_probabilities)
                self.probabilities = [p / prob_sum for p in dataset_probabilities]

            datasets_with_index = []
            for idx, ds in enumerate(datasets):
                ds_with_index = ds.map(lambda ex, i=idx: {**ex, '__dataset_index__': i})
                datasets_with_index.append(ds_with_index)
            
            self.dataset = interleave_datasets(
                datasets_with_index, probabilities=self.probabilities, seed=42, stopping_strategy='all_exhausted'
            )
        else:
            self.dataset = self._load_single_dataset(self.dataset_names[0], split, streaming=True)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        dataset_counters = {i: 0 for i in range(len(self.dataset_names))}
        
        yielded_samples = 0
        max_samples = None
        if self.split == 'validation': max_samples = self.max_val_samples
        if self.split == 'test': max_samples = self.max_test_samples
        
        if max_samples is not None and worker_info is not None:
            max_samples = max_samples // worker_info.num_workers

        for idx, item in enumerate(self.dataset):
            if worker_info is not None and idx % worker_info.num_workers != worker_info.id:
                continue
            
            if max_samples is not None and yielded_samples >= max_samples:
                return

            should_include = True
            if self.any_needs_filtering:
                ds_idx = item.get('__dataset_index__', 0)
                if self.needs_filtering[ds_idx]:
                    cnt = dataset_counters[ds_idx]
                    dataset_counters[ds_idx] += 1
                    if not self._apply_split_filter(cnt, self.split):
                        should_include = False
            
            if not should_include: continue

            text = item.get('text', '').strip()
            if len(text) == 0: continue
            
            encoded = self.tokenizer(
                text,
                max_length=self.max_seq_length,
                truncation=True,
                padding=False,
                return_overflowing_tokens=True,
                stride=0
            )
            
            for input_ids in encoded['input_ids']:
                if len(input_ids) < 10: continue
                
                yield {
                    'input_ids': torch.tensor(input_ids, dtype=torch.long)
                }
                
                yielded_samples += 1
                if max_samples is not None and yielded_samples >= max_samples:
                    return


class WikiTextDataset(Dataset, DatasetLoaderMixin):
    """
    Map-style dataset that returns raw token sequences.
    """
    
    def __init__(self, dataset_name: Union[str, List[str]], tokenizer_name: str,
                 max_seq_length: int = 512, split='train', samples_per_text=1,
                 max_val_samples: int = 5000, max_test_samples: int = 10000, **kwargs):
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length
        self.split = split
        self.samples_per_text = samples_per_text
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        
        if isinstance(dataset_name, str):
            self.dataset_names = [dataset_name]
        else:
            self.dataset_names = dataset_name
            
        all_tokenized = []
        for name in self.dataset_names:
            dataset = self._load_single_dataset(name, split, streaming=False)
            
            if self._needs_split_filtering(name):
                dataset = self._split_dataset_manual(dataset, split)
            
            tokenized = self._prepare_data(dataset)
            all_tokenized.extend(tokenized)
            
        self.tokenized_data = all_tokenized
        print(f"Loaded {len(self.tokenized_data)} sequences for split '{split}'")

    def _split_dataset_manual(self, dataset, split):
        total_size = len(dataset)
        idxs = list(range(total_size))
        
        if split == 'train':
            idxs = [i for i in idxs if self._apply_split_filter(i, 'train')]
        elif split == 'validation':
            idxs = [i for i in idxs if self._apply_split_filter(i, 'validation')]
        elif split == 'test':
            idxs = [i for i in idxs if self._apply_split_filter(i, 'test')]
            
        return dataset.select(idxs)

    def _prepare_data(self, dataset) -> List[torch.Tensor]:
        tokenized = []
        max_samples = self.max_val_samples if self.split == 'validation' else None
        
        for item in tqdm(dataset, desc=f"Tokenizing {self.split}"):
            if max_samples and len(tokenized) >= max_samples: break
            
            text = item['text'].strip()
            if len(text) == 0: continue
            
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
                    tokenized.append(torch.tensor(input_ids, dtype=torch.long))
                    if max_samples and len(tokenized) >= max_samples: break
                    
        return tokenized

    def __len__(self):
        return len(self.tokenized_data) * self.samples_per_text

    def __getitem__(self, idx):
        text_idx = idx // self.samples_per_text
        return {
            'input_ids': self.tokenized_data[text_idx]
        }


def simple_collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Collate function for RDT (GPU Preprocessing Mode).
    """
    input_ids = [item['input_ids'] for item in batch]
    padded_inputs = torch.nn.utils.rnn.pad_sequence(
        input_ids, 
        batch_first=True, 
        padding_value=pad_token_id
    )
    attention_mask = (padded_inputs != pad_token_id).long()
    
    return {
        'input_ids': padded_inputs,
        'attention_mask': attention_mask
    }


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders for RDT training"""
    data_config = config['data']
    training_config = config['training']
    streaming = data_config.get('streaming', False)
    
    tokenizer_name = data_config['tokenizer_name']
    
    # Load tokenizer for pad_token_id
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    common_kwargs = {
        'dataset_name': data_config['dataset_name'],
        'tokenizer_name': tokenizer_name,
        'max_seq_length': data_config['max_seq_length'],
        'dataset_probabilities': data_config.get('dataset_probabilities', None),
        'max_val_samples': data_config.get('max_val_samples', 5000),
        'max_test_samples': data_config.get('max_test_samples', 10000),
    }

    if streaming:
        print("Mode: Streaming Training")
        train_dataset = StreamingTextDataset(split='train', **common_kwargs)
    else:
        print("Mode: Map-style Training")
        train_dataset = WikiTextDataset(
            split='train', 
            samples_per_text=data_config.get('samples_per_text', 1),
            **common_kwargs
        )

    val_dataset = WikiTextDataset(split='validation', samples_per_text=1, **common_kwargs)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=not streaming,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        collate_fn=lambda x: simple_collate_fn(x, pad_token_id)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        collate_fn=lambda x: simple_collate_fn(x, pad_token_id)
    )
    
    return train_loader, val_loader


# ============================================================================
# Baseline Model Dataloaders (MLM, CMLM, MDLM)
# ============================================================================

def create_mlm_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders for MLM (BERT-style) training.
    Reuses WikiTextDataset but applies MLMCollator for standard masking.
    """
    from .collators import MLMCollator
    from transformers import AutoTokenizer
    
    data_config = config['data']
    training_config = config['training']
    
    common_kwargs = {
        'dataset_name': data_config['dataset_name'],
        'tokenizer_name': data_config['tokenizer_name'],
        'max_seq_length': data_config['max_seq_length'],
        'max_val_samples': data_config.get('max_val_samples', 5000),
        'max_test_samples': data_config.get('max_test_samples', 10000),
    }
    
    # MLM usually uses map-style dataset
    train_dataset = WikiTextDataset(
        split='train', 
        samples_per_text=data_config.get('samples_per_text', 1),
        **common_kwargs
    )
    val_dataset = WikiTextDataset(split='validation', samples_per_text=1, **common_kwargs)
    
    # Initialize Collator (Standard BERT Masking)
    tokenizer = AutoTokenizer.from_pretrained(data_config['tokenizer_name'])
    collator = MLMCollator(
        tokenizer=tokenizer,
        mlm_probability=data_config.get('mlm_probability', 0.15)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        collate_fn=collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        collate_fn=collator
    )
    
    return train_loader, val_loader


def create_cmlm_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders for CMLM/MDLM.
    Uses CMLMCollator (Padding only).
    """
    from .collators import CMLMCollator
    from transformers import AutoTokenizer
    
    data_config = config['data']
    training_config = config['training']
    
    common_kwargs = {
        'dataset_name': data_config['dataset_name'],
        'tokenizer_name': data_config['tokenizer_name'],
        'max_seq_length': data_config['max_seq_length'],
        'max_val_samples': data_config.get('max_val_samples', 5000),
        'max_test_samples': data_config.get('max_test_samples', 10000),
    }
    
    train_dataset = WikiTextDataset(
        split='train', 
        samples_per_text=data_config.get('samples_per_text', 1),
        **common_kwargs
    )
    val_dataset = WikiTextDataset(split='validation', samples_per_text=1, **common_kwargs)
    
    tokenizer = AutoTokenizer.from_pretrained(data_config['tokenizer_name'])
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    collator = CMLMCollator(pad_token_id=pad_token_id)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        collate_fn=collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        collate_fn=collator
    )
    
    return train_loader, val_loader


def create_mdlm_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders for MDLM.
    Identical to CMLM setup.
    """
    return create_cmlm_dataloaders(config)