"""Data loading and preprocessing for RDT"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import random
import numpy as np
from typing import List, Dict, Tuple, Iterator
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
        
        # Generate chain
        targets = []
        loss_masks = []
        gate_targets = []
        
        for i in range(chain_length):
            current_step = start_step - i - 1
            next_step = max(0, current_step - 1)
            
            num_visible_current = int(seq_len * (1 - current_step / self.total_steps))
            num_visible_next = int(seq_len * (1 - next_step / self.total_steps))
            
            newly_visible = restore_order[num_visible_current:num_visible_next]
            
            target = tokens.clone()
            loss_mask = torch.zeros(seq_len, dtype=torch.bool)
            
            if len(newly_visible) > 0:
                num_loss_tokens = max(1, int(len(newly_visible) * self.visible_loss_ratio))
                loss_indices = newly_visible[torch.randperm(len(newly_visible))[:num_loss_tokens]]
                loss_mask[loss_indices] = True
            
            targets.append(target)
            loss_masks.append(loss_mask)
            gate_targets.append(current_step / self.total_steps)
        
        gate_targets.append(0.0)
        
        return {
            'input': input_ids,
            'targets': torch.stack(targets),
            'loss_masks': torch.stack(loss_masks),
            'gate_targets': torch.tensor(gate_targets, dtype=torch.float),
            'chain_length': chain_length
        }


class StreamingTextDataset(IterableDataset, RDTDatasetBase):
    """Streaming dataset for on-the-fly loading"""
    
    def __init__(self, dataset_name='wikitext-2', split='train', **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.split = split
        
        print(f"Loading {dataset_name} ({split}) in streaming mode...")
        
        if 'bookcorpus' in dataset_name.lower():
            self.dataset = load_dataset('rojagtap/bookcorpus', split='train', streaming=True)
        elif 'wikipedia' in dataset_name.lower():
            self.dataset = load_dataset('wikimedia/wikipedia', '20231101.en', split='train', streaming=True)
        else:
            dataset_config = 'wikitext-2-raw-v1' if 'wikitext-2' in dataset_name else 'wikitext-103-raw-v1'
            self.dataset = load_dataset('wikitext', dataset_config, split=split, streaming=True)
        
        print(f"Dataset loaded in streaming mode")
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over streaming dataset with manual splitting and multi-worker support"""
        
        # Multi-worker support: prevent data duplication
        worker_info = torch.utils.data.get_worker_info()
        
        for idx, item in enumerate(self.dataset):
            # 1. Multi-worker handling: each worker processes different subset
            if worker_info is not None:
                if idx % worker_info.num_workers != worker_info.id:
                    continue
            
            # 2. Train/Val/Test split for datasets without built-in splits (BookCorpus, Wikipedia)
            if 'bookcorpus' in self.dataset_name.lower() or 'wikipedia' in self.dataset_name.lower():
                # Modulo-based split: 40 items → 38 train (95%), 1 val (2.5%), 1 test (2.5%)
                mod = idx % 40
                
                if self.split == 'train':
                    if mod >= 38:  # Skip indices 38, 39
                        continue
                elif self.split == 'validation':
                    if mod != 38:  # Only take index 38
                        continue
                elif self.split == 'test':
                    if mod != 39:  # Only take index 39
                        continue
            
            # 3. Process text - tokenize with return_overflowing_tokens for standard chunking
            text = item.get('text', '').strip()
            if len(text) == 0:
                continue
            
            # Standard approach: tokenizer handles chunking with stride
            encoded = self.tokenizer(
                text,
                max_length=self.max_seq_length,
                truncation=True,
                padding=False,
                return_overflowing_tokens=True,
                stride=0,
                return_tensors='pt'
            )
            
            # encoded['input_ids'] shape: (num_chunks, seq_len)
            for tokens in encoded['input_ids']:
                if len(tokens) < 10:
                    continue
                yield self._process_text(tokens)


class WikiTextDataset(Dataset, RDTDatasetBase):
    """Map-style dataset for preloading all data into memory"""
    
    def __init__(self, dataset_name='wikitext-2', split='train', samples_per_text=1, streaming=False, **kwargs):
        super().__init__(**kwargs)
        if streaming:
            raise ValueError("Use StreamingTextDataset for streaming mode instead")
        
        self.split = split
        self.samples_per_text = samples_per_text
        
        print(f"Loading {dataset_name} ({split})...")
        
        # Determine dataset source and config
        if 'bookcorpus' in dataset_name.lower():
            # BookCorpus only has train split, so we need to split it
            full_dataset = load_dataset('rojagtap/bookcorpus', split='train')
            
            # Split: 95% train, 2.5% validation, 2.5% test
            total_size = len(full_dataset)
            train_size = int(0.95 * total_size)
            val_size = int(0.025 * total_size)
            test_size = total_size - train_size - val_size
            
            splits = full_dataset.train_test_split(test_size=val_size + test_size, seed=42)
            train_data = splits['train']
            remaining = splits['test']
            
            val_test_splits = remaining.train_test_split(test_size=test_size, seed=42)
            val_data = val_test_splits['train']
            test_data = val_test_splits['test']
            
            # Select appropriate split
            if split == 'train':
                self.dataset = train_data
            elif split == 'validation':
                self.dataset = val_data
            elif split == 'test':
                self.dataset = test_data
            else:
                raise ValueError(f"Invalid split: {split}")
            
            self.tokenized_data = self._prepare_data()
            print(f"BookCorpus {split} split: {len(self.tokenized_data)} sequences")
            print(f"Total samples (with samples_per_text={samples_per_text}): {len(self.tokenized_data) * samples_per_text}")
        elif 'wikipedia' in dataset_name.lower():
            # Wikipedia 20231101.en only has train split, so we need to split it
            full_dataset = load_dataset('wikimedia/wikipedia', '20231101.en', split='train')
            
            # Split: 95% train, 2.5% validation, 2.5% test
            total_size = len(full_dataset)
            train_size = int(0.95 * total_size)
            val_size = int(0.025 * total_size)
            test_size = total_size - train_size - val_size
            
            splits = full_dataset.train_test_split(test_size=val_size + test_size, seed=42)
            train_data = splits['train']
            remaining = splits['test']
            
            val_test_splits = remaining.train_test_split(test_size=test_size, seed=42)
            val_data = val_test_splits['train']
            test_data = val_test_splits['test']
            
            # Select appropriate split
            if split == 'train':
                self.dataset = train_data
            elif split == 'validation':
                self.dataset = val_data
            elif split == 'test':
                self.dataset = test_data
            else:
                raise ValueError(f"Invalid split: {split}")
            
            self.tokenized_data = self._prepare_data()
            print(f"Wikipedia 20231101.en {split} split: {len(self.tokenized_data)} sequences")
            print(f"Total samples (with samples_per_text={samples_per_text}): {len(self.tokenized_data) * samples_per_text}")
        else:
            # WikiText datasets
            dataset_config = 'wikitext-2-raw-v1' if 'wikitext-2' in dataset_name else 'wikitext-103-raw-v1'
            self.dataset = load_dataset('wikitext', dataset_config, split=split)
            self.tokenized_data = self._prepare_data()
            print(f"Dataset loaded: {len(self.tokenized_data)} sequences")
            print(f"Total samples (with samples_per_text={samples_per_text}): {len(self.tokenized_data) * samples_per_text}")
    
    def _prepare_data(self) -> List[torch.Tensor]:
        """Tokenize dataset using fast batched processing and split into chunks"""
        print("Tokenizing texts...")
        
        # Create tokenizer instance with name (not self.tokenizer to avoid pickle issues)
        tokenizer_name = self.tokenizer.name_or_path
        
        # Standard approach: use return_overflowing_tokens for chunking
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        tokenized = []
        for item in tqdm(self.dataset, desc="Tokenizing and chunking"):
            text = item['text'].strip()
            if len(text) == 0:
                continue
            
            # Standard tokenization with chunking
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
    max_seq_len = max(item['input'].size(0) for item in batch)  # input\uc774 \uc774\uc81c 1D
    max_chain_len = max(item['chain_length'] for item in batch)
    batch_size = len(batch)
    
    inputs = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
    targets = torch.full((batch_size, max_chain_len, max_seq_len), pad_token_id, dtype=torch.long)
    loss_masks = torch.zeros((batch_size, max_chain_len, max_seq_len), dtype=torch.bool)
    gate_targets = torch.zeros((batch_size, max_chain_len + 1), dtype=torch.float)  # L+1\uac1c
    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    chain_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, item in enumerate(batch):
        seq_len = item['input'].size(0)  # 1D tensor
        chain_len = item['chain_length']
        
        inputs[i, :seq_len] = item['input']
        targets[i, :chain_len, :seq_len] = item['targets']
        loss_masks[i, :chain_len, :seq_len] = item['loss_masks']
        gate_targets[i, :chain_len + 1] = item['gate_targets']  # L+1\uac1c
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
    data_config = config['data']
    training_config = config['training']
    streaming = data_config.get('streaming', False)
    
    # BERT masking config
    bert_config = training_config.get('bert_masking', {})
    bert_masking_enabled = bert_config.get('enabled', True)
    mask_prob = bert_config.get('mask_prob', 0.8)
    random_prob = bert_config.get('random_prob', 0.1)
    keep_prob = bert_config.get('keep_prob', 0.1)
    
    if streaming:
        # Use StreamingTextDataset for streaming mode
        train_dataset = StreamingTextDataset(
            dataset_name=data_config['dataset_name'],
            split='train',
            tokenizer_name=data_config['tokenizer_name'],
            max_seq_length=data_config['max_seq_length'],
            total_steps=training_config['total_steps'],
            max_chain_length=training_config['max_chain_length'],
            visible_loss_ratio=training_config.get('visible_loss_ratio', 0.15),
            bert_masking_enabled=bert_masking_enabled,
            mask_prob=mask_prob,
            random_prob=random_prob,
            keep_prob=keep_prob
        )
        
        val_dataset = StreamingTextDataset(
            dataset_name=data_config['dataset_name'],
            split='validation',
            tokenizer_name=data_config['tokenizer_name'],
            max_seq_length=data_config['max_seq_length'],
            total_steps=training_config['total_steps'],
            max_chain_length=training_config['max_chain_length'],
            visible_loss_ratio=training_config.get('visible_loss_ratio', 0.15),
            bert_masking_enabled=bert_masking_enabled,
            mask_prob=mask_prob,
            random_prob=random_prob,
            keep_prob=keep_prob
        )
        
        # Get pad_token_id from tokenizer
        pad_token_id = train_dataset.tokenizer.pad_token_id if train_dataset.tokenizer.pad_token_id is not None else 0
        
        # Streaming datasets do not support shuffle
        train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], 
                                 num_workers=data_config['num_workers'], 
                                 pin_memory=data_config['pin_memory'], 
                                 collate_fn=lambda batch: collate_fn(batch, pad_token_id))
        val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'], 
                               num_workers=data_config['num_workers'], 
                               pin_memory=data_config['pin_memory'], 
                               collate_fn=lambda batch: collate_fn(batch, pad_token_id))
    else:
        # Use WikiTextDataset for normal mode
        train_dataset = WikiTextDataset(
            dataset_name=data_config['dataset_name'],
            split='train',
            tokenizer_name=data_config['tokenizer_name'],
            max_seq_length=data_config['max_seq_length'],
            total_steps=training_config['total_steps'],
            max_chain_length=training_config['max_chain_length'],
            visible_loss_ratio=training_config.get('visible_loss_ratio', 0.15),
            samples_per_text=data_config.get('samples_per_text', 1),
            streaming=streaming,
            bert_masking_enabled=bert_masking_enabled,
            mask_prob=mask_prob,
            random_prob=random_prob,
            keep_prob=keep_prob
        )
        
        val_dataset = WikiTextDataset(
            dataset_name=data_config['dataset_name'],
            split='validation',
            tokenizer_name=data_config['tokenizer_name'],
            max_seq_length=data_config['max_seq_length'],
            total_steps=training_config['total_steps'],
            max_chain_length=training_config['max_chain_length'],
            visible_loss_ratio=training_config.get('visible_loss_ratio', 0.15),
            samples_per_text=data_config.get('samples_per_text', 1),
            streaming=streaming,
            bert_masking_enabled=bert_masking_enabled,
            mask_prob=mask_prob,
            random_prob=random_prob,
            keep_prob=keep_prob
        )
        
        # Get pad_token_id from tokenizer
        pad_token_id = train_dataset.tokenizer.pad_token_id if train_dataset.tokenizer.pad_token_id is not None else 0
        
        train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], 
                                 shuffle=True, num_workers=data_config['num_workers'], 
                                 pin_memory=data_config['pin_memory'], 
                                 collate_fn=lambda batch: collate_fn(batch, pad_token_id))
        val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'], 
                               shuffle=False, num_workers=data_config['num_workers'], 
                               pin_memory=data_config['pin_memory'], 
                               collate_fn=lambda batch: collate_fn(batch, pad_token_id))
    
    return train_loader, val_loader


def create_mlm_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders for MLM training (BERT, RoBERTa, etc.)
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
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
    
    # Text chunking: split long texts into multiple samples
    max_seq_length = data_config['max_seq_length']
    overlap = data_config.get('chunk_overlap', 0)  # Overlap between chunks
    min_length = data_config.get('min_text_length', 50)
    
    def chunk_and_tokenize(examples):
        """Tokenize and chunk texts into multiple samples"""
        all_input_ids = []
        all_attention_masks = []
        
        for text in examples['text']:
            # Skip short texts
            if len(text.strip()) < min_length:
                continue
            
            # Tokenize full text
            encoded = tokenizer(
                text,
                truncation=False,  # Don't truncate yet
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