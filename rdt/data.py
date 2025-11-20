"""Data loading and preprocessing for RDT"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import random
from typing import List, Dict, Tuple
import numpy as np


class MaskingStrategy:
    """Generate progressively masked sequences"""
    
    def __init__(
        self,
        total_steps: int = 10,
        mask_token_id: int = 103,  # [MASK] for BERT
        strategy: str = 'linear'
    ):
        self.total_steps = total_steps
        self.mask_token_id = mask_token_id
        self.strategy = strategy
    
    def generate_masked_sequence(
        self,
        tokens: torch.Tensor,
        step: int
    ) -> torch.Tensor:
        """
        Generate masked sequence for a given step
        
        Args:
            tokens: (seq_len,) original token ids
            step: current step (0 = original, total_steps = fully masked)
        
        Returns:
            masked_tokens: (seq_len,)
        """
        if step == 0:
            return tokens.clone()
        
        # Calculate masking ratio
        if self.strategy == 'linear':
            mask_ratio = step / self.total_steps
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Random masking
        seq_len = len(tokens)
        num_mask = int(seq_len * mask_ratio)
        
        if num_mask == 0:
            return tokens.clone()
        
        # Randomly select positions to mask
        mask_positions = random.sample(range(seq_len), num_mask)
        
        masked_tokens = tokens.clone()
        masked_tokens[mask_positions] = self.mask_token_id
        
        return masked_tokens
    
    def generate_chain(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate full denoising chain: [a_0, a_1, ..., a_N]
        
        Args:
            tokens: (seq_len,) original tokens
        
        Returns:
            chain: list of masked sequences from step 0 to total_steps
        """
        chain = []
        for step in range(self.total_steps + 1):
            masked = self.generate_masked_sequence(tokens, step)
            chain.append(masked)
        
        return chain


class WikiTextDataset(Dataset):
    """WikiText dataset with progressive masking"""
    
    def __init__(
        self,
        dataset_name: str = 'wikitext-2',
        split: str = 'train',
        tokenizer_name: str = 'bert-base-uncased',
        max_seq_length: int = 512,
        total_steps: int = 10,
        max_chain_length: int = 5,
        masking_strategy: str = 'linear'
    ):
        self.split = split
        self.max_seq_length = max_seq_length
        self.total_steps = total_steps
        self.max_chain_length = max_chain_length
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mask_token_id = self.tokenizer.mask_token_id
        
        # Masking strategy
        self.masking = MaskingStrategy(
            total_steps=total_steps,
            mask_token_id=self.mask_token_id,
            strategy=masking_strategy
        )
        
        # Load dataset
        print(f"Loading {dataset_name} ({split})...")
        dataset_config = 'wikitext-2-raw-v1' if dataset_name == 'wikitext-2' else 'wikitext-103-raw-v1'
        self.dataset = load_dataset('wikitext', dataset_config, split=split)
        
        # Tokenize and filter
        self.tokenized_data = self._prepare_data()
        
        print(f"Dataset loaded: {len(self.tokenized_data)} sequences")
    
    def _prepare_data(self) -> List[torch.Tensor]:
        """Tokenize and prepare sequences"""
        tokenized = []
        
        for item in self.dataset:
            text = item['text'].strip()
            if len(text) == 0:
                continue
            
            # Tokenize
            encoded = self.tokenizer(
                text,
                max_length=self.max_seq_length,
                truncation=True,
                padding=False,
                return_tensors='pt'
            )
            
            tokens = encoded['input_ids'].squeeze(0)
            
            # Filter short sequences
            if len(tokens) < 10:
                continue
            
            tokenized.append(tokens)
        
        return tokenized
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get training sample with random chain segment
        
        Returns:
            {
                'input': (seq_len,) - masked input at step i
                'targets': (L, seq_len) - target sequences [a_{i-1}, ..., a_{i-L}]
                'gate_targets': (L,) - normalized step indices
                'chain_length': int
            }
        """
        tokens = self.tokenized_data[idx]
        
        # Generate full chain
        chain = self.masking.generate_chain(tokens)
        
        # Random sampling: pick starting step i and length L
        # i must be >= L so we can go backwards
        max_start = self.total_steps
        min_start = 1  # At least step 1
        
        if max_start < min_start:
            max_start = min_start
        
        start_step = random.randint(min_start, max_start)
        
        # Chain length: min(max_chain_length, start_step)
        chain_length = min(self.max_chain_length, start_step)
        
        # Input: a_i (most masked)
        input_tokens = chain[start_step]
        
        # Targets: [a_{i-1}, a_{i-2}, ..., a_{i-L}]
        target_sequences = []
        gate_targets = []
        
        for k in range(1, chain_length + 1):
            target_step = start_step - k
            target_sequences.append(chain[target_step])
            # Normalize gate target: remaining_steps / total_steps
            gate_targets.append(target_step / self.total_steps)
        
        # Stack
        targets = torch.stack(target_sequences)  # (L, seq_len)
        gate_targets = torch.tensor(gate_targets, dtype=torch.float)  # (L,)
        
        return {
            'input': input_tokens,
            'targets': targets,
            'gate_targets': gate_targets,
            'chain_length': chain_length
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader with padding
    """
    # Find max sequence length and max chain length in batch
    max_seq_len = max(item['input'].size(0) for item in batch)
    max_chain_len = max(item['chain_length'] for item in batch)
    
    batch_size = len(batch)
    
    # Pad token (usually 0 for BERT)
    pad_token_id = 0
    
    # Initialize tensors
    inputs = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
    targets = torch.full((batch_size, max_chain_len, max_seq_len), pad_token_id, dtype=torch.long)
    gate_targets = torch.zeros((batch_size, max_chain_len), dtype=torch.float)
    chain_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, item in enumerate(batch):
        seq_len = item['input'].size(0)
        chain_len = item['chain_length']
        
        inputs[i, :seq_len] = item['input']
        targets[i, :chain_len, :seq_len] = item['targets']
        gate_targets[i, :chain_len] = item['gate_targets']
        chain_lengths[i] = chain_len
    
    return {
        'input': inputs,
        'targets': targets,
        'gate_targets': gate_targets,
        'chain_lengths': chain_lengths
    }


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    
    data_config = config['data']
    training_config = config['training']
    
    # Train dataset
    train_dataset = WikiTextDataset(
        dataset_name=data_config['dataset_name'],
        split='train',
        tokenizer_name=data_config['tokenizer_name'],
        max_seq_length=data_config['max_seq_length'],
        total_steps=training_config['total_steps'],
        max_chain_length=training_config['max_chain_length'],
        masking_strategy=training_config['masking_strategy']
    )
    
    # Validation dataset
    val_dataset = WikiTextDataset(
        dataset_name=data_config['dataset_name'],
        split='validation',
        tokenizer_name=data_config['tokenizer_name'],
        max_seq_length=data_config['max_seq_length'],
        total_steps=training_config['total_steps'],
        max_chain_length=training_config['max_chain_length'],
        masking_strategy=training_config['masking_strategy']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader
