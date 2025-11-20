"""Data loading and preprocessing for RDT"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import random
import numpy as np
from typing import List, Dict, Tuple


class ShuffleIndexMasking:
    """Index-shuffling based masking strategy (memory efficient)"""
    
    def __init__(
        self,
        total_steps: int = 10,
        mask_token_id: int = 103,
        visible_loss_ratio: float = 0.15
    ):
        self.total_steps = total_steps
        self.mask_token_id = mask_token_id
        self.visible_loss_ratio = visible_loss_ratio
    
    def generate_shuffle_order(self, seq_len: int) -> np.ndarray:
        """Generate random shuffle order for restoration"""
        indices = np.arange(seq_len)
        np.random.shuffle(indices)
        return indices
    
    def create_step_data(
        self,
        tokens: torch.Tensor,
        shuffle_order: np.ndarray,
        step: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """
        Create input and targets for a given step with FIXED permutation
        
        Args:
            tokens: (seq_len,) original clean tokens
            shuffle_order: (seq_len,) shuffled restoration order (FIXED for all steps)
            step: current step (0 = clean, total_steps = fully masked)
        
        Returns:
            input_tokens: (seq_len,) reordered once, with masking
            target_tokens: (seq_len,) reordered once (clean)
            pos_ids: (seq_len,) original position indices for PE
            n_revealed: int, number of already revealed tokens
            n_delta: int, number of delta tokens (newly revealing)
        """
        seq_len = len(tokens)
        
        # Reorder tokens ONCE by shuffle_order (FIXED physical order)
        tokens_reordered = tokens[shuffle_order]
        pos_ids = torch.from_numpy(shuffle_order).long()
        
        # Calculate how many tokens are revealed at this step
        # step 0 → all visible, step N → all masked
        # Restoration order: shuffle_order[0] reveals first, shuffle_order[-1] reveals last
        num_visible = int(seq_len * (1 - step / self.total_steps))
        
        # Already revealed: shuffle_order[:num_visible]
        # Still masked: shuffle_order[num_visible:]
        
        # Create input with masking (physical order is FIXED)
        input_tokens = tokens_reordered.clone()
        input_tokens[num_visible:] = self.mask_token_id  # Mask unrevealed positions
        
        # Target is always clean (physical order is FIXED)
        target_tokens = tokens_reordered
        
        # Calculate loss region
        # Delta: newly revealed tokens
        # Context: sample from already revealed tokens for stability
        if num_visible > 0:
            n_context_loss = max(1, int(num_visible * self.visible_loss_ratio))
            n_delta = seq_len - num_visible  # All masked positions
            n_revealed = num_visible
        else:
            n_context_loss = 0
            n_delta = seq_len
            n_revealed = 0
        
        # Total loss region: delta + sampled context
        n_loss = n_delta + n_context_loss
        
        return input_tokens, target_tokens, pos_ids, n_revealed, n_delta


class WikiTextDataset(Dataset):
    """WikiText dataset with shuffle-index based progressive masking"""
    
    def __init__(
        self,
        dataset_name: str = 'wikitext-2',
        split: str = 'train',
        tokenizer_name: str = 'bert-base-uncased',
        max_seq_length: int = 512,
        total_steps: int = 10,
        max_chain_length: int = 5,
        visible_loss_ratio: float = 0.15
    ):
        self.split = split
        self.max_seq_length = max_seq_length
        self.total_steps = total_steps
        self.max_chain_length = max_chain_length
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mask_token_id = self.tokenizer.mask_token_id
        
        # Masking strategy
        self.masking = ShuffleIndexMasking(
            total_steps=total_steps,
            mask_token_id=self.mask_token_id,
            visible_loss_ratio=visible_loss_ratio
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
        Get training sample with fixed permutation and sliding window loss
        
        Returns:
            {
                'input': (L, seq_len) - fixed order, varying masks
                'targets': (L, seq_len) - fixed order targets
                'pos_ids': (seq_len,) - position IDs (same for all steps)
                'n_revealed': (L,) - number of already revealed tokens per step
                'n_delta': (L,) - number of delta tokens per step
                'gate_targets': (L,) - normalized step indices
                'chain_length': int
            }
        """
        tokens = self.tokenized_data[idx]
        seq_len = len(tokens)
        
        # Generate shuffle order (restoration order) - FIXED for all steps
        shuffle_order = self.masking.generate_shuffle_order(seq_len)
        
        # Random starting step
        max_start = self.total_steps
        min_start = 1
        start_step = random.randint(min_start, max_start)
        
        # Chain length
        chain_length = min(self.max_chain_length, start_step)
        
        # Generate L consecutive steps (SAME physical order, different masks)
        inputs = []
        targets = []
        n_revealed_list = []
        n_delta_list = []
        gate_targets = []
        
        for offset in range(chain_length):
            step = start_step - offset
            
            # Create data for this step (FIXED physical order)
            input_tokens, target_tokens, pos_ids, n_revealed, n_delta = self.masking.create_step_data(
                tokens, shuffle_order, step
            )
            
            inputs.append(input_tokens)
            targets.append(target_tokens)
            n_revealed_list.append(n_revealed)
            n_delta_list.append(n_delta)
            gate_targets.append(step / self.total_steps)  # Normalized
        
        # Stack
        inputs = torch.stack(inputs)  # (L, seq_len)
        targets = torch.stack(targets)  # (L, seq_len)
        n_revealed = torch.tensor(n_revealed_list, dtype=torch.long)  # (L,)
        n_delta = torch.tensor(n_delta_list, dtype=torch.long)  # (L,)
        gate_targets = torch.tensor(gate_targets, dtype=torch.float)  # (L,)
        
        return {
            'input': inputs[0],  # First step input
            'targets': targets,
            'pos_ids': pos_ids,  # Same for all steps!
            'n_revealed': n_revealed,
            'n_delta': n_delta,
            'gate_targets': gate_targets,
            'chain_length': chain_length
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader with padding and attention mask
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
    pos_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)  # Same for all steps!
    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)  # 0 for padding
    n_revealed = torch.zeros((batch_size, max_chain_len), dtype=torch.long)
    n_delta = torch.zeros((batch_size, max_chain_len), dtype=torch.long)
    gate_targets = torch.zeros((batch_size, max_chain_len), dtype=torch.float)
    chain_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, item in enumerate(batch):
        seq_len = item['input'].size(0)
        chain_len = item['chain_length']
        
        inputs[i, :seq_len] = item['input']
        targets[i, :chain_len, :seq_len] = item['targets']
        pos_ids[i, :seq_len] = item['pos_ids']  # Same for all steps!
        attention_mask[i, :seq_len] = 1  # Valid tokens
        n_revealed[i, :chain_len] = item['n_revealed']
        n_delta[i, :chain_len] = item['n_delta']
        gate_targets[i, :chain_len] = item['gate_targets']
        chain_lengths[i] = chain_len
    
    return {
        'input': inputs,
        'targets': targets,
        'pos_ids': pos_ids,  # (B, seq_len) - same for all steps!
        'attention_mask': attention_mask,
        'n_revealed': n_revealed,
        'n_delta': n_delta,
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
        visible_loss_ratio=training_config.get('visible_loss_ratio', 0.15)
    )
    
    # Validation dataset
    val_dataset = WikiTextDataset(
        dataset_name=data_config['dataset_name'],
        split='validation',
        tokenizer_name=data_config['tokenizer_name'],
        max_seq_length=data_config['max_seq_length'],
        total_steps=training_config['total_steps'],
        max_chain_length=training_config['max_chain_length'],
        visible_loss_ratio=training_config.get('visible_loss_ratio', 0.15)
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