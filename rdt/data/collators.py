"""Data collators for different training paradigms"""

import torch
from typing import Dict, List
from dataclasses import dataclass
from transformers import DataCollatorForLanguageModeling


class RDTCollator:
    """
    Collator for RDT chain-based training.
    Handles variable-length chains and padding.
    """
    
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of RDT samples with variable chain lengths.
        
        Args:
            batch: List of dicts with keys: input, targets, loss_masks, gate_targets, chain_length
            
        Returns:
            Dict with padded tensors
        """
        # Find max chain length and sequence length in batch
        max_chain_len = max(sample['chain_length'] for sample in batch)
        max_seq_len = max(sample['input'].size(0) for sample in batch)
        batch_size = len(batch)
        
        # Initialize padded tensors
        inputs = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
        targets = torch.full((batch_size, max_chain_len, max_seq_len), self.pad_token_id, dtype=torch.long)
        loss_masks = torch.zeros((batch_size, max_chain_len, max_seq_len), dtype=torch.bool)
        gate_targets = torch.zeros((batch_size, max_chain_len + 1), dtype=torch.float)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        chain_lengths = torch.zeros(batch_size, dtype=torch.long)
        
        # Fill tensors
        for i, sample in enumerate(batch):
            seq_len = sample['input'].size(0)
            chain_len = sample['chain_length']
            
            inputs[i, :seq_len] = sample['input']
            targets[i, :chain_len, :seq_len] = sample['targets']
            loss_masks[i, :chain_len, :seq_len] = sample['loss_masks']
            gate_targets[i, :chain_len + 1] = sample['gate_targets']
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


class MLMCollator(DataCollatorForLanguageModeling):
    """
    Collator for standard Masked Language Modeling (BERT, RoBERTa style).
    Extends HuggingFace's DataCollatorForLanguageModeling for consistency.
    """
    
    def __init__(self, tokenizer, mlm_probability=0.15):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            mlm_probability: Probability of masking tokens
        """
        super().__init__(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability
        )
    
    def __call__(self, examples):
        """
        Apply MLM masking to batch.
        
        Args:
            examples: List of tokenized examples (dicts with 'input_ids', 'attention_mask')
            
        Returns:
            Dict with input_ids, attention_mask, labels
        """
        # Handle both dict and tensor formats
        if isinstance(examples[0], dict):
            # Extract input_ids if in dict format
            batch = [
                {'input_ids': ex.get('input_ids', ex.get('input', ex))}
                for ex in examples
            ]
        else:
            # Convert tensors to dict format
            batch = [{'input_ids': ex} for ex in examples]
        
        # Use parent class collation
        return super().__call__(batch)


def get_collator(model_type: str, tokenizer=None, pad_token_id=0, mlm_probability=0.15):
    """
    Factory function to get appropriate collator based on model type.
    
    Args:
        model_type: 'rdt' or 'mlm'
        tokenizer: Required for MLM collator
        pad_token_id: Padding token ID for RDT collator
        mlm_probability: Masking probability for MLM collator
        
    Returns:
        Appropriate collator instance
    """
    if model_type.lower() == 'rdt':
        return RDTCollator(pad_token_id=pad_token_id)
    elif model_type.lower() == 'mlm':
        if tokenizer is None:
            raise ValueError("Tokenizer required for MLM collator")
        return MLMCollator(tokenizer=tokenizer, mlm_probability=mlm_probability)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'rdt' or 'mlm'")
