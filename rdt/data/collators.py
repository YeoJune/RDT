"""Data collators for different training paradigms"""

import torch
from typing import Dict, List
from transformers import DataCollatorForLanguageModeling


class RDTCollator:
    """
    Collator for RDT chain-based training.
    Simply pads and stacks pre-processed samples (v2 style).
    
    Processing is done in Dataset.__getitem__ (worker-parallel).
    """
    
    def __init__(self, pad_token_id=0):
        """
        Args:
            pad_token_id: Padding token ID
        """
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch by padding pre-processed samples.
        
        Args:
            batch: List of pre-processed dicts with keys:
                   'input', 'targets', 'loss_masks', 'gate_targets', 'chain_length'
            
        Returns:
            Dict with padded tensors
        """
        # Samples are already processed in __getitem__
        # Just pad and stack
        max_seq_len = max(sample['input'].size(0) for sample in batch)
        max_chain_len = max(sample['chain_length'] for sample in batch)
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
            attention_mask[i, :seq_len] = (sample['input'] != self.pad_token_id).long()
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


class CMLMCollator:
    """
    Collator for CMLM (Conditional Masked Language Model).
    Returns original tokens without pre-masking.
    Masking is done on-the-fly in train_step using uniform_masking().
    """
    
    def __init__(self, pad_token_id=0):
        """
        Args:
            pad_token_id: Padding token ID
        """
        self.pad_token_id = pad_token_id
    
    def __call__(self, examples):
        """
        Collate batch with padding only (no masking).
        
        Args:
            examples: List of tokenized examples (dicts with 'input_ids', 'attention_mask')
            
        Returns:
            Dict with input_ids, attention_mask (no labels - created on-the-fly)
        """
        # Extract input_ids and attention_mask
        if isinstance(examples[0], dict):
            input_ids = [ex['input_ids'] for ex in examples]
            attention_mask = [ex.get('attention_mask', None) for ex in examples]
            # Check if any attention_mask is None
            if any(mask is None for mask in attention_mask):
                attention_mask = None
        else:
            input_ids = examples
            attention_mask = None
        
        # Find max length
        if isinstance(input_ids[0], torch.Tensor):
            max_len = max(ids.size(0) for ids in input_ids)
        else:
            max_len = max(len(ids) for ids in input_ids)
        
        # Pad sequences
        padded_input_ids = []
        padded_attention_mask = []
        
        for i, ids in enumerate(input_ids):
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [self.pad_token_id] * pad_len)
            
            if attention_mask is not None:
                mask = attention_mask[i]
                if isinstance(mask, torch.Tensor):
                    mask = mask.tolist()
                padded_attention_mask.append(mask + [0] * pad_len)
            else:
                padded_attention_mask.append([1] * len(ids) + [0] * pad_len)
        
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_mask, dtype=torch.long)
        }


class MDLMCollator(CMLMCollator):
    """
    Collator for MDLM (Masked Diffusion Language Model).
    
    Identical to CMLM: returns original tokens without pre-masking.
    Masking is done on-the-fly in train_step using continuous_time_masking().
    
    This enables continuous-time diffusion training where masking ratio
    varies across time steps t ~ Uniform[0, 1].
    """
    pass  # Inherits all functionality from CMLMCollator


def get_collator(model_type: str, tokenizer=None, pad_token_id=0, mlm_probability=0.15):
    """
    Factory function to get appropriate collator based on model type.
    
    Args:
        model_type: 'rdt', 'mlm', 'cmlm', or 'mdlm'
        tokenizer: Required for MLM collator
        pad_token_id: Padding token ID for RDT/CMLM/MDLM collator
        mlm_probability: Masking probability for MLM collator
        
    Returns:
        Appropriate collator instance
    """
    model_type = model_type.lower()
    
    if model_type == 'rdt':
        return RDTCollator(pad_token_id=pad_token_id)
    elif model_type == 'mlm':
        if tokenizer is None:
            raise ValueError("Tokenizer required for MLM collator")
        return MLMCollator(tokenizer=tokenizer, mlm_probability=mlm_probability)
    elif model_type == 'cmlm':
        return CMLMCollator(pad_token_id=pad_token_id)
    elif model_type == 'mdlm':
        return MDLMCollator(pad_token_id=pad_token_id)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'rdt', 'mlm', 'cmlm', or 'mdlm'")