"""Baseline MLM model for comparison with RDT"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForMaskedLM,
    AutoConfig,
    AutoTokenizer
)


class MLM(nn.Module):
    """
    Wrapper for standard Masked Language Model baselines (BERT, RoBERTa, etc.)
    Provides consistent interface with RDT for unified training and evaluation.
    """
    
    def __init__(
        self,
        model_name='roberta-base',
        vocab_size=None,
        mask_token_id=None,
        pad_token_id=None
    ):
        """
        Args:
            model_name: HuggingFace model identifier (e.g., 'roberta-base', 'bert-base-uncased')
            vocab_size: Vocabulary size (optional, auto-detected from model)
            mask_token_id: ID of [MASK] token (auto-detected if None)
            pad_token_id: ID of [PAD] token (auto-detected if None)
        """
        super().__init__()
        self.model_name = model_name
        
        # Load pretrained model
        if vocab_size is not None:
            config = AutoConfig.from_pretrained(model_name)
            config.vocab_size = vocab_size
            self.model = AutoModelForMaskedLM.from_config(config)
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        
        self.vocab_size = self.model.config.vocab_size
        self.d_model = self.model.config.hidden_size
        
        # Special token IDs
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mask_token_id = mask_token_id if mask_token_id is not None else tokenizer.mask_token_id
        self.pad_token_id = pad_token_id if pad_token_id is not None else tokenizer.pad_token_id
        
        if self.mask_token_id is None:
            raise ValueError(f"Could not determine mask_token_id for {model_name}")
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Standard MLM forward pass
        
        Args:
            input_ids: [B, L] token indices
            attention_mask: [B, L] attention mask (1=valid, 0=padding)
            labels: [B, L] target token indices (optional, for loss computation)
            
        Returns:
            If labels provided: (loss, logits)
            Otherwise: logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        if labels is not None:
            return outputs.loss, outputs.logits
        return outputs.logits
    
    def standard_masking(self, input_ids, attention_mask=None, mask_prob=0.15):
        """
        Vectorized standard BERT-style masking (OPTIMIZED - no Python loops).
        
        Args:
            input_ids: [B, L] original token indices
            attention_mask: [B, L] attention mask (1=valid, 0=padding)
            mask_prob: Probability of masking each token (default: 0.15)
        
        Returns:
            masked_input_ids: [B, L] input with masks applied
            labels: [B, L] labels for loss (-100 for non-masked tokens)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 1. Create probability matrix for masking
        probability_matrix = torch.full((batch_size, seq_len), mask_prob, device=device)
        
        # 2. Zero out probabilities for padding tokens
        if attention_mask is not None:
            probability_matrix = probability_matrix.masked_fill(attention_mask == 0, 0.0)
        
        # Don't mask padding tokens (double check)
        probability_matrix = probability_matrix.masked_fill(input_ids == self.pad_token_id, 0.0)
        
        # 3. Sample mask decisions (vectorized Bernoulli sampling)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 4. Create labels (-100 for non-masked tokens)
        labels = input_ids.clone()
        labels = labels.masked_fill(~masked_indices, -100)
        
        # 5. Apply [MASK] token
        masked_input_ids = input_ids.clone()
        masked_input_ids = masked_input_ids.masked_fill(masked_indices, self.mask_token_id)
        
        return masked_input_ids, labels
    
    def get_embeddings(self, input_ids):
        """Get token embeddings without passing through transformer"""
        return self.model.get_input_embeddings()(input_ids)
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @classmethod
    def from_pretrained(cls, model_name):
        """Load pretrained model"""
        return cls(model_name=model_name)
    
    def save_pretrained(self, save_path):
        """Save model"""
        self.model.save_pretrained(save_path)
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict"""
        return self.model.load_state_dict(state_dict, strict=strict)
    
    def state_dict(self):
        """Get state dict"""
        return self.model.state_dict()