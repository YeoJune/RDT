"""Baseline models for comparison with RDT"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForMaskedLM,
    AutoConfig
)


class BaselineMLM(nn.Module):
    """
    Wrapper for standard Masked Language Model baselines (BERT, RoBERTa, etc.)
    Provides consistent interface with RDT for unified training and evaluation.
    """
    
    def __init__(self, model_name='roberta-base', vocab_size=None):
        """
        Args:
            model_name: HuggingFace model identifier (e.g., 'roberta-base', 'bert-base-uncased')
            vocab_size: Vocabulary size (optional, auto-detected from model)
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
