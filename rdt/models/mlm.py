"""Baseline MLM model for comparison with RDT"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForMaskedLM,
    AutoConfig,
    AutoTokenizer
)
from typing import Dict, Optional


class MLM(nn.Module):
    """
    Wrapper for standard Masked Language Model baselines (BERT, RoBERTa, etc.)
    Provides consistent interface with RDT for unified training and evaluation.
    
    Supports both pretrained and from-scratch training:
    - Pretrained: Load weights from HuggingFace hub
    - From scratch: Initialize with random weights using architecture config
    """
    
    def __init__(
        self,
        architecture: str = 'bert-base-uncased',
        pretrained: Optional[str] = None,
        vocab_size: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        config_overrides: Optional[Dict] = None,
        bert_masking_enabled: bool = False,
        mask_prob: float = 0.8,
        random_prob: float = 0.1,
        keep_prob: float = 0.1
    ):
        """
        Args:
            architecture: Model architecture identifier (e.g., 'bert-base-uncased', 'roberta-base')
                         Used for loading config and tokenizer
            pretrained: Model name/path to load pretrained weights from (None = from scratch)
                       If provided, loads pretrained weights from HuggingFace or local path
            vocab_size: Vocabulary size (required for from-scratch training)
            mask_token_id: ID of [MASK] token (auto-detected from tokenizer if None)
            pad_token_id: ID of [PAD] token (auto-detected from tokenizer if None)
            config_overrides: Dict of config parameters to override (e.g., {'num_hidden_layers': 6})
                            Useful for matching parameter count with other models
            bert_masking_enabled: Enable BERT-style masking (80% [MASK], 10% random, 10% keep)
            mask_prob: Probability of replacing with [MASK] token (default: 0.8)
            random_prob: Probability of replacing with random token (default: 0.1)
            keep_prob: Probability of keeping original token (default: 0.1)
        
        Examples:
            # Pretrained model
            model = MLM(architecture='bert-base-uncased', pretrained='bert-base-uncased')
            
            # From scratch with custom vocab
            model = MLM(architecture='bert-base-uncased', vocab_size=50000, 
                       mask_token_id=3, pad_token_id=0)
            
            # From scratch with custom architecture (match RDT size)
            model = MLM(architecture='bert-base-uncased', 
                       config_overrides={'num_hidden_layers': 6, 'hidden_size': 512})
        """
        super().__init__()
        self.architecture = architecture
        self.pretrained = pretrained
        
        # BERT-style masking parameters
        self.bert_masking_enabled = bert_masking_enabled
        self.mask_prob = mask_prob
        self.random_prob = random_prob
        self.keep_prob = keep_prob
        
        # Load model config
        config = AutoConfig.from_pretrained(architecture)
        
        # Apply config overrides (for fair comparison with custom models)
        if config_overrides:
            print(f"  Applying config overrides:")
            for k, v in config_overrides.items():
                if hasattr(config, k):
                    setattr(config, k, v)
                    print(f"    - {k}: {getattr(config, k)} → {v}")
                else:
                    print(f"    ⚠ Warning: config does not have attribute '{k}', skipping")
        
        # Override vocab size if provided
        if vocab_size is not None:
            config.vocab_size = vocab_size
        
        # Load model: pretrained or from scratch
        if pretrained is not None:
            # Load pretrained weights
            self.model = AutoModelForMaskedLM.from_pretrained(pretrained, config=config)
            print(f"✓ Loaded pretrained weights from: {pretrained}")
        else:
            # Initialize from scratch
            self.model = AutoModelForMaskedLM.from_config(config)
            print(f"✓ Initialized from scratch: {architecture}")
            if vocab_size is not None:
                print(f"  - Custom vocab_size: {vocab_size}")
        
        self.vocab_size = self.model.config.vocab_size
        self.d_model = self.model.config.hidden_size
        
        # Determine special token IDs
        if mask_token_id is None or pad_token_id is None:
            # Auto-detect from tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(architecture)
                self.mask_token_id = mask_token_id if mask_token_id is not None else tokenizer.mask_token_id
                self.pad_token_id = pad_token_id if pad_token_id is not None else tokenizer.pad_token_id
            except Exception as e:
                raise ValueError(
                    f"Could not load tokenizer for {architecture}. "
                    f"Please provide mask_token_id and pad_token_id explicitly. Error: {e}"
                )
        else:
            # Use provided IDs
            self.mask_token_id = mask_token_id
            self.pad_token_id = pad_token_id
        
        # Validate special tokens
        if self.mask_token_id is None:
            raise ValueError(
                f"Could not determine mask_token_id. "
                f"Please provide it explicitly via mask_token_id parameter."
            )
        if self.pad_token_id is None:
            raise ValueError(
                f"Could not determine pad_token_id. "
                f"Please provide it explicitly via pad_token_id parameter."
            )
        
        print(f"  - vocab_size: {self.vocab_size}")
        print(f"  - mask_token_id: {self.mask_token_id}")
        print(f"  - pad_token_id: {self.pad_token_id}")
        print(f"  - BERT masking: {self.bert_masking_enabled}")
    
    def _apply_bert_masking(self, input_ids: torch.Tensor, mask_decision: torch.Tensor) -> torch.Tensor:
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
    def from_config(cls, config: Dict):
        """
        Create model from config dictionary.
        
        Config format:
            model:
              architecture: "bert-base-uncased"  # Required: model architecture
              pretrained: "bert-base-uncased"    # Optional: None for from-scratch
              vocab_size: 50000                   # Optional: custom vocab size
              mask_token_id: 103                  # Optional: auto-detect from tokenizer
              pad_token_id: 0                     # Optional: auto-detect from tokenizer
              config_overrides:                   # Optional: override architecture params
                num_hidden_layers: 6
                hidden_size: 512
            training:
              bert_masking:                       # Optional: BERT-style masking
                enabled: true
                mask_prob: 0.8
                random_prob: 0.1
                keep_prob: 0.1
        
        Args:
            config: Configuration dictionary
            
        Returns:
            MLM instance
        """
        model_cfg = config['model']
        training_cfg = config.get('training', {})
        bert_cfg = training_cfg.get('bert_masking', {})
        
        return cls(
            architecture=model_cfg['architecture'],
            pretrained=model_cfg.get('pretrained'),
            vocab_size=model_cfg.get('vocab_size'),
            mask_token_id=model_cfg.get('mask_token_id'),
            pad_token_id=model_cfg.get('pad_token_id'),
            config_overrides=model_cfg.get('config_overrides'),
            bert_masking_enabled=bert_cfg.get('enabled', True),
            mask_prob=bert_cfg.get('mask_prob', 0.8),
            random_prob=bert_cfg.get('random_prob', 0.1),
            keep_prob=bert_cfg.get('keep_prob', 0.1)
        )
    
    @classmethod
    def from_pretrained(cls, model_name):
        """
        Load pretrained model (legacy API for compatibility).
        Equivalent to: MLM(architecture=model_name, pretrained=model_name)
        """
        return cls(architecture=model_name, pretrained=model_name)
    
    def save_pretrained(self, save_path):
        """Save model"""
        self.model.save_pretrained(save_path)
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict"""
        return self.model.load_state_dict(state_dict, strict=strict)
    
    def state_dict(self):
        """Get state dict"""
        return self.model.state_dict()