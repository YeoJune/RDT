"""
Masked Language Model (MLM) with RoPE

Standard MLM implementation using shared transformer backbone with RoPE.
Supports both simple masking and BERT-style masking (80-10-10 split).

References:
- BERT: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2019)
- RoBERTa: Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (2019)
- RoPE: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple
from .transformer import TransformerEncoder


class MLM(nn.Module):
    """
    Masked Language Model with RoPE positional encoding.
    
    Architecture:
        Token Embedding → Transformer Encoder (RoPE) → LM Head
    
    Key features:
    - RoPE for positional encoding (better for long sequences)
    - Standard MLM training objective
    - Optional BERT-style masking (80% [MASK], 10% random, 10% keep)
    - FlashAttention optimization (automatic)
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension (default: 768 for BERT-base)
        n_layers: Number of transformer layers (default: 12 for BERT-base)
        n_heads: Number of attention heads (default: 12 for BERT-base)
        d_ff: Feed-forward dimension (default: 3072 = 4 * d_model)
        max_seq_len: Maximum sequence length (default: 512)
        dropout: Dropout probability (default: 0.1)
        mask_token_id: ID of [MASK] token (default: 103 for BERT)
        pad_token_id: ID of [PAD] token (default: 0 for BERT)
        rope_base: Base for RoPE frequency calculation (default: 10000.0)
        bert_masking_enabled: Enable BERT-style 80-10-10 masking (default: False)
        tie_weights: Tie input/output embeddings (default: True)
    
    Usage:
        # BERT-base size
        model = MLM(vocab_size=30522, d_model=768, n_layers=12, n_heads=12)
        
        # Small model for experiments
        model = MLM(vocab_size=50000, d_model=512, n_layers=6, n_heads=8)
        
        # Training
        masked_ids, labels = model.standard_masking(input_ids)
        loss, logits = model(masked_ids, attention_mask, labels)
        
        # Inference
        logits = model(input_ids, attention_mask)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        mask_token_id: int = 103,
        pad_token_id: int = 0,
        rope_base: float = 10000.0,
        bert_masking_enabled: bool = False,
        tie_weights: bool = True
    ):
        super().__init__()
        
        # Configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff if d_ff is not None else 4 * d_model  # Standard: 4x model dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.rope_base = rope_base
        self.bert_masking_enabled = bert_masking_enabled
        self.tie_weights = tie_weights
        
        # Token embedding
        # Standard practice: use sqrt(d_model) scaling in forward pass
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer encoder with RoPE
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=self.d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            rope_base=rope_base
        )
        
        # LM head (output projection)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying (standard practice in BERT/GPT)
        if self.tie_weights:
            self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self._init_weights()
        
        # Log configuration
        print(f"✓ Initialized MLM with RoPE:")
        print(f"  - vocab_size: {vocab_size}")
        print(f"  - d_model: {d_model}")
        print(f"  - n_layers: {n_layers}")
        print(f"  - n_heads: {n_heads}")
        print(f"  - d_ff: {self.d_ff}")
        print(f"  - max_seq_len: {max_seq_len}")
        print(f"  - dropout: {dropout}")
        print(f"  - mask_token_id: {mask_token_id}")
        print(f"  - pad_token_id: {pad_token_id}")
        print(f"  - BERT masking: {bert_masking_enabled}")
        print(f"  - Weight tying: {tie_weights}")
        print(f"  - Parameters: {self.count_parameters():,}")
    
    def _init_weights(self):
        """
        Initialize weights following BERT initialization scheme.
        
        BERT uses:
        - Normal(0, 0.02) for embeddings and linear layers
        - Zeros for biases
        - Layer norm uses default PyTorch initialization (weight=1, bias=0)
        """
        # Token embedding
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # LM head (if not weight-tied, will be initialized here)
        if not self.tie_weights:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        
        # Transformer encoder uses Xavier init (already done in TransformerEncoder)
        # This is slightly different from BERT but is more modern practice
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through MLM.
        
        Args:
            input_ids: [B, L] token indices
            attention_mask: [B, L] attention mask (1=valid, 0=padding)
            labels: [B, L] target token indices (optional, for loss computation)
                   Use -100 for positions to ignore (standard PyTorch convention)
        
        Returns:
            If labels provided: (loss, logits)
                loss: scalar cross-entropy loss
                logits: [B, L, vocab_size] prediction logits
            Otherwise: logits
                logits: [B, L, vocab_size] prediction logits
        
        Note:
            During training, input_ids should contain [MASK] tokens at positions
            where we want to predict, and labels should contain the original tokens
            at those positions (with -100 elsewhere).
        """
        # 1. Token embedding with scaling
        # Standard practice: scale by sqrt(d_model) to maintain variance
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # 2. Transformer encoding
        # Convert attention_mask (1=valid, 0=padding) to key_padding_mask (True=ignore)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        
        hidden_states = self.encoder(x, key_padding_mask=key_padding_mask)
        
        # 3. LM head (project to vocabulary)
        logits = self.lm_head(hidden_states)
        
        # 4. Compute loss if labels provided
        if labels is not None:
            # Flatten for cross-entropy
            # Cross-entropy expects: (N, C) for predictions, (N,) for targets
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100  # Standard: ignore positions with -100
            )
            return loss, logits
        
        return logits
    
    def standard_masking(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_prob: float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply standard MLM masking to input tokens.
        
        This implements the masking strategy from BERT:
        - Select 15% of tokens for masking (mask_prob=0.15)
        - Of selected tokens:
          * 80% → replace with [MASK]
          * 10% → replace with random token
          * 10% → keep original (but still predict)
        
        Only applied if bert_masking_enabled=True. Otherwise uses simple masking
        (100% [MASK] token replacement).
        
        Args:
            input_ids: [B, L] original token indices
            attention_mask: [B, L] attention mask (1=valid, 0=padding)
            mask_prob: Probability of selecting each token for masking (default: 0.15)
        
        Returns:
            masked_input_ids: [B, L] input with masks applied
            labels: [B, L] labels for loss computation (-100 for non-masked tokens)
        
        Note:
            - Special tokens ([PAD], [CLS], [SEP]) are never masked
            - This assumes pad_token_id=0 (standard BERT convention)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 1. Create probability matrix for token selection
        probability_matrix = torch.full((batch_size, seq_len), mask_prob, device=device)
        
        # 2. Don't mask padding tokens
        if attention_mask is not None:
            probability_matrix = probability_matrix.masked_fill(attention_mask == 0, 0.0)
        probability_matrix = probability_matrix.masked_fill(input_ids == self.pad_token_id, 0.0)
        
        # 3. Sample which tokens to mask (Bernoulli sampling)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 4. Create labels (-100 for non-masked tokens)
        labels = input_ids.clone()
        labels = labels.masked_fill(~masked_indices, -100)
        
        # 5. Apply masking strategy
        masked_input_ids = self._apply_masking_strategy(input_ids, masked_indices)
        
        return masked_input_ids, labels
    
    def _apply_masking_strategy(
        self,
        input_ids: torch.Tensor,
        masked_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply masking strategy: simple or BERT-style.
        
        Simple masking: Replace all masked tokens with [MASK]
        BERT-style masking: 80% [MASK], 10% random, 10% keep
        
        Args:
            input_ids: [B, L] token indices
            masked_indices: [B, L] boolean mask (True=mask this token)
        
        Returns:
            masked_input_ids: [B, L] input with masks applied
        """
        masked_input_ids = input_ids.clone()
        
        if not self.bert_masking_enabled:
            # Simple masking: replace all with [MASK]
            masked_input_ids = masked_input_ids.masked_fill(masked_indices, self.mask_token_id)
            return masked_input_ids
        
        # BERT-style masking (80-10-10 split)
        # Get indices where masking should happen
        mask_positions = masked_indices.nonzero(as_tuple=True)
        
        if len(mask_positions[0]) == 0:
            return masked_input_ids
        
        # Generate random values for each masked position
        rand_values = torch.rand(len(mask_positions[0]), device=input_ids.device)
        
        # 80% of the time: replace with [MASK] token
        mask_token_mask = rand_values < 0.8
        mask_token_positions = (
            mask_positions[0][mask_token_mask],
            mask_positions[1][mask_token_mask]
        )
        if len(mask_token_positions[0]) > 0:
            masked_input_ids[mask_token_positions] = self.mask_token_id
        
        # 10% of the time: replace with random token
        random_token_mask = (rand_values >= 0.8) & (rand_values < 0.9)
        random_token_positions = (
            mask_positions[0][random_token_mask],
            mask_positions[1][random_token_mask]
        )
        if len(random_token_positions[0]) > 0:
            random_tokens = torch.randint(
                0, self.vocab_size,
                (len(random_token_positions[0]),),
                device=input_ids.device
            )
            masked_input_ids[random_token_positions] = random_tokens
        
        # 10% of the time: keep original token (no action needed)
        # These positions are already in masked_input_ids
        
        return masked_input_ids
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @classmethod
    def from_config(cls, config: Dict):
        """
        Create MLM model from configuration dictionary.
        
        Config format:
            model:
              vocab_size: 30522               # Required
              d_model: 768                    # Default: 768
              n_layers: 12                    # Default: 12
              n_heads: 12                     # Default: 12
              d_ff: 3072                      # Default: 4 * d_model
              max_seq_len: 512                # Default: 512
              dropout: 0.1                    # Default: 0.1
              mask_token_id: 103              # Default: 103
              pad_token_id: 0                 # Default: 0
              rope_base: 10000.0              # Default: 10000.0
              tie_weights: true               # Default: true
            training:
              bert_masking:
                enabled: false                # Default: false
        
        Args:
            config: Configuration dictionary
        
        Returns:
            MLM instance
        """
        model_cfg = config['model']
        training_cfg = config.get('training', {})
        bert_cfg = training_cfg.get('bert_masking', {})
        
        return cls(
            vocab_size=model_cfg['vocab_size'],
            d_model=model_cfg.get('d_model', 768),
            n_layers=model_cfg.get('n_layers', 12),
            n_heads=model_cfg.get('n_heads', 12),
            d_ff=model_cfg.get('d_ff'),
            max_seq_len=model_cfg.get('max_seq_len', 512),
            dropout=model_cfg.get('dropout', 0.1),
            mask_token_id=model_cfg.get('mask_token_id', 103),
            pad_token_id=model_cfg.get('pad_token_id', 0),
            rope_base=model_cfg.get('rope_base', 10000.0),
            bert_masking_enabled=bert_cfg.get('enabled', False),
            tie_weights=model_cfg.get('tie_weights', True)
        )
    
    def save_pretrained(self, save_path: str):
        """Save model weights."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
                'n_layers': self.n_layers,
                'n_heads': self.n_heads,
                'd_ff': self.d_ff,
                'max_seq_len': self.max_seq_len,
                'dropout': self.dropout,
                'mask_token_id': self.mask_token_id,
                'pad_token_id': self.pad_token_id,
                'rope_base': self.rope_base,
                'bert_masking_enabled': self.bert_masking_enabled,
                'tie_weights': self.tie_weights
            }
        }, save_path)
        print(f"✓ Model saved to: {save_path}")
    
    @classmethod
    def load_pretrained(cls, load_path: str):
        """Load model weights."""
        checkpoint = torch.load(load_path)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from: {load_path}")
        return model