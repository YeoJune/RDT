"""
RDT Models Module

This module provides transformer-based language models with shared backbone.

Models:
- MLM: Masked Language Model (BERT-style)
- CMLM: Conditional Masked Language Model (Mask-Predict)
- MDLM: Masked Discrete Language Model (uniform masking)
- RDT: Recursive Denoising Transformer (iterative refinement)

Shared Components (from transformer.py):
- RotaryPositionalEmbedding: RoPE implementation
- RoPESelfAttention: Self-attention with RoPE and FlashAttention
- RoPECrossAttention: Cross-attention with RoPE and FlashAttention
- TransformerEncoderLayer: Single encoder layer
- TransformerEncoder: Multi-layer encoder
"""

from .rdt import RDT
from .mlm import MLM
from .cmlm import CMLM
from .mdlm import MDLM
from .bert_init import initialize_rdt_with_bert, load_bert_weights_to_rdt

# Import shared transformer components
from .transformer import (
    RotaryPositionalEmbedding,
    RoPESelfAttention,
    RoPECrossAttention,
    TransformerEncoderLayer,
    TransformerEncoder,
)

__all__ = [
    # Models
    'RDT',
    'MLM',
    'CMLM',
    'MDLM',
    
    # Utilities
    'initialize_rdt_with_bert',
    'load_bert_weights_to_rdt',
    
    # Shared Transformer Components
    'RotaryPositionalEmbedding',
    'RoPESelfAttention',
    'RoPECrossAttention',
    'TransformerEncoderLayer',
    'TransformerEncoder',
]

__version__ = '0.1.0'