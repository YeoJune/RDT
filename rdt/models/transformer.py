"""
Shared Transformer Backbone with RoPE and FlashAttention

This module provides reusable transformer components for MLM, CMLM, and RDT models.
All models use the same backbone with RoPE positional encoding and optimized attention.

Key Components:
- RotaryPositionalEmbedding: RoPE implementation with caching
- RoPESelfAttention: Self-attention with RoPE and FlashAttention
- RoPECrossAttention: Cross-attention with RoPE and FlashAttention
- TransformerEncoderLayer: Single transformer encoder layer
- TransformerEncoder: Multi-layer transformer encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# ============================================================================
# Rotary Position Embedding (RoPE)
# ============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) as described in RoFormer paper.
    Reference: https://arxiv.org/abs/2104.09864
    
    RoPE encodes absolute position information by rotating query/key vectors
    in pairs, which naturally provides relative position information.
    
    Key properties:
    - Maintains relative position information
    - Works well with any sequence length (via extrapolation)
    - More efficient than learned position embeddings
    
    Implementation details:
    - Precomputes and caches sin/cos values for efficiency
    - Applies rotation to Q and K (NOT to V)
    - Uses standard rotation formula: x_rot = x * cos + rotate_half(x) * sin
    
    Args:
        dim: Dimension of the position embedding (typically head_dim)
        max_seq_len: Maximum sequence length to cache (default: 2048)
        base: Base for frequency calculation (default: 10000.0)
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute inverse frequencies: theta_i = base^(-2i/d) for i in [0, d/2)
        # These are the rotation frequencies for each dimension pair
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Cache for sin/cos values (built on first forward pass)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        
        # Build initial cache up to max_seq_len
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """
        Precompute and cache sin/cos values for positions [0, seq_len).
        
        This is called once during initialization and whenever we encounter
        a longer sequence than previously cached.
        
        Args:
            seq_len: Sequence length to cache
        """
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            
            # Position indices: [0, 1, 2, ..., seq_len-1]
            t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            
            # Compute outer product: [seq_len, dim/2]
            # freqs[pos, i] = pos * theta_i = pos * base^(-2i/d)
            freqs = torch.outer(t, self.inv_freq)
            
            # Duplicate for interleaving: [seq_len, dim]
            # This allows us to apply rotation to both elements of each pair
            emb = torch.cat([freqs, freqs], dim=-1)
            
            # Cache sin and cos
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half the hidden dims of the input.
        
        For x = [x0, x1, x2, x3, ...], returns [-x1, x0, -x3, x2, ...]
        This corresponds to rotating by 90 degrees in each 2D plane.
        
        Args:
            x: Input tensor [..., dim]
            
        Returns:
            Rotated tensor with same shape
        """
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, x: torch.Tensor, seq_dim: int = 1) -> torch.Tensor:
        """
        Apply rotary position embedding to input tensor.
        
        The rotation formula is:
            x_rotated = x * cos(theta * pos) + rotate_half(x) * sin(theta * pos)
        
        Args:
            x: Input tensor of shape [..., seq_len, ..., dim]
            seq_dim: Dimension index for sequence length (default: 1 for [B, L, ...])
        
        Returns:
            Tensor with same shape as x, with rotary embeddings applied
        """
        seq_len = x.shape[seq_dim]
        
        # Rebuild cache if we encounter a longer sequence
        if seq_len > self._seq_len_cached:
            self._build_cache(seq_len)
        
        # Get cached sin/cos values for current sequence length
        # Shape: [seq_len, dim]
        cos = self._cos_cached[:seq_len].to(x.device)
        sin = self._sin_cached[:seq_len].to(x.device)
        
        # Reshape for broadcasting to match x's shape
        # We need to add dimensions to match [..., seq_len, ..., dim]
        shape = [1] * x.ndim
        shape[seq_dim] = seq_len
        shape[-1] = self.dim
        
        cos = cos.view(shape)
        sin = sin.view(shape)
        
        # Apply rotation: x_rotated = x * cos + rotate_half(x) * sin
        return x * cos + self._rotate_half(x) * sin


# ============================================================================
# Attention Mechanisms with RoPE and FlashAttention
# ============================================================================

class RoPESelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with RoPE and FlashAttention.
    
    This implementation combines three key optimizations:
    1. RoPE: Applied to Q and K for position encoding (NOT to V)
    2. FlashAttention: Automatic via F.scaled_dot_product_attention (PyTorch 2.0+)
    3. Fused QKV projection: Single linear layer for efficiency
    
    FlashAttention benefits (automatic when available):
    - 20-30% faster inference/training
    - 30-50% less memory usage
    - Mathematically equivalent to standard attention
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        rope_layer: RoPE module instance
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model: int, n_heads: int, rope_layer: nn.Module, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope_layer
        
        # Fused QKV projection (more efficient than 3 separate projections)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Dropout
        self.dropout_p = dropout
        self.out_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with RoPE and FlashAttention.
        
        Args:
            x: [B, L, d_model] input tensor
            key_padding_mask: [B, L] boolean mask where True means ignore (padding)
        
        Returns:
            [B, L, d_model] attention output
        """
        B, L, D = x.shape
        
        # 1. Project to Q, K, V
        qkv = self.qkv_proj(x)  # [B, L, 3*d_model]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [B, L, d_model]
        
        # 2. Split heads: [B, L, n_heads, head_dim]
        q = q.view(B, L, self.n_heads, self.head_dim)
        k = k.view(B, L, self.n_heads, self.head_dim)
        v = v.view(B, L, self.n_heads, self.head_dim)
        
        # 3. Apply RoPE to Q and K ONLY (not to V)
        # RoPE provides position information through rotation
        q = self.rope(q, seq_dim=1)
        k = self.rope(k, seq_dim=1)
        
        # 4. Transpose for attention: [B, n_heads, L, head_dim]
        # F.scaled_dot_product_attention expects [B, H, L, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 5. Prepare attention mask for FlashAttention
        attn_mask = None
        if key_padding_mask is not None:
            # Convert boolean mask to attention mask
            # [B, L] -> [B, 1, 1, L]
            attn_mask = torch.zeros(B, 1, 1, L, dtype=q.dtype, device=q.device)
            attn_mask.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        
        # 6. Scaled Dot Product Attention with FlashAttention
        # PyTorch 2.0+ automatically uses FlashAttention when available
        # Falls back to standard implementation if not available
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        
        # 7. Concatenate heads: [B, L, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        
        # 8. Output projection
        output = self.out_proj(attn_output)
        output = self.out_dropout(output)
        
        return output


class RoPECrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention with RoPE and FlashAttention.
    
    Similar to RoPESelfAttention, but with separate Q (from target) and K/V (from source).
    Used in encoder-decoder architectures or when attending to a different sequence.
    
    Key differences from self-attention:
    - Separate Q projection (from x) and K/V projections (from context)
    - RoPE applied independently to Q and K
    - Source and target can have different sequence lengths
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        rope_layer: RoPE module instance
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(self, d_model: int, n_heads: int, rope_layer: nn.Module, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope_layer
        
        # Separate Q, K, V projections (cross-attention requires this)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Dropout
        self.dropout_p = dropout
        self.out_dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with RoPE and FlashAttention.
        
        Args:
            x: [B, L_target, d_model] target sequence (queries)
            context: [B, L_source, d_model] source sequence (keys/values)
            key_padding_mask: [B, L_source] boolean mask for source padding
        
        Returns:
            [B, L_target, d_model] attention output
        """
        B, L_target, D = x.shape
        L_source = context.shape[1]
        
        # 1. Project to Q, K, V
        q = self.q_proj(x)  # [B, L_target, d_model]
        k = self.k_proj(context)  # [B, L_source, d_model]
        v = self.v_proj(context)  # [B, L_source, d_model]
        
        # 2. Split heads
        q = q.view(B, L_target, self.n_heads, self.head_dim)
        k = k.view(B, L_source, self.n_heads, self.head_dim)
        v = v.view(B, L_source, self.n_heads, self.head_dim)
        
        # 3. Apply RoPE to Q and K
        q = self.rope(q, seq_dim=1)
        k = self.rope(k, seq_dim=1)
        
        # 4. Transpose for attention: [B, n_heads, L, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 5. Prepare attention mask
        attn_mask = None
        if key_padding_mask is not None:
            # [B, L_source] -> [B, 1, 1, L_source]
            attn_mask = torch.zeros(B, 1, 1, L_source, dtype=q.dtype, device=q.device)
            attn_mask.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        
        # 6. Scaled Dot Product Attention with FlashAttention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        
        # 7. Concatenate heads: [B, L_target, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_target, D)
        
        # 8. Output projection
        output = self.out_proj(attn_output)
        output = self.out_dropout(output)
        
        return output


# ============================================================================
# Transformer Encoder Components
# ============================================================================

class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer with RoPE.
    
    Architecture (Pre-LN variant):
        x -> LayerNorm -> Self-Attention -> Add -> LayerNorm -> FFN -> Add -> output
    
    Pre-LN (used here) vs Post-LN:
    - Pre-LN: More stable training, easier to train deep networks
    - Post-LN: Original transformer, sometimes better final performance
    
    Feed-Forward Network:
        Linear(d_model -> d_ff) -> GELU -> Dropout -> Linear(d_ff -> d_model)
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension (typically 4 * d_model)
        rope_layer: RoPE module instance
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        rope_layer: nn.Module,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Self-Attention with RoPE
        self.self_attn = RoPESelfAttention(d_model, n_heads, rope_layer, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through encoder layer.
        
        Args:
            x: [B, L, d_model] input tensor
            key_padding_mask: [B, L] boolean mask where True means ignore (padding)
        
        Returns:
            [B, L, d_model] output tensor
        """
        # Self-Attention block
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = residual + self.dropout1(x)
        
        # Feed-Forward block
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout2(x)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Multi-Layer Transformer Encoder with RoPE.
    
    RoPE can be provided externally for sharing, or created internally.
    """
    
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
        rope_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Create or use provided RoPE layer
        if rope_layer is not None:
            # Use shared RoPE
            self.rope = rope_layer
        else:
            # Create own RoPE
            head_dim = d_model // n_heads
            self.rope = RotaryPositionalEmbedding(
                dim=head_dim,
                max_seq_len=max_seq_len,
                base=rope_base
            )
        
        # Build transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                rope_layer=self.rope,  # 항상 self.rope 사용
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights using Xavier uniform initialization.
        
        Xavier initialization helps maintain gradient flow in deep networks
        by keeping variance consistent across layers.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through all encoder layers.
        
        Args:
            x: [B, L, d_model] input tensor
            key_padding_mask: [B, L] boolean mask where True means ignore (padding)
        
        Returns:
            [B, L, d_model] output tensor
        """
        hidden = x
        
        # Pass through all layers
        for layer in self.layers:
            hidden = layer(hidden, key_padding_mask=key_padding_mask)
        
        # Final normalization
        return self.norm(hidden)
    
    def get_rope_layer(self) -> nn.Module:
        """Get the RoPE layer for reuse in other components."""
        return self.rope

