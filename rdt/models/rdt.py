"""RDT Model Architecture (Refactored with RoPE & Transformer I/O)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional

# ============================================================================
# Rotary Position Embedding (RoPE)
# ============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) as described in RoFormer paper.
    Reference: https://arxiv.org/abs/2104.09864
    
    RoPE encodes position by rotating query/key vectors in pairs.
    This implementation caches sin/cos values for efficiency.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies: theta_i = base^(-2i/d) for i in [0, d/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Cache for sin/cos values
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        
        # Build initial cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Precompute and cache sin/cos values for positions [0, seq_len)"""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            
            # Position indices: [0, 1, 2, ..., seq_len-1]
            t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            
            # Compute outer product: [seq_len, dim/2]
            # freqs[pos, i] = pos * theta_i
            freqs = torch.outer(t, self.inv_freq)
            
            # Compute sin and cos
            # We'll interleave these: [cos, cos, ...], [sin, sin, ...]
            # Shape: [seq_len, dim]
            emb = torch.cat([freqs, freqs], dim=-1)  # Duplicate for interleaving
            
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half the hidden dims of the input.
        
        For x = [x0, x1, x2, x3, ...], returns [-x1, x0, -x3, x2, ...]
        This corresponds to rotating by 90 degrees in each 2D plane.
        """
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, x: torch.Tensor, seq_dim: int = 1) -> torch.Tensor:
        """
        Apply rotary position embedding to input tensor.
        
        Args:
            x: Input tensor of shape [..., seq_len, ..., dim]
            seq_dim: Dimension index for sequence length (default: 1 for [B, L, ...])
        
        Returns:
            Tensor with same shape as x, with rotary embeddings applied
        """
        seq_len = x.shape[seq_dim]
        
        # Rebuild cache if needed
        if seq_len > self._seq_len_cached:
            self._build_cache(seq_len)
        
        # Get cached values for current sequence length
        # Shape: [seq_len, dim]
        cos = self._cos_cached[:seq_len].to(x.device)
        sin = self._sin_cached[:seq_len].to(x.device)
        
        # Reshape for broadcasting
        # Need to match x's shape: [..., seq_len, ..., dim]
        # We'll add dimensions as needed
        shape = [1] * x.ndim
        shape[seq_dim] = seq_len
        shape[-1] = self.dim
        
        cos = cos.view(shape)
        sin = sin.view(shape)
        
        # Apply rotation: x_rotated = x * cos + rotate_half(x) * sin
        return x * cos + self._rotate_half(x) * sin


# ============================================================================
# Transformer Encoder Processor
# ============================================================================

class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer with RoPE
    
    Architecture: Self-Attention -> Add & Norm -> FFN -> Add & Norm
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
    
    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [B, L, d_model]
            key_padding_mask: [B, L] where True means ignore (padding)
        
        Returns:
            [B, L, d_model]
        """
        # Self-Attention
        res = x
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm, key_padding_mask=key_padding_mask)
        x = res + self.dropout1(attn_out)
        
        # Feed-Forward
        res = x
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = res + self.dropout2(ffn_out)
        
        return x


class TransformerProcessor(nn.Module):
    """
    Transformer Encoder for input/output processing.
    
    Uses standard transformer encoder layers with RoPE positional encoding.
    
    Args:
        d_model: Model dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        rope_layer: Rotary position embedding module
        dropout: Dropout probability
    """
    def __init__(
        self, 
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        rope_layer: nn.Module,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Build transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                rope_layer=rope_layer,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [B, L, d_model]
            key_padding_mask: [B, L] where True means ignore (padding)
        
        Returns:
            [B, L, d_model]
        """
        hidden = x
        
        for layer in self.layers:
            hidden = layer(hidden, key_padding_mask=key_padding_mask)
        
        return self.norm(hidden)


# ============================================================================
# Core RDT Components
# ============================================================================

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Classic DDPM Style: Sinusoidal -> Linear -> SiLU -> Linear
    """
    def __init__(self, hidden_dim, frequency_embedding_size=256, scale_factor = 20.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.gate_scale = 20.0
        self.scale_factor = scale_factor

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        t: [Batch] (1D Tensor)
        dim: Output dimension
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        
        if t.dim() == 2:
            t = t.squeeze(-1)
            
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t * self.scale_factor / self.gate_scale, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb.unsqueeze(1)


class AdaptiveLayerNorm(nn.Module):
    """
    Diffusion Style Adaptive LayerNorm (AdaLN)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_model, d_model * 2)
        
        # Zero-init for identity at start
        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x, emb):
        if emb.dim() == 3:
            emb = emb.squeeze(1)
            
        gamma, beta = self.proj(emb).chunk(2, dim=1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        
        return self.norm(x) * (1 + gamma) + beta

class RoPESelfAttention(nn.Module):
    """
    Self-Attention with RoPE applied only to Q and K (NOT to V)
    Optimized with F.scaled_dot_product_attention
    """
    def __init__(self, d_model: int, n_heads: int, rope_layer: nn.Module, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope_layer
        
        # QKV projections
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Dropout for SDPA
        self.dropout_p = dropout
        self.out_dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: [B, L, D]
            key_padding_mask: [B, L] where True means ignore (padding)
        """
        B, L, D = x.shape
        
        # 1. Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # 2. Split heads: [B, L, n_heads, head_dim]
        q = q.view(B, L, self.n_heads, self.head_dim)
        k = k.view(B, L, self.n_heads, self.head_dim)
        v = v.view(B, L, self.n_heads, self.head_dim)
        
        # 3. Apply RoPE to Q and K ONLY
        q = self.rope(q, seq_dim=1)
        k = self.rope(k, seq_dim=1)
        
        # 4. Transpose for attention: [B, n_heads, L, head_dim]
        # F.scaled_dot_product_attention expects [B, H, L, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 5. Scaled Dot Product Attention (Flash Attention applied automatically if available)
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: [B, L] -> [B, 1, 1, L] for broadcasting
            # SDPA supports boolean mask where True = masked out (ignore)
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        
        # 6. Concatenate heads: [B, L, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        
        # 7. Output projection
        output = self.out_proj(attn_output)
        output = self.out_dropout(output)
        
        return output


class RoPECrossAttention(nn.Module):
    """
    Cross-Attention with RoPE applied only to Q and K
    Optimized with F.scaled_dot_product_attention
    """
    def __init__(self, d_model: int, n_heads: int, rope_layer: nn.Module, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope_layer
        
        # Separate Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Dropout for SDPA
        self.dropout_p = dropout
        self.out_dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            query: [B, L_q, D]
            key: [B, L_kv, D]
            value: [B, L_kv, D]
            key_padding_mask: [B, L_kv]
        """
        B, L_q, D = query.shape
        L_kv = key.shape[1]
        
        # 1. Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 2. Split heads
        q = q.view(B, L_q, self.n_heads, self.head_dim)
        k = k.view(B, L_kv, self.n_heads, self.head_dim)
        v = v.view(B, L_kv, self.n_heads, self.head_dim)
        
        # 3. Apply RoPE to Q and K only
        q = self.rope(q, seq_dim=1)
        k = self.rope(k, seq_dim=1)
        
        # 4. Transpose for attention: [B, n_heads, L, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 5. Scaled Dot Product Attention
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: [B, L_kv] -> [B, 1, 1, L_kv]
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        
        # 6. Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, D)
        
        # 7. Output projection
        output = self.out_proj(attn_output)
        output = self.out_dropout(output)
        
        return output


class DirectionalRecursiveBlock(nn.Module):
    """Self-Attention + Optional Cross-Attention Block with AdaLN and RoPE"""
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_ff: int, 
        rope_layer: nn.Module,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Self-Attention with RoPE and AdaLN
        self.self_attn = RoPESelfAttention(d_model, n_heads, rope_layer, dropout)
        self.norm1 = AdaptiveLayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-Attention with RoPE and AdaLN
        self.cross_attn = RoPECrossAttention(d_model, n_heads, rope_layer, dropout)
        self.norm2 = AdaptiveLayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-Forward with AdaLN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm3 = AdaptiveLayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, noise_emb, context=None, src_key_padding_mask=None, context_key_padding_mask=None):
        """
        Args:
            x: [B, L, D] - input hidden states
            noise_emb: [B, 1, D] - noise level embedding for AdaLN
            context: [B, L_ctx, D] - optional context for cross-attention
            src_key_padding_mask: [B, L] - self-attention mask (True = ignore)
            context_key_padding_mask: [B, L_ctx] - cross-attention mask (True = ignore)
        
        Returns:
            [B, L, D] - output hidden states
        """
        # Self-Attention with AdaLN
        res = x
        x_norm = self.norm1(x, noise_emb)
        
        # RoPE is applied inside self_attn to Q and K only
        attn_out = self.self_attn(x_norm, key_padding_mask=src_key_padding_mask)
        x = res + self.dropout1(attn_out)
        
        # Cross-Attention with AdaLN (if context provided)
        if context is not None:
            res = x
            x_norm = self.norm2(x, noise_emb)
            
            # RoPE is applied inside cross_attn to Q and K only
            attn_out = self.cross_attn(
                query=x_norm,
                key=context,
                value=context,
                key_padding_mask=context_key_padding_mask
            )
            x = res + self.dropout2(attn_out)
        
        # Feed-Forward with AdaLN
        res = x
        x_norm = self.norm3(x, noise_emb)
        ffn_out = self.ffn(x_norm)
        x = res + self.dropout3(ffn_out)
        
        return x


class GateMLP(nn.Module):
    """
    Gate with residual prediction using concatenated features
    """
    def __init__(self, d_model: int, hidden_dim: int = 512, num_layers: int = 3, 
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gate_scale = 20.0 # for numerical stability
        
        # Multi-head attention pooling
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.normal_(self.query, std=0.02)
        
        # Fusion layer
        self.fusion = nn.Linear(d_model * 2, hidden_dim)
        
        # N-layer MLP with residual connections
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'linear': nn.Linear(hidden_dim, hidden_dim),
                'norm': nn.LayerNorm(hidden_dim),
                'dropout': nn.Dropout(dropout)
            }))
        
        # Separate output heads
        self.first_step_proj = nn.Linear(hidden_dim, 1)
        self.delta_proj = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
        
        for layer_dict in self.layers:
            nn.init.xavier_uniform_(layer_dict['linear'].weight)
            nn.init.zeros_(layer_dict['linear'].bias)
        
        nn.init.xavier_uniform_(self.first_step_proj.weight)
        nn.init.constant_(self.first_step_proj.bias, self.gate_scale / 2)
        
        nn.init.xavier_uniform_(self.delta_proj.weight)
        nn.init.constant_(self.delta_proj.bias, 1.0)
    
    def forward(self, x, mask=None, prev_pooled=None, prev_gate=None):
        x = x.detach()
        batch_size = x.size(0)
        
        # Multi-head attention pooling
        query = self.query.expand(batch_size, -1, -1)
        key_padding_mask = (mask == 0) if mask is not None else None
        
        pooled, _ = self.attention(
            query=query,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        pooled = pooled.squeeze(1)
        
        # Concatenate with previous pooled features
        if prev_pooled is None:
            prev_pooled = torch.zeros_like(pooled)

        prev_pooled = prev_pooled.detach()
        
        combined = torch.cat([pooled, prev_pooled], dim=-1)
        
        # Fusion layer
        h = self.fusion(combined)
        h = nn.functional.gelu(h)
        
        # N-layer MLP processing
        for i, layer_dict in enumerate(self.layers):
            h_new = layer_dict['linear'](h)
            h_new = layer_dict['norm'](h_new)
            h_new = nn.functional.gelu(h_new)
            h_new = layer_dict['dropout'](h_new)
            
            if i > 0:
                h = h + h_new
            else:
                h = h_new
        
        # Output prediction
        if prev_gate is None:
            raw = self.first_step_proj(h)
            decrease = nn.functional.softplus(raw)
            gate_output = torch.clamp(self.gate_scale - decrease, min=0.0)
        else:
            raw = self.delta_proj(h)
            delta = nn.functional.softplus(raw)
            gate_output = torch.clamp(prev_gate - delta, min=0.0)
        
        return gate_output, pooled


# ============================================================================
# Main RDT Model (Refactored with RoPE & Transformer I/O)
# ============================================================================

class RDT(nn.Module):
    """
    Recursive Denoising Transformer with:
    - RoPE for positional encoding
    - Transformer Encoder for input/output processing
    - AdaLN-based recursive blocks
    """
    def __init__(
        self, 
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        # Transformer I/O Configuration
        input_processor_layers: int = 1,
        output_processor_layers: int = 1,
        # Gate Configuration
        gate_hidden_dim: int = 512,
        gate_num_layers: int = 3,
        gate_num_heads: int = 8,
        gate_dropout: float = 0.3,
        # RoPE Configuration
        rope_base: float = 10000.0,
        # Training
        total_steps: int = 20,
        gradient_checkpointing: bool = False,
        weight_tying: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.total_steps = total_steps
        self.gradient_checkpointing = gradient_checkpointing
        self.weight_tying = weight_tying
        
        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Rotary Position Embedding (RoPE)
        # Each attention head gets its own rotary embedding
        head_dim = d_model // n_heads
        self.rope = RotaryPositionalEmbedding(
            dim=head_dim,  # Use head dimension, not model dimension!
            max_seq_len=max_seq_len,
            base=rope_base
        )
        
        # Input Processor (Transformer Encoder)
        self.input_processor = TransformerProcessor(
            d_model=d_model,
            n_layers=input_processor_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            rope_layer=self.rope,
            dropout=dropout
        )
        
        # Noise Level Embedding
        self.noise_emb = TimestepEmbedder(d_model, frequency_embedding_size=d_model, scale_factor=total_steps)
        
        # Recursive Encoder (AdaLN Blocks with RoPE)
        self.encoder_layers = nn.ModuleList([
            DirectionalRecursiveBlock(d_model, n_heads, d_ff, self.rope, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # Output Processor (Transformer Encoder)
        self.output_processor = TransformerProcessor(
            d_model=d_model,
            n_layers=output_processor_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            rope_layer=self.rope,
            dropout=dropout
        )
        
        # Output Projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Gate Network
        self.gate = GateMLP(
            d_model=d_model,
            hidden_dim=gate_hidden_dim,
            num_layers=gate_num_layers,
            num_heads=gate_num_heads,
            dropout=gate_dropout
        )
        
        # Weight Tying (conditional based on config)
        if self.weight_tying:
            self.output_projection.weight = self.token_embedding.weight
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def tie_weights(self):
        """Tie output projection weights with token embedding (only if weight_tying is enabled)"""
        if self.weight_tying:
            self.output_projection.weight = self.token_embedding.weight
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def encode(self, tokens, attention_mask=None):
        """
        Initial encoding: tokens → h_0
        
        Args:
            tokens: [B, L] token indices
            attention_mask: [B, L] attention mask (1=valid, 0=padding)
        
        Returns:
            h_0: [B, L, D] initial hidden states
        """
        # 1. Token embedding
        x = self.token_embedding(tokens) * math.sqrt(self.d_model)
        
        # 2. Input processing (Transformer Encoder with RoPE)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        h_0 = self.input_processor(x, key_padding_mask=key_padding_mask)
        
        return h_0


    def forward_step(self, hidden, noise_level, attention_mask=None):
        """
        Single recursive step: h_i → h_{i+1}
        Does NOT call gate network.
        
        Args:
            hidden: [B, L, D] current hidden states h_i
            noise_level: [B, 1] noise level for this step
            attention_mask: [B, L] attention mask (1=valid, 0=padding)
        
        Returns:
            h_{i+1}: [B, L, D] next hidden states
        """
        # 1. Create noise embedding
        noise_vec = self.noise_emb(noise_level)
        
        # 2. Prepare mask
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        
        # 3. Apply recursive encoder layers with AdaLN
        h_next = hidden
        for layer in self.encoder_layers:
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(h, n_emb, sm):
                        return module(h, noise_emb=n_emb, context=None,
                                    src_key_padding_mask=sm,
                                    context_key_padding_mask=None)
                    return custom_forward
                
                h_next = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    h_next, noise_vec, src_key_padding_mask,
                    use_reentrant=False
                )
            else:
                h_next = layer(
                    h_next,
                    noise_emb=noise_vec,
                    context=None,
                    src_key_padding_mask=src_key_padding_mask,
                    context_key_padding_mask=None
                )
        
        return h_next


    def decode(self, hidden, attention_mask=None):
        """
        Output decoder: h_n → logits
        
        Args:
            hidden: [B, L, D] final hidden states h_n
            attention_mask: [B, L] attention mask (1=valid, 0=padding)
        
        Returns:
            logits: [B, L, vocab_size]
        """
        # 1. Output processing (Transformer Encoder with RoPE)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        processed = self.output_processor(hidden, key_padding_mask=key_padding_mask)
        
        # 2. Project to vocabulary
        logits = self.output_projection(processed)
        
        return logits


    def forward(
        self,
        x,
        attention_mask=None,
        is_first_step=True,
        last_gate_score=None,
        last_pooled=None,
        gt_timestep=None,
        sampling_prob=0.0
    ):
        """
        Forward pass with gate network integration.
        
        If is_first_step=True:
            x [B, L] tokens → h_0 → gate_0 → h_1 → gate_1
            Returns: (h_1, gate_1, pooled_1)
        
        If is_first_step=False:
            x [B, L, D] hidden (h_i) → h_{i+1} → gate_{i+1}
            Returns: (h_{i+1}, gate_{i+1}, pooled_{i+1})
        
        Args:
            x: [B, L] (tokens) if is_first_step else [B, L, D] (hidden)
            attention_mask: [B, L] attention mask (1=valid, 0=padding)
            is_first_step: whether this is the first forward pass
            last_gate_score: [B, 1] previous gate prediction (required if not first step)
            last_pooled: [B, D] previous pooled features (required if not first step)
            gt_timestep: [B, 1] ground truth timestep for scheduled sampling
            sampling_prob: float or tensor, probability of using GT vs predicted gate
        
        Returns:
            hidden: [B, L, D] output hidden states (h_1 or h_{i+1})
            gate_pred: [B, 1] gate prediction for next step
            pooled: [B, D] pooled features for next step
        """
        
        if is_first_step:
            # ============================================================
            # FIRST STEP: tokens → h_0 → gate_0 → h_1 → gate_1
            # ============================================================
            
            # 1. Encode: tokens → h_0
            h_0 = self.encode(x, attention_mask)
            
            # 2. Initial gate prediction: gate(h_0, None, None)
            gate_0, pooled_0 = self.gate(
                h_0,
                attention_mask,
                prev_pooled=None,
                prev_gate=None
            )
            
            # 3. Scheduled sampling for noise level
            if self.training and gt_timestep is not None:
                # XLA-safe soft mixing
                noise_0 = sampling_prob * gt_timestep.detach() + (1.0 - sampling_prob) * gate_0.detach()
            else:
                noise_0 = gate_0.detach()
            
            # 4. Transform: h_0 → h_1
            h_1 = self.forward_step(h_0, noise_0, attention_mask)
            
            # 5. Next gate prediction: gate(h_1, pooled_0, gate_0)
            gate_1, pooled_1 = self.gate(
                h_1,
                attention_mask,
                prev_pooled=pooled_0,
                prev_gate=gate_0
            )
            
            return h_1, gate_1, pooled_1
        
        else:
            # ============================================================
            # SUBSEQUENT STEPS: h_i → h_{i+1} → gate_{i+1}
            # ============================================================
            
            h_i = x
            
            # 1. Scheduled sampling for noise level
            if self.training and gt_timestep is not None:
                # XLA-safe soft mixing
                noise_i = sampling_prob * gt_timestep.detach() + (1.0 - sampling_prob) * last_gate_score.detach()
            else:
                noise_i = last_gate_score.detach()
            
            # 2. Transform: h_i → h_{i+1}
            h_next = self.forward_step(h_i, noise_i, attention_mask)
            
            # 3. Next gate prediction: gate(h_{i+1}, last_pooled, last_gate_score)
            gate_next, pooled_next = self.gate(
                h_next,
                attention_mask,
                prev_pooled=last_pooled,
                prev_gate=last_gate_score
            )
            
            return h_next, gate_next, pooled_next
        
    def inference(self, tokens, attention_mask=None, max_steps=20, threshold=0.02, return_steps=False):
        """
        Clean inference using the standardized methods.
        Each sample is processed independently for exact step-by-step control.
        
        Args:
            tokens: [B, L] input tokens
            attention_mask: [B, L] attention mask (1=valid, 0=padding)
            max_steps: maximum recursive steps
            threshold: stop when gate score < threshold (per sample)
            return_steps: if True, return per-sample step counts
        
        Returns:
            output_tokens: [B, L] predicted tokens
            steps_taken: [B] (if return_steps=True) or float (mean steps)
        """
        self.eval()
        batch_size = tokens.size(0)
        device = tokens.device
        
        # Storage for results
        all_output_tokens = []
        all_steps_taken = []
        
        with torch.no_grad():
            # Process each sample independently
            for sample_idx in range(batch_size):
                # ================================================================
                # Extract single sample
                # ================================================================
                sample_tokens = tokens[sample_idx:sample_idx+1]  # [1, L]
                sample_mask = attention_mask[sample_idx:sample_idx+1] if attention_mask is not None else None
                
                # ================================================================
                # STEP 0: Initial encoding
                # ================================================================
                # h_0 = encode(tokens)
                h_0 = self.encode(sample_tokens, sample_mask)
                
                # gate_0, pooled_0 = gate(h_0, None, None)
                gate_0, pooled_0 = self.gate(
                    h_0,
                    sample_mask,
                    prev_pooled=None,
                    prev_gate=None
                )
                
                # ================================================================
                # STEP 1: First transformation
                # ================================================================
                # h_1 = forward_step(h_0, noise=gate_0)
                h_1 = self.forward_step(h_0, gate_0.detach(), sample_mask)
                
                # gate_1, pooled_1 = gate(h_1, pooled_0, gate_0)
                gate_1, pooled_1 = self.gate(
                    h_1,
                    sample_mask,
                    prev_pooled=pooled_0,
                    prev_gate=gate_0
                )
                
                # Current state
                h_current = h_1
                gate_current = gate_1
                pooled_current = pooled_1
                steps_taken = 1
                
                # ================================================================
                # STEP 2 to max_steps: Recursive steps
                # ================================================================
                for step in range(2, max_steps + 1):
                    # Check convergence BEFORE next transformation
                    gate_score = gate_current.squeeze(-1).item()
                    
                    if gate_score < threshold:
                        break
                    
                    # h_{i+1} = forward_step(h_i, noise=gate_i)
                    h_next = self.forward_step(h_current, gate_current.detach(), sample_mask)
                    
                    # gate_{i+1}, pooled_{i+1} = gate(h_{i+1}, pooled_i, gate_i)
                    gate_next, pooled_next = self.gate(
                        h_next,
                        sample_mask,
                        prev_pooled=pooled_current,
                        prev_gate=gate_current
                    )
                    
                    # Update state
                    h_current = h_next
                    gate_current = gate_next
                    pooled_current = pooled_next
                    steps_taken = step
                
                # ================================================================
                # FINAL: Decode
                # ================================================================
                # logits = decode(h_final)
                logits = self.decode(h_current, sample_mask)
                
                # output_tokens = argmax(logits)
                sample_output = logits.argmax(dim=-1)  # [1, L]
                
                # ================================================================
                # Store results
                # ================================================================
                all_output_tokens.append(sample_output)
                all_steps_taken.append(steps_taken)
        
        # ================================================================
        # Batch results
        # ================================================================
        output_tokens = torch.cat(all_output_tokens, dim=0)  # [B, L]
        steps_taken_tensor = torch.tensor(all_steps_taken, device=device)  # [B]
        
        if return_steps:
            return output_tokens, steps_taken_tensor
        else:
            return output_tokens, steps_taken_tensor.float().mean().item()