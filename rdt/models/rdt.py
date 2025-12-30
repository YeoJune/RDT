"""RDT Model Architecture (Refactored with RoPE & MLP I/O)"""

import torch
import torch.nn as nn
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
# Configurable MLP Processor
# ============================================================================

class MLPProcessor(nn.Module):
    """
    Configurable Multi-Layer Perceptron for input/output processing.
    
    Architecture: d_model -> hidden_dims[0] -> ... -> hidden_dims[-1] -> d_model
    
    Args:
        d_model: Input/output dimension
        hidden_dims: List of hidden layer dimensions e.g., [256] -> 512->256->512 (2 layers)
        dropout: Dropout probability
        activation: Activation function (default: GELU)
    """
    def __init__(
        self, 
        d_model: int, 
        hidden_dims: List[int], 
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dims = hidden_dims
        
        # Build layers
        layers = []
        
        # Input projection: d_model -> hidden_dims[0]
        layers.append(nn.Linear(d_model, hidden_dims[0]))
        layers.append(self._get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden layers: hidden_dims[i] -> hidden_dims[i+1]
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output projection: hidden_dims[-1] -> d_model
        layers.append(nn.Linear(hidden_dims[-1], d_model))
        
        self.mlp = nn.Sequential(*layers)

        self.norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh()
        }
        return activations.get(name.lower(), nn.GELU())
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform"""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
        last_linear = self.mlp[-1]
        
        if isinstance(last_linear, nn.Linear):
            # Zero Init
            nn.init.zeros_(last_linear.weight)
            
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [B, L, d_model]
        
        Returns:
            [B, L, d_model]
        """
        return x + self.mlp(self.norm(x))


# ============================================================================
# Core RDT Components (Unchanged)
# ============================================================================

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Classic DDPM Style: Sinusoidal -> Linear -> SiLU -> Linear
    """
    def __init__(self, hidden_dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.frequency_embedding_size = frequency_embedding_size

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
        t_freq = self.timestep_embedding(t * 50, self.frequency_embedding_size)
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
    This is the correct implementation as per RoFormer paper and all standard LLMs
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
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: [B, L, D]
            key_padding_mask: [B, L] where True means ignore (padding)
        
        Returns:
            [B, L, D]
        """
        B, L, D = x.shape
        
        # 1. Project to Q, K, V
        qkv = self.qkv_proj(x)  # [B, L, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [B, L, D]
        
        # 2. Split heads: [B, L, D] -> [B, L, n_heads, head_dim]
        q = q.view(B, L, self.n_heads, self.head_dim)
        k = k.view(B, L, self.n_heads, self.head_dim)
        v = v.view(B, L, self.n_heads, self.head_dim)
        
        # 3. Apply RoPE to Q and K ONLY (not V!)
        # RoPE's forward expects [..., seq_len, ..., dim]
        # Our shape is [B, L, n_heads, head_dim], seq_dim=1
        q = self.rope(q, seq_dim=1)
        k = self.rope(k, seq_dim=1)
        # V is NOT rotated
        
        # 4. Transpose for attention: [B, n_heads, L, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 5. Scaled dot-product attention
        # scores: [B, n_heads, L, L]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: [B, L] -> [B, 1, 1, L] for broadcasting
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))
        
        # 6. Attention weights and output
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)  # [B, n_heads, L, head_dim]
        
        # 7. Concatenate heads: [B, L, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        
        # 8. Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        return output


class RoPECrossAttention(nn.Module):
    """
    Cross-Attention with RoPE applied only to Q and K
    """
    def __init__(self, d_model: int, n_heads: int, rope_layer: nn.Module, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope_layer
        
        # Separate Q, K, V projections for cross-attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
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
        
        Returns:
            [B, L_q, D]
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
        
        # 4. Transpose for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 5. Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        
        # 6. Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, D)
        
        # 7. Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
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
# Main RDT Model (Refactored with RoPE & MLP I/O)
# ============================================================================

class RDT(nn.Module):
    """
    Recursive Denoising Transformer with:
    - RoPE for positional encoding
    - Configurable MLP for input/output processing
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
        # MLP I/O Configuration
        input_mlp_hidden: List[int] = [512],
        output_mlp_hidden: List[int] = [512],
        # Gate Configuration
        gate_hidden_dim: int = 512,
        gate_num_layers: int = 3,
        gate_num_heads: int = 8,
        gate_dropout: float = 0.3,
        # RoPE Configuration
        rope_base: float = 10000.0,
        # Training
        gradient_checkpointing: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.gradient_checkpointing = gradient_checkpointing
        
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
        
        # Input Processor (MLP-based)
        self.input_processor = MLPProcessor(
            d_model=d_model,
            hidden_dims=input_mlp_hidden,
            dropout=dropout
        )
        
        # Input Normalization
        self.input_norm = nn.LayerNorm(d_model)
        
        # Noise Level Embedding
        self.noise_emb = TimestepEmbedder(d_model, frequency_embedding_size=d_model)
        
        # Recursive Encoder (AdaLN Blocks with RoPE)
        self.encoder_layers = nn.ModuleList([
            DirectionalRecursiveBlock(d_model, n_heads, d_ff, self.rope, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # Output Processor (MLP-based)
        self.output_processor = MLPProcessor(
            d_model=d_model,
            hidden_dims=output_mlp_hidden,
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
        
        # Weight Tying
        self.output_projection.weight = self.token_embedding.weight
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, 
        x, 
        context=None, 
        attention_mask=None, 
        context_mask=None,
        last_gate_score=None, 
        last_pooled=None, 
        is_first_step=True, 
        gt_timestep=None, 
        sampling_prob=0.0
    ):
        """
        Forward pass with RoPE applied consistently throughout recursion.
        
        Args:
            x: [B, L] (tokens) or [B, L, D] (hidden)
            context: Optional context for cross-attention
            attention_mask: [B, L] (1=valid, 0=padding)
            context_mask: [B, L_ctx] (1=valid, 0=padding)
            last_gate_score: Previous gate prediction
            last_pooled: Previous pooled features
            is_first_step: Whether this is the first forward pass
            gt_timestep: Ground truth timestep for scheduled sampling
            sampling_prob: Probability of using GT vs predicted gate
        
        Returns:
            hidden: [B, L, D] - output hidden states
            next_gate_pred: [B, 1] - next gate prediction
            next_pooled: [B, D] - next pooled features
        """
        
        # 1. Initialization & Projection
        if is_first_step:
            # Token embedding
            x = self.token_embedding(x) * math.sqrt(self.d_model)
            
            # Input processing (MLP)
            # Note: RoPE is NOT applied here - it will be applied inside attention layers
            hidden = self.input_processor(x)
        else:
            # Already processed hidden states
            # Note: RoPE is NOT applied here - it will be applied inside attention layers
            hidden = x
        
        # Recursive Norm (Reset Distribution)
        hidden = self.input_norm(hidden)

        # 2. Gate Diagnosis
        predicted_gate, current_pooled = self.gate(
            hidden, 
            attention_mask, 
            prev_pooled=last_pooled, 
            prev_gate=last_gate_score
        )
        
        # 3. Scheduled Sampling
        if self.training and gt_timestep is not None:
            if sampling_prob > 0.0:
                use_gt = torch.rand(1).item() < sampling_prob
                if use_gt:
                    current_noise = gt_timestep
                else:
                    current_noise = predicted_gate.detach()
            else:
                current_noise = predicted_gate.detach()
        else:
            current_noise = predicted_gate.detach()

        # 4. Create Noise Embedding
        noise_vec = self.noise_emb(current_noise)

        # 5. Prepare Masks
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        context_key_padding_mask = (context_mask == 0) if context_mask is not None else None

        # 6. Recursive Encoding with AdaLN
        for layer in self.encoder_layers:
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(h, n_emb, c, sm, cm):
                        return module(h, noise_emb=n_emb, context=c, 
                                    src_key_padding_mask=sm, 
                                    context_key_padding_mask=cm)
                    return custom_forward
                
                hidden = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden, noise_vec, context, src_key_padding_mask, context_key_padding_mask,
                    use_reentrant=False
                )
            else:
                hidden = layer(
                    hidden,
                    noise_emb=noise_vec,
                    context=context,
                    src_key_padding_mask=src_key_padding_mask,
                    context_key_padding_mask=context_key_padding_mask
                )

        # 7. Next Gate Prediction
        next_gate_pred, next_pooled = self.gate(
            hidden, 
            attention_mask, 
            prev_pooled=current_pooled, 
            prev_gate=predicted_gate
        )
        
        return hidden, next_gate_pred, next_pooled
    
    def encode_tokens(self, tokens):
        """
        Initial encoding: tokens → h_0
        
        Args:
            tokens: [B, L] token indices
        
        Returns:
            hidden: [B, L, D] initial hidden states (before any recursive steps)
        """
        x = self.token_embedding(tokens) * math.sqrt(self.d_model)
        hidden = self.input_processor(x)
        hidden = self.input_norm(hidden)
        return hidden
    
    def forward_step(self, hidden, attention_mask=None,
                     last_gate_score=None, last_pooled=None,
                     gt_timestep=None, sampling_prob=0.0):
        """
        Single recursive step: h_i → h_{i+1}
        
        Args:
            hidden: [B, L, D] current hidden states
            attention_mask: [B, L] attention mask (1=valid, 0=padding)
            last_gate_score: [B, 1] previous gate prediction
            last_pooled: [B, D] previous pooled features
            gt_timestep: [B, 1] ground truth timestep for scheduled sampling
            sampling_prob: probability of using GT vs predicted gate
        
        Returns:
            hidden: [B, L, D] next hidden states (after recursive blocks)
            predicted_gate: [B, 1] gate prediction for current hidden
            current_pooled: [B, D] current pooled features
        """
        # 1. Gate prediction for current state
        predicted_gate, current_pooled = self.gate(
            hidden, attention_mask, last_pooled, last_gate_score
        )
        
        # 2. Scheduled sampling
        if self.training and gt_timestep is not None:
            if sampling_prob > 0.0:
                use_gt = torch.rand(1).item() < sampling_prob
                if use_gt:
                    current_noise = gt_timestep
                else:
                    current_noise = predicted_gate.detach()
            else:
                current_noise = predicted_gate.detach()
        else:
            current_noise = predicted_gate.detach()
        
        # 3. Create noise embedding
        noise_vec = self.noise_emb(current_noise)
        
        # 4. Prepare masks
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        
        # 5. Recursive encoding with AdaLN
        for layer in self.encoder_layers:
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(h, n_emb, sm):
                        return module(h, noise_emb=n_emb, context=None,
                                    src_key_padding_mask=sm,
                                    context_key_padding_mask=None)
                    return custom_forward
                
                hidden = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden, noise_vec, src_key_padding_mask,
                    use_reentrant=False
                )
            else:
                hidden = layer(
                    hidden,
                    noise_emb=noise_vec,
                    context=None,
                    src_key_padding_mask=src_key_padding_mask,
                    context_key_padding_mask=None
                )
        
        return hidden, predicted_gate, current_pooled
    
    def decode(self, hidden, attention_mask=None):
        """
        Output Decoder: hidden -> logits
        
        Args:
            hidden: [B, L, D]
            attention_mask: [B, L] (1=valid, 0=padding)
        
        Returns:
            logits: [B, L, vocab_size]
        """
        # Output processing (MLP)
        # Note: RoPE is NOT applied here - positional info is already in hidden states
        decoded = self.output_processor(hidden)
        
        # Project to vocabulary
        logits = self.output_projection(decoded)
        
        return logits

    def inference(self, x, context=None, attention_mask=None, context_mask=None,
                  max_steps=20, threshold=0.02, return_steps=False):
        """
        Inference with per-sample early stopping using masks.
        Each sample in the batch can stop independently when its gate score falls below threshold.
        
        Args:
            x: [B, L] - input tokens
            context: Optional context for cross-attention
            attention_mask: [B, L] - attention mask (1=valid, 0=padding)
            context_mask: [B, L_ctx] - context mask
            max_steps: Maximum recursive steps
            threshold: Stop when gate score < threshold (per sample)
            return_steps: If True, return per-sample step counts
        
        Returns:
            output_tokens: [B, L] - predicted tokens
            steps_taken: [B] - number of steps taken per sample (if return_steps=True)
                        or int - max steps taken (if return_steps=False)
        """
        self.eval()
        batch_size = x.size(0)
        device = x.device
        
        with torch.no_grad():
            # Initialize
            hidden, gate_pred, pooled = self.forward(
                x,
                context=context,
                attention_mask=attention_mask,
                context_mask=context_mask,
                is_first_step=True
            )
            
            # Track which samples are still active
            # active_mask: [B] - True if sample should continue processing
            active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            
            # Track steps taken per sample
            steps_taken = torch.ones(batch_size, dtype=torch.long, device=device)
            
            # Store final hidden states for each sample
            # We'll update this only for active samples at each step
            final_hidden = hidden.clone()
            
            step = 1
            
            # Continue until all samples converge or max_steps reached
            while step < max_steps and active_mask.any():
                # Check which samples should continue
                # gate_pred: [B, 1] -> [B]
                gate_scores = gate_pred.squeeze(-1)
                should_continue = gate_scores > threshold
                
                # Update active mask: only continue if both previously active AND score above threshold
                active_mask = active_mask & should_continue
                
                if not active_mask.any():
                    # All samples have converged
                    break
                
                # Perform forward pass for all samples (for efficiency)
                # But we'll only update hidden states for active samples
                hidden_next, gate_pred_next, pooled_next = self.forward(
                    hidden,
                    context=context,
                    attention_mask=attention_mask,
                    context_mask=context_mask,
                    last_gate_score=gate_pred,
                    last_pooled=pooled,
                    is_first_step=False
                )
                
                # Update hidden states only for active samples
                # This ensures converged samples retain their final state
                active_mask_expanded = active_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
                final_hidden = torch.where(
                    active_mask_expanded,
                    hidden_next,
                    final_hidden
                )
                
                # Update gate_pred and pooled for next iteration
                # (we process all samples for vectorization efficiency)
                hidden = hidden_next
                gate_pred = gate_pred_next
                pooled = pooled_next
                
                # Increment step counter for active samples only
                steps_taken[active_mask] += 1
                step += 1
            
            # Decode final hidden states
            logits = self.decode(final_hidden, attention_mask)
            output_tokens = logits.argmax(dim=-1)
        
        if return_steps:
            return output_tokens, steps_taken
        else:
            # Return mean steps for backward compatibility
            return output_tokens, steps_taken.mean().item()