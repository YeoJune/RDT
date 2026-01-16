"""RDT Model Architecture (Refactored with APE & Transformer I/O)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional


# ============================================================================
# Absolute Position Embedding (APE)
# ============================================================================

class AbsolutePositionalEmbedding(nn.Module):
    """
    Absolute Positional Embedding using sinusoidal functions.
    Reference: "Attention is All You Need" (Vaswani et al., 2017)
    
    This implementation uses fixed sinusoidal embeddings that are added
    to token embeddings once at the input layer.
    """
    def __init__(self, d_model: int, max_seq_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Create positional encoding matrix [max_seq_len, d_model]
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Compute division term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_seq_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape [B, L, d_model]
        
        Returns:
            Tensor with same shape as x, with positional encodings added
        """
        seq_len = x.size(1)
        
        # Add positional encoding
        # pe[:, :seq_len, :] has shape [1, L, d_model]
        return x + self.pe[:, :seq_len, :]


# ============================================================================
# Standard Self-Attention (without RoPE)
# ============================================================================

class StandardSelfAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention
    Optimized with F.scaled_dot_product_attention
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
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
        
        # 3. Transpose for attention: [B, n_heads, L, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 4. Scaled Dot Product Attention
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: [B, L] -> [B, 1, 1, L] for broadcasting
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        
        # 5. Concatenate heads: [B, L, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        
        # 6. Output projection
        output = self.out_proj(attn_output)
        output = self.out_dropout(output)
        
        return output


class StandardCrossAttention(nn.Module):
    """
    Standard Multi-Head Cross-Attention
    Optimized with F.scaled_dot_product_attention
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
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
        
        # 3. Transpose for attention: [B, n_heads, L, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 4. Scaled Dot Product Attention
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
        
        # 5. Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_q, D)
        
        # 6. Output projection
        output = self.out_proj(attn_output)
        output = self.out_dropout(output)
        
        return output


# ============================================================================
# Transformer Encoder Processor
# ============================================================================

class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer with standard attention
    
    Architecture: Self-Attention -> Add & Norm -> FFN -> Add & Norm
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Self-Attention
        self.self_attn = StandardSelfAttention(d_model, n_heads, dropout)
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
    
    Uses standard transformer encoder layers with APE positional encoding.
    
    Args:
        d_model: Model dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
    """
    def __init__(
        self, 
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
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


class DirectionalRecursiveBlock(nn.Module):
    """Self-Attention + Optional Cross-Attention Block with AdaLN"""
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Self-Attention with AdaLN
        self.self_attn = StandardSelfAttention(d_model, n_heads, dropout)
        self.norm1 = AdaptiveLayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-Attention with AdaLN
        self.cross_attn = StandardCrossAttention(d_model, n_heads, dropout)
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
        attn_out = self.self_attn(x_norm, key_padding_mask=src_key_padding_mask)
        x = res + self.dropout1(attn_out)
        
        # Cross-Attention with AdaLN (if context provided)
        if context is not None:
            res = x
            x_norm = self.norm2(x, noise_emb)
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
# Main RDT Model (Refactored with APE & Transformer I/O)
# ============================================================================

class RDT(nn.Module):
    """
    Recursive Denoising Transformer with:
    - APE (Absolute Positional Encoding) added once at input
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
        # Training
        total_steps: int = 20,
        gradient_checkpointing: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.total_steps = total_steps
        self.gradient_checkpointing = gradient_checkpointing
        
        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Absolute Positional Embedding (APE)
        self.pos_embedding = AbsolutePositionalEmbedding(
            d_model=d_model,
            max_seq_len=max_seq_len
        )
        
        # Input Processor (Transformer Encoder)
        self.input_processor = TransformerProcessor(
            d_model=d_model,
            n_layers=input_processor_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Input Normalization
        self.input_norm = nn.LayerNorm(d_model)
        
        # Noise Level Embedding
        self.noise_emb = TimestepEmbedder(d_model, frequency_embedding_size=d_model, scale_factor=total_steps)
        
        # Recursive Encoder (AdaLN Blocks)
        self.encoder_layers = nn.ModuleList([
            DirectionalRecursiveBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # Output Processor (Transformer Encoder)
        self.output_processor = TransformerProcessor(
            d_model=d_model,
            n_layers=output_processor_layers,
            n_heads=n_heads,
            d_ff=d_ff,
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
        Forward pass with APE added once at input.
        
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
            
            # Add positional encoding (APE - only once at input)
            x = self.pos_embedding(x)
            
            # Input processing (Transformer Encoder)
            key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
            hidden = self.input_processor(x, key_padding_mask=key_padding_mask)
        else:
            # Already processed hidden states
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
        
        # 3. Scheduled Sampling (TPU-Optimized: No .item() call, No Python control flow)
        current_noise = predicted_gate.detach()  # Default value
        
        if self.training and gt_timestep is not None:
            rand_tensor = torch.rand(1, device=predicted_gate.device)
            use_gt_mask = rand_tensor < sampling_prob
            current_noise = torch.where(use_gt_mask, gt_timestep, current_noise)

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
    
    def encode_tokens(self, tokens, attention_mask=None):
        """
        Initial encoding: tokens → h_0
        
        Args:
            tokens: [B, L] token indices
            attention_mask: [B, L] attention mask (1=valid, 0=padding)
        
        Returns:
            hidden: [B, L, D] initial hidden states (before any recursive steps)
        """
        x = self.token_embedding(tokens) * math.sqrt(self.d_model)
        
        # Add positional encoding (APE - only once)
        x = self.pos_embedding(x)
        
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        hidden = self.input_processor(x, key_padding_mask=key_padding_mask)
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
            sampling_prob: scalar tensor (0~1) for soft mixing
        
        Returns:
            hidden: [B, L, D] next hidden states (after recursive blocks)
            predicted_gate: None (not used, computed separately after transformation)
            current_pooled: None (not used, computed separately after transformation)
        """
        # Soft Mixing for Scheduled Sampling
        if self.training and gt_timestep is not None:
            current_noise = sampling_prob * gt_timestep.detach() + (1.0 - sampling_prob) * last_gate_score.detach()
        else:
            current_noise = last_gate_score.detach() if last_gate_score is not None else gt_timestep.detach()
        
        # Create noise embedding
        noise_vec = self.noise_emb(current_noise)
        
        # Prepare masks
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        
        # Recursive encoding with AdaLN
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
        
        return hidden, None, None
    
    def decode(self, hidden, attention_mask=None):
        """
        Output Decoder: hidden -> logits
        
        Args:
            hidden: [B, L, D]
            attention_mask: [B, L] (1=valid, 0=padding)
        
        Returns:
            logits: [B, L, vocab_size]
        """
        # Output processing (Transformer Encoder)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        decoded = self.output_processor(hidden, key_padding_mask=key_padding_mask)
        
        # Project to vocabulary
        logits = self.output_projection(decoded)
        
        return logits

    def inference(self, x, context=None, attention_mask=None, context_mask=None,
                max_steps=20, threshold=0.02, return_steps=False):
        """
        Inference with per-sample processing - ZERO ambiguity version.
        Each sample is processed independently in a loop, then batched for output.
        
        This ensures EXACT correspondence with training logic:
        - Step 0: encode_tokens + initial gate
        - Step 1~N: forward_step + gate + convergence check
        - Final: decode
        
        Args:
            x: [B, L] - input tokens
            context: Optional context for cross-attention (not used currently)
            attention_mask: [B, L] - attention mask (1=valid, 0=padding)
            context_mask: [B, L_ctx] - context mask (not used currently)
            max_steps: Maximum recursive steps
            threshold: Stop when gate score < threshold (per sample)
            return_steps: If True, return per-sample step counts
        
        Returns:
            output_tokens: [B, L] - predicted tokens
            steps_taken: [B] or float - number of steps taken
        """
        self.eval()
        batch_size = x.size(0)
        device = x.device
        
        # Storage for results
        all_output_tokens = []
        all_steps_taken = []
        
        with torch.no_grad():
            # Process each sample independently
            for sample_idx in range(batch_size):
                # Extract single sample
                sample_tokens = x[sample_idx:sample_idx+1]  # [1, L]
                sample_mask = attention_mask[sample_idx:sample_idx+1] if attention_mask is not None else None  # [1, L]
                
                # STEP 0: Initialization (exactly as in training)
                hidden = self.encode_tokens(sample_tokens, sample_mask)  # [1, L, D]
                
                gate_pred, pooled = self.gate(
                    hidden,
                    sample_mask,
                    None,
                    None
                )  # gate_pred: [1, 1], pooled: [1, D]
                
                # Track convergence for this sample
                converged = False
                steps_taken = 1  # We count step 0 as 1 step
                
                # STEP 1 to max_steps: Recursive steps
                for step in range(1, max_steps):
                    gate_score = gate_pred.squeeze(-1).item()
                    
                    if gate_score < threshold:
                        converged = True
                        break
                    
                    hidden, _, _ = self.forward_step(
                        hidden,
                        attention_mask=sample_mask,
                        last_gate_score=gate_pred,
                        last_pooled=pooled,
                        gt_timestep=None,
                        sampling_prob=0.0
                    )
                    
                    gate_pred, pooled = self.gate(
                        hidden,
                        sample_mask,
                        pooled,
                        gate_pred
                    )
                    
                    steps_taken += 1
                
                # FINAL: Decode
                logits = self.decode(hidden, sample_mask)  # [1, L, vocab_size]
                sample_output = logits.argmax(dim=-1)  # [1, L]
                
                # Store results
                all_output_tokens.append(sample_output)
                all_steps_taken.append(steps_taken)
        
        # Batch results
        output_tokens = torch.cat(all_output_tokens, dim=0)  # [B, L]
        steps_taken_tensor = torch.tensor(all_steps_taken, device=device)  # [B]
        
        if return_steps:
            return output_tokens, steps_taken_tensor
        else:
            return output_tokens, steps_taken_tensor.float().mean().item()
        
    def tie_weights(self):
        """Tie output projection weights with token embedding"""
        self.output_projection.weight = self.token_embedding.weight
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)