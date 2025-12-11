"""RDT Model Architecture (Enhanced with AdaLN & Cross-Attention) - Diffusion Style"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """BERT-style learned positional embeddings"""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        
    def forward(self, x):
        seq_len = x.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
        position_embeddings = self.position_embeddings(position_ids)
        return self.dropout(x + position_embeddings)

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
        # 1. Frequency 계산: exp(-log(10000) / (d-1))
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        
        # 2. Argument 계산: t * freq
        # t가 [Batch, 1]로 들어올 수도 있으므로 차원 맞춤
        if t.dim() == 2:
            t = t.squeeze(-1) # [Batch]
            
        args = t[:, None].float() * freqs[None]
        
        # 3. Sin/Cos 결합
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # 4. 차원이 홀수일 경우 0 패딩 (안전장치)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding

    def forward(self, t):
        # t: [Batch, 1] (0.0 ~ 1.0)
        t_freq = self.timestep_embedding(t * 50, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb.unsqueeze(1) # [Batch, 1, Hidden]

class AdaptiveLayerNorm(nn.Module):
    """
    Diffusion Style Adaptive LayerNorm (AdaLN)
    Noise Embedding을 받아 LayerNorm의 Scale(gamma)과 Shift(beta)를 동적으로 제어
    """
    def __init__(self, d_model: int):
        super().__init__()
        # elementwise_affine=False: 정적 파라미터 끔 (동적으로 생성할 것이므로)
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        
        # 임베딩을 Scale & Shift로 변환
        self.proj = nn.Linear(d_model, d_model * 2)
        
        # [Zero-Init Trick]
        # 학습 초기에는 조건이 영향을 주지 않도록(Identity) 가중치를 0으로 초기화
        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x, emb):
        # emb: [Batch, 1, D] -> [Batch, D]
        if emb.dim() == 3:
            emb = emb.squeeze(1)
            
        # Scale & Shift 예측
        gamma, beta = self.proj(emb).chunk(2, dim=1)
        
        # 차원 맞추기 (Broadcasting: [B, 1, D])
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        
        # Modulation: Norm(x) * (1 + gamma) + beta
        return self.norm(x) * (1 + gamma) + beta


class DirectionalRecursiveBlock(nn.Module):
    """Self-Attention + Optional Cross-Attention Block with AdaLN"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # 1. Self-Attention (AdaLN 적용)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = AdaptiveLayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # 2. Cross-Attention (Conditional, AdaLN 적용)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = AdaptiveLayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # 3. Feed-Forward (AdaLN 적용)
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
        noise_emb: [B, 1, D] - 필수 입력 (Gate Embedding)
        """
        # Phase 1: Self-Attention
        res = x
        x_norm = self.norm1(x, noise_emb) # AdaLN
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm, 
            key_padding_mask=src_key_padding_mask, 
            need_weights=False
        )
        x = res + self.dropout1(attn_out)
        
        # Phase 2: Cross-Attention
        if context is not None:
            res = x
            x_norm = self.norm2(x, noise_emb) # AdaLN
            attn_out, _ = self.cross_attn(
                query=x_norm, 
                key=context, 
                value=context, 
                key_padding_mask=context_key_padding_mask, 
                need_weights=False
            )
            x = res + self.dropout2(attn_out)
        
        # Phase 3: Feed-Forward
        res = x
        x_norm = self.norm3(x, noise_emb) # AdaLN
        ffn_out = self.ffn(x_norm)
        x = res + self.dropout3(ffn_out)
        
        return x

class GateMLP(nn.Module):
    """
    Enhanced Gate with configurable depth
    - Multi-head attention pooling
    - N-layer MLP with residual connections
    - Learnable temperature scaling
    """
    def __init__(self, d_model: int, hidden_dim: int = 512, num_layers: int = 3, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Multi-head attention pooling
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.normal_(self.query, std=0.02)
        
        # N-layer MLP with residual connections
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'linear': nn.Linear(d_model if i == 0 else hidden_dim, hidden_dim),
                'norm': nn.LayerNorm(hidden_dim),
                'dropout': nn.Dropout(dropout)
            }))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for layer_dict in self.layers:
            nn.init.xavier_uniform_(layer_dict['linear'].weight)
            nn.init.zeros_(layer_dict['linear'].bias)
        
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, x, mask=None):
        """
        x: [B, L, D] - hidden states
        mask: [B, L] - attention mask (1=valid, 0=padding)
        Returns: [B, 1] - gate prediction
        """
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
        pooled = pooled.squeeze(1)  # [B, D]
        
        # N-layer processing with residual connections
        h = pooled
        for i, layer_dict in enumerate(self.layers):
            h_new = layer_dict['linear'](h)
            h_new = layer_dict['norm'](h_new)
            h_new = nn.functional.gelu(h_new)
            h_new = layer_dict['dropout'](h_new)
            
            # Residual connection (skip first layer since dimension changes)
            if i > 0:
                h = h + h_new
            else:
                h = h_new
        
        # Output projection
        raw_output = self.output_proj(h)
        
        # Temperature-scaled softplus
        gate_output = nn.functional.softplus(raw_output)
        
        return gate_output

class RDT(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_encoder_layers=6, n_io_layers=1, 
                 d_ff=2048, dropout=0.1, max_seq_len=512,
                 gate_hidden_dim=512, gate_num_layers=3, gate_num_heads=8, gradient_checkpointing=False):
        super().__init__()
        self.d_model = d_model
        self.gradient_checkpointing = gradient_checkpointing
        
        # 1. Input Processing
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Input Encoder (1-2 layer Transformer)
        input_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, 
            dropout=dropout, batch_first=True
        )
        self.input_encoder = nn.TransformerEncoder(input_encoder_layer, num_layers=n_io_layers)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Noise Level Embedding
        self.noise_emb = TimestepEmbedder(d_model, frequency_embedding_size=d_model)
        
        # 2. Recursive Encoder (Using AdaLN Blocks)
        self.encoder_layers = nn.ModuleList([
            DirectionalRecursiveBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # 3. Output Decoder (1-2 layer Transformer)
        output_decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.output_decoder = nn.TransformerEncoder(output_decoder_layer, num_layers=n_io_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)
            
        self.gate = GateMLP(
            d_model=d_model, 
            hidden_dim=gate_hidden_dim, 
            num_layers=gate_num_layers,
            num_heads=gate_num_heads,
            dropout=dropout
        )
        
        # Weight Tying
        self.output_projection.weight = self.token_embedding.weight
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1: 
                nn.init.xavier_uniform_(p)

    def forward(self, x, context=None, attention_mask=None, context_mask=None, 
                last_gate_score=None, is_first_step=True, gt_timestep=None, sampling_prob=0.0):
        """
        x: [B, L] (tokens) or [B, L, D] (hidden)
        gt_timestep: [B, 1] Ground truth timestep (0~1), training 시 제공
        sampling_prob: Scheduled sampling probability (0~1)
                       - 0.0: 항상 predicted gate 사용 (inference)
                       - 1.0: 항상 gt_timestep 사용 (early training)
                       - 0~1: 확률적으로 혼합 (curriculum learning)
        """
        
        # 1. Initialization & Projection
        if is_first_step:
            x = self.token_embedding(x) * math.sqrt(self.d_model)
            x = self.pos_encoding(x)
            # Input Encoder
            src_key_padding_mask_temp = (attention_mask == 0) if attention_mask is not None else None
            hidden = self.input_encoder(x, src_key_padding_mask=src_key_padding_mask_temp)
        else:
            hidden = x
        
        # Recursive Norm (Reset Distribution)
        hidden = self.input_norm(hidden)

        # 2. Gate Diagnosis
        predicted_gate = last_gate_score if last_gate_score is not None else self.gate(hidden, attention_mask)
        
        # 3. Scheduled Sampling (Training Stabilization)
        if self.training and gt_timestep is not None:
            # Sampling probability로 GT vs Predicted 선택
            if sampling_prob > 0.0:
                # Bernoulli sampling: sampling_prob 확률로 GT 사용
                use_gt = torch.rand(1).item() < sampling_prob
                if use_gt:
                    current_noise = gt_timestep  # Ground Truth 사용
                else:
                    current_noise = predicted_gate.detach()  # Predicted 사용 (detached)
            else:
                current_noise = predicted_gate.detach()  # sampling_prob=0이면 항상 predicted
        else:
            # Inference 또는 GT 없을 때: 항상 predicted gate 사용
            current_noise = predicted_gate.detach()

        # 4. Create Noise Embedding (Condition)
        # Additive Injection 제거 -> AdaLN을 위한 임베딩 생성만 함
        noise_vec = self.noise_emb(current_noise)  # [B, 1, D]

        # 5. Prepare Masks
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        context_key_padding_mask = (context_mask == 0) if context_mask is not None else None

        # 6. Recursive Encoding with AdaLN
        for layer in self.encoder_layers:
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(h, n_emb, c, sm, cm):
                        # noise_emb를 블록 안으로 전달
                        return module(h, noise_emb=n_emb, context=c, src_key_padding_mask=sm, context_key_padding_mask=cm)
                    return custom_forward
                
                hidden = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer), 
                    hidden, noise_vec, context, src_key_padding_mask, context_key_padding_mask,
                    use_reentrant=False
                )
            else:
                hidden = layer(
                    hidden, 
                    noise_emb=noise_vec, # [중요] AdaLN을 위해 전달
                    context=context, 
                    src_key_padding_mask=src_key_padding_mask, 
                    context_key_padding_mask=context_key_padding_mask
                )

        # 7. Next Gate Prediction
        next_gate_pred = self.gate(hidden, attention_mask)
        
        return hidden, next_gate_pred
    
    def decode(self, hidden, attention_mask=None):
        """
        Output Decoder: hidden -> logits
        hidden: [B, L, D]
        attention_mask: [B, L] (1=valid, 0=padding)
        Returns: logits [B, L, vocab_size]
        """
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        decoded = self.output_decoder(hidden, src_key_padding_mask=src_key_padding_mask)
        logits = self.output_projection(decoded)
        return logits

    def inference(self, x, context=None, max_steps=20, threshold=0.02):
        self.eval()
        with torch.no_grad():
            hidden, gate_pred = self.forward(x, context=context, is_first_step=True)
            step = 1
            while step < max_steps and gate_pred.mean().item() > threshold:
                hidden, gate_pred = self.forward(
                    hidden, 
                    context=context, 
                    last_gate_score=gate_pred,
                    is_first_step=False
                )
                step += 1
            
            # Use decode method
            logits = self.decode(hidden)
            output_tokens = logits.argmax(dim=-1)
            
        return output_tokens, step