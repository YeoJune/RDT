"""RDT Model Architecture (Enhanced with AdaLN & Cross-Attention) - Diffusion Style"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        pos_emb = self.pe[:seq_len, :].unsqueeze(0)
        return self.dropout(x + pos_emb)

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
    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        # Attention-based pooling
        self.attention = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Gate prediction
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, mask=None):
        """
        x: [B, L, D]
        mask: [B, L] (1=valid, 0=padding)
        """
        if x.dim() == 2:  # Already pooled
            return torch.sigmoid(self.mlp(x))
        
        # Attention scores
        attn_scores = self.attention(x).squeeze(-1)  # [B, L]
        
        # Mask out padding
        if mask is not None:
            # -1e4 is safer than -1e9 for fp16
            attn_scores = attn_scores.masked_fill(mask == 0, -1e4)
        
        # Softmax
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # [B, L, 1]
        
        # Weighted sum
        pooled = (x * attn_weights).sum(dim=1)  # [B, D]
        
        raw_output = self.mlp(pooled)
        gate_output = nn.functional.softplus(raw_output)
        
        return gate_output

class RDT(nn.Module):
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
        self.gradient_checkpointing = gradient_checkpointing
        
        # 1. Input Processing
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Input Encoder (1-2 layer Transformer)
        input_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, 
            dropout=dropout, batch_first=True
        )
        self.input_encoder = nn.TransformerEncoder(input_encoder_layer, num_layers=input_processor_layers)
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
        self.output_decoder = nn.TransformerEncoder(output_decoder_layer, num_layers=output_processor_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)
            
        self.gate = GateMLP(d_model, gate_hidden_dim)
        
        # Weight Tying
        self.output_projection.weight = self.token_embedding.weight
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1: 
                nn.init.xavier_uniform_(p)
    
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
    
    
    def encode(self, tokens, attention_mask=None):
        """
        Initial encoding: tokens → h_0
        
        Args:
            tokens: [B, L] token indices
            attention_mask: [B, L] attention mask (1=valid, 0=padding)
        
        Returns:
            h_0: [B, L, D] initial hidden states
        """
        x = tokens
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        # Input Encoder
        src_key_padding_mask_temp = (attention_mask == 0) if attention_mask is not None else None
        h_0 = self.input_encoder(x, src_key_padding_mask=src_key_padding_mask_temp)
        
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
        
        # Recursive Norm (Reset Distribution)
        hidden = self.input_norm(hidden)

        # 4. Create Noise Embedding (Condition)
        # Additive Injection 제거 -> AdaLN을 위한 임베딩 생성만 함
        noise_vec = self.noise_emb(noise_level)  # [B, 1, D]

        # 5. Prepare Masks
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

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
                    hidden, noise_vec, None, src_key_padding_mask, None,
                    use_reentrant=False
                )
            else:
                hidden = layer(
                    hidden, 
                    noise_emb=noise_vec, # [중요] AdaLN을 위해 전달
                    context=None, 
                    src_key_padding_mask=src_key_padding_mask, 
                    context_key_padding_mask=None
                )
        
        return hidden


    def decode(self, hidden, attention_mask=None):
        """
        Output decoder: h_n → logits
        
        Args:
            hidden: [B, L, D] final hidden states h_n
            attention_mask: [B, L] attention mask (1=valid, 0=padding)
        
        Returns:
            logits: [B, L, vocab_size]
        """
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        decoded = self.output_decoder(hidden, src_key_padding_mask=src_key_padding_mask)
        logits = self.output_projection(decoded)
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
            
            h_i = x  # x is already hidden states
            
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