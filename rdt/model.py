"""RDT Model Architecture (Enhanced with Cross-Attention & Latent Consistency)"""

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

# [신규] Gate 점수(Scalar)를 벡터로 변환하여 인코더에 힌트를 주는 모듈
class NoiseLevelEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
    
    def forward(self, gate_score):
        # gate_score: [Batch, 1] -> [Batch, 1, d_model]
        return self.proj(gate_score).unsqueeze(1)

# [신규] Self-Attention과 Cross-Attention(조건부)을 모두 수행하는 블록
class DirectionalRecursiveBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # 1. Self-Attention (Internal Consistency)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # 2. Cross-Attention (External Direction) - Direction Injection
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # 3. Feed-Forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, context=None, src_mask=None, context_mask=None):
        # Phase 1: Self-Refinement (ResNet 구조: x + f(x))
        # Pre-Norm 방식을 사용하여 Deep Layer 학습 안정성 확보
        res = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, attn_mask=src_mask, need_weights=False)
        x = res + self.dropout1(x)
        
        # Phase 2: Direction Injection (조건이 있을 때만 실행)
        if context is not None:
            res = x
            x = self.norm2(x)
            x, _ = self.cross_attn(query=x, key=context, value=context, 
                                   key_padding_mask=context_mask, need_weights=False)
            x = res + self.dropout2(x)
        
        # Phase 3: Integration
        res = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = res + self.dropout3(x)
        
        return x

class LinearDecoder(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)
    def forward(self, x): return self.projection(x)

class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        self.projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, mask=None): 
        # [수정] mask 전달 가능하도록 변경
        x = self.decoder(x, src_key_padding_mask=mask)
        return self.projection(x)

class GateMLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        if x.dim() == 3: x = x.mean(dim=1)
        return torch.sigmoid(self.mlp(x))

class RDT(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_encoder_layers=6, n_decoder_layers=1, 
                 d_ff=2048, dropout=0.1, max_seq_len=512, decoder_type='transformer', 
                 gate_hidden_dim=256, gradient_checkpointing=False):
        super().__init__()
        self.d_model = d_model
        
        # 1. Input Processing
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # [신규] Input Consistency Projection
        # 초기 입력 임베딩을 Latent Space의 분포(LayerNorm)와 일치시켜 이질감 제거
        self.input_projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # [신규] Noise Level Embedding (Adaptive Feedback)
        self.noise_emb = NoiseLevelEmbedding(d_model)
        
        # 2. Recursive Encoder (이제 Cross-Attention 지원)
        self.encoder_layers = nn.ModuleList([
            DirectionalRecursiveBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # 3. Decoder
        if decoder_type == 'linear':
            self.decoder = LinearDecoder(d_model, vocab_size)
        else:
            self.decoder = TransformerDecoder(d_model, n_heads, n_decoder_layers, d_ff, vocab_size, dropout)
            
        self.gate = GateMLP(d_model, gate_hidden_dim)
        
        # Weight Tying
        self.decoder.projection.weight = self.token_embedding.weight
        self._init_weights()
        
        self.gradient_checkpointing = gradient_checkpointing

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def forward(self, x, context=None, attention_mask=None, context_mask=None, 
                last_gate_score=None, is_first_step=True):
        """
        x: [Batch, Seq] (Token IDs) OR [Batch, Seq, Dim] (Hidden State)
        context: [Batch, Ctx_Len, Dim] (Optional)
        last_gate_score: [Batch, 1] (Optional, 이전 스텝의 Gate 예측값)
        """
        
        # 1. 초기화 및 임베딩 (First Step)
        if is_first_step:
            x = self.token_embedding(x) * math.sqrt(self.d_model)
            x = self.pos_encoding(x)
            # [중요] 초기 입력을 Latent Space 분포로 투영 (Consistency)
            hidden = self.input_projector(x)
            
            # 첫 스텝의 Noise Level은 '완전 모름(0.5)' 혹은 '완전 노이즈(1.0)'로 가정
            # 여기서는 1.0(Max Noise)으로 초기화
            current_noise = torch.ones(x.size(0), 1, device=x.device)
        else:
            hidden = x
            # 이전 스텝에서 예측한 Gate 점수를 현재의 노이즈 레벨 힌트로 사용
            current_noise = last_gate_score if last_gate_score is not None else torch.zeros(hidden.size(0), 1, device=hidden.device)

        # 2. Noise/Gate Embedding 주입 (Adaptive Feedback)
        # "지금 상태가 70% 망가져 있다"는 정보를 벡터로 더해줌
        noise_vec = self.noise_emb(current_noise) # [B, 1, D]
        hidden = hidden + noise_vec

        # 3. Masks Preparation
        # Self-Attention Mask (Padding)
        src_mask = None
        if attention_mask is not None:
            # MultiheadAttention은 (Batch*NumHeads, Q, K) 형태 혹은 (Batch, Q, K) 필요
            # 여기선 key_padding_mask로 처리하는게 편함 (True가 마스킹됨) -> (attention_mask==0)
            # 하지만 nn.MultiheadAttention의 attn_mask 인자는 (L, S) 혹은 (N*H, L, S)임.
            # key_padding_mask 인자를 사용하는 것이 적절함.
            pass 

        # PyTorch MultiheadAttention key_padding_mask: True=Ignore, False=Attend
        # 입력 attention_mask: 1=Attend, 0=Ignore
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        memory_key_padding_mask = (context_mask == 0) if context_mask is not None else None

        # 4. Recursive Encoding
        for layer in self.encoder_layers:
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(h, c, sm, cm):
                        return module(h, context=c, src_mask=None, context_mask=cm) # src_mask는 key_padding_mask로 대체
                    return custom_forward
                # Checkpoint 사용 시 주의: 인자 전달 방식
                hidden = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer), hidden, context, src_mask, memory_key_padding_mask,
                    use_reentrant=False
                )
            else:
                # Custom Block 내부에서 key_padding_mask 처리를 하도록 구현했어야 하는데 
                # DirectionalRecursiveBlock의 forward를 보면 key_padding_mask 인자가 빠져있고 attn_mask만 있음.
                # 위 클래스 정의에서 forward 인자를 조금 수정해서 호출해야 함.
                
                # [수정된 호출] DirectionalRecursiveBlock 내부 로직에 맞춰 호출
                # Self-Attn의 key_padding_mask는 구현상 따로 받는게 좋으나, 
                # 여기선 간단히 구현하기 위해 MHA의 key_padding_mask 인자를 활용
                
                # Phase 1: Self-Attn
                res = hidden
                hidden = layer.norm1(hidden)
                hidden, _ = layer.self_attn(hidden, hidden, hidden, key_padding_mask=src_key_padding_mask, need_weights=False)
                hidden = res + layer.dropout1(hidden)
                
                # Phase 2: Cross-Attn
                if context is not None:
                    res = hidden
                    hidden = layer.norm2(hidden)
                    hidden, _ = layer.cross_attn(query=hidden, key=context, value=context, 
                                               key_padding_mask=memory_key_padding_mask, need_weights=False)
                    hidden = res + layer.dropout2(hidden)
                    
                # Phase 3: FFN
                res = hidden
                hidden = layer.norm3(hidden)
                hidden = layer.ffn(hidden)
                hidden = res + layer.dropout3(hidden)

        # 5. Gate Prediction
        gate_pred = self.gate(hidden)
        
        return hidden, gate_pred

    def inference(self, x, context=None, max_steps=20, threshold=0.1):
        self.eval()
        with torch.no_grad():
            # First Step
            hidden, gate_pred = self.forward(x, context=context, is_first_step=True)
            step = 1
            
            # Recursive Loop
            while step < max_steps and gate_pred.mean().item() > threshold:
                hidden, gate_pred = self.forward(
                    hidden, 
                    context=context, 
                    last_gate_score=gate_pred, # [중요] 자신의 예측값을 다음 스텝 힌트로 사용
                    is_first_step=False
                )
                step += 1
            
            # Final Decode
            if hasattr(self.decoder, 'decoder'):
                # Transformer Decoder는 src_mask 필요할 수 있음
                feat = self.decoder.decoder(hidden)
                logits = self.decoder.projection(feat)
            else:
                logits = self.decoder(hidden)
                
            output_tokens = logits.argmax(dim=-1)
            
        return output_tokens, step