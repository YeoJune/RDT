"""RDT Model Architecture (Enhanced with Cross-Attention & Latent Consistency) - Fixed"""

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


class NoiseLevelEmbedding(nn.Module):
    """Gate score를 벡터로 변환하여 인코더에 노이즈 레벨 힌트 제공"""
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


class DirectionalRecursiveBlock(nn.Module):
    """Self-Attention + Optional Cross-Attention Block"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # 1. Self-Attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # 2. Cross-Attention (Optional)
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

    def forward(self, x, context=None, src_key_padding_mask=None, context_key_padding_mask=None):
        """
        x: [B, L, D]
        context: [B, C, D] (optional)
        src_key_padding_mask: [B, L] (True = ignore)
        context_key_padding_mask: [B, C] (True = ignore)
        """
        # Phase 1: Self-Attention
        res = x
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm, 
            key_padding_mask=src_key_padding_mask, 
            need_weights=False
        )
        x = res + self.dropout1(attn_out)
        
        # Phase 2: Cross-Attention (Conditional)
        if context is not None:
            res = x
            x_norm = self.norm2(x)
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
        x_norm = self.norm3(x)
        ffn_out = self.ffn(x_norm)
        x = res + self.dropout3(ffn_out)
        
        return x


class LinearDecoder(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, x): 
        return self.projection(x)


class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        self.projection = nn.Linear(d_model, vocab_size)
        self.gradient_checkpointing = False
    
    def forward(self, x, mask=None): 
        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            def create_custom_forward(module):
                def custom_forward(x_input, mask_input):
                    return module(x_input, src_key_padding_mask=mask_input)
                return custom_forward
            x = checkpoint(create_custom_forward(self.decoder), x, mask, use_reentrant=False)
        else:
            x = self.decoder(x, src_key_padding_mask=mask)
        return self.projection(x)

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
            attn_scores = attn_scores.masked_fill(mask == 0, -1e4)
        
        # Softmax
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # [B, L, 1]
        
        # Weighted sum
        pooled = (x * attn_weights).sum(dim=1)  # [B, D]
        
        return torch.sigmoid(self.mlp(pooled))
    
class RDT(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_encoder_layers=6, n_decoder_layers=1, 
                 d_ff=2048, dropout=0.1, max_seq_len=512, decoder_type='transformer', 
                 gate_hidden_dim=256, gradient_checkpointing=False):
        super().__init__()
        self.d_model = d_model
        self.gradient_checkpointing = gradient_checkpointing
        
        # 1. Input Processing
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Input Consistency Projection
        self.input_projector = nn.Linear(d_model, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Noise Level Embedding
        self.noise_emb = NoiseLevelEmbedding(d_model)
        
        # 2. Recursive Encoder
        self.encoder_layers = nn.ModuleList([
            DirectionalRecursiveBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # 3. Decoder
        if decoder_type == 'linear':
            self.decoder = LinearDecoder(d_model, vocab_size)
        else:
            self.decoder = TransformerDecoder(d_model, n_heads, n_decoder_layers, d_ff, vocab_size, dropout)
            if gradient_checkpointing:
                self.decoder.gradient_checkpointing = True
            
        self.gate = GateMLP(d_model, gate_hidden_dim)
        
        # Weight Tying
        self.decoder.projection.weight = self.token_embedding.weight
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1: 
                nn.init.xavier_uniform_(p)

    def forward(self, x, context=None, attention_mask=None, context_mask=None, 
                last_gate_score=None, is_first_step=True):
        """
        x: [B, L] (tokens) or [B, L, D] (hidden)
        context: [B, C, D] (optional)
        attention_mask: [B, L] (1=attend, 0=ignore)
        context_mask: [B, C] (1=attend, 0=ignore)
        last_gate_score: [B, 1] (previous gate prediction)
        """
        
        # 1. Initialization
        if is_first_step:
            x = self.token_embedding(x) * math.sqrt(self.d_model)
            x = self.pos_encoding(x)
            hidden = self.input_projector(x)
        else:
            hidden = x
        
        hidden = self.input_norm(hidden) 

        # Use previous gate score as current noise level
        current_noise = last_gate_score if last_gate_score is not None else self.gate(hidden, attention_mask)

        # 2. Inject Noise Level Embedding
        noise_vec = self.noise_emb(current_noise)  # [B, 1, D]
        hidden = hidden + noise_vec

        # 3. Prepare Masks
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        context_key_padding_mask = (context_mask == 0) if context_mask is not None else None

        # 4. Recursive Encoding
        for layer in self.encoder_layers:
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(h, c, sm, cm):
                        return module(h, context=c, src_key_padding_mask=sm, context_key_padding_mask=cm)
                    return custom_forward
                hidden = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer), 
                    hidden, context, src_key_padding_mask, context_key_padding_mask,
                    use_reentrant=False
                )
            else:
                hidden = layer(
                    hidden, 
                    context=context, 
                    src_key_padding_mask=src_key_padding_mask, 
                    context_key_padding_mask=context_key_padding_mask
                )

        # 5. Gate Prediction
        gate_pred = self.gate(hidden, attention_mask)
        
        return hidden, gate_pred

    def inference(self, x, context=None, max_steps=20, threshold=0.01):
        """Inference with adaptive recursion"""
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
                    last_gate_score=gate_pred,
                    is_first_step=False
                )
                step += 1
            
            # Final Decode
            if hasattr(self.decoder, 'decoder'):
                feat = self.decoder.decoder(hidden)
                logits = self.decoder.projection(feat)
            else:
                logits = self.decoder(hidden)
                
            output_tokens = logits.argmax(dim=-1)
            
        return output_tokens, step