"""RDT Model Architecture"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with explicit position IDs support"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding table
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)  # (max_len, d_model)
    
    def forward(self, x=None, pos_ids=None):
        """
        Args:
            x: (batch_size, seq_len, d_model) - optional, for getting shape
            pos_ids: (batch_size, seq_len) - explicit position indices
        Returns:
            (batch_size, seq_len, d_model) or (seq_len, d_model)
        """
        if pos_ids is not None:
            # Training mode: use explicit position IDs
            # pos_ids: (B, seq_len) -> (B, seq_len, d_model)
            pos_emb = torch.nn.functional.embedding(pos_ids, self.pe)
            return self.dropout(pos_emb)
        else:
            # Inference mode: sequential positions
            if x is not None:
                seq_len = x.size(1)
                pos_emb = self.pe[:seq_len, :].unsqueeze(0)  # (1, seq_len, d_model)
                return self.dropout(x + pos_emb)
            else:
                raise ValueError("Either x or pos_ids must be provided")


class TransformerEncoder(nn.Module):
    """Shared Transformer Encoder (Recursive Core) with gradient checkpointing support"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        self.gradient_checkpointing = False
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len) - True for positions to IGNORE (padding)
        Returns:
            (batch_size, seq_len, d_model)
        """
        if self.gradient_checkpointing and self.training:
            # Use gradient checkpointing
            from torch.utils.checkpoint import checkpoint
            
            def create_custom_forward(module):
                def custom_forward(x_input, mask_input):
                    return module(x_input, src_key_padding_mask=mask_input)
                return custom_forward
            
            return checkpoint(
                create_custom_forward(self.encoder),
                x,
                mask,
                use_reentrant=False
            )
        else:
            return self.encoder(x, src_key_padding_mask=mask)


class LinearDecoder(nn.Module):
    """Linear Projection Decoder (Extreme lightweight)"""
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, vocab_size)
        """
        return self.projection(x)


class TransformerDecoder(nn.Module):
    """Lightweight Transformer Decoder (1-layer) with gradient checkpointing support"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        vocab_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        decoder_layer = nn.TransformerEncoderLayer(  # Self-attention only
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=n_layers
        )
        
        self.projection = nn.Linear(d_model, vocab_size)
        self.gradient_checkpointing = False
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, vocab_size)
        """
        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            x = checkpoint(
                create_custom_forward(self.decoder),
                x,
                use_reentrant=False
            )
        else:
            x = self.decoder(x)
        
        return self.projection(x)


class GateMLP(nn.Module):
    """Gate MLP: Predicts remaining denoising steps"""
    
    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Scalar output (normalized 0~1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model) or (batch_size, d_model)
        Returns:
            (batch_size, 1) - normalized step prediction
        """
        # Mean pooling over sequence
        if x.dim() == 3:
            x = x.mean(dim=1)  # (batch_size, d_model)
        
        return torch.sigmoid(self.mlp(x))  # Normalize to [0, 1]


class RDT(nn.Module):
    """Recursive Denoising Transformer"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 1,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        decoder_type: str = 'linear',  # 'linear' or 'transformer'
        gate_hidden_dim: int = 256,
        gradient_checkpointing: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.decoder_type = decoder_type
        self.gradient_checkpointing = gradient_checkpointing
        
        # Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Encoder (Shared, Recursive)
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Decoder (Shallow)
        if decoder_type == 'linear':
            self.decoder = LinearDecoder(d_model, vocab_size)
        else:
            self.decoder = TransformerDecoder(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_decoder_layers,
                d_ff=d_ff,
                vocab_size=vocab_size,
                dropout=dropout
            )
        
        # Gate
        self.gate = GateMLP(d_model, gate_hidden_dim)
        
        self._init_weights()
        
        # Enable gradient checkpointing if requested
        if gradient_checkpointing:
            print("Gradient checkpointing enabled")
            self.encoder.gradient_checkpointing = True
            if hasattr(self.decoder, 'gradient_checkpointing'):
                self.decoder.gradient_checkpointing = True
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, pos_ids=None, attention_mask=None, is_first_step: bool = True):
        """
        Single step forward pass
        
        Args:
            x: (batch_size, seq_len) if is_first_step else (batch_size, seq_len, d_model)
            pos_ids: (batch_size, seq_len) - explicit position IDs for training
            attention_mask: (batch_size, seq_len) - 1 for valid tokens, 0 for padding
            is_first_step: whether this is the first step (needs embedding)
        
        Returns:
            hidden: (batch_size, seq_len, d_model)
            gate_pred: (batch_size, 1)
        """
        # Embedding (only for first step)
        if is_first_step:
            x = self.token_embedding(x) * math.sqrt(self.d_model)
            
            # Positional encoding
            if pos_ids is not None:
                # Training: use explicit position IDs (for reordered input)
                pos_emb = self.pos_encoding(pos_ids=pos_ids)
                x = x + pos_emb
            else:
                # Inference: sequential positions
                x = self.pos_encoding(x=x)
        # Otherwise x is already hidden state from previous step
        
        # Create attention mask for Transformer
        # PyTorch Transformer expects: True for positions to IGNORE
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)  # True for padding
        
        # Encode
        hidden = self.encoder(x, mask=src_key_padding_mask)
        
        # Gate
        gate_pred = self.gate(hidden)
        
        return hidden, gate_pred
    
    def recursive_forward(self, x, num_steps: int):
        """
        Multiple recursive steps (for training)
        
        Args:
            x: (batch_size, seq_len) - initial masked input
            num_steps: number of recursive steps
        
        Returns:
            all_logits: list of (batch_size, seq_len, vocab_size)
            all_gate_preds: list of (batch_size, 1)
        """
        all_logits = []
        all_gate_preds = []
        
        # First step
        hidden, logits, gate_pred = self.forward(x, is_first_step=True)
        all_logits.append(logits)
        all_gate_preds.append(gate_pred)
        
        # Recursive steps (no positional encoding)
        for _ in range(num_steps - 1):
            hidden, logits, gate_pred = self.forward(hidden, is_first_step=False)
            all_logits.append(logits)
            all_gate_preds.append(gate_pred)
        
        return all_logits, all_gate_preds
    
    def inference(self, x, max_steps: int = 20, threshold: float = 0.1):
        """
        Adaptive inference with gate-based stopping
        
        Args:
            x: (batch_size, seq_len) - input tokens
            max_steps: maximum number of recursive steps
            threshold: stop when gate prediction < threshold
        
        Returns:
            output_tokens: (batch_size, seq_len)
            num_steps_taken: int
        """
        self.eval()
        with torch.no_grad():
            # First step
            hidden, logits, gate_pred = self.forward(x, is_first_step=True)
            
            step = 1
            while step < max_steps and gate_pred.mean().item() > threshold:
                hidden, logits, gate_pred = self.forward(hidden, is_first_step=False)
                step += 1
            
            # Final output
            output_tokens = logits.argmax(dim=-1)
            
        return output_tokens, step