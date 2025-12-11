"""BERT Weight Initialization for RDT - Optimized Layer Mapping"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional
import warnings


def load_bert_weights_to_rdt(
    rdt_model: nn.Module,
    bert_model_name: str = "prajjwal1/bert-medium",
    verbose: bool = True
) -> nn.Module:
    """
    Optimized BERT weight loading strategy for RDT.
    
    BERT Layer Mapping:
    -------------------
    BERT (8 layers) → RDT (1 + 6 + 1 = 8 layers)
    - BERT Layer 0    → RDT Input Encoder (low-level features)
    - BERT Layer 1-6  → RDT Main Encoder (semantic features, recursive)
    - BERT Layer 7    → RDT Output Decoder (high-level features)
    
    This preserves BERT's hierarchical learning:
    Syntactic → Semantic → Task-specific
    
    Args:
        rdt_model: RDT model instance
        bert_model_name: HuggingFace BERT model name
        verbose: Print loading progress
        
    Returns:
        RDT model with initialized BERT weights
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Loading BERT weights from: {bert_model_name}")
        print(f"{'='*60}\n")
    
    # Load BERT model and config (use safetensors to avoid security warning)
    bert = AutoModel.from_pretrained(bert_model_name, use_safetensors=True)
    bert_config = AutoConfig.from_pretrained(bert_model_name)
    
    num_bert_layers = bert_config.num_hidden_layers
    num_rdt_main = len(rdt_model.encoder_layers)
    
    # Validate layer compatibility
    required_layers = 1 + num_rdt_main + 1  # input + main + output
    if num_bert_layers < required_layers:
        raise ValueError(
            f"BERT has only {num_bert_layers} layers but RDT needs {required_layers} "
            f"(1 input + {num_rdt_main} main + 1 output). "
            f"Use a larger BERT model (e.g., 'bert-base-uncased' with 12 layers) "
            f"or reduce n_encoder_layers."
        )
    
    # Validate dimension compatibility
    if rdt_model.d_model != bert_config.hidden_size:
        raise ValueError(
            f"Dimension mismatch: RDT d_model={rdt_model.d_model}, "
            f"BERT hidden_size={bert_config.hidden_size}. They must match."
        )
    
    if verbose:
        print(f"BERT: {num_bert_layers} layers, d_model={bert_config.hidden_size}")
        print(f"RDT: 1 (input) + {num_rdt_main} (main) + 1 (output) = {required_layers} layers")
        print(f"\nLayer Mapping Strategy:")
        print(f"  BERT[0]     → RDT Input Encoder  (syntactic features)")
        print(f"  BERT[1-{num_rdt_main}]  → RDT Main Encoder   (semantic features)")
        print(f"  BERT[{num_bert_layers-1}]     → RDT Output Decoder (task features)")
        print()
    
    # =========================================================================
    # Step 1: Copy Token Embeddings
    # =========================================================================
    if verbose:
        print("[1/4] Copying token embeddings...")
    
    rdt_model.token_embedding.weight.data.copy_(
        bert.embeddings.word_embeddings.weight.data
    )
    
    if verbose:
        print(f"  ✓ Token embedding: {bert.embeddings.word_embeddings.weight.shape}")
    
    # =========================================================================
    # Step 2: Copy Input Encoder (BERT Layer 0)
    # =========================================================================
    if verbose:
        print("\n[2/4] Copying input encoder (BERT layer 0)...")
    
    bert_layer_0 = bert.encoder.layer[0]
    
    # RDT input_encoder is nn.TransformerEncoder with n_io_layers
    for i, rdt_layer in enumerate(rdt_model.input_encoder.layers):
        if i == 0:  # Only first layer gets BERT weights
            copy_standard_transformer_layer(bert_layer_0, rdt_layer)
            if verbose:
                print(f"  ✓ Input encoder layer 0 ← BERT layer 0")
        else:
            if verbose:
                print(f"  ⊘ Input encoder layer {i} (random init, n_io_layers > 1)")
    
    # =========================================================================
    # Step 3: Copy Main Encoder (BERT Middle Layers)
    # =========================================================================
    if verbose:
        print(f"\n[3/4] Copying main encoder (BERT layers 1-{num_rdt_main})...")
    
    for i in range(num_rdt_main):
        bert_layer_idx = i + 1  # BERT layers 1, 2, 3, 4, 5, 6
        bert_layer = bert.encoder.layer[bert_layer_idx]
        rdt_layer = rdt_model.encoder_layers[i]
        
        # Copy Self-Attention
        copy_attention_weights(
            bert_layer.attention.self,
            bert_layer.attention.output,
            rdt_layer.self_attn
        )
        
        # Copy FFN
        copy_ffn_weights(bert_layer, rdt_layer.ffn)
        
        # AdaLN remains zero-initialized (new conditioning mechanism)
        # norm1, norm2, norm3 are AdaptiveLayerNorm with zero-init projections
        
        if verbose:
            print(f"  ✓ RDT main[{i}] ← BERT layer[{bert_layer_idx}] (self-attn + FFN)")
            if i == 0:
                print(f"    (AdaLN projections remain zero-initialized for conditioning)")
    
    # =========================================================================
    # Step 4: Copy Output Decoder (BERT Last Layer)
    # =========================================================================
    if verbose:
        print(f"\n[4/4] Copying output decoder (BERT layer {num_bert_layers-1})...")
    
    bert_last_layer = bert.encoder.layer[num_bert_layers - 1]
    
    for i, rdt_layer in enumerate(rdt_model.output_decoder.layers):
        if i == 0:  # Only first layer gets BERT weights
            copy_standard_transformer_layer(bert_last_layer, rdt_layer)
            if verbose:
                print(f"  ✓ Output decoder layer 0 ← BERT layer {num_bert_layers-1}")
        else:
            if verbose:
                print(f"  ⊘ Output decoder layer {i} (random init, n_io_layers > 1)")
    
    # =========================================================================
    # Summary
    # =========================================================================
    if verbose:
        print(f"\n{'='*60}")
        print("✓ BERT weight loading complete!")
        print(f"{'='*60}")
        print("\nInitialized components:")
        print("  ✓ Token embeddings (tied with output projection)")
        print("  ✓ Input encoder (low-level syntactic features)")
        print("  ✓ Main encoder (mid-level semantic features, recursive)")
        print("  ✓ Output decoder (high-level task features)")
        print("\nRandomly initialized components:")
        print("  • Positional encoding")
        print("  • AdaLN conditioning projections (zero-init)")
        print("  • Timestep embedder (noise level)")
        print("  • Cross-attention (if used)")
        print("  • Gate MLP")
        print()
    
    return rdt_model


def copy_standard_transformer_layer(bert_layer, rdt_layer):
    """
    Copy BERT TransformerEncoderLayer to PyTorch nn.TransformerEncoderLayer.
    
    BERT Layer Structure:
    - attention.self: Q, K, V projections
    - attention.output: output projection + LayerNorm
    - intermediate: FFN first linear
    - output: FFN second linear + LayerNorm
    
    PyTorch TransformerEncoderLayer Structure:
    - self_attn: MultiheadAttention (fused QKV)
    - linear1: FFN first linear
    - linear2: FFN second linear
    - norm1, norm2: LayerNorm
    """
    # Self-Attention
    copy_attention_weights(
        bert_layer.attention.self,
        bert_layer.attention.output,
        rdt_layer.self_attn
    )
    
    # FFN - Linear1 (d_model -> d_ff)
    rdt_layer.linear1.weight.data.copy_(bert_layer.intermediate.dense.weight)
    rdt_layer.linear1.bias.data.copy_(bert_layer.intermediate.dense.bias)
    
    # FFN - Linear2 (d_ff -> d_model)
    rdt_layer.linear2.weight.data.copy_(bert_layer.output.dense.weight)
    rdt_layer.linear2.bias.data.copy_(bert_layer.output.dense.bias)
    
    # LayerNorm1 (after self-attention)
    rdt_layer.norm1.weight.data.copy_(bert_layer.attention.output.LayerNorm.weight)
    rdt_layer.norm1.bias.data.copy_(bert_layer.attention.output.LayerNorm.bias)
    
    # LayerNorm2 (after FFN)
    rdt_layer.norm2.weight.data.copy_(bert_layer.output.LayerNorm.weight)
    rdt_layer.norm2.bias.data.copy_(bert_layer.output.LayerNorm.bias)


def copy_attention_weights(bert_self_attn, bert_attn_output, rdt_mha):
    """
    Copy BERT self-attention weights to PyTorch MultiheadAttention.
    
    BERT:
    - attention.self.query/key/value: separate Q, K, V projections
    - attention.output.dense: output projection
    
    PyTorch MultiheadAttention:
    - in_proj_weight: fused [Q; K; V] weights (3 * d_model, d_model)
    - in_proj_bias: fused [Q; K; V] biases (3 * d_model)
    - out_proj: output projection
    """
    d_model = bert_self_attn.query.out_features
    
    # Get Q, K, V weights and biases
    q_weight = bert_self_attn.query.weight  # [d_model, d_model]
    k_weight = bert_self_attn.key.weight    # [d_model, d_model]
    v_weight = bert_self_attn.value.weight  # [d_model, d_model]
    
    q_bias = bert_self_attn.query.bias      # [d_model]
    k_bias = bert_self_attn.key.bias        # [d_model]
    v_bias = bert_self_attn.value.bias      # [d_model]
    
    # Fuse into single QKV projection
    if rdt_mha.in_proj_weight is not None:
        # Standard PyTorch MHA with fused projection
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)  # [3*d_model, d_model]
        qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)          # [3*d_model]
        
        rdt_mha.in_proj_weight.data.copy_(qkv_weight)
        rdt_mha.in_proj_bias.data.copy_(qkv_bias)
    else:
        # Separate Q, K, V projections (less common)
        rdt_mha.q_proj_weight.data.copy_(q_weight)
        rdt_mha.k_proj_weight.data.copy_(k_weight)
        rdt_mha.v_proj_weight.data.copy_(v_weight)
        
        if hasattr(rdt_mha, 'q_proj_bias'):
            rdt_mha.q_proj_bias.data.copy_(q_bias)
            rdt_mha.k_proj_bias.data.copy_(k_bias)
            rdt_mha.v_proj_bias.data.copy_(v_bias)
    
    # Output projection
    rdt_mha.out_proj.weight.data.copy_(bert_attn_output.dense.weight)
    rdt_mha.out_proj.bias.data.copy_(bert_attn_output.dense.bias)


def copy_ffn_weights(bert_layer, rdt_ffn):
    """
    Copy BERT FFN weights to RDT FFN.
    
    BERT FFN:
    - intermediate.dense: Linear(d_model, d_ff)
    - output.dense: Linear(d_ff, d_model)
    
    RDT FFN: nn.Sequential(
        [0] Linear(d_model, d_ff),
        [1] GELU(),
        [2] Dropout(),
        [3] Linear(d_ff, d_model)
    )
    """
    # First linear: d_model -> d_ff
    rdt_ffn[0].weight.data.copy_(bert_layer.intermediate.dense.weight)
    rdt_ffn[0].bias.data.copy_(bert_layer.intermediate.dense.bias)
    
    # Second linear: d_ff -> d_model
    rdt_ffn[3].weight.data.copy_(bert_layer.output.dense.weight)
    rdt_ffn[3].bias.data.copy_(bert_layer.output.dense.bias)


def initialize_rdt_with_bert(
    vocab_size: int,
    bert_model_name: str = "prajjwal1/bert-medium",
    d_model: Optional[int] = None,
    n_encoder_layers: int = 6,
    verbose: bool = True,
    **rdt_kwargs
) -> nn.Module:
    """
    Create RDT model and initialize with optimized BERT weight mapping.
    
    Usage:
        # Auto-detect d_model from BERT
        model = initialize_rdt_with_bert(
            vocab_size=30522,
            bert_model_name="prajjwal1/bert-medium",
            n_encoder_layers=6,
            n_heads=8,
            n_io_layers=1,
            d_ff=2048,
            dropout=0.1
        )
        
        # Explicit d_model (must match BERT)
        model = initialize_rdt_with_bert(
            vocab_size=30522,
            bert_model_name="bert-base-uncased",
            d_model=768,
            n_encoder_layers=10,
            n_heads=12,
            d_ff=3072
        )
    
    Args:
        vocab_size: Vocabulary size
        bert_model_name: HuggingFace BERT model name
        d_model: Hidden dimension (auto-detected from BERT if None)
        n_encoder_layers: Number of main encoder layers (recursive)
        verbose: Print initialization details
        **rdt_kwargs: Additional RDT model arguments
        
    Returns:
        RDT model with BERT-initialized weights
    """
    from .model import RDT
    
    # Load BERT config for validation
    bert_config = AutoConfig.from_pretrained(bert_model_name)
    
    # Auto-set d_model from BERT if not specified
    if d_model is None:
        d_model = bert_config.hidden_size
        if verbose:
            print(f"Auto-setting d_model={d_model} from BERT config")
    
    # Validate dimension match
    if d_model != bert_config.hidden_size:
        raise ValueError(
            f"d_model mismatch: RDT={d_model}, BERT={bert_config.hidden_size}. "
            f"They must match for weight transfer."
        )
    
    # Validate layer compatibility
    required_layers = 1 + n_encoder_layers + 1
    if bert_config.num_hidden_layers < required_layers:
        raise ValueError(
            f"BERT has {bert_config.num_hidden_layers} layers but RDT needs {required_layers} "
            f"(1 input + {n_encoder_layers} main + 1 output). "
            f"Use a larger BERT model or reduce n_encoder_layers."
        )
    
    # Create RDT model
    rdt_model = RDT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_encoder_layers=n_encoder_layers,
        **rdt_kwargs
    )
    
    # Load BERT weights with optimized mapping
    rdt_model = load_bert_weights_to_rdt(rdt_model, bert_model_name, verbose=verbose)
    
    return rdt_model


# ============================================================================
# Recommended BERT Models for RDT
# ============================================================================
RECOMMENDED_BERT_MODELS = {
    "prajjwal1/bert-tiny": {
        "layers": 2,
        "hidden": 128,
        "heads": 2,
        "notes": "Too small for RDT (needs min 8 layers)"
    },
    "prajjwal1/bert-mini": {
        "layers": 4,
        "hidden": 256,
        "heads": 4,
        "notes": "Too small for RDT (needs min 8 layers)"
    },
    "prajjwal1/bert-small": {
        "layers": 4,
        "hidden": 512,
        "heads": 8,
        "notes": "Too small for RDT (needs min 8 layers)"
    },
    "prajjwal1/bert-medium": {
        "layers": 8,
        "hidden": 512,
        "heads": 8,
        "notes": "✓ Perfect for n_encoder_layers=6 (default RDT config)"
    },
    "bert-base-uncased": {
        "layers": 12,
        "hidden": 768,
        "heads": 12,
        "notes": "✓ Good for n_encoder_layers=6-10"
    },
    "bert-large-uncased": {
        "layers": 24,
        "hidden": 1024,
        "heads": 16,
        "notes": "✓ Good for large RDT (n_encoder_layers=10-22)"
    }
}


def print_bert_compatibility():
    """Print recommended BERT models for RDT."""
    print("\n" + "="*70)
    print("Recommended BERT Models for RDT Initialization")
    print("="*70)
    print(f"\n{'Model':<30} {'Layers':>7} {'Hidden':>7} {'Heads':>6}  Notes")
    print("-"*70)
    
    for model_name, info in RECOMMENDED_BERT_MODELS.items():
        print(f"{model_name:<30} {info['layers']:>7} {info['hidden']:>7} {info['heads']:>6}  {info['notes']}")
    
    print("\n" + "="*70)
    print("RDT Layer Requirements: 1 (input) + N (main) + 1 (output) = N+2 layers")
    print("Example: n_encoder_layers=6 requires minimum 8 BERT layers")
    print("="*70 + "\n")


if __name__ == "__main__":
    print_bert_compatibility()
