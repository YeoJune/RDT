"""Test BERT weight initialization for RDT"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rdt.bert_init import (
    initialize_rdt_with_bert,
    print_bert_compatibility,
    load_bert_weights_to_rdt
)
from rdt.model import RDT
from rdt.utils import count_parameters


def test_bert_medium_init():
    """Test initialization with BERT-medium (8 layers, 512-dim)"""
    print("\n" + "="*70)
    print("TEST 1: BERT-medium → RDT (1 + 6 + 1 layers)")
    print("="*70)
    
    model = initialize_rdt_with_bert(
        vocab_size=30522,
        bert_model_name="prajjwal1/bert-medium",
        n_encoder_layers=6,
        n_heads=8,
        n_io_layers=1,
        d_ff=2048,
        dropout=0.1,
        max_seq_len=512,
        gate_hidden_dim=512,
        gate_num_layers=3,
        gate_num_heads=8,
        verbose=True
    )
    
    print(f"\n✓ Model created successfully!")
    print(f"  Total parameters: {count_parameters(model):,}")
    print(f"  d_model: {model.d_model}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    x = torch.randint(0, 30522, (batch_size, seq_len))
    
    model.eval()
    with torch.no_grad():
        hidden, gate = model(x, is_first_step=True)
        logits = model.decode(hidden)
    
    print(f"\n✓ Forward pass successful!")
    print(f"  Input: {x.shape}")
    print(f"  Hidden: {hidden.shape}")
    print(f"  Gate: {gate.shape}")
    print(f"  Logits: {logits.shape}")
    
    return model


def test_bert_base_init():
    """Test initialization with BERT-base (12 layers, 768-dim)"""
    print("\n" + "="*70)
    print("TEST 2: BERT-base → RDT (1 + 10 + 1 layers)")
    print("="*70)
    
    model = initialize_rdt_with_bert(
        vocab_size=30522,
        bert_model_name="bert-base-uncased",
        n_encoder_layers=10,  # Use more layers with BERT-base
        n_heads=12,
        n_io_layers=1,
        d_ff=3072,
        dropout=0.1,
        max_seq_len=512,
        gate_hidden_dim=768,
        gate_num_layers=3,
        gate_num_heads=12,
        verbose=True
    )
    
    print(f"\n✓ Model created successfully!")
    print(f"  Total parameters: {count_parameters(model):,}")
    print(f"  d_model: {model.d_model}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    x = torch.randint(0, 30522, (batch_size, seq_len))
    
    model.eval()
    with torch.no_grad():
        hidden, gate = model(x, is_first_step=True)
        logits = model.decode(hidden)
    
    print(f"\n✓ Forward pass successful!")
    print(f"  Input: {x.shape}")
    print(f"  Hidden: {hidden.shape}")
    print(f"  Gate: {gate.shape}")
    print(f"  Logits: {logits.shape}")
    
    return model


def test_manual_bert_loading():
    """Test manual BERT loading to existing RDT model"""
    print("\n" + "="*70)
    print("TEST 3: Manual BERT loading to existing RDT")
    print("="*70)
    
    # Create RDT model first
    model = RDT(
        vocab_size=30522,
        d_model=512,
        n_heads=8,
        n_encoder_layers=6,
        n_io_layers=1,
        d_ff=2048,
        dropout=0.1,
        max_seq_len=512,
        gate_hidden_dim=512,
        gate_num_layers=3,
        gate_num_heads=8
    )
    
    print(f"\nRDT model created with random weights")
    print(f"  Total parameters: {count_parameters(model):,}")
    
    # Load BERT weights
    model = load_bert_weights_to_rdt(
        model,
        bert_model_name="prajjwal1/bert-medium",
        verbose=True
    )
    
    print(f"\n✓ BERT weights loaded successfully!")
    
    return model


def test_weight_verification():
    """Verify that weights are actually copied correctly"""
    print("\n" + "="*70)
    print("TEST 4: Weight Copy Verification")
    print("="*70)
    
    from transformers import AutoModel
    
    # Load BERT
    bert = AutoModel.from_pretrained("prajjwal1/bert-medium")
    
    # Create RDT with BERT init
    rdt = initialize_rdt_with_bert(
        vocab_size=30522,
        bert_model_name="prajjwal1/bert-medium",
        n_encoder_layers=6,
        n_heads=8,
        verbose=False
    )
    
    print("\nVerifying weight copies...")
    
    # Check token embeddings
    bert_emb = bert.embeddings.word_embeddings.weight
    rdt_emb = rdt.token_embedding.weight
    assert torch.allclose(bert_emb, rdt_emb), "Token embeddings mismatch!"
    print("  ✓ Token embeddings match")
    
    # Check input encoder (BERT layer 0)
    bert_layer0_qw = bert.encoder.layer[0].attention.self.query.weight
    rdt_input_qw = rdt.input_encoder.layers[0].self_attn.in_proj_weight[:512, :]  # First d_model rows
    assert torch.allclose(bert_layer0_qw, rdt_input_qw), "Input encoder Q weights mismatch!"
    print("  ✓ Input encoder weights match")
    
    # Check main encoder layer 0 (BERT layer 1)
    bert_layer1_ffn1 = bert.encoder.layer[1].intermediate.dense.weight
    rdt_main0_ffn1 = rdt.encoder_layers[0].ffn[0].weight
    assert torch.allclose(bert_layer1_ffn1, rdt_main0_ffn1), "Main encoder FFN weights mismatch!"
    print("  ✓ Main encoder weights match")
    
    # Check output decoder (BERT layer 7)
    bert_layer7_ffn2 = bert.encoder.layer[7].output.dense.weight
    rdt_output_ffn2 = rdt.output_decoder.layers[0].linear2.weight
    assert torch.allclose(bert_layer7_ffn2, rdt_output_ffn2), "Output decoder weights mismatch!"
    print("  ✓ Output decoder weights match")
    
    print("\n✓ All weight copies verified successfully!")


def test_dimension_mismatch():
    """Test error handling for dimension mismatch"""
    print("\n" + "="*70)
    print("TEST 5: Dimension Mismatch Error Handling")
    print("="*70)
    
    try:
        # Try to use BERT-base (768-dim) with d_model=512
        model = initialize_rdt_with_bert(
            vocab_size=30522,
            bert_model_name="bert-base-uncased",  # 768-dim
            d_model=512,  # Wrong dimension!
            n_encoder_layers=6,
            n_heads=8
        )
        print("  ✗ Should have raised ValueError!")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly caught dimension mismatch:")
        print(f"    {e}")
        return True


def test_insufficient_layers():
    """Test error handling for insufficient BERT layers"""
    print("\n" + "="*70)
    print("TEST 6: Insufficient Layers Error Handling")
    print("="*70)
    
    try:
        # Try to use BERT-small (4 layers) with n_encoder_layers=6
        model = initialize_rdt_with_bert(
            vocab_size=30522,
            bert_model_name="prajjwal1/bert-small",  # Only 4 layers!
            n_encoder_layers=6,  # Needs 8 layers total
            n_heads=8
        )
        print("  ✗ Should have raised ValueError!")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly caught insufficient layers:")
        print(f"    {e}")
        return True


def test_inference():
    """Test inference with BERT-initialized model"""
    print("\n" + "="*70)
    print("TEST 7: Inference Test")
    print("="*70)
    
    model = initialize_rdt_with_bert(
        vocab_size=30522,
        bert_model_name="prajjwal1/bert-medium",
        n_encoder_layers=6,
        n_heads=8,
        verbose=False
    )
    
    # Prepare input
    batch_size = 2
    seq_len = 64
    x = torch.randint(0, 30522, (batch_size, seq_len))
    
    print(f"\nRunning inference...")
    print(f"  Input shape: {x.shape}")
    
    # Run inference
    output_tokens, num_steps = model.inference(x, max_steps=10, threshold=0.02)
    
    print(f"\n✓ Inference completed!")
    print(f"  Output shape: {output_tokens.shape}")
    print(f"  Steps taken: {num_steps}")
    print(f"  Gate threshold: 0.02")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("BERT Weight Initialization Tests for RDT")
    print("="*70)
    
    # Print compatibility info
    print_bert_compatibility()
    
    # Run tests
    try:
        test_bert_medium_init()
        test_manual_bert_loading()
        test_weight_verification()
        test_dimension_mismatch()
        test_insufficient_layers()
        test_inference()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70 + "\n")
        
    except Exception as e:
        print("\n" + "="*70)
        print("✗ TEST FAILED!")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Optional: Test BERT-base (requires downloading large model)
    run_bert_base = input("\nRun BERT-base test? (requires download, y/n): ").lower() == 'y'
    if run_bert_base:
        try:
            test_bert_base_init()
        except Exception as e:
            print(f"\n✗ BERT-base test failed: {e}")
    
    return 0


if __name__ == "__main__":
    exit(main())
