"""Quick test script to verify model implementation"""

import torch
from rdt.model import RDT
from rdt.utils import count_parameters

def test_model():
    print("Testing RDT Model Implementation...")
    print("=" * 50)
    
    # Small test configuration
    vocab_size = 30522  # BERT vocab size
    batch_size = 2
    seq_len = 32
    d_model = 256
    n_heads = 4
    n_encoder_layers = 3
    n_decoder_layers = 1
    
    # Create model
    print("\n1. Creating model...")
    model = RDT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        d_ff=1024,
        dropout=0.1,
        max_seq_len=512,
        decoder_type='linear',
        gate_hidden_dim=128
    )
    
    num_params = count_parameters(model)
    print(f"   Model created with {num_params:,} parameters")
    
    # Test single forward pass
    print("\n2. Testing single forward pass...")
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    hidden, logits, gate_pred = model(x, is_first_step=True)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Hidden shape: {hidden.shape}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Gate prediction shape: {gate_pred.shape}")
    
    assert hidden.shape == (batch_size, seq_len, d_model)
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert gate_pred.shape == (batch_size, 1)
    print("   ✓ Shapes correct!")
    
    # Test recursive forward
    print("\n3. Testing recursive forward (3 steps)...")
    num_steps = 3
    all_logits, all_gate_preds = model.recursive_forward(x, num_steps)
    
    print(f"   Number of logits: {len(all_logits)}")
    print(f"   Number of gate predictions: {len(all_gate_preds)}")
    
    assert len(all_logits) == num_steps
    assert len(all_gate_preds) == num_steps
    
    for i, (logits, gate) in enumerate(zip(all_logits, all_gate_preds)):
        print(f"   Step {i+1}: logits {logits.shape}, gate {gate.shape}")
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert gate.shape == (batch_size, 1)
    
    print("   ✓ Recursive forward works!")
    
    # Test inference
    print("\n4. Testing inference with adaptive stopping...")
    output_tokens, num_steps_taken = model.inference(x, max_steps=10, threshold=0.1)
    
    print(f"   Output shape: {output_tokens.shape}")
    print(f"   Steps taken: {num_steps_taken}/10")
    
    assert output_tokens.shape == (batch_size, seq_len)
    print("   ✓ Inference works!")
    
    # Test different decoder types
    print("\n5. Testing transformer decoder...")
    model_transformer = RDT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=1,
        d_ff=1024,
        dropout=0.1,
        decoder_type='transformer',
        gate_hidden_dim=128
    )
    
    hidden, logits, gate_pred = model_transformer(x, is_first_step=True)
    print(f"   Transformer decoder output: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, vocab_size)
    print("   ✓ Transformer decoder works!")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)


if __name__ == '__main__':
    test_model()
