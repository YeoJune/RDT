"""Debug checkpoint loading for RDT model"""

import torch
from transformers import AutoTokenizer
from rdt.models import RDT
from rdt.utils import load_config

def debug_checkpoint(config_path, checkpoint_path):
    """Debug checkpoint loading issue"""
    
    print("="*80)
    print("CHECKPOINT LOADING DEBUG")
    print("="*80)
    
    # Load config
    config = load_config(config_path)
    tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])
    vocab_size = tokenizer.vocab_size
    
    # Load checkpoint
    print(f"\n1. Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check checkpoint keys
    print(f"\n2. Checkpoint keys: {list(checkpoint.keys())}")
    
    state_dict = checkpoint['model_state_dict']
    print(f"\n3. State dict has {len(state_dict)} keys")
    print(f"   First 10 keys:")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        print(f"      {key}")
    
    # Check for _orig_mod prefix
    has_orig_mod = any(k.startswith('_orig_mod.') for k in state_dict.keys())
    print(f"\n4. Has '_orig_mod.' prefix: {has_orig_mod}")
    
    # Check for weight tying
    has_token_emb = 'token_embedding.weight' in state_dict or '_orig_mod.token_embedding.weight' in state_dict
    has_output_proj = 'output_projection.weight' in state_dict or '_orig_mod.output_projection.weight' in state_dict
    print(f"\n5. Checkpoint contains:")
    print(f"   - token_embedding.weight: {has_token_emb}")
    print(f"   - output_projection.weight: {has_output_proj}")
    
    # Create model
    print(f"\n6. Creating model...")
    model = RDT(
        vocab_size=vocab_size,
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_encoder_layers=config['model']['n_encoder_layers'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout'],
        max_seq_len=config['data']['max_seq_length'],
        input_processor_layers=config['model'].get('input_processor_layers', 1),
        output_processor_layers=config['model'].get('output_processor_layers', 1),
        gate_hidden_dim=config['model']['gate_hidden_dim'],
        gate_num_layers=config['model']['gate_num_layers'],
        gate_num_heads=config['model']['gate_num_heads'],
        gate_dropout=config['model']['gate_dropout'],
        rope_base=config['model'].get('rope_base', 10000.0),
        gradient_checkpointing=False
    )
    
    print(f"   Model created successfully")
    print(f"   Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Check initial weight tying
    print(f"\n7. Initial weight tying check:")
    is_tied_before = model.output_projection.weight is model.token_embedding.weight
    print(f"   Are weights tied (before loading)? {is_tied_before}")
    print(f"   token_embedding.weight[0,:5]: {model.token_embedding.weight[0,:5]}")
    print(f"   output_projection.weight[0,:5]: {model.output_projection.weight[0,:5]}")
    
    # Try loading with strict=True first
    print(f"\n8. Attempting to load with strict=True...")
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"   ✓ Loaded successfully with strict=True")
    except Exception as e:
        print(f"   ✗ Failed with strict=True: {str(e)[:200]}")
        
        # Remove _orig_mod prefix if exists
        if has_orig_mod:
            print(f"\n9. Removing '_orig_mod.' prefix...")
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[len('_orig_mod.'):]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            print(f"   New state dict first 10 keys:")
            for i, key in enumerate(list(new_state_dict.keys())[:10]):
                print(f"      {key}")
            
            # Check output_projection.weight
            if 'output_projection.weight' not in new_state_dict:
                print(f"\n10. WARNING: output_projection.weight missing from checkpoint!")
                print(f"    This is the root cause - weight tying was removed during save")
            
            # Try loading again
            print(f"\n11. Attempting to load with modified state_dict (strict=False)...")
            try:
                missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
                print(f"   ✓ Loaded successfully")
                print(f"   Missing keys: {missing}")
                print(f"   Unexpected keys: {unexpected}")
                state_dict = new_state_dict  # Use modified dict for further checks
            except Exception as e:
                print(f"   ✗ Still failed: {str(e)}")
                return
    
    # Check weight tying after loading
    print(f"\n12. Weight tying check after loading:")
    is_tied_after = model.output_projection.weight is model.token_embedding.weight
    print(f"   Are weights tied (after loading)? {is_tied_after}")
    print(f"   token_embedding.weight[0,:5]: {model.token_embedding.weight[0,:5]}")
    print(f"   output_projection.weight[0,:5]: {model.output_projection.weight[0,:5]}")
    print(f"   Are values same? {torch.allclose(model.token_embedding.weight, model.output_projection.weight)}")
    
    # Force weight tying
    print(f"\n13. Forcing weight tying...")
    model.output_projection.weight = model.token_embedding.weight
    is_tied_forced = model.output_projection.weight is model.token_embedding.weight
    print(f"   Are weights tied (after forcing)? {is_tied_forced}")
    
    # Test inference
    print(f"\n14. Testing inference...")
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Test with simple input
    test_input = torch.tensor([[101, 2000, 2003, 103, 102]], device=device)
    attention_mask = torch.ones_like(test_input)
    
    with torch.no_grad():
        # Test encode_tokens
        hidden = model.encode_tokens(test_input, attention_mask)
        print(f"   encode_tokens output[0,0,:5]: {hidden[0,0,:5]}")
        
        # Test decode
        logits = model.decode(hidden, attention_mask)
        print(f"   decode logits shape: {logits.shape}")
        print(f"   decode logits[0,3,:10]: {logits[0,3,:10]}")
        
        pred = logits.argmax(dim=-1)
        print(f"   Predicted tokens: {pred[0]}")
        
        # Test with different input
        test_input2 = torch.tensor([[101, 5678, 1234, 103, 102]], device=device)
        hidden2 = model.encode_tokens(test_input2, attention_mask)
        print(f"   encode_tokens output2[0,0,:5]: {hidden2[0,0,:5]}")
        print(f"   Are hidden states same for different inputs? {torch.allclose(hidden, hidden2)}")
        
        # Test full inference
        output_ids, steps = model.inference(
            test_input,
            attention_mask=attention_mask,
            max_steps=10,
            threshold=0.5,
            return_steps=True
        )
        print(f"   Full inference output: {output_ids[0]}")
        print(f"   Steps taken: {steps[0]}")
    
    print("\n" + "="*80)
    print("DIAGNOSIS:")
    print("="*80)
    
    if not has_output_proj:
        print("✗ CRITICAL: output_projection.weight was removed during checkpoint save")
        print("  → This is why all outputs are 101")
        print("  → Solution: Fix trainer.py to save weight-tied checkpoints properly")
    
    if not is_tied_forced:
        print("✗ CRITICAL: Weight tying cannot be restored")
    else:
        print("✓ Weight tying can be restored manually")
    
    if torch.allclose(hidden, hidden2):
        print("✗ CRITICAL: Model produces same hidden states for different inputs")
        print("  → Model parameters may not be loaded correctly")
    else:
        print("✓ Model processes different inputs differently")
    
    if (output_ids[0] == 101).all():
        print("✗ CRITICAL: Model outputs all 101 (CLS tokens)")
        print("  → Confirms weight tying issue")
    else:
        print("✓ Model produces varied outputs")
    
    print("="*80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    
    args = parser.parse_args()
    
    debug_checkpoint(args.config, args.checkpoint)