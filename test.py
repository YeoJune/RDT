"""Simple RDT Test - Minimal code to verify inference works"""

import torch
from transformers import AutoTokenizer
from rdt.models import RDT
from rdt.utils import load_config
import argparse

def test_single_sample(model, tokenizer, device):
    """Test with a single sample"""
    print("\n" + "="*80)
    print("Test 1: Single Sample (Original)")
    print("="*80)
    
    # Simple test
    text = "The quick brown fox jumps over the lazy dog."
    encoded = tokenizer(text, return_tensors='pt')
    tokens = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    print(f"Input: {text}")
    print(f"Tokens: {tokens[0][:10]}")
    
    with torch.no_grad():
        # Direct encode-decode
        hidden = model.encode_tokens(tokens, attention_mask)
        logits = model.decode(hidden, attention_mask)
        pred_direct = logits.argmax(dim=-1)
        
        print(f"Direct pred: {pred_direct[0][:10]}")
        print(f"Direct text: {tokenizer.decode(pred_direct[0], skip_special_tokens=True)}")
        
        # Full inference
        output, steps = model.inference(
            tokens,
            attention_mask=attention_mask,
            max_steps=10,
            threshold=0.5,
            return_steps=True
        )
        
        print(f"Inference output: {output[0][:10]}")
        print(f"Inference text: {tokenizer.decode(output[0], skip_special_tokens=True)}")
        print(f"Steps taken: {steps[0]}")


def test_masked_sample(model, tokenizer, device):
    """Test with masked input"""
    print("\n" + "="*80)
    print("Test 2: Masked Input (50% masking)")
    print("="*80)
    
    text = "The quick brown fox jumps over the lazy dog."
    encoded = tokenizer(text, return_tensors='pt')
    tokens = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Mask 50% of tokens (excluding special tokens)
    mask_token_id = tokenizer.mask_token_id
    masked_tokens = tokens.clone()
    
    # Mask positions 2, 4, 6, 8 (skip CLS and SEP)
    seq_len = tokens.shape[1]
    for i in range(2, min(seq_len-1, 10), 2):
        masked_tokens[0, i] = mask_token_id
    
    print(f"Original: {tokenizer.decode(tokens[0], skip_special_tokens=True)}")
    print(f"Masked:   {tokenizer.decode(masked_tokens[0], skip_special_tokens=True)}")
    print(f"Masked tokens: {masked_tokens[0][:10]}")
    
    with torch.no_grad():
        output, steps = model.inference(
            masked_tokens,
            attention_mask=attention_mask,
            max_steps=10,
            threshold=0.5,
            return_steps=True
        )
        
        print(f"Output tokens: {output[0][:10]}")
        print(f"Reconstructed: {tokenizer.decode(output[0], skip_special_tokens=True)}")
        print(f"Steps taken: {steps[0]}")
        
        # Check if all outputs are 101
        if (output[0] == 101).all():
            print("\n⚠️  WARNING: All outputs are 101 (CLS token) - MODEL NOT WORKING!")
        else:
            print("\n✓ Model producing varied outputs")


def test_batch(model, tokenizer, device):
    """Test with batch"""
    print("\n" + "="*80)
    print("Test 3: Batch Processing (4 samples)")
    print("="*80)
    
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world, this is a test.",
        "Machine learning is fascinating.",
        "Natural language processing works well."
    ]
    
    # Tokenize
    encoded = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    tokens = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    print(f"Batch shape: {tokens.shape}")
    print(f"Sample 1 tokens: {tokens[0][:10]}")
    
    with torch.no_grad():
        output, steps = model.inference(
            tokens,
            attention_mask=attention_mask,
            max_steps=10,
            threshold=0.5,
            return_steps=True
        )
        
        print(f"\nResults:")
        for i in range(len(texts)):
            pred_text = tokenizer.decode(output[i], skip_special_tokens=True)
            print(f"  [{i}] Steps: {steps[i]}, Output: {pred_text[:50]}...")
        
        # Check if all outputs are 101
        all_101 = all((output[i] == 101).sum() > output.shape[1] * 0.9 for i in range(len(texts)))
        if all_101:
            print("\n⚠️  WARNING: Most outputs are 101 - BATCH PROCESSING BROKEN!")
        else:
            print("\n✓ Batch processing working")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    
    from pathlib import Path
    checkpoint_path = Path(args.checkpoint)
    
    model = RDT(
        vocab_size=tokenizer.vocab_size,
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
    
    # Check if checkpoint is a directory or file
    if checkpoint_path.is_dir():
        # Load from accelerator saved directory
        safetensors_path = checkpoint_path / 'model.safetensors'
        if safetensors_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(safetensors_path))
            print(f"Loaded from {safetensors_path}")
        else:
            raise FileNotFoundError(f"No model.safetensors found in {checkpoint_path}")
    else:
        # Load from .pt file
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
    
    # Handle _orig_mod prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    
    # Force weight tying
    model.output_projection.weight = model.token_embedding.weight
    
    # Verify
    is_tied = model.output_projection.weight is model.token_embedding.weight
    print(f"Weight tying: {'✓' if is_tied else '✗'}")
    
    # Check parameter values
    print(f"token_embedding[101,:5]: {model.token_embedding.weight[101,:5]}")
    print(f"output_projection[101,:5]: {model.output_projection.weight[101,:5]}")
    
    model.eval()
    
    # Run tests
    test_single_sample(model, tokenizer, device)
    test_masked_sample(model, tokenizer, device)
    test_batch(model, tokenizer, device)
    
    print("\n" + "="*80)
    print("Testing Complete")
    print("="*80)


if __name__ == '__main__':
    main()