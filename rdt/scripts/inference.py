"""Inference script for RDT"""

import argparse
import torch
from transformers import AutoTokenizer

from rdt.models import RDT
from rdt.utils import load_config, load_checkpoint, get_device


def inference_interactive(model, tokenizer, device, max_steps=20, threshold=0.5):
    """Interactive inference mode"""
    model.eval()
    
    print("\n" + "="*50)
    print("RDT Interactive Inference")
    print("Enter text to denoise (or 'quit' to exit)")
    print("="*50 + "\n")
    
    while True:
        # Get input
        text = input("Input text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            break
        
        if not text:
            continue
        
        # Tokenize
        encoded = tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        input_ids = encoded['input_ids'].to(device)
        
        # Inference
        print("\nRunning inference...")
        with torch.no_grad():
            output_ids, num_steps = model.inference(
                input_ids,
                max_steps=max_steps,
                threshold=threshold
            )
        
        # Decode
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Display results
        print(f"\nInput:  {text}")
        print(f"Output: {output_text}")
        print(f"Steps taken: {num_steps}/{max_steps}")
        print("-" * 50 + "\n")


def inference_single(model, tokenizer, device, text, max_steps=20, threshold=0.5):
    """Single text inference"""
    model.eval()
    
    # Tokenize
    encoded = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    
    input_ids = encoded['input_ids'].to(device)
    
    # Inference
    with torch.no_grad():
        output_ids, num_steps = model.inference(
            input_ids,
            max_steps=max_steps,
            threshold=threshold
        )
    
    # Decode
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return output_text, num_steps


def main():
    parser = argparse.ArgumentParser(description='RDT Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional, will use checkpoint config)')
    parser.add_argument('--text', type=str, default=None,
                        help='Input text to denoise')
    parser.add_argument('--max_steps', type=int, default=20,
                        help='Maximum number of recursive steps')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Get config
    if args.config:
        config = load_config(args.config)
    else:
        config = checkpoint['config']
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])
    vocab_size = tokenizer.vocab_size
    
    # Create model
    print("Creating model...")
    model = RDT(
        vocab_size=vocab_size,
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_encoder_layers=config['model']['n_encoder_layers'],
        n_io_layers=config['model']['n_io_layers'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout'],
        max_seq_len=config['data']['max_seq_length'],
        gate_hidden_dim=config['model']['gate_hidden_dim'],
        gate_num_layers=config['model']['gate_num_layers'],
        gate_num_heads=config['model']['gate_num_heads'],
        gradient_checkpointing=config['model'].get('gradient_checkpointing', False)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded (epoch {checkpoint['epoch']}, step {checkpoint['step']})")
    
    # Run inference
    if args.interactive:
        inference_interactive(model, tokenizer, device, args.max_steps, config['model']['threshold'])
    elif args.text:
        output_text, num_steps = inference_single(
            model, tokenizer, device, args.text, args.max_steps, config['model']['threshold']
        )
        print(f"\nInput:  {args.text}")
        print(f"Output: {output_text}")
        print(f"Steps:  {num_steps}/{args.max_steps}")
    else:
        print("Please specify --text for single inference or --interactive for interactive mode")


if __name__ == '__main__':
    main()
