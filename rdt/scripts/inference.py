"""Unified inference script for RDT, MLM, and CMLM models"""

import argparse
import torch
from transformers import AutoTokenizer

from rdt.models import RDT, MLM
from rdt.models.cmlm import CMLM
from rdt.models.mdlm import MDLM
from rdt.utils import load_config, load_checkpoint, get_device, create_model_from_config


def inference_interactive_rdt(model, tokenizer, device, max_steps=20, threshold=0.5):
    """Interactive inference mode for RDT"""
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
        attention_mask = encoded.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Inference
        print("\nRunning inference...")
        with torch.no_grad():
            output_ids, num_steps = model.inference(
                input_ids,
                attention_mask=attention_mask,
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


def inference_interactive_mlm(model, tokenizer, device):
    """Interactive inference mode for MLM (single-pass)"""
    model.eval()
    
    print("\n" + "="*50)
    print("MLM Interactive Inference (Single-pass)")
    print("Enter text with [MASK] tokens (or 'quit' to exit)")
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
        attention_mask = encoded.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Inference
        print("\nRunning inference...")
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            output_ids = logits.argmax(dim=-1)
        
        # Decode
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Display results
        print(f"\nInput:  {text}")
        print(f"Output: {output_text}")
        print("-" * 50 + "\n")


def inference_interactive_mdlm(model, tokenizer, device, num_steps=1000, sampler='ddpm_cache'):
    """Interactive inference mode for MDLM (diffusion sampling)"""
    model.eval()
    
    print("\n" + "="*50)
    print(f"MDLM Interactive Inference ({num_steps} steps, {sampler} sampler)")
    print("Enter text (will be fully masked and reconstructed)")
    print("Or enter text with [MASK] tokens for partial reconstruction")
    print("Type 'quit' to exit")
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
        attention_mask = encoded.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Mask all non-special tokens if no [MASK] in input
        if tokenizer.mask_token_id not in input_ids:
            masked_input_ids = torch.full_like(input_ids, tokenizer.mask_token_id)
            # Keep special tokens
            special_mask = (input_ids == tokenizer.cls_token_id) | \
                          (input_ids == tokenizer.sep_token_id) | \
                          (input_ids == tokenizer.pad_token_id)
            masked_input_ids[special_mask] = input_ids[special_mask]
        else:
            masked_input_ids = input_ids
        
        # Inference
        print("\nRunning inference...")
        with torch.no_grad():
            output_ids, actual_steps = model.inference(
                masked_input_ids,
                attention_mask=attention_mask,
                num_steps=num_steps,
                sampler=sampler
            )
        
        # Decode
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Display results
        print(f"\nInput:  {text}")
        print(f"Output: {output_text}")
        print(f"Steps: {actual_steps} ({sampler} sampler)")
        print("-" * 50 + "\n")


def inference_interactive_cmlm(model, tokenizer, device, max_iterations=10):
    """Interactive inference mode for CMLM (iterative refinement)"""
    model.eval()
    
    print("\n" + "="*50)
    print(f"CMLM Interactive Inference ({max_iterations} iterations)")
    print("Enter text (will be fully masked and reconstructed)")
    print("Or enter text with [MASK] tokens for partial reconstruction")
    print("Type 'quit' to exit")
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
        attention_mask = encoded.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Mask all non-special tokens if no [MASK] in input
        if tokenizer.mask_token_id not in input_ids:
            masked_input_ids = torch.full_like(input_ids, tokenizer.mask_token_id)
            # Keep special tokens
            special_mask = (input_ids == tokenizer.cls_token_id) | \
                          (input_ids == tokenizer.sep_token_id) | \
                          (input_ids == tokenizer.pad_token_id)
            masked_input_ids[special_mask] = input_ids[special_mask]
        else:
            masked_input_ids = input_ids
        
        # Inference
        print("\nRunning inference...")
        with torch.no_grad():
            output_ids, _ = model.inference(
                masked_input_ids,
                attention_mask=attention_mask,
                max_iterations=max_iterations
            )
        
        # Decode
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Display results
        print(f"\nInput:  {text}")
        print(f"Output: {output_text}")
        print(f"Iterations: {max_iterations}")
        print("-" * 50 + "\n")


def inference_single(model, tokenizer, device, text, model_type='rdt', **kwargs):
    """Single text inference for any model type"""
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
    attention_mask = encoded.get('attention_mask')
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    # Inference based on model type
    with torch.no_grad():
        if model_type == 'rdt':
            max_steps = kwargs.get('max_steps', 20)
            threshold = kwargs.get('threshold', 0.5)
            output_ids, num_steps = model.inference(
                input_ids,
                attention_mask=attention_mask,
                max_steps=max_steps,
                threshold=threshold
            )
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return output_text, num_steps
            
        elif model_type == 'mlm':
            logits = model(input_ids, attention_mask)
            output_ids = logits.argmax(dim=-1)
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return output_text, 1
            
        elif model_type == 'cmlm':
            max_iterations = kwargs.get('max_iterations', 10)
            
            # Mask input if no [MASK] tokens present
            if tokenizer.mask_token_id not in input_ids:
                masked_input_ids = torch.full_like(input_ids, tokenizer.mask_token_id)
                special_mask = (input_ids == tokenizer.cls_token_id) | \
                              (input_ids == tokenizer.sep_token_id) | \
                              (input_ids == tokenizer.pad_token_id)
                masked_input_ids[special_mask] = input_ids[special_mask]
            else:
                masked_input_ids = input_ids
            
            output_ids, _ = model.inference(
                masked_input_ids,
                attention_mask=attention_mask,
                max_iterations=max_iterations
            )
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return output_text, max_iterations
            
        elif model_type == 'mdlm':
            num_steps = kwargs.get('num_steps', 1000)
            sampler = kwargs.get('sampler', 'ddpm_cache')
            
            # Mask input if no [MASK] tokens present
            if tokenizer.mask_token_id not in input_ids:
                masked_input_ids = torch.full_like(input_ids, tokenizer.mask_token_id)
                special_mask = (input_ids == tokenizer.cls_token_id) | \
                              (input_ids == tokenizer.sep_token_id) | \
                              (input_ids == tokenizer.pad_token_id)
                masked_input_ids[special_mask] = input_ids[special_mask]
            else:
                masked_input_ids = input_ids
            
            output_ids, actual_steps = model.inference(
                masked_input_ids,
                attention_mask=attention_mask,
                num_steps=num_steps,
                sampler=sampler
            )
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return output_text, actual_steps


def main():
    parser = argparse.ArgumentParser(description='Unified Model Inference (RDT/MLM/CMLM)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional, will use checkpoint config)')
    parser.add_argument('--text', type=str, default=None,
                        help='Input text for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    # RDT-specific args
    parser.add_argument('--max-steps', type=int, default=20,
                        help='Maximum number of recursive steps (RDT)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Gate threshold for stopping (RDT)')
    # CMLM-specific args
    parser.add_argument('--max-iterations', type=int, default=10,
                        help='Maximum mask-predict iterations (CMLM)')
    # MDLM-specific args
    parser.add_argument('--num-steps', type=int, default=1000,
                        help='Number of denoising steps (MDLM)')
    parser.add_argument('--sampler', type=str, default='ddpm_cache',
                        choices=['ddpm', 'ddpm_cache', 'analytic'],
                        help='Sampling strategy (MDLM)')
    
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
    
    # Determine model type
    model_type = config.get('model_type', 'rdt').lower()
    print(f"Model type: {model_type.upper()}")
    
    # Load tokenizer
    if model_type == 'rdt':
        tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])
    elif model_type in ['mlm', 'cmlm', 'mdlm']:
        # For baseline models, use architecture from config
        architecture = config['model'].get('architecture', config['model'].get('name', 'bert-base-uncased'))
        tokenizer = AutoTokenizer.from_pretrained(architecture)
    
    # Create and load model
    print("Creating model...")
    if model_type == 'rdt':
        vocab_size = tokenizer.vocab_size
        model = create_model_from_config(config, vocab_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    elif model_type == 'mlm':
        model = MLM.from_config(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    elif model_type == 'cmlm':
        model = CMLM.from_config(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    elif model_type == 'mdlm':
        model = MDLM.from_config(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded (epoch {checkpoint.get('epoch', 0)}, step {checkpoint.get('step', 0)})")
    
    # Get model-specific parameters
    if model_type == 'rdt':
        threshold = args.threshold if args.threshold is not None else config['model'].get('threshold', 0.02)
        max_steps = args.max_steps
        print(f"RDT parameters: max_steps={max_steps}, threshold={threshold}")
    elif model_type == 'cmlm':
        max_iterations = args.max_iterations
        print(f"CMLM parameters: max_iterations={max_iterations}")
    elif model_type == 'mdlm':
        num_steps = args.num_steps
        sampler = args.sampler
        print(f"MDLM parameters: num_steps={num_steps}, sampler={sampler}")
    
    # Run inference
    if args.interactive:
        if model_type == 'rdt':
            inference_interactive_rdt(model, tokenizer, device, max_steps, threshold)
        elif model_type == 'mlm':
            inference_interactive_mlm(model, tokenizer, device)
        elif model_type == 'cmlm':
            inference_interactive_cmlm(model, tokenizer, device, max_iterations)
        elif model_type == 'mdlm':
            inference_interactive_mdlm(model, tokenizer, device, num_steps, sampler)
            
    elif args.text:
        if model_type == 'rdt':
            output_text, num_steps = inference_single(
                model, tokenizer, device, args.text, model_type='rdt',
                max_steps=max_steps, threshold=threshold
            )
            print(f"\nInput:  {args.text}")
            print(f"Output: {output_text}")
            print(f"Steps:  {num_steps}/{max_steps}")
            
        elif model_type == 'mlm':
            output_text, _ = inference_single(
                model, tokenizer, device, args.text, model_type='mlm'
            )
            print(f"\nInput:  {args.text}")
            print(f"Output: {output_text}")
            
        elif model_type == 'cmlm':
            output_text, iterations = inference_single(
                model, tokenizer, device, args.text, model_type='cmlm',
                max_iterations=max_iterations
            )
            print(f"\nInput:  {args.text}")
            print(f"Output: {output_text}")
            print(f"Iterations: {iterations}")
            
        elif model_type == 'mdlm':
            output_text, steps = inference_single(
                model, tokenizer, device, args.text, model_type='mdlm',
                num_steps=num_steps, sampler=sampler
            )
            print(f"\nInput:  {args.text}")
            print(f"Output: {output_text}")
            print(f"Steps: {steps} ({sampler} sampler)")
    else:
        print("Please specify --text for single inference or --interactive for interactive mode")


if __name__ == '__main__':
    main()
