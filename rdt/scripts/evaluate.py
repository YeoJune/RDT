"""Unified evaluation script for all models"""

import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer

from rdt.models import RDT, MLM
from rdt.models.cmlm import CMLM
from rdt.models.mdlm import MDLM
from rdt.data import create_dataloaders, create_mlm_dataloaders, create_cmlm_dataloaders, create_mdlm_dataloaders
from rdt.evaluation import Evaluator
from rdt.utils import load_config, load_checkpoint, create_model_from_config


def main():
    parser = argparse.ArgumentParser(description='Evaluate RDT or Baseline Models')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Path to save evaluation results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'validation', 'test'],
                        help='Dataset split to evaluate on')
    # RDT-specific arguments
    parser.add_argument('--max-steps', type=int, default=20,
                        help='Max recursive steps for RDT')
    parser.add_argument('--threshold', type=float, default=0.02,
                        help='Gate threshold for RDT')
    # CMLM-specific arguments
    parser.add_argument('--max-iterations', type=int, default=10,
                        help='Max mask-predict iterations for CMLM')
    # MDLM-specific arguments
    parser.add_argument('--num-steps', type=int, default=1000,
                        help='Number of denoising steps for MDLM')
    parser.add_argument('--sampler', type=str, default='ddpm_cache',
                        choices=['ddpm', 'ddpm_cache', 'analytic'],
                        help='Sampling strategy for MDLM')
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Determine device
    if args.device:
        device_name = args.device
    else:
        device_name = config.get('device', 'cuda')
    
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Determine model type
    model_type = config.get('model_type', 'rdt').lower()
    print(f"Model type: {model_type.upper()}")
    
    # Load model
    print(f"\nLoading model from checkpoint: {args.checkpoint}")
    
    if model_type == 'rdt':
        # Load RDT model
        from rdt.data.rdt_preprocessor import RDTPreprocessor
        
        tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])
        vocab_size = tokenizer.vocab_size
        
        model = create_model_from_config(config, vocab_size)
        checkpoint = load_checkpoint(args.checkpoint, model)
        
        # Create preprocessor
        preprocessor = RDTPreprocessor(tokenizer, config).to(device)
        
        # Create dataloader
        print(f"\nPreparing RDT {args.split} data...")
        # Temporarily override split in config
        original_split = config['data'].get('split', 'train')
        config['data']['split'] = args.split
        
        if args.split == 'train':
            dataloader, _ = create_dataloaders(config)
        elif args.split == 'validation':
            _, dataloader = create_dataloaders(config)
        else:  # test
            # For test, use validation loader structure but with test split
            _, dataloader = create_dataloaders(config)
        
        config['data']['split'] = original_split
        
    elif model_type == 'mlm':
        # Load baseline MLM model
        model_name = config['model']['name']
        model = MLM(model_name=model_name)
        checkpoint = load_checkpoint(args.checkpoint, model)
        
        # Create dataloader
        print(f"\nPreparing MLM {args.split} data...")
        _, dataloader = create_mlm_dataloaders(config)
        
    elif model_type == 'cmlm':
        # Load CMLM model
        model = CMLM.from_config(config)
        checkpoint = load_checkpoint(args.checkpoint, model)
        
        # Create dataloader
        print(f"\nPreparing CMLM {args.split} data...")
        _, dataloader = create_cmlm_dataloaders(config)
        
    elif model_type == 'mdlm':
        # Load MDLM model
        model = MDLM.from_config(config)
        checkpoint = load_checkpoint(args.checkpoint, model)
        
        # Create dataloader
        print(f"\nPreparing MDLM {args.split} data...")
        _, dataloader = create_mdlm_dataloaders(config)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'rdt', 'mlm', 'cmlm', or 'mdlm'")
    
    model = model.to(device)
    print("Model loaded successfully!")
    
    # Create evaluator
    if model_type == 'rdt':
        evaluator = Evaluator(model, device, model_type=model_type, preprocessor=preprocessor)
    else:
        evaluator = Evaluator(model, device, model_type=model_type)
    
    # Evaluate
    print("\n" + "="*60)
    print(f"Evaluating on {args.split} split...")
    print("="*60 + "\n")
    
    if model_type == 'rdt':
        results = evaluator.evaluate(
            dataloader,
            max_steps=args.max_steps,
            threshold=args.threshold
        )
    elif model_type == 'cmlm':
        results = evaluator.evaluate(
            dataloader,
            max_iterations=args.max_iterations
        )
    elif model_type == 'mdlm':
        results = evaluator.evaluate(
            dataloader,
            num_steps=args.num_steps,
            sampler=args.sampler
        )
    else:  # mlm
        results = evaluator.evaluate(dataloader)
    
    # Add metadata
    results['model_type'] = model_type
    results['checkpoint'] = args.checkpoint
    results['split'] = args.split
    
    if model_type == 'rdt':
        results['max_steps'] = args.max_steps
        results['threshold'] = args.threshold
    elif model_type == 'cmlm':
        results['max_iterations'] = args.max_iterations
    elif model_type == 'mdlm':
        results['num_steps'] = args.num_steps
        results['sampler'] = args.sampler
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    output_path = Path(args.output)
    evaluator.save_results(results, output_path)


if __name__ == '__main__':
    main()
