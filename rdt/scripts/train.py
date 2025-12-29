"""Unified training script for all models (RDT and baselines)"""

import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer

from rdt.models import RDT, MLM
from rdt.models.cmlm import CMLM
from rdt.data import create_dataloaders, create_mlm_dataloaders, create_cmlm_dataloaders
from rdt.training import RDTTrainer, MLMTrainer
from rdt.utils import load_config, merge_configs, set_seed, get_device, create_model_from_config


def main():
    parser = argparse.ArgumentParser(description='Train RDT or Baseline Models')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--override', type=str, default=None,
                        help='Path to override config file')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from (loads model, optimizer, scheduler, and training state)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained weights to load (loads only model weights, starts training from scratch)')
    
    args = parser.parse_args()
    
    # Check for conflicting arguments
    if args.checkpoint and args.pretrained:
        raise ValueError("Cannot use both --checkpoint and --pretrained flags simultaneously. "
                         "Use --checkpoint to resume training with all states, or --pretrained to load only weights.")
    
    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    if args.override:
        print(f"Loading override config from {args.override}")
        override_config = load_config(args.override)
        config = merge_configs(config, override_config)
    
    # Override device if specified
    if args.device:
        config['device'] = args.device
    
    # Set seed
    set_seed(config['seed'])
    
    # Get device
    device = get_device(config['device'])
    print(f"Using device: {device}")
    
    # Determine model type
    model_type = config.get('model_type', 'rdt').lower()
    print(f"\nModel type: {model_type.upper()}")
    
    # Create model and dataloaders based on type
    if model_type == 'rdt':
        # RDT model
        print("\n" + "="*60)
        print("Training RDT Model")
        print("="*60)
        
        # Create dataloaders
        print("\nPreparing RDT data...")
        train_loader, val_loader = create_dataloaders(config)
        
        # Get vocab size
        tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])
        vocab_size = tokenizer.vocab_size
        print(f"Vocabulary size: {vocab_size}")
        
        # Create model
        print("\nInitializing RDT model...")
        model = create_model_from_config(config, vocab_size)
        
        # Create trainer
        trainer = RDTTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )
        
    elif model_type in ['mlm', 'cmlm']:
        # Baseline models (MLM or CMLM)
        print("\n" + "="*60)
        print(f"Training {model_type.upper()} Model")
        print("="*60)
        
        # Create model from config
        print(f"\nInitializing {model_type.upper()} model...")
        
        if model_type == 'mlm':
            model = MLM.from_config(config)
        elif model_type == 'cmlm':
            model = CMLM.from_config(config)
        
        print(f"\nModel parameters: {model.count_parameters()/1e6:.1f}M")
        
        # Create dataloaders
        print(f"\nPreparing {model_type.upper()} data...")
        if model_type == 'mlm':
            train_loader, val_loader = create_mlm_dataloaders(config)
        elif model_type == 'cmlm':
            train_loader, val_loader = create_cmlm_dataloaders(config)
        
        # Create unified trainer
        trainer = MLMTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )
        
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'rdt', 'mlm', or 'cmlm'")
    
    # Resume from checkpoint or load pretrained weights
    if args.checkpoint:
        print(f"\nResuming from checkpoint: {args.checkpoint}")
        from rdt.utils import load_checkpoint
        checkpoint = load_checkpoint(
            args.checkpoint,
            trainer.model,
            trainer.optimizer,
            trainer.scheduler
        )
        trainer.current_epoch = checkpoint.get('epoch', 0) + 1
        trainer.global_step = checkpoint.get('step', 0)
        print(f"Resuming from epoch {trainer.current_epoch}, step {trainer.global_step}")
    
    elif args.pretrained:
        print(f"\nLoading pretrained weights: {args.pretrained}")
        from rdt.utils import load_pretrained_weights
        load_pretrained_weights(
            args.pretrained,
            trainer.model
        )
        print("Starting training from epoch 0, step 0 with pretrained weights")
    
    # Train
    trainer.train()
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print(f"Checkpoints saved to: {config['output']['checkpoint_dir']}")
    print(f"Logs saved to: {config['output']['log_dir']}")
    print("="*60)


if __name__ == '__main__':
    main()