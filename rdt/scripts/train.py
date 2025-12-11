"""Training script for RDT"""

import argparse
from pathlib import Path
import torch

from rdt.model import RDT
from rdt.data import create_dataloaders
from rdt.trainer import RDTTrainer
from rdt.utils import load_config, merge_configs, set_seed, get_device, create_model_from_config
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description='Train Recursive Denoising Transformer')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                        help='Path to config file')
    parser.add_argument('--override', type=str, default=None,
                        help='Path to override config file')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
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
    
    # Create dataloaders
    print("\nPreparing data...")
    train_loader, val_loader = create_dataloaders(config)
    
    # Get vocab size from tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    # Create model (with optional BERT initialization)
    print("\nInitializing model...")
    model = create_model_from_config(config, vocab_size)
    
    # Create trainer
    trainer = RDTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.checkpoint:
        print(f"\nResuming from checkpoint: {args.checkpoint}")
        from rdt.utils import load_checkpoint
        checkpoint = load_checkpoint(
            args.checkpoint,
            model,
            trainer.optimizer,
            trainer.scheduler
        )
        trainer.current_epoch = checkpoint['epoch'] + 1
        trainer.global_step = checkpoint['step']
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()