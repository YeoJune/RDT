"""Unified training script for all models (RDT and baselines) using Accelerate"""

import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import os

from rdt.models import RDT, MLM
from rdt.models.cmlm import CMLM
from rdt.data import create_dataloaders, create_mlm_dataloaders, create_cmlm_dataloaders, create_mdlm_dataloaders
from rdt.training import RDTTrainer, MLMTrainer
from rdt.utils import load_config, merge_configs, set_seed, create_model_from_config


def main():
    parser = argparse.ArgumentParser(description='Train RDT or Baseline Models')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--override', type=str, default=None,
                        help='Path to override config file')
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
    config = load_config(args.config)
    
    if args.override:
        override_config = load_config(args.override)
        config = merge_configs(config, override_config)
    
    accelerator = Accelerator(
        log_with="wandb" if config.get('use_wandb', True) else None,
        project_config=ProjectConfiguration(
            project_dir=config['output']['log_dir'],
            logging_dir=config['output']['log_dir']
        ),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        # TPU RNG 동기화 문제 해결: even_batches를 False로 설정
        even_batches=False
    )

    if config.get('use_wandb', True):
        # run_name 설정 (config에 없으면 자동 생성)
        run_name = config.get('run_name', f"{config.get('model_type', 'model')}-run")
        
        # main.py에서 설정한 환경변수(WANDB_PROJECT 등)를 자동으로 가져감
        accelerator.init_trackers(
            project_name=os.environ.get("WANDB_PROJECT", "rdt"), 
            config=config,
            init_kwargs={"wandb": {"name": run_name}}
        )
    
    # Main Process에서만 출력
    if accelerator.is_main_process:
        print(f"Loading config from {args.config}")
        print(f"Accelerator Device: {accelerator.device}, Distributed: {accelerator.num_processes > 1}")
        print(f"Mixed Precision: {accelerator.mixed_precision}")
    
    # Set seed (Accelerate가 모든 프로세스에 시드 동기화)
    set_seed(config['seed'])
    
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
            accelerator=accelerator
        )
        
    elif model_type in ['mlm', 'cmlm', 'mdlm']:
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
        elif model_type == 'mdlm':
            from rdt.models.mdlm import MDLM
            model = MDLM.from_config(config)
        
        print(f"\nModel parameters: {model.count_parameters()/1e6:.1f}M")
        
        # Create dataloaders
        print(f"\nPreparing {model_type.upper()} data...")
        if model_type == 'mlm':
            train_loader, val_loader = create_mlm_dataloaders(config)
        elif model_type == 'mdlm':
            train_loader, val_loader = create_mdlm_dataloaders(config)
        elif model_type == 'cmlm':
            train_loader, val_loader = create_cmlm_dataloaders(config)
        
        # Create unified trainer
        trainer = MLMTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            accelerator=accelerator
        )
        
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'rdt', 'mlm', 'cmlm', or 'mdlm'")
    
    # Resume from checkpoint or load pretrained weights
    if args.checkpoint:
        if accelerator.is_main_process:
            print(f"\nResuming from checkpoint: {args.checkpoint}")
        trainer.resume_checkpoint = args.checkpoint
    
    elif args.pretrained:
        if accelerator.is_main_process:
            print(f"\nLoading pretrained weights: {args.pretrained}")
            from rdt.utils import load_pretrained_weights
            # unwrap model before loading
            unwrapped_model = accelerator.unwrap_model(trainer.model)
            load_pretrained_weights(args.pretrained, unwrapped_model)
            print("Starting training from epoch 0, step 0 with pretrained weights")
    
    # Train
    trainer.train()

    accelerator.end_training()
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print(f"Checkpoints saved to: {config['output']['checkpoint_dir']}")
    print(f"Logs saved to: {config['output']['log_dir']}")
    print("="*60)


if __name__ == '__main__':
    main()