"""Unified training script for all models (RDT and baselines) using Accelerate"""

import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer
from accelerate.utils import ProjectConfiguration
import os
import sys

# ===== 최우선: Accelerate import 전에 TPU 단일 디바이스 강제 =====
os.environ["TPU_NUM_DEVICES"] = "1"
os.environ["XLA_USE_BF16"] = "0"

# XLA 디바이스 강제 초기화
import torch_xla
import torch_xla.core.xla_model as xm

# 단일 디바이스만 사용하도록 강제
def get_single_xla_device():
    """Force single XLA device usage"""
    return xm.xla_device()

# Accelerate import (이제 단일 디바이스만 인식할 것)
from accelerate import Accelerator

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
                        help='Path to checkpoint to resume from')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained weights to load')
    
    args = parser.parse_args()
    
    if args.checkpoint and args.pretrained:
        raise ValueError("Cannot use both --checkpoint and --pretrained flags simultaneously.")
    
    # Load config
    config = load_config(args.config)
    
    if args.override:
        override_config = load_config(args.override)
        config = merge_configs(config, override_config)
    
    # 단일 디바이스 강제
    device = get_single_xla_device()
    print(f"\n[FORCED] Using single XLA device: {device}")
    
    # Accelerator 초기화
    accelerator = Accelerator(
        log_with="wandb" if config.get('use_wandb', True) else None,
        project_config=ProjectConfiguration(
            project_dir=config['output']['log_dir'],
            logging_dir=config['output']['log_dir']
        ),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1)
    )

    if config.get('use_wandb', True):
        run_name = config.get('run_name', f"{config.get('model_type', 'model')}-run")
        accelerator.init_trackers(
            project_name=os.environ.get("WANDB_PROJECT", "rdt"), 
            config=config,
            init_kwargs={"wandb": {"name": run_name}}
        )
    
    if accelerator.is_main_process:
        print(f"Loading config from {args.config}")
        print(f"Accelerator Device: {accelerator.device}")
        print(f"Mixed Precision: {accelerator.mixed_precision}")
        print(f"Num Processes: {accelerator.num_processes}")
        
        # 경고: 8개 프로세스 감지되면
        if accelerator.num_processes > 1:
            print(f"\n⚠️  WARNING: Accelerator detected {accelerator.num_processes} processes")
            print(f"⚠️  This will likely HANG. Forcing single device usage...")
    
    set_seed(config['seed'])
    
    model_type = config.get('model_type', 'rdt').lower()
    
    if model_type == 'rdt':
        print("\n" + "="*60)
        print("Training RDT Model")
        print("="*60)
        
        train_loader, val_loader = create_dataloaders(config)
        tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])
        vocab_size = tokenizer.vocab_size
        print(f"Vocabulary size: {vocab_size}")
        
        model = create_model_from_config(config, vocab_size)
        
        trainer = RDTTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            accelerator=accelerator
        )
        
    elif model_type in ['mlm', 'cmlm', 'mdlm']:
        print("\n" + "="*60)
        print(f"Training {model_type.upper()} Model")
        print("="*60)
        
        if model_type == 'mlm':
            model = MLM.from_config(config)
        elif model_type == 'cmlm':
            model = CMLM.from_config(config)
        elif model_type == 'mdlm':
            from rdt.models.mdlm import MDLM
            model = MDLM.from_config(config)
        
        print(f"\nModel parameters: {model.count_parameters()/1e6:.1f}M")
        
        if model_type == 'mlm':
            train_loader, val_loader = create_mlm_dataloaders(config)
        elif model_type == 'mdlm':
            train_loader, val_loader = create_mdlm_dataloaders(config)
        elif model_type == 'cmlm':
            train_loader, val_loader = create_cmlm_dataloaders(config)
        
        trainer = MLMTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            accelerator=accelerator
        )
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if args.checkpoint:
        if accelerator.is_main_process:
            print(f"\nResuming from checkpoint: {args.checkpoint}")
        trainer.resume_checkpoint = args.checkpoint
    
    elif args.pretrained:
        if accelerator.is_main_process:
            print(f"\nLoading pretrained weights: {args.pretrained}")
            from rdt.utils import load_pretrained_weights
            unwrapped_model = accelerator.unwrap_model(trainer.model)
            load_pretrained_weights(args.pretrained, unwrapped_model)
    
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