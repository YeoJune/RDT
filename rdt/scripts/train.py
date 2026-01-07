"""Unified training script for all models (RDT and baselines) using Accelerate"""

import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import os
import sys

from rdt.models import RDT, MLM
from rdt.models.cmlm import CMLM
from rdt.data import create_dataloaders, create_mlm_dataloaders, create_cmlm_dataloaders, create_mdlm_dataloaders
from rdt.training import RDTTrainer, MLMTrainer
from rdt.utils import load_config, merge_configs, set_seed, create_model_from_config


def is_kaggle_tpu():
    """Detect if running in Kaggle TPU environment"""
    try:
        import torch_xla.core.xla_model as xm
        # Check if we're in a notebook/Kaggle environment
        in_notebook = 'ipykernel' in sys.modules or 'IPython' in sys.modules
        # Check if TPU is available
        has_tpu = os.environ.get('PJRT_DEVICE') == 'TPU' or os.environ.get('TPU_NAME') is not None
        return in_notebook and has_tpu
    except ImportError:
        return False


def train_worker(index, args_tuple):
    """Worker function for TPU multiprocessing (fork mode)"""
    config_path, override_path, checkpoint_path, pretrained_path = args_tuple
    
    # Import here to avoid issues in main process
    from rdt.models import RDT, MLM
    from rdt.models.cmlm import CMLM
    from rdt.data import create_dataloaders, create_mlm_dataloaders, create_cmlm_dataloaders, create_mdlm_dataloaders
    from rdt.training import RDTTrainer, MLMTrainer
    from rdt.utils import load_config, merge_configs, set_seed, create_model_from_config
    
    # Load config
    config = load_config(config_path)
    
    if override_path:
        override_config = load_config(override_path)
        config = merge_configs(config, override_config)
    
    # Create accelerator
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
        print(f"Worker {index} | Device: {accelerator.device} | Distributed: {accelerator.num_processes > 1}")
    
    set_seed(config['seed'])
    
    # Determine model type
    model_type = config.get('model_type', 'rdt').lower()
    
    if model_type == 'rdt':
        print("\n" + "="*60)
        print("Training RDT Model")
        print("="*60)
        
        train_loader, val_loader = create_dataloaders(config)
        tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])
        vocab_size = tokenizer.vocab_size
        
        if accelerator.is_main_process:
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
            train_loader, val_loader = create_mlm_dataloaders(config)
        elif model_type == 'cmlm':
            model = CMLM.from_config(config)
            train_loader, val_loader = create_cmlm_dataloaders(config)
        elif model_type == 'mdlm':
            from rdt.models.mdlm import MDLM
            model = MDLM.from_config(config)
            train_loader, val_loader = create_mdlm_dataloaders(config)
        
        if accelerator.is_main_process:
            print(f"\nModel parameters: {model.count_parameters()/1e6:.1f}M")
        
        trainer = MLMTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            accelerator=accelerator
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Resume from checkpoint or load pretrained weights
    if checkpoint_path:
        if accelerator.is_main_process:
            print(f"\nResuming from checkpoint: {checkpoint_path}")
        trainer.resume_checkpoint = checkpoint_path
    
    elif pretrained_path:
        if accelerator.is_main_process:
            print(f"\nLoading pretrained weights: {pretrained_path}")
            from rdt.utils import load_pretrained_weights
            unwrapped_model = accelerator.unwrap_model(trainer.model)
            load_pretrained_weights(pretrained_path, unwrapped_model)
    
    # Train
    trainer.train()
    accelerator.end_training()
    
    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("Training completed successfully!")
        print(f"Checkpoints saved to: {config['output']['checkpoint_dir']}")
        print(f"Logs saved to: {config['output']['log_dir']}")
        print("="*60)


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
    
    # Detect environment
    if is_kaggle_tpu():
        print("\n" + "="*60)
        print("Detected Kaggle TPU environment - using xmp.spawn with fork")
        print("="*60)
        
        # Use xmp.spawn for Kaggle TPU (fork mode for notebook compatibility)
        import torch_xla.distributed.xla_multiprocessing as xmp
        
        args_tuple = (args.config, args.override, args.checkpoint, args.pretrained)
        
        xmp.spawn(
            train_worker,
            args=(args_tuple,),
            nprocs=8,
            start_method='fork'
        )
    else:
        print("\n" + "="*60)
        print("Using standard training mode (non-Kaggle or non-TPU)")
        print("="*60)
        
        # Standard training (GPU, CPU, or Cloud TPU with accelerate launch)
        train_worker(0, (args.config, args.override, args.checkpoint, args.pretrained))


if __name__ == '__main__':
    main()