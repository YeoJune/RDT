"""Unified training script for all models (RDT and baselines)"""

import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer

from rdt.models import RDT, BaselineMLM
from rdt.data import create_dataloaders
from rdt.data.collators import get_collator
from rdt.training import RDTTrainer, BaselineTrainer
from rdt.utils import load_config, merge_configs, set_seed, get_device, create_model_from_config


def create_mlm_dataloaders(config):
    """Create dataloaders for MLM training"""
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    
    data_config = config['data']
    training_config = config['training']
    
    tokenizer = AutoTokenizer.from_pretrained(data_config['tokenizer_name'])
    
    # Load dataset
    dataset_name = data_config['dataset_name']
    if 'wikitext-2' in dataset_name:
        dataset_config = 'wikitext-2-raw-v1'
    else:
        dataset_config = 'wikitext-103-raw-v1'
    
    print(f"Loading {dataset_name} dataset...")
    train_dataset = load_dataset('wikitext', dataset_config, split='train')
    val_dataset = load_dataset('wikitext', dataset_config, split='validation')
    
    # Tokenization function
    def tokenize_function(examples):
        texts = [text for text in examples['text'] if len(text.strip()) > 50]
        if not texts:
            return {'input_ids': [], 'attention_mask': []}
        
        return tokenizer(
            texts,
            truncation=True,
            max_length=data_config['max_seq_length'],
            padding='max_length',
            return_special_tokens_mask=True
        )
    
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train"
    )
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation"
    )
    
    # Filter empty examples
    train_dataset = train_dataset.filter(lambda x: len(x['input_ids']) > 0)
    val_dataset = val_dataset.filter(lambda x: len(x['input_ids']) > 0)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create collator
    collator = get_collator(
        model_type='mlm',
        tokenizer=tokenizer,
        mlm_probability=data_config.get('mlm_probability', 0.15)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        collate_fn=collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        collate_fn=collator
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train RDT or Baseline Models')
    parser.add_argument('--config', type=str, required=True,
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
        
    elif model_type == 'mlm':
        # Baseline MLM model
        print("\n" + "="*60)
        print("Training Baseline MLM Model")
        print("="*60)
        
        # Create model
        model_name = config['model']['name']
        print(f"\nLoading {model_name}...")
        model = BaselineMLM(model_name=model_name)
        print(f"Model loaded: {model.count_parameters()/1e6:.1f}M parameters")
        
        # Create dataloaders
        print("\nPreparing MLM data...")
        train_loader, val_loader = create_mlm_dataloaders(config)
        
        # Create trainer
        trainer = BaselineTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )
        
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'rdt' or 'mlm'")
    
    # Resume from checkpoint if specified
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
    
    # Train
    trainer.train()
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print(f"Checkpoints saved to: {config['output']['checkpoint_dir']}")
    print(f"Logs saved to: {config['output']['log_dir']}")
    print("="*60)


if __name__ == '__main__':
    main()
