"""Utility functions for RDT"""

import yaml
import torch
import random
import numpy as np
import csv
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import os


class CSVLogger:
    """Simple CSV logger for tracking training metrics"""
    
    def __init__(self, log_dir: str, filename: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'train_{timestamp}.csv'
        
        self.log_path = self.log_dir / filename
        self.writer = None
        self.file = None
        self.fieldnames = set()
        self.rows = []
        
    def log(self, metrics: Dict[str, float]):
        """Log metrics to CSV"""
        # Update fieldnames
        new_fields = set(metrics.keys()) - self.fieldnames
        if new_fields:
            self.fieldnames.update(new_fields)
            self._rewrite_csv()
        
        # Add row
        self.rows.append(metrics)
        
        # Write row
        if self.writer is not None:
            self.writer.writerow(metrics)
            self.file.flush()
    
    def _rewrite_csv(self):
        """Rewrite CSV with updated fieldnames"""
        if self.file is not None:
            self.file.close()
        
        # Sort fieldnames for consistent order
        sorted_fields = sorted(self.fieldnames, key=lambda x: (
            0 if x == 'epoch' else 1 if x == 'step' else 2
        ))
        
        self.file = open(self.log_path, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=sorted_fields, extrasaction='ignore')
        self.writer.writeheader()
        
        # Rewrite existing rows
        for row in self.rows[:-1]:  # Skip last row, will be written in log()
            self.writer.writerow(row)
        
        if len(self.rows) == 1:
            print(f"Logging to: {self.log_path}")
    
    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
            self.writer = None
    
    def __del__(self):
        self.close()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """Merge two configs (override takes precedence)"""
    merged = base_config.copy()
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    step: int,
    loss: float,
    config: Dict,
    checkpoint_dir: str,
    filename: str = None
):
    """Save training checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    
    checkpoint_path = checkpoint_dir / filename
    
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'config': config
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: Any = None
) -> Dict:
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded: {checkpoint_path}")
    return checkpoint


def load_pretrained_weights(
    checkpoint_path: str,
    model: torch.nn.Module
) -> None:
    """Load only model weights from checkpoint without optimizer/scheduler state"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Pretrained weights loaded from: {checkpoint_path}")
    print("Training will start from step 0 (optimizer and scheduler states not loaded)")


def cleanup_checkpoints(checkpoint_dir: str, keep_last_n: int = 3):
    """Keep only the last N checkpoints"""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return
    
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_epoch_*.pt"),
        key=lambda x: x.stat().st_mtime
    )
    
    if len(checkpoints) > keep_last_n:
        for ckpt in checkpoints[:-keep_last_n]:
            ckpt.unlink()
            print(f"Removed old checkpoint: {ckpt}")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_without_context(model: torch.nn.Module) -> int:
    """Count trainable parameters excluding context-related components in DirectionalRecursiveBlock"""
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Exclude cross_attn, norm2, and dropout2 from encoder_layers (only used when context is provided)
            if 'encoder_layers' in name and ('cross_attn' in name or 'norm2' in name or 'dropout2' in name):
                continue
            total += param.numel()
    return total


def get_device(device_name: str = 'cuda') -> torch.device:
    """Get torch device"""
    if device_name == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def create_model_from_config(config: Dict, vocab_size: int):
    """
    Create RDT model from config with optional BERT initialization.
    
    Args:
        config: Configuration dictionary
        vocab_size: Vocabulary size
        
    Returns:
        RDT model instance
    """
    from .models.rdt import RDT
    
    model_config = config['model']
    use_bert_init = model_config.get('use_bert_init', False)
    
    if use_bert_init:
        from .models.bert_init import initialize_rdt_with_bert
        
        bert_model_name = model_config.get('bert_model_name', 'prajjwal1/bert-medium')
        
        # Let bert_init auto-detect d_model if needed
        d_model = model_config.get('d_model', None)
        
        print(f"\n{'='*60}")
        print(f"Initializing RDT with BERT weights")
        print(f"{'='*60}")
        
        model = initialize_rdt_with_bert(
            vocab_size=vocab_size,
            bert_model_name=bert_model_name,
            d_model=d_model,
            n_heads=model_config['n_heads'],
            n_encoder_layers=model_config['n_encoder_layers'],
            d_ff=model_config['d_ff'],
            dropout=model_config['dropout'],
            max_seq_len=config['data']['max_seq_length'],
            input_processor_layers=model_config.get('input_processor_layers', 1),
            input_processor_heads=model_config.get('input_processor_heads', 8),
            input_processor_ff=model_config.get('input_processor_ff', 2048),
            output_processor_layers=model_config.get('output_processor_layers', 1),
            output_processor_heads=model_config.get('output_processor_heads', 8),
            output_processor_ff=model_config.get('output_processor_ff', 2048),
            gate_hidden_dim=model_config['gate_hidden_dim'],
            gate_num_layers=model_config['gate_num_layers'],
            gate_num_heads=model_config['gate_num_heads'],
            gate_dropout=model_config.get('gate_dropout', 0.1),
            rope_base=model_config.get('rope_base', 10000.0),
            gradient_checkpointing=model_config.get('gradient_checkpointing', False),
            verbose=True
        )
    else:
        print(f"\nInitializing RDT with random weights")
        model = RDT(
            vocab_size=vocab_size,
            d_model=model_config['d_model'],
            n_heads=model_config['n_heads'],
            n_encoder_layers=model_config['n_encoder_layers'],
            d_ff=model_config['d_ff'],
            dropout=model_config['dropout'],
            max_seq_len=config['data']['max_seq_length'],
            input_mlp_hidden=model_config.get('input_mlp_hidden', [512]),
            output_mlp_hidden=model_config.get('output_mlp_hidden', [512]),
            gate_hidden_dim=model_config['gate_hidden_dim'],
            gate_num_layers=model_config['gate_num_layers'],
            gate_num_heads=model_config['gate_num_heads'],
            gate_dropout=model_config.get('gate_dropout', 0.1),
            rope_base=model_config.get('rope_base', 10000.0),
            gradient_checkpointing=model_config.get('gradient_checkpointing', False)
        )
    
    return model
