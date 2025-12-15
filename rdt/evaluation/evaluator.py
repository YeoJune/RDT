"""Unified evaluator for all model types"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional
import json
from pathlib import Path
from datetime import datetime


from .metrics import calculate_metrics


class Evaluator:
    """
    Unified evaluator for RDT and baseline models.
    Provides standard evaluation metrics and comparison utilities.
    """
    
    def __init__(self, model, device: torch.device, model_type: str = 'rdt'):
        """
        Args:
            model: Model to evaluate (RDT or BaselineMLM)
            device: Device to run evaluation on
            model_type: 'rdt' or 'mlm'
        """
        self.model = model
        self.device = device
        self.model_type = model_type.lower()
        
        if self.model_type not in ['rdt', 'mlm']:
            raise ValueError(f"Unknown model type: {model_type}. Choose 'rdt' or 'mlm'")
    
    def evaluate_rdt(self, dataloader: DataLoader, max_steps: int = 20,
                     threshold: float = 0.02) -> Dict[str, float]:
        """
        Evaluate RDT model.
        
        Args:
            dataloader: DataLoader with RDT format data
            max_steps: Maximum recursive steps
            threshold: Gate threshold for stopping
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        total_steps_taken = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating RDT", leave=False):
                # Move batch to device
                input_ids = batch['input'].to(self.device)
                targets = batch['targets'].to(self.device)
                loss_masks = batch['loss_masks'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                chain_lengths = batch['chain_lengths']
                
                batch_size = input_ids.size(0)
                
                for b in range(batch_size):
                    chain_len = chain_lengths[b].item()
                    x = input_ids[b:b+1]
                    mask = attention_mask[b:b+1] if attention_mask is not None else None
                    
                    # Recursive inference
                    hidden, gate_pred, pooled = self.model(x, attention_mask=mask, is_first_step=True)
                    step = 1
                    
                    while step < max_steps and gate_pred.mean().item() > threshold:
                        hidden, gate_pred, pooled = self.model(
                            hidden,
                            attention_mask=mask,
                            last_gate_score=gate_pred,
                            last_pooled=pooled,
                            is_first_step=False
                        )
                        step += 1
                    
                    # Final decode
                    logits = self.model.decode(hidden, attention_mask=mask)
                    
                    # Calculate metrics on last target
                    if chain_len > 0:
                        target = targets[b, 0]  # Use first target
                        loss_mask = loss_masks[b, 0]
                        
                        # Apply loss mask
                        masked_logits = logits[0][loss_mask]
                        masked_target = target[loss_mask]
                        
                        if masked_target.numel() > 0:
                            metrics = calculate_metrics(
                                masked_logits.unsqueeze(0),
                                masked_target.unsqueeze(0),
                                ignore_index=-100
                            )
                            total_loss += metrics['loss']
                            total_accuracy += metrics['accuracy']
                    
                    total_steps_taken += step
                    num_batches += 1
        
        if num_batches == 0:
            return {'loss': 0.0, 'accuracy': 0.0, 'perplexity': float('inf'), 'avg_steps': 0}
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_steps = total_steps_taken / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'perplexity': perplexity,
            'avg_steps': avg_steps
        }
    
    def evaluate_mlm(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate standard MLM baseline.
        
        Args:
            dataloader: DataLoader with MLM format data
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        total_top5_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating MLM", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                loss, logits = self.model(input_ids, attention_mask, labels)
                
                # Calculate metrics
                metrics = calculate_metrics(logits, labels, ignore_index=-100)
                
                total_loss += metrics['loss']
                total_accuracy += metrics['accuracy']
                total_top5_accuracy += metrics['top5_accuracy']
                num_batches += 1
        
        if num_batches == 0:
            return {'loss': 0.0, 'accuracy': 0.0, 'perplexity': float('inf')}
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_top5_accuracy = total_top5_accuracy / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'top5_accuracy': avg_top5_accuracy,
            'perplexity': perplexity
        }
    
    def evaluate(self, dataloader: DataLoader, **kwargs) -> Dict[str, float]:
        """
        Unified evaluation interface.
        Automatically selects appropriate evaluation method based on model type.
        
        Args:
            dataloader: DataLoader with appropriate format
            **kwargs: Additional arguments for specific evaluators
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model_type == 'rdt':
            return self.evaluate_rdt(
                dataloader,
                max_steps=kwargs.get('max_steps', 20),
                threshold=kwargs.get('threshold', 0.02)
            )
        else:  # mlm
            return self.evaluate_mlm(dataloader)
    
    def save_results(self, results: Dict, save_path: str, add_metadata: bool = True):
        """
        Save evaluation results to JSON file with metadata.
        
        Args:
            results: Evaluation results dictionary
            save_path: Path to save results
            add_metadata: Whether to add timestamp and metadata
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if add_metadata:
            # Add metadata
            output = {
                'timestamp': datetime.now().isoformat(),
                'model_type': self.model_type,
                'results': results
            }
        else:
            output = results
        
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to: {save_path}")
    
    def print_results(self, results: Dict):
        """Pretty print evaluation results"""
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key:20s}: {value:.4f}")
            else:
                print(f"{key:20s}: {value}")
        print("="*60 + "\n")
