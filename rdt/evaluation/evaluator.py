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
    
    Note: RDT evaluation tests actual inference capability (no preprocessor).
          Training/validation use preprocessor for chain-based learning.
    """
    
    def __init__(self, model, device: torch.device, model_type: str = 'rdt'):
        """
        Args:
            model: Model to evaluate (RDT, MLM, CMLM, or MDLM)
            device: Device to run evaluation on
            model_type: 'rdt', 'mlm', 'cmlm', or 'mdlm'
        """
        self.model = model
        self.device = device
        self.model_type = model_type.lower()
        
        if self.model_type not in ['rdt', 'mlm', 'cmlm', 'mdlm']:
            raise ValueError(f"Unknown model type: {model_type}. Choose 'rdt', 'mlm', 'cmlm', or 'mdlm'")
    
    def evaluate_rdt(self, dataloader: DataLoader, max_steps: int = 20,
                 threshold: float = 0.02, mask_ratio: float = 0.15) -> Dict[str, float]:
        """
        Evaluate RDT model with actual inference capability (no preprocessor).
        
        Tests the model's ability to denoise masked input without intermediate targets.
        This is different from training/validation which use preprocessor chains.
        
        Args:
            dataloader: DataLoader with raw input_ids (no preprocessing)
            max_steps: Maximum recursive steps
            threshold: Gate threshold for stopping
            mask_ratio: Ratio of tokens to mask for evaluation (default: 0.15)
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        total_steps_taken = 0
        num_samples = 0
        
        # Get tokenizer info from model
        mask_token_id = getattr(self.model, 'mask_token_id', 103)  # Default BERT mask
        pad_token_id = getattr(self.model, 'pad_token_id', 0)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating RDT", leave=False):
                # Get raw input_ids (original tokens without preprocessing)
                raw_input_ids = batch['input_ids'].to(self.device)
                batch_size, seq_len = raw_input_ids.shape
                
                # Create attention mask
                attention_mask = (raw_input_ids != pad_token_id).long()
                
                # Randomly mask tokens for evaluation (BERT-style)
                # This simulates real denoising task without preprocessor chains
                masked_input_ids = raw_input_ids.clone()
                
                # Select tokens to mask (excluding special tokens)
                non_special_mask = attention_mask.bool()
                
                # Random masking
                rand_mask = torch.rand(batch_size, seq_len, device=self.device) < mask_ratio
                mask_positions = rand_mask & non_special_mask
                
                # Apply masking (80% [MASK], 10% random, 10% original)
                mask_token_mask = (torch.rand(batch_size, seq_len, device=self.device) < 0.8) & mask_positions
                random_token_mask = (torch.rand(batch_size, seq_len, device=self.device) < 0.5) & mask_positions & ~mask_token_mask
                
                masked_input_ids[mask_token_mask] = mask_token_id
                if hasattr(self.model, 'vocab_size'):
                    vocab_size = self.model.vocab_size
                    random_tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
                    masked_input_ids[random_token_mask] = random_tokens[random_token_mask]
                
                # Run inference (model denoises without intermediate targets)
                # Note: inference() decodes only ONCE at the end (not at every step)
                # This is the key difference from training where we decode at every step
                output_ids, steps_taken = self.model.inference(
                    masked_input_ids,
                    attention_mask=attention_mask,
                    max_steps=max_steps,
                    threshold=threshold,
                    return_steps=True
                )
                
                # Calculate accuracy on masked positions only
                # Accuracy measures exact token match after inference
                correct_predictions = (output_ids == raw_input_ids) & mask_positions
                total_masked = mask_positions.sum().item()
                
                if total_masked > 0:
                    accuracy = correct_predictions.sum().item() / total_masked
                    total_accuracy += accuracy * batch_size
                    
                    # Calculate loss on masked positions
                    # For loss calculation, we need logits (not just tokens)
                    # Re-encoding and decoding is necessary to get probability distributions
                    # This still respects the "decode once" principle of inference
                    with torch.no_grad():
                        # ✅ FIX: encode_tokens → encode
                        hidden = self.model.encode(output_ids, attention_mask)
                        logits = self.model.decode(hidden, attention_mask)
                    
                    # Loss only on masked tokens
                    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
                    token_losses = loss_fn(logits.view(-1, logits.size(-1)), raw_input_ids.view(-1))
                    token_losses = token_losses.view(batch_size, seq_len)
                    
                    masked_losses = token_losses[mask_positions]
                    if masked_losses.numel() > 0:
                        avg_loss = masked_losses.mean().item()
                        total_loss += avg_loss * batch_size
                
                # Track steps
                if isinstance(steps_taken, torch.Tensor):
                    total_steps_taken += steps_taken.sum().item()
                else:
                    total_steps_taken += steps_taken * batch_size
                
                num_samples += batch_size
        
        if num_samples == 0:
            return {'loss': 0.0, 'accuracy': 0.0, 'perplexity': float('inf'), 'avg_steps': 0}
        
        avg_loss = total_loss / num_samples
        avg_accuracy = total_accuracy / num_samples
        avg_steps = total_steps_taken / num_samples
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
    
    def evaluate_cmlm(self, dataloader: DataLoader, max_iterations: int = 10) -> Dict[str, float]:
        """
        Evaluate CMLM (Conditional Masked Language Model) with iterative refinement.
        
        Args:
            dataloader: DataLoader with CMLM format data (original tokens)
            max_iterations: Maximum number of mask-predict iterations (fixed, as per paper)
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        total_top5_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating CMLM", leave=False):
                input_ids = batch['input_ids'].to(self.device)  # Original tokens
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # CMLM inference with fixed iterations
                # Starts from fully masked and iteratively refines
                batch_size, seq_len = input_ids.shape
                
                # Start with all masks
                masked_input_ids = torch.full_like(input_ids, self.model.mask_token_id)
                
                # Copy special tokens (CLS, SEP, PAD)
                if attention_mask is not None:
                    special_mask = attention_mask == 0
                else:
                    special_mask = input_ids == self.model.pad_token_id
                masked_input_ids[special_mask] = input_ids[special_mask]
                
                # Iterative refinement
                output_ids, _ = self.model.inference(
                    masked_input_ids,
                    attention_mask=attention_mask,
                    max_iterations=max_iterations,
                    return_steps=False
                )
                
                # Get final logits for metric calculation
                logits = self.model(output_ids, attention_mask)
                
                # Create labels (all tokens except special tokens)
                labels = input_ids.clone()
                if attention_mask is not None:
                    labels[attention_mask == 0] = -100
                else:
                    labels[input_ids == self.model.pad_token_id] = -100
                
                # Calculate metrics
                metrics = calculate_metrics(logits, labels, ignore_index=-100)
                
                total_loss += metrics['loss']
                total_accuracy += metrics['accuracy']
                total_top5_accuracy += metrics['top5_accuracy']
                num_batches += 1
        
        if num_batches == 0:
            return {'loss': 0.0, 'accuracy': 0.0, 'perplexity': float('inf'), 'iterations': max_iterations}
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_top5_accuracy = total_top5_accuracy / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'top5_accuracy': avg_top5_accuracy,
            'perplexity': perplexity,
            'iterations': max_iterations
        }
    
    def evaluate_mdlm(self, dataloader: DataLoader, num_steps: int = 1000, 
                    sampler: str = 'ddpm_cache') -> Dict[str, float]:
        """
        Evaluate MDLM (Masked Diffusion Language Model) with iterative denoising.
        
        Args:
            dataloader: DataLoader with MDLM format data (original tokens)
            num_steps: Number of diffusion denoising steps
            sampler: Sampling strategy ('ddpm', 'ddpm_cache', 'analytic')
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        total_top5_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating MDLM", leave=False):
                input_ids = batch['input_ids'].to(self.device)  # Original tokens
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # MDLM inference with diffusion sampling
                batch_size, seq_len = input_ids.shape
                
                # Start with all masks
                masked_input_ids = torch.full_like(input_ids, self.model.mask_token_id)
                
                # Copy special tokens (CLS, SEP, PAD)
                if attention_mask is not None:
                    special_mask = attention_mask == 0
                else:
                    special_mask = input_ids == self.model.pad_token_id
                masked_input_ids[special_mask] = input_ids[special_mask]
                
                # Diffusion-based denoising
                output_ids, _ = self.model.inference(
                    masked_input_ids,
                    attention_mask=attention_mask,
                    num_steps=num_steps,
                    sampler=sampler,
                    return_intermediate=False
                )
                
                # Get final logits for metric calculation
                logits = self.model(output_ids, attention_mask)
                
                # Create labels (all tokens except special tokens)
                labels = input_ids.clone()
                if attention_mask is not None:
                    labels[attention_mask == 0] = -100
                else:
                    labels[input_ids == self.model.pad_token_id] = -100
                
                # Calculate metrics
                metrics = calculate_metrics(logits, labels, ignore_index=-100)
                
                total_loss += metrics['loss']
                total_accuracy += metrics['accuracy']
                total_top5_accuracy += metrics['top5_accuracy']
                num_batches += 1
        
        if num_batches == 0:
            return {
                'loss': 0.0, 
                'accuracy': 0.0, 
                'perplexity': float('inf'), 
                'num_steps': num_steps,
                'sampler': sampler
            }
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_top5_accuracy = total_top5_accuracy / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'top5_accuracy': avg_top5_accuracy,
            'perplexity': perplexity,
            'num_steps': num_steps,
            'sampler': sampler
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
        elif self.model_type == 'cmlm':
            return self.evaluate_cmlm(
                dataloader,
                max_iterations=kwargs.get('max_iterations', 10)
            )
        elif self.model_type == 'mdlm':
            return self.evaluate_mdlm(
                dataloader,
                num_steps=kwargs.get('num_steps', 1000),
                sampler=kwargs.get('sampler', 'ddpm_cache')
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
