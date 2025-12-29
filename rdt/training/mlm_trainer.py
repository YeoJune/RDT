"""Baseline trainer for standard MLM models (BERT, RoBERTa, etc.)"""

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional
import math
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import CSVLogger


class MLMTrainer:
    """
    Unified trainer for all MLM-based baselines (MLM, CMLM, future baselines).
    Automatically detects model type and applies appropriate training logic.
    """
    
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Detect model type
        self.model_type = self._detect_model_type(model)
        print(f"Detected model type: {self.model_type.upper()}")
        
        # Training config
        self.training_mode = config['training'].get('training_mode', 'epoch')
        self.num_epochs = config['training'].get('num_epochs', 10)
        self.max_training_steps = config['training'].get('max_training_steps', 100000)
        self.max_grad_norm = config['training']['max_grad_norm']
        self.gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Logging with W&B
        self.use_wandb = config.get('use_wandb', True)
        if self.use_wandb:
            wandb.init(
                project=config.get('wandb_project', 'rdt-baselines'),
                name=config.get('wandb_run_name', None),
                config=config,
                resume='allow'
            )
            wandb.watch(model, log='all', log_freq=config['output'].get('log_every_n_steps', 100))
        
        # CSV Logger
        log_dir = Path(config['output'].get('log_dir', 'outputs/logs'))
        self.csv_logger = CSVLogger(str(log_dir))
        
        self.log_every_n_steps = config['output']['log_every_n_steps']
        self.eval_every_n_epochs = config['output'].get('eval_every_n_epochs', 1)
        self.eval_every_n_steps = config['output'].get('eval_every_n_steps', 5000)
        
        # AMP
        self.use_amp = config.get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Checkpointing
        self.checkpoint_dir = Path(config['output']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.training_mode == 'epoch':
            self.save_every_n_epochs = config['training'].get('save_every_n_epochs', 1)
            self.save_every_n_steps = None
        else:
            self.save_every_n_steps = config['training'].get('save_every_n_steps', 10000)
            self.save_every_n_epochs = None
        self.keep_last_n_checkpoints = config['training']['keep_last_n_checkpoints']
        
        # State
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Print info
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")
    
    def _detect_model_type(self, model):
        """Detect model type from class name"""
        class_name = model.__class__.__name__.lower()
        if 'mdlm' in class_name:
            return 'mdlm'
        elif 'cmlm' in class_name:
            return 'cmlm'
        return 'mlm'
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.training_mode == 'epoch':
            total_steps = len(self.train_loader) * self.num_epochs
        else:
            total_steps = self.max_training_steps
        
        warmup_steps = int(total_steps * self.config['training'].get('warmup_ratio', 0.1))
        
        if self.config['training']['scheduler'] == 'cosine':
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config['training']['learning_rate'],
                total_steps=total_steps,
                pct_start=warmup_steps / total_steps,
                anneal_strategy='cos',
                final_div_factor=100
            )
        else:  # linear
            return optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: min(1.0, step / warmup_steps) if step < warmup_steps 
                else max(0.0, (total_steps - step) / (total_steps - warmup_steps))
            )
    
    def train_step(self, batch):
        """Unified training step with model-specific masking"""
        if self.model_type == 'mdlm':
            return self._train_step_mdlm(batch)
        if self.model_type == 'cmlm':
            return self._train_step_cmlm(batch)
        else:
            return self._train_step_mlm(batch)
    
    def _train_step_mlm(self, batch):
        """Standard MLM training (masking already done by collator)"""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass with AMP
        if self.use_amp:
            with autocast('cuda'):
                loss, logits = self.model(input_ids, attention_mask, labels)
        else:
            loss, logits = self.model(input_ids, attention_mask, labels)
        
        # Store original loss for logging
        original_loss = loss.item()
        
        # Backward pass with gradient accumulation
        loss = loss / self.gradient_accumulation_steps  # Scale loss
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Only update weights every gradient_accumulation_steps
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
        
        # Calculate accuracy (on masked tokens only)
        mask = labels != -100
        if mask.sum() > 0:
            pred_tokens = logits.argmax(dim=-1)
            correct = (pred_tokens == labels) & mask
            accuracy = correct.sum().float() / mask.sum().float()
        else:
            accuracy = torch.tensor(0.0)
        
        return {
            'loss': original_loss,
            'accuracy': accuracy.item(),
            'lr': self.scheduler.get_last_lr()[0]
        }
    
    def _train_step_cmlm(self, batch):
        """CMLM training with on-the-fly uniform masking"""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)  # Original tokens
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Apply uniform masking on-the-fly
        masked_input_ids, labels, mask_ratio = self.model.uniform_masking(
            input_ids, attention_mask
        )
        
        # Forward pass with AMP
        if self.use_amp:
            with autocast('cuda'):
                loss, logits = self.model(masked_input_ids, attention_mask, labels)
        else:
            loss, logits = self.model(masked_input_ids, attention_mask, labels)
        
        # Store original loss for logging
        original_loss = loss.item()
        
        # Backward pass with gradient accumulation
        loss = loss / self.gradient_accumulation_steps  # Scale loss
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Only update weights every gradient_accumulation_steps
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
        
        # Calculate accuracy (on masked tokens only)
        mask = labels != -100
        if mask.sum() > 0:
            pred_tokens = logits.argmax(dim=-1)
            correct = (pred_tokens == labels) & mask
            accuracy = correct.sum().float() / mask.sum().float()
        else:
            accuracy = torch.tensor(0.0)
        
        return {
            'loss': original_loss,
            'accuracy': accuracy.item(),
            'mask_ratio': mask_ratio,
            'lr': self.scheduler.get_last_lr()[0]
        }
    
    def _train_step_mdlm(self, batch):
        """MDLM training with continuous-time masking and weighted loss"""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)  # Original tokens
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Apply continuous-time masking on-the-fly
        masked_input_ids, labels, t, loss_weight = self.model.continuous_time_masking(
            input_ids, 
            attention_mask,
            low_discrepancy_sampling=True
        )
        
        # Forward pass with time-weighted loss
        if self.use_amp:
            with autocast('cuda'):
                loss, logits = self.model.forward_with_time_weighting(
                    masked_input_ids, attention_mask, labels, loss_weight
                )
        else:
            loss, logits = self.model.forward_with_time_weighting(
                masked_input_ids, attention_mask, labels, loss_weight
            )
        
        # Store original loss for logging
        original_loss = loss.item()
        
        # Backward pass with gradient accumulation
        loss = loss / self.gradient_accumulation_steps  # Scale loss
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Only update weights every gradient_accumulation_steps
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
        
        # Calculate accuracy (on masked tokens only)
        mask = labels != -100
        if mask.sum() > 0:
            pred_tokens = logits.argmax(dim=-1)
            correct = (pred_tokens == labels) & mask
            accuracy = correct.sum().float() / mask.sum().float()
        else:
            accuracy = torch.tensor(0.0)
        
        # Calculate average time and mask ratio for logging
        avg_t = t.mean().item()
        mask_ratio = (labels != -100).float().sum() / mask.numel()
        
        return {
            'loss': original_loss,
            'accuracy': accuracy.item(),
            'avg_time': avg_t,
            'mask_ratio': mask_ratio.item(),
            'avg_loss_weight': loss_weight.mean().item(),
            'lr': self.scheduler.get_last_lr()[0]
        }
    
    def evaluate(self):
        """Evaluate on validation set with model-specific logic"""
        if self.model_type == 'mdlm':
            return self._evaluate_mdlm()
        elif self.model_type == 'cmlm':
            return self._evaluate_cmlm()
        else:
            return self._evaluate_mlm()
        
    def _evaluate_mlm(self):
        """Standard MLM evaluation"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.use_amp:
                    with autocast():
                        loss, logits = self.model(input_ids, attention_mask, labels)
                else:
                    loss, logits = self.model(input_ids, attention_mask, labels)
                
                # Calculate accuracy
                mask = labels != -100
                if mask.sum() > 0:
                    pred_tokens = logits.argmax(dim=-1)
                    correct = (pred_tokens == labels) & mask
                    accuracy = correct.sum().float() / mask.sum().float()
                else:
                    accuracy = torch.tensor(0.0)
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': avg_accuracy
        }
    
    def _evaluate_cmlm(self):
        """CMLM evaluation with uniform masking"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Apply uniform masking for CMLM
                masked_input_ids, labels, _ = self.model.uniform_masking(
                    input_ids, attention_mask
                )
                
                if self.use_amp:
                    with autocast():
                        loss, logits = self.model(masked_input_ids, attention_mask, labels)
                else:
                    loss, logits = self.model(masked_input_ids, attention_mask, labels)
                
                # Calculate accuracy
                mask = labels != -100
                if mask.sum() > 0:
                    pred_tokens = logits.argmax(dim=-1)
                    correct = (pred_tokens == labels) & mask
                    accuracy = correct.sum().float() / mask.sum().float()
                else:
                    accuracy = torch.tensor(0.0)
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': avg_accuracy
        }
    
    def _evaluate_mdlm(self):
        """
        MDLM evaluation using standard Monte Carlo integration.
        - Loss: Weighted NLL (ELBO) matching the training objective.
        - Accuracy: Unweighted average accuracy over timesteps.
        """
        self.model.eval()
        total_elbo = 0
        total_accuracy = 0  # Accuracy 누적 변수
        total_tokens = 0
        num_batches = 0
        
        # Monte Carlo samples
        num_mc_samples = self.config.get('eval', {}).get('mc_samples', 10)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating MDLM", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Valid token count
                if attention_mask is not None:
                    num_valid_tokens = attention_mask.sum().item()
                else:
                    num_valid_tokens = (input_ids != self.model.pad_token_id).sum().item()

                # MC Integration
                batch_elbo_sum = 0
                batch_acc_sum = 0 # 배치 내 MC 샘플들의 정확도 합
                
                for _ in range(num_mc_samples):
                    masked_ids, labels, t, loss_weight = self.model.continuous_time_masking(
                        input_ids, attention_mask, low_discrepancy_sampling=True
                    )
                    
                    if self.use_amp:
                        with autocast('cuda'):
                            # Loss (Weighted)
                            loss, logits = self.model.forward_with_time_weighting(
                                masked_ids, attention_mask, labels, loss_weight
                            )
                    else:
                        loss, logits = self.model.forward_with_time_weighting(
                            masked_ids, attention_mask, labels, loss_weight
                        )
                    
                    # 1. Loss Accumulation (Weighted Sum)
                    batch_elbo_sum += loss.item() * num_valid_tokens
                    
                    # 2. Accuracy Calculation (Unweighted)
                    mask = labels != -100
                    if mask.sum() > 0:
                        pred_tokens = logits.argmax(dim=-1)
                        correct = (pred_tokens == labels) & mask
                        # 해당 타임스텝 t에서의 정확도
                        acc = correct.sum().float() / mask.sum().float()
                        batch_acc_sum += acc.item()
                    # (만약 mask.sum() == 0이면 acc는 0으로 간주하거나 skip)

                # Average over MC samples
                batch_avg_elbo = batch_elbo_sum / num_mc_samples
                batch_avg_acc = batch_acc_sum / num_mc_samples # 배치의 평균 정확도
                
                total_elbo += batch_avg_elbo
                total_accuracy += batch_avg_acc
                total_tokens += num_valid_tokens
                num_batches += 1
        
        # Final Metrics
        if total_tokens > 0:
            avg_elbo = total_elbo / total_tokens
        else:
            avg_elbo = 0.0
            
        avg_accuracy = total_accuracy / max(1, num_batches)
        
        return {
            'val_loss': avg_elbo,     # Weighted NLL (Train Loss와 비교용)
            'val_accuracy': avg_accuracy, # Average Reconstruction Acc
            'mc_samples': num_mc_samples
        }

    def save_checkpoint(self, filename=None):
        """Save checkpoint"""
        if filename is None:
            if self.training_mode == 'epoch':
                filename = f"checkpoint_epoch_{self.current_epoch}_step_{self.global_step}.pt"
            else:
                filename = f"checkpoint_step_{self.global_step}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        torch.save({
            'epoch': self.current_epoch,
            'step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }, checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        return checkpoint_path
    
    def _cleanup_checkpoints(self):
        """Keep only last N checkpoints"""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda x: x.stat().st_mtime
        )
        
        if len(checkpoints) > self.keep_last_n_checkpoints:
            for ckpt in checkpoints[:-self.keep_last_n_checkpoints]:
                ckpt.unlink()
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print(f"Starting {self.model_type.upper()} Training")
        print("="*60)
        print(f"Training mode: {self.training_mode}")
        if self.training_mode == 'epoch':
            print(f"Epochs: {self.num_epochs}")
        else:
            print(f"Max steps: {self.max_training_steps}")
        print("="*60 + "\n")
        
        if self.training_mode == 'epoch':
            self._train_by_epoch()
        else:
            self._train_by_step()
        
        print("\n" + "="*60)
        print("Training completed!")
        print("="*60)
    
    def _train_by_epoch(self):
        """Train by epochs"""
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Training loop
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
            for batch in pbar:
                metrics = self.train_step(batch)
                self.global_step += 1
                
                # Logging
                if self.global_step % self.log_every_n_steps == 0:
                    log_data = {'epoch': epoch, 'step': self.global_step, **metrics}
                    self.csv_logger.log(log_data)
                    
                    if self.use_wandb:
                        wandb.log({'train/' + k if k not in ['epoch', 'step'] else k: v 
                                  for k, v in log_data.items()})
                    pbar.set_postfix(loss=f"{metrics['loss']:.4f}", 
                                    acc=f"{metrics['accuracy']:.4f}")
            
            # Evaluation
            if (epoch + 1) % self.eval_every_n_epochs == 0:
                val_metrics = self.evaluate()
                val_data = {'epoch': epoch, 'step': self.global_step, **val_metrics}
                self.csv_logger.log(val_data)
                
                if self.use_wandb:
                    wandb.log({
                        'val/loss': val_metrics['val_loss'],
                        'val/accuracy': val_metrics['val_accuracy'],
                        'epoch': epoch
                    })
                
                print(f"\nValidation - Loss: {val_metrics['val_loss']:.4f}, "
                      f"Accuracy: {val_metrics['val_accuracy']:.4f}, ")
                
                # Save best model
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint('best_model.pt')
            
            # Save checkpoint
            if (epoch + 1) % self.save_every_n_epochs == 0:
                self.save_checkpoint()
    
    def _train_by_step(self):
        """Train by steps"""
        pbar = tqdm(total=self.max_training_steps, desc="Training")
        pbar.update(self.global_step)
        
        epoch = 0
        while self.global_step < self.max_training_steps:
            for batch in self.train_loader:
                if self.global_step >= self.max_training_steps:
                    break
                
                metrics = self.train_step(batch)
                self.global_step += 1
                pbar.update(1)
                
                # Logging
                if self.global_step % self.log_every_n_steps == 0:
                    log_data = {'epoch': epoch, 'step': self.global_step, **metrics}
                    self.csv_logger.log(log_data)
                    
                    if self.use_wandb:
                        wandb.log({'train/' + k if k not in ['epoch', 'step'] else k: v 
                                  for k, v in log_data.items()})
                    pbar.set_postfix(loss=f"{metrics['loss']:.4f}",
                                    acc=f"{metrics['accuracy']:.4f}")
                
                # Evaluation
                if self.global_step % self.eval_every_n_steps == 0:
                    val_metrics = self.evaluate()
                    val_data = {'epoch': epoch, 'step': self.global_step, **val_metrics}
                    self.csv_logger.log(val_data)
                    
                    if self.use_wandb:
                        wandb.log({
                            'val/loss': val_metrics['val_loss'],
                            'val/accuracy': val_metrics['val_accuracy'],
                            'step': self.global_step
                        })
                    
                    print(f"\nStep {self.global_step} - Val Loss: {val_metrics['val_loss']:.4f}, "
                          f"Accuracy: {val_metrics['val_accuracy']:.4f}, ")
                    
                    # Save best model
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.save_checkpoint('best_model.pt')
                
                # Save checkpoint
                if self.global_step % self.save_every_n_steps == 0:
                    self.save_checkpoint()
            
            epoch += 1
        
        pbar.close()