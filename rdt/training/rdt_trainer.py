"""Training logic for RDT"""

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, Optional
import math

from ..models.rdt import RDT
from ..utils import save_checkpoint, cleanup_checkpoints, count_parameters, count_parameters_without_context, CSVLogger
from ..data.rdt_preprocessor import RDTPreprocessor


class RDTTrainer:
    """Trainer for Recursive Denoising Transformer"""
    
    def __init__(
        self,
        model: RDT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device
    ):
        self.model = model.to(device)
        if hasattr(torch, 'compile') and device.type == 'cuda':
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Training config
        self.training_mode = config['training'].get('training_mode', 'epoch')
        self.num_epochs = config['training'].get('num_epochs', 10)
        self.max_training_steps = config['training'].get('max_training_steps', 100000)
        self.max_grad_norm = config['training']['max_grad_norm']
        self.gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
        self.loss_weight_recon = config['training']['loss_weight_recon']
        self.loss_weight_gate = config['training']['loss_weight_gate']
        self.loss_weight_aux = config['training'].get('loss_weight_aux', 0.1)
        self.aux_ratio = config['training'].get('aux_ratio', 0.25)
        self.aux_temp = config['training'].get('aux_temp', 0.5)
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])

        self.preprocessor = RDTPreprocessor(tokenizer, config).to(device)
        
        # Loss functions
        self.recon_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        self.gate_criterion = nn.MSELoss()
        
        # Logging with W&B
        self.use_wandb = config.get('use_wandb', True)
        if self.use_wandb:
            wandb.init(
                project=config.get('wandb_project', 'rdt'),
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
        
        if self.use_amp:
            print("Using Automatic Mixed Precision (AMP)")
        
        # Checkpointing
        self.checkpoint_dir = config['output']['checkpoint_dir']
        if self.training_mode == 'epoch':
            self.save_every_n_epochs = config['training'].get('save_every_n_epochs', 1)
            self.save_every_n_steps = None
        else:  # step mode
            self.save_every_n_steps = config['training'].get('save_every_n_steps', 10000)
            self.save_every_n_epochs = None
        self.keep_last_n_checkpoints = config['training']['keep_last_n_checkpoints']
        
        # State
        self.global_step = 0
        self.current_epoch = 0
        
        # Print model info
        num_params = count_parameters(model)
        num_params_no_context = count_parameters_without_context(model)
        print(f"Model parameters: {num_params:,}")
        print(f"Model parameters (without context): {num_params_no_context:,}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.training_mode == 'epoch':
            total_batches = len(self.train_loader) * self.num_epochs
        else:
            total_batches = self.max_training_steps
        
        # 실제 optimizer update 횟수
        total_steps = total_batches // self.gradient_accumulation_steps
        
        if self.config['training']['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config['training']['learning_rate'],
                total_steps=total_steps,
                pct_start=self.config['training']['warmup_ratio'],
                anneal_strategy='cos',
                final_div_factor=100
            )
        elif self.config['training']['scheduler'] == 'linear':
            warmup_iters = int(total_steps * self.config['training']['warmup_ratio'])
            scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=warmup_iters
            )
        else:
            scheduler = None
        
        return scheduler
    
    def get_sampling_prob(self, epoch: int = 0, step: int = 0) -> float:
        """
        Scheduled Sampling Probability Scheduler
        - Epoch mode: 1.0 -> 0.0 (Linear Decay over epochs)
        - Step mode: 1.0 -> 0.0 (Linear Decay over steps)
        - Early training: Use GT timestep for stability
        - Late training: Use predicted gate for inference alignment
        """
        if self.training_mode == 'epoch':
            if self.num_epochs <= 1:
                return 0.0  # Single epoch -> no scheduling
            # Linear decay: 1.0 -> 0.0
            sampling_prob = max(0.0, 1.0 - (epoch / (self.num_epochs - 1)))
        else:  # step mode
            if self.max_training_steps <= 1:
                return 0.0
            # Linear decay: 1.0 -> 0.0 based on current step
            sampling_prob = max(0.0, 1.0 - (step / self.max_training_steps))
        
        return sampling_prob
    
    def train_step(self, raw_batch: Dict) -> Tuple[float, float, float, float]:
        """
        - Recon Loss: h_i -> Target Index (미래 추론)
        - Aux Loss: D(h_i) -> D(E(GT)) (현재 위상 보존)
        """
        self.model.train()

        raw_input_ids = raw_batch['input_ids'].to(self.device)

        with torch.no_grad():
            batch = self.preprocessor(raw_input_ids)
        
        # 데이터 준비
        input_tokens = batch['input'].to(self.device)
        targets = batch['targets'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        loss_masks = batch['loss_masks'].to(self.device)
        gate_targets = batch['gate_targets'].to(self.device)
        chain_lengths = batch['chain_lengths']
        
        batch_size = input_tokens.shape[0]
        actual_max_length = chain_lengths.max().item()
        sampling_prob = self.get_sampling_prob(self.current_epoch, self.global_step)
        
        # Aux batch 샘플링
        aux_batch_size = max(1, int(batch_size * self.aux_ratio))
        
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # --- Step 0: 초기 상태 ---
            h_0 = self.model.encode_tokens(input_tokens)
            gate_pred_0, pooled_0 = self.model.gate(h_0, attention_mask)
            gate_loss_0 = self.gate_criterion(gate_pred_0, gate_targets[:, 0].unsqueeze(1))
            
            accumulated_loss = self.loss_weight_gate * gate_loss_0
            total_recon_loss = 0.0
            total_gate_loss = gate_loss_0.item()
            total_aux_loss = 0.0
            
            num_valid_steps = 1
            num_aux_steps = 0
            
            hidden = h_0
            gate_pred = gate_pred_0
            pooled = pooled_0
            
            # --- Recursive Steps ---
            for step_idx in range(actual_max_length):
                valid_mask = chain_lengths > step_idx
                if valid_mask.sum() == 0:
                    break
                
                # Forward step
                step_gt_gate = gate_targets[:, step_idx].unsqueeze(1)
                hidden, gate_pred, pooled = self.model.forward_step(
                    hidden,
                    attention_mask=attention_mask,
                    last_gate_score=gate_pred,
                    last_pooled=pooled,
                    gt_timestep=step_gt_gate,
                    sampling_prob=sampling_prob
                )
                
                # 현재 step 타겟과 마스크
                step_targets = targets[:, step_idx, :]
                step_loss_mask = loss_masks[:, step_idx, :]
                
                # [A] Main Reconstruction Loss
                logits_pred = self.model.decode(hidden, attention_mask)
                
                if step_loss_mask.sum() > 0:
                    recon_loss = self.recon_criterion(
                        logits_pred[step_loss_mask],
                        step_targets[step_loss_mask]
                    )
                else:
                    recon_loss = torch.tensor(0.0, device=self.device)
                
                # [B] Aux Loss: KL Divergence
                aux_kl = torch.tensor(0.0, device=self.device)
                
                if self.loss_weight_aux > 0:
                    aux_mask = step_loss_mask[:aux_batch_size]
                    
                    if aux_mask.sum() > 0:
                        # Teacher: D(E(GT))
                        with torch.no_grad():
                            h_GT = self.model.encode_tokens(step_targets[:aux_batch_size])
                            logits_GT = self.model.decode(h_GT, attention_mask[:aux_batch_size])
                            # Target distribution (with temperature)
                            # Note: Temperature > 1 makes distribution softer
                            # Temperature < 1 makes distribution sharper
                            # For semantic consistency, τ=1.0 is reasonable
                            target_dist = torch.softmax(logits_GT / self.aux_temp, dim=-1)
                        
                        # Student: D(h_recursive)
                        log_probs_pred = torch.log_softmax(
                            logits_pred[:aux_batch_size] / self.aux_temp,
                            dim=-1
                        )
                        
                        # KL Divergence: KL(P_target || P_pred)
                        # PyTorch KLDivLoss expects (log_pred, target)
                        kl_criterion = nn.KLDivLoss(reduction='none')
                        kl_loss_all = kl_criterion(log_probs_pred, target_dist)
                        
                        # Apply mask and average
                        # kl_loss_all: [B_aux, L, V]
                        # aux_mask: [B_aux, L]
                        aux_kl = kl_loss_all.sum(dim=-1)[aux_mask].mean()
                        
                        # Temperature scaling correction
                        # When using temperature τ, gradient is scaled by 1/τ²
                        # So we multiply loss by τ² to compensate
                        aux_kl = aux_kl * (self.aux_temp ** 2)
                        
                        total_aux_loss += aux_kl.item()
                        num_aux_steps += 1
                
                # [C] Gate Loss
                step_gate_target = gate_targets[:, step_idx + 1].unsqueeze(1)
                gate_loss = self.gate_criterion(
                    gate_pred[valid_mask],
                    step_gate_target[valid_mask]
                )
                
                # Step loss 누적
                step_loss = (
                    self.loss_weight_recon * recon_loss +
                    self.loss_weight_gate * gate_loss +
                    self.loss_weight_aux * aux_kl
                )
                
                accumulated_loss += step_loss
                
                total_recon_loss += recon_loss.item()
                total_gate_loss += gate_loss.item()
                num_valid_steps += 1
            
            # 최종 손실
            final_loss = accumulated_loss / num_valid_steps
        
        # 역전파
        original_final_loss = final_loss.item()
        final_loss = final_loss / self.gradient_accumulation_steps
        
        if self.use_amp:
            self.scaler.scale(final_loss).backward()
        else:
            final_loss.backward()
        
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
            if self.scheduler:
                self.scheduler.step()
        
        # 평균 계산 (로깅용)
        avg_aux = total_aux_loss / num_aux_steps if num_aux_steps > 0 else 0.0
        
        return (
            original_final_loss,
            total_recon_loss / num_valid_steps,
            total_gate_loss / num_valid_steps,
            avg_aux
        )
    
    def validate(self) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_gate = 0
        num_batches = 0
        
        with torch.no_grad():
            for raw_batch in tqdm(self.val_loader, desc="Validating", leave=False):
                raw_input_ids = raw_batch['input_ids'].to(self.device)
                batch = self.preprocessor(raw_input_ids)
                
                input_tokens = batch['input'].to(self.device)
                targets = batch['targets'].to(self.device)
                loss_masks = batch['loss_masks'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                gate_targets = batch['gate_targets'].to(self.device)
                chain_lengths = batch['chain_lengths']
                
                actual_max_length = chain_lengths.max().item()
                batch_recon = 0
                batch_gate = 0
                num_valid = 0
                
                # Initial encoding
                h_0 = self.model.encode_tokens(input_tokens)
                gate_pred_0, pooled_0 = self.model.gate(h_0, attention_mask, None, None)
                
                # Gate loss for h_0
                gate_target_0 = gate_targets[:, 0].unsqueeze(1)
                gate_loss_0 = self.gate_criterion(gate_pred_0, gate_target_0)
                batch_gate += gate_loss_0.item()
                num_valid += 1
                
                # Iterative steps
                hidden = h_0
                gate_pred = gate_pred_0
                pooled = pooled_0
                
                for step_idx in range(actual_max_length):
                    valid_mask = chain_lengths > step_idx
                    if valid_mask.sum() == 0:
                        break
                    
                    # Forward step
                    hidden, gate_pred, pooled = self.model.forward_step(
                        hidden,
                        attention_mask=attention_mask,
                        last_gate_score=gate_pred,
                        last_pooled=pooled
                    )
                    
                    # Reconstruction loss
                    step_targets = targets[:, step_idx, :]
                    step_loss_mask = loss_masks[:, step_idx, :]
                    
                    if step_loss_mask.sum() > 0:
                        logits = self.model.decode(hidden, attention_mask)
                        sel_logits = logits[step_loss_mask]
                        sel_targ = step_targets[step_loss_mask]
                        recon_loss = self.recon_criterion(sel_logits, sel_targ)
                    else:
                        recon_loss = torch.tensor(0.0, device=self.device)
                    
                    # Gate loss
                    step_gate_targets = gate_targets[:, step_idx + 1].unsqueeze(1)
                    gate_pred_valid = gate_pred[valid_mask]
                    gate_target_valid = step_gate_targets[valid_mask]
                    gate_loss = self.gate_criterion(gate_pred_valid, gate_target_valid) if len(gate_pred_valid) > 0 else torch.tensor(0.0, device=self.device)
                    
                    batch_recon += recon_loss.item()
                    batch_gate += gate_loss.item()
                    num_valid += 1
                
                if num_valid > 0:
                    avg_recon = batch_recon / num_valid
                    avg_gate = batch_gate / num_valid
                    avg_total = self.loss_weight_recon * avg_recon + self.loss_weight_gate * avg_gate
                    
                    total_loss += avg_total
                    total_recon += avg_recon
                    total_gate += avg_gate
                    num_batches += 1
        
        if num_batches == 0:
            return 0, 0, 0
        
        return (
            total_loss / num_batches,
            total_recon / num_batches,
            total_gate / num_batches
        )

    def train(self):
        if self.training_mode == 'epoch':
            self._train_by_epoch()
        else:
            self._train_by_step()
    
    def _train_by_epoch(self):
        print(f"\nStarting epoch-based training for {self.num_epochs} epochs...")
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            sampling_prob = self.get_sampling_prob(epoch=epoch)
            print(f"\nEpoch {epoch + 1}/{self.num_epochs} | Sampling Prob: {sampling_prob:.3f}")
            
            epoch_loss = 0; epoch_recon = 0; epoch_gate = 0; epoch_aux = 0
            progress_bar = tqdm(self.train_loader, desc="Training")
            
            for batch in progress_bar:
                loss, recon, gate, aux = self.train_step(batch)
                epoch_loss += loss; epoch_recon += recon; epoch_gate += gate; epoch_aux += aux
                self.global_step += 1
                
                if self.global_step % self.log_every_n_steps == 0:
                    log_data = {
                        'epoch': epoch,
                        'step': self.global_step,
                        'loss': loss,
                        'recon_loss': recon,
                        'gate_loss': gate,
                        'aux_loss': aux,
                        'lr': self.optimizer.param_groups[0]['lr'],
                        'sampling_prob': sampling_prob
                    }
                    
                    # CSV logging
                    self.csv_logger.log(log_data)
                    
                    # W&B logging
                    if self.use_wandb:
                        wandb.log({'train/' + k if not k in ['epoch', 'step'] else k: v 
                                  for k, v in log_data.items()})
                
                progress_bar.set_postfix({'loss': f'{loss:.4f}', 'recon': f'{recon:.4f}', 'aux': f'{aux:.4f}'})
            
            # Validation
            if (epoch + 1) % self.eval_every_n_epochs == 0:
                val_loss, val_recon, val_gate = self.validate()
                print(f"Val - Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, Gate: {val_gate:.4f}")
                
                val_data = {
                    'epoch': epoch,
                    'step': self.global_step,
                    'val_loss': val_loss,
                    'val_recon': val_recon,
                    'val_gate': val_gate
                }
                self.csv_logger.log(val_data)
                
                if self.use_wandb:
                    wandb.log({
                        'val/loss': val_loss,
                        'val/recon_loss': val_recon,
                        'val/gate_loss': val_gate,
                        'epoch': epoch
                    })
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, self.global_step, val_loss, self.config, self.checkpoint_dir, 'best_model.pt')
            
            # Checkpoint
            if (epoch + 1) % self.save_every_n_epochs == 0:
                save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, self.global_step, epoch_loss/len(self.train_loader), self.config, self.checkpoint_dir)
                cleanup_checkpoints(self.checkpoint_dir, self.keep_last_n_checkpoints)
        
        print("\nTraining completed!")
        self.csv_logger.close()
    
    def _train_by_step(self):
        print(f"\nStarting step-based training for {self.max_training_steps} steps...")
        best_val_loss = float('inf')
        
        step = 0
        epoch = 0
        
        while step < self.max_training_steps:
            self.current_epoch = epoch
            sampling_prob = self.get_sampling_prob(step=step)
            print(f"\nEpoch {epoch + 1} | Step {step}/{self.max_training_steps} | Sampling Prob: {sampling_prob:.3f}")
            
            progress_bar = tqdm(self.train_loader, desc=f"Training (Step {step})")
            
            for batch in progress_bar:
                if step >= self.max_training_steps:
                    break
                
                loss, recon, gate, aux = self.train_step(batch)
                step += 1
                self.global_step = step
                
                if step % self.log_every_n_steps == 0:
                    log_data = {
                        'epoch': epoch,
                        'step': step,
                        'loss': loss,
                        'recon_loss': recon,
                        'gate_loss': gate,
                        'aux_loss': aux,
                        'lr': self.optimizer.param_groups[0]['lr'],
                        'sampling_prob': self.get_sampling_prob(step=step)
                    }
                    
                    self.csv_logger.log(log_data)
                    
                    if self.use_wandb:
                        wandb.log({'train/' + k if not k in ['epoch', 'step'] else k: v 
                                  for k, v in log_data.items()})
                
                # Validation at regular step intervals
                if step % self.eval_every_n_steps == 0:
                    val_loss, val_recon, val_gate = self.validate()
                    print(f"\nStep {step} - Val Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, Gate: {val_gate:.4f}")
                    
                    val_data = {
                        'epoch': epoch,
                        'step': step,
                        'val_loss': val_loss,
                        'val_recon': val_recon,
                        'val_gate': val_gate
                    }
                    self.csv_logger.log(val_data)
                    
                    if self.use_wandb:
                        wandb.log({
                            'val/loss': val_loss,
                            'val/recon_loss': val_recon,
                            'val/gate_loss': val_gate,
                            'step': step
                        })
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, step, val_loss, self.config, self.checkpoint_dir, 'best_model.pt')
                
                # Checkpoint
                if step % self.save_every_n_steps == 0:
                    save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, step, loss, self.config, self.checkpoint_dir)
                    cleanup_checkpoints(self.checkpoint_dir, self.keep_last_n_checkpoints)
                
                progress_bar.set_postfix({'step': step, 'loss': f'{loss:.4f}', 'recon': f'{recon:.4f}', 'aux': f'{aux:.4f}'})
            
            epoch += 1
        
        print("\nTraining completed!")
        self.csv_logger.close()