"""Training logic for RDT"""

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, Optional
import math

from ..models.rdt_model import RDT
from ..utils import save_checkpoint, cleanup_checkpoints, count_parameters, CSVLogger


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
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Training config
        self.training_mode = config['training'].get('training_mode', 'epoch')
        self.num_epochs = config['training'].get('num_epochs', 10)
        self.max_training_steps = config['training'].get('max_training_steps', 100000)
        self.max_grad_norm = config['training']['max_grad_norm']
        self.loss_weight_recon = config['training']['loss_weight_recon']
        self.loss_weight_gate = config['training']['loss_weight_gate']
        self.loss_weight_aux = config['training'].get('loss_weight_aux', 0.1)
        self.aux_ratio = config['training'].get('aux_ratio', 0.25)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
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
        print(f"Model parameters: {num_params:,}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        # Calculate total steps based on training mode
        if self.training_mode == 'epoch':
            total_steps = len(self.train_loader) * self.num_epochs
        else:  # step mode
            total_steps = self.max_training_steps
        
        if self.config['training']['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config['training']['learning_rate'],
                total_steps=total_steps,
                pct_start=self.config['training']['warmup_ratio'],
                anneal_strategy='cos',
                eta_min=self.config['training']['learning_rate'] * 1e-2
            )
        elif self.config['training']['scheduler'] == 'linear':
            scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=int(total_steps * self.config['training']['warmup_ratio'])
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
    
    def train_step(self, batch: Dict) -> Tuple[float, float, float]:
        self.model.train()
        
        # 데이터 로드
        input_tokens = batch['input'].to(self.device)
        targets = batch['targets'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        loss_masks = batch['loss_masks'].to(self.device)
        gate_targets = batch['gate_targets'].to(self.device)  # [B, L+1] - s_0~s_L의 step
        chain_lengths = batch['chain_lengths']
        
        batch_size, seq_len = input_tokens.shape
        actual_max_length = chain_lengths.max().item()
        
        accumulated_loss = 0
        total_recon_loss = 0
        total_gate_loss = 0
        num_valid_steps = 0
        
        # Scheduled Sampling Probability
        sampling_prob = self.get_sampling_prob(self.current_epoch, self.global_step)
        
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # 0. Initial embedding and gate prediction for h_0
            # s_0 -> input_encoder -> h_0
            # h_0 -> gate -> gate_pred_0 (should predict s_0's step)
            init_emb = self.model.token_embedding(input_tokens) * math.sqrt(self.model.d_model)
            init_emb = self.model.pos_encoding(init_emb)
            src_key_padding_mask = (attention_mask == 0)
            h_0 = self.model.input_encoder(init_emb, src_key_padding_mask=src_key_padding_mask)
            h_0 = self.model.input_norm(h_0)
            # First gate prediction (no previous prediction, so prev_pooled=None, prev_gate=None)
            gate_pred_0, pooled_0 = self.model.gate(h_0, attention_mask, prev_pooled=None, prev_gate=None)
            
            # Gate Loss for h_0 (predicting s_0's step)
            gate_target_0 = gate_targets[:, 0].unsqueeze(1)  # s_0의 step
            gate_loss_0 = self.gate_criterion(gate_pred_0, gate_target_0)
            accumulated_loss = self.loss_weight_gate * gate_loss_0
            total_gate_loss += gate_loss_0.item()
            num_valid_steps += 1
            
            # Store hidden states for aux loss (if enabled)
            hidden_states = [] if self.loss_weight_aux > 0 else None
            
            # 1. First main encoder forward: h_0 -> h_1
            step_gt_timestep = gate_targets[:, 0].unsqueeze(1)  # s_0의 step (for scheduled sampling)
            hidden, gate_pred, pooled = self.model(
                h_0,  # Pass h_0 directly
                attention_mask=attention_mask,
                last_gate_score=gate_pred_0,
                last_pooled=pooled_0,  # Pass pooled features from h_0
                is_first_step=False,  # Already processed by input_encoder
                gt_timestep=step_gt_timestep,
                sampling_prob=sampling_prob
            )
            
            for step_idx in range(actual_max_length):
                valid_mask = chain_lengths > step_idx
                if valid_mask.sum() == 0: break
                
                # Store hidden for aux loss (gradient flow 유지!)
                if hidden_states is not None:
                    hidden_states.append(hidden)
                
                step_targets = targets[:, step_idx, :]
                step_loss_mask = loss_masks[:, step_idx, :]
                step_gate_targets = gate_targets[:, step_idx + 1].unsqueeze(1)  # L+1\uac1c \uc911 1~L\ubc88\uc9f8
                
                # 2. Reconstruction Loss (Masked Selection)
                if step_loss_mask.sum() > 0:
                    logits = self.model.decode(hidden, attention_mask)
                    selected_logits = logits[step_loss_mask]
                    selected_targets = step_targets[step_loss_mask]
                    recon_loss = self.recon_criterion(selected_logits, selected_targets)
                else:
                    recon_loss = torch.tensor(0.0, device=self.device)
                
                # 3. Gate Loss (h_{step_idx+1}이 s_{step_idx+1}의 step 예측)
                gate_pred_valid = gate_pred[valid_mask]
                gate_target_valid = step_gate_targets[valid_mask]
                gate_loss = self.gate_criterion(gate_pred_valid, gate_target_valid) if len(gate_pred_valid) > 0 else torch.tensor(0.0, device=self.device)
                
                # Step Loss Accumulation
                step_loss = self.loss_weight_recon * recon_loss + self.loss_weight_gate * gate_loss
                accumulated_loss = accumulated_loss + step_loss
                
                total_recon_loss += recon_loss.item()
                total_gate_loss += gate_loss.item()
                num_valid_steps += 1
                
                # Next Step (Recursive)
                if step_idx < actual_max_length - 1:
                    next_gt_timestep = gate_targets[:, step_idx + 1].unsqueeze(1)  # [B, 1]
                    hidden, gate_pred, pooled = self.model.forward(
                        hidden,
                        attention_mask=attention_mask,
                        last_gate_score=gate_pred,
                        last_pooled=pooled,
                        is_first_step=False,
                        gt_timestep=next_gt_timestep,
                        sampling_prob=sampling_prob
                    )
            
            # Final Loss Calculation
            main_loss = accumulated_loss / max(1, num_valid_steps)
            
            # === Auxiliary Loss: Latent Consistency (Sub-sampling) ===
            if self.loss_weight_aux > 0:
                # Compute micro-batch size (at least 1 sample)
                aux_batch_size = max(1, int(batch_size * self.aux_ratio))
                
                aux_loss = 0
                num_aux_steps = 0
                
                src_key_padding_mask = (attention_mask[:aux_batch_size, :] == 0)
                
                # Latent Consistency Loss for each step
                for step_idx in range(len(hidden_states)):
                    # 1. Ground Truth hidden: s_{i+1} -> input_encoder -> h_GT (detached!)
                    step_target = targets[:aux_batch_size, step_idx, :]  # [Aux_B, Seq]
                    
                    target_emb = self.model.token_embedding(step_target) * math.sqrt(self.model.d_model)
                    target_emb = self.model.pos_encoding(target_emb)
                    h_GT = self.model.input_encoder(target_emb, src_key_padding_mask=src_key_padding_mask)
                    h_GT = self.model.input_norm(h_GT)
                    
                    # 2. Predicted hidden: h_{i+1} (from main loop, already computed)
                    h_pred = hidden_states[step_idx][:aux_batch_size]  # [Aux_B, Seq, D]
                    
                    # 3. Latent Consistency Loss (MSE on hidden vectors)
                    latent_loss = nn.functional.mse_loss(h_pred, h_GT)
                    aux_loss += latent_loss
                    num_aux_steps += 1
                
                aux_loss = aux_loss / max(1, num_aux_steps)
                final_loss = main_loss + self.loss_weight_aux * aux_loss
            else:
                aux_loss = torch.tensor(0.0, device=self.device)
                final_loss = main_loss
        
        # [수정] Standard Optimization (No Accumulation)
        self.optimizer.zero_grad()
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        if self.scheduler:
            self.scheduler.step()
        
        return final_loss.item(), total_recon_loss / max(1, num_valid_steps), total_gate_loss / max(1, num_valid_steps), aux_loss.item()
    
    def validate(self) -> Tuple[float, float, float, float]:
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_gate = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                input_tokens = batch['input'].to(self.device)
                targets = batch['targets'].to(self.device)
                loss_masks = batch['loss_masks'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                gate_targets = batch['gate_targets'].to(self.device)  # [B, L+1] - s_0~s_L의 step
                chain_lengths = batch['chain_lengths']
                
                actual_max_length = chain_lengths.max().item()
                batch_recon = 0
                batch_gate = 0
                num_valid = 0
                
                # 0. Initial embedding and gate prediction for h_0
                init_emb = self.model.token_embedding(input_tokens) * math.sqrt(self.model.d_model)
                init_emb = self.model.pos_encoding(init_emb)
                src_key_padding_mask = (attention_mask == 0)
                h_0 = self.model.input_encoder(init_emb, src_key_padding_mask=src_key_padding_mask)
                h_0 = self.model.input_norm(h_0)
                # First gate prediction (no previous prediction, so prev_pooled=None, prev_gate=None)
                gate_pred_0, pooled_0 = self.model.gate(h_0, attention_mask, prev_pooled=None, prev_gate=None)
                
                # Gate Loss for h_0
                gate_target_0 = gate_targets[:, 0].unsqueeze(1)
                gate_loss_0 = self.gate_criterion(gate_pred_0, gate_target_0)
                batch_gate += gate_loss_0.item()
                num_valid += 1
                
                hidden, gate_pred, pooled = self.model(
                    h_0,
                    attention_mask=attention_mask,
                    last_gate_score=gate_pred_0,
                    last_pooled=pooled_0,
                    is_first_step=False
                )
                
                for step_idx in range(actual_max_length):
                    valid_mask = chain_lengths > step_idx
                    if valid_mask.sum() == 0: break
                    
                    step_targets = targets[:, step_idx, :]
                    step_loss_mask = loss_masks[:, step_idx, :]
                    step_gate_targets = gate_targets[:, step_idx + 1].unsqueeze(1)  # L+1\uac1c \uc911 1~L\ubc88\uc9f8
                    
                    if step_loss_mask.sum() > 0:
                        logits = self.model.decode(hidden, attention_mask)
                        sel_logits = logits[step_loss_mask]
                        sel_targ = step_targets[step_loss_mask]
                        recon_loss = self.recon_criterion(sel_logits, sel_targ)
                    else:
                        recon_loss = torch.tensor(0.0, device=self.device)
                    
                    gate_pred_valid = gate_pred[valid_mask]
                    gate_target_valid = step_gate_targets[valid_mask]
                    gate_loss = self.gate_criterion(gate_pred_valid, gate_target_valid) if len(gate_pred_valid) > 0 else torch.tensor(0.0, device=self.device)
                    
                    batch_recon += recon_loss.item()
                    batch_gate += gate_loss.item()
                    num_valid += 1
                    
                    if step_idx < actual_max_length - 1:
                        hidden, gate_pred, pooled = self.model.forward(
                            hidden,
                            attention_mask=attention_mask,
                            last_gate_score=gate_pred,
                            last_pooled=pooled,
                            is_first_step=False
                        )
                
                if num_valid > 0:
                    avg_recon = batch_recon / num_valid
                    avg_gate = batch_gate / num_valid
                    avg_total = self.loss_weight_recon * avg_recon + self.loss_weight_gate * avg_gate
                    
                    total_loss += avg_total
                    total_recon += avg_recon
                    total_gate += avg_gate
                    num_batches += 1
        
        if num_batches == 0: return 0, 0, 0, 0
        return total_loss / num_batches, total_recon / num_batches, total_gate / num_batches

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