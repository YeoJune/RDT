"""Training logic for RDT"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple
import math

from .model import RDT
from .utils import save_checkpoint, cleanup_checkpoints, count_parameters


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
        self.num_epochs = config['training']['num_epochs']
        self.max_grad_norm = config['training']['max_grad_norm']
        self.loss_weight_recon = config['training']['loss_weight_recon']
        self.loss_weight_gate = config['training']['loss_weight_gate']
        self.loss_weight_aux = config['training'].get('loss_weight_aux', 0.1)
        
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
        
        # Logging
        log_dir = Path(config['output']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_every_n_steps = config['output']['log_every_n_steps']

        # AMP
        self.use_amp = config.get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        if self.use_amp:
            print("Using Automatic Mixed Precision (AMP)")
        
        # Checkpointing
        self.checkpoint_dir = config['output']['checkpoint_dir']
        self.save_every_n_epochs = config['training']['save_every_n_epochs']
        self.keep_last_n_checkpoints = config['training']['keep_last_n_checkpoints']
        
        # State
        self.global_step = 0
        self.current_epoch = 0
        
        # Print model info
        num_params = count_parameters(model)
        print(f"Model parameters: {num_params:,}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        # [수정] Accumulation 제거로 Total Steps 계산 단순화
        total_steps = len(self.train_loader) * self.num_epochs
        
        if self.config['training']['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config['training']['learning_rate'],
                total_steps=total_steps,
                pct_start=self.config['training']['warmup_ratio'],
                anneal_strategy='cos'
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
    
    def get_sampling_prob(self, epoch: int) -> float:
        """
        Scheduled Sampling Probability Scheduler
        - Epoch 0 ~ num_epochs: 1.0 -> 0.0 (Linear Decay)
        - Early training: Use GT timestep for stability
        - Late training: Use predicted gate for inference alignment
        """
        if self.num_epochs <= 1:
            return 0.0  # Single epoch -> no scheduling
        
        # Linear decay: 1.0 -> 0.0
        sampling_prob = max(0.0, 1.0 - (epoch / (self.num_epochs - 1)))
        return sampling_prob
    
    def train_step(self, batch: Dict) -> Tuple[float, float, float]:
        self.model.train()
        
        # 데이터 로드
        input_tokens = batch['input'].to(self.device)
        targets = batch['targets'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        loss_masks = batch['loss_masks'].to(self.device)
        gate_targets = batch['gate_targets'].to(self.device)
        chain_lengths = batch['chain_lengths']
        
        batch_size, seq_len = input_tokens.shape
        actual_max_length = chain_lengths.max().item()
        
        accumulated_loss = 0
        total_recon_loss = 0
        total_gate_loss = 0
        num_valid_steps = 0
        
        # Scheduled Sampling Probability
        sampling_prob = self.get_sampling_prob(self.current_epoch)
        
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # 1. First Forward (with Scheduled Sampling)
            step_gt_timestep = gate_targets[:, 0].unsqueeze(1)  # [B, 1]
            hidden, gate_pred = self.model(
                input_tokens,
                attention_mask=attention_mask,
                is_first_step=True,
                gt_timestep=step_gt_timestep,
                sampling_prob=sampling_prob
            )
            
            for step_idx in range(actual_max_length):
                valid_mask = chain_lengths > step_idx
                if valid_mask.sum() == 0: break
                
                step_targets = targets[:, step_idx, :]
                step_loss_mask = loss_masks[:, step_idx, :]
                step_gate_targets = gate_targets[:, step_idx].unsqueeze(1)
                
                # 2. Reconstruction Loss (Masked Selection)
                if step_loss_mask.sum() > 0:
                    # Use decode method
                    logits = self.model.decode(hidden, attention_mask)
                    selected_logits = logits[step_loss_mask]
                    selected_targets = step_targets[step_loss_mask]
                    recon_loss = self.recon_criterion(selected_logits, selected_targets)
                else:
                    recon_loss = torch.tensor(0.0, device=self.device)
                
                # 3. Gate Loss
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
                    hidden, gate_pred = self.model.forward(
                        hidden,
                        attention_mask=attention_mask,
                        last_gate_score=gate_pred,
                        is_first_step=False,
                        gt_timestep=next_gt_timestep,
                        sampling_prob=sampling_prob
                    )
            
            # Final Loss Calculation
            main_loss = accumulated_loss / max(1, num_valid_steps)
            
            # === Auxiliary Loss: I/O Reconstruction ===
            aux_loss = 0
            num_aux_steps = 0
            
            # targets의 각 중간 step을 Input Encoder -> Output Decoder로 재구성
            for step_idx in range(actual_max_length):
                step_target = targets[:, step_idx, :]  # [B, Seq]
                
                # Input Encoder로 인코딩
                target_emb = self.model.token_embedding(step_target) * math.sqrt(self.model.d_model)
                target_emb = self.model.pos_encoding(target_emb)
                src_key_padding_mask = (attention_mask == 0)
                target_hidden = self.model.input_encoder(target_emb, src_key_padding_mask=src_key_padding_mask)
                
                # Output Decoder로 디코딩
                recon_logits = self.model.decode(target_hidden, attention_mask)
                
                # Reconstruction Loss (전체 시퀀스, padding 제외)
                recon_logits_flat = recon_logits.reshape(-1, recon_logits.size(-1))
                target_flat = step_target.reshape(-1)
                step_aux_loss = self.recon_criterion(recon_logits_flat, target_flat)
                
                aux_loss += step_aux_loss
                num_aux_steps += 1
            
            aux_loss = aux_loss / max(1, num_aux_steps)
            
            # Total Loss
            final_loss = main_loss + self.loss_weight_aux * aux_loss
        
        # [수정] Standard Optimization (No Accumulation)
        self.optimizer.zero_grad()
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        if self.scheduler:
            self.scheduler.step()
        
        return final_loss.item(), total_recon_loss / max(1, num_valid_steps), total_gate_loss / max(1, num_valid_steps), aux_loss.item()
    
    def validate(self) -> Tuple[float, float, float]:
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
                gate_targets = batch['gate_targets'].to(self.device)
                chain_lengths = batch['chain_lengths']
                
                actual_max_length = chain_lengths.max().item()
                batch_recon = 0
                batch_gate = 0
                num_valid = 0
                
                hidden, gate_pred = self.model(
                    input_tokens, 
                    attention_mask=attention_mask, 
                    is_first_step=True
                )
                
                for step_idx in range(actual_max_length):
                    valid_mask = chain_lengths > step_idx
                    if valid_mask.sum() == 0: break
                    
                    step_targets = targets[:, step_idx, :]
                    step_loss_mask = loss_masks[:, step_idx, :]
                    step_gate_targets = gate_targets[:, step_idx].unsqueeze(1)
                    
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
                        hidden, gate_pred = self.model.forward(
                            hidden,
                            attention_mask=attention_mask,
                            last_gate_score=gate_pred,
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
        
        if num_batches == 0: return 0, 0, 0
        return total_loss / num_batches, total_recon / num_batches, total_gate / num_batches

    def train(self):
        print(f"\nStarting training for {self.num_epochs} epochs...")
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            sampling_prob = self.get_sampling_prob(epoch)
            print(f"\nEpoch {epoch + 1}/{self.num_epochs} | Sampling Prob: {sampling_prob:.3f}")
            
            epoch_loss = 0; epoch_recon = 0; epoch_gate = 0; epoch_aux = 0
            progress_bar = tqdm(self.train_loader, desc="Training")
            
            for batch in progress_bar:
                loss, recon, gate, aux = self.train_step(batch)
                epoch_loss += loss; epoch_recon += recon; epoch_gate += gate; epoch_aux += aux
                self.global_step += 1
                
                if self.global_step % self.log_every_n_steps == 0:
                    self.writer.add_scalar('train/loss', loss, self.global_step)
                    self.writer.add_scalar('train/recon_loss', recon, self.global_step)
                    self.writer.add_scalar('train/gate_loss', gate, self.global_step)
                    self.writer.add_scalar('train/aux_loss', aux, self.global_step)
                    self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                    self.writer.add_scalar('train/sampling_prob', sampling_prob, self.global_step)
                
                progress_bar.set_postfix({'loss': f'{loss:.4f}', 'recon': f'{recon:.4f}', 'aux': f'{aux:.4f}'})
            
            # Validation
            if (epoch + 1) % self.config['output']['eval_every_n_epochs'] == 0:
                val_loss, val_recon, val_gate = self.validate()
                print(f"Val - Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, Gate: {val_gate:.4f}")
                self.writer.add_scalar('val/loss', val_loss, epoch)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, self.global_step, val_loss, self.config, self.checkpoint_dir, 'best_model.pt')
            
            # Checkpoint
            if (epoch + 1) % self.save_every_n_epochs == 0:
                save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, self.global_step, epoch_loss/len(self.train_loader), self.config, self.checkpoint_dir)
                cleanup_checkpoints(self.checkpoint_dir, self.keep_last_n_checkpoints)
        
        print("\nTraining completed!")
        self.writer.close()