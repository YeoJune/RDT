"""Training logic for RDT"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
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
        total_steps = len(self.train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.config['training']['warmup_ratio'])
        
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
                total_iters=warmup_steps
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_step(self, batch: Dict) -> Tuple[float, float, float]:
        """
        Single training step with memory-aligned slicing
        
        Args:
            batch: dictionary with 'input', 'targets', 'pos_ids', 'n_loss', 'gate_targets', 'chain_lengths'
        
        Returns:
            total_loss, recon_loss, gate_loss
        """
        self.model.train()
        
        # Move to device
        input_tokens = batch['input'].to(self.device)  # (B, seq_len)
        targets = batch['targets'].to(self.device)  # (B, L, seq_len)
        pos_ids = batch['pos_ids'].to(self.device)  # (B, L, seq_len)
        n_loss = batch['n_loss']  # (B, L) - CPU is fine
        gate_targets = batch['gate_targets'].to(self.device)  # (B, L)
        chain_lengths = batch['chain_lengths']  # (B,)
        
        batch_size = input_tokens.size(0)
        actual_max_length = chain_lengths.max().item()
        
        # Accumulate loss step by step (memory efficient)
        total_recon_loss = 0.0
        total_gate_loss = 0.0
        num_valid_steps = 0
        
        accumulated_loss = 0
        
        # First forward pass (with pos_ids for reordered input)
        hidden, gate_pred = self.model(
            input_tokens,
            pos_ids=pos_ids[:, 0, :],  # First step pos_ids
            is_first_step=True
        )
        
        for step_idx in range(actual_max_length):
            # Check which samples need this step
            valid_mask = chain_lengths > step_idx
            if valid_mask.sum() == 0:
                break
            
            # Get target and n_loss for this step
            step_targets = targets[:, step_idx, :]  # (B, seq_len)
            step_n_loss = n_loss[:, step_idx]  # (B,)
            step_gate_targets = gate_targets[:, step_idx].unsqueeze(1)  # (B, 1)
            
            # Get max n_loss in this step
            max_n_loss = step_n_loss.max().item()
            
            # Reconstruction loss (slice-based, memory-aligned!)
            if max_n_loss > 0:
                # For Transformer Decoder: need full sequence for attention
                if hasattr(self.model.decoder, 'decoder'):
                    # Transformer Decoder: run full attention
                    decoder_features = self.model.decoder.decoder(hidden)  # (B, seq_len, d_model)
                    
                    # Slice delta region (memory-aligned!)
                    delta_features = decoder_features[:, :max_n_loss, :]  # (B, n_delta, d_model)
                    delta_targets = step_targets[:, :max_n_loss]  # (B, n_delta)
                    
                    # Flatten and filter valid samples
                    delta_features_flat = delta_features.reshape(-1, delta_features.size(-1))  # (B*n_delta, d_model)
                    delta_targets_flat = delta_targets.reshape(-1)  # (B*n_delta,)
                    
                    # Create mask for valid samples (where n_loss > position)
                    valid_positions = torch.arange(max_n_loss, device=self.device).unsqueeze(0) < step_n_loss.unsqueeze(1).to(self.device)  # (B, n_delta)
                    valid_positions_flat = valid_positions.reshape(-1)  # (B*n_delta,)
                    
                    if valid_positions_flat.sum() > 0:
                        delta_features_valid = delta_features_flat[valid_positions_flat]
                        delta_targets_valid = delta_targets_flat[valid_positions_flat]
                        
                        # Project only valid positions
                        logits_valid = self.model.decoder.projection(delta_features_valid)
                        recon_loss = self.recon_criterion(logits_valid, delta_targets_valid)
                    else:
                        recon_loss = torch.tensor(0.0, device=self.device)
                else:
                    # Linear Decoder: can directly slice and project
                    delta_hidden = hidden[:, :max_n_loss, :]  # (B, n_delta, d_model)
                    delta_targets = step_targets[:, :max_n_loss]  # (B, n_delta)
                    
                    delta_hidden_flat = delta_hidden.reshape(-1, delta_hidden.size(-1))
                    delta_targets_flat = delta_targets.reshape(-1)
                    
                    valid_positions = torch.arange(max_n_loss, device=self.device).unsqueeze(0) < step_n_loss.unsqueeze(1).to(self.device)
                    valid_positions_flat = valid_positions.reshape(-1)
                    
                    if valid_positions_flat.sum() > 0:
                        delta_hidden_valid = delta_hidden_flat[valid_positions_flat]
                        delta_targets_valid = delta_targets_flat[valid_positions_flat]
                        
                        logits_valid = self.model.decoder(delta_hidden_valid)
                        recon_loss = self.recon_criterion(logits_valid, delta_targets_valid)
                    else:
                        recon_loss = torch.tensor(0.0, device=self.device)
            else:
                recon_loss = torch.tensor(0.0, device=self.device)
            
            # Gate loss
            gate_pred_valid = gate_pred[valid_mask]
            gate_target_valid = step_gate_targets[valid_mask]
            
            if len(gate_pred_valid) > 0:
                gate_loss = self.gate_criterion(gate_pred_valid, gate_target_valid)
            else:
                gate_loss = torch.tensor(0.0, device=self.device)
            
            # Step loss
            step_loss = (
                self.loss_weight_recon * recon_loss +
                self.loss_weight_gate * gate_loss
            )
            accumulated_loss = accumulated_loss + step_loss
            
            # Accumulate for logging (detached)
            total_recon_loss += recon_loss.item()
            total_gate_loss += gate_loss.item()
            num_valid_steps += 1
            
            # Next step (reuse hidden state)
            if step_idx < actual_max_length - 1:
                hidden = self.model.encoder(hidden)
                gate_pred = self.model.gate(hidden)
        
        # Average loss
        if num_valid_steps > 0:
            final_loss = accumulated_loss / num_valid_steps
            avg_recon_loss = total_recon_loss / num_valid_steps
            avg_gate_loss = total_gate_loss / num_valid_steps
        else:
            final_loss = accumulated_loss
            avg_recon_loss = 0.0
            avg_gate_loss = 0.0
        
        # Backward
        self.optimizer.zero_grad()
        if isinstance(final_loss, torch.Tensor):
            final_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # Update
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        
        return (
            final_loss.item() if isinstance(final_loss, torch.Tensor) else 0.0,
            avg_recon_loss,
            avg_gate_loss
        )
    
    def validate(self) -> Tuple[float, float, float]:
        """Validation loop with memory-aligned slicing"""
        self.model.eval()
        
        total_loss = 0
        total_recon_loss = 0
        total_gate_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                input_tokens = batch['input'].to(self.device)
                targets = batch['targets'].to(self.device)
                pos_ids = batch['pos_ids'].to(self.device)
                n_loss = batch['n_loss']
                gate_targets = batch['gate_targets'].to(self.device)
                chain_lengths = batch['chain_lengths']
                
                actual_max_length = chain_lengths.max().item()
                
                # Forward (step by step for memory efficiency)
                batch_recon_loss = 0
                batch_gate_loss = 0
                num_valid_steps = 0
                
                # First forward (with pos_ids)
                hidden, gate_pred = self.model(
                    input_tokens,
                    pos_ids=pos_ids[:, 0, :],
                    is_first_step=True
                )
                
                for step_idx in range(actual_max_length):
                    valid_mask = chain_lengths > step_idx
                    if valid_mask.sum() == 0:
                        break
                    
                    step_targets = targets[:, step_idx, :]
                    step_n_loss = n_loss[:, step_idx]
                    step_gate_targets = gate_targets[:, step_idx].unsqueeze(1)
                    
                    max_n_loss = step_n_loss.max().item()
                    
                    # Reconstruction loss (sliced)
                    if max_n_loss > 0:
                        if hasattr(self.model.decoder, 'decoder'):
                            decoder_features = self.model.decoder.decoder(hidden)
                            delta_features = decoder_features[:, :max_n_loss, :]
                            delta_targets = step_targets[:, :max_n_loss]
                            
                            delta_features_flat = delta_features.reshape(-1, delta_features.size(-1))
                            delta_targets_flat = delta_targets.reshape(-1)
                            
                            valid_positions = torch.arange(max_n_loss, device=self.device).unsqueeze(0) < step_n_loss.unsqueeze(1).to(self.device)
                            valid_positions_flat = valid_positions.reshape(-1)
                            
                            if valid_positions_flat.sum() > 0:
                                delta_features_valid = delta_features_flat[valid_positions_flat]
                                delta_targets_valid = delta_targets_flat[valid_positions_flat]
                                
                                logits_valid = self.model.decoder.projection(delta_features_valid)
                                recon_loss = self.recon_criterion(logits_valid, delta_targets_valid)
                            else:
                                recon_loss = torch.tensor(0.0, device=self.device)
                        else:
                            delta_hidden = hidden[:, :max_n_loss, :]
                            delta_targets = step_targets[:, :max_n_loss]
                            
                            delta_hidden_flat = delta_hidden.reshape(-1, delta_hidden.size(-1))
                            delta_targets_flat = delta_targets.reshape(-1)
                            
                            valid_positions = torch.arange(max_n_loss, device=self.device).unsqueeze(0) < step_n_loss.unsqueeze(1).to(self.device)
                            valid_positions_flat = valid_positions.reshape(-1)
                            
                            if valid_positions_flat.sum() > 0:
                                delta_hidden_valid = delta_hidden_flat[valid_positions_flat]
                                delta_targets_valid = delta_targets_flat[valid_positions_flat]
                                
                                logits_valid = self.model.decoder(delta_hidden_valid)
                                recon_loss = self.recon_criterion(logits_valid, delta_targets_valid)
                            else:
                                recon_loss = torch.tensor(0.0, device=self.device)
                    else:
                        recon_loss = torch.tensor(0.0, device=self.device)
                    
                    # Gate loss
                    gate_pred_valid = gate_pred[valid_mask]
                    gate_target_valid = step_gate_targets[valid_mask]
                    
                    if len(gate_pred_valid) > 0:
                        gate_loss = self.gate_criterion(gate_pred_valid, gate_target_valid)
                    else:
                        gate_loss = torch.tensor(0.0, device=self.device)
                    
                    batch_recon_loss += recon_loss.item()
                    batch_gate_loss += gate_loss.item()
                    num_valid_steps += 1
                    
                    # Next step
                    if step_idx < actual_max_length - 1:
                        hidden = self.model.encoder(hidden)
                        gate_pred = self.model.gate(hidden)
                
                if num_valid_steps > 0:
                    avg_recon = batch_recon_loss / num_valid_steps
                    avg_gate = batch_gate_loss / num_valid_steps
                    avg_total = (
                        self.loss_weight_recon * avg_recon +
                        self.loss_weight_gate * avg_gate
                    )
                    
                    total_loss += avg_total
                    total_recon_loss += avg_recon
                    total_gate_loss += avg_gate
                    num_batches += 1
        
        if num_batches == 0:
            return 0, 0, 0
        
        return (
            total_loss / num_batches,
            total_recon_loss / num_batches,
            total_gate_loss / num_batches
        )
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.num_epochs} epochs...")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Training
            epoch_loss = 0
            epoch_recon_loss = 0
            epoch_gate_loss = 0
            
            progress_bar = tqdm(self.train_loader, desc="Training")
            for batch_idx, batch in enumerate(progress_bar):
                total_loss, recon_loss, gate_loss = self.train_step(batch)
                
                epoch_loss += total_loss
                epoch_recon_loss += recon_loss
                epoch_gate_loss += gate_loss
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.log_every_n_steps == 0:
                    self.writer.add_scalar('train/total_loss', total_loss, self.global_step)
                    self.writer.add_scalar('train/recon_loss', recon_loss, self.global_step)
                    self.writer.add_scalar('train/gate_loss', gate_loss, self.global_step)
                    self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{total_loss:.4f}',
                    'recon': f'{recon_loss:.4f}',
                    'gate': f'{gate_loss:.4f}'
                })
            
            # Epoch summary
            avg_loss = epoch_loss / len(self.train_loader)
            avg_recon = epoch_recon_loss / len(self.train_loader)
            avg_gate = epoch_gate_loss / len(self.train_loader)
            
            print(f"Train - Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, Gate: {avg_gate:.4f}")
            
            # Validation
            if (epoch + 1) % self.config['output']['eval_every_n_epochs'] == 0:
                val_loss, val_recon, val_gate = self.validate()
                print(f"Val   - Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, Gate: {val_gate:.4f}")
                
                self.writer.add_scalar('val/total_loss', val_loss, epoch)
                self.writer.add_scalar('val/recon_loss', val_recon, epoch)
                self.writer.add_scalar('val/gate_loss', val_gate, epoch)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch,
                        self.global_step,
                        val_loss,
                        self.config,
                        self.checkpoint_dir,
                        filename='best_model.pt'
                    )
            
            # Regular checkpointing
            if (epoch + 1) % self.save_every_n_epochs == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    self.global_step,
                    avg_loss,
                    self.config,
                    self.checkpoint_dir
                )
                cleanup_checkpoints(self.checkpoint_dir, self.keep_last_n_checkpoints)
        
        print("\nTraining completed!")
        self.writer.close()