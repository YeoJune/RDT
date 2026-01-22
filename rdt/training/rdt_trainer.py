"""Training logic for RDT (1번 인터페이스 + 2번 구현)"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, Optional
import math
import wandb

from ..models.rdt import RDT
from ..utils import save_checkpoint, cleanup_checkpoints, count_parameters, count_parameters_without_context, CSVLogger


class RDTTrainer:
    """
    Trainer for Recursive Denoising Transformer
    
    ⚠️ DEBUG VERSION: 1번 인터페이스 유지 + 2번 내부 구현
    """
    
    def __init__(
        self,
        model: RDT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device
    ):
        self.config = config
        self.device = device
        self.resume_checkpoint = None
        
        # ============================================================================
        # Collate Function 참조 저장 (1번 인터페이스 유지)
        # ============================================================================
        self.train_collate_fn = train_loader.collate_fn if hasattr(train_loader, 'collate_fn') else None
        self.val_collate_fn = val_loader.collate_fn if hasattr(val_loader, 'collate_fn') else None
        
        # Store original dataloaders
        self.original_train_loader = train_loader
        self.original_val_loader = val_loader
        
        # Move model to device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
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
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Weight tying status
        self.weight_tying = config['model'].get('weight_tying', True)
        
        # ============================================================================
        # Weight Tying 처리 (1번 인터페이스 유지)
        # ============================================================================
        if self.weight_tying:
            if hasattr(model, 'tie_weights'):
                model.tie_weights()
            elif hasattr(model, 'output_projection') and hasattr(model, 'token_embedding'):
                model.output_projection.weight = model.token_embedding.weight
        
        # ============================================================================
        # Scheduler Creation
        # ============================================================================
        self.scheduler = self._create_scheduler()
        
        # Mixed Precision (2번 스타일)
        self.use_amp = config.get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        if self.use_amp:
            print("Using Automatic Mixed Precision (AMP)")
        
        # Loss functions
        self.recon_criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        self.gate_criterion = nn.MSELoss()
        
        # Logging with W&B (1번 인터페이스 유지)
        self.use_wandb = config.get('use_wandb', True)
        if self.use_wandb:
            # W&B는 train.py에서 초기화
            pass
        
        # CSV Logger
        log_dir = Path(config['output'].get('log_dir', 'outputs/logs'))
        self.csv_logger = CSVLogger(str(log_dir))
        
        self.log_every_n_steps = config['output']['log_every_n_steps']
        self.use_tqdm = config['output'].get('use_tqdm', True)
        self.eval_every_n_epochs = config['output'].get('eval_every_n_epochs', 1)
        self.eval_every_n_steps = config['output'].get('eval_every_n_steps', 5000)
        
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
        num_params = count_parameters(self.model)
        num_params_no_context = count_parameters_without_context(self.model)
        print(f"Model parameters: {num_params:,}")
        print(f"Model parameters (without context): {num_params_no_context:,}")
        
        # ============================================================================
        # Weight Tying Verification (1번 인터페이스 유지)
        # ============================================================================
        if hasattr(self.model, 'output_projection') and hasattr(self.model, 'token_embedding'):
            is_tied = self.model.output_projection.weight is self.model.token_embedding.weight
            
            if self.weight_tying:
                print(f"\nWeight Tying Status: {'✓ ENABLED' if is_tied else '✗ FAILED'}")
                
                if not is_tied:
                    print("ERROR: Weight tying verification failed!")
                    print("Attempting to fix...")
                    
                    if hasattr(self.model, 'tie_weights'):
                        self.model.tie_weights()
                    else:
                        self.model.output_projection.weight = self.model.token_embedding.weight
                    
                    is_tied_after_fix = self.model.output_projection.weight is self.model.token_embedding.weight
                    if is_tied_after_fix:
                        print("✓ Weight tying fixed successfully!")
                    else:
                        print("✗ CRITICAL: Weight tying could not be fixed!")
                else:
                    emb_addr = id(self.model.token_embedding.weight)
                    proj_addr = id(self.model.output_projection.weight)
                    print(f"  - Token embedding: {emb_addr}")
                    print(f"  - Output projection: {proj_addr}")
                    print(f"  - Same object: {emb_addr == proj_addr}")
            else:
                print(f"\nWeight Tying Status: {'✓ DISABLED (as configured)' if not is_tied else '⚠️ UNEXPECTED - Still tied'}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.training_mode == 'epoch':
            total_batches = len(self.train_loader) * self.num_epochs
        else:
            total_batches = self.max_training_steps
        
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
        Scheduled Sampling Probability Scheduler (1번 인터페이스)
        """
        if self.training_mode == 'epoch':
            if self.num_epochs <= 1:
                return 0.0
            sampling_prob = max(0.0, 1.0 - (epoch / (self.num_epochs - 1)))
        else:  # step mode
            if self.max_training_steps <= 1:
                return 0.0
            sampling_prob = max(0.0, 1.0 - (step / self.max_training_steps))
        
        return sampling_prob

    def train_step(self, batch: Dict) -> Tuple[float, float, float, float]:
        """
        Training step (2번 스타일 구현)
        
        Returns:
            final_loss, avg_recon, avg_gate, avg_aux
        """
        self.model.train()
        
        # Load data (2번 스타일 - .to(device))
        input_tokens = batch['input'].to(self.device)
        targets = batch['targets'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        loss_masks = batch['loss_masks'].to(self.device)
        gate_targets = batch['gate_targets'].to(self.device)
        chain_lengths = batch['chain_lengths']
        
        batch_size, seq_len = input_tokens.shape
        actual_max_length = chain_lengths.max().item()
        
        # Scheduled Sampling
        sampling_prob = self.get_sampling_prob(self.current_epoch, self.global_step)
        
        # ============================================================================
        # 2번 스타일: Step-level averaging
        # ============================================================================
        accumulated_loss = 0
        total_recon_loss = 0
        total_gate_loss = 0
        num_valid_steps = 0
        
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # ================================================================
            # STEP 0: Manual First Forward (2번 스타일)
            # ================================================================
            init_emb = self.model.token_embedding(input_tokens) * math.sqrt(self.model.d_model)
            init_emb = self.model.pos_encoding(init_emb)
            src_key_padding_mask = (attention_mask == 0)
            h_0 = self.model.input_encoder(init_emb, src_key_padding_mask=src_key_padding_mask)
            h_0 = self.model.input_norm(h_0)
            gate_pred_0, _ = self.model.gate(h_0, attention_mask)
            
            # Gate Loss for h_0
            gate_target_0 = gate_targets[:, 0].unsqueeze(1)
            gate_loss_0 = self.gate_criterion(gate_pred_0, gate_target_0)
            accumulated_loss = self.loss_weight_gate * gate_loss_0
            total_gate_loss += gate_loss_0.item()
            num_valid_steps += 1
            
            # Store hidden states for aux loss
            hidden_states = [] if self.loss_weight_aux > 0 else None
            
            # First main encoder forward
            step_gt_timestep = gate_targets[:, 1].unsqueeze(1)
            hidden, gate_pred, _ = self.model(
                h_0,
                attention_mask=attention_mask,
                is_first_step=False,
                gt_timestep=step_gt_timestep,
                sampling_prob=sampling_prob
            )
            
            # ================================================================
            # STEP 1 to max_length: Recursive Steps (2번 스타일)
            # ================================================================
            for step_idx in range(actual_max_length):
                valid_mask = chain_lengths > step_idx
                if valid_mask.sum() == 0:
                    break
                
                # Store hidden for aux loss
                if hidden_states is not None:
                    hidden_states.append(hidden)
                
                step_targets = targets[:, step_idx, :]
                step_loss_mask = loss_masks[:, step_idx, :]
                step_gate_targets = gate_targets[:, step_idx + 1].unsqueeze(1)
                
                # Reconstruction Loss (2번 스타일 - masked selection)
                if step_loss_mask.sum() > 0:
                    logits = self.model.decode(hidden, attention_mask)
                    selected_logits = logits[step_loss_mask]
                    selected_targets = step_targets[step_loss_mask]
                    recon_loss = self.recon_criterion(selected_logits, selected_targets)
                else:
                    recon_loss = torch.tensor(0.0, device=self.device)
                
                # Gate Loss
                gate_pred_valid = gate_pred[valid_mask]
                gate_target_valid = step_gate_targets[valid_mask]
                gate_loss = self.gate_criterion(gate_pred_valid, gate_target_valid) if len(gate_pred_valid) > 0 else torch.tensor(0.0, device=self.device)
                
                # Step Loss Accumulation
                step_loss = self.loss_weight_recon * recon_loss + self.loss_weight_gate * gate_loss
                accumulated_loss = accumulated_loss + step_loss
                
                total_recon_loss += recon_loss.item()
                total_gate_loss += gate_loss.item()
                num_valid_steps += 1
                
                # Next Step
                if step_idx < actual_max_length - 1:
                    next_gt_timestep = gate_targets[:, step_idx + 2].unsqueeze(1)
                    hidden, gate_pred, _ = self.model.forward(
                        hidden,
                        attention_mask=attention_mask,
                        last_gate_score=gate_pred,
                        is_first_step=False,
                        gt_timestep=next_gt_timestep,
                        sampling_prob=sampling_prob
                    )
            
            # ================================================================
            # Final Loss Calculation (2번 스타일)
            # ================================================================
            main_loss = accumulated_loss / max(1, num_valid_steps)
            
            # Auxiliary Loss: MSE on Hidden States (2번 스타일!)
            if self.loss_weight_aux > 0:
                aux_batch_size = max(1, int(batch_size * self.aux_ratio))
                
                aux_loss = 0
                num_aux_steps = 0
                
                src_key_padding_mask = (attention_mask[:aux_batch_size, :] == 0)
                
                for step_idx in range(len(hidden_states)):
                    # Ground Truth hidden
                    step_target = targets[:aux_batch_size, step_idx, :]
                    
                    target_emb = self.model.token_embedding(step_target) * math.sqrt(self.model.d_model)
                    target_emb = self.model.pos_encoding(target_emb)
                    h_GT = self.model.input_encoder(target_emb, src_key_padding_mask=src_key_padding_mask)
                    
                    # Predicted hidden
                    h_pred = hidden_states[step_idx][:aux_batch_size]
                    
                    # MSE loss (2번 스타일!)
                    latent_loss = nn.functional.mse_loss(h_pred, h_GT)
                    aux_loss += latent_loss
                    num_aux_steps += 1
                
                aux_loss = aux_loss / max(1, num_aux_steps)
                final_loss = main_loss + self.loss_weight_aux * aux_loss
            else:
                aux_loss = torch.tensor(0.0, device=self.device)
                final_loss = main_loss
        
        # ================================================================
        # Optimization (2번 스타일 - no gradient accumulation)
        # ================================================================
        self.optimizer.zero_grad()
        
        if self.use_amp:
            self.scaler.scale(final_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        if self.scheduler:
            self.scheduler.step()
        
        return (
            final_loss.item(),
            total_recon_loss / max(1, num_valid_steps),
            total_gate_loss / max(1, num_valid_steps),
            aux_loss.item()
        )

    def validate(self) -> Tuple[float, float, float]:
        """
        Validation (2번 스타일 구현)
        
        Returns:
            avg_total_loss, avg_recon_loss, avg_gate_loss
        """
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_gate = 0
        num_batches = 0
        
        with torch.no_grad():
            val_iter = self.val_loader
            
            if self.use_tqdm:
                val_iter = tqdm(val_iter, desc="Validating", leave=False)
            
            for batch in val_iter:
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
                
                # STEP 0: Manual encoding (2번 스타일)
                init_emb = self.model.token_embedding(input_tokens) * math.sqrt(self.model.d_model)
                init_emb = self.model.pos_encoding(init_emb)
                src_key_padding_mask = (attention_mask == 0)
                h_0 = self.model.input_encoder(init_emb, src_key_padding_mask=src_key_padding_mask)
                h_0 = self.model.input_norm(h_0)
                gate_pred_0, _ = self.model.gate(h_0, attention_mask)
                
                # Gate Loss for h_0
                gate_target_0 = gate_targets[:, 0].unsqueeze(1)
                gate_loss_0 = self.gate_criterion(gate_pred_0, gate_target_0)
                batch_gate += gate_loss_0.item()
                num_valid += 1
                
                hidden, gate_pred, _ = self.model(
                    h_0,
                    attention_mask=attention_mask,
                    is_first_step=False
                )
                
                for step_idx in range(actual_max_length):
                    valid_mask = chain_lengths > step_idx
                    if valid_mask.sum() == 0:
                        break
                    
                    step_targets = targets[:, step_idx, :]
                    step_loss_mask = loss_masks[:, step_idx, :]
                    step_gate_targets = gate_targets[:, step_idx + 1].unsqueeze(1)
                    
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
                        hidden, gate_pred, _ = self.model.forward(
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
        
        if num_batches == 0:
            return 0.0, 0.0, 0.0
        return total_loss / num_batches, total_recon / num_batches, total_gate / num_batches

    def _update_curriculum_progress(self, progress: float):
        """Update curriculum progress (1번 인터페이스)"""
        if self.train_collate_fn is not None and hasattr(self.train_collate_fn, 'current_progress'):
            self.train_collate_fn.current_progress = progress
    
    def get_curriculum_progress(self) -> float:
        """Calculate curriculum progress (1번 인터페이스)"""
        if self.training_mode == 'epoch':
            if self.num_epochs <= 1:
                return 1.0
            return min(1.0, self.current_epoch / (self.num_epochs - 1))
        else:
            if self.max_training_steps <= 1:
                return 1.0
            return min(1.0, self.global_step / self.max_training_steps)
        
    def train(self):
        if self.training_mode == 'epoch':
            self._train_by_epoch()
        else:
            self._train_by_step()
    
    def _train_by_epoch(self):
        print(f"\nStarting epoch-based training for {self.num_epochs} epochs...")
        
        # Resume checkpoint
        if self.resume_checkpoint:
            print(f"Loading checkpoint from {self.resume_checkpoint}")
            from ..utils import load_checkpoint
            checkpoint = load_checkpoint(
                self.resume_checkpoint,
                self.model,
                self.optimizer,
                self.scheduler
            )
            self.current_epoch = checkpoint.get('epoch', 0)
            self.global_step = checkpoint.get('step', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
        else:
            best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            sampling_prob = self.get_sampling_prob(epoch=epoch)
            
            # Update curriculum (1번 인터페이스)
            curriculum_progress = self.get_curriculum_progress()
            self._update_curriculum_progress(curriculum_progress)
            
            log_msg = f"\nEpoch {epoch + 1}/{self.num_epochs} | Sampling Prob: {sampling_prob:.3f}"
            
            # Curriculum info (1번 인터페이스)
            if self.train_collate_fn and hasattr(self.train_collate_fn, 'curriculum_enabled'):
                if self.train_collate_fn.curriculum_enabled:
                    min_step, max_step = self.train_collate_fn.get_start_step_range(curriculum_progress)
                    avg_start_step = (min_step + max_step) / 2
                    avg_mask_ratio = avg_start_step / self.train_collate_fn.total_steps
                    log_msg += f" | Curriculum: [{min_step}, {max_step}] (~{avg_mask_ratio:.1%} mask)"
            
            print(log_msg)
            
            train_iter = self.train_loader
            
            if self.use_tqdm:
                progress_bar = tqdm(train_iter, desc="Training")
                train_iter = progress_bar
            
            for batch in train_iter:
                loss, recon, gate, aux = self.train_step(batch)
                self.global_step += 1
                
                # Logging (1번 인터페이스)
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
                                  for k, v in log_data.items()}, step=self.global_step)
                    
                    # tqdm update
                    if self.use_tqdm:
                        progress_bar.set_postfix({
                            'step': self.global_step,
                            'loss': f'{loss:.4f}',
                            'recon': f'{recon:.4f}'
                        })
                    else:
                        print(f"Epoch {epoch+1}/{self.num_epochs} | Step {self.global_step} | Loss: {loss:.4f} | Recon: {recon:.4f} | Gate: {gate:.4f} | Aux: {aux:.4f} | LR: {log_data['lr']:.6f}")
            
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
                    }, step=self.global_step)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        step=self.global_step,
                        loss=val_loss,
                        config=self.config,
                        checkpoint_dir=self.checkpoint_dir,
                        filename="best_model.pt"
                    )
                    print(f"Best checkpoint saved at epoch {epoch+1}, step {self.global_step}")
            
            # Checkpoint
            if (epoch + 1) % self.save_every_n_epochs == 0:
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    step=self.global_step,
                    loss=0.0,
                    config=self.config,
                    checkpoint_dir=self.checkpoint_dir,
                    filename=f"checkpoint_epoch_{epoch+1}_step_{self.global_step}.pt"
                )
                cleanup_checkpoints(self.checkpoint_dir, self.keep_last_n_checkpoints)
        
        print("\nTraining completed!")
        self.csv_logger.close()
    
    def _train_by_step(self):
        print(f"\nStarting step-based training for {self.max_training_steps} steps...")
        
        # Resume checkpoint
        if self.resume_checkpoint:
            print(f"Loading checkpoint from {self.resume_checkpoint}")
            from ..utils import load_checkpoint
            checkpoint = load_checkpoint(
                self.resume_checkpoint,
                self.model,
                self.optimizer,
                self.scheduler
            )
            epoch = checkpoint.get('epoch', 0)
            step = checkpoint.get('step', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.current_epoch = epoch
            self.global_step = step
            print(f"Resumed from epoch {epoch}, step {step}")
        else:
            best_val_loss = float('inf')
            step = 0
            epoch = 0
        
        while step < self.max_training_steps:
            self.current_epoch = epoch
            sampling_prob = self.get_sampling_prob(step=step)
            
            # Update curriculum (1번 인터페이스)
            curriculum_progress = self.get_curriculum_progress()
            self._update_curriculum_progress(curriculum_progress)
            
            log_msg = f"\nEpoch {epoch + 1} | Step {step}/{self.max_training_steps} | Sampling Prob: {sampling_prob:.3f}"
            
            # Curriculum info (1번 인터페이스)
            if self.train_collate_fn and hasattr(self.train_collate_fn, 'curriculum_enabled'):
                if self.train_collate_fn.curriculum_enabled:
                    min_step, max_step = self.train_collate_fn.get_start_step_range(curriculum_progress)
                    avg_start_step = (min_step + max_step) / 2
                    avg_mask_ratio = avg_start_step / self.train_collate_fn.total_steps
                    log_msg += f" | Curriculum: [{min_step}, {max_step}] (~{avg_mask_ratio:.1%} mask)"
            
            print(log_msg)
            
            train_iter = self.train_loader
            
            if self.use_tqdm:
                progress_bar = tqdm(train_iter, desc=f"Training (Step {step})")
                train_iter = progress_bar
            
            for batch in train_iter:
                if step >= self.max_training_steps:
                    break
                
                loss, recon, gate, aux = self.train_step(batch)
                step += 1
                self.global_step = step
                
                # Logging (1번 인터페이스)
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
                                  for k, v in log_data.items()}, step=step)
                    
                    # tqdm update
                    if self.use_tqdm:
                        progress_bar.set_postfix({
                            'step': step,
                            'loss': f'{loss:.4f}',
                            'recon': f'{recon:.4f}'
                        })
                    else:
                        print(f"Epoch {epoch+1} | Step {step}/{self.max_training_steps} | Loss: {loss:.4f} | Recon: {recon:.4f} | Gate: {gate:.4f} | Aux: {aux:.4f} | LR: {log_data['lr']:.6f}")
                
                # Validation
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
                        }, step=step)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            epoch=epoch,
                            step=step,
                            loss=val_loss,
                            config=self.config,
                            checkpoint_dir=self.checkpoint_dir,
                            filename="best_model.pt"
                        )
                        print(f"Best checkpoint saved at step {step}")
                
                # Checkpoint
                if step % self.save_every_n_steps == 0:
                    save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        step=step,
                        loss=0.0,
                        config=self.config,
                        checkpoint_dir=self.checkpoint_dir,
                        filename=f"checkpoint_epoch_{epoch}_step_{step}.pt"
                    )
                    cleanup_checkpoints(self.checkpoint_dir, self.keep_last_n_checkpoints)
            
            epoch += 1
        
        print("\nTraining completed!")
        self.csv_logger.close()