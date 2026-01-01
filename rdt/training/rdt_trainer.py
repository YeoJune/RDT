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
            self.model = torch.compile(self.model, mode='reduce-overhead')
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
        self.model.train()
        raw_input_ids = raw_batch['input_ids'].to(self.device)

        # 1. Preprocessing
        with torch.no_grad():
            batch = self.preprocessor(raw_input_ids)
        
        input_tokens = batch['input'].to(self.device)
        targets = batch['targets'].to(self.device)                 # [B, MaxChain, L]
        attention_mask = batch['attention_mask'].to(self.device)   # [B, L]
        loss_masks = batch['loss_masks'].to(self.device)           # [B, MaxChain, L]
        gate_targets = batch['gate_targets'].to(self.device)
        chain_lengths = batch['chain_lengths']
        
        batch_size = input_tokens.shape[0]
        actual_max_length = chain_lengths.max().item()
        sampling_prob = self.get_sampling_prob(self.current_epoch, self.global_step)
        
        # Aux batch size
        aux_batch_size = max(1, int(batch_size * self.aux_ratio))

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # =================================================================
            # [Phase 1] Teacher (Aux) Pre-computation (Batch Processing)
            # 루프 밖에서 미리 계산하여 GPU 병렬성 극대화
            # =================================================================
            teacher_dist_all = None
            if self.loss_weight_aux > 0:
                with torch.no_grad():
                    # Aux용 Target만 추출: [B_aux, MaxChain, L]
                    aux_targets = targets[:aux_batch_size, :actual_max_length, :]
                    
                    # Flatten: [B_aux * Steps, L]
                    flat_aux_targets = aux_targets.reshape(-1, aux_targets.size(-1))
                    
                    # Attention Mask 확장 (Broadcasting)
                    # [B_aux, L] -> [B_aux, Steps, L] -> [B_aux * Steps, L]
                    aux_att_mask = attention_mask[:aux_batch_size].unsqueeze(1)\
                        .expand(-1, actual_max_length, -1).reshape(-1, attention_mask.size(-1))
                    
                    # 한 번에 인코딩 + 디코딩 (Teacher Forcing)
                    h_GT = self.model.encode_tokens(flat_aux_targets, attention_mask=aux_att_mask)
                    logits_GT = self.model.decode(h_GT, attention_mask=aux_att_mask)
                    
                    # Softmax & Reshape back to [B_aux, Steps, L, V]
                    flat_teacher_dist = torch.softmax(logits_GT / self.aux_temp, dim=-1)
                    teacher_dist_all = flat_teacher_dist.view(aux_batch_size, actual_max_length, -1, flat_teacher_dist.size(-1))

            # =================================================================
            # [Phase 2] Recursive Loop (Light-weight)
            # 디코딩 없이 Hidden State 전이만 수행
            # =================================================================
            # Step 0
            h_0 = self.model.encode_tokens(input_tokens, attention_mask)
            gate_pred_0, pooled_0 = self.model.gate(h_0, attention_mask)
            
            # Gate Loss 0
            gate_loss_0 = self.gate_criterion(gate_pred_0, gate_targets[:, 0].unsqueeze(1))
            total_gate_loss = gate_loss_0
            
            # State Collectors
            collected_hidden = []      # h_1 ... h_L 저장
            collected_gate_preds = []  # g_1 ... g_L 저장
            
            hidden = h_0
            gate_pred = gate_pred_0
            pooled = pooled_0
            
            # Loop runs fast because heavy decoding is removed
            for step_idx in range(actual_max_length):
                valid_mask = chain_lengths > step_idx
                if valid_mask.sum() == 0:
                    break
                
                step_gt_gate = gate_targets[:, step_idx].unsqueeze(1)
                
                # Forward Step (Only recursive logic)
                hidden, gate_pred, pooled = self.model.forward_step(
                    hidden,
                    attention_mask=attention_mask,
                    last_gate_score=gate_pred,
                    last_pooled=pooled,
                    gt_timestep=step_gt_gate,
                    sampling_prob=sampling_prob
                )
                
                collected_hidden.append(hidden)
                collected_gate_preds.append(gate_pred)

            # =================================================================
            # [Phase 3] Batch Decoding (Student)
            # 모아둔 Hidden State를 한 번에 디코딩
            # =================================================================
            # Stack Hiddens: [Steps, B, L, D] -> [B, Steps, L, D]
            # (Stacking이 메모리를 쓰지만 BPTT를 위해 어차피 필요함)
            all_hidden_states = torch.stack(collected_hidden, dim=1) 
            
            # Flatten for Batch Decode: [B * Steps, L, D]
            flat_hidden = all_hidden_states.reshape(-1, *all_hidden_states.shape[2:])
            
            # Mask 확장 (For Student Decode)
            flat_att_mask = attention_mask.unsqueeze(1)\
                .expand(-1, actual_max_length, -1).reshape(-1, attention_mask.size(-1))
            
            # One-Shot Decoding
            flat_logits = self.model.decode(flat_hidden, attention_mask=flat_att_mask)
            
            # Reshape back: [B, Steps, L, V]
            all_logits = flat_logits.view(batch_size, actual_max_length, -1, flat_logits.size(-1))

            # =================================================================
            # [Phase 4] Vectorized Loss Calculation
            # =================================================================
            # 1. Recon Loss (Masking 활용하여 루프 없이 계산 가능하지만, 메모리 아끼기 위해 flatten 사용)
            # targets: [B, Steps, L]
            # loss_masks: [B, Steps, L]
            
            # Valid Mask (Chain Length에 따른 마스크): [B, Steps]
            step_indices = torch.arange(actual_max_length, device=self.device).unsqueeze(0) # [1, Steps]
            chain_len_mask = step_indices < chain_lengths.unsqueeze(1) # [B, Steps]
            
            # Final Loss Mask = Chain Valid & Random Mask
            # [B, Steps, L]
            final_loss_mask = loss_masks & chain_len_mask.unsqueeze(-1)
            
            # Flatten for CrossEntropy
            active_logits = all_logits[final_loss_mask] # [N_active, V]
            active_targets = targets[:batch_size, :actual_max_length, :][final_loss_mask] # [N_active]
            
            if len(active_targets) > 0:
                recon_loss = self.recon_criterion(active_logits, active_targets)
            else:
                recon_loss = torch.tensor(0.0, device=self.device)

            # 2. Gate Loss
            # collected_gate_preds: List of [B, 1] -> [B, Steps]
            all_gate_preds = torch.cat(collected_gate_preds, dim=1) 
            target_gates = gate_targets[:, 1:actual_max_length+1] # [B, Steps]
            
            # Valid한 부분만 MSE
            if chain_len_mask.any():
                gate_loss_steps = self.gate_criterion(
                    all_gate_preds[chain_len_mask], 
                    target_gates[chain_len_mask]
                )
                total_gate_loss += gate_loss_steps

            # 3. Aux Loss (KL)
            avg_aux_loss = 0.0
            if self.loss_weight_aux > 0 and teacher_dist_all is not None:
                # Student Logits: [B_aux, Steps, L, V]
                aux_student_logits = all_logits[:aux_batch_size]
                
                # Log Softmax
                log_probs_student = torch.log_softmax(aux_student_logits / self.aux_temp, dim=-1)
                
                # KL Calculation (Element-wise first)
                kl_crit = nn.KLDivLoss(reduction='none')
                # [B_aux, Steps, L]
                kl_div = kl_crit(log_probs_student, teacher_dist_all).sum(dim=-1)
                
                # Masking
                aux_valid_mask = final_loss_mask[:aux_batch_size] # [B_aux, Steps, L]
                
                if aux_valid_mask.sum() > 0:
                    aux_loss_val = kl_div[aux_valid_mask].mean()
                    avg_aux_loss = aux_loss_val * (self.aux_temp ** 2)

            # Total Loss
            total_loss = (
                self.loss_weight_recon * recon_loss +
                self.loss_weight_gate * total_gate_loss +
                self.loss_weight_aux * avg_aux_loss
            )

        # Backward & Optimizer Step
        # (기존 코드와 동일)
        self.scaler.scale(total_loss / self.gradient_accumulation_steps).backward()
        
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.scheduler: self.scheduler.step()

        return (
            total_loss.item(),
            recon_loss.item(),
            total_gate_loss.item(),
            avg_aux_loss.item() if isinstance(avg_aux_loss, torch.Tensor) else avg_aux_loss
        )
    
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