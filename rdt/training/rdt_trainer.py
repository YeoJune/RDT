"""Training logic for RDT with Accelerate"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate import Accelerator
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, Optional
import math

from ..models.rdt import RDT
from ..utils import save_checkpoint, cleanup_checkpoints, count_parameters, count_parameters_without_context, CSVLogger


class RDTTrainer:
    """Trainer for Recursive Denoising Transformer"""
    
    def __init__(
        self,
        model: RDT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        accelerator: Accelerator
    ):
        self.accelerator = accelerator
        self.config = config
        self.resume_checkpoint = None  # main에서 설정 가능하도록

        self.train_loader = train_loader
        self.val_loader = val_loader
        
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
        
        # Accelerate Prepare: 모델, 옵티마이저, 데이터로더를 래핑하여 장치 분산 및 정밀도 관리 자동화
        self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler = \
            self.accelerator.prepare(
                model, self.optimizer, train_loader, val_loader, self.scheduler
            )
        
        # torch.compile (optional, 선택적으로 적용)
        # if hasattr(torch, 'compile') and self.accelerator.device.type == 'cuda':
        #     if self.accelerator.is_main_process:
        #         print("Compiling model with torch.compile...")
        #     self.model = torch.compile(self.model)
        
        # Loss functions
        self.recon_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        self.gate_criterion = nn.MSELoss()
        
        # Logging with W&B (Main Process Only)
        self.use_wandb = config.get('use_wandb', True)
        if self.use_wandb and self.accelerator.is_main_process:
            # Accelerator가 tracker 초기화를 처리하므로 여기서는 생략 가능
            # self.accelerator.init_trackers()는 train.py에서 호출됨
            pass
        
        # CSV Logger (Main Process Only)
        if self.accelerator.is_main_process:
            log_dir = Path(config['output'].get('log_dir', 'outputs/logs'))
            self.csv_logger = CSVLogger(str(log_dir))
        
        self.log_every_n_steps = config['output']['log_every_n_steps']
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
        
        # Print model info (Main Process Only)
        if self.accelerator.is_main_process:
            # unwrap해서 파라미터 계산
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            num_params = count_parameters(unwrapped_model)
            num_params_no_context = count_parameters_without_context(unwrapped_model)
            print(f"Model parameters: {num_params:,}")
            print(f"Model parameters (without context): {num_params_no_context:,}")
        
        self.is_tpu = self.accelerator.device.type == 'xla' or (hasattr(self.accelerator.state, "distributed_type") and self.accelerator.state.distributed_type == "TPU")
    
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

    def train_step(self, batch: Dict) -> Tuple[float, float, float, float]:
        self.model.train()
        raw_model = self.accelerator.unwrap_model(self.model)

        # 데이터 로드 (이미 전처리됨)
        input_tokens = batch['input'].to(self.accelerator.device)
        targets = batch['targets'].to(self.accelerator.device)
        attention_mask = batch['attention_mask'].to(self.accelerator.device)
        loss_masks = batch['loss_masks'].to(self.accelerator.device)
        gate_targets = batch['gate_targets'].to(self.accelerator.device)
        chain_lengths = batch['chain_lengths'].to(self.accelerator.device)
        
        batch_size = input_tokens.shape[0]
        sampling_prob = self.get_sampling_prob(self.current_epoch, self.global_step)

        # [분기점 1] TPU: 고정 길이 루프 / GPU: Dynamic 길이 루프
        if self.is_tpu:
            # TPU: Config에 설정된 최대 길이로 고정 (중간 탈출 절대 금지)
            max_length = self.config['training']['max_chain_length']
        else:
            # GPU: 실제 데이터 길이만큼만 돌기 (Dynamic)
            max_length = chain_lengths.max().item()
        
        aux_batch_size = max(1, int(batch_size * self.aux_ratio))
        
        # --- Step 0 ---
        h_0 = raw_model.encode_tokens(input_tokens)
        gate_pred_0, pooled_0 = raw_model.gate(h_0, attention_mask)
        
        # Gate Loss at Step 0 (전체 배치 계산 - GPU/TPU 공통)
        gate_loss_0 = torch.nn.functional.mse_loss(gate_pred_0, gate_targets[:, 0].unsqueeze(1))
        
        accumulated_loss = self.loss_weight_gate * gate_loss_0
        
        # TPU 최적화: detach()된 텐서로 누적
        total_recon_loss_tensor = torch.tensor(0.0, device=self.accelerator.device)
        total_gate_loss_tensor = gate_loss_0.detach()
        total_aux_loss_tensor = torch.tensor(0.0, device=self.accelerator.device)
        
        num_valid_steps = 1
        num_aux_steps = 0
        
        hidden = h_0
        gate_pred = gate_pred_0
        pooled = pooled_0
        
        # --- Recursive Steps ---
        for step_idx in range(max_length):
            valid_mask = chain_lengths > step_idx
            
            # [분기점 1-2] GPU: Early Break 허용 / TPU: Break 금지
            if not self.is_tpu:
                # GPU: 더 이상 계산할 게 없으면 칼같이 종료 (속도 이득)
                if valid_mask.sum() == 0:
                    break
            # TPU: break 없음. 끝까지 돔 (valid_mask로 Loss를 0으로 만들어서 무효화)
            
            step_gt_gate = gate_targets[:, step_idx].unsqueeze(1)
            hidden, gate_pred, pooled = raw_model.forward_step(
                hidden,
                attention_mask=attention_mask,
                last_gate_score=gate_pred,
                last_pooled=pooled,
                gt_timestep=step_gt_gate,
                sampling_prob=sampling_prob
            )
            
            step_targets = targets[:, step_idx, :]
            step_loss_mask = loss_masks[:, step_idx, :]
            
            # [A] Reconstruction Loss
            logits_pred = raw_model.decode(hidden, attention_mask)
            
            # [분기점 2] Recon Loss 계산 방식 분기
            if self.is_tpu:
                # TPU: 전체 계산 -> 마스킹 (Dynamic Shape 방지)
                loss_per_token = torch.nn.functional.cross_entropy(
                    logits_pred.transpose(1, 2), 
                    step_targets, 
                    reduction='none'
                )
                mask_float = step_loss_mask.float()
                recon_loss = (loss_per_token * mask_float).sum() / (mask_float.sum() + 1e-6)
            else:
                # GPU: 인덱싱 (속도/메모리 최적화)
                if step_loss_mask.sum() > 0:
                    recon_loss = self.recon_criterion(
                        logits_pred[step_loss_mask],
                        step_targets[step_loss_mask]
                    )
                else:
                    recon_loss = torch.tensor(0.0, device=self.accelerator.device)
            
            # [B] Aux Loss (KL Divergence)
            aux_kl = torch.tensor(0.0, device=self.accelerator.device)
            if self.loss_weight_aux > 0:
                aux_mask = step_loss_mask[:aux_batch_size]
                
                with torch.no_grad():
                    h_GT = raw_model.encode_tokens(step_targets[:aux_batch_size])
                    logits_GT = raw_model.decode(h_GT, attention_mask[:aux_batch_size])
                    target_dist = torch.softmax(logits_GT / self.aux_temp, dim=-1)
                
                log_probs_pred = torch.log_softmax(logits_pred[:aux_batch_size] / self.aux_temp, dim=-1)
                
                # [분기점 5] KL Divergence 계산 방식 분기
                if self.is_tpu:
                    # TPU: 수동 계산 (Compile 이슈 회피)
                    kl_loss_all = target_dist * (torch.log(target_dist + 1e-9) - log_probs_pred)
                else:
                    # GPU: nn.KLDivLoss 사용
                    kl_criterion = torch.nn.KLDivLoss(reduction='none')
                    kl_loss_all = kl_criterion(log_probs_pred, target_dist)
                
                kl_per_token = kl_loss_all.sum(dim=-1)  # [B, L]
                
                # [분기점 3] Aux Loss 계산 방식 분기
                if self.is_tpu:
                    # TPU: 마스킹 처리
                    aux_mask_float = aux_mask.float()
                    aux_kl = (kl_per_token * aux_mask_float).sum() / (aux_mask_float.sum() + 1e-6)
                else:
                    # GPU: 인덱싱 처리
                    if aux_mask.sum() > 0:
                        aux_kl = kl_per_token[aux_mask].mean()
                    else:
                        aux_kl = torch.tensor(0.0, device=self.accelerator.device)
                
                aux_kl = aux_kl * (self.aux_temp ** 2)
                total_aux_loss_tensor += aux_kl.detach()
                num_aux_steps += 1
            
            # [C] Gate Loss
            step_gate_target = gate_targets[:, step_idx + 1].unsqueeze(1)
            
            # [분기점 4] Gate Loss 계산 방식 분기
            if self.is_tpu:
                # TPU: 전체 MSE 계산 후 valid_mask 곱하기
                gate_loss_raw = torch.nn.functional.mse_loss(gate_pred, step_gate_target, reduction='none')
                valid_mask_float = valid_mask.float().unsqueeze(1)
                gate_loss = (gate_loss_raw * valid_mask_float).sum() / (valid_mask_float.sum() + 1e-6)
            else:
                # GPU: 유효한 샘플만 인덱싱
                if valid_mask.sum() > 0:
                    gate_pred_valid = gate_pred[valid_mask]
                    gate_target_valid = step_gate_target[valid_mask]
                    gate_loss = self.gate_criterion(gate_pred_valid, gate_target_valid)
                else:
                    gate_loss = torch.tensor(0.0, device=self.accelerator.device)
            
            step_loss = (
                self.loss_weight_recon * recon_loss +
                self.loss_weight_gate * gate_loss +
                self.loss_weight_aux * aux_kl
            )
            
            accumulated_loss += step_loss
            
            total_recon_loss_tensor += recon_loss.detach()
            total_gate_loss_tensor += gate_loss.detach()
            num_valid_steps += 1
        
        final_loss = accumulated_loss / num_valid_steps
        
        with self.accelerator.accumulate(self.model):
            self.accelerator.backward(final_loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler:
                self.scheduler.step()
        
        return (
            final_loss.item(),
            (total_recon_loss_tensor / num_valid_steps).item(),
            (total_gate_loss_tensor / num_valid_steps).item(),
            (total_aux_loss_tensor / num_aux_steps).item() if num_aux_steps > 0 else 0.0
        )
    
    def validate(self) -> Tuple[float, float, float]:
        self.model.eval()
        
        # Unwrap model to access custom methods (DDP-safe)
        raw_model = self.accelerator.unwrap_model(self.model)
        
        total_loss = 0
        total_recon = 0
        total_gate = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False, disable=not self.accelerator.is_local_main_process):
                # 데이터 로드 (이미 전처리됨)
                input_tokens = batch['input'].to(self.accelerator.device)
                targets = batch['targets'].to(self.accelerator.device)
                loss_masks = batch['loss_masks'].to(self.accelerator.device)
                attention_mask = batch['attention_mask'].to(self.accelerator.device)
                gate_targets = batch['gate_targets'].to(self.accelerator.device)
                chain_lengths = batch['chain_lengths'].to(self.accelerator.device)
                
                actual_max_length = chain_lengths.max().item()
                batch_recon = 0
                batch_gate = 0
                num_valid = 0
                
                # Initial encoding
                h_0 = raw_model.encode_tokens(input_tokens)
                gate_pred_0, pooled_0 = raw_model.gate(h_0, attention_mask, None, None)
                
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
                    hidden, gate_pred, pooled = raw_model.forward_step(
                        hidden,
                        attention_mask=attention_mask,
                        last_gate_score=gate_pred,
                        last_pooled=pooled
                    )
                    
                    # Reconstruction loss
                    step_targets = targets[:, step_idx, :]
                    step_loss_mask = loss_masks[:, step_idx, :]
                    
                    if step_loss_mask.sum() > 0:
                        logits = raw_model.decode(hidden, attention_mask)
                        sel_logits = logits[step_loss_mask]
                        sel_targ = step_targets[step_loss_mask]
                        recon_loss = self.recon_criterion(sel_logits, sel_targ)
                    else:
                        recon_loss = torch.tensor(0.0, device=self.accelerator.device)
                    
                    # Gate loss
                    step_gate_targets = gate_targets[:, step_idx + 1].unsqueeze(1)
                    gate_pred_valid = gate_pred[valid_mask]
                    gate_target_valid = step_gate_targets[valid_mask]
                    gate_loss = self.gate_criterion(gate_pred_valid, gate_target_valid) if len(gate_pred_valid) > 0 else torch.tensor(0.0, device=self.accelerator.device)
                    
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
        if self.accelerator.is_main_process:
            print(f"\nStarting epoch-based training for {self.num_epochs} epochs...")
        
        # Resume checkpoint if specified
        if self.resume_checkpoint:
            if self.accelerator.is_main_process:
                print(f"Loading checkpoint from {self.resume_checkpoint}")
            self.accelerator.load_state(self.resume_checkpoint)
            # TODO: 저장된 epoch/step 정보 복원 로직 추가 필요
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            sampling_prob = self.get_sampling_prob(epoch=epoch)
            
            if self.accelerator.is_main_process:
                print(f"\nEpoch {epoch + 1}/{self.num_epochs} | Sampling Prob: {sampling_prob:.3f}")
            
            epoch_loss = 0; epoch_recon = 0; epoch_gate = 0; epoch_aux = 0
            progress_bar = tqdm(
                self.train_loader, 
                desc="Training",
                disable=not self.accelerator.is_local_main_process
            )
            
            for batch in progress_bar:
                loss, recon, gate, aux = self.train_step(batch)
                epoch_loss += loss; epoch_recon += recon; epoch_gate += gate; epoch_aux += aux
                self.global_step += 1
                
                if self.global_step % self.log_every_n_steps == 0 and self.accelerator.is_main_process:
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
                        self.accelerator.log({'train/' + k if not k in ['epoch', 'step'] else k: v 
                                  for k, v in log_data.items()}, step=self.global_step)
                
                progress_bar.set_postfix({'loss': f'{loss:.4f}', 'recon': f'{recon:.4f}', 'aux': f'{aux:.4f}'})
            
            # Validation
            if (epoch + 1) % self.eval_every_n_epochs == 0:
                val_loss, val_recon, val_gate = self.validate()
                
                if self.accelerator.is_main_process:
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
                        self.accelerator.log({
                            'val/loss': val_loss,
                            'val/recon_loss': val_recon,
                            'val/gate_loss': val_gate,
                            'epoch': epoch
                        }, step=self.global_step)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        # Accelerator의 save_state 사용 (표준화)
                        save_path = f"{self.checkpoint_dir}/checkpoint-best"
                        self.accelerator.save_state(save_path)
            
            # Checkpoint
            if (epoch + 1) % self.save_every_n_epochs == 0:
                self.accelerator.wait_for_everyone()
                save_path = f"{self.checkpoint_dir}/checkpoint-epoch-{epoch+1}"
                self.accelerator.save_state(save_path)
                # cleanup_checkpoints는 필요시 별도 구현
        
        if self.accelerator.is_main_process:
            print("\nTraining completed!")
            self.csv_logger.close()
    
    def _train_by_step(self):
        if self.accelerator.is_main_process:
            print(f"\nStarting step-based training for {self.max_training_steps} steps...")
        
        # Resume checkpoint if specified
        if self.resume_checkpoint:
            if self.accelerator.is_main_process:
                print(f"Loading checkpoint from {self.resume_checkpoint}")
            self.accelerator.load_state(self.resume_checkpoint)
            # TODO: 저장된 epoch/step 정보 복원 로직 추가 필요
        
        best_val_loss = float('inf')
        
        step = 0
        epoch = 0
        
        while step < self.max_training_steps:
            self.current_epoch = epoch
            sampling_prob = self.get_sampling_prob(step=step)
            
            if self.accelerator.is_main_process:
                print(f"\nEpoch {epoch + 1} | Step {step}/{self.max_training_steps} | Sampling Prob: {sampling_prob:.3f}")
            
            progress_bar = tqdm(
                self.train_loader, 
                desc=f"Training (Step {step})",
                disable=not self.accelerator.is_local_main_process
            )
            
            for batch in progress_bar:
                if step >= self.max_training_steps:
                    break
                
                loss, recon, gate, aux = self.train_step(batch)
                step += 1
                self.global_step = step
                
                if step % self.log_every_n_steps == 0 and self.accelerator.is_main_process:
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
                        self.accelerator.log({'train/' + k if not k in ['epoch', 'step'] else k: v 
                                  for k, v in log_data.items()}, step=step)
                
                # Validation at regular step intervals
                if step % self.eval_every_n_steps == 0:
                    val_loss, val_recon, val_gate = self.validate()
                    
                    if self.accelerator.is_main_process:
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
                            self.accelerator.log({
                                'val/loss': val_loss,
                                'val/recon_loss': val_recon,
                                'val/gate_loss': val_gate,
                                'step': step
                            }, step=step)
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            # Accelerator의 save_state 사용
                            save_path = f"{self.checkpoint_dir}/checkpoint-best"
                            self.accelerator.save_state(save_path)
                
                # Checkpoint
                if step % self.save_every_n_steps == 0:
                    self.accelerator.wait_for_everyone()
                    save_path = f"{self.checkpoint_dir}/checkpoint-step-{step}"
                    self.accelerator.save_state(save_path)
                    # cleanup_checkpoints는 필요시 별도 구현
                
                progress_bar.set_postfix({'step': step, 'loss': f'{loss:.4f}', 'recon': f'{recon:.4f}', 'aux': f'{aux:.4f}'})
            
            epoch += 1
        
        if self.accelerator.is_main_process:
            print("\nTraining completed!")
            self.csv_logger.close()
