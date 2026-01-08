"""Training logic for RDT with Accelerate"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate import Accelerator, DistributedType
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
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # TPU 환경 감지 (표준 방식)
        self.is_tpu = self.accelerator.state.distributed_type == DistributedType.XLA
        
        # ============================================================================
        # [CRITICAL] Weight Tying - Step 1: Apply BEFORE prepare()
        # ============================================================================
        # Weight tying must be done before DDP/FSDP wrapping to ensure proper sharing
        if hasattr(model, 'tie_weights'):
            model.tie_weights()
        elif hasattr(model, 'output_projection') and hasattr(model, 'token_embedding'):
            model.output_projection.weight = model.token_embedding.weight
        
        # ============================================================================
        # Accelerate Prepare - Standard Pattern
        # ============================================================================
        # Prepare model, optimizer, and dataloaders together (scheduler comes later)
        self.model, self.optimizer, self.train_loader, self.val_loader = \
            self.accelerator.prepare(
                model, self.optimizer, train_loader, val_loader
            )
        
        # ============================================================================
        # [CRITICAL] Weight Tying - Step 2: Re-apply AFTER prepare()
        # ============================================================================
        # DDP/FSDP wrapping can break weight sharing, so we must re-apply
        # Use unwrap_model to access the underlying model
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped_model, 'tie_weights'):
            unwrapped_model.tie_weights()
        elif hasattr(unwrapped_model, 'output_projection') and hasattr(unwrapped_model, 'token_embedding'):
            unwrapped_model.output_projection.weight = unwrapped_model.token_embedding.weight
        
        # ============================================================================
        # Scheduler Creation - AFTER prepare() to use prepared DataLoader length
        # ============================================================================
        self.scheduler = self._create_scheduler()
        
        # Prepare scheduler separately (standard Accelerate pattern)
        if self.scheduler:
            self.scheduler = self.accelerator.prepare(self.scheduler)
        
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
        
        # Print model info (Main Process Only)
        if self.accelerator.is_main_process:
            # unwrap해서 파라미터 계산
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            num_params = count_parameters(unwrapped_model)
            num_params_no_context = count_parameters_without_context(unwrapped_model)
            print(f"Model parameters: {num_params:,}")
            print(f"Model parameters (without context): {num_params_no_context:,}")
            
            # ========================================================================
            # [VERIFICATION] Weight Tying Status Check
            # ========================================================================
            if hasattr(unwrapped_model, 'output_projection') and hasattr(unwrapped_model, 'token_embedding'):
                is_tied = unwrapped_model.output_projection.weight is unwrapped_model.token_embedding.weight
                print(f"\nWeight Tying Status: {'✓ ENABLED' if is_tied else '✗ FAILED'}")
                
                if not is_tied:
                    print("ERROR: Weight tying verification failed!")
                    print("Attempting to fix...")
                    
                    # Try to fix
                    if hasattr(unwrapped_model, 'tie_weights'):
                        unwrapped_model.tie_weights()
                    else:
                        unwrapped_model.output_projection.weight = unwrapped_model.token_embedding.weight
                    
                    # Final verification
                    is_tied_after_fix = unwrapped_model.output_projection.weight is unwrapped_model.token_embedding.weight
                    if is_tied_after_fix:
                        print("✓ Weight tying fixed successfully!")
                    else:
                        print("✗ CRITICAL: Weight tying could not be fixed!")
                        print("This may lead to incorrect training behavior.")
                else:
                    # Show memory addresses to confirm
                    emb_addr = id(unwrapped_model.token_embedding.weight)
                    proj_addr = id(unwrapped_model.output_projection.weight)
                    print(f"  - Token embedding: {emb_addr}")
                    print(f"  - Output projection: {proj_addr}")
                    print(f"  - Same object: {emb_addr == proj_addr}")
    
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

        # 데이터 로드 (prepare된 DataLoader는 이미 올바른 device에 배치됨)
        input_tokens = batch['input']
        targets = batch['targets']
        attention_mask = batch['attention_mask']
        loss_masks = batch['loss_masks']
        gate_targets = batch['gate_targets']
        chain_lengths = batch['chain_lengths']
        
        batch_size = input_tokens.shape[0]
        # [Standard Fix] Float를 Tensor로 변환하여 XLA 컴파일 최적화
        sampling_prob_val = self.get_sampling_prob(self.current_epoch, self.global_step)
        sampling_prob = torch.tensor(sampling_prob_val, device=self.accelerator.device)

        # [TPU Optimization] max_length 계산
        if self.is_tpu:
            # TPU: 고정 길이 사용하여 그래프 안정성 확보
            max_length = self.config['training']['max_chain_length']
        else:
            # GPU: 동적 길이로 효율성 확보
            max_length = chain_lengths.max().item()
        
        aux_batch_size = max(1, int(batch_size * self.aux_ratio))
        
        # --- Step 0 ---
        h_0 = raw_model.encode_tokens(input_tokens)
        gate_pred_0, pooled_0 = raw_model.gate(h_0, attention_mask)
        
        gate_loss_0 = torch.nn.functional.mse_loss(gate_pred_0, gate_targets[:, 0].unsqueeze(1))
        accumulated_loss = self.loss_weight_gate * gate_loss_0
        
        # Logging용 텐서 (detached internally where created or via logic below)
        total_recon_loss_tensor = torch.tensor(0.0, device=self.accelerator.device)
        total_gate_loss_tensor = gate_loss_0.detach()
        total_aux_loss_tensor = torch.tensor(0.0, device=self.accelerator.device)
        
        num_valid_steps = torch.tensor(1.0, device=self.accelerator.device)
        num_aux_steps = torch.tensor(0.0, device=self.accelerator.device)
        
        hidden = h_0
        gate_pred = gate_pred_0
        pooled = pooled_0
        
        # --- Recursive Steps ---
        for step_idx in range(max_length):
            valid_mask = chain_lengths > step_idx
            if not self.is_tpu and valid_mask.sum() == 0:
                break
            
            step_gt_gate = gate_targets[:, step_idx].unsqueeze(1)
            hidden, gate_pred, pooled = raw_model.forward_step(
                hidden, attention_mask=attention_mask, last_gate_score=gate_pred,
                last_pooled=pooled, gt_timestep=step_gt_gate, sampling_prob=sampling_prob
            )
            
            step_targets = targets[:, step_idx, :]
            step_loss_mask = loss_masks[:, step_idx, :]
            
            # [A] Recon Loss
            logits_pred = raw_model.decode(hidden, attention_mask)
            if self.is_tpu:
                loss_per_token = torch.nn.functional.cross_entropy(logits_pred.transpose(1, 2), step_targets, reduction='none')
                mask_float = step_loss_mask.float()
                recon_loss = (loss_per_token * mask_float).sum() / (mask_float.sum() + 1e-6)
            else:
                if step_loss_mask.sum() > 0:
                    recon_loss = self.recon_criterion(logits_pred[step_loss_mask], step_targets[step_loss_mask])
                else:
                    recon_loss = torch.tensor(0.0, device=self.accelerator.device)
            
            # [B] Aux Loss
            aux_kl = torch.tensor(0.0, device=self.accelerator.device)
            if self.loss_weight_aux > 0:
                aux_mask = step_loss_mask[:aux_batch_size]
                with torch.no_grad():
                    h_GT = raw_model.encode_tokens(step_targets[:aux_batch_size])
                    logits_GT = raw_model.decode(h_GT, attention_mask[:aux_batch_size])
                    target_dist = torch.softmax(logits_GT / self.aux_temp, dim=-1)
                
                log_probs_pred = torch.log_softmax(logits_pred[:aux_batch_size] / self.aux_temp, dim=-1)
                
                if self.is_tpu:
                    kl_loss_all = target_dist * (torch.log(target_dist + 1e-9) - log_probs_pred)
                    kl_per_token = kl_loss_all.sum(dim=-1)
                    aux_mask_float = aux_mask.float()
                    aux_kl = (kl_per_token * aux_mask_float).sum() / (aux_mask_float.sum() + 1e-6)
                else:
                    if aux_mask.sum() > 0:
                        kl_criterion = torch.nn.KLDivLoss(reduction='none')
                        kl_per_token = kl_criterion(log_probs_pred, target_dist).sum(dim=-1)
                        aux_kl = kl_per_token[aux_mask].mean()
                
                aux_kl = aux_kl * (self.aux_temp ** 2)
                total_aux_loss_tensor += aux_kl.detach()
                num_aux_steps += 1.0
            
            # [C] Gate Loss
            step_gate_target = gate_targets[:, step_idx + 1].unsqueeze(1)
            if self.is_tpu:
                gate_loss_raw = torch.nn.functional.mse_loss(gate_pred, step_gate_target, reduction='none')
                valid_mask_float = valid_mask.float().unsqueeze(1)
                gate_loss = (gate_loss_raw * valid_mask_float).sum() / (valid_mask_float.sum() + 1e-6)
            else:
                if valid_mask.sum() > 0:
                    gate_loss = self.gate_criterion(gate_pred[valid_mask], step_gate_target[valid_mask])
                else:
                    gate_loss = torch.tensor(0.0, device=self.accelerator.device)
            
            step_loss = (self.loss_weight_recon * recon_loss + self.loss_weight_gate * gate_loss + self.loss_weight_aux * aux_kl)
            accumulated_loss += step_loss
            
            total_recon_loss_tensor += recon_loss.detach()
            total_gate_loss_tensor += gate_loss.detach()
            num_valid_steps += 1.0
        
        # ============================================================================
        # Loss Calculation - Accelerate Standard Pattern
        # ============================================================================
        # Calculate average loss per recursive step
        final_loss = accumulated_loss / num_valid_steps
        
        # IMPORTANT: Loss scaling is handled automatically by accelerator.accumulate()
        # 
        # How it works:
        # 1. We pass gradient_accumulation_steps to Accelerator.__init__()
        # 2. accelerator.accumulate() context manager automatically:
        #    - Scales gradients by 1/gradient_accumulation_steps
        #    - Controls when to sync gradients across devices
        #    - Handles optimizer stepping at the right intervals
        # 
        # Reference: https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation
        # 
        # DO NOT manually divide loss by gradient_accumulation_steps here!
        # That would cause double-scaling and incorrect training.
        
        # Return final_loss for backward pass (not detached)
        # Return logging tensors (detached)
        return (
            final_loss,
            (total_recon_loss_tensor / num_valid_steps).detach(),
            (total_gate_loss_tensor / num_valid_steps).detach(),
            (total_aux_loss_tensor / (num_aux_steps + 1e-6)).detach()
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
            # Accelerate가 준비한 DataLoader 그대로 사용
            val_iter = self.val_loader
            
            if self.use_tqdm:
                val_iter = tqdm(val_iter, desc="Validating", leave=False, disable=not self.accelerator.is_local_main_process)
            
            for batch in val_iter:
                # 데이터 로드 (prepare된 DataLoader는 이미 올바른 device에 배치됨)
                input_tokens = batch['input']
                targets = batch['targets']
                loss_masks = batch['loss_masks']
                attention_mask = batch['attention_mask']
                gate_targets = batch['gate_targets']
                chain_lengths = batch['chain_lengths']
                
                # [TPU Optimization] max_length 계산
                if self.is_tpu:
                    actual_max_length = self.config['training']['max_chain_length']
                else:
                    actual_max_length = chain_lengths.max().item()
                # [최적화] 텐서로 누적 (매 스텝마다 .item() 호출하지 않음)
                batch_recon_tensor = torch.tensor(0.0, device=self.accelerator.device)
                batch_gate_tensor = torch.tensor(0.0, device=self.accelerator.device)
                num_valid = 0
                
                # Initial encoding
                h_0 = raw_model.encode_tokens(input_tokens)
                gate_pred_0, pooled_0 = raw_model.gate(h_0, attention_mask, None, None)
                
                # Gate loss for h_0
                gate_target_0 = gate_targets[:, 0].unsqueeze(1)
                gate_loss_0 = self.gate_criterion(gate_pred_0, gate_target_0)
                batch_gate_tensor += gate_loss_0
                num_valid += 1
                
                # Iterative steps
                hidden = h_0
                gate_pred = gate_pred_0
                pooled = pooled_0
                
                for step_idx in range(actual_max_length):
                    valid_mask = chain_lengths > step_idx
                    # [TPU Consistency] train_step과 동일한 패턴 적용
                    if not self.is_tpu and valid_mask.sum() == 0:
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
                    
                    # [최적화] 텐서끼리 덧셈 (동기화 없음)
                    batch_recon_tensor += recon_loss
                    batch_gate_tensor += gate_loss
                    num_valid += 1
                
                # [최적화] 배치 끝에 한 번만 .item() 호출
                if num_valid > 0:
                    avg_recon = (batch_recon_tensor / num_valid).item()
                    avg_gate = (batch_gate_tensor / num_valid).item()
                    avg_total = self.loss_weight_recon * avg_recon + self.loss_weight_gate * avg_gate
                    
                    total_loss += avg_total
                    total_recon += avg_recon
                    total_gate += avg_gate
                    num_batches += 1
        
        if num_batches == 0:
            return 0.0, 0.0, 0.0
        
        # [Distributed Validation] 모든 프로세스에서 값을 수집하여 평균 계산
        # 이렇게 하면 전체 validation set의 정확한 평균을 얻을 수 있음
        total_loss_tensor = torch.tensor(total_loss, device=self.accelerator.device)
        total_recon_tensor = torch.tensor(total_recon, device=self.accelerator.device)
        total_gate_tensor = torch.tensor(total_gate, device=self.accelerator.device)
        num_batches_tensor = torch.tensor(num_batches, device=self.accelerator.device)
        
        # gather: 모든 프로세스의 값을 수집
        total_loss_tensor = self.accelerator.gather(total_loss_tensor).mean()
        total_recon_tensor = self.accelerator.gather(total_recon_tensor).mean()
        total_gate_tensor = self.accelerator.gather(total_gate_tensor).mean()
        num_batches_tensor = self.accelerator.gather(num_batches_tensor).sum()
        
        if num_batches_tensor == 0:
            return 0.0, 0.0, 0.0
        
        return (
            total_loss_tensor.item(),
            total_recon_tensor.item(),
            total_gate_tensor.item()
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
            from ..utils import load_checkpoint
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            checkpoint = load_checkpoint(
                self.resume_checkpoint,
                unwrapped_model,
                self.optimizer,
                self.scheduler
            )
            self.current_epoch = checkpoint.get('epoch', 0)
            self.global_step = checkpoint.get('step', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            if self.accelerator.is_main_process:
                print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
        else:
            best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            sampling_prob = self.get_sampling_prob(epoch=epoch)
            
            if self.accelerator.is_main_process:
                print(f"\nEpoch {epoch + 1}/{self.num_epochs} | Sampling Prob: {sampling_prob:.3f}")
            
            # [Standard Fix] 에폭 단위 누적 변수 제거 (TPU 그래프 폭발 방지)
            
            # Accelerate가 준비한 DataLoader 그대로 사용
            train_iter = self.train_loader
            
            if self.use_tqdm:
                progress_bar = tqdm(
                    train_iter, 
                    desc="Training",
                    disable=not self.accelerator.is_local_main_process
                )
                train_iter = progress_bar
            
            for batch in train_iter:
                # Accelerate의 gradient accumulation 자동 처리
                with self.accelerator.accumulate(self.model):
                    loss, recon, gate, aux = self.train_step(batch)

                    # Backward & Optimizer Step (accumulate 컨텍스트가 자동 처리)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if self.scheduler:
                        self.scheduler.step()
                
                # [Standard Fix] 루프 내 불필요한 연산 및 동기화 제거
                # epoch_loss += loss.item()  <-- 삭제 (매 스텝 Sync 유발)
                # epoch_loss += loss.detach() <-- 삭제 (그래프 폭발 유발)
                
                self.global_step += 1
                
                # ====================================================================
                # Logging - Standard Pattern
                # ====================================================================
                if self.global_step % self.log_every_n_steps == 0:
                    # Get scalar values (accumulate() context handles averaging)
                    # No need to manually scale - the reported loss is already correct
                    current_loss = loss.item()
                    current_recon = recon.item()
                    current_gate = gate.item()
                    current_aux = aux.item()
                    
                    if self.accelerator.is_main_process:
                        log_data = {
                            'epoch': epoch,
                            'step': self.global_step,
                            'loss': current_loss,
                            'recon_loss': current_recon,
                            'gate_loss': current_gate,
                            'aux_loss': current_aux,
                            'lr': self.optimizer.param_groups[0]['lr'],
                            'sampling_prob': sampling_prob
                        }
                        
                        # CSV logging
                        self.csv_logger.log(log_data)
                        
                        # W&B logging
                        if self.use_wandb:
                            self.accelerator.log({'train/' + k if not k in ['epoch', 'step'] else k: v 
                                      for k, v in log_data.items()}, step=self.global_step)
                        
                        # tqdm update
                        if self.use_tqdm:
                            progress_bar.set_postfix({
                                'step': self.global_step,
                                'loss': f'{current_loss:.4f}',
                                'recon': f'{current_recon:.4f}',
                                'aux': f'{current_aux:.4f}'
                            })
                        else:
                            # Print logging when tqdm is disabled
                            print(f"Epoch {epoch+1}/{self.num_epochs} | Step {self.global_step} | Loss: {current_loss:.4f} | Recon: {current_recon:.4f} | Gate: {current_gate:.4f} | Aux: {current_aux:.4f} | LR: {log_data['lr']:.6f}")
            
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
                        # torch.save 사용 (torch.compile + accelerator.save_state 문제 해결)
                        self.accelerator.wait_for_everyone()
                        if self.accelerator.is_main_process:
                            unwrapped_model = self.accelerator.unwrap_model(self.model)
                            # torch.compile으로 인한 _orig_mod prefix 제거
                            if hasattr(unwrapped_model, '_orig_mod'):
                                unwrapped_model = unwrapped_model._orig_mod
                            
                            from ..utils import save_checkpoint
                            save_checkpoint(
                                model=unwrapped_model,
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
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    # torch.compile으로 인한 _orig_mod prefix 제거
                    if hasattr(unwrapped_model, '_orig_mod'):
                        unwrapped_model = unwrapped_model._orig_mod
                    
                    from ..utils import save_checkpoint, cleanup_checkpoints
                    save_checkpoint(
                        model=unwrapped_model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch + 1,
                        step=self.global_step,
                        loss=0.0,  # Not applicable for regular checkpoints
                        config=self.config,
                        checkpoint_dir=self.checkpoint_dir,
                        filename=f"checkpoint_epoch_{epoch+1}_step_{self.global_step}.pt"
                    )
                    cleanup_checkpoints(self.checkpoint_dir, self.keep_last_n_checkpoints)
        
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
            from ..utils import load_checkpoint
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            checkpoint = load_checkpoint(
                self.resume_checkpoint,
                unwrapped_model,
                self.optimizer,
                self.scheduler
            )
            epoch = checkpoint.get('epoch', 0)
            step = checkpoint.get('step', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.current_epoch = epoch
            self.global_step = step
            if self.accelerator.is_main_process:
                print(f"Resumed from epoch {epoch}, step {step}")
        else:
            best_val_loss = float('inf')
            step = 0
            epoch = 0
        
        while step < self.max_training_steps:
            self.current_epoch = epoch
            sampling_prob = self.get_sampling_prob(step=step)
            
            if self.accelerator.is_main_process:
                print(f"\nEpoch {epoch + 1} | Step {step}/{self.max_training_steps} | Sampling Prob: {sampling_prob:.3f}")
            
            # Accelerate가 준비한 DataLoader 그대로 사용
            train_iter = self.train_loader
            
            if self.use_tqdm:
                progress_bar = tqdm(
                    train_iter, 
                    desc=f"Training (Step {step})",
                    disable=not self.accelerator.is_local_main_process
                )
                train_iter = progress_bar
            
            for batch in train_iter:
                if step >= self.max_training_steps:
                    break
                
                # Accelerate의 gradient accumulation 자동 처리
                with self.accelerator.accumulate(self.model):
                    loss, recon, gate, aux = self.train_step(batch)
                    
                    # Backward & Optimizer Step (accumulate 컨텍스트가 자동 처리)
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if self.scheduler:
                        self.scheduler.step()
                
                step += 1
                self.global_step = step
                
                # ====================================================================
                # Logging - Standard Pattern
                # ====================================================================
                if step % self.log_every_n_steps == 0 and self.accelerator.is_main_process:
                    # Get scalar values (accumulate() context handles averaging)
                    current_loss = loss.item()
                    current_recon = recon.item()
                    current_gate = gate.item()
                    current_aux = aux.item()

                    log_data = {
                        'epoch': epoch,
                        'step': step,
                        'loss': current_loss,
                        'recon_loss': current_recon,
                        'gate_loss': current_gate,
                        'aux_loss': current_aux,
                        'lr': self.optimizer.param_groups[0]['lr'],
                        'sampling_prob': self.get_sampling_prob(step=step)
                    }
                    
                    self.csv_logger.log(log_data)
                    
                    if self.use_wandb:
                        self.accelerator.log({'train/' + k if not k in ['epoch', 'step'] else k: v 
                                  for k, v in log_data.items()}, step=step)
                    
                    # tqdm update
                    if self.use_tqdm:
                        progress_bar.set_postfix({
                            'step': step,
                            'loss': f'{current_loss:.4f}',
                            'recon': f'{current_recon:.4f}',
                            'aux': f'{current_aux:.4f}'
                        })
                    else:
                        # Print logging when tqdm is disabled
                        print(f"Epoch {epoch+1} | Step {step}/{self.max_training_steps} | Loss: {current_loss:.4f} | Recon: {current_recon:.4f} | Gate: {current_gate:.4f} | Aux: {current_aux:.4f} | LR: {log_data['lr']:.6f}")
                
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
                            # torch.save 사용 (torch.compile + accelerator.save_state 문제 해결)
                            self.accelerator.wait_for_everyone()
                            if self.accelerator.is_main_process:
                                unwrapped_model = self.accelerator.unwrap_model(self.model)
                                # torch.compile으로 인한 _orig_mod prefix 제거
                                if hasattr(unwrapped_model, '_orig_mod'):
                                    unwrapped_model = unwrapped_model._orig_mod
                                
                                from ..utils import save_checkpoint
                                save_checkpoint(
                                    model=unwrapped_model,
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
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.is_main_process:
                        unwrapped_model = self.accelerator.unwrap_model(self.model)
                        # torch.compile으로 인한 _orig_mod prefix 제거
                        if hasattr(unwrapped_model, '_orig_mod'):
                            unwrapped_model = unwrapped_model._orig_mod
                        
                        from ..utils import save_checkpoint, cleanup_checkpoints
                        save_checkpoint(
                            model=unwrapped_model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            epoch=epoch,
                            step=step,
                            loss=0.0,  # Not applicable for regular checkpoints
                            config=self.config,
                            checkpoint_dir=self.checkpoint_dir,
                            filename=f"checkpoint_epoch_{epoch}_step_{step}.pt"
                        )
                        cleanup_checkpoints(self.checkpoint_dir, self.keep_last_n_checkpoints)
            
            epoch += 1
        
        if self.accelerator.is_main_process:
            print("\nTraining completed!")
            self.csv_logger.close()
