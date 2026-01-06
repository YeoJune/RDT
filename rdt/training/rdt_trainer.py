"""Training logic for RDT with Accelerate"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate import Accelerator
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
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
        
        # TPU 환경 감지 (prepare 전에 미리 확인)
        self.is_tpu = self.accelerator.device.type == 'xla' or (hasattr(self.accelerator.state, "distributed_type") and self.accelerator.state.distributed_type == "TPU")
        
        # 1. 모델, 옵티마이저, 스케줄러만 Accelerate로 준비 (로더 제외!)
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            model, self.optimizer, self.scheduler
        )
        
        # 2. 데이터로더는 TPU일 때 '수동'으로 분산 처리 (Accelerate Wrapper 제거)
        if self.is_tpu:
            from torch.utils.data.distributed import DistributedSampler
            
            # [Train Loader 재구성]
            # Accelerate Wrapper 대신 순정 DataLoader + DistributedSampler 사용
            train_sampler = DistributedSampler(
                train_loader.dataset,
                num_replicas=xr.world_size(),
                rank=xm.get_ordinal(),
                shuffle=True
            )
            self.train_loader = DataLoader(
                train_loader.dataset,
                batch_size=train_loader.batch_size,
                sampler=train_sampler,
                num_workers=train_loader.num_workers,
                collate_fn=train_loader.collate_fn,
                pin_memory=train_loader.pin_memory,
                drop_last=train_loader.drop_last
            )
            
            # [Val Loader 재구성]
            val_sampler = DistributedSampler(
                val_loader.dataset,
                num_replicas=xr.world_size(),
                rank=xm.get_ordinal(),
                shuffle=False
            )
            self.val_loader = DataLoader(
                val_loader.dataset,
                batch_size=val_loader.batch_size,
                sampler=val_sampler,
                num_workers=val_loader.num_workers,
                collate_fn=val_loader.collate_fn,
                pin_memory=val_loader.pin_memory,
                drop_last=val_loader.drop_last
            )
        else:
            # GPU/CPU일 때는 Accelerate가 알아서 잘 하므로 맡김
            self.train_loader, self.val_loader = self.accelerator.prepare(
                train_loader, val_loader
            )
        
        # Unwrap을 통해 원본 모델 접근 후 재할당
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped_model, 'output_projection') and hasattr(unwrapped_model, 'token_embedding'):
            unwrapped_model.output_projection.weight = unwrapped_model.token_embedding.weight
        
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

        # 데이터 로드
        input_tokens = batch['input'].to(self.accelerator.device)
        targets = batch['targets'].to(self.accelerator.device)
        attention_mask = batch['attention_mask'].to(self.accelerator.device)
        loss_masks = batch['loss_masks'].to(self.accelerator.device)
        gate_targets = batch['gate_targets'].to(self.accelerator.device)
        chain_lengths = batch['chain_lengths'].to(self.accelerator.device)
        
        batch_size = input_tokens.shape[0]
        sampling_prob = self.get_sampling_prob(self.current_epoch, self.global_step)

        if self.is_tpu:
            max_length = self.config['training']['max_chain_length']
        else:
            max_length = chain_lengths.max().item()
        
        aux_batch_size = max(1, int(batch_size * self.aux_ratio))
        
        # --- Step 0 ---
        h_0 = raw_model.encode_tokens(input_tokens)
        gate_pred_0, pooled_0 = raw_model.gate(h_0, attention_mask)
        
        gate_loss_0 = torch.nn.functional.mse_loss(gate_pred_0, gate_targets[:, 0].unsqueeze(1))
        accumulated_loss = self.loss_weight_gate * gate_loss_0
        
        # Logging용 텐서 (Gradient 계산엔 영향 X)
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
                num_aux_steps += 1
            
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
            num_valid_steps += 1
        
        # --- Gradient Accumulation Logic ---
        
        # 1. 평균 Loss 계산
        final_loss = accumulated_loss / num_valid_steps
        
        # 2. Loss Scaling
        if self.gradient_accumulation_steps > 1:
            final_loss = final_loss / self.gradient_accumulation_steps
        
        # 3. Backward (매 배치마다 수행하여 그래프 생성)
        self.accelerator.backward(final_loss)
        
        # 4. [필수] mark_step (Backward 직후): 여기서 미분 계산을 실행하라고 TPU에 명령
        # 이걸 안 하면 Accumulation 동안 그래프가 메모리에 계속 쌓여서 OOM 남
        xm.mark_step() 
        
        # 5. 조건부 Update (Accumulation이 찼을 때만 수행)
        # self.global_step은 현재 완료된 배치가 아니라 '시작 전' 카운트이므로 +1 해서 체크
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # 6. [필수] mark_step (Update 직후): 가중치 갱신 실행 명령
            xm.mark_step() 
            
            self.optimizer.zero_grad()
            
            if self.scheduler:
                self.scheduler.step()
        
        # [수정] .item() 제거! 텐서 그대로 반환해야 파이프라인이 안 끊김
        # CPU로 값을 가져오지 말고 TPU 메모리에 둔 채로 레퍼런스만 넘깁니다.
        return (
            (final_loss.detach() * self.gradient_accumulation_steps) if self.gradient_accumulation_steps > 1 else final_loss.detach(),
            (total_recon_loss_tensor / num_valid_steps).detach(),
            (total_gate_loss_tensor / num_valid_steps).detach(),
            (total_aux_loss_tensor / num_aux_steps).detach() if num_aux_steps > 0 else torch.tensor(0.0, device=self.accelerator.device)
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
            # [TPU Fix] ParallelLoader 강제 적용으로 데이터 공급 데드락 방지
            if self.is_tpu:
                # XLA 전용 로더를 lazy import로 가져옴 (CPU 환경 호환성 유지)
                import torch_xla.distributed.parallel_loader as pl
                
                # ParallelLoader로 래핑하여 백그라운드에서 TPU로 데이터 고속 전송
                val_device_loader = pl.ParallelLoader(
                    self.val_loader, 
                    [self.accelerator.device]
                ).per_device_loader(self.accelerator.device)
                
                val_iter = val_device_loader
            else:
                val_iter = self.val_loader
            
            # tqdm은 TPU I/O 락을 유발하므로 XLA 환경에서는 강제로 끄거나 주의해서 사용
            if self.use_tqdm and not self.is_tpu:
                val_iter = tqdm(val_iter, desc="Validating", leave=False, disable=not self.accelerator.is_local_main_process)
            
            for batch in val_iter:
                # 데이터 로드 (이미 전처리됨)
                input_tokens = batch['input'].to(self.accelerator.device)
                targets = batch['targets'].to(self.accelerator.device)
                loss_masks = batch['loss_masks'].to(self.accelerator.device)
                attention_mask = batch['attention_mask'].to(self.accelerator.device)
                gate_targets = batch['gate_targets'].to(self.accelerator.device)
                chain_lengths = batch['chain_lengths'].to(self.accelerator.device)
                
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
            
            # [TPU Fix] ParallelLoader 강제 적용으로 데이터 공급 데드락 방지
            if self.is_tpu:
                # XLA 전용 로더를 lazy import로 가져옴 (CPU 환경 호환성 유지)
                import torch_xla.distributed.parallel_loader as pl
                
                # ParallelLoader로 래핑하여 백그라운드에서 TPU로 데이터 고속 전송
                train_device_loader = pl.ParallelLoader(
                    self.train_loader, 
                    [self.accelerator.device]
                ).per_device_loader(self.accelerator.device)
                
                train_iter = train_device_loader
            else:
                train_iter = self.train_loader
            
            # tqdm은 TPU I/O 락을 유발하므로 XLA 환경에서는 강제로 끄거나 주의해서 사용
            if self.use_tqdm and not self.is_tpu:
                progress_bar = tqdm(
                    train_iter, 
                    desc="Training",
                    disable=not self.accelerator.is_local_main_process
                )
                train_iter = progress_bar
            
            for batch in train_iter:
                loss, recon, gate, aux = self.train_step(batch)
                
                # [수정] 텐서 누적 (CPU 동기화 X)
                epoch_loss += loss
                epoch_recon += recon
                epoch_gate += gate
                epoch_aux += aux
                
                self.global_step += 1
                
                if self.global_step % self.log_every_n_steps == 0 and self.accelerator.is_main_process:
                    # [핵심] 로그 찍는 순간에만 .item() 호출해서 값 가져옴
                    # 이렇게 해야 N steps 마다 한 번만 멈추고, 나머지는 전속력으로 달림
                    current_loss = loss.item()
                    current_recon = recon.item()
                    current_gate = gate.item()
                    current_aux = aux.item()

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
                    
                    # Print logging when tqdm is disabled
                    if not self.use_tqdm:
                        print(f"Epoch {epoch+1}/{self.num_epochs} | Step {self.global_step} | Loss: {current_loss:.4f} | Recon: {current_recon:.4f} | Gate: {current_gate:.4f} | Aux: {current_aux:.4f} | LR: {log_data['lr']:.6f}")
                
                if self.use_tqdm:
                    # tqdm도 로깅 시점에만 .item() 호출
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'recon': f'{recon.item():.4f}', 'aux': f'{aux.item():.4f}'})
            
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
            
            # [TPU Fix] ParallelLoader 강제 적용으로 데이터 공급 데드락 방지
            if self.is_tpu:
                # XLA 전용 로더를 lazy import로 가져옴 (CPU 환경 호환성 유지)
                import torch_xla.distributed.parallel_loader as pl
                
                # ParallelLoader로 래핑하여 백그라운드에서 TPU로 데이터 고속 전송
                train_device_loader = pl.ParallelLoader(
                    self.train_loader, 
                    [self.accelerator.device]
                ).per_device_loader(self.accelerator.device)
                
                train_iter = train_device_loader
            else:
                train_iter = self.train_loader
            
            # tqdm은 TPU I/O 락을 유발하므로 XLA 환경에서는 강제로 끄거나 주의해서 사용
            if self.use_tqdm and not self.is_tpu:
                progress_bar = tqdm(
                    train_iter, 
                    desc=f"Training (Step {step})",
                    disable=not self.accelerator.is_local_main_process
                )
                train_iter = progress_bar
            
            for batch in train_iter:
                if step >= self.max_training_steps:
                    break
                
                loss, recon, gate, aux = self.train_step(batch)
                step += 1
                self.global_step = step
                
                if step % self.log_every_n_steps == 0 and self.accelerator.is_main_process:
                    # [핵심] 여기서만 .item() 호출
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
                    
                    # Print logging when tqdm is disabled
                    if not self.use_tqdm:
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
                            # Accelerator의 save_state 사용
                            save_path = f"{self.checkpoint_dir}/checkpoint-best"
                            self.accelerator.save_state(save_path)
                
                # Checkpoint
                if step % self.save_every_n_steps == 0:
                    self.accelerator.wait_for_everyone()
                    save_path = f"{self.checkpoint_dir}/checkpoint-step-{step}"
                    self.accelerator.save_state(save_path)
                    # cleanup_checkpoints는 필요시 별도 구현
                
                if self.use_tqdm:
                    # tqdm도 로깅 시점에만 .item() 호출
                    progress_bar.set_postfix({'step': step, 'loss': f'{loss.item():.4f}', 'recon': f'{recon.item():.4f}', 'aux': f'{aux.item():.4f}'})
            
            epoch += 1
        
        if self.accelerator.is_main_process:
            print("\nTraining completed!")
            self.csv_logger.close()
