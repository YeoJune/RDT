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
        self.model.train()
        
        # 데이터 로드
        input_tokens = batch['input'].to(self.device)
        targets = batch['targets'].to(self.device)
        pos_ids = batch['pos_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        n_revealed = batch['n_revealed'].to(self.device) # GPU로 이동
        n_delta = batch['n_delta'].to(self.device)       # GPU로 이동
        gate_targets = batch['gate_targets'].to(self.device)
        chain_lengths = batch['chain_lengths']
        visible_ratio = self.config['training'].get('visible_loss_ratio', 0.15)
        
        batch_size, seq_len = input_tokens.shape
        actual_max_length = chain_lengths.max().item()
        
        accumulated_loss = 0
        total_recon_loss = 0
        total_gate_loss = 0
        num_valid_steps = 0
        
        # 1. 인코더 Forward (문맥 파악)
        hidden, gate_pred = self.model(
            input_tokens,
            pos_ids=pos_ids,
            attention_mask=attention_mask,
            is_first_step=True
        )
        
        # Position Index 미리 생성 (마스킹용)
        # (1, seq_len) 형태: [[0, 1, 2, ...]]
        pos_range = torch.arange(seq_len, device=self.device).unsqueeze(0)
        
        for step_idx in range(actual_max_length):
            valid_mask = chain_lengths > step_idx
            if valid_mask.sum() == 0: break
            
            step_targets = targets[:, step_idx, :]      # (B, Seq)
            step_n_revealed = n_revealed[:, step_idx].unsqueeze(1) # (B, 1)
            step_n_delta = n_delta[:, step_idx].unsqueeze(1)       # (B, 1)
                        
            # 1. Delta Mask (이번에 꼭 맞춰야 하는 구간)
            # 조건: n_revealed <= pos < n_revealed + n_delta
            delta_mask = (pos_range >= step_n_revealed) & (pos_range < (step_n_revealed + step_n_delta))
            
            # 2. Visible Mask (이미 밝혀진 구간 중 랜덤 샘플링)
            # 조건 1: pos < n_revealed (이미 밝혀진 구간)
            visible_region = (pos_range < step_n_revealed)
            
            # 조건 2: 랜덤 확률 (ratio)
            # torch.rand_like로 같은 크기의 랜덤 텐서 생성 후 비율보다 작은 곳만 True
            if visible_ratio > 0:
                random_select = torch.rand_like(pos_range, dtype=torch.float) < visible_ratio
                maintenance_mask = visible_region & random_select
            else:
                maintenance_mask = torch.zeros_like(delta_mask, dtype=torch.bool)
            
            # 3. 최종 Mask 합치기 (OR 연산)
            loss_mask = delta_mask | maintenance_mask
            
            if loss_mask.sum() > 0:
                # 2. Transformer Decoder Body 실행 (전체 문맥 유지)
                if hasattr(self.model.decoder, 'decoder'):
                    # (B, Seq, D_model) -> 아직 Vocab으로 확장 안 됨 (메모리 안전)
                    decoder_features = self.model.decoder.decoder(
                        hidden, 
                        src_key_padding_mask=(attention_mask == 0)
                    )
                    
                    # 3. 필요한 토큰만 쏙 뽑아내기 (Selection)
                    # (N_total_delta, D_model) 형태로 평탄화됨
                    selected_features = decoder_features[loss_mask]
                    selected_targets = step_targets[loss_mask]
                    
                    # 4. 뽑아낸 것만 단어로 변환 (Projection)
                    # 여기서 메모리 절약 효과 발생! (512개 다 하는 게 아니라 50개 정도만 함)
                    logits_active = self.model.decoder.projection(selected_features)
                    
                    recon_loss = self.recon_criterion(logits_active, selected_targets)
                else:
                    # Linear Decoder인 경우
                    selected_hidden = hidden[loss_mask]
                    selected_targets = step_targets[loss_mask]
                    logits_active = self.model.decoder(selected_hidden)
                    recon_loss = self.recon_criterion(logits_active, selected_targets)
            else:
                recon_loss = torch.tensor(0.0, device=self.device)
            
            # === Gate Loss ===
            step_gate_targets = gate_targets[:, step_idx].unsqueeze(1)
            gate_pred_valid = gate_pred[valid_mask]
            gate_target_valid = step_gate_targets[valid_mask]
            
            gate_loss = self.gate_criterion(gate_pred_valid, gate_target_valid) if len(gate_pred_valid) > 0 else torch.tensor(0.0, device=self.device)
            
            # Step Loss 합산
            step_loss = self.loss_weight_recon * recon_loss + self.loss_weight_gate * gate_loss
            accumulated_loss = accumulated_loss + step_loss
            
            total_recon_loss += recon_loss.item()
            total_gate_loss += gate_loss.item()
            num_valid_steps += 1
            
            # 다음 스텝 준비 (Recursive)
            if step_idx < actual_max_length - 1:
                hidden = self.model.encoder(hidden, mask=(attention_mask == 0))
                gate_pred = self.model.gate(hidden)
        
        # 최종 Loss
        final_loss = accumulated_loss / max(1, num_valid_steps)
        
        # Backward & Update (Accumulation 없이 매번 실행)
        self.optimizer.zero_grad()
        final_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        
        return final_loss.item(), total_recon_loss / max(1, num_valid_steps), total_gate_loss / max(1, num_valid_steps)
    
    def validate(self) -> Tuple[float, float, float]:
        """Validation loop with memory-aligned slicing"""
        self.model.eval()
        
        total_loss = 0
        total_recon_loss = 0
        total_gate_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                # 데이터 로드 (Train과 동일)
                input_tokens = batch['input'].to(self.device)
                targets = batch['targets'].to(self.device)
                pos_ids = batch['pos_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                n_revealed = batch['n_revealed'].to(self.device)
                n_delta = batch['n_delta'].to(self.device)
                gate_targets = batch['gate_targets'].to(self.device)
                chain_lengths = batch['chain_lengths']
                
                batch_size, seq_len = input_tokens.shape
                actual_max_length = chain_lengths.max().item()
                
                batch_recon_loss = 0
                batch_gate_loss = 0
                num_valid_steps = 0
                
                # 1. 인코더 Forward
                hidden, gate_pred = self.model(
                    input_tokens,
                    pos_ids=pos_ids, # (B, Seq) 그대로 전달
                    attention_mask=attention_mask,
                    is_first_step=True
                )
                
                # Position Index 미리 생성
                pos_range = torch.arange(seq_len, device=self.device).unsqueeze(0)
                
                for step_idx in range(actual_max_length):
                    valid_mask = chain_lengths > step_idx
                    if valid_mask.sum() == 0:
                        break
                    
                    step_targets = targets[:, step_idx, :]
                    step_n_revealed = n_revealed[:, step_idx].unsqueeze(1)
                    step_n_delta = n_delta[:, step_idx].unsqueeze(1)
                    
                    # === Slicing & Projection 최적화 (Train과 동일) ===
                    loss_mask = (pos_range >= step_n_revealed) & (pos_range < (step_n_revealed + step_n_delta))
                    
                    if loss_mask.sum() > 0:
                        if hasattr(self.model.decoder, 'decoder'):
                            # Decoder Body (Full Context)
                            decoder_features = self.model.decoder.decoder(
                                hidden,
                                src_key_padding_mask=(attention_mask == 0)
                            )
                            
                            # Selection (VRAM 절약)
                            selected_features = decoder_features[loss_mask]
                            selected_targets = step_targets[loss_mask]
                            
                            # Projection
                            logits_active = self.model.decoder.projection(selected_features)
                            
                            recon_loss = self.recon_criterion(logits_active, selected_targets)
                        else:
                            # Linear Decoder Case
                            selected_hidden = hidden[loss_mask]
                            selected_targets = step_targets[loss_mask]
                            logits_active = self.model.decoder(selected_hidden)
                            recon_loss = self.recon_criterion(logits_active, selected_targets)
                    else:
                        recon_loss = torch.tensor(0.0, device=self.device)
                    
                    # === Gate Loss ===
                    step_gate_targets = gate_targets[:, step_idx].unsqueeze(1)
                    gate_pred_valid = gate_pred[valid_mask]
                    gate_target_valid = step_gate_targets[valid_mask]
                    
                    gate_loss = self.gate_criterion(gate_pred_valid, gate_target_valid) if len(gate_pred_valid) > 0 else torch.tensor(0.0, device=self.device)
                    
                    batch_recon_loss += recon_loss.item()
                    batch_gate_loss += gate_loss.item()
                    num_valid_steps += 1
                    
                    # Next Step (Recursive)
                    if step_idx < actual_max_length - 1:
                        hidden = self.model.encoder(hidden, mask=(attention_mask == 0))
                        gate_pred = self.model.gate(hidden)
                
                # 배치 통계 집계
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