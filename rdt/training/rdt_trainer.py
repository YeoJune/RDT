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
        self.resume_checkpoint = None
        
        # ============================================================================
        # ✅ STANDARD PATTERN: Save collate_fn reference BEFORE prepare()
        # ============================================================================
        # This is the standard way to maintain access to dataloader components
        # that need to be modified during training (e.g., curriculum learning)
        self.train_collate_fn = train_loader.collate_fn if hasattr(train_loader, 'collate_fn') else None
        self.val_collate_fn = val_loader.collate_fn if hasattr(val_loader, 'collate_fn') else None
        
        # Store original dataloaders for reference if needed
        self.original_train_loader = train_loader
        self.original_val_loader = val_loader

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
        
        # TPU 환경 감지 (표준 방식)
        self.is_tpu = self.accelerator.state.distributed_type == DistributedType.XLA
        
        # Weight tying enabled status from config
        self.weight_tying = config['model'].get('weight_tying', True)
        
        # ============================================================================
        # [CRITICAL] Weight Tying - Step 1: Apply BEFORE prepare() (if enabled)
        # ============================================================================
        # Weight tying must be done before DDP/FSDP wrapping to ensure proper sharing
        if self.weight_tying:
            if hasattr(model, 'tie_weights'):
                model.tie_weights()
            elif hasattr(model, 'output_projection') and hasattr(model, 'token_embedding'):
                model.output_projection.weight = model.token_embedding.weight
        
        # ============================================================================
        # Accelerate Prepare - Standard Pattern
        # ============================================================================
        self.model, self.optimizer, self.train_loader, self.val_loader = \
            self.accelerator.prepare(
                model, self.optimizer, train_loader, val_loader
            )
        
        # ============================================================================
        # [CRITICAL] Weight Tying - Step 2: Re-apply AFTER prepare() (if enabled)
        # ============================================================================
        # DDP/FSDP wrapping can break weight sharing, so we must re-apply
        # Use unwrap_model to access the underlying model
        if self.weight_tying:
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
        self.recon_criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)  # Ignore padding
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
                
                if self.weight_tying:
                    # Weight tying should be enabled
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
                else:
                    # Weight tying should be disabled
                    print(f"\nWeight Tying Status: {'✓ DISABLED (as configured)' if not is_tied else '⚠️ UNEXPECTED - Still tied'}")
                    if is_tied:
                        print("WARNING: Weights are tied even though weight_tying=False in config.")
                        print("This might be due to model checkpoint having tied weights.")
    
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

    def train_step(self, batch: Dict) -> Tuple[float, float, float, float, float]:
        """
        Standard loss computation with token-weighted averaging (MLM standard).
        
        Returns:
            final_loss: Total weighted loss (used for backprop)
            avg_recon: Average reconstruction loss (token-weighted)
            avg_gate: Average gate loss (sample-weighted)
            avg_aux: Average auxiliary loss (token-weighted)
            accuracy: Token-level accuracy
        """
        self.model.train()

        # Load data
        input_tokens = batch['input']
        targets = batch['targets']
        attention_mask = batch['attention_mask']
        loss_masks = batch['loss_masks']
        gate_targets = batch['gate_targets']
        chain_lengths = batch['chain_lengths']
        
        batch_size = input_tokens.shape[0]
        sampling_prob_val = self.get_sampling_prob(self.current_epoch, self.global_step)
        sampling_prob = torch.tensor(sampling_prob_val, device=self.accelerator.device)

        # [TPU Optimization] max_length 계산
        if self.is_tpu:
            max_length = self.config['training']['max_chain_length']
        else:
            max_length = chain_lengths.max().item()
        
        aux_batch_size = max(1, int(batch_size * self.aux_ratio))
        
        # ============================================================================
        # ✅ Token-weighted and Sample-weighted Loss Accumulation (MLM Standard)
        # ============================================================================
        # Reconstruction: Token-weighted
        total_recon_loss_raw = torch.tensor(0.0, device=self.accelerator.device)
        total_recon_tokens = torch.tensor(0.0, device=self.accelerator.device)
        
        # Gate: Sample-weighted
        total_gate_loss_raw = torch.tensor(0.0, device=self.accelerator.device)
        total_gate_samples = torch.tensor(0.0, device=self.accelerator.device)
        
        # Auxiliary: Token-weighted
        total_aux_loss_raw = torch.tensor(0.0, device=self.accelerator.device)
        total_aux_tokens = torch.tensor(0.0, device=self.accelerator.device)
        
        # Accuracy
        total_correct_tokens = torch.tensor(0.0, device=self.accelerator.device)
        total_target_tokens = torch.tensor(0.0, device=self.accelerator.device)
        
        # ============================================================================
        # STEP 0: Manual First Forward (tokens → h_0 → h_1)
        # ============================================================================
        
        # 1. Encode: tokens → h_0
        h_0 = self.model.encode(input_tokens, attention_mask)
        
        # 2. Gate prediction for h_0: gate_0, pooled_0
        gate_0, pooled_0 = self.model.gate(
            h_0,
            attention_mask,
            prev_pooled=None,
            prev_gate=None
        )
        
        # 3. Gate loss for gate_0 (predicting step 0's noise level)
        gate_target_0 = gate_targets[:, 0].unsqueeze(1)
        
        if self.is_tpu:
            gate_loss_raw_0 = torch.nn.functional.mse_loss(
                gate_0, gate_target_0, reduction='none'
            )
            # All samples are valid for step 0
            num_samples_0 = torch.tensor(batch_size, dtype=torch.float, device=self.accelerator.device)
            gate_loss_sum_0 = gate_loss_raw_0.sum()
        else:
            gate_loss_sum_0 = torch.nn.functional.mse_loss(
                gate_0, gate_target_0, reduction='sum'
            )
            num_samples_0 = torch.tensor(batch_size, dtype=torch.float, device=self.accelerator.device)
        
        total_gate_loss_raw += gate_loss_sum_0
        total_gate_samples += num_samples_0

        # Update state for next steps
        hidden, gate_pred, pooled = h_0, gate_0, pooled_0
        
        # ============================================================================
        # STEP 1 to max_length: Recursive Steps
        # ============================================================================
        
        for step_idx in range(0, max_length):
            valid_mask = chain_lengths > step_idx
            if not self.is_tpu and valid_mask.sum() == 0:
                break
            
            # GT timestep for this step
            gt_timestep = gate_targets[:, step_idx].unsqueeze(1)
            gt_noise = torch.randn_like(gt_timestep) * 0.2
            gt_timestep = gt_timestep + gt_noise
            
            # Forward: h_i → h_{i+1}
            hidden, gate_pred, pooled = self.model.forward(
                x=hidden,
                attention_mask=attention_mask,
                is_first_step=False,
                last_gate_score=gate_pred,
                last_pooled=pooled,
                gt_timestep=gt_timestep,
                sampling_prob=sampling_prob
            )
            
            # ========================================================================
            # Compute losses for step_idx
            # ========================================================================
            step_targets = targets[:, step_idx, :]
            step_loss_mask = loss_masks[:, step_idx, :]
            
            # [A] Reconstruction Loss & Accuracy (Token-weighted)
            logits_pred = self.model.decode(hidden, attention_mask)
            
            # Accuracy calculation (EM)
            pred_tokens = logits_pred.argmax(dim=-1)
            if self.is_tpu:
                correct = (pred_tokens == step_targets).float() * step_loss_mask.float()
                total_correct_tokens += correct.sum()
                total_target_tokens += step_loss_mask.float().sum()
            else:
                if step_loss_mask.sum() > 0:
                    correct = (pred_tokens[step_loss_mask] == step_targets[step_loss_mask]).float().sum()
                    total_correct_tokens += correct
                    total_target_tokens += step_loss_mask.sum().float()
            
            # ✅ Token-weighted Loss (MLM Standard)
            if self.is_tpu:
                loss_per_token = torch.nn.functional.cross_entropy(
                    logits_pred.transpose(1, 2), step_targets, 
                    reduction='none',
                    ignore_index=self.pad_token_id  # ✅ Added for efficiency
                )
                mask_float = step_loss_mask.float()
                step_recon_loss_sum = (loss_per_token * mask_float).sum()
                step_num_tokens = mask_float.sum()
            else:
                if step_loss_mask.sum() > 0:
                    # Use reduction='sum' for token-weighted averaging
                    criterion_sum = torch.nn.CrossEntropyLoss(
                        ignore_index=self.pad_token_id, 
                        reduction='sum'
                    )
                    step_recon_loss_sum = criterion_sum(
                        logits_pred[step_loss_mask], 
                        step_targets[step_loss_mask]
                    )
                    step_num_tokens = step_loss_mask.sum().float()
                else:
                    step_recon_loss_sum = torch.tensor(0.0, device=self.accelerator.device)
                    step_num_tokens = torch.tensor(0.0, device=self.accelerator.device)
            
            total_recon_loss_raw += step_recon_loss_sum
            total_recon_tokens += step_num_tokens
            
            # [B] Auxiliary Loss (Token-weighted)
            if self.loss_weight_aux > 0:
                aux_mask = step_loss_mask[:aux_batch_size]
                with torch.no_grad():
                    h_GT = self.model.encode(step_targets[:aux_batch_size], attention_mask[:aux_batch_size])
                    logits_GT = self.model.decode(h_GT, attention_mask[:aux_batch_size])
                    target_dist = torch.softmax(logits_GT / self.aux_temp, dim=-1)
                
                log_probs_pred = torch.log_softmax(logits_pred[:aux_batch_size] / self.aux_temp, dim=-1)
                
                if self.is_tpu:
                    kl_loss_all = target_dist * (torch.log(target_dist + 1e-9) - log_probs_pred)
                    kl_per_token = kl_loss_all.sum(dim=-1)  # [B_aux, L]
                    aux_mask_float = aux_mask.float()
                    step_aux_loss_sum = (kl_per_token * aux_mask_float).sum()
                    step_aux_tokens = aux_mask_float.sum()
                else:
                    if aux_mask.sum() > 0:
                        kl_criterion = torch.nn.KLDivLoss(reduction='none')
                        kl_per_token = kl_criterion(log_probs_pred, target_dist).sum(dim=-1)  # [B_aux, L]
                        step_aux_loss_sum = kl_per_token[aux_mask].sum()
                        step_aux_tokens = aux_mask.sum().float()
                    else:
                        step_aux_loss_sum = torch.tensor(0.0, device=self.accelerator.device)
                        step_aux_tokens = torch.tensor(0.0, device=self.accelerator.device)
                
                step_aux_loss_sum = step_aux_loss_sum * (self.aux_temp ** 2)
                total_aux_loss_raw += step_aux_loss_sum
                total_aux_tokens += step_aux_tokens
            
            # [C] Gate Loss (Sample-weighted)
            gate_target_next = gate_targets[:, step_idx + 1].unsqueeze(1)
            
            if self.is_tpu:
                gate_loss_raw = torch.nn.functional.mse_loss(
                    gate_pred, gate_target_next, reduction='none'
                )
                valid_mask_float = valid_mask.float().unsqueeze(1)
                step_gate_loss_sum = (gate_loss_raw * valid_mask_float).sum()
                step_num_samples = valid_mask_float.sum()
            else:
                if valid_mask.sum() > 0:
                    step_gate_loss_sum = torch.nn.functional.mse_loss(
                        gate_pred[valid_mask], 
                        gate_target_next[valid_mask], 
                        reduction='sum'
                    )
                    step_num_samples = valid_mask.sum().float()
                else:
                    step_gate_loss_sum = torch.tensor(0.0, device=self.accelerator.device)
                    step_num_samples = torch.tensor(0.0, device=self.accelerator.device)
            
            total_gate_loss_raw += step_gate_loss_sum
            total_gate_samples += step_num_samples
        
        # ============================================================================
        # ✅ Final Loss Calculation (Token/Sample-weighted Averages - MLM Standard)
        # ============================================================================
        
        # Token-weighted average for reconstruction
        avg_recon = total_recon_loss_raw / (total_recon_tokens + 1e-6)
        
        # Sample-weighted average for gate
        avg_gate = total_gate_loss_raw / (total_gate_samples + 1e-6)
        
        # Token-weighted average for auxiliary
        avg_aux = total_aux_loss_raw / (total_aux_tokens + 1e-6) if total_aux_tokens > 0 else torch.tensor(0.0, device=self.accelerator.device)
        
        # Weighted sum for backprop
        final_loss = (
            self.loss_weight_recon * avg_recon + 
            self.loss_weight_gate * avg_gate + 
            self.loss_weight_aux * avg_aux
        )
        
        # Accuracy
        accuracy = total_correct_tokens / (total_target_tokens + 1e-6)
        
        return (
            final_loss,
            avg_recon.detach(),
            avg_gate.detach(),
            avg_aux.detach(),
            accuracy.detach()
        )


    def validate(self) -> Tuple[float, float, float, float]:
        """
        Standard validation with token/sample-weighted loss computation.
        
        Returns:
            avg_total_loss: Weighted average of all losses
            avg_recon_loss: Average reconstruction loss (token-weighted)
            avg_gate_loss: Average gate loss (sample-weighted)
            avg_accuracy: Token-level accuracy
        """
        self.model.eval()
        
        # ✅ Global accumulators with token/sample weighting
        total_recon_loss_raw = torch.tensor(0.0, device=self.accelerator.device)
        total_recon_tokens = torch.tensor(0, device=self.accelerator.device)
        
        total_gate_loss_raw = torch.tensor(0.0, device=self.accelerator.device)
        total_gate_samples = torch.tensor(0, device=self.accelerator.device)
        
        total_correct_tokens = torch.tensor(0.0, device=self.accelerator.device)
        total_target_tokens = torch.tensor(0.0, device=self.accelerator.device)
        
        with torch.no_grad():
            val_iter = self.val_loader
            
            if self.use_tqdm:
                val_iter = tqdm(val_iter, desc="Validating", leave=False, 
                            disable=not self.accelerator.is_local_main_process)
            
            for batch in val_iter:
                input_tokens = batch['input']
                targets = batch['targets']
                loss_masks = batch['loss_masks']
                attention_mask = batch['attention_mask']
                gate_targets = batch['gate_targets']
                chain_lengths = batch['chain_lengths']
                
                batch_size = input_tokens.shape[0]
                
                # [TPU Optimization] max_length 계산
                if self.is_tpu:
                    actual_max_length = self.config['training']['max_chain_length']
                else:
                    actual_max_length = chain_lengths.max().item()
                
                # ================================================================
                # STEP 0: Manual First Forward (tokens → h_0 → h_1)
                # ================================================================
                
                # 1. Encode: tokens → h_0
                h_0 = self.model.encode(input_tokens, attention_mask)
                
                # 2. Gate prediction for h_0: gate_0, pooled_0
                gate_0, pooled_0 = self.model.gate(
                    h_0,
                    attention_mask,
                    prev_pooled=None,
                    prev_gate=None
                )
                
                # 3. Gate loss for gate_0
                gate_target_0 = gate_targets[:, 0].unsqueeze(1)
                
                if self.is_tpu:
                    gate_loss_raw_0 = torch.nn.functional.mse_loss(
                        gate_0, gate_target_0, reduction='none'
                    )
                    num_samples_0 = torch.tensor(batch_size, dtype=torch.float, device=self.accelerator.device)
                    gate_loss_sum_0 = gate_loss_raw_0.sum()
                else:
                    gate_loss_sum_0 = torch.nn.functional.mse_loss(
                        gate_0, gate_target_0, reduction='sum'
                    )
                    num_samples_0 = torch.tensor(batch_size, dtype=torch.float, device=self.accelerator.device)
                
                total_gate_loss_raw += gate_loss_sum_0
                total_gate_samples += num_samples_0
                
                # Update state for next steps
                hidden, gate_pred, pooled = h_0, gate_0, pooled_0
                
                # ================================================================
                # STEP 1 to actual_max_length: Recursive Steps
                # ================================================================
                for step_idx in range(0, actual_max_length):
                    valid_mask = chain_lengths > step_idx
                    if not self.is_tpu and valid_mask.sum() == 0:
                        break
                    
                    # Forward: h_i → h_{i+1}
                    hidden, gate_pred, pooled = self.model.forward(
                        x=hidden,
                        attention_mask=attention_mask,
                        is_first_step=False,
                        last_gate_score=gate_pred,
                        last_pooled=pooled,
                        gt_timestep=None,
                        sampling_prob=0.0
                    )
                    
                    # Reconstruction loss & Accuracy
                    step_targets = targets[:, step_idx, :]
                    step_loss_mask = loss_masks[:, step_idx, :]
                    
                    logits = self.model.decode(hidden, attention_mask)
                    
                    # Accuracy
                    pred_tokens = logits.argmax(dim=-1)
                    if self.is_tpu:
                        correct = (pred_tokens == step_targets).float() * step_loss_mask.float()
                        total_correct_tokens += correct.sum()
                        total_target_tokens += step_loss_mask.float().sum()
                    else:
                        if step_loss_mask.sum() > 0:
                            correct = (pred_tokens[step_loss_mask] == step_targets[step_loss_mask]).float().sum()
                            total_correct_tokens += correct
                            total_target_tokens += step_loss_mask.sum().float()
                    
                    # ✅ Token-weighted Recon Loss
                    if self.is_tpu:
                        loss_per_token = torch.nn.functional.cross_entropy(
                            logits.transpose(1, 2), step_targets, 
                            reduction='none',
                            ignore_index=self.pad_token_id  # ✅ Added
                        )
                        mask_float = step_loss_mask.float()
                        step_recon_loss_sum = (loss_per_token * mask_float).sum()
                        step_num_tokens = mask_float.sum()
                    else:
                        if step_loss_mask.sum() > 0:
                            criterion_sum = torch.nn.CrossEntropyLoss(
                                ignore_index=self.pad_token_id, 
                                reduction='sum'
                            )
                            step_recon_loss_sum = criterion_sum(
                                logits[step_loss_mask], 
                                step_targets[step_loss_mask]
                            )
                            step_num_tokens = step_loss_mask.sum().float()
                        else:
                            step_recon_loss_sum = torch.tensor(0.0, device=self.accelerator.device)
                            step_num_tokens = torch.tensor(0.0, device=self.accelerator.device)
                    
                    total_recon_loss_raw += step_recon_loss_sum
                    total_recon_tokens += step_num_tokens
                    
                    # ✅ Sample-weighted Gate loss
                    gate_target_next = gate_targets[:, step_idx + 1].unsqueeze(1)
                    
                    if self.is_tpu:
                        gate_loss_raw = torch.nn.functional.mse_loss(
                            gate_pred, gate_target_next, reduction='none'
                        )
                        valid_mask_float = valid_mask.float().unsqueeze(1)
                        step_gate_loss_sum = (gate_loss_raw * valid_mask_float).sum()
                        step_num_samples = valid_mask_float.sum()
                    else:
                        if valid_mask.sum() > 0:
                            step_gate_loss_sum = torch.nn.functional.mse_loss(
                                gate_pred[valid_mask], 
                                gate_target_next[valid_mask], 
                                reduction='sum'
                            )
                            step_num_samples = valid_mask.sum().float()
                        else:
                            step_gate_loss_sum = torch.tensor(0.0, device=self.accelerator.device)
                            step_num_samples = torch.tensor(0.0, device=self.accelerator.device)
                    
                    total_gate_loss_raw += step_gate_loss_sum
                    total_gate_samples += step_num_samples
            
            # ================================================================
            # ✅ Final Averaging (Token/Sample-weighted - MLM Standard)
            # ================================================================
            
            # Token-weighted average for reconstruction
            avg_recon = total_recon_loss_raw / (total_recon_tokens + 1e-6)
            
            # Sample-weighted average for gate
            avg_gate = total_gate_loss_raw / (total_gate_samples + 1e-6)
            
            # Token-level accuracy
            avg_accuracy = total_correct_tokens / (total_target_tokens + 1e-6)
            
            # Total loss (weighted sum)
            avg_total = self.loss_weight_recon * avg_recon + self.loss_weight_gate * avg_gate
            
            # ================================================================
            # Distributed gathering (sum raw values, then average)
            # ================================================================
            total_recon_loss_raw = self.accelerator.gather(total_recon_loss_raw).sum()
            total_recon_tokens = self.accelerator.gather(total_recon_tokens).sum()
            
            total_gate_loss_raw = self.accelerator.gather(total_gate_loss_raw).sum()
            total_gate_samples = self.accelerator.gather(total_gate_samples).sum()
            
            total_correct_tokens = self.accelerator.gather(total_correct_tokens).sum()
            total_target_tokens = self.accelerator.gather(total_target_tokens).sum()
            
            # Final averages after gathering
            if total_recon_tokens == 0 or total_gate_samples == 0:
                return 0.0, 0.0, 0.0, 0.0
            
            final_avg_recon = (total_recon_loss_raw / total_recon_tokens).item()
            final_avg_gate = (total_gate_loss_raw / total_gate_samples).item()
            final_avg_accuracy = (total_correct_tokens / total_target_tokens).item()
            final_avg_total = self.loss_weight_recon * final_avg_recon + self.loss_weight_gate * final_avg_gate
            
            return final_avg_total, final_avg_recon, final_avg_gate, final_avg_accuracy

    def _update_curriculum_progress(self, progress: float):
        """
        Update curriculum progress in collate_fn.
        
        This is the standard pattern for dynamic training parameters:
        1. Save reference to collate_fn before prepare()
        2. Update it directly during training
        
        This works across all backends (GPU/TPU/CPU) because we're
        modifying the original object, not trying to access it through wrappers.
        """
        if self.train_collate_fn is not None and hasattr(self.train_collate_fn, 'current_progress'):
            self.train_collate_fn.current_progress = progress
    
    def get_curriculum_progress(self) -> float:
        """
        Calculate current curriculum progress.
        
        Returns:
            progress: Float in [0.0, 1.0]
                     0.0 = training start
                     1.0 = training end
        """
        if self.training_mode == 'epoch':
            if self.num_epochs <= 1:
                return 1.0
            return min(1.0, self.current_epoch / (self.num_epochs - 1))
        else:  # step mode
            if self.max_training_steps <= 1:
                return 1.0
            return min(1.0, self.global_step / self.max_training_steps)
        
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
            
            # ✅ STANDARD PATTERN: Update curriculum using saved reference
            curriculum_progress = self.get_curriculum_progress()
            self._update_curriculum_progress(curriculum_progress)
            
            if self.accelerator.is_main_process:
                log_msg = f"\nEpoch {epoch + 1}/{self.num_epochs} | Sampling Prob: {sampling_prob:.3f}"
                
                # Add curriculum info if enabled
                if self.train_collate_fn and hasattr(self.train_collate_fn, 'curriculum_enabled'):
                    if self.train_collate_fn.curriculum_enabled:
                        min_step, max_step = self.train_collate_fn.get_start_step_range(curriculum_progress)
                        avg_start_step = (min_step + max_step) / 2
                        avg_mask_ratio = avg_start_step / self.train_collate_fn.total_steps
                        log_msg += f" | Curriculum: [{min_step}, {max_step}] (~{avg_mask_ratio:.1%} mask)"
                
                print(log_msg)
            
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
                    loss, recon, gate, aux, acc = self.train_step(batch)

                    # Backward & Optimizer Step (accumulate 컨텍스트가 자동 처리)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        if self.scheduler:
                            self.scheduler.step()
                        self.optimizer.zero_grad()
                
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
                    current_acc = acc.item()
                    
                    if self.accelerator.is_main_process:
                        log_data = {
                            'epoch': epoch,
                            'step': self.global_step,
                            'loss': current_loss,
                            'recon_loss': current_recon,
                            'gate_loss': current_gate,
                            'aux_loss': current_aux,
                            'accuracy': current_acc,
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
                                'acc': f'{current_acc:.3f}'
                            })
                        else:
                            # Print logging when tqdm is disabled
                            print(f"Epoch {epoch+1}/{self.num_epochs} | Step {self.global_step} | Loss: {current_loss:.4f} | Recon: {current_recon:.4f} | Gate: {current_gate:.4f} | Aux: {current_aux:.4f} | Acc: {current_acc:.3f} | LR: {log_data['lr']:.6f}")
            
            # Validation
            if (epoch + 1) % self.eval_every_n_epochs == 0:
                val_loss, val_recon, val_gate, val_acc = self.validate()
                
                if self.accelerator.is_main_process:
                    print(f"Val - Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, Gate: {val_gate:.4f}, Acc: {val_acc:.3f}")
                    
                    val_data = {
                        'epoch': epoch,
                        'step': self.global_step,
                        'val_loss': val_loss,
                        'val_recon': val_recon,
                        'val_gate': val_gate,
                        'val_accuracy': val_acc
                    }
                    self.csv_logger.log(val_data)
                    
                    if self.use_wandb:
                        self.accelerator.log({
                            'val/loss': val_loss,
                            'val/recon_loss': val_recon,
                            'val/gate_loss': val_gate,
                            'val/accuracy': val_acc,
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
            
            # ✅ STANDARD PATTERN: Update curriculum using saved reference
            curriculum_progress = self.get_curriculum_progress()
            self._update_curriculum_progress(curriculum_progress)
            
            if self.accelerator.is_main_process:
                log_msg = f"\nEpoch {epoch + 1} | Step {step}/{self.max_training_steps} | Sampling Prob: {sampling_prob:.3f}"
                
                # Add curriculum info if enabled
                if self.train_collate_fn and hasattr(self.train_collate_fn, 'curriculum_enabled'):
                    if self.train_collate_fn.curriculum_enabled:
                        min_step, max_step = self.train_collate_fn.get_start_step_range(curriculum_progress)
                        avg_start_step = (min_step + max_step) / 2
                        avg_mask_ratio = avg_start_step / self.train_collate_fn.total_steps
                        log_msg += f" | Curriculum: [{min_step}, {max_step}] (~{avg_mask_ratio:.1%} mask)"
                
                print(log_msg)
                
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
                    loss, recon, gate, aux, acc = self.train_step(batch)
                    
                    # Backward & Optimizer Step (accumulate 컨텍스트가 자동 처리)
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        if self.scheduler:
                            self.scheduler.step()
                        self.optimizer.zero_grad()
                
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
                    current_acc = acc.item()

                    log_data = {
                        'epoch': epoch,
                        'step': step,
                        'loss': current_loss,
                        'recon_loss': current_recon,
                        'gate_loss': current_gate,
                        'aux_loss': current_aux,
                        'accuracy': current_acc,
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
                            'acc': f'{current_acc:.3f}'
                        })
                    else:
                        # Print logging when tqdm is disabled
                        print(f"Epoch {epoch+1} | Step {step}/{self.max_training_steps} | Loss: {current_loss:.4f} | Recon: {current_recon:.4f} | Gate: {current_gate:.4f} | Aux: {current_aux:.4f} | Acc: {current_acc:.3f} | LR: {log_data['lr']:.6f}")
                
                # Validation at regular step intervals
                if step % self.eval_every_n_steps == 0:
                    val_loss, val_recon, val_gate, val_acc = self.validate()
                    
                    if self.accelerator.is_main_process:
                        print(f"\nStep {step} - Val Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, Gate: {val_gate:.4f}, Acc: {val_acc:.3f}")
                        
                        val_data = {
                            'epoch': epoch,
                            'step': step,
                            'val_loss': val_loss,
                            'val_recon': val_recon,
                            'val_gate': val_gate,
                            'val_accuracy': val_acc
                        }
                        self.csv_logger.log(val_data)
                        
                        if self.use_wandb:
                            self.accelerator.log({
                                'val/loss': val_loss,
                                'val/recon_loss': val_recon,
                                'val/gate_loss': val_gate,
                                'val/accuracy': val_acc,
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
