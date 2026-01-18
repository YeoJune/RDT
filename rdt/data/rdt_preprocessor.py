import torch
import torch.nn as nn
from typing import Dict, List

class RDTPreprocessor:
    """
    RDT 데이터 생성 로직을 수행하는 Collator.
    DataLoader의 collate_fn으로 사용되어 배치 단위로 전처리를 수행.
    TPU 최적화: CPU에서 동작하며 고정 크기 텐서를 생성.
    """
    def __init__(self, tokenizer, config, device='cpu'):
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.mask_token_id = tokenizer.mask_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.vocab_size = tokenizer.vocab_size
        self.device = device
        
        # ✅ BEST PRACTICE: Store all special token IDs
        self.all_special_ids = set(tokenizer.all_special_ids)
        
        self.max_chain_length = config['training']['max_chain_length']
        self.total_steps = config['training']['total_steps']
        self.visible_loss_ratio = config['training'].get('visible_loss_ratio', 0.15)
        
        # BERT Masking Config
        bert_config = config['training'].get('bert_masking', {})
        self.bert_masking_enabled = bert_config.get('enabled', True)
        self.mask_prob = bert_config.get('mask_prob', 0.8)
        self.random_prob = bert_config.get('random_prob', 0.1)
        
        # Curriculum Learning Config
        curriculum_config = config['training'].get('curriculum', {})
        self.curriculum_enabled = curriculum_config.get('enabled', False)
        self.curriculum_start_step = curriculum_config.get('start_step', self.total_steps)
        
        # Current progress (updated by trainer)
        self.current_progress = 0.0
    
    def get_start_step_range(self, progress: float) -> tuple:
        """
        Get allowed start_step range based on curriculum progress.
        
        Args:
            progress: Training progress in [0.0, 1.0]
        
        Returns:
            (min_step, max_step): Allowed range for start_step sampling
        
        Examples (total_steps=20, start_step=5, max_chain_length=2):
            progress=0.0 → (2, 5)    # Narrow range (20~25% mask)
            progress=0.5 → (2, 12)   # Expanding range (10~90% mask)
            progress=1.0 → (2, 20)   # Full range (0~90% mask)
        
        Critical: start_step MUST be >= max_chain_length for curriculum to work!
        """
        if not self.curriculum_enabled:
            return (self.max_chain_length, self.total_steps)
        
        # Min: Always fixed at max_chain_length
        min_step = self.max_chain_length
        
        # Max: Linear interpolation from start_step to total_steps
        max_step = self.curriculum_start_step + progress * (
            self.total_steps - self.curriculum_start_step
        )
        
        # Clamp max_step to be at least min_step
        max_step = max(min_step, int(max_step))
        
        return (min_step, max_step)

    @torch.no_grad()
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([item['input_ids'] for item in batch])
        device = self.device
        input_ids = input_ids.to(device)
        
        B, L = input_ids.shape
        
        # ========================================================================
        # [FIX] Special Token Identification using all_special_ids
        # ========================================================================
        # Create mask for ALL special tokens (more robust and standard)
        special_token_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
        for special_id in self.all_special_ids:
            special_token_mask |= (input_ids == special_id)
        
        # Non-special token mask for actual length calculation
        non_special_mask = ~special_token_mask
        
        # 1. Chain Length
        chain_lengths = torch.randint(1, self.max_chain_length + 1, (B,), device=device)
        
        # 2. Start Step Sampling with Curriculum
        if self.curriculum_enabled:
            curriculum_min, curriculum_max = self.get_start_step_range(self.current_progress)
            
            # Respect chain_length constraint
            min_starts = torch.maximum(
                chain_lengths,
                torch.tensor(curriculum_min, device=device)
            )
            max_starts = torch.full((B,), curriculum_max, device=device)
            max_starts = torch.maximum(max_starts, min_starts)  # Ensure min <= max
            
            # Uniform sampling in [min_starts, max_starts]
            ranges = max_starts - min_starts + 1
            start_steps = min_starts + torch.rand(B, device=device).mul(ranges).long()
        else:
            # Original logic: uniform sampling in [chain_length, total_steps]
            min_starts = chain_lengths
            ranges = self.total_steps - min_starts + 1
            start_steps = min_starts + torch.rand(B, device=device).mul(ranges).long()
        
        # 3. Random Permutation Logic
        rand_scores = torch.rand(B, L, device=device)
        # ✅ FIX: Exclude ALL special tokens from ranking
        rand_scores = torch.where(
            special_token_mask, 
            torch.tensor(1e9, device=device), 
            rand_scores
        )
        restore_ranks = rand_scores.argsort(dim=1).argsort(dim=1)
        
        # 4. Init Tensors
        inputs = torch.full((B, L), self.pad_token_id, dtype=torch.long, device=device)
        targets = torch.full((B, self.max_chain_length, L), self.pad_token_id, dtype=torch.long, device=device)
        loss_masks = torch.zeros((B, self.max_chain_length, L), dtype=torch.bool, device=device)
        gate_targets = torch.zeros((B, self.max_chain_length + 1), dtype=torch.float, device=device)
        attention_mask = (input_ids != self.pad_token_id).long()
        
        # ✅ FIX: Count only non-special tokens for actual length
        actual_lengths = non_special_mask.sum(dim=1).float()
        
        # 5. Input Generation (s_0)
        input_steps = start_steps
        visible_ratios = 1.0 - (input_steps.float() / self.total_steps)
        visible_counts = (visible_ratios * actual_lengths).long()
        
        is_masked = restore_ranks >= visible_counts.unsqueeze(1)
        # ✅ FIX: Only mask non-special tokens
        is_masked = is_masked & non_special_mask
        
        inputs = input_ids.clone()
        if self.bert_masking_enabled:
            mask_token_mask = (torch.rand(B, L, device=device) < self.mask_prob) & is_masked
            random_token_mask = (torch.rand(B, L, device=device) < (self.mask_prob + self.random_prob)) & is_masked & ~mask_token_mask
            
            inputs = torch.where(mask_token_mask, torch.tensor(self.mask_token_id, device=device), inputs)
            random_tokens_full = torch.randint(self.cls_token_id + 1, self.vocab_size, (B, L), device=device)
            inputs = torch.where(random_token_mask, random_tokens_full, inputs)
        else:
            inputs = torch.where(is_masked, torch.tensor(self.mask_token_id, device=device), inputs)
            
        gate_targets[:, 0] = (input_steps.float() / self.total_steps) * 20.0
        
        # 6. Chain Generation
        for i in range(self.max_chain_length):
            valid_samples = chain_lengths > i
            target_steps = start_steps - (i + 1)
            
            curr_visible_ratios = 1.0 - (start_steps - i).float() / self.total_steps
            target_visible_ratios = 1.0 - target_steps.float() / self.total_steps
            
            curr_visible_counts = (curr_visible_ratios * actual_lengths).long()
            target_visible_counts = (target_visible_ratios * actual_lengths).long()
            
            lower_bound = curr_visible_counts.unsqueeze(1)
            upper_bound = target_visible_counts.unsqueeze(1)
            
            step_loss_mask = (restore_ranks >= lower_bound) & (restore_ranks < upper_bound)
            
            if self.visible_loss_ratio > 0:
                already_visible = restore_ranks < lower_bound
                rehearsal_mask = (torch.rand(B, L, device=device) < self.visible_loss_ratio) & already_visible
                step_loss_mask = step_loss_mask | rehearsal_mask
            
            # ✅ FIX: Exclude special tokens from loss mask
            step_loss_mask = step_loss_mask & non_special_mask
            
            still_masked = restore_ranks >= upper_bound
            step_targets = torch.where(
                still_masked,
                torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
                input_ids
            )
            
            targets[:, i, :] = step_targets
            loss_masks[:, i, :] = step_loss_mask & valid_samples.unsqueeze(1)
            gate_targets[:, i + 1] = (target_steps.float() / self.total_steps) * 20.0
            
        return {
            'input': inputs,
            'targets': targets,
            'loss_masks': loss_masks,
            'gate_targets': gate_targets,
            'attention_mask': attention_mask,
            'chain_lengths': chain_lengths
        }