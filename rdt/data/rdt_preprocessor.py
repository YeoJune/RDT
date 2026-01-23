import torch
import torch.nn as nn
from typing import Dict, Optional

class RDTPreprocessor:
    """
    RDT 마스킹 로직을 제공하는 유틸리티 클래스.
    개별 샘플 단위로 처리하며, Collator가 이를 사용.
    
    Version 2 Style:
    - Individual sample processing (not batch)
    - Used by RDTCollator for encapsulation
    - Permutation-based indexing (like v2)
    - Worker-local processing
    """
    def __init__(self, tokenizer, config):
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.mask_token_id = tokenizer.mask_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.vocab_size = tokenizer.vocab_size
        
        # ✅ KEEP: Store all special token IDs (from v1)
        self.all_special_ids = set(tokenizer.all_special_ids)
        
        self.min_chain_length = config['training'].get('min_chain_length', 1)
        self.max_chain_length = config['training']['max_chain_length']
        self.total_steps = config['training']['total_steps']
        self.visible_loss_ratio = config['training'].get('visible_loss_ratio', 0.15)
        
        # BERT Masking Config
        bert_config = config['training'].get('bert_masking', {})
        self.bert_masking_enabled = bert_config.get('enabled', True)
        self.mask_prob = bert_config.get('mask_prob', 0.8)
        self.random_prob = bert_config.get('random_prob', 0.1)
    
    def _create_special_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create mask for ALL special tokens (from v1).
        
        Args:
            input_ids: (L,) token tensor
        
        Returns:
            special_mask: (L,) bool tensor, True for special tokens
        """
        special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for special_id in self.all_special_ids:
            special_mask |= (input_ids == special_id)
        return special_mask
    
    def _apply_bert_masking(self, input_ids: torch.Tensor, mask_indices: torch.Tensor) -> torch.Tensor:
        """
        Apply BERT-style masking to specific indices.
        80% [MASK], 10% random, 10% keep original.
        
        Args:
            input_ids: (L,) token tensor (will be modified in-place)
            mask_indices: (M,) indices to mask
        
        Returns:
            Modified input_ids
        """
        if len(mask_indices) == 0 or not self.bert_masking_enabled:
            input_ids[mask_indices] = self.mask_token_id
            return input_ids
        
        rand = torch.rand(len(mask_indices))
        
        # 80%: [MASK]
        mask_token_indices = mask_indices[rand < self.mask_prob]
        input_ids[mask_token_indices] = self.mask_token_id
        
        # 10%: random token
        random_token_indices = mask_indices[
            (rand >= self.mask_prob) & (rand < self.mask_prob + self.random_prob)
        ]
        if len(random_token_indices) > 0:
            random_tokens = torch.randint(
                self.cls_token_id + 1, 
                self.vocab_size, 
                (len(random_token_indices),)
            )
            input_ids[random_token_indices] = random_tokens
        
        # 10%: keep original (do nothing)
        
        return input_ids
    
    def process_single_sample(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process a single sample to generate RDT training data.
        Version 2 style: individual sample, permutation-based.
        
        Args:
            input_ids: (L,) raw token tensor
        
        Returns:
            Dict with keys: 'input', 'targets', 'loss_masks', 'gate_targets', 'chain_length'
        """
        seq_len = len(input_ids)
        
        # ✅ Special token handling (from v1)
        special_mask = self._create_special_token_mask(input_ids)
        non_special_mask = ~special_mask
        actual_length = non_special_mask.sum().item()
        
        # ✅ Version 2 style: Random permutation for non-special tokens only
        # Create permutation that only includes non-special token positions
        non_special_positions = torch.where(non_special_mask)[0]  # Indices of non-special tokens
        perm = torch.randperm(len(non_special_positions))
        restore_order = non_special_positions[perm]  # Permuted non-special positions
        
        # Sample chain length (v2 style)
        chain_length = torch.randint(self.min_chain_length, self.max_chain_length + 1, (1,)).item()
        
        # Sample start step (v2 style: no curriculum)
        min_start = chain_length
        max_start = self.total_steps
        start_step = torch.randint(min_start, max_start + 1, (1,)).item()
        
        # ========================================================================
        # Generate Input (s_0)
        # ========================================================================
        input_step = start_step
        visible_ratio = 1.0 - (input_step / self.total_steps)
        num_visible = int(visible_ratio * actual_length)
        
        # Visible and masked indices (from permuted non-special positions)
        visible_indices = restore_order[:num_visible]
        masked_indices = restore_order[num_visible:]
        
        # Create input with masking
        inputs = input_ids.clone()
        if len(masked_indices) > 0:
            inputs = self._apply_bert_masking(inputs, masked_indices)
        
        # ========================================================================
        # Generate Chain (targets and loss_masks)
        # ========================================================================
        targets = []
        loss_masks = []
        gate_targets = [input_step / self.total_steps * 20.0]
        
        for i in range(chain_length):
            target_step = start_step - (i + 1)
            
            curr_visible_ratio = 1.0 - (start_step - i) / self.total_steps
            target_visible_ratio = 1.0 - target_step / self.total_steps
            
            curr_num_visible = int(curr_visible_ratio * actual_length)
            target_num_visible = int(target_visible_ratio * actual_length)
            
            # Ensure monotonic increase
            target_num_visible = max(curr_num_visible + 1, target_num_visible)
            
            # Create target (mask tokens beyond target_num_visible)
            target_visible_indices = restore_order[:target_num_visible]
            target_masked_indices = restore_order[target_num_visible:]
            
            target_ids = input_ids.clone()
            if len(target_masked_indices) > 0:
                target_ids[target_masked_indices] = self.mask_token_id
            
            targets.append(target_ids)
            
            # Create loss mask (delta region)
            delta_indices = restore_order[curr_num_visible:target_num_visible]
            
            loss_mask = torch.zeros(seq_len, dtype=torch.bool)
            if len(delta_indices) > 0:
                loss_mask[delta_indices] = True
            
            # Visible loss (rehearsal)
            if curr_num_visible > 0 and self.visible_loss_ratio > 0:
                num_rehearsal = max(1, int(curr_num_visible * self.visible_loss_ratio))
                rehearsal_perm = torch.randperm(curr_num_visible)[:num_rehearsal]
                rehearsal_indices = restore_order[rehearsal_perm]
                loss_mask[rehearsal_indices] = True
            
            loss_masks.append(loss_mask)
            gate_targets.append(target_step / self.total_steps * 20.0)
        
        return {
            'input': inputs,
            'targets': torch.stack(targets),
            'loss_masks': torch.stack(loss_masks),
            'gate_targets': torch.tensor(gate_targets, dtype=torch.float),
            'chain_length': chain_length
        }