import torch
import torch.nn as nn
from typing import Dict, List

class RDTPreprocessor:
    """
    RDT 데이터 생성 로직을 수행하는 Collator.
    DataLoader의 collate_fn으로 사용되어 배치 단위로 전처리를 수행.
    
    ⚠️ DEBUG VERSION: 2번 구현의 단순 로직 사용 (special token 처리 없음)
    """
    def __init__(self, tokenizer, config, device='cpu'):
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.mask_token_id = tokenizer.mask_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.vocab_size = tokenizer.vocab_size
        self.device = device
        
        # Config 파싱 (1번과 동일)
        self.min_chain_length = config['training'].get('min_chain_length', 1)
        self.max_chain_length = config['training']['max_chain_length']
        self.total_steps = config['training']['total_steps']
        self.visible_loss_ratio = config['training'].get('visible_loss_ratio', 0.15)
        
        # BERT Masking Config
        bert_config = config['training'].get('bert_masking', {})
        self.bert_masking_enabled = bert_config.get('enabled', True)
        self.mask_prob = bert_config.get('mask_prob', 0.8)
        self.random_prob = bert_config.get('random_prob', 0.1)
        
        # Curriculum Learning Config (파싱만 하고 사용 안 함)
        curriculum_config = config['training'].get('curriculum', {})
        self.curriculum_enabled = curriculum_config.get('enabled', False)
        self.curriculum_start_step = curriculum_config.get('start_step', self.total_steps)
        
        # Current progress (not used in this version)
        self.current_progress = 0.0

    @torch.no_grad()
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([item['input_ids'] for item in batch])
        device = self.device
        input_ids = input_ids.to(device)
        
        B, L = input_ids.shape
        
        # ========================================================================
        # 2번 스타일: Special Token 처리 없음
        # ========================================================================
        
        # 1. Chain Length (배치 내 가변)
        chain_lengths = torch.randint(self.min_chain_length, self.max_chain_length + 1, (B,), device=device)
        
        # 2. Start Step Sampling (Curriculum 없이 단순 샘플링)
        min_starts = chain_lengths
        ranges = self.total_steps - min_starts + 1
        start_steps = min_starts + torch.rand(B, device=device).mul(ranges).long()
        
        # 3. Random Permutation Logic (Special token 구분 없음)
        rand_scores = torch.rand(B, L, device=device)
        restore_ranks = rand_scores.argsort(dim=1).argsort(dim=1)
        
        # 4. Init Tensors
        inputs = torch.full((B, L), self.pad_token_id, dtype=torch.long, device=device)
        targets = torch.full((B, self.max_chain_length, L), self.pad_token_id, dtype=torch.long, device=device)
        loss_masks = torch.zeros((B, self.max_chain_length, L), dtype=torch.bool, device=device)
        gate_targets = torch.zeros((B, self.max_chain_length + 1), dtype=torch.float, device=device)
        attention_mask = (input_ids != self.pad_token_id).long()
        
        # 5. Input Generation (s_0)
        input_steps = start_steps
        visible_ratios = 1.0 - (input_steps.float() / self.total_steps)
        # ⚠️ 2번 스타일: seq_len 전체 사용 (special token 구분 없음)
        visible_counts = (visible_ratios * L).long()
        
        is_masked = restore_ranks >= visible_counts.unsqueeze(1)
        
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
            
            # ⚠️ 2번 스타일: L 전체 사용
            curr_visible_counts = (curr_visible_ratios * L).long()
            target_visible_counts = (target_visible_ratios * L).long()

            target_visible_counts = torch.maximum(
                curr_visible_counts + 1,
                target_visible_counts
            )
            
            lower_bound = curr_visible_counts.unsqueeze(1)
            upper_bound = target_visible_counts.unsqueeze(1)
            
            step_loss_mask = (restore_ranks >= lower_bound) & (restore_ranks < upper_bound)
            
            if self.visible_loss_ratio > 0:
                already_visible = restore_ranks < lower_bound
                rehearsal_mask = (torch.rand(B, L, device=device) < self.visible_loss_ratio) & already_visible
                step_loss_mask = step_loss_mask | rehearsal_mask
            
            # ⚠️ 2번 스타일: Special token 제외 없음
            
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