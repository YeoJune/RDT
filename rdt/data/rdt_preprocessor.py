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
        
        self.max_chain_length = config['training']['max_chain_length']
        self.total_steps = config['training']['total_steps']
        self.visible_loss_ratio = config['training'].get('visible_loss_ratio', 0.15)
        
        # BERT Masking Config
        bert_config = config['training'].get('bert_masking', {})
        self.bert_masking_enabled = bert_config.get('enabled', True)
        self.mask_prob = bert_config.get('mask_prob', 0.8)
        self.random_prob = bert_config.get('random_prob', 0.1)

    @torch.no_grad()
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collator function that processes a batch of samples.
        
        Args:
            batch: List of dicts with 'input_ids' key
        Returns:
            processed_batch: Dict - RDT 모델 입력 형태
        """
        # Stack input_ids from batch
        input_ids = torch.stack([item['input_ids'] for item in batch])
        device = self.device
        input_ids = input_ids.to(device)
        
        B, L = input_ids.shape
        
        # 1. Chain Length & Start Step
        chain_lengths = torch.randint(1, self.max_chain_length + 1, (B,), device=device)
        max_chain_len_in_batch = self.max_chain_length #chain_lengths.max().item()
        
        min_starts = chain_lengths
        ranges = self.total_steps - min_starts + 1

        max_start = 5  # 이 값 조정
        ranges = torch.clamp(ranges, max=max_start - min_starts + 1)

        start_steps = min_starts + torch.rand(B, device=device).mul(ranges).long()
        
        # 2. Random Permutation Logic (GPU Friendly)
        rand_scores = torch.rand(B, L, device=device)
        
        # 패딩 토큰은 아주 큰 값으로 설정하여 항상 마지막 랭크를 받도록
        # 이렇게 하면 visible_counts 계산이 실제 토큰만 기준으로 정확하게 작동
        padding_mask = (input_ids == self.pad_token_id)
        rand_scores = torch.where(padding_mask, torch.tensor(1e9, device=device), rand_scores)
        
        restore_ranks = rand_scores.argsort(dim=1).argsort(dim=1)
        
        # 3. Init Tensors
        inputs = torch.full((B, L), self.pad_token_id, dtype=torch.long, device=device)
        targets = torch.full((B, max_chain_len_in_batch, L), self.pad_token_id, dtype=torch.long, device=device)
        loss_masks = torch.zeros((B, max_chain_len_in_batch, L), dtype=torch.bool, device=device)
        gate_targets = torch.zeros((B, max_chain_len_in_batch + 1), dtype=torch.float, device=device)
        attention_mask = (input_ids != self.pad_token_id).long()
        
        # 실제 토큰 개수 계산 (패딩 제외)
        # [B] - 각 샘플의 실제 토큰 개수
        actual_lengths = attention_mask.sum(dim=1).float()
        
        # 4. Input Generation (s_0)
        input_steps = start_steps
        visible_ratios = 1.0 - (input_steps.float() / self.total_steps)
        # 실제 토큰 개수 기준으로 visible_counts 계산 [B]
        visible_counts = (visible_ratios * actual_lengths).long()
        
        is_masked = restore_ranks >= visible_counts.unsqueeze(1)
        is_masked = is_masked & (attention_mask.bool()) # 패딩은 마스킹 제외
        
        inputs = input_ids.clone()
        if self.bert_masking_enabled:
            mask_token_mask = (torch.rand(B, L, device=device) < self.mask_prob) & is_masked
            random_token_mask = (torch.rand(B, L, device=device) < (self.mask_prob + self.random_prob)) & is_masked & ~mask_token_mask
            
            # TPU 최적화: Boolean indexing 대신 torch.where 사용 (Shape 고정)
            inputs = torch.where(mask_token_mask, torch.tensor(self.mask_token_id, device=device), inputs)
            
            # Random 토큰 적용: 전체 크기 텐서를 미리 생성하여 dynamic shape 방지
            random_tokens_full = torch.randint(self.cls_token_id + 1, self.vocab_size, (B, L), device=device)
            inputs = torch.where(random_token_mask, random_tokens_full, inputs)
        else:
            inputs = torch.where(is_masked, torch.tensor(self.mask_token_id, device=device), inputs)
            
        gate_targets[:, 0] = (input_steps.float() / self.total_steps) * 20.0
        
        # 5. Chain Generation
        # TPU 최적화: break 없이 고정 횟수 루프 실행
        # valid_samples가 False인 경우 loss_masks가 0이 되어 학습에 영향 없음
        for i in range(max_chain_len_in_batch):
            valid_samples = chain_lengths > i
                
            # 타겟 스텝 계산
            target_steps = start_steps - (i + 1)
            
            # Loss Mask 계산 (현재 스텝에서는 안 보였지만 다음 스텝에 보여야 할 구간)
            # 실제 토큰 개수 기준으로 계산
            curr_visible_ratios = 1.0 - (start_steps - i).float() / self.total_steps
            target_visible_ratios = 1.0 - target_steps.float() / self.total_steps
            
            # [B] shape - 각 샘플별 visible 토큰 개수
            curr_visible_counts = (curr_visible_ratios * actual_lengths).long()
            target_visible_counts = (target_visible_ratios * actual_lengths).long()
            
            # [B, 1] shape로 확장하여 [B, L]과 비교 가능하도록
            lower_bound = curr_visible_counts.unsqueeze(1)
            upper_bound = target_visible_counts.unsqueeze(1)
            
            # 기본 복원 구간
            step_loss_mask = (restore_ranks >= lower_bound) & (restore_ranks < upper_bound)
            
            # Rehearsal (Visible Loss)
            if self.visible_loss_ratio > 0:
                already_visible = restore_ranks < lower_bound
                rehearsal_mask = (torch.rand(B, L, device=device) < self.visible_loss_ratio) & already_visible
                step_loss_mask = step_loss_mask | rehearsal_mask
            
            # 패딩 부분은 Loss 마스크 제외
            step_loss_mask = step_loss_mask & attention_mask.bool()
            
            # Target Tokens
            # 아직 마스킹되어야 할 토큰은 [MASK]로, 복원된 토큰은 원본 유지
            still_masked = restore_ranks >= upper_bound
            step_targets = torch.where(
                still_masked,
                torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
                input_ids
            )
            
            # Assign
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