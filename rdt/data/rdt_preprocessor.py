import torch
import torch.nn as nn

class RDTPreprocessor(nn.Module):
    """
    RDT 데이터 생성 로직을 GPU에서 배치 단위로 수행하는 모듈.
    CPU 부하와 PCI-e 병목을 제거함.
    """
    def __init__(self, tokenizer, config):
        super().__init__()
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.mask_token_id = tokenizer.mask_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.vocab_size = tokenizer.vocab_size
        
        self.max_chain_length = config['training']['max_chain_length']
        self.total_steps = config['training']['total_steps']
        self.visible_loss_ratio = config['training'].get('visible_loss_ratio', 0.15)
        
        # BERT Masking Config
        bert_config = config['training'].get('bert_masking', {})
        self.bert_masking_enabled = bert_config.get('enabled', True)
        self.mask_prob = bert_config.get('mask_prob', 0.8)
        self.random_prob = bert_config.get('random_prob', 0.1)

    @torch.no_grad()
    def forward(self, input_ids):
        """
        Args:
            input_ids: [Batch, SeqLen] - 원본 토큰 (GPU Tensor)
        Returns:
            processed_batch: Dict - RDT 모델 입력 형태
        """
        device = input_ids.device
        B, L = input_ids.shape
        
        # 1. Chain Length & Start Step
        chain_lengths = torch.randint(1, self.max_chain_length + 1, (B,), device=device)
        max_chain_len_in_batch = chain_lengths.max().item()
        
        min_starts = chain_lengths
        ranges = self.total_steps - min_starts + 1
        start_steps = min_starts + torch.rand(B, device=device).mul(ranges).long()
        
        # 2. Random Permutation Logic (GPU Friendly)
        rand_scores = torch.rand(B, L, device=device)
        # 패딩 토큰은 항상 가장 나중에 복원되도록(또는 마스킹 안되도록) 점수 조작 가능하지만
        # 여기서는 attention_mask로 처리되므로 단순 랜덤
        restore_ranks = rand_scores.argsort(dim=1).argsort(dim=1)
        
        # 3. Init Tensors
        inputs = torch.full((B, L), self.pad_token_id, dtype=torch.long, device=device)
        targets = torch.full((B, max_chain_len_in_batch, L), self.pad_token_id, dtype=torch.long, device=device)
        loss_masks = torch.zeros((B, max_chain_len_in_batch, L), dtype=torch.bool, device=device)
        gate_targets = torch.zeros((B, max_chain_len_in_batch + 1), dtype=torch.float, device=device)
        attention_mask = (input_ids != self.pad_token_id).long()
        
        # 4. Input Generation (s_0)
        input_steps = start_steps
        visible_ratios = 1.0 - (input_steps.float() / self.total_steps)
        visible_counts = (visible_ratios * L).long()
        
        is_masked = restore_ranks >= visible_counts.unsqueeze(1)
        is_masked = is_masked & (attention_mask.bool()) # 패딩은 마스킹 제외
        
        inputs = input_ids.clone()
        if self.bert_masking_enabled:
            mask_token_mask = (torch.rand(B, L, device=device) < self.mask_prob) & is_masked
            random_token_mask = (torch.rand(B, L, device=device) < (self.mask_prob + self.random_prob)) & is_masked & ~mask_token_mask
            
            inputs[mask_token_mask] = self.mask_token_id
            if random_token_mask.any():
                random_tokens = torch.randint(self.cls_token_id + 1, self.vocab_size, (random_token_mask.sum(),), device=device)
                inputs[random_token_mask] = random_tokens
        else:
            inputs[is_masked] = self.mask_token_id
            
        gate_targets[:, 0] = (input_steps.float() / self.total_steps) * 20.0
        
        # 5. Chain Generation
        for i in range(max_chain_len_in_batch):
            valid_samples = chain_lengths > i
            if not valid_samples.any():
                break
                
            # 타겟 스텝 계산
            target_steps = start_steps - (i + 1)
            
            # Loss Mask 계산 (현재 스텝에서는 안 보였지만 다음 스텝에 보여야 할 구간)
            curr_visible_counts = ((1.0 - (start_steps - i).float() / self.total_steps) * L).long()
            target_visible_counts = ((1.0 - target_steps.float() / self.total_steps) * L).long()
            
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
            step_targets = input_ids.clone()
            still_masked = restore_ranks >= upper_bound
            step_targets[still_masked] = self.mask_token_id
            
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