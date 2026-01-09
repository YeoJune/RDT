import torch
import torch.nn as nn
from typing import Dict, List

class RDTPreprocessor:
    """
    RDT 데이터 생성 로직을 수행하는 Collator.
    명확성을 위해 최적화보다 정확성과 해석 가능성에 집중.
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
        명확하고 해석 가능한 데이터 생성 로직.
        
        핵심 아이디어:
        1. 실제 토큰들만 추출
        2. 복원 순서 배열 생성 (0-indexed, dense)
        3. 순서 배열 기반으로 마스킹 및 타겟 생성
        """
        input_ids = torch.stack([item['input_ids'] for item in batch])
        device = self.device
        input_ids = input_ids.to(device)
        
        B, L = input_ids.shape
        attention_mask = (input_ids != self.pad_token_id).long()
        
        # ================================================================
        # Step 1: Chain length & Start step 결정
        # ================================================================
        chain_lengths = torch.randint(1, self.max_chain_length + 1, (B,), device=device)
        
        min_starts = chain_lengths
        ranges = self.total_steps - min_starts + 1
        start_steps = min_starts + torch.rand(B, device=device).mul(ranges).long()
        
        # ================================================================
        # Step 2: 각 샘플별로 복원 순서 배열 생성
        # ================================================================
        # restore_order[b, i] = i번째로 복원될 실제 토큰의 position
        # 예: restore_order[0] = [5, 2, 8, 1, ...] 
        #     → Position 5가 가장 먼저, Position 2가 두 번째, ...
        
        restore_order = []  # List of tensors, each [actual_length]
        actual_lengths = []  # List of ints
        
        for b in range(B):
            # 실제 토큰 위치들
            real_positions = attention_mask[b].nonzero(as_tuple=True)[0]  # [actual_length]
            actual_length = len(real_positions)
            actual_lengths.append(actual_length)
            
            # 랜덤 순서 생성
            perm = torch.randperm(actual_length, device=device)
            
            # restore_order: i번째로 복원될 position
            order = real_positions[perm]  # [actual_length]
            restore_order.append(order)
        
        # ================================================================
        # Step 3: 텐서 초기화
        # ================================================================
        max_chain_len = self.max_chain_length
        
        inputs = torch.full((B, L), self.pad_token_id, dtype=torch.long, device=device)
        targets = torch.full((B, max_chain_len, L), self.pad_token_id, dtype=torch.long, device=device)
        loss_masks = torch.zeros((B, max_chain_len, L), dtype=torch.bool, device=device)
        gate_targets = torch.zeros((B, max_chain_len + 1), dtype=torch.float, device=device)
        
        # ================================================================
        # Step 4: 각 샘플별로 Input 생성 (Step 0)
        # ================================================================
        for b in range(B):
            actual_length = actual_lengths[b]
            order = restore_order[b]  # [actual_length]
            
            # 현재 step에서 보이는 토큰 개수
            visible_ratio = 1.0 - (start_steps[b].float() / self.total_steps)
            num_visible = int(visible_ratio * actual_length)
            
            # 가장 먼저 복원되는 num_visible개 토큰들
            visible_positions = order[:num_visible]  # [num_visible]
            masked_positions = order[num_visible:]   # [actual_length - num_visible]
            
            # Input 생성: pad_token으로 초기화
            sample_input = torch.full((L,), self.pad_token_id, dtype=torch.long, device=device)
            
            # 보이는 위치에 원본 토큰 배치
            if len(visible_positions) > 0:
                sample_input[visible_positions] = input_ids[b, visible_positions]
            
            # 마스킹된 위치 처리
            if len(masked_positions) > 0:
                if self.bert_masking_enabled:
                    # BERT-style masking: 80% [MASK], 10% random, 10% keep
                    num_masked = len(masked_positions)
                    rand = torch.rand(num_masked, device=device)
                    
                    # 80% [MASK]
                    mask_indices = masked_positions[rand < 0.8]
                    if len(mask_indices) > 0:
                        sample_input[mask_indices] = self.mask_token_id
                    
                    # 10% random token
                    random_indices = masked_positions[(rand >= 0.8) & (rand < 0.9)]
                    if len(random_indices) > 0:
                        random_tokens = torch.randint(
                            self.cls_token_id + 1, 
                            self.vocab_size, 
                            (len(random_indices),), 
                            device=device
                        )
                        sample_input[random_indices] = random_tokens
                    
                    # 10% keep original
                    keep_indices = masked_positions[rand >= 0.9]
                    if len(keep_indices) > 0:
                        sample_input[keep_indices] = input_ids[b, keep_indices]
                else:
                    # 단순 masking: 전부 [MASK]
                    sample_input[masked_positions] = self.mask_token_id
            
            inputs[b] = sample_input
            
            # Gate target
            gate_targets[b, 0] = (start_steps[b].float() / self.total_steps) * 20.0
        
        # ================================================================
        # Step 5: Chain 생성
        # ================================================================
        for step_idx in range(max_chain_len):
            for b in range(B):
                # 이 샘플의 chain이 step_idx까지 유효한지
                if chain_lengths[b] <= step_idx:
                    continue
                
                actual_length = actual_lengths[b]
                order = restore_order[b]
                
                # 현재 step과 타겟 step
                current_step = start_steps[b] - step_idx
                target_step = start_steps[b] - (step_idx + 1)
                
                # 현재 보이는 개수와 다음에 보일 개수
                curr_visible_ratio = 1.0 - (current_step.float() / self.total_steps)
                target_visible_ratio = 1.0 - (target_step.float() / self.total_steps)
                
                curr_num_visible = int(curr_visible_ratio * actual_length)
                target_num_visible = int(target_visible_ratio * actual_length)
                
                # ============================================================
                # Loss Mask 생성
                # ============================================================
                step_loss_mask = torch.zeros(L, dtype=torch.bool, device=device)
                
                # 이번 step에서 새로 복원할 토큰들
                newly_visible_positions = order[curr_num_visible:target_num_visible]
                if len(newly_visible_positions) > 0:
                    step_loss_mask[newly_visible_positions] = True
                
                # Rehearsal: 이미 보이는 토큰 중 일부도 loss 계산
                if self.visible_loss_ratio > 0 and curr_num_visible > 0:
                    already_visible_positions = order[:curr_num_visible]
                    num_rehearsal = int(self.visible_loss_ratio * len(already_visible_positions))
                    
                    if num_rehearsal > 0:
                        rehearsal_indices = torch.randperm(len(already_visible_positions), device=device)[:num_rehearsal]
                        rehearsal_positions = already_visible_positions[rehearsal_indices]
                        step_loss_mask[rehearsal_positions] = True
                
                # ============================================================
                # Target 생성
                # ============================================================
                # 초기화: 전부 패딩
                step_target = torch.full((L,), self.pad_token_id, dtype=torch.long, device=device)
                
                # 다음 step에서 보일 위치: 원본 토큰
                visible_until_target = order[:target_num_visible]
                if len(visible_until_target) > 0:
                    step_target[visible_until_target] = input_ids[b, visible_until_target]
                
                # 아직 마스킹된 위치: [MASK]
                still_masked = order[target_num_visible:]
                if len(still_masked) > 0:
                    step_target[still_masked] = self.mask_token_id
                
                # 패딩 위치는 초기값 그대로 (pad_token_id)
                
                # ============================================================
                # 할당
                # ============================================================
                targets[b, step_idx] = step_target
                loss_masks[b, step_idx] = step_loss_mask
                gate_targets[b, step_idx + 1] = (target_step.float() / self.total_steps) * 20.0
        
        return {
            'input': inputs,
            'targets': targets,
            'loss_masks': loss_masks,
            'gate_targets': gate_targets,
            'attention_mask': attention_mask,
            'chain_lengths': chain_lengths
        }