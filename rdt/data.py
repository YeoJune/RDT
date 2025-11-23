"""Data loading and preprocessing for RDT"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import random
import numpy as np
from typing import List, Dict, Tuple

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class WikiTextDataset(Dataset):
    def __init__(self, dataset_name='wikitext-2', split='train', tokenizer_name='bert-base-uncased', 
                 max_seq_length=512, total_steps=10, max_chain_length=5, visible_loss_ratio=0.15, samples_per_text=1):
        self.split = split
        self.max_seq_length = max_seq_length
        self.total_steps = total_steps
        self.max_chain_length = max_chain_length
        self.visible_loss_ratio = visible_loss_ratio
        self.samples_per_text = samples_per_text
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mask_token_id = self.tokenizer.mask_token_id
        
        print(f"Loading {dataset_name} ({split})...")
        # wikitext-103-raw-v1 or wikitext-2-raw-v1
        dataset_config = 'wikitext-2-raw-v1' if 'wikitext-2' in dataset_name else 'wikitext-103-raw-v1'
        self.dataset = load_dataset('wikitext', dataset_config, split=split)
        self.tokenized_data = self._prepare_data()
        print(f"Dataset loaded: {len(self.tokenized_data)} sequences")
        print(f"Total samples (with samples_per_text={samples_per_text}): {len(self.tokenized_data) * samples_per_text}")
    
    def _prepare_data(self) -> List[torch.Tensor]:
        tokenized = []
        for item in self.dataset:
            text = item['text'].strip()
            if len(text) == 0: continue
            encoded = self.tokenizer(text, max_length=self.max_seq_length, truncation=True, padding=False, return_tensors='pt')
            tokens = encoded['input_ids'].squeeze(0)
            if len(tokens) < 10: continue
            tokenized.append(tokens)
        return tokenized
    
    def __len__(self):
        return len(self.tokenized_data) * self.samples_per_text
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 실제 텍스트 인덱스와 샘플 번호 계산
        text_idx = idx // self.samples_per_text
        sample_idx = idx % self.samples_per_text
        
        tokens = self.tokenized_data[text_idx]
        seq_len = len(tokens)
        
        # 1. 복원 순서 결정 (랜덤 순열)
        # 이 순서대로 토큰이 하나씩 밝혀진다고 가정
        restore_order = torch.randperm(seq_len)
        
        max_start = self.total_steps
        min_start = 1
        start_step = random.randint(min_start, max_start)
        chain_length = min(self.max_chain_length, start_step)
        
        inputs = []
        targets = []
        loss_masks = []
        gate_targets = []
        
        # 한 번의 체인에 대해 연속적인 스텝 생성
        for offset in range(chain_length):
            step = start_step - offset
            
            # 현재 스텝에서 보여야 할 개수 (Visible Count)
            # step=10 (Full Mask) -> visible=0
            # step=0 (Clean) -> visible=seq_len
            num_visible = int(seq_len * (1 - step / self.total_steps))
            
            # 인덱스 구분
            visible_indices = restore_order[:num_visible]
            masked_indices = restore_order[num_visible:]
            
            # 1. 입력 생성 (원본 위치 그대로, 마스킹만 적용)
            input_ids = tokens.clone()
            if len(masked_indices) > 0:
                input_ids[masked_indices] = self.mask_token_id
            
            # 2. Loss Mask 생성 (여기가 핵심)
            # Delta: 이번 스텝의 목표 (다음 스텝에서 밝혀질 구간)
            step_size = int(seq_len / self.total_steps) + 1
            # restore_order 상에서 [num_visible ~ num_visible + step_size] 구간이 이번 Delta
            delta_end = min(seq_len, num_visible + step_size)
            delta_indices = restore_order[num_visible : delta_end]
            
            mask_bool = torch.zeros(seq_len, dtype=torch.bool)
            mask_bool[delta_indices] = True
            
            # Maintenance: 이미 밝혀진 곳 중 일부 랜덤 샘플링
            if num_visible > 0 and self.visible_loss_ratio > 0:
                num_maint = max(1, int(num_visible * self.visible_loss_ratio))
                # visible_indices 중에서 랜덤 선택
                perm = torch.randperm(len(visible_indices))[:num_maint]
                maint_indices = visible_indices[perm]
                mask_bool[maint_indices] = True
            
            inputs.append(input_ids)
            targets.append(tokens) # 타겟은 항상 원본
            loss_masks.append(mask_bool)
            gate_targets.append(step / self.total_steps)
            
        return {
            'input': torch.stack(inputs),       # (L, Seq)
            'targets': torch.stack(targets),    # (L, Seq)
            'loss_masks': torch.stack(loss_masks), # (L, Seq)
            'gate_targets': torch.tensor(gate_targets, dtype=torch.float),
            'chain_length': chain_length
        }

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    max_seq_len = max(item['input'].size(1) for item in batch)
    max_chain_len = max(item['chain_length'] for item in batch)
    batch_size = len(batch)
    pad_token_id = 0
    
    inputs = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
    targets = torch.full((batch_size, max_chain_len, max_seq_len), pad_token_id, dtype=torch.long)
    loss_masks = torch.zeros((batch_size, max_chain_len, max_seq_len), dtype=torch.bool)
    gate_targets = torch.zeros((batch_size, max_chain_len), dtype=torch.float)
    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long) # 0: Pad, 1: Valid
    chain_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, item in enumerate(batch):
        seq_len = item['input'].size(1)
        chain_len = item['chain_length']
        
        # 입력 (첫 스텝만 가져와서 모델에 넣고 돌림 -> Trainer에서 loop 처리하므로 첫 입력만 필요? 
        # 아니요, Trainer는 hidden을 재사용하므로 첫 입력만 필요합니다.
        # 하지만 여기선 item['input']이 (L, Seq) 형태입니다.
        # Trainer 로직상 첫 입력(Start Step)만 필요하므로 0번째만 씁니다.
        inputs[i, :seq_len] = item['input'][0] 
        
        targets[i, :chain_len, :seq_len] = item['targets']
        loss_masks[i, :chain_len, :seq_len] = item['loss_masks']
        gate_targets[i, :chain_len] = item['gate_targets']
        attention_mask[i, :seq_len] = 1
        chain_lengths[i] = chain_len
    
    return {
        'input': inputs,
        'targets': targets,
        'loss_masks': loss_masks,
        'gate_targets': gate_targets,
        'attention_mask': attention_mask,
        'chain_lengths': chain_lengths
    }

def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    # 기존과 동일
    data_config = config['data']
    training_config = config['training']
    
    train_dataset = WikiTextDataset(
        dataset_name=data_config['dataset_name'],
        split='train',
        tokenizer_name=data_config['tokenizer_name'],
        max_seq_length=data_config['max_seq_length'],
        total_steps=training_config['total_steps'],
        max_chain_length=training_config['max_chain_length'],
        visible_loss_ratio=training_config.get('visible_loss_ratio', 0.15),
        samples_per_text=data_config.get('samples_per_text', 1)
    )
    
    val_dataset = WikiTextDataset(
        dataset_name=data_config['dataset_name'],
        split='validation',
        tokenizer_name=data_config['tokenizer_name'],
        max_seq_length=data_config['max_seq_length'],
        total_steps=training_config['total_steps'],
        max_chain_length=training_config['max_chain_length'],
        visible_loss_ratio=training_config.get('visible_loss_ratio', 0.15),
        samples_per_text=data_config.get('samples_per_text', 1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True, num_workers=data_config['num_workers'], pin_memory=data_config['pin_memory'], collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'], shuffle=False, num_workers=data_config['num_workers'], pin_memory=data_config['pin_memory'], collate_fn=collate_fn)
    
    return train_loader, val_loader