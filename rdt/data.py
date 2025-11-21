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

class ShuffleIndexMasking:
    def __init__(self, total_steps: int = 10, mask_token_id: int = 103, visible_loss_ratio: float = 0.15):
        self.total_steps = total_steps
        self.mask_token_id = mask_token_id
        self.visible_loss_ratio = visible_loss_ratio
    
    def generate_shuffle_order(self, seq_len: int) -> np.ndarray:
        indices = np.arange(seq_len)
        np.random.shuffle(indices)
        return indices
    
    def create_step_data(self, tokens: torch.Tensor, shuffle_order: np.ndarray, step: int):
        seq_len = len(tokens)
        tokens_reordered = tokens[shuffle_order]
        pos_ids = torch.from_numpy(shuffle_order).long()
        
        # 현재 스텝에서 보여줄 개수
        num_visible = int(seq_len * (1 - step / self.total_steps))
        
        # 입력 생성 (뒷부분 마스킹)
        input_tokens = tokens_reordered.clone()
        input_tokens[num_visible:] = self.mask_token_id
        target_tokens = tokens_reordered
        
        # [수정] Delta(이번 목표) 구간 계산
        # 전체를 다 예측하는 게 아니라, 이번 스텝(1/N) 만큼만 예측하여 VRAM 절약
        step_size = int(seq_len / self.total_steps)
        if step_size == 0: step_size = 1
        
        # 이번에 새로 밝힐 구간 (Next Chunk)
        delta_start = num_visible
        delta_end = min(seq_len, num_visible + step_size)
        
        n_delta = max(0, delta_end - delta_start) # 이번 타겟 길이
        
        return input_tokens, target_tokens, pos_ids, num_visible, n_delta


class WikiTextDataset(Dataset):
    def __init__(self, dataset_name='wikitext-2', split='train', tokenizer_name='bert-base-uncased', 
                 max_seq_length=512, total_steps=10, max_chain_length=5, visible_loss_ratio=0.15):
        self.split = split
        self.max_seq_length = max_seq_length
        self.total_steps = total_steps
        self.max_chain_length = max_chain_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.masking = ShuffleIndexMasking(total_steps, self.mask_token_id, visible_loss_ratio)
        
        print(f"Loading {dataset_name} ({split})...")
        dataset_config = 'wikitext-2-raw-v1' if dataset_name == 'wikitext-2' else 'wikitext-103-raw-v1'
        self.dataset = load_dataset('wikitext', dataset_config, split=split)
        self.tokenized_data = self._prepare_data()
        print(f"Dataset loaded: {len(self.tokenized_data)} sequences")
    
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
        return len(self.tokenized_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.tokenized_data[idx]
        seq_len = len(tokens)
        shuffle_order = self.masking.generate_shuffle_order(seq_len)
        
        max_start = self.total_steps
        min_start = 1
        start_step = random.randint(min_start, max_start)
        chain_length = min(self.max_chain_length, start_step)
        
        inputs, targets = [], []
        n_revealed_list, n_delta_list = [], []
        gate_targets = []
        
        # [수정] pos_ids는 루프 밖에서 한 번만 생성 (모든 스텝 동일)
        # create_step_data는 내부적으로 pos_ids를 만들지만 여기선 한 번만 받으면 됨
        _, _, pos_ids_fixed, _, _ = self.masking.create_step_data(tokens, shuffle_order, start_step)

        for offset in range(chain_length):
            step = start_step - offset
            input_tokens, target_tokens, _, n_rev, n_del = self.masking.create_step_data(tokens, shuffle_order, step)
            
            inputs.append(input_tokens)
            targets.append(target_tokens)
            n_revealed_list.append(n_rev)
            n_delta_list.append(n_del)
            gate_targets.append(step / self.total_steps)
        
        return {
            'input': inputs[0],
            'targets': torch.stack(targets),
            'pos_ids': pos_ids_fixed, # [수정] Stack 아님, (seq_len,)
            'n_revealed': torch.tensor(n_revealed_list, dtype=torch.long),
            'n_delta': torch.tensor(n_delta_list, dtype=torch.long),
            'gate_targets': torch.tensor(gate_targets, dtype=torch.float),
            'chain_length': chain_length
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    max_seq_len = max(item['input'].size(0) for item in batch)
    max_chain_len = max(item['chain_length'] for item in batch)
    batch_size = len(batch)
    pad_token_id = 0
    
    inputs = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
    targets = torch.full((batch_size, max_chain_len, max_seq_len), pad_token_id, dtype=torch.long)
    pos_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long) # [수정] (B, Seq)
    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    
    n_revealed = torch.zeros((batch_size, max_chain_len), dtype=torch.long)
    n_delta = torch.zeros((batch_size, max_chain_len), dtype=torch.long)
    gate_targets = torch.zeros((batch_size, max_chain_len), dtype=torch.float)
    chain_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, item in enumerate(batch):
        seq_len = item['input'].size(0)
        chain_len = item['chain_length']
        
        inputs[i, :seq_len] = item['input']
        targets[i, :chain_len, :seq_len] = item['targets']
        pos_ids[i, :seq_len] = item['pos_ids'] # [수정] 차원 일치
        attention_mask[i, :seq_len] = 1
        n_revealed[i, :chain_len] = item['n_revealed']
        n_delta[i, :chain_len] = item['n_delta']
        gate_targets[i, :chain_len] = item['gate_targets']
        chain_lengths[i] = chain_len
    
    return {
        'input': inputs, 'targets': targets, 'pos_ids': pos_ids,
        'attention_mask': attention_mask, 'n_revealed': n_revealed,
        'n_delta': n_delta, 'gate_targets': gate_targets, 'chain_lengths': chain_lengths
    }

def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    
    data_config = config['data']
    training_config = config['training']
    
    # Train dataset
    train_dataset = WikiTextDataset(
        dataset_name=data_config['dataset_name'],
        split='train',
        tokenizer_name=data_config['tokenizer_name'],
        max_seq_length=data_config['max_seq_length'],
        total_steps=training_config['total_steps'],
        max_chain_length=training_config['max_chain_length'],
        visible_loss_ratio=training_config.get('visible_loss_ratio', 0.15)
    )
    
    # Validation dataset
    val_dataset = WikiTextDataset(
        dataset_name=data_config['dataset_name'],
        split='validation',
        tokenizer_name=data_config['tokenizer_name'],
        max_seq_length=data_config['max_seq_length'],
        total_steps=training_config['total_steps'],
        max_chain_length=training_config['max_chain_length'],
        visible_loss_ratio=training_config.get('visible_loss_ratio', 0.15)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=data_config['pin_memory'],
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader