"""Data loading and preprocessing for RDT"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import random
import numpy as np
from typing import List, Dict, Tuple, Iterator
from tqdm import tqdm

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class StreamingTextDataset(IterableDataset):
    """Streaming dataset for on-the-fly loading without preloading all data"""
    
    def __init__(self, dataset_name='wikitext-2', split='train', tokenizer_name='bert-base-uncased',
                 max_seq_length=512, total_steps=10, max_chain_length=5, visible_loss_ratio=0.15):
        self.dataset_name = dataset_name
        self.split = split
        self.max_seq_length = max_seq_length
        self.total_steps = total_steps
        self.max_chain_length = max_chain_length
        self.visible_loss_ratio = visible_loss_ratio
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mask_token_id = self.tokenizer.mask_token_id
        
        print(f"Loading {dataset_name} ({split}) in streaming mode...")
        
        # Load streaming dataset
        if 'bookcorpus' in dataset_name.lower():
            self.dataset = load_dataset('rojagtap/bookcorpus', split='train', streaming=True)
            print(f"BookCorpus loaded in streaming mode (split not supported)")
        elif 'wikipedia' in dataset_name.lower():
            # Wikipedia 20231101.en - only train split available
            self.dataset = load_dataset('wikimedia/wikipedia', '20231101.en', split='train', streaming=True)
            print(f"Wikipedia 20231101.en loaded in streaming mode (split not supported)")
        else:
            dataset_config = 'wikitext-2-raw-v1' if 'wikitext-2' in dataset_name else 'wikitext-103-raw-v1'
            self.dataset = load_dataset('wikitext', dataset_config, split=split, streaming=True)
            print(f"Dataset loaded in streaming mode")
    
    def _process_text(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process a single tokenized text into training sample"""
        seq_len = len(tokens)
        
        # Random restore order
        restore_order = torch.randperm(seq_len)
        
        # Random start step
        max_start = self.total_steps
        min_start = 1
        start_step = random.randint(min_start, max_start)
        chain_length = min(self.max_chain_length, start_step)
        
        inputs = []
        gate_targets = []
        
        # Generate inputs for each step
        for offset in range(chain_length):
            step = start_step - offset
            num_visible = int(seq_len * (1 - step / self.total_steps))
            
            visible_indices = restore_order[:num_visible]
            masked_indices = restore_order[num_visible:]
            
            input_ids = tokens.clone()
            if len(masked_indices) > 0:
                input_ids[masked_indices] = self.mask_token_id
            
            inputs.append(input_ids)
            gate_targets.append(step / self.total_steps * 20)
        
        # Generate targets and loss masks
        targets = []
        loss_masks = []
        
        for i in range(chain_length):
            if i < chain_length - 1:
                target = inputs[i + 1]
                
                current_step = start_step - i
                next_step = start_step - (i + 1)
                
                current_visible = int(seq_len * (1 - current_step / self.total_steps))
                next_visible = int(seq_len * (1 - next_step / self.total_steps))
                
                delta_indices = restore_order[current_visible:next_visible]
                
                mask_bool = torch.zeros(seq_len, dtype=torch.bool)
                mask_bool[delta_indices] = True
                
                if current_visible > 0 and self.visible_loss_ratio > 0:
                    num_maint = max(1, int(current_visible * self.visible_loss_ratio))
                    perm = torch.randperm(current_visible)[:num_maint]
                    maint_indices = restore_order[perm]
                    mask_bool[maint_indices] = True
            else:
                target = tokens
                
                last_step = start_step - (chain_length - 1)
                last_visible = int(seq_len * (1 - last_step / self.total_steps))
                remaining_indices = restore_order[last_visible:]
                
                mask_bool = torch.zeros(seq_len, dtype=torch.bool)
                mask_bool[remaining_indices] = True
            
            targets.append(target)
            loss_masks.append(mask_bool)
        
        return {
            'input': torch.stack(inputs),
            'targets': torch.stack(targets),
            'loss_masks': torch.stack(loss_masks),
            'gate_targets': torch.tensor(gate_targets, dtype=torch.float),
            'chain_length': chain_length
        }
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over streaming dataset"""
        for item in self.dataset:
            text = item.get('text', '').strip()
            if len(text) == 0:
                continue
            
            encoded = self.tokenizer(text, max_length=self.max_seq_length, truncation=True, 
                                   padding=False, return_tensors='pt')
            tokens = encoded['input_ids'].squeeze(0)
            
            if len(tokens) < 10:
                continue
            
            yield self._process_text(tokens)


class WikiTextDataset(Dataset):
    """Map-style dataset for preloading all data into memory"""
    
    def __init__(self, dataset_name='wikitext-2', split='train', tokenizer_name='bert-base-uncased', 
                 max_seq_length=512, total_steps=10, max_chain_length=5, visible_loss_ratio=0.15, samples_per_text=1, streaming=False):
        if streaming:
            raise ValueError("Use StreamingTextDataset for streaming mode instead")
        
        self.split = split
        self.max_seq_length = max_seq_length
        self.total_steps = total_steps
        self.max_chain_length = max_chain_length
        self.visible_loss_ratio = visible_loss_ratio
        self.samples_per_text = samples_per_text
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mask_token_id = self.tokenizer.mask_token_id
        
        print(f"Loading {dataset_name} ({split})...")
        
        # Determine dataset source and config
        if 'bookcorpus' in dataset_name.lower():
            # BookCorpus only has train split, so we need to split it
            full_dataset = load_dataset('rojagtap/bookcorpus', split='train')
            
            # Split: 95% train, 2.5% validation, 2.5% test
            total_size = len(full_dataset)
            train_size = int(0.95 * total_size)
            val_size = int(0.025 * total_size)
            test_size = total_size - train_size - val_size
            
            splits = full_dataset.train_test_split(test_size=val_size + test_size, seed=42)
            train_data = splits['train']
            remaining = splits['test']
            
            val_test_splits = remaining.train_test_split(test_size=test_size, seed=42)
            val_data = val_test_splits['train']
            test_data = val_test_splits['test']
            
            # Select appropriate split
            if split == 'train':
                self.dataset = train_data
            elif split == 'validation':
                self.dataset = val_data
            elif split == 'test':
                self.dataset = test_data
            else:
                raise ValueError(f"Invalid split: {split}")
            
            self.tokenized_data = self._prepare_data()
            print(f"BookCorpus {split} split: {len(self.tokenized_data)} sequences")
            print(f"Total samples (with samples_per_text={samples_per_text}): {len(self.tokenized_data) * samples_per_text}")
        elif 'wikipedia' in dataset_name.lower():
            # Wikipedia 20231101.en only has train split, so we need to split it
            full_dataset = load_dataset('wikimedia/wikipedia', '20231101.en', split='train')
            
            # Split: 95% train, 2.5% validation, 2.5% test
            total_size = len(full_dataset)
            train_size = int(0.95 * total_size)
            val_size = int(0.025 * total_size)
            test_size = total_size - train_size - val_size
            
            splits = full_dataset.train_test_split(test_size=val_size + test_size, seed=42)
            train_data = splits['train']
            remaining = splits['test']
            
            val_test_splits = remaining.train_test_split(test_size=test_size, seed=42)
            val_data = val_test_splits['train']
            test_data = val_test_splits['test']
            
            # Select appropriate split
            if split == 'train':
                self.dataset = train_data
            elif split == 'validation':
                self.dataset = val_data
            elif split == 'test':
                self.dataset = test_data
            else:
                raise ValueError(f"Invalid split: {split}")
            
            self.tokenized_data = self._prepare_data()
            print(f"Wikipedia 20231101.en {split} split: {len(self.tokenized_data)} sequences")
            print(f"Total samples (with samples_per_text={samples_per_text}): {len(self.tokenized_data) * samples_per_text}")
        else:
            # WikiText datasets
            dataset_config = 'wikitext-2-raw-v1' if 'wikitext-2' in dataset_name else 'wikitext-103-raw-v1'
            self.dataset = load_dataset('wikitext', dataset_config, split=split)
            self.tokenized_data = self._prepare_data()
            print(f"Dataset loaded: {len(self.tokenized_data)} sequences")
            print(f"Total samples (with samples_per_text={samples_per_text}): {len(self.tokenized_data) * samples_per_text}")
    
    def _prepare_data(self) -> List[torch.Tensor]:
        """Tokenize dataset using fast batched processing"""
        print("Tokenizing texts...")
        
        # Create tokenizer instance with name (not self.tokenizer to avoid pickle issues)
        tokenizer_name = self.tokenizer.name_or_path
        
        # Use HuggingFace's fast .map() method with batching
        def tokenize_function(examples):
            # Create tokenizer inside function to avoid multiprocessing pickle issues
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            return tokenizer(
                examples['text'],
                max_length=self.max_seq_length,
                truncation=True,
                padding=False,
            )
        
        # Apply tokenization with batching (single process to avoid memory/pickle issues)
        tokenized_dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            batch_size=1024,
            remove_columns=self.dataset.column_names,
            desc="Tokenizing"
        )
        
        # Convert to list of tensors, filtering out short sequences
        tokenized = []
        for item in tqdm(tokenized_dataset, desc="Processing tokens"):
            tokens = torch.tensor(item['input_ids'], dtype=torch.long)
            if len(tokens) >= 10:
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
        
        # 복원 순서 결정 (랜덤 순열)
        restore_order = torch.randperm(seq_len)
        
        max_start = self.total_steps
        min_start = 1
        start_step = random.randint(min_start, max_start)
        chain_length = min(self.max_chain_length, start_step)
        
        inputs = []
        gate_targets = []
        
        # Step 1: 각 step의 입력 생성
        for offset in range(chain_length):
            step = start_step - offset
            num_visible = int(seq_len * (1 - step / self.total_steps))
            
            visible_indices = restore_order[:num_visible]
            masked_indices = restore_order[num_visible:]
            
            input_ids = tokens.clone()
            if len(masked_indices) > 0:
                input_ids[masked_indices] = self.mask_token_id
            
            inputs.append(input_ids)
            gate_targets.append(step / self.total_steps * 20)
        
        # Step 2: Target과 Loss Mask 생성 (점진적 디노이징)
        targets = []
        loss_masks = []
        
        for i in range(chain_length):
            if i < chain_length - 1:
                # Target: 다음 step의 입력 (덜 마스킹된 버전)
                target = inputs[i + 1]
                
                # Loss Mask: 다음 step에서 새로 드러나는 위치만
                current_step = start_step - i
                next_step = start_step - (i + 1)
                
                current_visible = int(seq_len * (1 - current_step / self.total_steps))
                next_visible = int(seq_len * (1 - next_step / self.total_steps))
                
                # Delta: 다음 step에서 새로 드러나는 구간
                delta_indices = restore_order[current_visible:next_visible]
                
                mask_bool = torch.zeros(seq_len, dtype=torch.bool)
                mask_bool[delta_indices] = True
                
                # Maintenance: 이미 드러난 것 중 일부 (일관성 유지)
                if current_visible > 0 and self.visible_loss_ratio > 0:
                    num_maint = max(1, int(current_visible * self.visible_loss_ratio))
                    perm = torch.randperm(current_visible)[:num_maint]
                    maint_indices = restore_order[perm]
                    mask_bool[maint_indices] = True
                
            else:
                # 마지막 step: 완전히 복원된 원본이 target
                target = tokens
                
                # Loss Mask: 아직 마스킹된 모든 위치
                last_step = start_step - (chain_length - 1)
                last_visible = int(seq_len * (1 - last_step / self.total_steps))
                remaining_indices = restore_order[last_visible:]
                
                mask_bool = torch.zeros(seq_len, dtype=torch.bool)
                mask_bool[remaining_indices] = True
            
            targets.append(target)
            loss_masks.append(mask_bool)
        
        return {
            'input': torch.stack(inputs),           # (L, Seq)
            'targets': torch.stack(targets),        # (L, Seq)
            'loss_masks': torch.stack(loss_masks),  # (L, Seq)
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
    data_config = config['data']
    training_config = config['training']
    streaming = data_config.get('streaming', False)
    
    if streaming:
        # Use StreamingTextDataset for streaming mode
        train_dataset = StreamingTextDataset(
            dataset_name=data_config['dataset_name'],
            split='train',
            tokenizer_name=data_config['tokenizer_name'],
            max_seq_length=data_config['max_seq_length'],
            total_steps=training_config['total_steps'],
            max_chain_length=training_config['max_chain_length'],
            visible_loss_ratio=training_config.get('visible_loss_ratio', 0.15)
        )
        
        val_dataset = StreamingTextDataset(
            dataset_name=data_config['dataset_name'],
            split='validation',
            tokenizer_name=data_config['tokenizer_name'],
            max_seq_length=data_config['max_seq_length'],
            total_steps=training_config['total_steps'],
            max_chain_length=training_config['max_chain_length'],
            visible_loss_ratio=training_config.get('visible_loss_ratio', 0.15)
        )
        
        # Streaming datasets do not support shuffle
        train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], 
                                 num_workers=data_config['num_workers'], 
                                 pin_memory=data_config['pin_memory'], 
                                 collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'], 
                               num_workers=data_config['num_workers'], 
                               pin_memory=data_config['pin_memory'], 
                               collate_fn=collate_fn)
    else:
        # Use WikiTextDataset for normal mode
        train_dataset = WikiTextDataset(
            dataset_name=data_config['dataset_name'],
            split='train',
            tokenizer_name=data_config['tokenizer_name'],
            max_seq_length=data_config['max_seq_length'],
            total_steps=training_config['total_steps'],
            max_chain_length=training_config['max_chain_length'],
            visible_loss_ratio=training_config.get('visible_loss_ratio', 0.15),
            samples_per_text=data_config.get('samples_per_text', 1),
            streaming=streaming
        )
        
        val_dataset = WikiTextDataset(
            dataset_name=data_config['dataset_name'],
            split='validation',
            tokenizer_name=data_config['tokenizer_name'],
            max_seq_length=data_config['max_seq_length'],
            total_steps=training_config['total_steps'],
            max_chain_length=training_config['max_chain_length'],
            visible_loss_ratio=training_config.get('visible_loss_ratio', 0.15),
            samples_per_text=data_config.get('samples_per_text', 1),
            streaming=streaming
        )
        
        train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], 
                                 shuffle=True, num_workers=data_config['num_workers'], 
                                 pin_memory=data_config['pin_memory'], 
                                 collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'], 
                               shuffle=False, num_workers=data_config['num_workers'], 
                               pin_memory=data_config['pin_memory'], 
                               collate_fn=collate_fn)
    
    return train_loader, val_loader