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
                 max_seq_length=512, total_steps=10, max_chain_length=5, visible_loss_ratio=0.15,
                 bert_masking_enabled=True, mask_prob=0.8, random_prob=0.1, keep_prob=0.1):
        self.dataset_name = dataset_name
        self.split = split
        self.max_seq_length = max_seq_length
        self.total_steps = total_steps
        self.max_chain_length = max_chain_length
        self.visible_loss_ratio = visible_loss_ratio
        
        # BERT-style masking parameters
        self.bert_masking_enabled = bert_masking_enabled
        self.mask_prob = mask_prob
        self.random_prob = random_prob
        self.keep_prob = keep_prob
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.vocab_size = self.tokenizer.vocab_size
        
        print(f"Loading {dataset_name} ({split}) in streaming mode...")
        
        # Load streaming dataset (split processing will be done in __iter__)
        if 'bookcorpus' in dataset_name.lower():
            self.dataset = load_dataset('rojagtap/bookcorpus', split='train', streaming=True)
            print(f"BookCorpus loaded in streaming mode")
        elif 'wikipedia' in dataset_name.lower():
            self.dataset = load_dataset('wikimedia/wikipedia', '20231101.en', split='train', streaming=True)
            print(f"Wikipedia 20231101.en loaded in streaming mode")
        else:
            # WikiText already has splits
            dataset_config = 'wikitext-2-raw-v1' if 'wikitext-2' in dataset_name else 'wikitext-103-raw-v1'
            self.dataset = load_dataset('wikitext', dataset_config, split=split, streaming=True)
            print(f"Dataset loaded in streaming mode")
    
    def _apply_bert_masking(self, input_ids: torch.Tensor, masked_indices: torch.Tensor, original_tokens: torch.Tensor) -> torch.Tensor:
        """Apply BERT-style masking: 80% [MASK], 10% random, 10% keep original"""
        if len(masked_indices) == 0 or not self.bert_masking_enabled:
            input_ids[masked_indices] = self.mask_token_id
            return input_ids
        
        # 마스킹할 토큰들에 대해 BERT 스타일 적용
        rand = torch.rand(len(masked_indices))
        
        # 80%: [MASK] 토큰
        mask_indices = masked_indices[rand < self.mask_prob]
        input_ids[mask_indices] = self.mask_token_id
        
        # 10%: 랜덤 토큰 (special tokens 제외)
        random_indices = masked_indices[(rand >= self.mask_prob) & (rand < self.mask_prob + self.random_prob)]
        if len(random_indices) > 0:
            random_tokens = torch.randint(self.tokenizer.cls_token_id + 1, self.vocab_size, (len(random_indices),))
            input_ids[random_indices] = random_tokens
        
        # 10%: 원본 유지 (keep_prob) - 아무것도 안 함
        
        return input_ids
    
    def _process_text(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process a single tokenized text into training sample"""
        seq_len = len(tokens)
        
        # Random restore order
        restore_order = torch.randperm(seq_len)
        
        # Chain length is fixed, adjust start_step range
        chain_length = random.randint(1, self.max_chain_length)
        max_start = self.total_steps
        min_start = chain_length  # s_L이 step 0이 되도록 보장
        start_step = random.randint(min_start, max_start)
        
        # Generate input (s_0 only)
        step_0 = start_step
        num_visible_0 = int(seq_len * (1 - step_0 / self.total_steps))
        visible_indices_0 = restore_order[:num_visible_0]
        masked_indices_0 = restore_order[num_visible_0:]
        
        input_ids = tokens.clone()
        if len(masked_indices_0) > 0:
            input_ids = self._apply_bert_masking(input_ids, masked_indices_0, tokens)
        
        # Generate targets (s_1, s_2, ..., s_L) and gate_targets (s_0, s_1, ..., s_L)
        targets = []
        gate_targets = []
        
        # s_0의 gate target
        gate_targets.append(step_0 / self.total_steps * 20)
        
        # s_1 ~ s_L 의 target 및 gate target 생성
        for offset in range(1, chain_length + 1):
            step = start_step - offset
            num_visible = int(seq_len * (1 - step / self.total_steps))
            
            visible_indices = restore_order[:num_visible]
            masked_indices = restore_order[num_visible:]
            
            target_ids = tokens.clone()
            if len(masked_indices) > 0:
                target_ids[masked_indices] = self.mask_token_id
            
            targets.append(target_ids)
            gate_targets.append(step / self.total_steps * 20)
        
        # Generate loss masks
        loss_masks = []
        
        for i in range(chain_length):
            current_step = start_step - i
            next_step = start_step - (i + 1)
            
            current_visible = int(seq_len * (1 - current_step / self.total_steps))
            next_visible = int(seq_len * (1 - next_step / self.total_steps))
            
            # Delta: 다음 step에서 새로 드러나는 위치
            delta_indices = restore_order[current_visible:next_visible]
            
            mask_bool = torch.zeros(seq_len, dtype=torch.bool)
            if len(delta_indices) > 0:
                mask_bool[delta_indices] = True
            
            # Maintenance: 이미 드러난 것 중 일부
            if current_visible > 0 and self.visible_loss_ratio > 0:
                num_maint = max(1, int(current_visible * self.visible_loss_ratio))
                perm = torch.randperm(current_visible)[:num_maint]
                maint_indices = restore_order[perm]
                mask_bool[maint_indices] = True
            
            loss_masks.append(mask_bool)
        
        return {
            'input': input_ids,  # s_0 (Seq,)
            'targets': torch.stack(targets),  # (L, Seq) - s_1~s_L
            'loss_masks': torch.stack(loss_masks),  # (L, Seq)
            'gate_targets': torch.tensor(gate_targets, dtype=torch.float),  # (L+1,) - s_0~s_L의 step
            'chain_length': chain_length
        }
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over streaming dataset with manual splitting and multi-worker support"""
        
        # Multi-worker support: prevent data duplication
        worker_info = torch.utils.data.get_worker_info()
        
        for idx, item in enumerate(self.dataset):
            # 1. Multi-worker handling: each worker processes different subset
            if worker_info is not None:
                if idx % worker_info.num_workers != worker_info.id:
                    continue
            
            # 2. Train/Val/Test split for datasets without built-in splits (BookCorpus, Wikipedia)
            if 'bookcorpus' in self.dataset_name.lower() or 'wikipedia' in self.dataset_name.lower():
                # Modulo-based split: 40 items → 38 train (95%), 1 val (2.5%), 1 test (2.5%)
                mod = idx % 40
                
                if self.split == 'train':
                    if mod >= 38:  # Skip indices 38, 39
                        continue
                elif self.split == 'validation':
                    if mod != 38:  # Only take index 38
                        continue
                elif self.split == 'test':
                    if mod != 39:  # Only take index 39
                        continue
            
            # 3. Process text
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
                 max_seq_length=512, total_steps=10, max_chain_length=5, visible_loss_ratio=0.15, samples_per_text=1, streaming=False,
                 bert_masking_enabled=True, mask_prob=0.8, random_prob=0.1, keep_prob=0.1):
        if streaming:
            raise ValueError("Use StreamingTextDataset for streaming mode instead")
        
        self.split = split
        self.max_seq_length = max_seq_length
        self.total_steps = total_steps
        self.max_chain_length = max_chain_length
        self.visible_loss_ratio = visible_loss_ratio
        self.samples_per_text = samples_per_text
        
        # BERT-style masking parameters
        self.bert_masking_enabled = bert_masking_enabled
        self.mask_prob = mask_prob
        self.random_prob = random_prob
        self.keep_prob = keep_prob
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.vocab_size = self.tokenizer.vocab_size
        
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
            num_proc=4,
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
    
    def _apply_bert_masking(self, input_ids: torch.Tensor, masked_indices: torch.Tensor, original_tokens: torch.Tensor) -> torch.Tensor:
        """Apply BERT-style masking: 80% [MASK], 10% random, 10% keep original"""
        if len(masked_indices) == 0 or not self.bert_masking_enabled:
            input_ids[masked_indices] = self.mask_token_id
            return input_ids
        
        # 마스킹할 토큰들에 대해 BERT 스타일 적용
        rand = torch.rand(len(masked_indices))
        
        # 80%: [MASK] 토큰
        mask_indices = masked_indices[rand < self.mask_prob]
        input_ids[mask_indices] = self.mask_token_id
        
        # 10%: 랜덤 토큰 (special tokens 제외)
        random_indices = masked_indices[(rand >= self.mask_prob) & (rand < self.mask_prob + self.random_prob)]
        if len(random_indices) > 0:
            random_tokens = torch.randint(self.tokenizer.cls_token_id + 1, self.vocab_size, (len(random_indices),))
            input_ids[random_indices] = random_tokens
        
        # 10%: 원본 유지 (keep_prob) - 아무것도 안 함
        
        return input_ids
    
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
        
        # Chain length is fixed, adjust start_step range
        chain_length = random.randint(1, self.max_chain_length)
        max_start = self.total_steps
        min_start = chain_length  # s_L이 step 0이 되도록 보장
        start_step = random.randint(min_start, max_start)
        
        # Generate input (s_0 only)
        step_0 = start_step
        num_visible_0 = int(seq_len * (1 - step_0 / self.total_steps))
        visible_indices_0 = restore_order[:num_visible_0]
        masked_indices_0 = restore_order[num_visible_0:]
        
        input_ids = tokens.clone()
        if len(masked_indices_0) > 0:
            input_ids = self._apply_bert_masking(input_ids, masked_indices_0, tokens)
        
        # Generate targets (s_1, s_2, ..., s_L) and gate_targets (s_0, s_1, ..., s_L)
        targets = []
        gate_targets = []
        
        # s_0의 gate target
        gate_targets.append(step_0 / self.total_steps * 20)
        
        # s_1 ~ s_L 의 target 및 gate target 생성
        for offset in range(1, chain_length + 1):
            step = start_step - offset
            num_visible = int(seq_len * (1 - step / self.total_steps))
            
            visible_indices = restore_order[:num_visible]
            masked_indices = restore_order[num_visible:]
            
            target_ids = tokens.clone()
            if len(masked_indices) > 0:
                target_ids[masked_indices] = self.mask_token_id
            
            targets.append(target_ids)
            gate_targets.append(step / self.total_steps * 20)
        
        # Step 2: Loss Mask 생성
        loss_masks = []
        
        for i in range(chain_length):
            current_step = start_step - i
            next_step = start_step - (i + 1)
            
            current_visible = int(seq_len * (1 - current_step / self.total_steps))
            next_visible = int(seq_len * (1 - next_step / self.total_steps))
            
            # Delta: 다음 step에서 새로 드러나는 위치
            delta_indices = restore_order[current_visible:next_visible]
            
            mask_bool = torch.zeros(seq_len, dtype=torch.bool)
            if len(delta_indices) > 0:
                mask_bool[delta_indices] = True
            
            # Maintenance: 이미 드러난 것 중 일부
            if current_visible > 0 and self.visible_loss_ratio > 0:
                num_maint = max(1, int(current_visible * self.visible_loss_ratio))
                perm = torch.randperm(current_visible)[:num_maint]
                maint_indices = restore_order[perm]
                mask_bool[maint_indices] = True
            
            loss_masks.append(mask_bool)
        
        return {
            'input': input_ids,  # s_0 (Seq,)
            'targets': torch.stack(targets),  # (L, Seq) - s_1~s_L
            'loss_masks': torch.stack(loss_masks),  # (L, Seq)
            'gate_targets': torch.tensor(gate_targets, dtype=torch.float),  # (L+1,) - s_0~s_L의 step
            'chain_length': chain_length
        }

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    max_seq_len = max(item['input'].size(0) for item in batch)  # input\uc774 \uc774\uc81c 1D
    max_chain_len = max(item['chain_length'] for item in batch)
    batch_size = len(batch)
    pad_token_id = 0
    
    inputs = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
    targets = torch.full((batch_size, max_chain_len, max_seq_len), pad_token_id, dtype=torch.long)
    loss_masks = torch.zeros((batch_size, max_chain_len, max_seq_len), dtype=torch.bool)
    gate_targets = torch.zeros((batch_size, max_chain_len + 1), dtype=torch.float)  # L+1\uac1c
    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    chain_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, item in enumerate(batch):
        seq_len = item['input'].size(0)  # 1D tensor
        chain_len = item['chain_length']
        
        inputs[i, :seq_len] = item['input']
        targets[i, :chain_len, :seq_len] = item['targets']
        loss_masks[i, :chain_len, :seq_len] = item['loss_masks']
        gate_targets[i, :chain_len + 1] = item['gate_targets']  # L+1\uac1c
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
    
    # BERT masking config
    bert_config = training_config.get('bert_masking', {})
    bert_masking_enabled = bert_config.get('enabled', True)
    mask_prob = bert_config.get('mask_prob', 0.8)
    random_prob = bert_config.get('random_prob', 0.1)
    keep_prob = bert_config.get('keep_prob', 0.1)
    
    if streaming:
        # Use StreamingTextDataset for streaming mode
        train_dataset = StreamingTextDataset(
            dataset_name=data_config['dataset_name'],
            split='train',
            tokenizer_name=data_config['tokenizer_name'],
            max_seq_length=data_config['max_seq_length'],
            total_steps=training_config['total_steps'],
            max_chain_length=training_config['max_chain_length'],
            visible_loss_ratio=training_config.get('visible_loss_ratio', 0.15),
            bert_masking_enabled=bert_masking_enabled,
            mask_prob=mask_prob,
            random_prob=random_prob,
            keep_prob=keep_prob
        )
        
        val_dataset = StreamingTextDataset(
            dataset_name=data_config['dataset_name'],
            split='validation',
            tokenizer_name=data_config['tokenizer_name'],
            max_seq_length=data_config['max_seq_length'],
            total_steps=training_config['total_steps'],
            max_chain_length=training_config['max_chain_length'],
            visible_loss_ratio=training_config.get('visible_loss_ratio', 0.15),
            bert_masking_enabled=bert_masking_enabled,
            mask_prob=mask_prob,
            random_prob=random_prob,
            keep_prob=keep_prob
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
            streaming=streaming,
            bert_masking_enabled=bert_masking_enabled,
            mask_prob=mask_prob,
            random_prob=random_prob,
            keep_prob=keep_prob
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
            streaming=streaming,
            bert_masking_enabled=bert_masking_enabled,
            mask_prob=mask_prob,
            random_prob=random_prob,
            keep_prob=keep_prob
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