"""
Precision Debugging: Training Validate vs Test Masking
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

from rdt.models import RDT
from rdt.utils import load_config
from rdt.data.rdt_preprocessor import RDTPreprocessor


def create_masked_input(tokens, mask_ratio, mask_token_id, special_token_ids=None):
    """Create masked input - exact copy from test_masking.py"""
    seq_len = len(tokens)
    
    if special_token_ids is not None:
        maskable_positions = torch.tensor(
            [i for i in range(seq_len) if tokens[i].item() not in special_token_ids],
            dtype=torch.long
        )
    else:
        maskable_positions = torch.arange(seq_len)
    
    num_maskable = len(maskable_positions)
    if num_maskable == 0:
        return tokens.clone(), torch.zeros(seq_len, dtype=torch.bool)
    
    num_mask = int(num_maskable * mask_ratio)
    
    if num_mask == 0:
        return tokens.clone(), torch.zeros(seq_len, dtype=torch.bool)
    
    mask_positions_indices = torch.randperm(num_maskable)[:num_mask]
    mask_indices = maskable_positions[mask_positions_indices]
    
    masked_tokens = tokens.clone()
    masked_tokens[mask_indices] = mask_token_id
    
    eval_mask = torch.zeros(seq_len, dtype=torch.bool)
    eval_mask[mask_indices] = True
    
    return masked_tokens, eval_mask


def load_model_and_tokenizer(config_path: str, checkpoint_path: str, device: str):
    """Load RDT model and tokenizer"""
    config = load_config(config_path)
    
    tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])
    vocab_size = tokenizer.vocab_size
    
    model = RDT(
        vocab_size=vocab_size,
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_encoder_layers=config['model']['n_encoder_layers'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout'],
        max_seq_len=config['data']['max_seq_length'],
        input_processor_layers=config['model']['input_processor_layers'],
        output_processor_layers=config['model']['output_processor_layers'],
        gate_hidden_dim=config['model']['gate_hidden_dim'],
        gate_num_layers=config['model']['gate_num_layers'],
        gate_num_heads=config['model']['gate_num_heads'],
        gate_dropout=config['model']['gate_dropout'],
        rope_base=config['model']['rope_base'],
        gradient_checkpointing=config['model']['gradient_checkpointing']
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, config


def prepare_test_data(num_samples: int = 100):
    """Simple test data"""
    sample_texts = [
        "The quick brown fox jumps over the lazy dog in the forest.",
        "Machine learning is a subset of artificial intelligence that focuses on data.",
        "Natural language processing enables computers to understand human language.",
        "Deep neural networks have revolutionized computer vision and speech recognition.",
        "Transformer models use self-attention mechanisms for sequence processing.",
        "Gradient descent is an optimization algorithm used in machine learning.",
        "Convolutional neural networks are particularly effective for image tasks.",
        "Recurrent neural networks can process sequential data like text and speech.",
        "Reinforcement learning agents learn through interaction with environments.",
        "Transfer learning allows models to leverage knowledge from related tasks.",
    ]
    
    texts = []
    while len(texts) < num_samples:
        texts.extend(sample_texts)
    
    return texts[:num_samples]


def run_train_validate(model, tokenizer, config, test_texts, device):
    """
    Scenario 1: EXACT replica of RDTTrainer.validate()
    """
    print("\n" + "="*80)
    print("SCENARIO 1: Training Validate (RDTTrainer.validate())")
    print("="*80)
    
    model.eval()
    
    # Preprocessor
    preprocessor = RDTPreprocessor(tokenizer, config, device=device)
    
    # Prepare batch
    batch_data = []
    for text in test_texts[:32]:
        input_ids = tokenizer(
            text,
            max_length=config['data']['max_seq_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )['input_ids'].squeeze(0)
        batch_data.append({'input_ids': input_ids})
    
    batch = preprocessor(batch_data)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    
    print("\nBatch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
    
    # Extract batch data
    input_tokens = batch['input']
    targets = batch['targets']
    loss_masks = batch['loss_masks']
    attention_mask = batch['attention_mask']
    gate_targets = batch['gate_targets']
    chain_lengths = batch['chain_lengths']
    
    actual_max_length = chain_lengths.max().item()
    
    # Metrics
    recon_criterion = nn.CrossEntropyLoss()
    gate_criterion = nn.MSELoss()
    
    batch_recon_tensor = torch.tensor(0.0, device=device)
    batch_gate_tensor = torch.tensor(0.0, device=device)
    batch_correct_tokens = torch.tensor(0.0, device=device)
    batch_target_tokens = torch.tensor(0.0, device=device)
    num_valid = 0
    
    with torch.no_grad():
        # Step 0: Initial encoding
        h_0 = model.encode_tokens(input_tokens)
        gate_pred_0, pooled_0 = model.gate(h_0, attention_mask, None, None)
        
        # Gate loss for h_0
        gate_target_0 = gate_targets[:, 0].unsqueeze(1)
        gate_loss_0 = gate_criterion(gate_pred_0, gate_target_0)
        batch_gate_tensor += gate_loss_0
        num_valid += 1
        
        # Iterative steps
        hidden = h_0
        gate_pred = gate_pred_0
        pooled = pooled_0
        
        for step_idx in range(actual_max_length):
            valid_mask = chain_lengths > step_idx
            if valid_mask.sum() == 0:
                break
            
            # Forward step
            hidden, _, _ = model.forward_step(
                hidden,
                attention_mask=attention_mask,
                last_gate_score=gate_pred,
                last_pooled=pooled
            )
            
            # Gate prediction for transformed hidden
            gate_pred, pooled = model.gate(hidden, attention_mask, pooled, gate_pred)
            
            # Reconstruction loss & Accuracy
            step_targets = targets[:, step_idx, :]
            step_loss_mask = loss_masks[:, step_idx, :]
            
            if step_loss_mask.sum() > 0:
                logits = model.decode(hidden, attention_mask)
                sel_logits = logits[step_loss_mask]
                sel_targ = step_targets[step_loss_mask]
                recon_loss = recon_criterion(sel_logits, sel_targ)
                
                # Accuracy
                pred_tokens = logits.argmax(dim=-1)
                correct = (pred_tokens[step_loss_mask] == step_targets[step_loss_mask]).float().sum()
                batch_correct_tokens += correct
                batch_target_tokens += step_loss_mask.sum().float()
            else:
                recon_loss = torch.tensor(0.0, device=device)
            
            # Gate loss
            step_gate_targets = gate_targets[:, step_idx + 1].unsqueeze(1)
            gate_pred_valid = gate_pred[valid_mask]
            gate_target_valid = step_gate_targets[valid_mask]
            gate_loss = gate_criterion(gate_pred_valid, gate_target_valid) if len(gate_pred_valid) > 0 else torch.tensor(0.0, device=device)
            
            batch_recon_tensor += recon_loss
            batch_gate_tensor += gate_loss
            num_valid += 1
    
    # Calculate final metrics
    if num_valid > 0:
        avg_recon = (batch_recon_tensor / num_valid).item()
        avg_gate = (batch_gate_tensor / num_valid).item()
        batch_acc = (batch_correct_tokens / (batch_target_tokens + 1e-6)).item()
    else:
        avg_recon = avg_gate = batch_acc = 0.0
    
    print("\nTraining Validate Results:")
    print(f"  Reconstruction Loss: {avg_recon:.4f}")
    print(f"  Gate Loss: {avg_gate:.4f}")
    print(f"  Accuracy (masked positions): {batch_acc:.4f}")
    print(f"  Total masked positions: {batch_target_tokens.item():.0f}")
    print(f"  Average chain length: {chain_lengths.float().mean().item():.2f}")
    
    return {
        'recon_loss': avg_recon,
        'gate_loss': avg_gate,
        'accuracy': batch_acc,
        'total_masked': batch_target_tokens.item()
    }


def run_test_masking(model, tokenizer, config, test_texts, device, mask_ratio=0.3):
    """
    Scenario 2: EXACT replica of test_masking.py test_rdt_model()
    """
    print("\n" + "="*80)
    print(f"SCENARIO 2: Test Masking (test_masking.py, mask_ratio={mask_ratio})")
    print("="*80)
    
    model.eval()
    
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id or 0
    special_token_ids = set(tokenizer.all_special_ids)
    
    max_seq_len = config['data']['max_seq_length']
    max_steps = 20
    threshold = 0.5
    batch_size = 32
    
    # Tokenize
    all_tokens = []
    all_texts = []
    
    for text in test_texts[:32]:
        encoded = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_seq_len,
            padding=False
        )
        tokens = encoded['input_ids'].squeeze(0)
        
        if len(tokens) >= 10:
            all_tokens.append(tokens)
            all_texts.append(tokenizer.decode(tokens, skip_special_tokens=True))
    
    # Results
    exact_match_scores = []
    steps_taken_list = []
    
    # Process batch
    batch_tokens = all_tokens
    batch_input_ids = []
    batch_eval_masks = []
    batch_original_tokens = []
    
    for tokens in batch_tokens:
        masked_tokens, eval_mask = create_masked_input(tokens, mask_ratio, mask_token_id, special_token_ids)
        
        batch_input_ids.append(masked_tokens)
        batch_eval_masks.append(eval_mask)
        batch_original_tokens.append(tokens)
    
    # Pad to max length in batch
    max_len = max(len(t) for t in batch_input_ids)
    padded_input_ids = []
    padded_attention_masks = []
    
    for masked_tokens in batch_input_ids:
        pad_len = max_len - len(masked_tokens)
        if pad_len > 0:
            padded_input_ids.append(
                torch.cat([masked_tokens, torch.full((pad_len,), pad_token_id)])
            )
            padded_attention_masks.append(
                torch.cat([torch.ones(len(masked_tokens)), torch.zeros(pad_len)])
            )
        else:
            padded_input_ids.append(masked_tokens)
            padded_attention_masks.append(torch.ones(len(masked_tokens)))
    
    # Stack
    batch_input_ids_tensor = torch.stack(padded_input_ids).to(device)
    batch_attention_masks_tensor = torch.stack(padded_attention_masks).to(device)
    
    print(f"\nBatch info:")
    print(f"  Input shape: {batch_input_ids_tensor.shape}")
    print(f"  Total masked positions: {sum(m.sum().item() for m in batch_eval_masks)}")
    
    # Inference
    with torch.no_grad():
        output_ids, steps_taken = model.inference(
            batch_input_ids_tensor,
            attention_mask=batch_attention_masks_tensor,
            max_steps=max_steps,
            threshold=threshold,
            return_steps=True
        )
        
        pred_tokens_batch = output_ids.cpu()
        steps_batch = steps_taken.cpu()
    
    # Calculate metrics per sample
    for j in range(len(batch_tokens)):
        original_tokens = batch_original_tokens[j]
        eval_mask = batch_eval_masks[j]
        
        # Extract predictions (remove padding)
        orig_len = len(original_tokens)
        pred_tokens = pred_tokens_batch[j][:orig_len]
        num_steps = steps_batch[j].item()
        
        # Exact Match Accuracy
        if eval_mask.sum() > 0:
            correct = (pred_tokens[eval_mask] == original_tokens[eval_mask]).sum().item()
            total = eval_mask.sum().item()
            accuracy = correct / total
        else:
            accuracy = 1.0
        
        exact_match_scores.append(accuracy)
        steps_taken_list.append(num_steps)
    
    # Aggregate
    avg_accuracy = np.mean(exact_match_scores)
    avg_steps = np.mean(steps_taken_list)
    
    print(f"\nTest Masking Results:")
    print(f"  Accuracy (masked positions): {avg_accuracy:.4f}")
    print(f"  Average steps: {avg_steps:.2f}")
    
    return {
        'accuracy': avg_accuracy,
        'avg_steps': avg_steps
    }


def detailed_comparison(results1, results2):
    """Comparison"""
    print("\n" + "="*80)
    print("DETAILED COMPARISON")
    print("="*80)
    
    print(f"\n{'Metric':<30} {'Train Validate':<20} {'Test Masking':<20} {'Difference':<20}")
    print("-"*90)
    
    acc_diff = results2['accuracy'] - results1['accuracy']
    acc_diff_pct = (acc_diff / results1['accuracy']) * 100 if results1['accuracy'] > 0 else 0
    print(f"{'Accuracy':<30} {results1['accuracy']:<20.4f} {results2['accuracy']:<20.4f} {acc_diff:+.4f} ({acc_diff_pct:+.1f}%)")
    
    print("\n" + "="*80)


def main():
    config_path = 'rdt/configs/rdt.yaml'
    checkpoint_path = 'checkpoints/best_model.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_samples = 100
    
    print("="*80)
    print("RDT PRECISION DEBUGGING: Train Validate vs Test Masking")
    print("="*80)
    print(f"\nConfig: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    model, tokenizer, config = load_model_and_tokenizer(config_path, checkpoint_path, device)
    print(f"Model loaded successfully")
    
    test_texts = prepare_test_data(num_samples)
    
    results1 = run_train_validate(model, tokenizer, config, test_texts, device)
    results2 = run_test_masking(model, tokenizer, config, test_texts, device, mask_ratio=0.3)
    
    detailed_comparison(results1, results2)
    
    print("\n" + "="*80)
    print("DEBUGGING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()