"""
Precision Debugging: Training Validate vs Test Masking
상세 디버깅 버전
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
    """Create masked input"""
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
    """Scenario 1: Training Validate with detailed debugging"""
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
    
    print("\n[BATCH INFO]")
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
    
    # Detailed analysis
    print("\n[PREPROCESSING ANALYSIS]")
    print(f"  Chain lengths: min={chain_lengths.min().item()}, max={chain_lengths.max().item()}, mean={chain_lengths.float().mean().item():.2f}")
    print(f"  Gate targets range: [{gate_targets.min().item():.2f}, {gate_targets.max().item():.2f}]")
    
    total_tokens = (attention_mask == 1).sum().item()
    total_masked_in_input = (input_tokens == tokenizer.mask_token_id).sum().item()
    print(f"  Total valid tokens: {total_tokens}")
    print(f"  Masked tokens in input: {total_masked_in_input} ({100*total_masked_in_input/total_tokens:.1f}%)")
    
    # Loss mask analysis
    for step_idx in range(actual_max_length):
        step_loss_mask = loss_masks[:, step_idx, :]
        num_masked = step_loss_mask.sum().item()
        if num_masked > 0:
            print(f"  Step {step_idx}: {num_masked} positions to predict")
    
    # Metrics
    recon_criterion = nn.CrossEntropyLoss()
    gate_criterion = nn.MSELoss()
    
    batch_recon_tensor = torch.tensor(0.0, device=device)
    batch_gate_tensor = torch.tensor(0.0, device=device)
    batch_correct_tokens = torch.tensor(0.0, device=device)
    batch_target_tokens = torch.tensor(0.0, device=device)
    num_valid = 0
    
    step_details = []
    
    # ========================================================================
    # DETAILED LOGGING FOR FIRST SAMPLE
    # ========================================================================
    print("\n[DETAILED FORWARD - FIRST SAMPLE]")
    
    # Get first sample details
    sample_0_input = input_tokens[0:1]
    sample_0_targets = targets[0]
    sample_0_loss_masks = loss_masks[0]
    sample_0_attention_mask = attention_mask[0:1]
    sample_0_chain_length = chain_lengths[0].item()
    
    print(f"Sample 0 Details:")
    print(f"  Chain length: {sample_0_chain_length}")
    
    # Find masked positions in input
    sample_0_input_cpu = sample_0_input.squeeze(0).cpu()
    masked_in_input = (sample_0_input_cpu == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    print(f"  Masked in input: {len(masked_in_input)} positions")
    print(f"  Masked indices: {masked_in_input.tolist()[:10]}")
    
    # Show what needs to be predicted at each step
    for step_idx in range(sample_0_chain_length):
        step_mask = sample_0_loss_masks[step_idx]
        positions_to_predict = step_mask.nonzero(as_tuple=True)[0]
        if len(positions_to_predict) > 0:
            print(f"  Step {step_idx}: Predict {len(positions_to_predict)} positions - {positions_to_predict.tolist()[:5]}")
    
    with torch.no_grad():
        # Step 0: Initial encoding
        print("\n[FORWARD PASS]")
        h_0 = model.encode_tokens(input_tokens)
        gate_pred_0, pooled_0 = model.gate(h_0, attention_mask, None, None)
        
        print(f"Step 0 (Encoding):")
        print(f"  Hidden shape: {h_0.shape}")
        print(f"  Gate prediction: mean={gate_pred_0.mean().item():.4f}, std={gate_pred_0.std().item():.4f}")
        print(f"  Gate target: mean={gate_targets[:, 0].mean().item():.4f}")
        
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
            
            # Gate prediction
            gate_pred, pooled = model.gate(hidden, attention_mask, pooled, gate_pred)
            
            # Reconstruction & Accuracy
            step_targets = targets[:, step_idx, :]
            step_loss_mask = loss_masks[:, step_idx, :]
            
            step_info = {
                'step': step_idx + 1,
                'valid_samples': valid_mask.sum().item(),
                'positions_to_predict': step_loss_mask.sum().item(),
            }
            
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
                
                step_info['recon_loss'] = recon_loss.item()
                step_info['accuracy'] = (correct / step_loss_mask.sum().float()).item()
                step_info['correct'] = correct.item()
                
                # Prediction confidence
                pred_probs = torch.softmax(sel_logits, dim=-1)
                max_probs, _ = pred_probs.max(dim=-1)
                step_info['confidence'] = max_probs.mean().item()
                
                # Top-5 accuracy
                _, top5_preds = pred_probs.topk(5, dim=-1)
                top5_correct = (top5_preds == sel_targ.unsqueeze(-1)).any(dim=-1).float().sum()
                step_info['top5_accuracy'] = (top5_correct / step_loss_mask.sum().float()).item()
                
                # ============================================================
                # DETAILED LOGGING FOR FIRST SAMPLE
                # ============================================================
                sample_0_loss_mask = step_loss_mask[0]
                if sample_0_loss_mask.sum() > 0:
                    sample_0_logits = logits[0]
                    sample_0_pred = pred_tokens[0].cpu()
                    sample_0_target = step_targets[0].cpu()
                    sample_0_probs = torch.softmax(sample_0_logits, dim=-1)
                    
                    positions = sample_0_loss_mask.nonzero(as_tuple=True)[0]
                    
                    print(f"\n  [Sample 0 Predictions - Step {step_idx + 1}]")
                    unique_preds = set()
                    for pos in positions[:5]:  # First 5
                        pos_idx = pos.item()
                        pred_id = sample_0_pred[pos_idx].item()
                        true_id = sample_0_target[pos_idx].item()
                        prob = sample_0_probs[pos_idx, pred_id].item()
                        
                        pred_str = tokenizer.decode([pred_id])
                        true_str = tokenizer.decode([true_id])
                        
                        unique_preds.add(pred_id)
                        match = "✓" if pred_id == true_id else "✗"
                        print(f"    Pos {pos_idx}: pred='{pred_str}' ({pred_id}), true='{true_str}' ({true_id}), prob={prob:.4f} {match}")
                    
                    if len(unique_preds) == 1:
                        print(f"    ⚠️  All positions predict the SAME token!")
                    else:
                        print(f"    ✓ Predicting {len(unique_preds)} different tokens")
            else:
                recon_loss = torch.tensor(0.0, device=device)
                step_info.update({
                    'recon_loss': 0.0,
                    'accuracy': 0.0,
                    'correct': 0,
                    'confidence': 0.0,
                    'top5_accuracy': 0.0
                })
            
            # Gate loss
            step_gate_targets = gate_targets[:, step_idx + 1].unsqueeze(1)
            gate_pred_valid = gate_pred[valid_mask]
            gate_target_valid = step_gate_targets[valid_mask]
            gate_loss = gate_criterion(gate_pred_valid, gate_target_valid) if len(gate_pred_valid) > 0 else torch.tensor(0.0, device=device)
            
            step_info['gate_loss'] = gate_loss.item()
            step_info['gate_pred_mean'] = gate_pred.mean().item()
            step_info['gate_target_mean'] = step_gate_targets.mean().item()
            
            batch_recon_tensor += recon_loss
            batch_gate_tensor += gate_loss
            num_valid += 1
            
            step_details.append(step_info)
            
            print(f"\nStep {step_idx + 1} (Recursive):")
            print(f"  Valid samples: {step_info['valid_samples']}/{len(chain_lengths)}")
            print(f"  Positions to predict: {step_info['positions_to_predict']}")
            if step_info['positions_to_predict'] > 0:
                print(f"  Recon loss: {step_info['recon_loss']:.4f}")
                print(f"  Accuracy: {step_info['accuracy']:.4f} ({step_info['correct']:.0f}/{step_info['positions_to_predict']})")
                print(f"  Top-5 Accuracy: {step_info['top5_accuracy']:.4f}")
                print(f"  Confidence: {step_info['confidence']:.4f}")
            print(f"  Gate loss: {step_info['gate_loss']:.4f}")
            print(f"  Gate pred: {step_info['gate_pred_mean']:.4f}, target: {step_info['gate_target_mean']:.4f}")
    
    # Final metrics
    if num_valid > 0:
        avg_recon = (batch_recon_tensor / num_valid).item()
        avg_gate = (batch_gate_tensor / num_valid).item()
        batch_acc = (batch_correct_tokens / (batch_target_tokens + 1e-6)).item()
    else:
        avg_recon = avg_gate = batch_acc = 0.0
    
    print("\n[FINAL RESULTS]")
    print(f"  Reconstruction Loss: {avg_recon:.4f}")
    print(f"  Gate Loss: {avg_gate:.4f}")
    print(f"  Accuracy (masked positions): {batch_acc:.4f}")
    print(f"  Total masked positions: {batch_target_tokens.item():.0f}")
    print(f"  Average chain length: {chain_lengths.float().mean().item():.2f}")
    
    return {
        'recon_loss': avg_recon,
        'gate_loss': avg_gate,
        'accuracy': batch_acc,
        'total_masked': batch_target_tokens.item(),
        'step_details': step_details
    }


def run_test_masking_with_gt_gate(model, tokenizer, config, test_texts, device, mask_ratio=0.3, training_batch=None):
    """
    Scenario 3: Test Masking with GT GATE (Control Experiment)
    
    Uses SAME input as training if training_batch is provided,
    otherwise creates its own masked input
    """
    print("\n" + "="*80)
    print(f"SCENARIO 3: Test Masking with GT GATE (mask_ratio={mask_ratio})")
    print("="*80)
    
    model.eval()
    
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id or 0
    special_token_ids = set(tokenizer.all_special_ids)
    
    max_seq_len = config['data']['max_seq_length']
    
    # Use GT gate schedule
    total_steps = config['training']['total_steps']
    estimated_step = int(total_steps * (1 - 0.7))
    gt_gate_start = (estimated_step / total_steps) * 20.0
    
    print(f"\n[GT GATE SETUP]")
    print(f"  Estimated denoising step for 30% masking: {estimated_step}/{total_steps}")
    print(f"  GT gate value: {gt_gate_start:.4f}")
    
    # ========================================================================
    # CRITICAL: Use SAME input as training for fair comparison
    # ========================================================================
    if training_batch is not None:
        print(f"\n[USING TRAINING BATCH INPUT]")
        print(f"  This ensures we compare on IDENTICAL inputs")
        
        batch_input_ids_tensor = training_batch['input'].to(device)
        batch_attention_masks_tensor = training_batch['attention_mask'].to(device)
        
        # For evaluation, we need to know what the original tokens were
        # We'll reconstruct from targets (step 1 has the unmasked tokens)
        # This is a bit hacky but necessary for fair comparison
        print(f"  Note: Using training preprocessed inputs")
        
        # We can't easily get eval_mask from training batch, so create new masks
        batch_eval_masks = []
        batch_original_tokens = []
        
        for i in range(batch_input_ids_tensor.size(0)):
            # Get the input
            input_seq = batch_input_ids_tensor[i].cpu()
            
            # Get original from training targets (last step should have originals)
            # This is approximate - for demo purposes
            # In reality we'd need the original unmasked sequence
            eval_mask = (input_seq == mask_token_id)
            batch_eval_masks.append(eval_mask)
            
            # Use targets from training as "original"
            # Note: This is not perfect but shows the concept
            original = input_seq.clone()
            batch_original_tokens.append(original)
        
        all_tokens = batch_original_tokens  # For iteration
        
    else:
        # Create new masked input
        all_tokens = []
        
        for text in test_texts[:32]:
            encoded = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=max_seq_len,
                padding='max_length'
            )
            tokens = encoded['input_ids'].squeeze(0)
            
            if len(tokens) >= 10:
                all_tokens.append(tokens)
        
        # Process batch
        batch_input_ids = []
        batch_eval_masks = []
        batch_original_tokens = []
        
        for tokens in all_tokens:
            masked_tokens, eval_mask = create_masked_input(tokens, mask_ratio, mask_token_id, special_token_ids)
            batch_input_ids.append(masked_tokens)
            batch_eval_masks.append(eval_mask)
            batch_original_tokens.append(tokens)
        
        # Pad
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
        
        batch_input_ids_tensor = torch.stack(padded_input_ids).to(device)
        batch_attention_masks_tensor = torch.stack(padded_attention_masks).to(device)
    
    print(f"\n[INFERENCE WITH GT GATE]")
    print(f"  Using fixed GT gate schedule")
    
    # ========================================================================
    # DETAILED LOGGING FOR FIRST SAMPLE
    # ========================================================================
    print(f"\n[DETAILED INFERENCE - FIRST SAMPLE WITH GT GATE]")
    
    sample_idx = 0
    sample_tokens = batch_input_ids_tensor[sample_idx:sample_idx+1]
    sample_mask = batch_attention_masks_tensor[sample_idx:sample_idx+1]
    original_tokens = batch_original_tokens[sample_idx]
    eval_mask = batch_eval_masks[sample_idx]
    
    print(f"Sample 0 Details:")
    print(f"  Original length: {len(original_tokens)}")
    print(f"  Masked positions: {eval_mask.sum().item()}")
    print(f"  Masked indices: {eval_mask.nonzero(as_tuple=True)[0].tolist()}")
    
    for idx in eval_mask.nonzero(as_tuple=True)[0][:5]:
        orig_token = tokenizer.decode([original_tokens[idx].item()])
        print(f"    Position {idx}: '{orig_token}' (token_id={original_tokens[idx].item()})")
    
    with torch.no_grad():
        # Step 0: Initial encoding
        print(f"\n--- Step 0: Initial Encoding ---")
        h_0 = model.encode_tokens(sample_tokens, sample_mask)
        gate_pred_0, pooled_0 = model.gate(h_0, sample_mask, None, None)
        
        print(f"  Hidden shape: {h_0.shape}")
        print(f"  Gate pred (ignored): {gate_pred_0.item():.4f}")
        print(f"  Using GT gate: {gt_gate_start:.4f}")
        
        # Use GT gate instead of predicted
        gt_gate_0 = torch.tensor([[gt_gate_start]], device=device)
        
        # First forward_step with GT gate
        print(f"\n--- Step 1: First Forward Step (GT Gate) ---")
        hidden, _, _ = model.forward_step(
            h_0,
            attention_mask=sample_mask,
            last_gate_score=gt_gate_0,
            last_pooled=pooled_0,
            gt_timestep=None,
            sampling_prob=0.0
        )
        
        gate_pred, pooled = model.gate(hidden, sample_mask, pooled_0, gt_gate_0)
        print(f"  Gate pred (ignored): {gate_pred.item():.4f}")
        
        # Decode and check
        logits = model.decode(hidden, sample_mask)
        pred_tokens = logits.argmax(dim=-1).squeeze(0).cpu()
        pred_probs = torch.softmax(logits.squeeze(0), dim=-1)
        
        print(f"  Predictions at masked positions:")
        unique_preds = set()
        for idx in eval_mask.nonzero(as_tuple=True)[0][:5]:
            if idx < len(pred_tokens):
                pred_token_id = pred_tokens[idx].item()
                true_token_id = original_tokens[idx].item()
                pred_prob = pred_probs[idx, pred_token_id].item()
                
                pred_token_str = tokenizer.decode([pred_token_id])
                true_token_str = tokenizer.decode([true_token_id])
                
                unique_preds.add(pred_token_id)
                match = "✓" if pred_token_id == true_token_id else "✗"
                print(f"    Pos {idx}: pred='{pred_token_str}' ({pred_token_id}), true='{true_token_str}' ({true_token_id}), prob={pred_prob:.4f} {match}")
        
        if len(unique_preds) == 1:
            print(f"  ⚠️  All positions predict the SAME token!")
        else:
            print(f"  ✓ Predicting {len(unique_preds)} different tokens")
        
        # Second forward_step with GT gate
        gt_gate_1 = torch.tensor([[max(0.0, gt_gate_start - 1.0)]], device=device)
        
        print(f"\n--- Step 2: Second Forward Step (GT Gate) ---")
        print(f"  Using GT gate: {gt_gate_1.item():.4f}")
        
        hidden, _, _ = model.forward_step(
            hidden,
            attention_mask=sample_mask,
            last_gate_score=gt_gate_1,
            last_pooled=pooled,
            gt_timestep=None,
            sampling_prob=0.0
        )
        
        # Final decode
        logits = model.decode(hidden, sample_mask)
        pred_tokens = logits.argmax(dim=-1).squeeze(0).cpu()
        pred_probs = torch.softmax(logits.squeeze(0), dim=-1)
        
        print(f"  Final predictions at masked positions:")
        unique_preds = set()
        for idx in eval_mask.nonzero(as_tuple=True)[0][:5]:
            if idx < len(pred_tokens):
                pred_token_id = pred_tokens[idx].item()
                true_token_id = original_tokens[idx].item()
                pred_prob = pred_probs[idx, pred_token_id].item()
                
                pred_token_str = tokenizer.decode([pred_token_id])
                true_token_str = tokenizer.decode([true_token_id])
                
                unique_preds.add(pred_token_id)
                match = "✓" if pred_token_id == true_token_id else "✗"
                print(f"    Pos {idx}: pred='{pred_token_str}' ({pred_token_id}), true='{true_token_str}' ({true_token_id}), prob={pred_prob:.4f} {match}")
        
        if len(unique_preds) == 1:
            print(f"  ⚠️  All positions predict the SAME token!")
        else:
            print(f"  ✓ Predicting {len(unique_preds)} different tokens")
        
        # Calculate accuracy
        if eval_mask.sum() > 0:
            correct = (pred_tokens[eval_mask] == original_tokens[eval_mask]).sum().item()
            total = eval_mask.sum().item()
            print(f"\n  Sample 0 final accuracy: {correct}/{total} = {correct/total:.4f}")
    
    print(f"\n[BATCH INFERENCE WITH GT GATE]")
    
    # Manual inference with GT gate for all samples
    all_output_tokens = []
    
    with torch.no_grad():
        for sample_idx in range(len(all_tokens)):
            sample_tokens = batch_input_ids_tensor[sample_idx:sample_idx+1]
            sample_mask = batch_attention_masks_tensor[sample_idx:sample_idx+1]
            
            # Step 0: Initial encoding
            h_0 = model.encode_tokens(sample_tokens, sample_mask)
            gate_pred_0, pooled_0 = model.gate(h_0, sample_mask, None, None)
            
            # Use GT gate instead of predicted
            gt_gate_0 = torch.tensor([[gt_gate_start]], device=device)
            
            # First forward_step with GT gate
            hidden, _, _ = model.forward_step(
                h_0,
                attention_mask=sample_mask,
                last_gate_score=gt_gate_0,  # GT gate
                last_pooled=pooled_0,
                gt_timestep=None,
                sampling_prob=0.0
            )
            
            # Subsequent step with decremented GT gate
            gt_gate_1 = torch.tensor([[max(0.0, gt_gate_start - 1.0)]], device=device)
            gate_pred, pooled = model.gate(hidden, sample_mask, pooled_0, gt_gate_0)
            
            # Second forward_step with GT gate
            hidden, _, _ = model.forward_step(
                hidden,
                attention_mask=sample_mask,
                last_gate_score=gt_gate_1,  # GT gate
                last_pooled=pooled,
                gt_timestep=None,
                sampling_prob=0.0
            )
            
            # Final decode
            logits = model.decode(hidden, sample_mask)
            sample_output = logits.argmax(dim=-1)
            
            all_output_tokens.append(sample_output)
    
    output_tokens = torch.cat(all_output_tokens, dim=0).cpu()
    
    # Calculate metrics
    exact_match_scores = []
    sample_details = []
    
    for j in range(len(all_tokens)):
        original_tokens = batch_original_tokens[j]
        eval_mask = batch_eval_masks[j]
        
        orig_len = len(original_tokens)
        pred_tokens = output_tokens[j][:orig_len]
        
        if eval_mask.sum() > 0:
            correct = (pred_tokens[eval_mask] == original_tokens[eval_mask]).sum().item()
            total = eval_mask.sum().item()
            accuracy = correct / total
        else:
            accuracy = 1.0
            correct = 0
            total = 0
        
        exact_match_scores.append(accuracy)
        
        sample_details.append({
            'sample_idx': j,
            'seq_len': orig_len,
            'num_masked': eval_mask.sum().item(),
            'correct': correct,
            'total': total,
            'accuracy': accuracy
        })
    
    print(f"\n[SAMPLE DETAILS (first 5)]")
    for detail in sample_details[:5]:
        print(f"  Sample {detail['sample_idx']}: len={detail['seq_len']}, masked={detail['num_masked']}, "
              f"correct={detail['correct']}/{detail['total']}, acc={detail['accuracy']:.4f}")
    
    # Aggregate
    avg_accuracy = np.mean(exact_match_scores)
    
    print(f"\n[FINAL RESULTS]")
    print(f"  Accuracy (masked positions): {avg_accuracy:.4f}")
    
    return {
        'accuracy': avg_accuracy,
        'sample_details': sample_details
    }


def run_test_masking(model, tokenizer, config, test_texts, device, mask_ratio=0.3):
    """Scenario 2: Test Masking with DETAILED inference debugging"""
    print("\n" + "="*80)
    print(f"SCENARIO 2: Test Masking (test_masking.py, mask_ratio={mask_ratio})")
    print("="*80)
    
    model.eval()
    
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id or 0
    special_token_ids = set(tokenizer.all_special_ids)
    
    max_seq_len = config['data']['max_seq_length']
    max_steps = 10
    threshold = 0.5
    
    # Tokenize
    all_tokens = []
    all_texts = []
    
    for text in test_texts[:32]:
        encoded = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_seq_len,
            padding='max_length'
        )
        tokens = encoded['input_ids'].squeeze(0)
        
        if len(tokens) >= 10:
            all_tokens.append(tokens)
            all_texts.append(tokenizer.decode(tokens, skip_special_tokens=True))
    
    print(f"\n[TOKENIZATION]")
    print(f"  Num samples: {len(all_tokens)}")
    print(f"  Sequence lengths: min={min(len(t) for t in all_tokens)}, max={max(len(t) for t in all_tokens)}, mean={np.mean([len(t) for t in all_tokens]):.1f}")
    
    # Process batch
    batch_tokens = all_tokens
    batch_input_ids = []
    batch_eval_masks = []
    batch_original_tokens = []
    
    total_maskable = 0
    total_masked = 0
    
    for tokens in batch_tokens:
        masked_tokens, eval_mask = create_masked_input(tokens, mask_ratio, mask_token_id, special_token_ids)
        
        batch_input_ids.append(masked_tokens)
        batch_eval_masks.append(eval_mask)
        batch_original_tokens.append(tokens)
        
        maskable = sum(1 for t in tokens if t.item() not in special_token_ids)
        total_maskable += maskable
        total_masked += eval_mask.sum().item()
    
    print(f"\n[MASKING STATISTICS]")
    print(f"  Total maskable tokens: {total_maskable}")
    print(f"  Total masked tokens: {total_masked}")
    print(f"  Actual mask ratio: {total_masked/total_maskable:.4f}")
    
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
    
    print(f"\n[BATCH INFO]")
    print(f"  Input shape: {batch_input_ids_tensor.shape}")
    print(f"  Attention mask shape: {batch_attention_masks_tensor.shape}")
    print(f"  Valid tokens: {batch_attention_masks_tensor.sum().item()}")
    
    # ========================================================================
    # MANUAL INFERENCE with DETAILED LOGGING (first sample only for clarity)
    # ========================================================================
    print(f"\n[DETAILED INFERENCE - FIRST SAMPLE]")
    print(f"  Max steps: {max_steps}")
    print(f"  Threshold: {threshold}")
    
    sample_idx = 0
    x = batch_input_ids_tensor
    attention_mask = batch_attention_masks_tensor
    
    with torch.no_grad():
        # Get first sample for detailed logging
        sample_tokens = x[sample_idx:sample_idx+1]
        sample_mask = attention_mask[sample_idx:sample_idx+1]
        original_tokens = batch_original_tokens[sample_idx]
        eval_mask = batch_eval_masks[sample_idx]
        
        print(f"\nSample 0 Details:")
        print(f"  Original length: {len(original_tokens)}")
        print(f"  Masked positions: {eval_mask.sum().item()}")
        print(f"  Masked indices: {eval_mask.nonzero(as_tuple=True)[0].tolist()}")
        
        # Show what's masked
        for idx in eval_mask.nonzero(as_tuple=True)[0][:5]:  # First 5
            orig_token = tokenizer.decode([original_tokens[idx].item()])
            print(f"    Position {idx}: '{orig_token}' (token_id={original_tokens[idx].item()})")
        
        # Step 0: Initial encoding
        print(f"\n--- Step 0: Initial Encoding ---")
        h_0 = model.encode_tokens(sample_tokens, sample_mask)
        gate_pred_0, pooled_0 = model.gate(h_0, sample_mask, None, None)
        
        print(f"  Hidden shape: {h_0.shape}")
        print(f"  Gate pred: {gate_pred_0.item():.4f}")
        print(f"  Hidden stats: mean={h_0.mean().item():.4f}, std={h_0.std().item():.4f}")
        print(f"  Pooled stats: mean={pooled_0.mean().item():.4f}, std={pooled_0.std().item():.4f}")
        
        # First forward_step
        print(f"\n--- Step 1: First Forward Step ---")
        hidden, _, _ = model.forward_step(
            h_0,
            attention_mask=sample_mask,
            last_gate_score=gate_pred_0,
            last_pooled=pooled_0,
            gt_timestep=None,
            sampling_prob=0.0
        )
        
        print(f"  Hidden after transform: mean={hidden.mean().item():.4f}, std={hidden.std().item():.4f}")
        
        gate_pred, pooled = model.gate(hidden, sample_mask, pooled_0, gate_pred_0)
        print(f"  Gate pred after transform: {gate_pred.item():.4f}")
        print(f"  Pooled after gate: mean={pooled.mean().item():.4f}, std={pooled.std().item():.4f}")
        
        # CRITICAL: Check if gate network is broken
        if gate_pred.item() == 0.0:
            print(f"  ⚠️  WARNING: Gate immediately dropped to 0.0!")
            print(f"  This suggests gate network may not be properly trained")
            print(f"  Previous gate: {gate_pred_0.item():.4f}")
            print(f"  Gate delta: {gate_pred.item() - gate_pred_0.item():.4f}")
        
        # Decode and check predictions
        logits = model.decode(hidden, sample_mask)
        pred_tokens = logits.argmax(dim=-1).squeeze(0)
        pred_probs = torch.softmax(logits.squeeze(0), dim=-1)
        
        print(f"  Logits stats: mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")
        print(f"  Predictions at masked positions:")
        
        unique_preds = set()
        for idx in eval_mask.nonzero(as_tuple=True)[0][:5]:
            if idx < len(pred_tokens):
                pred_token_id = pred_tokens[idx].item()
                true_token_id = original_tokens[idx].item()
                pred_prob = pred_probs[idx, pred_token_id].item()
                
                pred_token_str = tokenizer.decode([pred_token_id])
                true_token_str = tokenizer.decode([true_token_id])
                
                unique_preds.add(pred_token_id)
                match = "✓" if pred_token_id == true_token_id else "✗"
                print(f"    Pos {idx}: pred='{pred_token_str}' ({pred_token_id}), true='{true_token_str}' ({true_token_id}), prob={pred_prob:.4f} {match}")
        
        if len(unique_preds) == 1:
            print(f"  ⚠️  WARNING: All masked positions predict the SAME token!")
            print(f"  This suggests model is not attending to position-specific context")
        
        steps_taken = 1
        
        # Continue iterations
        for step in range(1, max_steps):
            gate_score = gate_pred.squeeze(-1).item()
            
            print(f"\n--- Step {step + 1}: Iteration ---")
            print(f"  Gate score: {gate_score:.4f}")
            
            if gate_score < threshold:
                print(f"  → Converged (gate < threshold)")
                break
            
            hidden, _, _ = model.forward_step(
                hidden,
                attention_mask=sample_mask,
                last_gate_score=gate_pred,
                last_pooled=pooled,
                gt_timestep=None,
                sampling_prob=0.0
            )
            
            gate_pred, pooled = model.gate(hidden, sample_mask, pooled, gate_pred)
            
            # Decode and check
            logits = model.decode(hidden, sample_mask)
            pred_tokens = logits.argmax(dim=-1).squeeze(0)
            pred_probs = torch.softmax(logits.squeeze(0), dim=-1)
            
            print(f"  Gate pred after transform: {gate_pred.item():.4f}")
            print(f"  Predictions at masked positions:")
            for idx in eval_mask.nonzero(as_tuple=True)[0][:5]:
                if idx < len(pred_tokens):
                    pred_token_id = pred_tokens[idx].item()
                    true_token_id = original_tokens[idx].item()
                    pred_prob = pred_probs[idx, pred_token_id].item()
                    
                    pred_token_str = tokenizer.decode([pred_token_id])
                    true_token_str = tokenizer.decode([true_token_id])
                    
                    match = "✓" if pred_token_id == true_token_id else "✗"
                    print(f"    Pos {idx}: pred='{pred_token_str}' ({pred_token_id}), true='{true_token_str}' ({true_token_id}), prob={pred_prob:.4f} {match}")
            
            steps_taken += 1
        
        print(f"\nFinal step count: {steps_taken}")
        
        # Final accuracy for this sample
        final_logits = model.decode(hidden, sample_mask)
        final_pred = final_logits.argmax(dim=-1).squeeze(0).cpu()  # CPU로 이동
        
        # eval_mask는 원본 길이, final_pred는 padded 길이일 수 있음
        orig_len = len(original_tokens)
        final_pred_trimmed = final_pred[:orig_len]
        
        if eval_mask.sum() > 0:
            correct = (final_pred_trimmed[eval_mask] == original_tokens[eval_mask]).sum().item()
            total = eval_mask.sum().item()
            print(f"Final accuracy: {correct}/{total} = {correct/total:.4f}")
        else:
            print(f"No masked positions to evaluate")
        
    # ========================================================================
    # Run full batch inference (using original method)
    # ========================================================================
    print(f"\n[BATCH INFERENCE]")
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
    
    print(f"  Steps taken: min={steps_batch.min().item()}, max={steps_batch.max().item()}, mean={steps_batch.float().mean().item():.2f}")
    
    # Calculate metrics per sample
    exact_match_scores = []
    sample_details = []
    
    for j in range(len(batch_tokens)):
        original_tokens = batch_original_tokens[j]
        eval_mask = batch_eval_masks[j]
        
        orig_len = len(original_tokens)
        pred_tokens = pred_tokens_batch[j][:orig_len]
        num_steps = steps_batch[j].item()
        
        if eval_mask.sum() > 0:
            correct = (pred_tokens[eval_mask] == original_tokens[eval_mask]).sum().item()
            total = eval_mask.sum().item()
            accuracy = correct / total
        else:
            accuracy = 1.0
            correct = 0
            total = 0
        
        exact_match_scores.append(accuracy)
        
        sample_details.append({
            'sample_idx': j,
            'seq_len': orig_len,
            'num_masked': eval_mask.sum().item(),
            'correct': correct,
            'total': total,
            'accuracy': accuracy,
            'steps': num_steps
        })
    
    print(f"\n[SAMPLE DETAILS (first 5)]")
    for detail in sample_details[:5]:
        print(f"  Sample {detail['sample_idx']}: len={detail['seq_len']}, masked={detail['num_masked']}, "
              f"correct={detail['correct']}/{detail['total']}, acc={detail['accuracy']:.4f}, steps={detail['steps']}")
    
    # Aggregate
    avg_accuracy = np.mean(exact_match_scores)
    avg_steps = np.mean(steps_batch.numpy())
    
    print(f"\n[FINAL RESULTS]")
    print(f"  Accuracy (masked positions): {avg_accuracy:.4f}")
    print(f"  Average steps: {avg_steps:.2f}")
    
    return {
        'accuracy': avg_accuracy,
        'avg_steps': avg_steps,
        'sample_details': sample_details
    }


def detailed_comparison(results1, results2, results3):
    """Detailed comparison including GT gate control"""
    print("\n" + "="*80)
    print("DETAILED COMPARISON")
    print("="*80)
    
    print(f"\n{'Metric':<30} {'Train Validate':<20} {'Test (Pred Gate)':<20} {'Test (GT Gate)':<20}")
    print("-"*100)
    
    print(f"{'Accuracy':<30} {results1['accuracy']:<20.4f} {results2['accuracy']:<20.4f} {results3['accuracy']:<20.4f}")
    
    print("\n" + "="*100)
    
    print("\n[KEY FINDINGS]")
    
    # Compare predicted gate vs GT gate
    pred_gate_acc = results2['accuracy']
    gt_gate_acc = results3['accuracy']
    
    if pred_gate_acc == 0.0 and gt_gate_acc > 0.0:
        print("✓ CONFIRMED: Gate collapse is the PRIMARY issue")
        print(f"  - With predicted gate: {pred_gate_acc:.4f} (model stops too early)")
        print(f"  - With GT gate: {gt_gate_acc:.4f} (model can denoise)")
        print("  → Gate network is NOT properly trained for inference")
    elif gt_gate_acc == 0.0:
        print("✗ PROBLEM: Even with GT gate, accuracy is 0%")
        print("  → Fundamental model capability issue, not just gate")
    elif pred_gate_acc > 0.0 and gt_gate_acc > pred_gate_acc:
        print("⚠️  PARTIAL: Gate helps but both have issues")
        print(f"  - Improvement with GT gate: {(gt_gate_acc - pred_gate_acc):.4f}")
    
    print("\n[COMPARISON WITH TRAINING]")
    train_acc = results1['accuracy']
    
    if gt_gate_acc < train_acc:
        gap = train_acc - gt_gate_acc
        print(f"Gap between training and GT gate test: {gap:.4f} ({gap/train_acc*100:.1f}%)")
        print("Possible reasons:")
        print("  1. Different masking distributions (chain-based vs fixed-ratio)")
        print("  2. Different data (preprocessed vs raw)")
        print("  3. Training uses GT timestep in forward_step, test doesn't")
    else:
        print(f"GT gate test matches or exceeds training performance")
        print("The model works - gate is the only issue")


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
    results3 = run_test_masking_with_gt_gate(model, tokenizer, config, test_texts, device, mask_ratio=0.3)
    
    detailed_comparison(results1, results2, results3)
    
    print("\n" + "="*80)
    print("DEBUGGING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()