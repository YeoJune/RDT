"""
Precision Debugging: Training Validate vs Test Masking
버그 수준의 성능 차이를 분석하기 위한 정밀 재현 코드

두 상황을 완전히 동일하게 재현:
1. Train 시의 Validate (trainer.validate())
2. Test Masking (test_masking.py의 test_rdt_model())
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
import yaml

# Import using rdt module pattern (same as train.py and test_masking.py)
from rdt.models import RDT
from rdt.utils import load_config
from rdt.data.preprocessing import RDTPreprocessor


def create_masked_input(tokens, mask_ratio, mask_token_id, special_token_ids=None):
    """Create masked input with specified ratio, excluding special tokens"""
    seq_len = len(tokens)
    
    # Identify maskable positions (exclude special tokens)
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
    
    # Random masking positions from maskable positions only
    mask_positions_indices = torch.randperm(num_maskable)[:num_mask]
    mask_indices = maskable_positions[mask_positions_indices]
    
    masked_tokens = tokens.clone()
    masked_tokens[mask_indices] = mask_token_id
    
    # Create mask for evaluation
    eval_mask = torch.zeros(seq_len, dtype=torch.bool)
    eval_mask[mask_indices] = True
    
    return masked_tokens, eval_mask


def load_model_and_tokenizer(config_path: str, checkpoint_path: str, device: str):
    """Load RDT model and tokenizer from config and checkpoint"""
    config = load_config(config_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])
    vocab_size = tokenizer.vocab_size
    
    # Create model - exact parameters from provided RDT code
    model = RDT(
        vocab_size=vocab_size,
        d_model=config['model']['d_model'],
        n_heads=config['model']['num_heads'],
        n_encoder_layers=config['model']['num_layers'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout'],
        max_seq_len=config['model']['max_seq_length'],
        # Transformer I/O Configuration
        input_processor_layers=config['model'].get('input_processor_layers', 1),
        output_processor_layers=config['model'].get('output_processor_layers', 1),
        # Gate Configuration
        gate_hidden_dim=config['model'].get('gate_hidden_dim', 512),
        gate_num_layers=config['model'].get('gate_num_layers', 3),
        gate_num_heads=config['model'].get('gate_num_heads', 8),
        gate_dropout=config['model'].get('gate_dropout', 0.3),
        # RoPE Configuration
        rope_base=config['model'].get('rope_base', 10000.0),
        # Training
        gradient_checkpointing=False  # Disabled for inference
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, config


def prepare_test_data(config: dict, tokenizer, num_samples: int = 100):
    """Load test data - simple version without streaming dataset dependency"""
    print("\nLoading test data...")
    
    # For debugging, we'll use a simple text generation instead of actual dataset
    # This ensures the test runs without dataset loading issues
    
    # Create dummy test texts that are realistic
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
    
    # Replicate to get desired number of samples
    texts = []
    while len(texts) < num_samples:
        texts.extend(sample_texts)
    
    texts = texts[:num_samples]
    
    print(f"Loaded {len(texts)} test texts")
    return texts


def run_train_validate(model, tokenizer, config, test_texts, device):
    """
    Scenario 1: Training시의 Validate 재현
    
    Training에서는 RDTPreprocessor를 사용해 chain을 생성하고,
    각 chain step마다 forward_step + decode를 수행
    
    이것은 RDTTrainer.validate() 메서드와 동일한 로직
    """
    print("\n" + "="*80)
    print("SCENARIO 1: Training Validate (trainer.validate())")
    print("="*80)
    
    model.eval()
    
    # Create RDT Preprocessor (collator)
    preprocessor = RDTPreprocessor(tokenizer, config, device=device)
    
    # Prepare batch from test texts
    batch_data = []
    for text in test_texts[:32]:  # Use batch size 32
        input_ids = tokenizer(
            text,
            max_length=config['data']['max_seq_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )['input_ids'].squeeze(0)
        
        batch_data.append({'input_ids': input_ids})
    
    # Apply RDT preprocessing (exactly as in training)
    batch = preprocessor(batch_data)
    
    # Move to device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    
    print("\nBatch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
    
    # ================================================================
    # Training-style forward pass with chain processing
    # ================================================================
    with torch.no_grad():
        input_tokens = batch['input']  # [B, L]
        targets = batch['targets']  # [B, max_chain_len, L]
        loss_masks = batch['loss_masks']  # [B, max_chain_len, L]
        gate_targets = batch['gate_targets']  # [B, max_chain_len + 1]
        attention_mask = batch['attention_mask']  # [B, L]
        chain_lengths = batch['chain_lengths']  # [B]
        
        B, max_chain_len, L = targets.shape
        V = tokenizer.vocab_size
        
        # Initialize
        all_logits = []
        all_gate_preds = []
        
        # ============================================================
        # Step 0: Initial encoding
        # ============================================================
        hidden = model.encode_tokens(input_tokens, attention_mask)  # [B, L, D]
        
        # Initial gate
        gate_pred, pooled = model.gate(
            hidden,
            attention_mask,
            prev_pooled=None,
            prev_gate=None
        )
        all_gate_preds.append(gate_pred)
        
        # ============================================================
        # Steps 1 to max_chain_len: Recursive processing
        # ============================================================
        for step in range(max_chain_len):
            # Forward step (transformation)
            hidden, _, _ = model.forward_step(
                hidden,
                attention_mask=attention_mask,
                last_gate_score=gate_pred,
                last_pooled=pooled,
                gt_timestep=gate_targets[:, step:step+1],  # GT for this step
                sampling_prob=0.0  # No sampling in validation
            )
            
            # Decode to logits
            logits = model.decode(hidden, attention_mask)  # [B, L, V]
            all_logits.append(logits)
            
            # Next gate prediction
            gate_pred, pooled = model.gate(
                hidden,
                attention_mask,
                prev_pooled=pooled,
                prev_gate=gate_pred
            )
            all_gate_preds.append(gate_pred)
        
        # Stack logits: [B, max_chain_len, L, V]
        logits = torch.stack(all_logits, dim=1)
        
        # Stack gate predictions: [B, max_chain_len + 1]
        gate_logits = torch.cat(all_gate_preds, dim=1)
    
    # ================================================================
    # Calculate metrics (same as trainer.validate())
    # ================================================================
    # Reconstruction loss
    logits_flat = logits.reshape(B * max_chain_len * L, V)
    targets_flat = targets.reshape(B * max_chain_len * L)
    loss_mask_flat = loss_masks.reshape(B * max_chain_len * L)
    
    recon_criterion = nn.CrossEntropyLoss(reduction='none')
    recon_loss_all = recon_criterion(logits_flat, targets_flat)
    recon_loss_all = recon_loss_all.reshape(B, max_chain_len, L)
    
    # Apply mask
    masked_recon_loss = recon_loss_all * loss_masks.float()
    recon_loss = masked_recon_loss.sum() / (loss_masks.sum() + 1e-8)
    
    # Gate loss
    gate_criterion = nn.MSELoss()
    gate_loss = gate_criterion(gate_logits, gate_targets)
    
    # Accuracy on masked positions
    pred_tokens = torch.argmax(logits, dim=-1)
    correct = (pred_tokens == targets) & loss_masks
    accuracy = correct.sum().float() / (loss_masks.sum() + 1e-8)
    
    print("\nTraining Validate Results:")
    print(f"  Reconstruction Loss: {recon_loss.item():.4f}")
    print(f"  Gate Loss: {gate_loss.item():.4f}")
    print(f"  Accuracy (masked positions): {accuracy.item():.4f}")
    print(f"  Total masked positions: {loss_masks.sum().item()}")
    
    # Detailed analysis
    print("\nDetailed Analysis:")
    print(f"  Average chain length: {chain_lengths.float().mean().item():.2f}")
    print(f"  Mask ratio per sample: {loss_masks.sum(dim=(1,2)).float().mean().item() / L:.4f}")
    
    return {
        'recon_loss': recon_loss.item(),
        'gate_loss': gate_loss.item(),
        'accuracy': accuracy.item(),
        'total_masked': loss_masks.sum().item()
    }


def run_test_masking(model, tokenizer, config, test_texts, device, mask_ratio=0.3):
    """
    Scenario 2: Test Masking 재현
    
    이것은 test_masking.py의 test_rdt_model()과 동일한 로직
    정확한 RDT inference() 메서드 사용
    """
    print("\n" + "="*80)
    print(f"SCENARIO 2: Test Masking (test_masking.py, mask_ratio={mask_ratio})")
    print("="*80)
    
    model.eval()
    
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id or 0
    special_token_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id, pad_token_id}
    
    # Process samples
    all_accuracies = []
    all_recon_losses = []
    total_steps_list = []
    
    batch_size = 32
    for i in range(0, min(len(test_texts), 32), batch_size):
        batch_texts = test_texts[i:i+batch_size]
        
        # Tokenize
        batch_inputs = tokenizer(
            batch_texts,
            max_length=config['data']['max_seq_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = batch_inputs['input_ids'].to(device)
        attention_mask = batch_inputs['attention_mask'].to(device)
        
        B, L = input_ids.shape
        
        # Create masked inputs
        masked_inputs = torch.zeros_like(input_ids)
        eval_masks = torch.zeros_like(input_ids, dtype=torch.bool)
        
        for j in range(B):
            masked_inp, eval_mask = create_masked_input(
                input_ids[j], mask_ratio, mask_token_id, special_token_ids
            )
            masked_inputs[j] = masked_inp
            eval_masks[j] = eval_mask
        
        print(f"\nBatch {i//batch_size + 1}:")
        print(f"  Input shape: {masked_inputs.shape}")
        print(f"  Masked positions: {eval_masks.sum().item()}")
        print(f"  Mask ratio: {eval_masks.sum().item() / (attention_mask.sum().item() + 1e-8):.4f}")
        
        # ================================================================
        # Use model.inference() - exact same as test_masking.py
        # ================================================================
        max_steps = 20
        threshold = 0.02  # Same threshold as in inference method
        
        with torch.no_grad():
            # Model's built-in inference with per-sample processing
            output_tokens, steps_taken = model.inference(
                x=masked_inputs,
                attention_mask=attention_mask,
                max_steps=max_steps,
                threshold=threshold,
                return_steps=True  # Get per-sample steps
            )
        
        # steps_taken is [B] tensor
        step_count = steps_taken.float().mean().item()
        
        print(f"  Steps taken: {step_count:.2f}")
        
        # ================================================================
        # Calculate metrics on masked positions
        # ================================================================
        # Accuracy
        correct = (output_tokens == input_ids) & eval_masks
        batch_accuracy = correct.sum().float() / (eval_masks.sum() + 1e-8)
        
        # Reconstruction loss
        # Need to get logits for the final output
        with torch.no_grad():
            # Encode final output tokens
            hidden = model.encode_tokens(output_tokens, attention_mask)
            
            # Decode to logits
            final_logits = model.decode(hidden, attention_mask)  # [B, L, V]
            
            # Cross-entropy loss on evaluated positions
            recon_criterion = nn.CrossEntropyLoss(reduction='none')
            losses = recon_criterion(
                final_logits.reshape(-1, final_logits.size(-1)),
                input_ids.reshape(-1)
            ).reshape(B, L)
            
            masked_losses = losses * eval_masks.float()
            batch_recon_loss = masked_losses.sum() / (eval_masks.sum() + 1e-8)
        
        all_accuracies.append(batch_accuracy.item())
        all_recon_losses.append(batch_recon_loss.item())
        total_steps_list.append(step_count)
        
        print(f"  Accuracy: {batch_accuracy.item():.4f}")
        print(f"  Recon Loss: {batch_recon_loss.item():.4f}")
    
    # Aggregate results
    avg_accuracy = np.mean(all_accuracies)
    avg_recon_loss = np.mean(all_recon_losses)
    avg_steps = np.mean(total_steps_list)
    
    print("\nTest Masking Results:")
    print(f"  Reconstruction Loss: {avg_recon_loss:.4f}")
    print(f"  Accuracy (masked positions): {avg_accuracy:.4f}")
    print(f"  Average steps: {avg_steps:.2f}")
    
    return {
        'recon_loss': avg_recon_loss,
        'accuracy': avg_accuracy,
        'avg_steps': avg_steps
    }


def detailed_comparison(results1, results2):
    """Detailed comparison between two scenarios"""
    print("\n" + "="*80)
    print("DETAILED COMPARISON")
    print("="*80)
    
    print(f"\n{'Metric':<30} {'Train Validate':<20} {'Test Masking':<20} {'Difference':<20}")
    print("-"*90)
    
    # Accuracy
    acc_diff = results2['accuracy'] - results1['accuracy']
    acc_diff_pct = (acc_diff / results1['accuracy']) * 100 if results1['accuracy'] > 0 else 0
    print(f"{'Accuracy':<30} {results1['accuracy']:<20.4f} {results2['accuracy']:<20.4f} {acc_diff:+.4f} ({acc_diff_pct:+.1f}%)")
    
    # Recon Loss
    loss_diff = results2['recon_loss'] - results1['recon_loss']
    loss_diff_pct = (loss_diff / results1['recon_loss']) * 100 if results1['recon_loss'] > 0 else 0
    print(f"{'Reconstruction Loss':<30} {results1['recon_loss']:<20.4f} {results2['recon_loss']:<20.4f} {loss_diff:+.4f} ({loss_diff_pct:+.1f}%)")
    
    print("\n" + "="*80)
    
    # Analysis
    print("\nANALYSIS:")
    
    if abs(acc_diff) > 0.1:
        print(f"⚠️  CRITICAL: Accuracy difference of {abs(acc_diff):.4f} ({abs(acc_diff_pct):.1f}%) is significant!")
    
    if abs(loss_diff) > 0.5:
        print(f"⚠️  CRITICAL: Loss difference of {abs(loss_diff):.4f} ({abs(loss_diff_pct):.1f}%) is significant!")
    
    print("\nPossible causes:")
    print("1. Different data preprocessing (RDTPreprocessor vs create_masked_input)")
    print("2. Different masking strategies (chain-based vs fixed ratio)")
    print("3. Different evaluation methods (single step vs iterative refinement)")
    print("4. Different model behaviors (training mode vs inference mode)")
    print("5. Different loss calculation methods")


def main():
    """Main debugging script"""
    # Configuration
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
    print(f"Num samples: {num_samples}")
    
    # Load model
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    model, tokenizer, config = load_model_and_tokenizer(config_path, checkpoint_path, device)
    print(f"Model loaded successfully")
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # Load test data
    test_texts = prepare_test_data(config, tokenizer, num_samples)
    
    # Run both scenarios
    results1 = run_train_validate(model, tokenizer, config, test_texts, device)
    results2 = run_test_masking(model, tokenizer, config, test_texts, device, mask_ratio=0.3)
    
    # Compare
    detailed_comparison(results1, results2)
    
    print("\n" + "="*80)
    print("DEBUGGING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()