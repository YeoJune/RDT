"""Test RoBERTa-base reconstruction capability across different masking levels"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, RobertaForMaskedLM
from tqdm import tqdm
from pathlib import Path


def create_masked_input(tokens, mask_ratio, mask_token_id):
    """Create masked input with specified ratio"""
    seq_len = len(tokens)
    num_mask = int(seq_len * mask_ratio)
    
    if num_mask == 0:
        return tokens.clone(), torch.zeros(seq_len, dtype=torch.bool)
    
    # Random masking positions
    mask_indices = torch.randperm(seq_len)[:num_mask]
    
    masked_tokens = tokens.clone()
    masked_tokens[mask_indices] = mask_token_id
    
    # Create mask for evaluation
    eval_mask = torch.zeros(seq_len, dtype=torch.bool)
    eval_mask[mask_indices] = True
    
    return masked_tokens, eval_mask


def calculate_accuracy(pred_tokens, target_tokens, eval_mask):
    """Calculate accuracy on masked positions"""
    if eval_mask.sum() == 0:
        return 1.0
    
    masked_pred = pred_tokens[eval_mask]
    masked_target = target_tokens[eval_mask]
    
    correct = (masked_pred == masked_target).sum().item()
    total = eval_mask.sum().item()
    
    return correct / total


def roberta_single_pass_inference(model, input_ids, attention_mask):
    """Single forward pass through RoBERTa"""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred_tokens = torch.argmax(logits, dim=-1)
    return pred_tokens


def roberta_iterative_inference(model, input_ids, attention_mask, mask_token_id, max_steps=20, threshold=0.95):
    """Iterative refinement similar to RDT"""
    current_ids = input_ids.clone()
    
    for step in range(max_steps):
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=current_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_tokens = torch.argmax(logits, dim=-1)
            max_probs = torch.max(probs, dim=-1)[0]
        
        # Find masked positions
        mask_positions = (current_ids == mask_token_id)
        
        if not mask_positions.any():
            return current_ids, step + 1
        
        # Get confidence for masked positions
        masked_probs = max_probs[mask_positions]
        
        # Update high-confidence predictions
        high_conf_mask = masked_probs > threshold
        
        if not high_conf_mask.any():
            # If no high confidence, update all
            current_ids[mask_positions] = pred_tokens[mask_positions]
            return current_ids, step + 1
        
        # Update only high-confidence positions
        mask_positions_idx = torch.where(mask_positions)[1]
        high_conf_idx = mask_positions_idx[high_conf_mask]
        
        for idx in high_conf_idx:
            current_ids[0, idx] = pred_tokens[0, idx]
    
    # Final pass to fill any remaining masks
    mask_positions = (current_ids == mask_token_id)
    if mask_positions.any():
        with torch.no_grad():
            outputs = model(input_ids=current_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred_tokens = torch.argmax(logits, dim=-1)
        current_ids[mask_positions] = pred_tokens[mask_positions]
    
    return current_ids, max_steps


def test_roberta(model, tokenizer, test_texts, mask_ratios, device, max_seq_len, mode='single', max_steps=20, threshold=0.95):
    """Test RoBERTa across different masking levels"""
    model.eval()
    mask_token_id = tokenizer.mask_token_id
    
    results = {ratio: [] for ratio in mask_ratios}
    steps_taken = {ratio: [] for ratio in mask_ratios}
    
    mode_name = "Single-pass" if mode == 'single' else "Iterative"
    print(f"\nTesting RoBERTa {mode_name} reconstruction (max_seq_len={max_seq_len})...")
    
    for text in tqdm(test_texts, desc="Processing texts"):
        # Tokenize
        encoded = tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=max_seq_len,
            padding='max_length'
        )
        tokens = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # Filter out padding for evaluation
        valid_positions = attention_mask.bool()
        valid_tokens = tokens[valid_positions]
        
        if len(valid_tokens) < 10:
            continue
        
        # Test each masking ratio
        for ratio in mask_ratios:
            # Create masked input (only on valid positions)
            masked_tokens, eval_mask = create_masked_input(valid_tokens, ratio, mask_token_id)
            
            # Put back into full sequence
            full_masked = tokens.clone()
            full_masked[valid_positions] = masked_tokens
            
            input_ids = full_masked.unsqueeze(0).to(device)
            attn_mask = attention_mask.unsqueeze(0).to(device)
            
            # Inference
            if mode == 'single':
                pred_tokens_full = roberta_single_pass_inference(model, input_ids, attn_mask)
                num_steps = 1
            else:
                pred_tokens_full, num_steps = roberta_iterative_inference(
                    model, input_ids, attn_mask, mask_token_id, max_steps, threshold
                )
            
            # Extract predictions on valid positions
            pred_tokens = pred_tokens_full.squeeze(0).cpu()[valid_positions]
            
            # Calculate accuracy
            accuracy = calculate_accuracy(pred_tokens, valid_tokens, eval_mask)
            
            results[ratio].append(accuracy)
            steps_taken[ratio].append(num_steps)
    
    # Aggregate results
    avg_results = {}
    avg_steps = {}
    
    for ratio in mask_ratios:
        if results[ratio]:
            avg_results[ratio] = np.mean(results[ratio])
            avg_steps[ratio] = np.mean(steps_taken[ratio])
        else:
            avg_results[ratio] = 0.0
            avg_steps[ratio] = 0.0
    
    return avg_results, avg_steps


def visualize_results(mask_ratios, accuracies, steps, mode='single', save_path=None):
    """Visualize test results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    mode_name = "Single-pass" if mode == 'single' else "Iterative"
    
    # Plot 1: Accuracy vs Masking Ratio
    mask_percentages = [r * 100 for r in mask_ratios]
    ax1.plot(mask_percentages, list(accuracies.values()), 'g-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Masking Ratio (%)', fontsize=12)
    ax1.set_ylabel('Reconstruction Accuracy', fontsize=12)
    ax1.set_title(f'RoBERTa-base {mode_name} Reconstruction Accuracy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Add value labels
    for ratio, acc in accuracies.items():
        x = ratio * 100
        ax1.annotate(f'{acc:.3f}', (x, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # Plot 2: Steps Taken vs Masking Ratio
    ax2.plot(mask_percentages, list(steps.values()), 'purple', marker='s', linewidth=2, markersize=8)
    ax2.set_xlabel('Masking Ratio (%)', fontsize=12)
    ax2.set_ylabel('Average Steps Taken', fontsize=12)
    ax2.set_title(f'{mode_name} Inference Steps', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for ratio, step in steps.items():
        x = ratio * 100
        ax2.annotate(f'{step:.1f}', (x, step), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
    
    plt.show()


def load_test_texts(tokenizer, split='test', num_samples=100):
    """Load test texts from WikiText"""
    from datasets import load_dataset
    
    print(f"Loading test data...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
    
    texts = []
    for item in dataset:
        text = item['text'].strip()
        if len(text) > 50:
            texts.append(text)
        if len(texts) >= num_samples:
            break
    
    return texts


def main():
    parser = argparse.ArgumentParser(description='Test RoBERTa-base Masking Reconstruction')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of test samples')
    parser.add_argument('--max_seq_len', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'iterative'],
                        help='Inference mode: single-pass or iterative')
    parser.add_argument('--max_steps', type=int, default=20,
                        help='Maximum inference steps (for iterative mode)')
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Confidence threshold (for iterative mode)')
    parser.add_argument('--output', type=str, default='roberta_baseline_results.png',
                        help='Output visualization path')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load RoBERTa model and tokenizer
    print("Loading RoBERTa-base model...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    model = model.to(device)
    model.eval()
    print("Model loaded")
    
    # Load test data
    test_texts = load_test_texts(tokenizer, num_samples=args.num_samples)
    print(f"Loaded {len(test_texts)} test texts")
    
    # Define masking ratios
    mask_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Test model
    accuracies, steps = test_roberta(
        model, tokenizer, test_texts, mask_ratios, 
        device, args.max_seq_len, args.mode, args.max_steps, args.threshold
    )
    
    # Print results
    mode_name = "Single-pass" if args.mode == 'single' else "Iterative"
    print("\n" + "="*60)
    print(f"RoBERTa-base {mode_name} Test Results")
    print("="*60)
    print(f"{'Masking %':<12} {'Accuracy':<12} {'Avg Steps':<12}")
    print("-"*60)
    
    for ratio in mask_ratios:
        mask_pct = ratio * 100
        acc = accuracies[ratio]
        step = steps[ratio]
        print(f"{mask_pct:>10.0f}%  {acc:>10.4f}  {step:>10.2f}")
    
    print("="*60)
    
    # Visualize
    visualize_results(mask_ratios, accuracies, steps, args.mode, args.output)


if __name__ == '__main__':
    main()