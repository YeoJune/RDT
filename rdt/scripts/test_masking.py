"""Test reconstruction capability across different masking levels for RDT and RoBERTa models"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, RobertaTokenizer, RobertaForMaskedLM
from tqdm import tqdm
from pathlib import Path

from rdt.models import RDT
from rdt.utils import load_config, get_device


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


def test_rdt_model(model, tokenizer, test_texts, mask_ratios, device, max_seq_len, max_steps=20, threshold=0.5):
    """Test RDT model across different masking levels"""
    model.eval()
    mask_token_id = tokenizer.mask_token_id
    
    results = {ratio: [] for ratio in mask_ratios}
    steps_taken = {ratio: [] for ratio in mask_ratios}
    
    print(f"\nTesting RDT reconstruction capability (max_seq_len={max_seq_len})...")
    
    for text in tqdm(test_texts, desc="Processing texts"):
        # Tokenize with config max_length
        encoded = tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=max_seq_len,
            padding=False
        )
        tokens = encoded['input_ids'].squeeze(0)
        
        if len(tokens) < 10:
            continue
        
        # Test each masking ratio
        for ratio in mask_ratios:
            # Create masked input
            masked_tokens, eval_mask = create_masked_input(tokens, ratio, mask_token_id)
            input_ids = masked_tokens.unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                output_ids, num_steps = model.inference(
                    input_ids,
                    max_steps=max_steps,
                    threshold=threshold
                )
            
            # Calculate accuracy
            pred_tokens = output_ids.squeeze(0).cpu()
            accuracy = calculate_accuracy(pred_tokens, tokens, eval_mask)
            
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


def test_roberta_model(model, tokenizer, test_texts, mask_ratios, device, max_seq_len, mode='single', max_steps=20, threshold=0.95):
    """Test RoBERTa model across different masking levels"""
    """Test model across different masking levels"""
    model.eval()
    mask_token_id = tokenizer.mask_token_id
    
    results = {ratio: [] for ratio in mask_ratios}
    steps_taken = {ratio: [] for ratio in mask_ratios}
    
    print(f"\nTesting reconstruction capability (max_seq_len={max_seq_len})...")
    
    for text in tqdm(test_texts, desc="Processing texts"):
        # Tokenize with config max_length
        encoded = tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=max_seq_len,
            padding=False
        )
        tokens = encoded['input_ids'].squeeze(0)
        
        if len(tokens) < 10:
            continue
        
        # Test each masking ratio
        for ratio in mask_ratios:
            # Create masked input
            masked_tokens, eval_mask = create_masked_input(tokens, ratio, mask_token_id)
            input_ids = masked_tokens.unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                output_ids, num_steps = model.inference(
                    input_ids,
                    max_steps=max_steps,
                    threshold=threshold
                )
            
            # Calculate accuracy
            pred_tokens = output_ids.squeeze(0).cpu()
            accuracy = calculate_accuracy(pred_tokens, tokens, eval_mask)
            
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


def visualize_results(mask_ratios, accuracies, steps, model_type='rdt', mode='single', save_path=None):
    """Visualize test results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Determine title and color based on model type
    if model_type == 'rdt':
        title_prefix = "RDT"
        color_acc = 'b'
        color_steps = 'r'
    else:
        mode_name = "Single-pass" if mode == 'single' else "Iterative"
        title_prefix = f"RoBERTa-base {mode_name}"
        color_acc = 'g'
        color_steps = 'purple'
    
    # Plot 1: Accuracy vs Masking Ratio
    mask_percentages = [r * 100 for r in mask_ratios]
    ax1.plot(mask_percentages, list(accuracies.values()), f'{color_acc}-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Masking Ratio (%)', fontsize=12)
    ax1.set_ylabel('Reconstruction Accuracy', fontsize=12)
    ax1.set_title(f'{title_prefix} Reconstruction Accuracy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Add value labels
    for ratio, acc in accuracies.items():
        x = ratio * 100
        ax1.annotate(f'{acc:.3f}', (x, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # Plot 2: Steps Taken vs Masking Ratio
    ax2.plot(mask_percentages, list(steps.values()), f'{color_steps}-s', linewidth=2, markersize=8)
    ax2.set_xlabel('Masking Ratio (%)', fontsize=12)
    ax2.set_ylabel('Average Steps Taken', fontsize=12)
    ax2.set_title('Inference Steps by Masking Level', fontsize=14, fontweight='bold')
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


def load_test_texts_rdt(config, num_samples=100):
    """Load test texts using WikiTextDataset for RDT"""
    from rdt.data import WikiTextDataset
    
    print(f"Loading test data from {config['data']['dataset_name']}...")
    
    # Create dataset using existing WikiTextDataset (same logic as training)
    dataset = WikiTextDataset(
        dataset_name=config['data']['dataset_name'],
        split='test',
        tokenizer_name=config['data']['tokenizer_name'],
        max_seq_length=config['data']['max_seq_length'],
        total_steps=config['training']['total_steps'],
        max_chain_length=config['training']['max_chain_length'],
        visible_loss_ratio=config['training'].get('visible_loss_ratio', 0.15),
        samples_per_text=1,
        streaming=False  # Use non-streaming for test
    )
    
    # Extract raw texts from tokenized data
    texts = []
    tokenizer = dataset.tokenizer
    
    for tokens in dataset.tokenized_data[:num_samples]:
        # Decode tokens back to text
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        if len(text) > 50:  # Only use substantial texts
            texts.append(text)
    
    return texts


def load_test_texts_roberta(tokenizer, num_samples=100):
    """Load test texts from WikiText for RoBERTa"""
    from datasets import load_dataset
    
    print(f"Loading test data from WikiText-103...")
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
    
    texts = []
    for item in dataset:
        text = item['text'].strip()
        if len(text) > 50:
            texts.append(text)
        if len(texts) >= num_samples:
            break
    
    return texts


def main():
    parser = argparse.ArgumentParser(description='Test Masking Reconstruction for RDT and RoBERTa')
    parser.add_argument('--model_type', type=str, required=True, choices=['rdt', 'roberta'],
                        help='Model type: rdt or roberta')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (required for RDT)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of test samples')
    parser.add_argument('--max_seq_len', type=int, default=None,
                        help='Maximum sequence length (optional for RoBERTa, uses config for RDT)')
    parser.add_argument('--max_steps', type=int, default=20,
                        help='Maximum inference steps')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Confidence threshold (default: 0.5 for RDT, 0.95 for RoBERTa)')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'iterative'],
                        help='RoBERTa inference mode: single-pass or iterative')
    parser.add_argument('--output', type=str, default=None,
                        help='Output visualization path')
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        args.output = f'{args.model_type}_masking_results.png'
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.model_type == 'rdt':
        # ===== RDT Model =====
        # Load checkpoint
        print(f"Loading RDT checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        
        # Get config
        if args.config:
            config = load_config(args.config)
        else:
            config = checkpoint['config']
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])
        vocab_size = tokenizer.vocab_size
        
        # Create model
        print("Creating RDT model...")
        model = RDT(
            vocab_size=vocab_size,
            d_model=config['model']['d_model'],
            n_heads=config['model']['n_heads'],
            n_encoder_layers=config['model']['n_encoder_layers'],
            n_io_layers=config['model']['n_io_layers'],
            d_ff=config['model']['d_ff'],
            dropout=config['model']['dropout'],
            max_seq_len=config['data']['max_seq_length'],
            gate_hidden_dim=config['model']['gate_hidden_dim'],
            gate_num_layers=config['model']['gate_num_layers'],
            gate_num_heads=config['model']['gate_num_heads'],
            gradient_checkpointing=config['model'].get('gradient_checkpointing', False)
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded (epoch {checkpoint['epoch']}, step {checkpoint['step']})")
        
        # Load test data
        test_texts = load_test_texts_rdt(config, num_samples=args.num_samples)
        print(f"Loaded {len(test_texts)} test texts")
        
        # Define masking ratios
        mask_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        # Set threshold
        threshold = args.threshold if args.threshold else config['model']['threshold']
        max_steps = args.max_steps if args.max_steps else config['training']['total_steps']
        
        # Test model
        accuracies, steps = test_rdt_model(
            model, tokenizer, test_texts, mask_ratios, 
            device, config['data']['max_seq_length'], max_steps, threshold
        )
        
        # Print results
        print("\n" + "="*60)
        print("RDT Test Results")
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
        visualize_results(mask_ratios, accuracies, steps, 'rdt', args.mode, args.output)
    
    else:
        # ===== RoBERTa Model =====
        # Load model and tokenizer
        print(f"Loading RoBERTa model from {args.checkpoint}...")
        tokenizer = RobertaTokenizer.from_pretrained(args.checkpoint)
        model = RobertaForMaskedLM.from_pretrained(args.checkpoint)
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded from {args.checkpoint}")
        
        # Load test data
        test_texts = load_test_texts_roberta(tokenizer, num_samples=args.num_samples)
        print(f"Loaded {len(test_texts)} test texts")
        
        # Define masking ratios
        mask_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        # Set default parameters
        max_seq_len = args.max_seq_len if args.max_seq_len else 128
        threshold = args.threshold if args.threshold else 0.95
        
        # Test model
        accuracies, steps = test_roberta_model(
            model, tokenizer, test_texts, mask_ratios, 
            device, max_seq_len, args.mode, args.max_steps, threshold
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
        visualize_results(mask_ratios, accuracies, steps, 'roberta', args.mode, args.output)


if __name__ == '__main__':
    main()