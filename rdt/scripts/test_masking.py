"""Test RDT reconstruction capability across different masking levels"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path

from rdt.model import RDT
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


def test_model(model, tokenizer, test_texts, mask_ratios, device, max_seq_len, max_steps=20, threshold=0.5):
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


def visualize_results(mask_ratios, accuracies, steps, save_path=None):
    """Visualize test results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy vs Masking Ratio
    mask_percentages = [r * 100 for r in mask_ratios]
    ax1.plot(mask_percentages, list(accuracies.values()), 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Masking Ratio (%)', fontsize=12)
    ax1.set_ylabel('Reconstruction Accuracy', fontsize=12)
    ax1.set_title('RDT Reconstruction Accuracy by Masking Level', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Add value labels
    for ratio, acc in accuracies.items():
        x = ratio * 100
        ax1.annotate(f'{acc:.3f}', (x, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # Plot 2: Steps Taken vs Masking Ratio
    ax2.plot(mask_percentages, list(steps.values()), 'r-s', linewidth=2, markersize=8)
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


def load_test_texts(config, num_samples=100):
    """Load test texts using WikiTextDataset"""
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


def main():
    parser = argparse.ArgumentParser(description='Test RDT Masking Reconstruction')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of test samples')
    parser.add_argument('--max_steps', type=int, default=20,
                        help='Maximum inference steps')
    parser.add_argument('--output', type=str, default='masking_test_results.png',
                        help='Output visualization path')
    
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Get config
    if args.config:
        config = load_config(args.config)
    else:
        config = checkpoint['config']
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])
    vocab_size = tokenizer.vocab_size
    
    # Create model
    print("Creating model...")
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
    test_texts = load_test_texts(config, num_samples=args.num_samples)
    print(f"Loaded {len(test_texts)} test texts")
    
    # Define masking ratios (0% to 100%)
    mask_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Test model
    accuracies, steps = test_model(
        model, tokenizer, test_texts, mask_ratios, 
        device, config['data']['max_seq_length'], args.max_steps, config['model']['threshold']
    )
    
    # Print results
    print("\n" + "="*60)
    print("Test Results")
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
    visualize_results(mask_ratios, accuracies, steps, args.output)


if __name__ == '__main__':
    main()