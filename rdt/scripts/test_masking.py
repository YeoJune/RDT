"""Test reconstruction capability across different masking levels for RDT and RoBERTa models
Enhanced with multiple metrics: BERTScore, Oracle PPL, BLEU-4, Exact Match
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, RobertaForMaskedLM, GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Metric libraries
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

from rdt.models import RDT
from rdt.utils import load_config, get_device


# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class MetricCalculator:
    """Batch-optimized metric calculator"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Load GPT-2 Large for perplexity
        print("Loading GPT-2 Large for perplexity calculation...")
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device)
        self.gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')
        self.gpt2_model.eval()
        
        self.smoothing = SmoothingFunction()
    
    def calculate_bertscore(self, references: List[str], predictions: List[str]) -> float:
        """Calculate BERTScore F1 (batch)"""
        if len(references) == 0:
            return 0.0
        
        P, R, F1 = bert_score(
            predictions, 
            references, 
            lang='en', 
            model_type='bert-base-uncased',
            device=self.device,
            batch_size=32,
            verbose=False
        )
        
        return F1.mean().item()
    
    def calculate_perplexity_batch(self, texts: List[str], batch_size: int = 8) -> float:
        """Calculate average perplexity using GPT-2 Large (per-sample, batched for speed)"""
        if len(texts) == 0:
            return float('inf')
        
        ppls = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Process each sample individually to get per-sample PPL
                for text in batch_texts:
                    # Tokenize
                    encodings = self.gpt2_tokenizer(
                        text,
                        return_tensors='pt',
                        truncation=True,
                        max_length=1024
                    ).to(self.device)
                    
                    input_ids = encodings['input_ids']
                    
                    # Calculate loss for this sample
                    outputs = self.gpt2_model(input_ids, labels=input_ids)
                    
                    # Perplexity = exp(loss)
                    sample_loss = outputs.loss.item()
                    ppl = np.exp(sample_loss)
                    ppls.append(ppl)
        
        return np.mean(ppls) if ppls else float('inf')
    
    def calculate_bleu4(self, references: List[str], predictions: List[str]) -> float:
        """Calculate average BLEU-4 score"""
        if len(references) == 0:
            return 0.0
        
        bleu_scores = []
        
        for ref, pred in zip(references, predictions):
            # Tokenize
            ref_tokens = ref.split()
            pred_tokens = pred.split()
            
            if len(pred_tokens) == 0:
                bleu_scores.append(0.0)
                continue
            
            # Calculate BLEU-4
            score = sentence_bleu(
                [ref_tokens], 
                pred_tokens, 
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=self.smoothing.method1
            )
            bleu_scores.append(score)
        
        return np.mean(bleu_scores)
    
    def calculate_exact_match(self, pred_tokens: torch.Tensor, target_tokens: torch.Tensor, 
                             eval_mask: torch.Tensor) -> float:
        """Calculate exact match accuracy on masked positions"""
        if eval_mask.sum() == 0:
            return 1.0
        
        masked_pred = pred_tokens[eval_mask]
        masked_target = target_tokens[eval_mask]
        
        correct = (masked_pred == masked_target).sum().item()
        total = eval_mask.sum().item()
        
        return correct / total


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


def roberta_single_pass_inference(model, input_ids, attention_mask):
    """Single forward pass through RoBERTa"""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred_tokens = torch.argmax(logits, dim=-1)
    return pred_tokens


def test_rdt_model(model, tokenizer, test_texts, mask_ratios, device, max_seq_len, 
                   metric_calc, max_steps=20, threshold=0.5, batch_size=32):
    """Test RDT model across different masking levels with multiple metrics (batched)"""
    model.eval()
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id or 0
    
    # Initialize result storage
    results = {
        'exact_match': {ratio: [] for ratio in mask_ratios},
        'bertscore': {ratio: {'refs': [], 'preds': []} for ratio in mask_ratios},
        'perplexity': {ratio: [] for ratio in mask_ratios},
        'bleu4': {ratio: {'refs': [], 'preds': []} for ratio in mask_ratios},
        'steps': {ratio: [] for ratio in mask_ratios}
    }
    
    print(f"\nTesting RDT reconstruction capability (max_seq_len={max_seq_len}, batch_size={batch_size})...")
    
    # Tokenize all texts first
    all_tokens = []
    all_texts = []
    
    for text in tqdm(test_texts, desc="Tokenizing"):
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
    
    # Process in batches for each masking ratio
    for ratio in mask_ratios:
        print(f"\nProcessing masking ratio: {ratio*100:.0f}%")
        
        for i in tqdm(range(0, len(all_tokens), batch_size), desc=f"Batches ({ratio*100:.0f}%)"):
            batch_tokens = all_tokens[i:i+batch_size]
            batch_texts = all_texts[i:i+batch_size]
            
            # Prepare batch
            max_len = max(len(t) for t in batch_tokens)
            batch_input_ids = []
            batch_attention_mask = []
            batch_eval_masks = []
            batch_original_tokens = []
            
            for tokens in batch_tokens:
                # Create masked input
                masked_tokens, eval_mask = create_masked_input(tokens, ratio, mask_token_id)
                
                # Pad to max_len
                pad_len = max_len - len(masked_tokens)
                if pad_len > 0:
                    masked_tokens = torch.cat([masked_tokens, torch.full((pad_len,), pad_token_id)])
                    eval_mask = torch.cat([eval_mask, torch.zeros(pad_len, dtype=torch.bool)])
                    padded_tokens = torch.cat([tokens, torch.full((pad_len,), pad_token_id)])
                    attention_mask = torch.cat([torch.ones(len(tokens)), torch.zeros(pad_len)])
                else:
                    padded_tokens = tokens
                    attention_mask = torch.ones(len(tokens))
                
                batch_input_ids.append(masked_tokens)
                batch_attention_mask.append(attention_mask)
                batch_eval_masks.append(eval_mask)
                batch_original_tokens.append(padded_tokens)
            
            # Stack into batch tensors
            batch_input_ids = torch.stack(batch_input_ids).to(device)
            batch_attention_mask = torch.stack(batch_attention_mask).to(device)
            
            # Batch inference
            with torch.no_grad():
                # Note: RDT.inference doesn't support batching natively, so we process one by one
                # but keep them on GPU for efficiency
                for j in range(len(batch_tokens)):
                    input_ids = batch_input_ids[j:j+1]
                    
                    output_ids, num_steps = model.inference(
                        input_ids,
                        max_steps=max_steps,
                        threshold=threshold
                    )
                    
                    pred_tokens = output_ids.squeeze(0).cpu()
                    original_tokens = batch_original_tokens[j]
                    eval_mask = batch_eval_masks[j]
                    
                    # Remove padding for evaluation
                    valid_len = len(batch_tokens[j])
                    pred_tokens = pred_tokens[:valid_len]
                    original_tokens = original_tokens[:valid_len]
                    eval_mask = eval_mask[:valid_len]
                    
                    # 1. Exact Match Accuracy
                    accuracy = metric_calc.calculate_exact_match(pred_tokens, original_tokens, eval_mask)
                    results['exact_match'][ratio].append(accuracy)
                    
                    # 2. Reconstructed text
                    reconstructed_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
                    original_text = batch_texts[j]
                    
                    results['bertscore'][ratio]['refs'].append(original_text)
                    results['bertscore'][ratio]['preds'].append(reconstructed_text)
                    results['perplexity'][ratio].append(reconstructed_text)
                    results['bleu4'][ratio]['refs'].append(original_text)
                    results['bleu4'][ratio]['preds'].append(reconstructed_text)
                    
                    # 3. Steps taken
                    results['steps'][ratio].append(num_steps)
    
    # Aggregate results
    aggregated = {
        'exact_match': {},
        'bertscore': {},
        'perplexity': {},
        'bleu4': {},
        'steps': {}
    }
    
    print("\nCalculating metrics...")
    for ratio in tqdm(mask_ratios, desc="Aggregating"):
        # Exact Match
        if results['exact_match'][ratio]:
            aggregated['exact_match'][ratio] = np.mean(results['exact_match'][ratio])
        else:
            aggregated['exact_match'][ratio] = 0.0
        
        # BERTScore
        if results['bertscore'][ratio]['refs']:
            aggregated['bertscore'][ratio] = metric_calc.calculate_bertscore(
                results['bertscore'][ratio]['refs'],
                results['bertscore'][ratio]['preds']
            )
        else:
            aggregated['bertscore'][ratio] = 0.0
        
        # Perplexity
        if results['perplexity'][ratio]:
            aggregated['perplexity'][ratio] = metric_calc.calculate_perplexity_batch(
                results['perplexity'][ratio]
            )
        else:
            aggregated['perplexity'][ratio] = float('inf')
        
        # BLEU-4
        if results['bleu4'][ratio]['refs']:
            aggregated['bleu4'][ratio] = metric_calc.calculate_bleu4(
                results['bleu4'][ratio]['refs'],
                results['bleu4'][ratio]['preds']
            )
        else:
            aggregated['bleu4'][ratio] = 0.0
        
        # Steps
        if results['steps'][ratio]:
            aggregated['steps'][ratio] = np.mean(results['steps'][ratio])
        else:
            aggregated['steps'][ratio] = 0.0
    
    return aggregated


def test_roberta_model(model, tokenizer, test_texts, mask_ratios, device, max_seq_len, 
                      metric_calc, batch_size=32):
    """Test RoBERTa model with single-pass inference (batched)"""
    model.eval()
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id or 0
    
    # Initialize result storage
    results = {
        'exact_match': {ratio: [] for ratio in mask_ratios},
        'bertscore': {ratio: {'refs': [], 'preds': []} for ratio in mask_ratios},
        'perplexity': {ratio: [] for ratio in mask_ratios},
        'bleu4': {ratio: {'refs': [], 'preds': []} for ratio in mask_ratios},
        'steps': {ratio: [] for ratio in mask_ratios}
    }
    
    print(f"\nTesting RoBERTa Single-pass reconstruction (max_seq_len={max_seq_len}, batch_size={batch_size})...")
    
    # Tokenize all texts first
    all_tokens = []
    all_texts = []
    all_attention_masks = []
    
    for text in tqdm(test_texts, desc="Tokenizing"):
        encoded = tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=max_seq_len,
            padding='max_length'
        )
        tokens = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # Filter out padding
        valid_positions = attention_mask.bool()
        valid_tokens = tokens[valid_positions]
        
        if len(valid_tokens) >= 10:
            all_tokens.append((tokens, attention_mask, valid_positions, valid_tokens))
            all_texts.append(tokenizer.decode(valid_tokens, skip_special_tokens=True))
    
    # Process in batches for each masking ratio
    for ratio in mask_ratios:
        print(f"\nProcessing masking ratio: {ratio*100:.0f}%")
        
        for i in tqdm(range(0, len(all_tokens), batch_size), desc=f"Batches ({ratio*100:.0f}%)"):
            batch_data = all_tokens[i:i+batch_size]
            batch_texts = all_texts[i:i+batch_size]
            
            # Prepare batch
            batch_input_ids = []
            batch_attention_masks = []
            batch_eval_masks = []
            batch_valid_tokens = []
            
            for tokens, attention_mask, valid_positions, valid_tokens in batch_data:
                # Create masked input (only on valid positions)
                masked_tokens, eval_mask = create_masked_input(valid_tokens, ratio, mask_token_id)
                
                # Put back into full sequence
                full_masked = tokens.clone()
                full_masked[valid_positions] = masked_tokens
                
                batch_input_ids.append(full_masked)
                batch_attention_masks.append(attention_mask)
                batch_eval_masks.append((eval_mask, valid_positions))
                batch_valid_tokens.append(valid_tokens)
            
            # Stack into batch tensors
            batch_input_ids = torch.stack(batch_input_ids).to(device)
            batch_attention_masks = torch.stack(batch_attention_masks).to(device)
            
            # Batch inference
            with torch.no_grad():
                outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks)
                logits = outputs.logits
                pred_tokens_batch = torch.argmax(logits, dim=-1).cpu()
            
            # Process each sample in batch
            for j in range(len(batch_data)):
                eval_mask, valid_positions = batch_eval_masks[j]
                valid_tokens = batch_valid_tokens[j]
                
                # Extract predictions on valid positions
                pred_tokens = pred_tokens_batch[j][valid_positions]
                
                # 1. Exact Match Accuracy
                accuracy = metric_calc.calculate_exact_match(pred_tokens, valid_tokens, eval_mask)
                results['exact_match'][ratio].append(accuracy)
                
                # 2. Reconstructed text
                reconstructed_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
                original_text = batch_texts[j]
                
                results['bertscore'][ratio]['refs'].append(original_text)
                results['bertscore'][ratio]['preds'].append(reconstructed_text)
                results['perplexity'][ratio].append(reconstructed_text)
                results['bleu4'][ratio]['refs'].append(original_text)
                results['bleu4'][ratio]['preds'].append(reconstructed_text)
                
                # 3. Steps (always 1 for single-pass)
                results['steps'][ratio].append(1)
    
    # Aggregate results
    aggregated = {
        'exact_match': {},
        'bertscore': {},
        'perplexity': {},
        'bleu4': {},
        'steps': {}
    }
    
    print("\nCalculating metrics...")
    for ratio in tqdm(mask_ratios, desc="Aggregating"):
        # Exact Match
        if results['exact_match'][ratio]:
            aggregated['exact_match'][ratio] = np.mean(results['exact_match'][ratio])
        else:
            aggregated['exact_match'][ratio] = 0.0
        
        # BERTScore
        if results['bertscore'][ratio]['refs']:
            aggregated['bertscore'][ratio] = metric_calc.calculate_bertscore(
                results['bertscore'][ratio]['refs'],
                results['bertscore'][ratio]['preds']
            )
        else:
            aggregated['bertscore'][ratio] = 0.0
        
        # Perplexity
        if results['perplexity'][ratio]:
            aggregated['perplexity'][ratio] = metric_calc.calculate_perplexity_batch(
                results['perplexity'][ratio]
            )
        else:
            aggregated['perplexity'][ratio] = float('inf')
        
        # BLEU-4
        if results['bleu4'][ratio]['refs']:
            aggregated['bleu4'][ratio] = metric_calc.calculate_bleu4(
                results['bleu4'][ratio]['refs'],
                results['bleu4'][ratio]['preds']
            )
        else:
            aggregated['bleu4'][ratio] = 0.0
        
        # Steps
        if results['steps'][ratio]:
            aggregated['steps'][ratio] = np.mean(results['steps'][ratio])
        else:
            aggregated['steps'][ratio] = 1.0
    
    return aggregated


def visualize_comparison(mask_ratios, results1, results2, model1_name, model2_name, output_dir):
    """Visualize comparison between two models"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mask_percentages = [r * 100 for r in mask_ratios]
    
    metrics = [
        ('exact_match', 'Exact Match Accuracy', (0, 1.05)),
        ('bertscore', 'BERTScore F1', (0, 1.05)),
        ('bleu4', 'BLEU-4 Score', (0, 1.05)),
        ('perplexity', 'Perplexity (GPT-2 Large)', None),
        ('steps', 'Average Inference Steps', None)
    ]
    
    for metric_key, metric_name, ylim in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot model 1
        values1 = [results1[metric_key][r] for r in mask_ratios]
        ax.plot(mask_percentages, values1, marker='o', linewidth=2, markersize=8, 
                label=model1_name, color='#2E86DE')
        
        # Plot model 2
        values2 = [results2[metric_key][r] for r in mask_ratios]
        ax.plot(mask_percentages, values2, marker='s', linewidth=2, markersize=8, 
                label=model2_name, color='#EE5A6F')
        
        ax.set_xlabel('Masking Ratio (%)', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if ylim:
            ax.set_ylim(ylim)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{metric_key}_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nVisualizations saved to: {output_dir}")


def visualize_single(mask_ratios, results, model_name, output_dir):
    """Visualize results for a single model"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mask_percentages = [r * 100 for r in mask_ratios]
    
    # Create 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    metrics = [
        ('exact_match', 'Exact Match Accuracy', (0, 1.05), '#2E86DE'),
        ('bertscore', 'BERTScore F1', (0, 1.05), '#10AC84'),
        ('bleu4', 'BLEU-4 Score', (0, 1.05), '#EE5A6F'),
        ('perplexity', 'Perplexity (GPT-2 Large)', None, '#F79F1F'),
        ('steps', 'Average Inference Steps', None, '#5F27CD')
    ]
    
    for idx, (metric_key, metric_name, ylim, color) in enumerate(metrics):
        ax = axes[idx]
        
        values = [results[metric_key][r] for r in mask_ratios]
        ax.plot(mask_percentages, values, marker='o', linewidth=2, markersize=8, color=color)
        
        ax.set_xlabel('Masking Ratio (%)', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if ylim:
            ax.set_ylim(ylim)
        
        # Add value labels
        for ratio, val in zip(mask_percentages, values):
            if metric_key == 'perplexity':
                ax.annotate(f'{val:.1f}', (ratio, val), textcoords="offset points", 
                           xytext=(0,5), ha='center', fontsize=8)
            else:
                ax.annotate(f'{val:.3f}', (ratio, val), textcoords="offset points", 
                           xytext=(0,5), ha='center', fontsize=8)
    
    # Hide the last subplot (6th position)
    axes[5].axis('off')
    
    fig.suptitle(f'{model_name} - Reconstruction Metrics', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'all_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_dir}")


def save_results_text(mask_ratios, results, model_name, output_path):
    """Save results to text file"""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{model_name} - Reconstruction Test Results\n")
        f.write("=" * 80 + "\n\n")
        
        # Table header
        f.write(f"{'Mask %':<10} {'Exact Match':<15} {'BERTScore':<15} {'BLEU-4':<15} {'PPL':<15} {'Steps':<10}\n")
        f.write("-" * 80 + "\n")
        
        for ratio in mask_ratios:
            mask_pct = ratio * 100
            exact = results['exact_match'][ratio]
            bert = results['bertscore'][ratio]
            bleu = results['bleu4'][ratio]
            ppl = results['perplexity'][ratio]
            steps = results['steps'][ratio]
            
            f.write(f"{mask_pct:>8.0f}%  {exact:>13.4f}  {bert:>13.4f}  {bleu:>13.4f}  "
                   f"{ppl:>13.2f}  {steps:>8.2f}\n")
        
        f.write("=" * 80 + "\n")


def save_comparison_text(mask_ratios, results1, results2, model1_name, model2_name, output_path):
    """Save comparison results to text file"""
    with open(output_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write(f"Comparison: {model1_name} vs {model2_name}\n")
        f.write("=" * 100 + "\n\n")
        
        metrics = ['exact_match', 'bertscore', 'bleu4', 'perplexity', 'steps']
        metric_names = ['Exact Match', 'BERTScore F1', 'BLEU-4', 'Perplexity', 'Steps']
        
        for metric_key, metric_name in zip(metrics, metric_names):
            f.write(f"\n{metric_name}:\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Mask %':<10} {model1_name:<25} {model2_name:<25} {'Difference':<20}\n")
            f.write("-" * 100 + "\n")
            
            for ratio in mask_ratios:
                mask_pct = ratio * 100
                val1 = results1[metric_key][ratio]
                val2 = results2[metric_key][ratio]
                
                if metric_key == 'perplexity':
                    diff = val1 - val2
                    f.write(f"{mask_pct:>8.0f}%  {val1:>23.2f}  {val2:>23.2f}  {diff:>18.2f}\n")
                else:
                    diff = val1 - val2
                    f.write(f"{mask_pct:>8.0f}%  {val1:>23.4f}  {val2:>23.4f}  {diff:>+18.4f}\n")
            
            f.write("\n")
        
        f.write("=" * 100 + "\n")


def save_detailed_csv(mask_ratios, results, model_name, output_path):
    """Save detailed results to CSV"""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Model', 'Masking_Ratio', 'Exact_Match', 'BERTScore_F1', 
                        'BLEU4', 'Perplexity', 'Avg_Steps'])
        
        # Data
        for ratio in mask_ratios:
            writer.writerow([
                model_name,
                f"{ratio*100:.0f}%",
                f"{results['exact_match'][ratio]:.4f}",
                f"{results['bertscore'][ratio]:.4f}",
                f"{results['bleu4'][ratio]:.4f}",
                f"{results['perplexity'][ratio]:.2f}",
                f"{results['steps'][ratio]:.2f}"
            ])


def load_test_texts_rdt(config, num_samples=1000):
    """Load test texts using WikiTextDataset for RDT"""
    from rdt.data import WikiTextDataset
    
    print(f"Loading test data from {config['data']['dataset_name']}...")
    
    # Create dataset
    dataset = WikiTextDataset(
        dataset_name=config['data']['dataset_name'],
        split='test',
        tokenizer_name=config['data']['tokenizer_name'],
        max_seq_length=config['data']['max_seq_length'],
        total_steps=config['training']['total_steps'],
        max_chain_length=config['training']['max_chain_length'],
        visible_loss_ratio=config['training'].get('visible_loss_ratio', 0.15),
        samples_per_text=1,
        streaming=False
    )
    
    # Extract raw texts
    texts = []
    tokenizer = dataset.tokenizer
    
    for tokens in dataset.tokenized_data[:num_samples]:
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        if len(text) > 50:
            texts.append(text)
    
    return texts


def load_test_texts_roberta(config, tokenizer, num_samples=1000):
    """Load test texts from WikiText for RoBERTa/MLM models"""
    from datasets import load_dataset
    
    dataset_name = config['data'].get('dataset_name', 'wikitext-2')
    
    # Map dataset names
    dataset_map = {
        'wikitext-2': 'wikitext-2-raw-v1',
        'wikitext-103': 'wikitext-103-raw-v1'
    }
    
    if dataset_name in dataset_map:
        full_dataset_name = dataset_map[dataset_name]
    else:
        full_dataset_name = dataset_name
    
    print(f"Loading test data from {full_dataset_name}...")
    dataset = load_dataset('wikitext', full_dataset_name, split='test')
    
    texts = []
    for item in dataset:
        text = item['text'].strip()
        if len(text) > 50:
            texts.append(text)
        if len(texts) >= num_samples:
            break
    
    return texts


def run_single_model_test(config_path, checkpoint_path, device, num_samples, 
                         max_seq_len, max_steps, threshold, output_dir):
    """Run test for a single model"""
    # Load config
    config = load_config(config_path)
    model_type = config.get('model_type', 'rdt').lower()
    
    # Get batch size from config
    batch_size = config['training'].get('batch_size', 32)
    
    print(f"\n{'='*80}")
    print(f"Testing Model: {model_type.upper()}")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Batch size: {batch_size} (from config)")
    print(f"{'='*80}")
    
    # Initialize metric calculator
    metric_calc = MetricCalculator(device=device)
    
    # Define masking ratios
    mask_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    if model_type == 'rdt':
        # Load RDT model
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])
        vocab_size = tokenizer.vocab_size
        
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
            gate_dropout=config['model']['gate_dropout'],
            gradient_checkpointing=config['model'].get('gradient_checkpointing', False)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Load test data
        test_texts = load_test_texts_rdt(config, num_samples=num_samples)
        print(f"Loaded {len(test_texts)} test texts")
        
        # Set parameters
        threshold = threshold if threshold is not None else config['model']['threshold']
        max_steps = max_steps if max_steps is not None else config['training']['total_steps']
        max_seq_len = max_seq_len if max_seq_len is not None else config['data']['max_seq_length']
        
        # Test model
        results = test_rdt_model(
            model, tokenizer, test_texts, mask_ratios, 
            device, max_seq_len, metric_calc, max_steps, threshold, batch_size
        )
        
        model_name = f"RDT"
        
    elif model_type == 'mlm':
        # Load RoBERTa model
        model_name_hf = config['model']['name']
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.is_dir():
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            model = RobertaForMaskedLM.from_pretrained(checkpoint_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name_hf)
            model = RobertaForMaskedLM.from_pretrained(model_name_hf)
            
            if checkpoint_path.suffix == '.pt':
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        # Load test data
        test_texts = load_test_texts_roberta(config, tokenizer, num_samples=num_samples)
        print(f"Loaded {len(test_texts)} test texts")
        
        # Set parameters
        max_seq_len = max_seq_len if max_seq_len is not None else config['data'].get('max_seq_length', 128)
        
        # Test model
        results = test_roberta_model(
            model, tokenizer, test_texts, mask_ratios, 
            device, max_seq_len, metric_calc, batch_size
        )
        
        model_name = "RoBERTa-base"
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Print results
    print("\n" + "="*80)
    print(f"{model_name} Test Results")
    print("="*80)
    print(f"{'Mask %':<10} {'Exact Match':<15} {'BERTScore':<15} {'BLEU-4':<15} {'PPL':<15} {'Steps':<10}")
    print("-"*80)
    
    for ratio in mask_ratios:
        mask_pct = ratio * 100
        exact = results['exact_match'][ratio]
        bert = results['bertscore'][ratio]
        bleu = results['bleu4'][ratio]
        ppl = results['perplexity'][ratio]
        steps = results['steps'][ratio]
        
        print(f"{mask_pct:>8.0f}%  {exact:>13.4f}  {bert:>13.4f}  {bleu:>13.4f}  "
              f"{ppl:>13.2f}  {steps:>8.2f}")
    
    print("="*80)
    
    return results, model_name, mask_ratios


def main():
    parser = argparse.ArgumentParser(description='Test Masking Reconstruction with Multiple Metrics')
    
    # Single model mode
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    
    # Comparison mode
    parser.add_argument('--config1', type=str, help='Path to first model config')
    parser.add_argument('--checkpoint1', type=str, help='Path to first model checkpoint')
    parser.add_argument('--config2', type=str, help='Path to second model config')
    parser.add_argument('--checkpoint2', type=str, help='Path to second model checkpoint')
    
    # Common arguments
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of test samples')
    parser.add_argument('--max_seq_len', type=int, default=None, help='Maximum sequence length')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum inference steps (RDT)')
    parser.add_argument('--threshold', type=float, default=None, help='Confidence threshold (RDT)')
    parser.add_argument('--output', type=str, default='results/', help='Output directory')
    
    args = parser.parse_args()
    
    # Determine mode
    is_comparison = args.config1 is not None and args.config2 is not None
    
    if not is_comparison and (args.config is None or args.checkpoint is None):
        parser.error("Either provide --config and --checkpoint, or --config1, --checkpoint1, --config2, --checkpoint2")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if is_comparison:
        # Comparison mode
        print("\n" + "="*80)
        print("COMPARISON MODE: Testing two models")
        print("="*80)
        
        # Test model 1
        results1, name1, mask_ratios = run_single_model_test(
            args.config1, args.checkpoint1, device, args.num_samples,
            args.max_seq_len, args.max_steps, args.threshold, output_dir
        )
        
        # Test model 2
        results2, name2, _ = run_single_model_test(
            args.config2, args.checkpoint2, device, args.num_samples,
            args.max_seq_len, args.max_steps, args.threshold, output_dir
        )
        
        # Generate comparison visualizations
        visualize_comparison(mask_ratios, results1, results2, name1, name2, output_dir)
        
        # Save comparison text
        save_comparison_text(mask_ratios, results1, results2, name1, name2, 
                            output_dir / 'comparison_summary.txt')
        
        # Save individual CSVs
        save_detailed_csv(mask_ratios, results1, name1, output_dir / f'{name1}_results.csv')
        save_detailed_csv(mask_ratios, results2, name2, output_dir / f'{name2}_results.csv')
        
        print(f"\nComparison results saved to: {output_dir}")
        
    else:
        # Single model mode
        print("\n" + "="*80)
        print("SINGLE MODEL MODE")
        print("="*80)
        
        results, model_name, mask_ratios = run_single_model_test(
            args.config, args.checkpoint, device, args.num_samples,
            args.max_seq_len, args.max_steps, args.threshold, output_dir
        )
        
        # Generate visualization
        visualize_single(mask_ratios, results, model_name, output_dir)
        
        # Save text results
        save_results_text(mask_ratios, results, model_name, output_dir / 'summary.txt')
        
        # Save CSV
        save_detailed_csv(mask_ratios, results, model_name, output_dir / 'detailed_results.csv')
        
        print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()