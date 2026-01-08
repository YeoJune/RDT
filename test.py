"""RDT Overfitting Sanity Check - 100 samples training and test-masking with EM metric"""

import argparse
import torch
import numpy as np
from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from pathlib import Path
from tqdm import tqdm
import json

from rdt.models import RDT
from rdt.utils import load_config, merge_configs, set_seed, create_model_from_config
from rdt.data import WikiTextDataset, RDTPreprocessor
from rdt.training import RDTTrainer
from torch.utils.data import DataLoader, Subset


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


def calculate_exact_match(pred_tokens: torch.Tensor, target_tokens: torch.Tensor, 
                         eval_mask: torch.Tensor) -> float:
    """Calculate exact match accuracy on masked positions"""
    if eval_mask.sum() == 0:
        return 1.0
    
    masked_pred = pred_tokens[eval_mask]
    masked_target = target_tokens[eval_mask]
    
    correct = (masked_pred == masked_target).sum().item()
    total = eval_mask.sum().item()
    
    return correct / total


def test_rdt_model(model, tokenizer, test_texts, mask_ratios, device, max_seq_len, 
                   max_steps=20, threshold=0.5, batch_size=32):
    """Test RDT model across different masking levels with EM metric (batched)"""
    model.eval()
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id or 0
    
    # Initialize result storage
    results = {
        'exact_match': {ratio: [] for ratio in mask_ratios}
    }
    
    print(f"\nTesting RDT reconstruction capability (max_seq_len={max_seq_len}, batch_size={batch_size})...")
    
    # Get special token IDs to exclude from masking
    special_token_ids = set(tokenizer.all_special_ids)
    
    # Tokenize all texts first
    all_tokens = []
    
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
    
    # Process in batches for each masking ratio
    for ratio in mask_ratios:
        print(f"\nProcessing masking ratio: {ratio*100:.0f}%")
        
        for i in tqdm(range(0, len(all_tokens), batch_size), desc=f"Batches ({ratio*100:.0f}%)"):
            batch_tokens = all_tokens[i:i+batch_size]
            
            # Prepare batch with dynamic padding
            batch_input_ids = []
            batch_eval_masks = []
            batch_original_tokens = []
            
            for tokens in batch_tokens:
                # Create masked input
                masked_tokens, eval_mask = create_masked_input(tokens, ratio, mask_token_id, special_token_ids)
                
                batch_input_ids.append(masked_tokens)
                batch_eval_masks.append(eval_mask)
                batch_original_tokens.append(tokens)
            
            # Pad to max length in this batch
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
            
            # Stack into batch tensors
            batch_input_ids_tensor = torch.stack(padded_input_ids).to(device)
            batch_attention_masks_tensor = torch.stack(padded_attention_masks).to(device)
            
            # Batch inference
            with torch.no_grad():
                output_ids, _ = model.inference(
                    batch_input_ids_tensor,
                    attention_mask=batch_attention_masks_tensor,
                    max_steps=max_steps,
                    threshold=threshold,
                    return_steps=True
                )
                
                pred_tokens_batch = output_ids.cpu()
            
            # Process each sample in batch
            for j in range(len(batch_tokens)):
                original_tokens = batch_original_tokens[j]
                eval_mask = batch_eval_masks[j]
                
                # Extract predictions (remove padding to match original length)
                orig_len = len(original_tokens)
                pred_tokens = pred_tokens_batch[j][:orig_len]
                
                # Exact Match Accuracy (only on masked positions)
                accuracy = calculate_exact_match(pred_tokens, original_tokens, eval_mask)
                results['exact_match'][ratio].append(accuracy)
    
    # Aggregate results
    aggregated = {'exact_match': {}}
    
    print("\nCalculating metrics...")
    for ratio in tqdm(mask_ratios, desc="Aggregating"):
        if results['exact_match'][ratio]:
            aggregated['exact_match'][ratio] = np.mean(results['exact_match'][ratio])
        else:
            aggregated['exact_match'][ratio] = 0.0
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description='RDT Overfitting Sanity Check')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to overfit on (default: 100)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--output_dir', type=str, default='test_outputs',
                        help='Output directory for checkpoints and results')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size for testing (default: use config value)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config for overfitting experiment
    config['training']['num_epochs'] = args.num_epochs
    config['training']['training_mode'] = 'epoch'
    config['output']['checkpoint_dir'] = f"{args.output_dir}/checkpoints"
    config['output']['log_dir'] = f"{args.output_dir}/logs"
    config['use_wandb'] = False  # Disable wandb for testing
    
    # Override batch size if specified
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    
    # Set seed
    set_seed(config['seed'])
    
    # Create accelerator (no wandb)
    accelerator = Accelerator(
        log_with=None,
        project_config=ProjectConfiguration(
            project_dir=config['output']['log_dir'],
            logging_dir=config['output']['log_dir']
        ),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1)
    )
    
    print(f"\n{'='*80}")
    print("RDT Overfitting Sanity Check")
    print(f"{'='*80}")
    print(f"Num samples: {args.num_samples}")
    print(f"Num epochs: {args.num_epochs}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {accelerator.device}")
    print(f"{'='*80}\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])
    vocab_size = tokenizer.vocab_size
    
    # =========================================================================
    # Step 1: Create small dataset (100 samples)
    # =========================================================================
    print("Step 1: Creating small dataset...")
    
    full_dataset = WikiTextDataset(
        dataset_name=config['data']['dataset_name'],
        tokenizer_name=config['data']['tokenizer_name'],
        max_seq_length=config['data']['max_seq_length'],
        split='train',
        samples_per_text=1
    )
    
    # Select first N samples
    indices = list(range(min(args.num_samples, len(full_dataset))))
    small_dataset = Subset(full_dataset, indices)
    
    print(f"Created dataset with {len(small_dataset)} samples")
    
    # Create collator
    rdt_collator = RDTPreprocessor(tokenizer, config, device='cpu')
    
    # Create dataloader
    train_loader = DataLoader(
        small_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,  # Use 0 for small dataset
        pin_memory=True,
        collate_fn=rdt_collator,
        drop_last=False
    )
    
    val_loader = train_loader  # Use same data for validation (overfitting check)
    
    # =========================================================================
    # Step 2: Create and train model
    # =========================================================================
    print("\nStep 2: Creating model...")
    
    model = create_model_from_config(config, vocab_size)
    
    print(f"Model parameters: {model.count_parameters()/1e6:.2f}M")
    
    # Create trainer
    trainer = RDTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        accelerator=accelerator
    )
    
    print("\nStep 3: Training model (overfitting)...")
    trainer.train()
    
    # End training properly (like train.py)
    accelerator.end_training()
    
    print(f"\nTraining completed!")
    print(f"Checkpoints saved to: {config['output']['checkpoint_dir']}")
    
    # =========================================================================
    # Step 4: Test-masking with different ratios
    # =========================================================================
    print("\nStep 4: Testing with different masking ratios...")
    
    # Get test texts (same as training data for overfitting check)
    # Use the exact same method as test_masking.py's load_test_texts_rdt
    test_texts = []
    for tokens in [full_dataset.tokenized_data[idx] for idx in indices]:
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        if len(text) > 50:  # Same filtering as test_masking.py
            test_texts.append(text)
    
    print(f"Number of test texts: {len(test_texts)}")
    
    # Masking ratios from 0% to 100% (same as test_masking.py default)
    mask_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Get trained model (unwrap exactly like test_masking.py)
    trained_model = accelerator.unwrap_model(trainer.model)
    trained_model.eval()
    
    # Run test-masking (exactly like test_masking.py)
    results = test_rdt_model(
        model=trained_model,
        tokenizer=tokenizer,
        test_texts=test_texts,
        mask_ratios=mask_ratios,
        device=accelerator.device,
        max_seq_len=config['data']['max_seq_length'],
        max_steps=config['training']['total_steps'],
        threshold=config['model']['threshold'],
        batch_size=args.batch_size if args.batch_size is not None else config['training']['batch_size']
    )
    
    # =========================================================================
    # Step 5: Print and save results
    # =========================================================================
    print(f"\n{'='*80}")
    print("Test Results (Exact Match Accuracy)")
    print(f"{'='*80}")
    print(f"{'Mask Ratio':<15} {'EM Accuracy':<15}")
    print("-" * 80)
    
    for ratio in mask_ratios:
        em = results['exact_match'][ratio]
        print(f"{ratio*100:>6.0f}%{'':<8} {em:>6.4f}")
    
    print(f"{'='*80}\n")
    
    # Save results to JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    print("\n" + "="*80)
    print("Overfitting sanity check completed successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
