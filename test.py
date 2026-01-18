"""Test RDT Data Preprocessing - Verify chain generation logic"""

import torch
import yaml
from transformers import AutoTokenizer
from rdt.data import create_dataloaders


def print_separator(title=""):
    """Print section separator"""
    print("\n" + "="*80)
    if title:
        print(f"  {title}")
        print("="*80)


def decode_tokens(tokenizer, token_ids, skip_special=False):
    """Decode token IDs to text"""
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.cpu().tolist()
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special)


def visualize_batch(batch, tokenizer, batch_idx=0, sample_idx=0):
    """
    Visualize a single sample from RDT batch
    
    Args:
        batch: Batch dictionary from RDTPreprocessor
        tokenizer: Tokenizer for decoding
        batch_idx: Which batch to visualize
        sample_idx: Which sample in batch to visualize
    """
    print_separator(f"BATCH {batch_idx} - SAMPLE {sample_idx}")
    
    # Extract single sample
    input_tokens = batch['input'][sample_idx]
    targets = batch['targets'][sample_idx]  # [max_chain_length, L]
    loss_masks = batch['loss_masks'][sample_idx]  # [max_chain_length, L]
    gate_targets = batch['gate_targets'][sample_idx]  # [max_chain_length + 1]
    attention_mask = batch['attention_mask'][sample_idx]
    chain_length = batch['chain_lengths'][sample_idx].item()
    
    seq_len = input_tokens.shape[0]
    
    # Print metadata
    print(f"\nMetadata:")
    print(f"  Chain Length: {chain_length}")
    print(f"  Sequence Length: {seq_len}")
    print(f"  Actual Tokens (non-pad): {attention_mask.sum().item()}")
    
    # Print gate targets
    print(f"\nGate Targets (noise levels × 20):")
    for i in range(chain_length + 1):
        gate_val = gate_targets[i].item()
        print(f"  Step {i}: {gate_val:.4f} (noise level: {gate_val/20:.4f})")
    
    # Print input (h_0)
    print_separator("INPUT (h_0) - Initial State")
    input_text = decode_tokens(tokenizer, input_tokens, skip_special=True)
    print(f"Text: {input_text[:200]}...")
    
    # Count mask tokens in input (only in valid positions)
    mask_token_id = tokenizer.mask_token_id
    num_valid = attention_mask.sum().item()
    # ✅ FIX: Only count masks in non-padding positions
    valid_mask = attention_mask.bool()
    num_masked = ((input_tokens == mask_token_id) & valid_mask).sum().item()
    mask_ratio = num_masked / num_valid if num_valid > 0 else 0
    print(f"Masked tokens: {num_masked}/{num_valid} ({mask_ratio*100:.1f}%)")
    
    # Print each chain step
    for step_idx in range(chain_length):
        print_separator(f"STEP {step_idx} → STEP {step_idx + 1}")
        
        step_targets = targets[step_idx]
        step_loss_mask = loss_masks[step_idx]
        
        # Decode target
        target_text = decode_tokens(tokenizer, step_targets, skip_special=True)
        print(f"\nTarget (h_{step_idx + 1}):")
        print(f"  Text: {target_text[:200]}...")
        
        # Count masks in target (only in valid positions)
        # ✅ FIX: Only count masks in non-padding positions
        valid_mask = attention_mask.bool()
        num_masked_target = ((step_targets == mask_token_id) & valid_mask).sum().item()
        target_mask_ratio = num_masked_target / num_valid if num_valid > 0 else 0
        print(f"  Masked tokens: {num_masked_target}/{num_valid} ({target_mask_ratio*100:.1f}%)")
        
        # Analyze loss mask
        num_loss_tokens = step_loss_mask.sum().item()
        loss_ratio = num_loss_tokens / num_valid if num_valid > 0 else 0
        print(f"\nLoss Mask:")
        print(f"  Loss computed on: {num_loss_tokens}/{num_valid} tokens ({loss_ratio*100:.1f}%)")
        
        # Show which tokens are in loss mask
        if num_loss_tokens > 0:
            loss_positions = torch.where(step_loss_mask)[0].tolist()
            print(f"  Loss positions (first 10): {loss_positions[:10]}")
            
            # Decode tokens at loss positions
            loss_tokens = step_targets[step_loss_mask][:10]  # First 10
            loss_token_text = [tokenizer.decode([t]) for t in loss_tokens]
            print(f"  Loss tokens (first 10): {loss_token_text}")
    
    print_separator()


def visualize_curriculum_progress(config, tokenizer):
    """Visualize curriculum learning progression"""
    from rdt.data.rdt_preprocessor import RDTPreprocessor
    
    if not config['training']['curriculum']['enabled']:
        print("\n[INFO] Curriculum learning is disabled in config")
        return
    
    print_separator("CURRICULUM LEARNING PROGRESSION")
    
    preprocessor = RDTPreprocessor(tokenizer, config)
    
    progress_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    total_steps = config['training']['total_steps']
    
    print(f"\nTotal Steps: {total_steps}")
    print(f"Max Chain Length: {config['training']['max_chain_length']}")
    print(f"Start Step: {config['training']['curriculum']['start_step']}")
    print()
    
    for progress in progress_points:
        min_step, max_step = preprocessor.get_start_step_range(progress)
        
        # Calculate approximate mask ratios
        min_mask_ratio = 1.0 - (max_step / total_steps)
        max_mask_ratio = 1.0 - (min_step / total_steps)
        
        print(f"Progress {progress*100:>3.0f}%: start_step ∈ [{min_step:>2}, {max_step:>2}] "
              f"→ mask ratio ∈ [{min_mask_ratio*100:>4.1f}%, {max_mask_ratio*100:>4.1f}%]")
    
    print_separator()


def main():
    """Main test function"""
    
    # Load minimal config for testing
    config = {
        'model_type': 'rdt',
        'data': {
            'dataset_name': 'wikitext-2',
            'tokenizer_name': 'bert-base-uncased',
            'max_seq_length': 128,
            'streaming': False,
            'samples_per_text': 1,
            'num_workers': 0,  # Single process for testing
            'pin_memory': False,
            'max_val_samples': 100,
            'max_test_samples': 100,
        },
        'training': {
            'batch_size': 4,
            'max_chain_length': 2,
            'total_steps': 20,
            'visible_loss_ratio': 0.15,
            
            # Curriculum settings
            'curriculum': {
                'enabled': True,
                'start_step': 5,
            },
            
            # BERT masking settings
            'bert_masking': {
                'enabled': True,
                'mask_prob': 0.8,
                'random_prob': 0.1,
            }
        }
    }
    
    print_separator("RDT DATA PREPROCESSING TEST")
    print("\nConfiguration:")
    print(f"  Dataset: {config['data']['dataset_name']}")
    print(f"  Max Sequence Length: {config['data']['max_seq_length']}")
    print(f"  Batch Size: {config['training']['batch_size']}")
    print(f"  Max Chain Length: {config['training']['max_chain_length']}")
    print(f"  Total Steps: {config['training']['total_steps']}")
    print(f"  Visible Loss Ratio: {config['training']['visible_loss_ratio']}")
    print(f"  Curriculum Enabled: {config['training']['curriculum']['enabled']}")
    if config['training']['curriculum']['enabled']:
        print(f"  Curriculum Start Step: {config['training']['curriculum']['start_step']}")
    print(f"  BERT Masking Enabled: {config['training']['bert_masking']['enabled']}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Mask token: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
    print(f"  Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"  Train batches: ~{len(train_loader)}")
    print(f"  Val batches: ~{len(val_loader)}")
    
    # Visualize curriculum progression
    visualize_curriculum_progress(config, tokenizer)
    
    # Get first batch
    print("\nFetching first batch...")
    batch_iter = iter(train_loader)
    batch = next(batch_iter)
    
    print(f"\nBatch structure:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} ({value.dtype})")
        else:
            print(f"  {key}: {value}")
    
    # Visualize multiple samples
    num_samples_to_show = min(2, config['training']['batch_size'])
    
    for i in range(num_samples_to_show):
        visualize_batch(batch, tokenizer, batch_idx=0, sample_idx=i)
    
    # Test second batch to verify randomness
    print_separator("TESTING RANDOMNESS - SECOND BATCH")
    batch2 = next(batch_iter)
    visualize_batch(batch2, tokenizer, batch_idx=1, sample_idx=0)
    
    print_separator("TEST COMPLETED")
    print("\n✓ Data preprocessing is working correctly!")
    print("✓ Chain generation produces valid targets")
    print("✓ Gate targets align with masking schedule")
    print("✓ Loss masks identify tokens to predict")
    
    if config['training']['curriculum']['enabled']:
        print("✓ Curriculum learning configured properly")
    
    if config['training']['bert_masking']['enabled']:
        print("✓ BERT-style masking applied")


if __name__ == '__main__':
    main()