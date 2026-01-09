"""
Compare training data (RDTPreprocessor) vs test data (create_masked_input)
"""

import torch
from transformers import AutoTokenizer
from rdt.utils import load_config
from rdt.data import WikiTextDataset
from rdt.data.rdt_preprocessor import RDTPreprocessor

def create_masked_input(tokens, mask_ratio, mask_token_id, special_token_ids=None):
    """Test masking function"""
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


def main():
    # Load config
    config_path = 'rdt/configs/rdt.yaml'
    config = load_config(config_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])
    mask_token_id = tokenizer.mask_token_id
    special_token_ids = set(tokenizer.all_special_ids)
    
    print("="*80)
    print("DATA COMPARISON: Training (RDTPreprocessor) vs Test (create_masked_input)")
    print("="*80)
    
    # Load dataset - get more samples to find longer sequences
    dataset = WikiTextDataset(
        dataset_name=config['data']['dataset_name'],
        split='validation',
        tokenizer_name=config['data']['tokenizer_name'],
        max_seq_length=config['data']['max_seq_length'],
        samples_per_text=1,
        max_val_samples=100  # Get more samples
    )
    
    # Find a sample with sufficient length
    sample = None
    for i in range(len(dataset)):
        s = dataset[i]
        tokens = s['input_ids']
        non_pad = (tokens != tokenizer.pad_token_id).sum().item()
        if non_pad >= 50:  # At least 50 real tokens
            sample = s
            tokens = sample['input_ids']
            print(f"\nUsing sample {i} with {non_pad} non-padding tokens")
            break
    
    if sample is None:
        print("ERROR: Could not find sample with sufficient length")
        return
    
    print(f"Original tokens shape: {tokens.shape}")
    print(f"Original text: {tokenizer.decode(tokens, skip_special_tokens=True)[:100]}...")
    
    # Create RDT training data
    print("\n" + "="*80)
    print("TRAINING DATA (RDTPreprocessor)")
    print("="*80)
    
    preprocessor = RDTPreprocessor(tokenizer, config, device='cpu')
    training_batch = preprocessor([{'input_ids': tokens}])
    
    training_input = training_batch['input'][0]
    training_targets = training_batch['targets'][0]
    training_loss_masks = training_batch['loss_masks'][0]
    training_attention_mask = training_batch['attention_mask'][0]
    chain_length = training_batch['chain_lengths'][0].item()
    
    # Calculate masking ratio
    real_length = training_attention_mask.sum().item()
    masked_count = (training_input == mask_token_id).sum().item()
    training_masking_ratio = masked_count / real_length
    
    print(f"\nInput shape: {training_input.shape}")
    print(f"Targets shape: {training_targets.shape}")
    print(f"Loss masks shape: {training_loss_masks.shape}")
    print(f"Chain length: {chain_length}")
    print(f"Real length (non-padding): {real_length}")
    print(f"Masked count: {masked_count}")
    print(f"Masking ratio: {training_masking_ratio:.2%}")
    
    print(f"\nInput text (first 200 chars): {tokenizer.decode(training_input, skip_special_tokens=False)[:200]}...")
    
    # Show step-by-step targets
    for step in range(min(3, chain_length)):
        step_target = training_targets[step]
        step_loss_mask = training_loss_masks[step]
        loss_positions = step_loss_mask.nonzero().squeeze()
        if loss_positions.dim() == 0:
            loss_positions = [loss_positions.item()]
        else:
            loss_positions = loss_positions.tolist()
        
        print(f"\nStep {step}:")
        print(f"  Loss mask positions (first 10): {loss_positions[:10]}")
        print(f"  Loss mask count: {step_loss_mask.sum().item()}")
    
    # Create test data with similar masking ratio
    print("\n" + "="*80)
    print("TEST DATA (create_masked_input)")
    print("="*80)
    
    test_masking_ratio = 0.1  # Use 10% for comparison
    test_masked_tokens, test_eval_mask = create_masked_input(
        tokens, test_masking_ratio, mask_token_id, special_token_ids
    )
    
    test_real_length = len([t for t in tokens if t.item() not in special_token_ids])
    test_masked_count = (test_masked_tokens == mask_token_id).sum().item()
    actual_test_ratio = test_masked_count / test_real_length
    
    print(f"\nMasked tokens shape: {test_masked_tokens.shape}")
    print(f"Eval mask shape: {test_eval_mask.shape}")
    print(f"Real length (non-special): {test_real_length}")
    print(f"Masked count: {test_masked_count}")
    print(f"Requested masking ratio: {test_masking_ratio:.2%}")
    print(f"Actual masking ratio: {actual_test_ratio:.2%}")
    
    eval_positions = test_eval_mask.nonzero().squeeze()
    if eval_positions.dim() == 0:
        eval_positions = [eval_positions.item()]
    else:
        eval_positions = eval_positions.tolist()
    
    print(f"\nEval mask positions (first 10): {eval_positions[:10]}")
    print(f"Eval mask count: {test_eval_mask.sum().item()}")
    print(f"\nMasked text (first 200 chars): {tokenizer.decode(test_masked_tokens, skip_special_tokens=False)[:200]}...")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    print(f"\nTraining masking ratio: {training_masking_ratio:.2%}")
    print(f"Test masking ratio: {actual_test_ratio:.2%}")
    print(f"\nTraining: Uses [MASK] token? {(training_input == mask_token_id).any().item()}")
    print(f"Test: Uses [MASK] token? {(test_masked_tokens == mask_token_id).any().item()}")
    
    # Total loss mask count across all steps
    total_loss_mask_count = training_loss_masks.sum().item()
    total_loss_ratio = total_loss_mask_count / (real_length * chain_length)
    
    print(f"\nTraining: Total loss mask count across {chain_length} steps: {total_loss_mask_count}")
    print(f"Training: Average loss ratio per step: {total_loss_ratio:.2%}")
    print(f"Test: Eval mask count: {test_eval_mask.sum().item()}")
    

if __name__ == '__main__':
    main()