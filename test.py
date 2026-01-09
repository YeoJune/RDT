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
    
    # Load dataset
    dataset = WikiTextDataset(
        dataset_name=config['data']['dataset_name'],
        split='validation',
        tokenizer_name=config['data']['tokenizer_name'],
        max_seq_length=config['data']['max_seq_length'],
        samples_per_text=1,
        max_val_samples=10
    )
    
    # Get one sample
    sample = dataset[0]
    tokens = sample['input_ids']
    
    print(f"\nOriginal tokens shape: {tokens.shape}")
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
    
    print(f"\nInput text: {tokenizer.decode(training_input, skip_special_tokens=False)[:150]}...")
    
    # Show step-by-step targets
    for step in range(min(3, chain_length)):
        step_target = training_targets[step]
        step_loss_mask = training_loss_masks[step]
        loss_positions = step_loss_mask.nonzero().squeeze().tolist()
        if not isinstance(loss_positions, list):
            loss_positions = [loss_positions]
        
        print(f"\nStep {step}:")
        print(f"  Loss mask positions (first 10): {loss_positions[:10]}")
        print(f"  Loss mask count: {step_loss_mask.sum().item()}")
        print(f"  Target text (first 100 chars): {tokenizer.decode(step_target, skip_special_tokens=False)[:100]}...")
    
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
    
    eval_positions = test_eval_mask.nonzero().squeeze().tolist()
    if not isinstance(eval_positions, list):
        eval_positions = [eval_positions]
    
    print(f"\nEval mask positions (first 10): {eval_positions[:10]}")
    print(f"Eval mask count: {test_eval_mask.sum().item()}")
    print(f"\nMasked text: {tokenizer.decode(test_masked_tokens, skip_special_tokens=False)[:150]}...")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    print(f"\nTraining masking ratio: {training_masking_ratio:.2%}")
    print(f"Test masking ratio: {actual_test_ratio:.2%}")
    print(f"\nTraining: Uses [MASK] token? {(training_input == mask_token_id).any().item()}")
    print(f"Test: Uses [MASK] token? {(test_masked_tokens == mask_token_id).any().item()}")
    print(f"\nTraining: Mask token ID = {mask_token_id}")
    print(f"Test: Mask token ID = {mask_token_id}")
    
    # Check if positions match for similar ratio
    print("\n" + "="*80)
    print("POSITION ANALYSIS")
    print("="*80)
    
    # Training step 0 loss mask
    step0_loss_mask = training_loss_masks[0]
    step0_loss_positions = step0_loss_mask.nonzero().squeeze().tolist()
    if not isinstance(step0_loss_positions, list):
        step0_loss_positions = [step0_loss_positions]
    
    print(f"\nTraining step 0 loss positions (first 20): {step0_loss_positions[:20]}")
    print(f"Test eval positions (first 20): {eval_positions[:20]}")
    print(f"\nAre they using the same masking strategy? No - different random seeds and methods")
    

if __name__ == '__main__':
    main()