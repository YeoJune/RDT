"""CMLM (Conditional Masked Language Model) - Optimized Implementation"""

import torch
import torch.nn.functional as F
from .mlm import MLM
from typing import Dict, Optional


class CMLM(MLM):
    """
    Conditional Masked Language Model (CMLM) with optimized Mask-Predict.
    
    As described in:
    "Mask-Predict: Parallel Decoding of Conditional Masked Language Models" 
    (Ghazvininejad et al., EMNLP 2019)
    
    Key optimizations:
    1. Vectorized Uniform Masking (no Python loops, GPU-accelerated)
    2. Efficient Inference (1 forward pass per iteration instead of 2)
    3. Confidence reuse across iterations
    
    This provides a fair, efficient baseline for comparing with RDT.
    """
    
    def __init__(
        self,
        architecture: str = 'bert-base-uncased',
        pretrained: Optional[str] = None,
        vocab_size: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None
    ):
        """
        Initialize CMLM by extending MLM.
        
        Args:
            architecture: Model architecture identifier (e.g., 'bert-base-uncased')
            pretrained: Model name/path to load pretrained weights from (None = from scratch)
            vocab_size: Vocabulary size (required for from-scratch training)
            mask_token_id: ID of [MASK] token (auto-detected if None)
            pad_token_id: ID of [PAD] token (auto-detected if None)
        """
        super().__init__(
            architecture=architecture,
            pretrained=pretrained,
            vocab_size=vocab_size,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id
        )
    
    @classmethod
    def from_config(cls, config: Dict):
        """
        Create CMLM model from config dictionary.
        
        Config format:
            model:
              architecture: "bert-base-uncased"  # Required
              pretrained: "bert-base-uncased"    # Optional: None for from-scratch
              vocab_size: 50000                   # Optional
              mask_token_id: 103                  # Optional
              pad_token_id: 0                     # Optional
        
        Args:
            config: Configuration dictionary
            
        Returns:
            CMLM instance
        """
        model_cfg = config['model']
        
        return cls(
            architecture=model_cfg['architecture'],
            pretrained=model_cfg.get('pretrained'),
            vocab_size=model_cfg.get('vocab_size'),
            mask_token_id=model_cfg.get('mask_token_id'),
            pad_token_id=model_cfg.get('pad_token_id')
        )
    
    def uniform_masking(self, input_ids, attention_mask=None):
        """
        Vectorized uniform distribution masking (OPTIMIZED - no Python loops).
        
        "We train with a simple masking scheme where the number of masked target 
        tokens is distributed uniformly, presenting the model with both easy 
        (single mask) and difficult (completely masked) examples."
        
        Masking ratio is sampled uniformly from [0, 1] per sample in batch,
        allowing model to learn reconstruction from 0% to 100% masked examples.
        
        Optimization: Uses torch.argsort for fully GPU-accelerated masking.
        
        Args:
            input_ids: [B, L] original token indices
            attention_mask: [B, L] attention mask (1=valid, 0=padding)
        
        Returns:
            masked_input_ids: [B, L] input with masks applied
            labels: [B, L] labels for loss (-100 for non-masked tokens)
            mask_ratio: float - average masking ratio used
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 1. Sample masking ratio per sample from Uniform[0, 1]
        # [B, 1]
        mask_ratios = torch.rand(batch_size, 1, device=device)
        
        # 2. Create random selection scores for each token
        # [B, L]
        selection_scores = torch.rand(batch_size, seq_len, device=device)
        
        # 3. Determine which tokens can be masked (exclude padding)
        if attention_mask is not None:
            maskable = attention_mask.bool()
        else:
            maskable = torch.ones_like(input_ids, dtype=torch.bool)
        
        maskable = maskable & (input_ids != self.pad_token_id)
        
        # Make non-maskable tokens unselectable by setting score to -inf
        selection_scores = selection_scores.masked_fill(~maskable, -1e9)
        
        # 4. Calculate number of tokens to mask per sample
        num_maskable = maskable.sum(dim=1, keepdim=True).float()  # [B, 1]
        num_to_mask = (num_maskable * mask_ratios).long()  # [B, 1]
        
        # 5. Select top-k tokens to mask using argsort (fully vectorized)
        # Get ranking of each token by selection score
        sorted_indices = torch.argsort(selection_scores, dim=1, descending=True)
        ranks = torch.argsort(sorted_indices, dim=1)  # [B, L]
        
        # Mask tokens with rank < num_to_mask
        mask_decision = ranks < num_to_mask  # [B, L]
        
        # 6. Apply masking
        masked_input_ids = input_ids.clone()
        masked_input_ids = masked_input_ids.masked_fill(mask_decision, self.mask_token_id)
        
        # 7. Create labels (-100 for non-masked tokens)
        labels = input_ids.clone()
        labels = labels.masked_fill(~mask_decision, -100)
        
        return masked_input_ids, labels, mask_ratios.mean().item()

    
    def inference(
        self,
        input_ids=None,
        attention_mask=None,
        max_iterations=10,
        length=None,
        return_steps=False
    ):
        """
        Optimized Mask-Predict inference (SINGLE forward pass per iteration).
        
        "Decoding starts with a completely masked target text, to predict all of 
        the words in parallel, and ends after a constant number of mask-predict cycles."
        
        Key optimization: Reuses confidence scores from previous iteration,
        eliminating redundant forward passes.
        
        Args:
            input_ids: [B, L] initial tokens (if None, starts with all masks)
            attention_mask: [B, L] attention mask
            max_iterations: Maximum number of mask-predict iterations
            length: Sequence length (required if input_ids is None)
            return_steps: If True, return intermediate predictions
        
        Returns:
            final_tokens: [B, L] final predictions
            steps_taken: int or List[Tensor] - iterations or intermediate states
        """
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            # 1. Initialize
            if input_ids is None:
                if length is None:
                    raise ValueError("Must provide either input_ids or length")
                batch_size = 1
                current_ids = torch.full((batch_size, length), self.mask_token_id, 
                                       dtype=torch.long, device=device)
                if attention_mask is None:
                    attention_mask = torch.ones_like(current_ids)
            else:
                current_ids = input_ids.clone()
                batch_size, length = input_ids.shape
                if attention_mask is None:
                    attention_mask = (input_ids != self.pad_token_id).long()
            
            # Initialize confidence scores (0 for all masked positions initially)
            current_confidences = torch.zeros(batch_size, length, device=device)
            
            intermediate_steps = [] if return_steps else None
            
            # 2. Iterative Mask-Predict refinement
            for iteration in range(max_iterations):
                if return_steps:
                    intermediate_steps.append(current_ids.clone())
                
                # --- Step A: Selective Masking (skip at iteration 0 if starting fully masked) ---
                if iteration > 0:
                    # Calculate number to mask with linear decay
                    # Standard Mask-Predict: mask_ratio linearly decreases from 1.0 to 0.0
                    mask_ratio = 1.0 - (iteration / max_iterations)
                    num_maskable = attention_mask.sum(dim=1, keepdim=True).float()  # [B, 1]
                    num_to_mask = (num_maskable * mask_ratio).long()  # [B, 1]
                    
                    # Early stopping if nothing to mask
                    if num_to_mask.max() == 0:
                        break
                    
                    # Select least confident tokens to re-mask
                    # Add small Gumbel noise to break ties randomly
                    gumbel_noise = -torch.log(-torch.log(torch.rand_like(current_confidences) + 1e-9) + 1e-9)
                    selection_scores = current_confidences + gumbel_noise * 0.001
                    
                    # Prevent masking of padding tokens
                    selection_scores = selection_scores.masked_fill(attention_mask == 0, 1e9)
                    
                    # Vectorized top-k selection (fully GPU-accelerated)
                    sorted_indices = torch.argsort(selection_scores, dim=1)  # ascending order
                    ranks = torch.argsort(sorted_indices, dim=1)  # [B, L]
                    
                    # Mask tokens with lowest confidence (rank < num_to_mask)
                    mask_decision = ranks < num_to_mask  # [B, L]
                    
                    # Apply masks
                    current_ids = current_ids.masked_fill(mask_decision, self.mask_token_id)
                
                # --- Step B: Prediction (SINGLE forward pass) ---
                logits = self.forward(current_ids, attention_mask)  # [B, L, V]
                probs = F.softmax(logits, dim=-1)  # [B, L, V]
                
                # Get predicted tokens and their confidences
                new_ids = logits.argmax(dim=-1)  # [B, L]
                new_confidences = probs.max(dim=-1).values  # [B, L]
                
                # Update only masked positions (preserve non-masked tokens and their confidences)
                is_masked = (current_ids == self.mask_token_id)
                current_ids = torch.where(is_masked, new_ids, current_ids)
                current_confidences = torch.where(is_masked, new_confidences, current_confidences)
            
            final_tokens = current_ids
        
        if return_steps:
            return final_tokens, intermediate_steps
        else:
            return final_tokens, max_iterations