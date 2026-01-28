"""
Conditional Masked Language Model (CMLM) with RoPE

CMLM extends MLM with iterative mask-predict decoding and uniform masking.
Uses same backbone as MLM but adds specialized training and inference strategies.

References:
- Mask-Predict: Ghazvininejad et al., "Mask-Predict: Parallel Decoding of 
  Conditional Masked Language Models" (EMNLP 2019)
"""

import torch
import torch.nn.functional as F
from .mlm import MLM
from typing import Dict, Optional, Tuple, List


class CMLM(MLM):
    """
    Conditional Masked Language Model (CMLM) with optimized Mask-Predict.
    
    CMLM extends MLM with:
    1. Uniform masking: Train on variable masking ratios (0% to 100%)
    2. Iterative inference: Mask-Predict decoding with confidence-based remasking
    
    Key features:
    - Uniform masking distribution for robust training
    - Single forward pass per iteration (efficient)
    - Confidence-based token selection
    - RoPE positional encoding (inherited from MLM)
    - FlashAttention optimization (inherited from MLM)
    
    Training:
        Use uniform_masking() instead of standard_masking()
        This exposes model to all masking ratios equally
    
    Inference:
        Use inference() for iterative mask-predict decoding
        Starts with fully masked sequence, refines over iterations
    
    Usage:
        # Same initialization as MLM
        model = CMLM(vocab_size=30522, d_model=768, n_layers=12, n_heads=12)
        
        # Training with uniform masking
        masked_ids, labels, mask_ratio = model.uniform_masking(input_ids)
        loss, logits = model(masked_ids, attention_mask, labels)
        
        # Iterative inference
        output_tokens, iterations = model.inference(
            input_ids=None,  # Start from scratch
            length=50,
            max_iterations=10
        )
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize CMLM (same parameters as MLM).
        
        See MLM.__init__ for full parameter documentation.
        """
        super().__init__(*args, **kwargs)
        
        # Log CMLM-specific info
        print(f"✓ CMLM extensions enabled:")
        print(f"  - Uniform masking for training")
        print(f"  - Iterative mask-predict inference")
    
    def uniform_masking(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Uniform distribution masking for CMLM training.
        
        Key idea from Mask-Predict paper:
        "We train with a simple masking scheme where the number of masked target 
        tokens is distributed uniformly, presenting the model with both easy 
        (single mask) and difficult (completely masked) examples."
        
        Implementation:
        - Sample masking ratio uniformly from [0, 1] per sample
        - Select tokens to mask using random sampling
        - Model learns to reconstruct from any masking level
        
        This is vectorized and GPU-optimized using torch.argsort.
        
        Args:
            input_ids: [B, L] original token indices
            attention_mask: [B, L] attention mask (1=valid, 0=padding)
        
        Returns:
            masked_input_ids: [B, L] input with masks applied
            labels: [B, L] labels for loss (-100 for non-masked tokens)
            avg_mask_ratio: float - average masking ratio across batch
        
        Example:
            With batch_size=4:
            - Sample 1: mask 5% of tokens
            - Sample 2: mask 43% of tokens
            - Sample 3: mask 87% of tokens
            - Sample 4: mask 12% of tokens
            → Model sees diverse masking levels
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 1. Sample masking ratio per sample from Uniform[0, 1]
        mask_ratios = torch.rand(batch_size, 1, device=device)  # [B, 1]
        
        # 2. Create random selection scores for each token
        selection_scores = torch.rand(batch_size, seq_len, device=device)  # [B, L]
        
        # 3. Determine which tokens can be masked (exclude padding)
        if attention_mask is not None:
            maskable = attention_mask.bool()
        else:
            maskable = torch.ones_like(input_ids, dtype=torch.bool)
        
        maskable = maskable & (input_ids != self.pad_token_id)
        
        # Make non-maskable tokens unselectable
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
        
        # 6. Apply masking strategy (inherited from MLM)
        masked_input_ids = self._apply_masking_strategy(input_ids, mask_decision)
        
        # 7. Create labels (-100 for non-masked tokens)
        labels = input_ids.clone()
        labels = labels.masked_fill(~mask_decision, -100)
        
        return masked_input_ids, labels, mask_ratios.mean().item()
    
    def inference(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_iterations: int = 10,
        length: Optional[int] = None,
        return_steps: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Iterative mask-predict inference with confidence-based remasking.
        
        Algorithm (from Mask-Predict paper):
        1. Start with fully masked sequence (or partially masked input)
        2. For each iteration:
           a. Predict all masked positions in parallel
           b. Get confidence scores for predictions
           c. Keep highest-confidence predictions
           d. Re-mask lowest-confidence predictions
        3. Repeat until convergence or max iterations
        
        Key optimization: Single forward pass per iteration
        (Original paper did two passes: one for prediction, one for confidence)
        
        Args:
            input_ids: [B, L] initial tokens (if None, starts with all masks)
            attention_mask: [B, L] attention mask (1=valid, 0=padding)
            max_iterations: Maximum number of mask-predict cycles (default: 10)
            length: Sequence length (required if input_ids is None)
            return_steps: If True, return intermediate predictions
        
        Returns:
            final_tokens: [B, L] final predictions
            steps_info: int (iterations used) or List[Tensor] (intermediate states)
        
        Example:
            # Generate from scratch
            tokens, iters = model.inference(length=50, max_iterations=10)
            
            # Refine partially masked input
            tokens, iters = model.inference(input_ids=masked_input, max_iterations=5)
            
            # Get intermediate steps for visualization
            tokens, steps = model.inference(length=50, return_steps=True)
        """
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            # 1. Initialize sequence
            if input_ids is None:
                if length is None:
                    raise ValueError("Must provide either input_ids or length")
                batch_size = 1
                current_ids = torch.full(
                    (batch_size, length),
                    self.mask_token_id,
                    dtype=torch.long,
                    device=device
                )
                if attention_mask is None:
                    attention_mask = torch.ones_like(current_ids)
            else:
                current_ids = input_ids.clone()
                batch_size, length = input_ids.shape
                if attention_mask is None:
                    attention_mask = (input_ids != self.pad_token_id).long()
            
            # Initialize confidence scores (0 for masked positions)
            current_confidences = torch.zeros(batch_size, length, device=device)
            
            intermediate_steps = [] if return_steps else None
            
            # 2. Iterative mask-predict refinement
            for iteration in range(max_iterations):
                if return_steps:
                    intermediate_steps.append(current_ids.clone())
                
                # --- Step A: Selective remasking (skip at iteration 0) ---
                if iteration > 0:
                    # Calculate number to mask with linear decay
                    # Mask ratio decreases: 1.0 → 0.0 over iterations
                    mask_ratio = 1.0 - (iteration / max_iterations)
                    num_maskable = attention_mask.sum(dim=1, keepdim=True).float()
                    num_to_mask = (num_maskable * mask_ratio).long()
                    
                    # Early stopping if nothing to mask
                    if num_to_mask.max() == 0:
                        break
                    
                    # Select least confident tokens to re-mask
                    # Add small Gumbel noise to break ties randomly
                    gumbel_noise = -torch.log(-torch.log(
                        torch.rand_like(current_confidences) + 1e-9
                    ) + 1e-9)
                    selection_scores = current_confidences + gumbel_noise * 0.001
                    
                    # Prevent masking of padding tokens
                    selection_scores = selection_scores.masked_fill(
                        attention_mask == 0, 1e9
                    )
                    
                    # Vectorized top-k selection
                    sorted_indices = torch.argsort(selection_scores, dim=1)
                    ranks = torch.argsort(sorted_indices, dim=1)
                    
                    # Mask tokens with lowest confidence
                    mask_decision = ranks < num_to_mask
                    
                    # Apply masks
                    current_ids = current_ids.masked_fill(
                        mask_decision, self.mask_token_id
                    )
                
                # --- Step B: Prediction (single forward pass) ---
                logits = self.forward(current_ids, attention_mask)  # [B, L, V]
                
                # Get predicted tokens
                new_ids = logits.argmax(dim=-1)  # [B, L]
                
                # Get confidences (optimized with log-sum-exp trick)
                # Instead of: probs = softmax(logits); conf = probs.max()
                # Use: conf = exp(max_logit - logsumexp(logits))
                # This saves memory (no full [B, L, V] probability tensor)
                max_logits = logits.max(dim=-1, keepdim=True).values  # [B, L, 1]
                log_sum_exp = torch.logsumexp(logits - max_logits, dim=-1)  # [B, L]
                new_confidences = torch.exp(max_logits.squeeze(-1) - log_sum_exp)  # [B, L]
                
                # Update only masked positions
                is_masked = (current_ids == self.mask_token_id)
                current_ids = torch.where(is_masked, new_ids, current_ids)
                current_confidences = torch.where(
                    is_masked, new_confidences, current_confidences
                )
            
            final_tokens = current_ids
        
        if return_steps:
            return final_tokens, intermediate_steps
        else:
            return final_tokens, max_iterations
    
    @classmethod
    def from_config(cls, config: Dict):
        """
        Create CMLM model from configuration dictionary.
        
        Uses same config format as MLM. See MLM.from_config for details.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            CMLM instance
        """
        model_cfg = config['model']
        training_cfg = config.get('training', {})
        bert_cfg = training_cfg.get('bert_masking', {})
        
        return cls(
            vocab_size=model_cfg['vocab_size'],
            d_model=model_cfg.get('d_model', 768),
            n_layers=model_cfg.get('n_layers', 12),
            n_heads=model_cfg.get('n_heads', 12),
            d_ff=model_cfg.get('d_ff'),
            max_seq_len=model_cfg.get('max_seq_len', 512),
            dropout=model_cfg.get('dropout', 0.1),
            mask_token_id=model_cfg.get('mask_token_id', 103),
            pad_token_id=model_cfg.get('pad_token_id', 0),
            rope_base=model_cfg.get('rope_base', 10000.0),
            bert_masking_enabled=bert_cfg.get('enabled', False),
            tie_weights=model_cfg.get('tie_weights', True)
        )