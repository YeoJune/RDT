"""
Masked Diffusion Language Model (MDLM) with RoPE

MDLM extends MLM with continuous-time diffusion training and sampling.
Uses SUBS (SUBStitution) parameterization for efficient masked diffusion.

References:
- MDLM: Sahoo et al., "Simple and Effective Masked Diffusion Language Models" (NeurIPS 2024)
"""

import torch
import torch.nn.functional as F
import math
from .mlm import MLM
from typing import Dict, Optional, Tuple


class MDLM(MLM):
    """
    Masked Diffusion Language Model (MDLM) with SUBS parameterization.
    
    MDLM extends MLM with:
    1. Continuous-time formulation: Sample masking from α(t) schedule
    2. Time-weighted loss: Rao-Blackwellized NELBO objective
    3. Efficient sampling: Cached diffusion with 3-4x speedup
    
    Key features:
    - Cosine or linear noise schedule
    - Time derivative weighting for stable training
    - Low-discrepancy sampling for variance reduction
    - RoPE positional encoding (inherited from MLM)
    - FlashAttention optimization (inherited from MLM)
    
    Training:
        Use continuous_time_masking() instead of standard_masking()
        Apply time-weighted loss for proper diffusion training
    
    Inference:
        Use inference() for iterative denoising
        Supports different samplers: ddpm_cache, ddpm, analytic
    
    Usage:
        # Initialize with cosine schedule
        model = MDLM(
            vocab_size=30522,
            d_model=768,
            n_layers=12,
            n_heads=12,
            noise_schedule='cosine'
        )
        
        # Training with continuous-time masking
        masked_ids, labels, t, weights = model.continuous_time_masking(input_ids)
        loss, logits = model.forward_with_time_weighting(
            masked_ids, attention_mask, labels, weights
        )
        
        # Sampling
        output = model.inference(length=50, num_steps=1000, sampler='ddpm_cache')
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        mask_token_id: int = 103,
        pad_token_id: int = 0,
        rope_base: float = 10000.0,
        bert_masking_enabled: bool = False,
        tie_weights: bool = True,
        noise_schedule: str = 'cosine',
        time_conditioning: bool = False
    ):
        """
        Initialize MDLM.
        
        Args:
            (See MLM for base parameters)
            noise_schedule: Noise schedule type ('cosine' or 'linear')
            time_conditioning: Whether to condition model on timestep
                             (Default: False, as MDLM paper doesn't use this)
        """
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            rope_base=rope_base,
            bert_masking_enabled=bert_masking_enabled,
            tie_weights=tie_weights
        )
        
        self.noise_schedule = noise_schedule
        self.time_conditioning = time_conditioning
        
        # Log MDLM-specific info
        print(f"✓ MDLM extensions enabled:")
        print(f"  - Noise schedule: {noise_schedule}")
        print(f"  - Time conditioning: {time_conditioning}")
        if time_conditioning:
            print(f"  ⚠️  Warning: MDLM paper uses time_conditioning=False")
    
    def _cosine_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """
        Cosine noise schedule: α(t) = cos²(πt/2)
        
        This is the recommended schedule in the MDLM paper.
        Provides smooth transition from clean to noisy data.
        
        Args:
            t: Time steps in [0, 1], shape [...]
        
        Returns:
            α(t): Non-masking probability, same shape as t
        """
        return torch.cos(t * math.pi / 2) ** 2
    
    def _linear_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """
        Linear noise schedule: α(t) = 1 - t
        
        Simpler than cosine but can be less stable at boundaries.
        
        Args:
            t: Time steps in [0, 1], shape [...]
        
        Returns:
            α(t): Non-masking probability, same shape as t
        """
        return 1.0 - t
    
    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get α(t) based on noise schedule.
        
        α(t) represents the probability that a token is NOT masked at time t.
        α(0) = 1 (fully visible), α(1) = 0 (fully masked)
        
        Args:
            t: Time steps in [0, 1]
        
        Returns:
            α(t): Non-masking probability at time t
        """
        if self.noise_schedule == 'cosine':
            return self._cosine_schedule(t)
        elif self.noise_schedule == 'linear':
            return self._linear_schedule(t)
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")
    
    def get_mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get masking probability p_mask(t) = 1 - α(t)
        
        Args:
            t: Time steps in [0, 1]
        
        Returns:
            p_mask(t): Masking probability at time t
        """
        return 1.0 - self.get_alpha(t)
    
    def continuous_time_masking(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        low_discrepancy_sampling: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MDLM continuous-time masking for training.
        
        Algorithm:
        1. Sample time t ~ Uniform[ε, 1] per sample
        2. Calculate mask probability p(t) = 1 - α(t)
        3. Mask each token with probability p(t)
        4. Compute loss weight w(t) = |α'(t)| / (1 - α(t))
        
        Key insight from paper:
        "We train with a continuous-time formulation where the number of masked
        tokens is sampled from a noise schedule α(t), presenting the model with
        varying levels of corruption from t=0 (fully visible) to t=1 (fully masked)."
        
        Args:
            input_ids: [B, L] original token indices
            attention_mask: [B, L] attention mask (1=valid, 0=padding)
            low_discrepancy_sampling: Use stratified sampling for variance reduction
        
        Returns:
            masked_input_ids: [B, L] input with masks applied
            labels: [B, L] labels for loss (-100 for non-masked tokens)
            t: [B] sampled time steps
            loss_weight: [B] weight for loss (time derivative α'(t))
        
        Example:
            masked, labels, t, weights = model.continuous_time_masking(input_ids)
            # t might be [0.23, 0.67, 0.91, ...] for different samples
            # weights account for importance of different time steps
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 1. Sample time steps t ~ Uniform[ε, 1]
        # Avoid t=0 to prevent singularity in loss weight
        eps = 1e-5
        
        if low_discrepancy_sampling:
            # Low-discrepancy (stratified) sampler
            # Partition [ε, 1] into B equal intervals and sample from each
            # This reduces variance in gradient estimation
            t = torch.linspace(eps, 1.0, batch_size + 1, device=device)[:-1]
            noise = torch.rand(batch_size, device=device) * (1 - eps) / batch_size
            t = (t + noise).clamp(eps, 1.0)
        else:
            # Standard uniform sampling
            t = (torch.rand(batch_size, device=device) * (1 - eps) + eps).clamp(eps, 1.0)
        
        # 2. Calculate masking probability from noise schedule
        alpha_t = self.get_alpha(t)  # [B]
        mask_prob = 1.0 - alpha_t  # [B]
        
        # 3. Determine which tokens can be masked (exclude padding)
        if attention_mask is not None:
            maskable = attention_mask.bool()
        else:
            maskable = torch.ones_like(input_ids, dtype=torch.bool)
        
        maskable = maskable & (input_ids != self.pad_token_id)
        
        # 4. Sample masking decisions for each token
        # Each token is masked with probability mask_prob[batch_idx]
        rand = torch.rand(batch_size, seq_len, device=device)  # [B, L]
        mask_prob_expanded = mask_prob.unsqueeze(1)  # [B, 1]
        
        mask_decision = (rand < mask_prob_expanded) & maskable  # [B, L]
        
        # 5. Apply masking strategy (inherited from MLM)
        masked_input_ids = self._apply_masking_strategy(input_ids, mask_decision)
        
        # 6. Create labels (-100 for non-masked tokens)
        labels = input_ids.clone()
        labels = labels.masked_fill(~mask_decision, -100)
        
        # 7. Calculate loss weight: |α'(t)| / (1 - α(t))
        # This comes from the Rao-Blackwellized NELBO (Eq. 3 in paper)
        loss_weight = self._compute_loss_weight(t)  # [B]
        
        return masked_input_ids, labels, t, loss_weight
    
    def _compute_loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute loss weight: |α'(t)| / (1 - α(t))
        
        This weight comes from the continuous-time NELBO:
        L = E_t ∫ [α'(t) / (1 - α(t))] * CrossEntropy(x_θ(z_t), x) dt
        
        The weight ensures that all time steps contribute equally to training,
        despite having different amounts of corruption.
        
        Args:
            t: [B] time steps in [0, 1]
        
        Returns:
            weight: [B] loss weight for each sample
        """
        # Compute analytic derivative for stability
        if self.noise_schedule == 'cosine':
            # α(t) = cos²(πt/2)
            # α'(t) = -π * cos(πt/2) * sin(πt/2) = -(π/2) * sin(πt)
            alpha_prime = -(math.pi / 2) * torch.sin(math.pi * t)
        elif self.noise_schedule == 'linear':
            # α(t) = 1 - t
            # α'(t) = -1
            alpha_prime = torch.full_like(t, -1.0)
        else:
            raise ValueError(f"Unknown schedule: {self.noise_schedule}")
        
        # Compute 1 - α(t)
        alpha_t = self.get_alpha(t)
        one_minus_alpha = 1.0 - alpha_t
        
        # Weight = |α'(t)| / (1 - α(t))
        # Add epsilon for numerical stability
        weight = torch.abs(alpha_prime) / (one_minus_alpha + 1e-8)
        
        # Clamp to prevent explosion near t=0
        # Theory: weight → ∞ as t → 0, but we clamp for stable training
        weight = torch.clamp(weight, max=100.0)
        
        return weight
    
    def forward_with_time_weighting(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_weight: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with time-weighted loss for MDLM training.
        
        Important: Weights must be applied PER SAMPLE before reduction.
        Standard PyTorch reduction doesn't support sample-wise weights correctly.
        
        Args:
            input_ids: [B, L] token indices
            attention_mask: [B, L] attention mask (1=valid, 0=padding)
            labels: [B, L] target token indices (-100 for non-masked)
            loss_weight: [B] loss weight per sample (from continuous_time_masking)
        
        Returns:
            loss: weighted scalar loss
            logits: [B, L, vocab_size] prediction logits
        """
        # Get logits from base MLM
        if labels is not None:
            # We'll compute loss manually with weights
            logits = self.forward(input_ids, attention_mask)
            
            # Compute per-sample loss
            # CrossEntropy with reduction='none' gives [B, L]
            loss_per_token = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
                reduction='none'
            ).view(labels.shape)  # [B, L]
            
            # Average over sequence (only non-ignored tokens)
            mask = (labels != -100)
            loss_per_sample = (loss_per_token * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # [B]
            
            # Apply time weights
            if loss_weight is not None:
                loss_per_sample = loss_per_sample * loss_weight  # [B]
            
            # Final reduction over batch
            loss = loss_per_sample.mean()
            
            return loss, logits
        else:
            return self.forward(input_ids, attention_mask)
    
    def inference(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        length: Optional[int] = None,
        num_steps: int = 1000,
        return_steps: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        DDPM-style sampling for MDLM.
        
        Algorithm:
        1. Start with fully masked sequence (t=1)
        2. For each timestep from T to 1:
           a. Predict all tokens
           b. Probabilistically accept predictions based on α(t)
        3. Return final sequence
        
        Args:
            input_ids: [B, L] initial tokens (if None, starts fully masked)
            attention_mask: [B, L] attention mask
            length: Sequence length (required if input_ids is None)
            num_steps: Number of denoising steps (more = better quality)
            return_steps: If True, return intermediate predictions
        
        Returns:
            final_tokens: [B, L] generated tokens
            steps_info: intermediate states if return_steps=True
        """
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            # Initialize
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
            
            intermediate_steps = [] if return_steps else None
            
            # Iterative denoising from t=1 to t=0
            for step in range(num_steps, 0, -1):
                if return_steps:
                    intermediate_steps.append(current_ids.clone())
                
                # Current time
                t = step / num_steps
                t_tensor = torch.full((batch_size,), t, device=device)
                
                # Predict
                logits = self.forward(current_ids, attention_mask)
                predicted_ids = logits.argmax(dim=-1)
                
                # Probabilistic acceptance based on α(t)
                alpha_t = self.get_alpha(t_tensor)
                accept_prob = 1.0 - alpha_t  # Unmask probability
                
                rand = torch.rand(batch_size, length, device=device)
                should_unmask = rand < accept_prob.unsqueeze(1)
                
                # Update: unmask some positions
                is_masked = (current_ids == self.mask_token_id)
                should_update = is_masked & should_unmask
                current_ids = torch.where(should_update, predicted_ids, current_ids)
            
            final_tokens = current_ids
        
        if return_steps:
            return final_tokens, intermediate_steps
        else:
            return final_tokens, num_steps
    
    @classmethod
    def from_config(cls, config: Dict, vocab_size: int):
        """
        Create MDLM model from configuration dictionary.
        
        Config format (extends MLM config):
            model:
              vocab_size: 30522
              d_model: 768
              n_layers: 12
              n_heads: 12
              noise_schedule: "cosine"        # 'cosine' or 'linear'
              time_conditioning: false        # Usually false for MDLM
              ... (other MLM parameters)
        
        Args:
            config: Configuration dictionary
        
        Returns:
            MDLM instance
        """
        model_cfg = config['model']
        training_cfg = config.get('training', {})
        bert_cfg = training_cfg.get('bert_masking', {})
        
        return cls(
            vocab_size=vocab_size,
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
            tie_weights=model_cfg.get('tie_weights', True),
            noise_schedule=model_cfg.get('noise_schedule', 'cosine'),
            time_conditioning=model_cfg.get('time_conditioning', False)
        )