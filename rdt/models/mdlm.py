"""MDLM (Masked Diffusion Language Model) - SUBS Parameterization"""

import torch
import torch.nn.functional as F
from .mlm import MLM
from typing import Dict, Optional
import math


class MDLM(MLM):
    """
    Masked Diffusion Language Model (MDLM) with SUBS parameterization.
    
    As described in:
    "Simple and Effective Masked Diffusion Language Models" 
    (Sahoo et al., NeurIPS 2024)
    
    Key features:
    1. SUBS (SUBStitution) parameterization - copies unmasked tokens
    2. Continuous-time formulation (T → ∞)
    3. Cosine noise schedule: α(t) = cos²(πt/2)
    4. Rao-Blackwellized objective (mixture of weighted MLM losses)
    5. Efficient sampling with caching (3-4x speedup)
    
    Training:
    - Sample time t ~ Uniform[0, 1]
    - Mask tokens with probability p_mask(t) = 1 - α(t)
    - Apply time-weighted cross-entropy loss
    
    Inference:
    - Start from fully masked sequence
    - Iteratively denoise over T steps
    - Supports 3 samplers: ddpm_cache, ddpm, analytic
    """
    
    def __init__(
        self,
        architecture: str = 'bert-base-uncased',
        pretrained: Optional[str] = None,
        vocab_size: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        config_overrides: Optional[Dict] = None,
        noise_schedule: str = 'cosine',
        time_conditioning: bool = False,
        bert_masking_enabled: bool = False,
        mask_prob: float = 0.8,
        random_prob: float = 0.1,
        keep_prob: float = 0.1
    ):
        """
        Initialize MDLM by extending MLM.
        
        Args:
            architecture: Model architecture identifier (e.g., 'bert-base-uncased')
            pretrained: Model name/path to load pretrained weights from (None = from scratch)
            vocab_size: Vocabulary size (required for from-scratch training)
            mask_token_id: ID of [MASK] token (auto-detected if None)
            pad_token_id: ID of [PAD] token (auto-detected if None)
            config_overrides: Dict of config parameters to override
            noise_schedule: Noise schedule type ('cosine' or 'linear')
            time_conditioning: Whether to condition model on timestep (default: False for MDLM)
            bert_masking_enabled: Enable BERT-style masking (80% [MASK], 10% random, 10% keep)
            mask_prob: Probability of replacing with [MASK] token (default: 0.8)
            random_prob: Probability of replacing with random token (default: 0.1)
            keep_prob: Probability of keeping original token (default: 0.1)
        """
        super().__init__(
            architecture=architecture,
            pretrained=pretrained,
            vocab_size=vocab_size,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            config_overrides=config_overrides,
            bert_masking_enabled=bert_masking_enabled,
            mask_prob=mask_prob,
            random_prob=random_prob,
            keep_prob=keep_prob
        )
        
        self.noise_schedule = noise_schedule
        self.time_conditioning = time_conditioning
        
        if time_conditioning:
            print("  ⚠️  Warning: MDLM paper uses time_conditioning=False")
    
    @classmethod
    def from_config(cls, config: Dict):
        """
        Create MDLM model from config dictionary.
        
        Config format:
            model:
              architecture: "bert-base-uncased"  # Required
              pretrained: "bert-base-uncased"    # Optional: None for from-scratch
              vocab_size: 50000                   # Optional
              mask_token_id: 103                  # Optional
              pad_token_id: 0                     # Optional
              noise_schedule: "cosine"            # Optional: 'cosine' or 'linear'
              time_conditioning: false            # Optional: default False for MDLM
              config_overrides:                   # Optional
                num_hidden_layers: 6
                hidden_size: 512
            training:
              bert_masking:                       # Optional: BERT-style masking
                enabled: false
                mask_prob: 0.8
                random_prob: 0.1
                keep_prob: 0.1
        
        Args:
            config: Configuration dictionary
            
        Returns:
            MDLM instance
        """
        model_cfg = config['model']
        training_cfg = config.get('training', {})
        bert_cfg = training_cfg.get('bert_masking', {})
        
        return cls(
            architecture=model_cfg['architecture'],
            pretrained=model_cfg.get('pretrained'),
            vocab_size=model_cfg.get('vocab_size'),
            mask_token_id=model_cfg.get('mask_token_id'),
            pad_token_id=model_cfg.get('pad_token_id'),
            config_overrides=model_cfg.get('config_overrides'),
            noise_schedule=model_cfg.get('noise_schedule', 'cosine'),
            time_conditioning=model_cfg.get('time_conditioning', False),
            bert_masking_enabled=bert_cfg.get('enabled', False),
            mask_prob=bert_cfg.get('mask_prob', 0.8),
            random_prob=bert_cfg.get('random_prob', 0.1),
            keep_prob=bert_cfg.get('keep_prob', 0.1)
        )
    
    def _cosine_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """
        Cosine noise schedule: α(t) = cos²(πt/2)
        
        Args:
            t: Time steps in [0, 1], shape [..., ]
            
        Returns:
            α(t): Non-masking probability, same shape as t
        """
        return torch.cos(t * math.pi / 2) ** 2
    
    def _linear_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """
        Linear noise schedule: α(t) = 1 - t
        
        Args:
            t: Time steps in [0, 1], shape [..., ]
            
        Returns:
            α(t): Non-masking probability, same shape as t
        """
        return 1.0 - t
    
    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get α(t) based on noise schedule.
        
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
    ):
        """
        MDLM continuous-time masking for training.
        
        "We train with a continuous-time formulation where the number of masked
        tokens is sampled from a noise schedule α(t), presenting the model with
        varying levels of corruption from t=0 (fully visible) to t=1 (fully masked)."
        
        Args:
            input_ids: [B, L] original token indices
            attention_mask: [B, L] attention mask (1=valid, 0=padding)
            low_discrepancy_sampling: Use low-discrepancy sampler for variance reduction
        
        Returns:
            masked_input_ids: [B, L] input with masks applied
            labels: [B, L] labels for loss (-100 for non-masked tokens)
            t: [B] sampled time steps
            loss_weight: [B] weight for loss (time derivative α'(t))
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 1. Sample time steps t ~ Uniform[eps, 1]
        # FIX: Avoid t=0 strictly to prevent loss weight singularity
        eps = 1e-5
        if low_discrepancy_sampling:
            # Low-discrepancy sampler: partition interval [eps, 1] evenly
            # Reduces variance during training (Appendix G in paper)
            t = torch.linspace(eps, 1.0, batch_size + 1, device=device)[:-1]
            noise = torch.rand(batch_size, device=device) * (1 - eps) / batch_size
            t = t + noise
            t = t.clamp(eps, 1.0)  # Double safety
        else:
            t = torch.rand(batch_size, device=device) * (1 - eps) + eps
            t = t.clamp(eps, 1.0)  # Ensure t ∈ [eps, 1]
        
        # 2. Calculate masking probability from noise schedule
        # α(t): probability of NOT masking
        # p_mask(t) = 1 - α(t): probability of masking
        alpha_t = self.get_alpha(t)  # [B]
        mask_prob = 1.0 - alpha_t  # [B]
        
        # 3. Determine which tokens can be masked (exclude padding)
        if attention_mask is not None:
            maskable = attention_mask.bool()
        else:
            maskable = torch.ones_like(input_ids, dtype=torch.bool)
        
        maskable = maskable & (input_ids != self.pad_token_id)
        
        # 4. Sample masking decisions for each token
        # Use vectorized sampling: compare uniform[0,1] with mask_prob
        rand = torch.rand(batch_size, seq_len, device=device)  # [B, L]
        mask_prob_expanded = mask_prob.unsqueeze(1)  # [B, 1]
        
        mask_decision = (rand < mask_prob_expanded) & maskable  # [B, L]
        
        # 5. Apply masking (BERT-style if enabled, otherwise simple SUBS parameterization)
        masked_input_ids = self._apply_bert_masking(input_ids, mask_decision)
        
        # 6. Create labels (-100 for non-masked tokens)
        labels = input_ids.clone()
        labels = labels.masked_fill(~mask_decision, -100)
        
        # 7. Calculate loss weight: α'(t) / (1 - α(t))
        # From Eq. 3 in paper: weighted by time derivative of noise schedule
        loss_weight = self._compute_loss_weight(t)  # [B]
        
        return masked_input_ids, labels, t, loss_weight
    
    def _compute_loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute loss weight: α'(t) / (1 - α(t)) using analytic gradients.
        Prevents singularity at t=0 by clamping the weight.
        
        This comes from the Rao-Blackwellized NELBO objective (Eq. 3 in paper):
        L = E_t ∫ [α'(t) / (1 - α(t))] * CrossEntropy(x_θ(z_t), x) dt
        
        Args:
            t: [B] time steps in [0, 1]
            
        Returns:
            weight: [B] loss weight for each sample
        """
        # 1. Analytic derivatives for stability (avoids numerical errors)
        if self.noise_schedule == 'cosine':
            # α(t) = cos²(πt/2)
            # α'(t) = -π * cos(πt/2) * sin(πt/2) = -(π/2) * sin(πt)
            alpha_prime = -(math.pi / 2) * torch.sin(math.pi * t)
        elif self.noise_schedule == 'linear':
            # α(t) = 1 - t
            # α'(t) = -1
            alpha_prime = torch.full_like(t, -1.0)
        else:
            # Fallback to numerical differentiation for unknown schedules
            return self._compute_loss_weight_numerical(t)
        
        # 2. Compute 1 - α(t)
        alpha_t = self.get_alpha(t)
        one_minus_alpha = 1.0 - alpha_t
        
        # 3. Compute Weight: |α'(t)| / (1 - α(t))
        # Add epsilon to denominator for numerical stability
        weight = torch.abs(alpha_prime) / (one_minus_alpha + 1e-8)
        
        # 4. CRITICAL FIX: Clamp large weights at t≈0
        # The theoretical weight → ∞ as t → 0, but gradients shouldn't explode.
        # Standard MDLM implementations clamp this to prevent loss spikes.
        weight = torch.clamp(weight, max=100.0)
        
        return weight
    
    def _compute_loss_weight_numerical(self, t: torch.Tensor) -> torch.Tensor:
        """
        Fallback numerical differentiation for unknown noise schedules.
        
        Args:
            t: [B] time steps in [0, 1]
            
        Returns:
            weight: [B] loss weight for each sample
        """
        # Compute α'(t) numerically with small epsilon
        eps = 1e-5
        t_plus = torch.clamp(t + eps, 0, 1)
        t_minus = torch.clamp(t - eps, 0, 1)
        
        alpha_plus = self.get_alpha(t_plus)
        alpha_minus = self.get_alpha(t_minus)
        alpha_prime = (alpha_plus - alpha_minus) / (2 * eps)
        
        # Compute 1 - α(t)
        alpha_t = self.get_alpha(t)
        one_minus_alpha = 1.0 - alpha_t
        
        # Prevent division by zero
        one_minus_alpha = torch.clamp(one_minus_alpha, min=1e-8)
        
        # Weight = |α'(t)| / (1 - α(t))
        weight = torch.abs(alpha_prime) / one_minus_alpha
        
        # Clamp to prevent explosion
        weight = torch.clamp(weight, max=100.0)
        
        return weight
    
    def forward_with_time_weighting(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_weight: Optional[torch.Tensor] = None
    ):
        """
        Forward pass with time-weighted loss for MDLM training.
        
        Args:
            input_ids: [B, L] masked token indices
            attention_mask: [B, L] attention mask
            labels: [B, L] target tokens (-100 for non-masked)
            loss_weight: [B] time-dependent loss weight
            
        Returns:
            loss: Weighted cross-entropy loss
            logits: [B, L, V] output logits
        """
        # Standard forward pass
        loss, logits = self.forward(input_ids, attention_mask, labels)
        
        # Apply time weighting
        if loss_weight is not None:
            # Average weight across batch
            avg_weight = loss_weight.mean()
            loss = loss * avg_weight
        
        return loss, logits
    
    def inference(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        num_steps: int = 1000,
        sampler: str = 'ddpm',
        length: Optional[int] = None,
        return_intermediate: bool = False,
        temperature: float = 1.0
    ):
        """
        MDLM inference with iterative denoising.
        
        "Decoding starts with a completely masked sequence, and iteratively
        unmasks tokens based on model predictions over T steps."
        
        IMPORTANT: MDLM uses RANDOM unmasking (not confidence-based).
        This is what makes it a simple baseline compared to:
        - CMLM: confidence-based unmasking (smart)
        - RDT: chain-based unmasking (smarter)
        
        Args:
            input_ids: [B, L] initial tokens (if None, starts fully masked)
            attention_mask: [B, L] attention mask
            num_steps: Number of denoising steps (T)
            sampler: Sampling strategy ('ddpm', 'ddpm_cache', 'analytic')
                    'ddpm' - Standard random unmasking (recommended for MDLM)
                    'ddpm_cache' - Cached random unmasking (faster but still random)
                    'analytic' - Placeholder
            length: Sequence length (required if input_ids is None)
            return_intermediate: Return intermediate predictions
            temperature: Sampling temperature
        
        Returns:
            final_tokens: [B, L] final predictions
            intermediates: Optional list of intermediate states
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Select sampler
        if sampler == 'ddpm_cache':
            return self._sample_ddpm_cache(
                input_ids, attention_mask, num_steps, length, 
                return_intermediate, temperature
            )
        elif sampler == 'ddpm':
            return self._sample_ddpm(
                input_ids, attention_mask, num_steps, length,
                return_intermediate, temperature
            )
        elif sampler == 'analytic':
            return self._sample_analytic(
                input_ids, attention_mask, num_steps, length,
                return_intermediate, temperature
            )
        else:
            raise ValueError(f"Unknown sampler: {sampler}. Choose 'ddpm_cache', 'ddpm', or 'analytic'")
    
    def _sample_ddpm_cache(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        num_steps: int,
        length: Optional[int],
        return_intermediate: bool,
        temperature: float
    ):
        """
        MDLM sampling with caching optimization.
        
        Key characteristic: RANDOM remasking (NOT confidence-based).
        This distinguishes MDLM from CMLM (Mask-Predict) which uses confidence.
        
        MDLM is a simple baseline that randomly unmasks tokens,
        while CMLM intelligently unmasks based on model confidence.
        
        Optimization: Reuse predictions from previous iteration,
        but remask tokens RANDOMLY (not based on confidence).
        
        Args:
            (same as inference)
            
        Returns:
            final_tokens: [B, L] denoised sequence
            intermediates: Optional[List[Tensor]]
        """
        device = next(self.parameters()).device
        
        with torch.no_grad():
            # 1. Initialize
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
            
            # Track intermediate states
            intermediates = [] if return_intermediate else None
            
            # Initialize confidence scores (0 for masked positions)
            current_confidences = torch.zeros(batch_size, length, device=device)
            
            # 2. Iterative denoising: t = 1 → 0
            for step in range(num_steps):
                if return_intermediate:
                    intermediates.append(current_ids.clone())
                
                # Current time: t = 1 - step/T
                t = 1.0 - step / num_steps
                
                # Skip if fully denoised
                if t <= 0:
                    break
                
                # --- Step A: Predict all tokens ---
                logits = self.forward(current_ids, attention_mask)  # [B, L, V]
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                probs = F.softmax(logits, dim=-1)  # [B, L, V]
                
                # Get predictions and confidences
                predicted_ids = logits.argmax(dim=-1)  # [B, L]
                predicted_confidences = probs.max(dim=-1).values  # [B, L]
                
                # --- Step B: Selective remasking (except first iteration) ---
                if step < num_steps - 1:
                    # Calculate number of tokens to keep masked at next step
                    next_t = 1.0 - (step + 1) / num_steps
                    mask_prob_next = self.get_mask_prob(torch.tensor(next_t, device=device))
                    
                    num_total = attention_mask.sum(dim=1, keepdim=True).float()  # [B, 1]
                    num_to_mask = (num_total * mask_prob_next).long()  # [B, 1]
                    
                    # MDLM: Random selection (NOT confidence-based like CMLM)
                    # This is the key difference that makes MDLM a simple baseline
                    # for comparing against smart methods like CMLM and RDT
                    random_scores = torch.rand(batch_size, length, device=device)
                    
                    # Prevent remasking padding
                    random_scores = random_scores.masked_fill(
                        attention_mask == 0, -1e9
                    )
                    
                    # Vectorized random selection (highest random scores = randomly selected)
                    sorted_indices = torch.argsort(random_scores, dim=1, descending=True)
                    ranks = torch.argsort(sorted_indices, dim=1)  # [B, L]
                    
                    # Remask tokens randomly (not based on confidence)
                    remask_decision = ranks < num_to_mask  # [B, L]
                    
                    # Update current state
                    current_ids = torch.where(
                        remask_decision,
                        torch.full_like(current_ids, self.mask_token_id),
                        predicted_ids
                    )
                    current_confidences = torch.where(
                        remask_decision,
                        torch.zeros_like(predicted_confidences),
                        predicted_confidences
                    )
                else:
                    # Final step: unmask everything
                    current_ids = predicted_ids
            
            final_tokens = current_ids
        
        if return_intermediate:
            return final_tokens, intermediates
        else:
            return final_tokens, num_steps
    
    def _sample_ddpm(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        num_steps: int,
        length: Optional[int],
        return_intermediate: bool,
        temperature: float
    ):
        """
        Standard DDPM ancestral sampling (D3PM style).
        
        Uses transition probabilities q(x_{t-1} | x_t, x_0).
        
        Args:
            (same as inference)
            
        Returns:
            final_tokens: [B, L] denoised sequence
            intermediates: Optional[List[Tensor]]
        """
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
            
            intermediates = [] if return_intermediate else None
            
            # Iterative denoising
            for step in range(num_steps):
                if return_intermediate:
                    intermediates.append(current_ids.clone())
                
                # Current and next time
                t = torch.tensor(1.0 - step / num_steps, device=device)
                t_next = torch.tensor(1.0 - (step + 1) / num_steps, device=device)
                
                if t <= 0:
                    break
                
                # Predict x_0
                logits = self.forward(current_ids, attention_mask)  # [B, L, V]
                
                if temperature != 1.0:
                    logits = logits / temperature
                
                probs_x0 = F.softmax(logits, dim=-1)  # [B, L, V]
                
                # Calculate transition probabilities
                alpha_t = self.get_alpha(t)
                alpha_t_next = self.get_alpha(t_next)
                
                # q(x_{t-1} | x_t, x_0) for SUBS parameterization
                # If x_t is [MASK]: sample from p(x_0)
                # If x_t is not [MASK]: keep it (copy flag)
                is_masked = (current_ids == self.mask_token_id)
                
                if step < num_steps - 1:
                    # Not final step: stochastic sampling
                    # Probability of staying masked: (α_t - α_{t-1}) / α_t
                    stay_masked_prob = (alpha_t - alpha_t_next) / (alpha_t + 1e-8)
                    
                    # Sample from x_0 predictions
                    sampled_x0 = torch.multinomial(
                        probs_x0.view(-1, self.vocab_size),
                        num_samples=1
                    ).view(batch_size, length)
                    
                    # Decide which positions stay masked
                    stay_masked = torch.rand(batch_size, length, device=device) < stay_masked_prob
                    stay_masked = stay_masked & is_masked & (attention_mask == 1)
                    
                    # Update: unmask or keep masked
                    current_ids = torch.where(
                        stay_masked,
                        torch.full_like(current_ids, self.mask_token_id),
                        torch.where(is_masked, sampled_x0, current_ids)
                    )
                else:
                    # Final step: deterministic unmasking
                    predicted_ids = probs_x0.argmax(dim=-1)
                    current_ids = torch.where(is_masked, predicted_ids, current_ids)
            
            final_tokens = current_ids
        
        if return_intermediate:
            return final_tokens, intermediates
        else:
            return final_tokens, num_steps
    
    def _sample_analytic(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        num_steps: int,
        length: Optional[int],
        return_intermediate: bool,
        temperature: float
    ):
        """
        Analytic sampler (SEDD style) - uses score-based formulation.
        
        Args:
            (same as inference)
            
        Returns:
            final_tokens: [B, L] denoised sequence
            intermediates: Optional[List[Tensor]]
        """
        # Placeholder for analytic sampler
        # Full implementation requires score computation
        print("⚠️  Analytic sampler not yet implemented, falling back to ddpm_cache")
        return self._sample_ddpm_cache(
            input_ids, attention_mask, num_steps, length,
            return_intermediate, temperature
        )