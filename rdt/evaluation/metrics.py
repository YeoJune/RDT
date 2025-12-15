"""Evaluation metrics for language models"""

import torch
import torch.nn.functional as F
import math
from typing import Dict, Optional


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss value
        
    Returns:
        Perplexity value
    """
    return math.exp(loss)


def calculate_accuracy(logits: torch.Tensor, targets: torch.Tensor, 
                       ignore_index: int = -100) -> float:
    """
    Calculate token-level accuracy.
    
    Args:
        logits: [B, L, V] model predictions
        targets: [B, L] target tokens
        ignore_index: Index to ignore (e.g., padding, -100 for MLM)
        
    Returns:
        Accuracy as float
    """
    predictions = logits.argmax(dim=-1)
    mask = targets != ignore_index
    
    if mask.sum() == 0:
        return 0.0
    
    correct = (predictions == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    return accuracy.item()


def calculate_top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor,
                              k: int = 5, ignore_index: int = -100) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        logits: [B, L, V] model predictions
        targets: [B, L] target tokens
        k: Top-k value
        ignore_index: Index to ignore
        
    Returns:
        Top-k accuracy as float
    """
    mask = targets != ignore_index
    
    if mask.sum() == 0:
        return 0.0
    
    # Get top-k predictions
    _, top_k_preds = logits.topk(k, dim=-1)  # [B, L, k]
    
    # Expand targets to match top_k_preds shape
    targets_expanded = targets.unsqueeze(-1).expand_as(top_k_preds)  # [B, L, k]
    
    # Check if true target is in top-k
    correct = (top_k_preds == targets_expanded).any(dim=-1) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    return accuracy.item()


def calculate_loss(logits: torch.Tensor, targets: torch.Tensor,
                   ignore_index: int = -100) -> float:
    """
    Calculate cross-entropy loss.
    
    Args:
        logits: [B, L, V] model predictions
        targets: [B, L] target tokens
        ignore_index: Index to ignore
        
    Returns:
        Loss value
    """
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=ignore_index
    )
    return loss.item()


def calculate_metrics(logits: torch.Tensor, targets: torch.Tensor,
                      ignore_index: int = -100) -> Dict[str, float]:
    """
    Calculate all standard metrics at once.
    
    Args:
        logits: [B, L, V] model predictions
        targets: [B, L] target tokens
        ignore_index: Index to ignore
        
    Returns:
        Dictionary of metrics
    """
    loss = calculate_loss(logits, targets, ignore_index)
    accuracy = calculate_accuracy(logits, targets, ignore_index)
    top5_accuracy = calculate_top_k_accuracy(logits, targets, k=5, ignore_index=ignore_index)
    perplexity = calculate_perplexity(loss)
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'perplexity': perplexity
    }
