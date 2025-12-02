"""RDT: Recursive Denoising Transformer

A PyTorch implementation of Recursive Denoising Transformer for progressive text denoising.
Enhanced with noise-level conditioning and directional recursive blocks.
"""

__version__ = "0.2.0"

from .model import (
    RDT,
    PositionalEncoding,
    TimestepEmbedder,
    DirectionalRecursiveBlock,
    GateMLP,
)
from .data import WikiTextDataset, create_dataloaders
from .trainer import RDTTrainer
from .utils import (
    load_config,
    merge_configs,
    set_seed,
    save_checkpoint,
    load_checkpoint,
    get_device,
    count_parameters,
)

__all__ = [
    # Model components
    "RDT",
    "PositionalEncoding",
    "TimestepEmbedder",
    "DirectionalRecursiveBlock",
    "GateMLP",
    # Data
    "WikiTextDataset",
    "create_dataloaders",
    # Training
    "RDTTrainer",
    # Utilities
    "load_config",
    "merge_configs",
    "set_seed",
    "save_checkpoint",
    "load_checkpoint",
    "get_device",
    "count_parameters",
]