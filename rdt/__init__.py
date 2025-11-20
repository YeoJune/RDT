"""RDT: Recursive Denoising Transformer

A PyTorch implementation of Recursive Denoising Transformer for progressive text denoising.
"""

__version__ = "0.1.0"

from .model import RDT, TransformerEncoder, LinearDecoder, TransformerDecoder, GateMLP
from .data import WikiTextDataset, ShuffleIndexMasking, create_dataloaders
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
    "TransformerEncoder",
    "LinearDecoder",
    "TransformerDecoder",
    "GateMLP",
    # Data
    "WikiTextDataset",
    "ShuffleIndexMasking",
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
