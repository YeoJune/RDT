"""RDT: Recursive Denoising Transformer

A PyTorch implementation of Recursive Denoising Transformer for progressive text denoising.
Enhanced with noise-level conditioning and directional recursive blocks.
"""

__version__ = "0.2.0"

# Import main modules
from . import models
from . import data  
from . import training
from . import evaluation
from . import utils

# Import commonly used classes for convenience
from .models import RDT, BaselineMLM
from .data import create_dataloaders
from .training import RDTTrainer, BaselineTrainer
from .evaluation import Evaluator
from .utils import (
    load_config,
    merge_configs,
    set_seed,
    save_checkpoint,
    load_checkpoint,
    load_pretrained_weights,
    get_device,
    count_parameters,
)

__all__ = [
    # Modules
    'models',
    'data',
    'training',
    'evaluation',
    'utils',
    # Models
    'RDT',
    'BaselineMLM',
    # Data
    'create_dataloaders',
    # Training
    'RDTTrainer',
    'BaselineTrainer',
    # Evaluation
    'Evaluator',
    # Utilities
    'load_config',
    'merge_configs',
    'set_seed',
    'save_checkpoint',
    'load_checkpoint',
    'load_pretrained_weights',
    'get_device',
    'count_parameters',
]
