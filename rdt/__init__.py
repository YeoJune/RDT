"""RDT: Recursive Denoising Transformer

A PyTorch implementation of Recursive Denoising Transformer for progressive text denoising.
Includes baseline models: MLM (single-pass), CMLM (mask-predict), MDLM (diffusion).
"""

__version__ = "0.2.0"

# Import main modules
from . import models
from . import data  
from . import training
from . import evaluation
from . import utils

# Import commonly used classes for convenience
from .models import RDT, MLM, CMLM, MDLM
from .data import create_dataloaders, create_mlm_dataloaders, create_cmlm_dataloaders, create_mdlm_dataloaders
from .training import RDTTrainer, MLMTrainer
from .evaluation import Evaluator
from .utils import (
    load_config,
    merge_configs,
    set_seed,
    save_checkpoint,
    load_checkpoint,
    load_pretrained_weights,
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
    'MLM',
    'CMLM',
    'MDLM',
    # Data
    'create_dataloaders',
    'create_mlm_dataloaders',
    'create_cmlm_dataloaders',
    'create_mdlm_dataloaders',
    # Training
    'RDTTrainer',
    'MLMTrainer',
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
