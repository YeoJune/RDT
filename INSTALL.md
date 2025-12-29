# RDT Installation Guide

## Quick Install

### From Source (Development)

```bash
# Clone or download the repository
cd RDT

# Install in editable mode with all dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

**Note:** All dependencies are defined in `pyproject.toml` and installed automatically.

### From PyPI (When Published)

```bash
pip install rdt-transformer
```

## Verification

### Test Installation

```bash
# Test import
python -c "import rdt; print(rdt.__version__)"

# Test model import
python -c "from rdt.models import RDT, MLM, CMLM; print('Models loaded successfully')"

# Check CLI commands
rdt-train --help
rdt-evaluate --help
rdt-inference --help
```

### Quick Start

```bash
# Train with base config
rdt-train --config rdt/configs/base.yaml

# Or if config not in package path
rdt-train --config configs/base.yaml

# Inference
rdt-inference --checkpoint checkpoints/best_model.pt --text "Hello world"
```

## Installation Options

### Standard Installation

```bash
pip install rdt-transformer
# or for development
pip install -e .
```

Includes:

- Core dependencies (torch, transformers, datasets, etc.)
- Evaluation metrics (bert-score, nltk)
- Training tools (wandb, matplotlib)
- Command-line tools (rdt-train, rdt-evaluate, rdt-inference, etc.)
- Configuration files

### Development Installation

```bash
pip install -e ".[dev]"
```

Additional tools:

- pytest (testing)
- black (code formatting)
- flake8 (linting)
- isort (import sorting)

### Minimal Installation (Not Recommended)

```bash
pip install --no-deps rdt-transformer
# Then manually install only what you need
pip install torch transformers datasets
```

## Requirements

### Python Version

- Python >= 3.8
- Tested on: 3.8, 3.9, 3.10, 3.11

### Core Dependencies

All dependencies are specified in `pyproject.toml`:

- **Deep Learning**: torch >= 2.0.0, transformers >= 4.30.0
- **Data Processing**: datasets >= 2.14.0, numpy >= 1.24.0
- **Evaluation**: bert-score >= 0.3.13, nltk >= 3.8
- **Training & Logging**: wandb >= 0.16.0, matplotlib >= 3.5.0
- **Utilities**: pyyaml >= 6.0, tqdm >= 4.65.0

### Hardware Requirements

- **Minimum**: 8GB RAM, CPU only
- **Recommended**: 16GB RAM, NVIDIA GPU (4GB+ VRAM)
- **Optimal**: 32GB RAM, NVIDIA GPU (8GB+ VRAM)

## Platform-Specific Notes

### Linux (Ubuntu/Debian)

```bash
# No additional steps needed
pip install rdt-transformer
```

### macOS

```bash
# Install with CPU-only PyTorch (recommended for M1/M2)
pip install rdt-transformer

# PyTorch will automatically use Apple Silicon GPU if available
```

### Windows

```bash
# Standard installation
pip install rdt-transformer

# If you have CUDA GPU, install PyTorch with CUDA first:
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install rdt-transformer --no-deps
pip install transformers datasets pyyaml tensorboard tqdm numpy
```

## GPU Support

### CUDA (NVIDIA GPUs)

```bash
# Install PyTorch with CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Then install RDT
pip install rdt-transformer
```

### ROCm (AMD GPUs)

```bash
# Install PyTorch with ROCm
pip install torch --index-url https://download.pytorch.org/whl/rocm5.6

# Then install RDT
pip install rdt-transformer
```

### MPS (Apple Silicon)

```bash
# Standard installation (MPS support included in PyTorch 2.0+)
pip install rdt-transformer
```

## Troubleshooting

### Import Errors

```bash
# If "No module named 'rdt'" error:
pip install --force-reinstall rdt-transformer

# Or install from source:
git clone <repository>
cd RDT
pip install -e .
```

### PyTorch Installation Issues

```bash
# Remove existing PyTorch
pip uninstall torch torchvision torchaudio

# Install specific version
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118
```

### CUDA Version Mismatch

```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch version
# For CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Memory Issues

```bash
# Reduce batch size in config
training:
  batch_size: 8  # or lower
```

### Dataset Download Issues

```bash
# If Hugging Face datasets fails, set cache directory:
export HF_DATASETS_CACHE="/path/to/cache"

# Or specify in code:
from datasets import load_dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir='/path/to/cache')
```

## Virtual Environment (Recommended)

### Using venv

```bash
# Create virtual environment
python -m venv rdt-env

# Activate
source rdt-env/bin/activate  # Linux/Mac
rdt-env\Scripts\activate     # Windows

# Install
pip install rdt-transformer
```

### Using conda

```bash
# Create conda environment
conda create -n rdt python=3.10

# Activate
conda activate rdt

# Install PyTorch with conda (recommended for conda users)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install RDT
pip install rdt-transformer
```

## Development Setup

```bash
# Clone repository
git clone <repository>
cd RDT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
python test_model.py
pytest tests/  # if tests directory exists

# Format code
black rdt/
isort rdt/

# Lint
flake8 rdt/
```

## Docker (Optional)

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU version)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install RDT
COPY . .
RUN pip install .

# Run
CMD ["rdt-train", "--config", "rdt/configs/base.yaml"]
```

Build and run:

```bash
docker build -t rdt .
docker run -v $(pwd)/checkpoints:/app/checkpoints rdt
```

## Verifying Installation

### Quick Test

```python
import torch
import rdt

# Check version
print(f"RDT version: {rdt.__version__}")

# Create small model
model = rdt.RDT(
    vocab_size=1000,
    d_model=128,
    n_encoder_layers=2,
    n_decoder_layers=1
)

# Test forward pass
x = torch.randint(0, 1000, (2, 10))
hidden, logits, gate = model(x)

print(f"✓ Model works! Output shape: {logits.shape}")
```

### Full Test

```bash
# Run provided test script
python test_model.py

# Expected output:
# Testing RDT Model Implementation...
# ==================================================
# 1. Creating model...
#    Model created with X parameters
# ...
# All tests passed! ✓
```

## Getting Help

If installation issues persist:

1. Check Python version: `python --version`
2. Check pip version: `pip --version`
3. Update pip: `pip install --upgrade pip`
4. Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
5. Create issue on GitHub with:
   - OS and Python version
   - Full error message
   - Installation command used

## Next Steps

After successful installation:

1. Read [USAGE_GUIDE.md](USAGE_GUIDE.md) for detailed usage
2. Check [README.md](README.md) for quick start
3. Explore configuration files in `rdt/configs/`
4. Try training: `rdt-train --config rdt/configs/base.yaml`
