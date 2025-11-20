# RDT: Recursive Denoising Transformer

A PyTorch implementation of Recursive Denoising Transformer, a model that learns to progressively denoise masked text through iterative refinement.

## Architecture

- **Shared Encoder**: Recursive transformer encoder that refines latent representations
- **Shallow Decoder**: Lightweight decoder (linear or 1-layer) for token generation
- **Gate MLP**: Predicts remaining denoising steps for adaptive computation

## Key Features

- **Progressive Denoising**: Trains on sequences with gradually decreasing mask ratios
- **Adaptive Inference**: Gate-controlled recursive steps (easy inputs → fewer steps)
- **Memory Efficient**: Chain sampling reduces VRAM usage during training
- **Parameter Efficient**: Single encoder reused recursively

## Installation

```bash
# Extract package
tar -xzf RDT-v0.1.0.tar.gz  # or unzip RDT-v0.1.0.zip
cd RDT

# Install
pip install -e .

# Verify
python check_compatibility.py
```

For detailed installation, see [INSTALL.md](INSTALL.md).

## Quick Start
```

## Quick Start

### Training

Train on WikiText-2 with base configuration:
```bash
# Using CLI command (if installed as package)
rdt-train --config rdt/configs/base.yaml

# Or using Python script directly
python rdt/scripts/train.py --config configs/base.yaml
```

Train on WikiText-103 with experiment config:
```bash
rdt-train --config rdt/configs/base.yaml --override rdt/configs/experiment.yaml
```

Resume from checkpoint:
```bash
rdt-train --config rdt/configs/base.yaml --checkpoint checkpoints/checkpoint_epoch_5.pt
```

### Inference

Single text inference:
```bash
# Using CLI command
rdt-inference --checkpoint checkpoints/best_model.pt --text "Your text here"

# Or using Python script directly
python rdt/scripts/inference.py --checkpoint checkpoints/best_model.pt --text "Your text here"
```

Interactive mode:
```bash
rdt-inference --checkpoint checkpoints/best_model.pt --interactive
```

With custom parameters:
```bash
rdt-inference --checkpoint checkpoints/best_model.pt \
    --text "Sample text" \
    --max_steps 30 \
    --threshold 0.05
```

## Configuration

See `configs/base.yaml` for all configuration options:

### Model Architecture
- `d_model`: Hidden dimension (default: 512)
- `n_encoder_layers`: Encoder depth (default: 6)
- `n_decoder_layers`: Decoder depth (default: 1)
- `decoder_type`: 'linear' or 'transformer'

### Training Strategy
- `max_chain_length`: Training segment length (default: 5)
- `total_steps`: Total denoising steps (default: 10)
- `masking_strategy`: 'linear' (step k → k*10% masking)

### Data
- `dataset_name`: 'wikitext-2' or 'wikitext-103'
- `max_seq_length`: Maximum sequence length (default: 512)

## Project Structure

```
RDT/
├── rdt/                      # Main package
│   ├── __init__.py          # Package exports
│   ├── model.py             # RDT architecture
│   ├── data.py              # Data loading & masking
│   ├── trainer.py           # Training loop
│   ├── utils.py             # Utilities
│   ├── configs/             # Configuration files
│   │   ├── base.yaml
│   │   └── experiment.yaml
│   └── scripts/             # Command-line scripts
│       ├── train.py
│       └── inference.py
├── configs/                 # User-editable configs
│   ├── base.yaml
│   └── experiment.yaml
├── checkpoints/             # Model checkpoints
├── runs/                    # TensorBoard logs
├── setup.py                 # Package setup (legacy)
├── pyproject.toml          # Modern package config
├── requirements.txt
├── README.md
├── INSTALL.md              # Detailed installation guide
├── USAGE_GUIDE.md          # Detailed usage guide
└── LICENSE
```

## Monitoring

View training progress with TensorBoard:
```bash
tensorboard --logdir runs/
```

Metrics logged:
- Train/Val: Total loss, Reconstruction loss, Gate loss
- Learning rate schedule
- Per-step losses during recursive forward pass

## Design Philosophy

Based on research showing deep encoders with shallow decoders achieve optimal performance:
- **MAE (He et al., 2022)**: Encoder extracts features, decoder just reconstructs
- **ALBERT (Lan et al., 2020)**: Parameter sharing (recursion) maintains performance

In RDT:
1. Encoder does the "thinking" (recursive refinement)
2. Decoder is just a "translator" (latent → tokens)
3. Gate learns when to stop (adaptive computation)

## Citation

If you use this code, please cite:
```bibtex
@software{rdt2024,
  title={RDT: Recursive Denoising Transformer},
  author={Your Name},
  year={2024}
}
```

## License

MIT License
