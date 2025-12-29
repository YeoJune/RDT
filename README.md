# RDT: Recursive Denoising Transformer

> **An Iterative Text Refinement Framework via Latent Space Denoising**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pytorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![Architecture: Recursive](https://img.shields.io/badge/Architecture-Recursive_Transformer-purple)]()

**Recursive Denoising Transformer (RDT)** proposes a novel architecture that bridges the gap between **Autoregressive Transformers** and **Latent Diffusion Models**. Unlike traditional BERT-like models that attempt to reconstruct corrupted tokens directly in a single pass, RDT separates the generation process into **Latent Mapping** and **Latent Denoising**.

RDT operates on the insight that while text masking is a discrete and non-differentiable operation, the underlying semantic manifold is smooth. By projecting discrete corruptions into a continuous latent space, RDT transforms the "token prediction" problem into a simpler "vector field estimation" problem. Using a state-aware recursive mechanism and **Adaptive Layer Normalization (AdaLN)**, the model iteratively purifies hidden representations, allowing for parameter-efficient deep computation and robust reconstruction of complex semantic structures.

---

## 🧩 Methodology

### 1. Discrete Corruption, Continuous Refinement

Traditional MLMs struggle because they mix the burden of "understanding syntax" and "predicting discrete tokens" in every layer. RDT decouples these responsibilities:

1.  **Interface (I/O Encoders)**: Defines the topology of the latent manifold. It handles the translation between discrete tokens and continuous embeddings.
2.  **Engine (Recursive Block)**: Focuses solely on **Manifold Projection**. It learns a vector field that moves points from "noisy regions" back to the "semantic manifold center" ($h_{noisy} \to h_{clean}$).

$$
h_{t+1} = \mathcal{F}_\theta(h_t, \text{Emb}(g_t))
$$

### 2. Model Architecture

The inference process resembles a **Latent Diffusion** trajectory, but specifically adapted for discrete text via recursive computation.

```mermaid
graph LR
    subgraph "Interface (Token Space)"
        Input[Corrupted Tokens]
        Output[Refined Tokens]
    end

    subgraph "Recursive Engine (Latent Space)"
        H_in((h_t)) --> Norm[Input Norm]
        Norm --> Gate{Gate MLP}

        Gate --"Noise Level (t)"--> Time[Timestep Embedder]
        Time --"AdaLN Modulation (γ, β)"--> Block[Recursive Denoising Block]

        H_in --> Block
        Block --> H_out((h_t+1))
    end

    Input --"Input Encoder"--> H_in
    H_out --"Output Decoder"--> Output
    H_out -."Recursion Loop".-> H_in
```

#### A. Adaptive Layer Normalization (AdaLN)

To effectively reuse weights across different denoising stages, the model injects timestep information directly into the normalization layers. The affine parameters are dynamically generated based on the Gate's output:

$$ \text{AdaLN}(x, t) = (1 + \gamma(t)) \cdot \text{LayerNorm}(x) + \beta(t) $$

We utilize a **Zero-Initialization** strategy for $\gamma$ and $\beta$, ensuring that the recursive block starts as an identity function and gradually learns to modulate features as training progresses.

#### B. Self-Regulated Gating Mechanism

RDT includes a lightweight **Gate MLP** that acts as an internal clock. It diagnoses the entropy of the current hidden state to predict the restoration progress ($g_t$).

- **Residual Prediction**: The gate predicts the _decrease_ in noise ($\Delta$) rather than the absolute value ($g_{t+1} = g_t - \Delta$), ensuring a monotonically decreasing trajectory.
- **Adaptive Stopping**: During inference, the recursion terminates automatically when the gate score drops below a threshold.

---

## 📉 Optimization Objectives

The model is trained using a multi-task objective function ($ \mathcal{L}\_{total} $) that enforces structural integrity and temporal coherence in the latent space.

$$ \mathcal{L}_{total} = \mathcal{L}_{recon} + \lambda*{gate}\mathcal{L}*{gate} + \lambda*{latent}\mathcal{L}*{latent} $$

| Component              |         Symbol         | Description                                                                                                                                                              |
| :--------------------- | :--------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- | ------------------------------------------- | --- | ---------------------------------------------------------------------------------------------------- |
| **Reconstruction**     | $\mathcal{L}_{recon}$  | Cross-Entropy loss applied to the final logits. Ensures the final latent representation decodes into correct tokens.                                                     |
| **Gate Consistency**   |  $\mathcal{L}_{gate}$  | MSE loss ensuring the Gate MLP accurately estimates the ground-truth restoration percentage ($s_{GT}$).                                                                  |
| **Latent Consistency** | $\mathcal{L}_{latent}$ | **The Core Constraint.** We minimize the distance between the recursive state $h_t$ and the "ideal" state encoded from the ground-truth text by the Input Encoder.<br>$$ |     | h*{pred}^{(t)} - \text{Encoder}(x*{target}) |     | ^2 $$<br>This acts as **"Teacher Forcing" in the latent space**, simplifying the learning landscape. |

---

## 📂 Project Structure

The project is organized to clearly separate the neural architecture, training logic, and data pipeline.

```bash
rdt/
├── models/              # Core Neural Architectures
│   ├── rdt.py              # RDT Implementation (RoPE, MLP I/O, AdaLN, Gate MLP)
│   ├── mlm.py              # MLM Wrapper for BERT/RoBERTa
│   ├── cmlm.py             # Conditional MLM (Mask-Predict)
│   └── bert_init.py        # Pretrained weight initialization utilities
│
├── training/            # Training Logic
│   ├── rdt_trainer.py      # RDT Trainer (Latent Consistency + Gate Loss)
│   │                       #   with Scheduled Sampling for gate predictions
│   └── mlm_trainer.py      # Standard MLM Trainer for baselines
│
├── data/                # Data Pipeline
│   ├── datasets.py         # StreamingTextDataset & WikiTextDataset
│   └── collators.py        # RDTCollator, MLMCollator (masking & chain generation)
│
├── evaluation/          # Evaluation Tools
│   ├── evaluator.py        # Unified evaluator for RDT/MLM/CMLM
│   └── metrics.py          # Perplexity, Accuracy calculations
│
├── scripts/             # CLI Entry Points
│   ├── train.py            # Unified training script (RDT/MLM/CMLM)
│   ├── evaluate.py         # Model evaluation script
│   ├── inference.py        # Interactive inference for all models
│   ├── test_masking.py     # Masking sensitivity analysis
│   └── plot_training.py    # Training metrics visualization
│
├── configs/             # Hyperparameter Configurations
│   ├── base.yaml           # Default RDT configuration
│   ├── bert_baseline.yaml  # BERT baseline settings
│   └── roberta_baseline.yaml # RoBERTa baseline settings
│
└── utils/
    ├── utils.py            # General utilities & model creation
    └── logger.py           # Logging utilities
```

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/YeoJune/rdt.git
cd rdt

# Install dependencies (Editable mode for development)
pip install -e .

# Or install with development tools (pytest, black, flake8, isort)
pip install -e ".[dev]"
```

**Requirements:**

- Python 3.8+
- PyTorch 2.0+ (with CUDA support recommended)
- All dependencies are defined in `pyproject.toml` and installed automatically

---

## 🚀 Usage

### 1. Training

Train RDT, MLM, or CMLM models using the unified training script. The RDT trainer supports **Scheduled Sampling**, transitioning from Ground-Truth gate scores to Predicted gate scores to reduce exposure bias.

```bash
# Train RDT with default configuration
rdt-train --config rdt/configs/base.yaml --model_type rdt

# Train BERT baseline
rdt-train --config rdt/configs/bert_baseline.yaml --model_type mlm

# Train CMLM baseline
rdt-train --config rdt/configs/base.yaml --model_type cmlm --model_name bert-base-uncased

# Custom output directory
rdt-train --config rdt/configs/base.yaml --output_dir ./checkpoints/exp_01

# Resume from checkpoint
rdt-train --config rdt/configs/base.yaml --resume ./checkpoints/exp_01/checkpoint_epoch_5.pt
```

### 2. Inference

Run inference with any trained model (RDT/MLM/CMLM). RDT uses **Adaptive Stopping** based on gate scores for efficient computation.

```bash
# Interactive mode - RDT
rdt-inference --checkpoint checkpoints/rdt_best.pt --model_type rdt --interactive

# Interactive mode - MLM
rdt-inference --checkpoint checkpoints/mlm_best.pt --model_type mlm --interactive

# Interactive mode - CMLM
rdt-inference --checkpoint checkpoints/cmlm_best.pt --model_type cmlm --interactive

# Single text inference - RDT
rdt-inference --checkpoint checkpoints/rdt_best.pt --model_type rdt \
    --text "The quick brown [MASK] jumps over the lazy [MASK]." \
    --max_steps 10 --threshold 0.05

# Single text inference - MLM (single pass)
rdt-inference --checkpoint checkpoints/mlm_best.pt --model_type mlm \
    --text "The capital of France is [MASK]."

# Single text inference - CMLM (iterative refinement)
rdt-inference --checkpoint checkpoints/cmlm_best.pt --model_type cmlm \
    --text "The quick brown [MASK] jumps over the lazy [MASK]." \
    --max_iterations 10
```

**RDT Output Example:**

```text
[Step 0] Gate: 1.000 | The quick brown [MASK] jumps over the lazy [MASK].
[Step 1] Gate: 0.782 | The quick brown fox jumps over the lazy [MASK].
[Step 2] Gate: 0.234 | The quick brown fox jumps over the lazy dog.
[Step 3] Gate: 0.043 | The quick brown fox jumps over the lazy dog.
✓ Completed in 3 steps (Gate < 0.05)
```

### 3. Evaluation

Evaluate model performance on standard benchmarks. Supports accuracy, BLEU, and BERTScore metrics.

```bash
# Evaluate RDT model
rdt-evaluate --checkpoint checkpoints/rdt_best.pt --model_type rdt \
    --dataset wikitext --split test --mask_ratio 0.5

# Evaluate MLM baseline
rdt-evaluate --checkpoint checkpoints/mlm_best.pt --model_type mlm \
    --dataset wikitext --split test

# Evaluate CMLM baseline
rdt-evaluate --checkpoint checkpoints/cmlm_best.pt --model_type cmlm \
    --dataset wikitext --split test --max_iterations 10

# Masking sensitivity analysis
rdt-test-masking --checkpoint checkpoints/rdt_best.pt --model_type rdt \
    --dataset wikitext --num_samples 100
```

**Available Commands:**

- `rdt-train` - Train models (RDT/MLM/CMLM)
- `rdt-evaluate` - Evaluate trained models
- `rdt-inference` - Interactive or single-text inference
- `rdt-test-masking` - Analyze performance across different masking ratios
- `rdt-plot` - Visualize training metrics from logs

---

## 📊 Performance & Logging

RDT integrates with **Weights & Biases (W&B)** for real-time experiment tracking.

- **Training Metrics**: Loss (Total, Recon, Gate, Latent), Learning Rate, Sampling Probability.
- **Validation Metrics**: Accuracy, Perplexity, Gate Error.
- **Visualizations**: Gate score trajectories, Latent space convergence analysis.

To enable W&B, set `use_wandb: true` in your `configs/base.yaml`.

## 📜 Citation

If you find this code or architecture useful for your research, please cite:

```bibtex
@misc{rdt2025,
  title={RDT: Recursive Denoising Transformer via Latent Space Refinement},
  author={Yeo Joon},
  year={2025},
  publisher={GitHub},
  journal={arXiv preprint},
  howpublished={\url{https://github.com/YeoJune/rdt}}
}
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
