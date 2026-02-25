# RDT: Recursive Denoising Transformer

> **An Iterative Text Refinement Framework via Latent Space Denoising**

**Recursive Denoising Transformer (RDT)** proposes a novel architecture that bridges the gap between Autoregressive Transformers and Latent Diffusion Models. Unlike traditional BERT-like models or standard diffusion models that rely on fixed denoising steps, RDT separates the generation process into **Latent Mapping** and **Latent Denoising**, enabling adaptive, dynamic text refinement.

By projecting discrete corrupted tokens into a continuous latent space, RDT utilizes a state-aware recursive mechanism and **Adaptive Layer Normalization (AdaLN)** to iteratively purify hidden representations. A built-in Gate MLP predicts the remaining noise level, allowing the model to dynamically halt computation (Adaptive Stopping) once the text is fully refined, drastically reducing average inference steps.

---

## 🧩 Methodology

### 1. Discrete Corruption, Continuous Refinement

RDT decouples the responsibility of syntax understanding and token prediction:

- **Interface (I/O Encoders):** Translates between discrete tokens and continuous embeddings.
- **Engine (Recursive Block):** A shared Transformer block that repeatedly projects the latent state $h_t$ towards the semantic manifold center ($h_{t+1}$).

### 2. Adaptive Stopping & AdaLN

- **AdaLN:** Injects the current noise level (Gate Score) into the normalization layers. Initialized with zeros, it starts as an identity function and smoothly learns to modulate features.
- **Gate MLP:** Diagnoses the entropy of the current hidden state to predict restoration progress. During inference, recursion terminates automatically when the gate score drops below a predefined threshold.

---

## 📊 Experimental Results

RDT was evaluated on the **Wikitext-103** dataset against MDLM (Diffusion) and CMLM (Mask-Predict) baselines.

### Small-Scale Setup (Validation)

Initial validation experiments were conducted with a compact configuration:

- **Architecture:** `d_model=512`, `n_layers=6`, `seq_len=128`
- **Training:** `lr=2e-4`, 30 epochs
- **Parameters:** RDT ~40M (due to AdaLN modules), Baselines ~34M
- **RDT Config:** `total_steps=20`, `chain_length=1~2`, layered structure (1-4-1: input processor, recursive encoder, output processor)
- **Metrics:** Exact Match, BERTScore (F1, via `bert-large`), BLEU4, and Perplexity (via `gpt2-large`)

### Base-Scale Setup (Main)

The main experiments use standard BERT-base scale for fair comparison:

- **Architecture:** `d_model=768`, `n_heads=12`, `seq_len=256`
- **Layers:** MDLM/CMLM: 12 layers, RDT: 2-6-2 (10 total: input processor, recursive encoder, output processor)
- **Training:** `lr=2e-4`, 30 epochs, `batch_size=64`
- **RDT Config:** `total_steps=20`, `chain_length=1~2`
- **Rationale:** RDT uses fewer layers to match parameter count with baselines, as AdaLN modules increase parameters by ~15%

### Reconstruction Quality & Efficiency (Summary)

RDT achieves comparable or slightly superior generation quality compared to a 1000-step MDLM, while drastically reducing the average inference steps to under 15 via its adaptive stopping mechanism.

| Model    | Params | Masking | Exact Match | BERTScore  | BLEU4      | PPL (↓) | Avg Steps |
| -------- | ------ | ------- | ----------- | ---------- | ---------- | ------- | --------- |
| **MDLM** | 34M    | 10%     | 0.5307      | 0.9852     | 0.8717     | 55.08   | 1000.00   |
| **CMLM** | 34M    | 10%     | 0.5771      | 0.9866     | 0.8842     | 53.76   | 10.00     |
| **RDT**  | 40M    | 10%     | **0.5481**  | **0.9863** | **0.8788** | 60.53   | **2.38**  |
|          |        |         |             |            |            |         |           |
| **MDLM** | 34M    | 50%     | 0.3332      | 0.9022     | 0.3456     | 109.49  | 1000.00   |
| **CMLM** | 34M    | 50%     | 0.2505      | 0.8793     | 0.3051     | 171.07  | 10.00     |
| **RDT**  | 40M    | 50%     | **0.3338**  | **0.9064** | **0.3471** | 118.77  | **7.62**  |
|          |        |         |             |            |            |         |           |
| **MDLM** | 34M    | 90%     | 0.1196      | 0.7774     | 0.0327     | 36.84   | 1000.00   |
| **CMLM** | 34M    | 90%     | 0.0906      | 0.8159     | 0.0423     | 46.44   | 10.00     |
| **RDT**  | 40M    | 90%     | **0.1102**  | **0.7918** | **0.0306** | 65.47   | **13.83** |

<details>
<summary><b>View Full Experimental Results (0% ~ 100% Masking)</b></summary>

| Model    | Masking | Exact Match | BERTScore | BLEU4  | PPL    | Avg Steps |
| -------- | ------- | ----------- | --------- | ------ | ------ | --------- |
| **MDLM** | 0%      | 1.0000      | 1.0000    | 1.0000 | 49.34  | 1000.00   |
| **MDLM** | 20%     | 0.4856      | 0.9668    | 0.7298 | 64.45  | 1000.00   |
| **MDLM** | 40%     | 0.3867      | 0.9257    | 0.4670 | 91.28  | 1000.00   |
| **MDLM** | 60%     | 0.2796      | 0.8790    | 0.2439 | 120.24 | 1000.00   |
| **MDLM** | 80%     | 0.1732      | 0.8217    | 0.0829 | 85.94  | 1000.00   |
| **MDLM** | 100%    | 0.0697      | 0.7423    | 0.0040 | 1.96   | 1000.00   |
| **CMLM** | 0%      | 1.0000      | 1.0000    | 1.0000 | 49.34  | 10.00     |
| **CMLM** | 20%     | 0.4630      | 0.9590    | 0.7205 | 84.14  | 10.00     |
| **CMLM** | 40%     | 0.2988      | 0.9029    | 0.4179 | 161.05 | 10.00     |
| **CMLM** | 60%     | 0.2066      | 0.8607    | 0.2138 | 157.79 | 10.00     |
| **CMLM** | 80%     | 0.1354      | 0.8316    | 0.0893 | 85.04  | 10.00     |
| **CMLM** | 100%    | 0.0368      | 0.7671    | 0.0076 | 10.77  | 10.00     |
| **RDT**  | 0%      | 1.0000      | 1.0000    | 1.0000 | 55.97  | 1.75      |
| **RDT**  | 20%     | 0.4998      | 0.9690    | 0.7401 | 68.57  | 3.54      |
| **RDT**  | 40%     | 0.3909      | 0.9292    | 0.4689 | 98.27  | 6.18      |
| **RDT**  | 60%     | 0.2772      | 0.8836    | 0.2439 | 139.03 | 9.02      |
| **RDT**  | 80%     | 0.1615      | 0.8300    | 0.0788 | 131.71 | 12.20     |
| **RDT**  | 100%    | 0.0582      | 0.7367    | 0.0037 | 2.60   | 14.93     |

</details>

---

## � Hardware Requirements

### Recommended Specifications (Base-Scale)

- **GPU:** 32GB VRAM (e.g., NVIDIA RTX 5090, A40, V100)
- **Training Time** (per epoch on RTX 5090, Base config):
  - MDLM: ~1h 40min
  - CMLM: ~1h 20min
  - RDT: ~2h 10min (longer due to chain-based training)
- **Inference:** 8GB+ VRAM recommended

### Memory Optimization Tips

If GPU memory is limited, consider:

- Reduce `batch_size` (default: 64)
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Enable `gradient_checkpointing: true` in model config
- Reduce `max_seq_length` (default: 256)

---

## �📂 Project Structure

```bash
rdt/
├── models/              # Core Neural Architectures (RDT, MLM, CMLM, MDLM)
├── training/            # Training Logic (Latent Consistency, Gate Loss, Scheduled Sampling)
├── data/                # Data Pipeline (StreamingTextDataset, WikiTextDataset)
├── evaluation/          # Unified evaluator & metrics (Perplexity, BERTScore, Accuracy)
├── scripts/             # CLI Entry Points (train, evaluate, inference)
└── configs/             # Hyperparameter Configurations

```

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/YeoJune/rdt.git
cd rdt

# Install dependencies (Editable mode for development)
pip install -e .

```

_Requirements: Python 3.8+, PyTorch 2.0+ (CUDA support recommended)_

---

## ⚡ Quick Start

### Minimal Working Example

```python
import torch
from transformers import AutoTokenizer
from rdt.models import RDT
from rdt.utils import load_config, create_model_from_config

# Load config and tokenizer
config = load_config("rdt/configs/rdt.yaml")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Create model
model = create_model_from_config(config, vocab_size=tokenizer.vocab_size)
model.eval()

# Prepare corrupted input
text = "The quick brown [MASK] jumps over the lazy [MASK]."
inputs = tokenizer(text, return_tensors="pt")

# Run adaptive denoising
with torch.no_grad():
    output_ids, num_steps = model.inference(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_steps=20,
        threshold=0.05
    )

# Decode result
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Input:  {text}")
print(f"Output: {output_text}")
print(f"Steps:  {num_steps}/20")
```

---

## 🚀 Usage

### 1. Training (Multi-GPU via Accelerate)

RDT supports distributed training via **Hugging Face Accelerate** for efficient scaling.

```bash
# Configure Accelerate (one-time setup)
accelerate config

# Train RDT (Recursive Denoising Transformer)
accelerate launch rdt/scripts/train.py --config rdt/configs/rdt.yaml

# Train baselines
accelerate launch rdt/scripts/train.py --config rdt/configs/mdlm.yaml
accelerate launch rdt/scripts/train.py --config rdt/configs/cmlm.yaml

```

### 2. Inference

Run interactive or single-text inference. RDT utilizes its internal Gate MLP for adaptive stopping (`--threshold`).

```bash
# Interactive mode - RDT
rdt-inference --checkpoint checkpoints/rdt/best_model.pt --config rdt/configs/rdt.yaml --interactive

# Single text inference - RDT
rdt-inference --checkpoint checkpoints/rdt/best_model.pt --config rdt/configs/rdt.yaml \
    --text "The quick brown [MASK] jumps over the lazy [MASK]." \
    --max-steps 20 --threshold 0.05

```

### 3. Evaluation

Evaluate models on benchmarks (Accuracy, BLEU, BERTScore).

```bash
rdt-evaluate --config rdt/configs/rdt.yaml --checkpoint checkpoints/rdt/best_model.pt \
    --split test --max-steps 20 --threshold 0.05

```

---

## � Configuration Parameters

### Key RDT Parameters

The behavior of RDT can be controlled via config files (`rdt/configs/*.yaml`). Key parameters:

#### Model Architecture

- `d_model`: Hidden dimension (768 for Base, 512 for Small)
- `n_encoder_layers`: Recursive transformer depth (6 for Base)
- `input_processor_layers` / `output_processor_layers`: I/O MLP depths (2/2 for Base)
- `threshold`: Gate stopping criterion (0.02-0.05 recommended for inference)

#### Training Strategy

- `total_steps`: Total denoising steps N (20 recommended)
- `min_chain_length` / `max_chain_length`: Training segment length L (1~2 recommended)
- `loss_weight_recon`: Reconstruction loss weight (1.0)
- `loss_weight_gate`: Gate prediction loss weight (1.0)

#### Inference

- `--max-steps`: Maximum recursive iterations (20 default)
- `--threshold`: Gate score threshold for early stopping (0.02-0.05)
  - Lower = more refinement steps
  - Higher = faster inference, slightly lower quality

### Config Files Overview

| Config      | Model Type | Layers | Key Characteristics                    |
| ----------- | ---------- | ------ | -------------------------------------- |
| `rdt.yaml`  | RDT        | 2-6-2  | Recursive denoising, adaptive stopping |
| `mdlm.yaml` | MDLM       | 12     | Continuous-time diffusion, 1000 steps  |
| `cmlm.yaml` | CMLM       | 12     | Mask-predict, 10 iterations            |
| `mlm.yaml`  | MLM        | 12     | Standard BERT-style, single-pass       |

---

## �📜 Citation

If you find this code or architecture useful for your research, please cite:

```bibtex
@misc{rdt2026,
  title={RDT: Recursive Denoising Transformer via Latent Space Refinement},
  author={Yeo Joon},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/YeoJune/rdt}}
}

```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.
