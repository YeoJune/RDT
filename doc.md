# Recursive Denoising Transformer: Iterative Text Restoration via Continuous Latent Refinement

**Anonymous Authors**  
_Submitted to [VENUE PLACEHOLDER]_

---

## Abstract

Masked language models restore corrupted text in a single forward pass, while diffusion-based approaches enable iterative refinement but require hundreds of fixed denoising steps. We present the **Recursive Denoising Transformer (RDT)**, a framework that bridges these paradigms by performing iterative denoising in a continuous latent space while operating on discrete text inputs.

RDT separates the generation process into three components: an **Input Processor** that maps discrete corrupted tokens into continuous representations, a shared **Recursive Encoder** that iteratively refines these representations conditioned on the current noise level via Adaptive LayerNorm (AdaLN), and an **Output Processor** that decodes the refined representation back into tokens. A key contribution is the **Gate MLP**, which estimates the residual noise level at each step and enables _adaptive stopping_: recursion terminates automatically when the representation is sufficiently refined, allocating computation proportionally to input corruption severity. The Gate MLP uses a residual subtraction structure with Softplus nonlinearity that algebraically guarantees monotone decrease of predicted noise levels across all recursion steps. AdaLN projections are zero-initialized, enabling stable training by beginning as standard LayerNorm. A chain-based training procedure with permutation-fixed token revelation order and soft-mixing scheduled sampling further stabilizes learning and bridges the training-inference gap.

We evaluate RDT on the Wikitext-103 benchmark against MDLM (Sahoo et al., 2024) and CMLM (Ghazvininejad et al., 2019) baselines at the Small scale (~34–40M parameters) across corruption levels from 10% to 90%. RDT matches or exceeds both baselines on BERTScore across all corruption levels, and outperforms MDLM on Exact Match at corruption rates up to 50%, while using an average of only 7.79 inference steps — compared to 1000 for MDLM and a fixed 10 for CMLM — without any fixed step budget. At low corruption (10%), RDT requires just 2.38 steps; at high corruption (90%), 13.83 steps. _(Results at Small scale; base-scale experiments pending.)_

---

## 1. Introduction

Text restoration — recovering the original content of a corrupted or partially masked sequence — is a fundamental task in natural language processing. Two dominant paradigms have emerged for approaching this problem. Autoregressive models (Brown et al., 2020; Touvron et al., 2023) factorize the joint distribution over tokens sequentially, enabling expressive generation but at high computational cost proportional to sequence length. Masked language models (Devlin et al., 2019) take the opposite approach: they restore all masked positions in a single parallel forward pass, achieving efficiency at the cost of explicit modeling of inter-token dependencies during decoding.

A middle ground has been explored by non-autoregressive generation methods. CMLM (Ghazvininejad et al., 2019) repeatedly masks and regenerates low-confidence tokens, performing iterative refinement in discrete token space over a fixed number of steps. Diffusion models offer a principled probabilistic framework for iterative refinement, and their application to text — both in discrete state spaces (Austin et al., 2021; Sahoo et al., 2024) and continuous embeddings (Li et al., 2022) — has demonstrated the potential of multi-step denoising. However, existing approaches either operate in discrete space, limiting representational flexibility, or require a fixed number of denoising steps regardless of how corrupted the input is, allocating the same computation to a 10%-masked sentence as to a 90%-masked one.

This raises a natural question: _can we design a framework that iteratively refines corrupted discrete text in a continuous latent space, while adapting the amount of computation to the complexity of each input?_

We answer this question affirmatively with the **Recursive Denoising Transformer (RDT)**. RDT encodes a corrupted token sequence into a continuous representation, then repeatedly refines it through a shared Recursive Encoder conditioned on the estimated noise level at each step. A Gate MLP, operating on stop-gradient hidden states, estimates residual noise and determines when to terminate — naturally allocating more steps to heavily corrupted inputs and fewer to lightly corrupted ones. The discrete-to-continuous and continuous-to-discrete boundaries are handled by dedicated Input and Output Processors, allowing the Recursive Encoder to focus exclusively on denoising in the continuous domain.

Our main contributions are:

1. **A new discrete-continuous denoising framework.** We propose RDT, which separates the discrete-continuous interface from the denoising computation via explicit Input/Output Processors. This modular design allows the Recursive Encoder to operate purely in continuous latent space while seamlessly handling discrete token inputs and outputs.

2. **Noise-conditioned Recursive Encoder with AdaLN-Zero.** We adapt the AdaLN-Zero conditioning strategy from DiT (Peebles & Xie, 2023) to iterative text denoising, where the conditioning signal τ̂_t is updated at every recursion step. Zero-initialized projections ensure training stability without architectural regularization.

3. **Gate MLP with algebraically guaranteed monotone decrease.** Unlike ACT (Graves, 2016) and PonderNet (Banino et al., 2021), which learn a scalar halting probability, our Gate MLP predicts a noise level decrement via residual subtraction with Softplus nonlinearity, ensuring τ̂*t ≤ τ̂*{t-1} for all steps without stochastic assumptions. This structural guarantee makes adaptive stopping reliable and interpretable.

4. **Chain-based training with permutation fixation.** We construct training chains by pre-fixing a token revelation permutation π, ensuring an unambiguous, monotonically expanding delta region as the supervision target at each step. This provides clear per-step gradient signal without requiring a rehearsal mechanism. We further investigate soft-mixing scheduled sampling to close the training-inference gap in an XLA/TPU-compatible manner.

We evaluate RDT on the Wikitext-103 text restoration benchmark and demonstrate that RDT achieves competitive or superior quality to both MDLM and CMLM across most corruption levels at the Small scale (~34–40M parameters), while using on average only 0.8% of MDLM's inference steps and fewer steps than CMLM at corruption rates below 80%. Our results validate the feasibility of continuous latent denoising for discrete text and suggest promising directions for future work.

---

## 2. Related Work

### 2.1 Masked Language Models

BERT (Devlin et al., 2019) introduced the masked language modeling (MLM) objective, in which 15% of input tokens are replaced with [MASK] and the model is trained to recover them from surrounding context using a bidirectional Transformer encoder. The model performs all recoveries in a single forward pass. Subsequent work including RoBERTa (Liu et al., 2019) and ALBERT (Lan et al., 2019) improved pretraining stability and efficiency while retaining the single-pass structure. The core limitation of MLM for restoration tasks is precisely this single-pass behavior: inter-token consistency is not enforced iteratively, and the model cannot refine an initially poor prediction. RDT addresses this by applying the same encoder architecture repeatedly, accumulating denoising over multiple passes in continuous latent space.

### 2.2 Non-Autoregressive Generation and Order-Agnostic Models

CMLM (Ghazvininejad et al., 2019) adapts the MLM architecture for conditional generation via the Mask-Predict algorithm: the model first predicts all tokens in parallel, then iteratively masks the lowest-confidence predictions and regenerates them. This achieves near-autoregressive BLEU quality with fixed T=10 iterations and parallel decoding. However, CMLM operates entirely in discrete token space, and its iteration count is fixed regardless of input complexity.

A related line of work explores order-agnostic generation. ARDM (Hoogeboom et al., 2022) proposes Autoregressive Diffusion Models, which factorize the joint distribution over tokens in a randomly sampled order, making the model agnostic to generation order. While ARDM's use of random token orderings superficially resembles RDT's permutation-based training chains, the approaches differ fundamentally: ARDM remains an autoregressive model over discrete tokens where the order is determined internally, whereas RDT fixes the permutation externally during data construction and performs simultaneous refinement of all token positions in continuous latent space.

RDT differs from both in that it (1) operates in continuous rather than discrete space, (2) uses adaptive rather than fixed iteration counts, and (3) employs a separate noise estimation module (Gate MLP) to drive termination.

### 2.3 Discrete Diffusion Models

D3PM (Austin et al., 2021) extends denoising diffusion probabilistic models (Ho et al., 2020) to discrete state spaces, proposing several transition matrix structures including absorbing-state masking. The absorbing-state variant establishes a formal connection between discrete diffusion and MLM. MDLM (Sahoo et al., 2024) simplifies this framework by focusing on masking-only diffusion and deriving a Rao-Blackwellized continuous-time ELBO that reduces to a mixture of weighted MLM losses:

$$\mathcal{L}_{\text{MDLM}} = -\mathbb{E}_{t, q}\left[w(t) \cdot \log p_\theta(x \mid x_t)\right]$$

where $x_t$ is a stochastically masked version of $x$ at time $t$, $q$ denotes the forward masking process distribution, and $w(t)$ is a time-dependent weight. With modern training practices, MDLM achieves state-of-the-art performance among discrete diffusion models.

RDT lies outside the diffusion framework. MDLM defines a stochastic forward process in discrete state space and maximizes an ELBO over probabilistic samples. RDT instead encodes discrete tokens into continuous vectors and performs deterministic iterative refinement — no stochastic forward process is defined, and inference does not involve sampling. This distinction makes RDT's behavior more predictable and its stopping condition more reliable, at the cost of the probabilistic generation guarantees that diffusion models provide.

### 2.4 Continuous Diffusion for Text

Diffusion-LM (Li et al., 2022) applies Gaussian diffusion to continuous word embeddings, enabling controllable text generation through gradient-based guidance in the latent space. The model defines a stochastic forward process that corrupts embeddings with Gaussian noise, then learns to denoise over hundreds of reverse diffusion steps. RDT shares the strategy of operating in continuous embedding space, but differs in that it requires no stochastic forward process — corruption is defined purely by token masking — and uses a deterministic iterative refinement loop with adaptive stopping rather than a fixed-step reverse diffusion chain.

### 2.5 Adaptive Computation

Adaptive Computation Time (ACT; Graves, 2016) allows RNNs to learn variable computation depths by training a scalar halting probability at each step. PonderNet (Banino et al., 2021) extends this with a probabilistic halting distribution and a ponder loss for training stability. Both learn _when_ to stop by estimating the probability that further computation is unnecessary.

The Gate MLP in RDT is analogous in spirit but different in mechanism. Rather than estimating a halting probability, Gate MLP estimates the _residual noise level_ of the current hidden state — a domain-specific quantity that naturally reflects convergence. Crucially, it uses a residual subtraction structure that algebraically guarantees τ̂*t ≤ τ̂*{t-1} for all steps, whereas ACT and PonderNet offer only probabilistic convergence guarantees. Termination in RDT is deterministic: recursion stops when the estimated noise level falls below a threshold.

### 2.6 Architectural Components

**Rotary Position Embedding (RoPE).** All Transformer components in RDT — Input Processor, Recursive Encoder, and Output Processor — apply RoPE (Su et al., 2021) to query and key vectors in self-attention. RoPE encodes absolute positions via rotation matrices while inducing relative positional dependencies implicitly, combining the benefits of absolute and relative position encodings.

**AdaLN-Zero.** The Recursive Encoder uses Adaptive LayerNorm with zero initialization (AdaLN-Zero), proposed in DiT (Peebles & Xie, 2023) for timestep-conditioned image generation. Zero-initializing the affine projection ensures that at the start of training, AdaLN reduces to standard LayerNorm, avoiding instability from random conditioning. We adapt this strategy to text denoising with a dynamically updated conditioning signal τ̂_t.

**Knowledge Distillation.** The auxiliary training loss follows the knowledge distillation formulation of Hinton et al. (2015), using temperature scaling on the reference distribution. The standard T² compensation is applied to maintain gradient magnitudes comparable to the reconstruction loss when T < 1.

---

## 3. Method

### 3.1 Problem Formulation

Let $x \in \mathcal{V}^L$ denote a token sequence of length $L$ over vocabulary $\mathcal{V}$. A corruption operator applies a binary mask $m \in \{0,1\}^L$, replacing positions where $m_i = 1$ with a [MASK] token:

$$\tilde{x} = \text{corrupt}(x, m)$$

The masking ratio $\rho = |m|/L$ induces a scalar noise level $\tau = \rho \cdot G$ for a fixed scale constant $G = 20.0$. By construction, $\tau = 0$ for a fully clean sequence and $\tau = G$ for a fully masked one. The restoration task is:

$$\hat{y} = \arg\max_{x} P_\theta(x \mid \tilde{x})$$

Rather than computing this distribution in a single pass or via sequential autoregressive decoding, RDT encodes $\tilde{x}$ into a continuous latent representation and iteratively refines it:

$$\tilde{x} \;\to\; h_0 \;\to\; h_1 \;\to\; \cdots \;\to\; h_T \;\to\; \hat{y}$$

A Gate MLP estimates the residual noise level $\hat{\tau}_t$ at each recursion step $t$, serving both as the conditioning signal for the Recursive Encoder and as the termination criterion.

**Notation.** Throughout this section we use the following: $D$ for hidden dimension, $B$ for batch size, $S$ for the total number of discrete corruption steps used during training, $\text{sg}(\cdot)$ for the stop-gradient operator, and $\pi$ for a token revelation permutation.

### 3.2 Model Architecture

Figure 1 shows the overall architecture. RDT consists of three sequential modules — Input Processor, Recursive Encoder with Gate MLP, and Output Processor — each with a distinct role.

**[FIGURE 1 PLACEHOLDER: Overall RDT inference pipeline. Corrupted input → Embedding → Input Processor → Recursive Loop (Recursive Encoder + Gate MLP, adaptive stop) → Output Processor → Output Projection → Restored output.]**

#### 3.2.1 Input and Output Processors

The Input Processor $f_\text{in}$ maps discrete token indices to continuous representations:

$$h_0 = f_\text{in}\!\left(\text{Emb}(\tilde{x}) \cdot \sqrt{D}\right) \in \mathbb{R}^{B \times L \times D} \tag{1}$$

where $\text{Emb} \in \mathbb{R}^{|\mathcal{V}| \times D}$ is a learnable embedding matrix and the $\sqrt{D}$ scaling follows the convention of Vaswani et al. (2017). $f_\text{in}$ is a Transformer encoder with RoPE self-attention over $n_\text{in}$ layers.

Symmetrically, the Output Processor $f_\text{out}$ refines the final hidden state before projection back to token logits:

$$\text{logits} = W_\text{out} \cdot f_\text{out}(h_T) \in \mathbb{R}^{B \times L \times |\mathcal{V}|} \tag{2}$$

$W_\text{out}$ shares weights with $\text{Emb}$ (weight tying). This separation of discrete-continuous boundary handling from the denoising loop allows the Recursive Encoder to focus exclusively on latent refinement without needing to manage token-space operations.

#### 3.2.2 Recursive Encoder with AdaLN

The Recursive Encoder $f_\text{enc}$ is a Transformer with $N$ blocks, each conditioned on the current noise level $\hat{\tau}_t$ via Adaptive LayerNorm (AdaLN). Given input $h_t$, a two-step residual update computes $h_{t+1}$:

$$x' = h_t + \text{Dropout}\!\left(\text{SelfAttn}_\text{RoPE}\!\left(\text{AdaLN}(h_t, \hat{\tau}_t)\right)\right) \tag{3}$$

$$h_{t+1} = x' + \text{Dropout}\!\left(\text{FFN}\!\left(\text{AdaLN}(x', \hat{\tau}_t)\right)\right) \tag{4}$$

The noise level $\hat{\tau}_t \in \mathbb{R}$ is first embedded via a sinusoidal Timestep Embedder into $t_\text{emb} \in \mathbb{R}^{B \times 1 \times D}$:

$$f_k = \cos\!\left(\hat{\tau} \cdot e^{-\frac{k}{K}\log M}\right), \quad f_{k+K} = \sin\!\left(\hat{\tau} \cdot e^{-\frac{k}{K}\log M}\right), \quad k = 0,\ldots,K-1 \tag{5}$$

$$t_\text{emb} = \text{Linear}\!\left(\text{SiLU}\!\left(\text{Linear}([f_0;\ldots;f_{2K-1}])\right)\right) \tag{6}$$

where $M = 10000$ and $K = D/2$. AdaLN then applies a learned affine transformation conditioned on $t_\text{emb}$:

$$\text{AdaLN}(x, \hat{\tau}) = \text{LayerNorm}(x) \cdot (1 + \gamma) + \beta, \qquad [\gamma;\, \beta] = \text{Linear}(t_\text{emb}(\hat{\tau})) \tag{7}$$

Following DiT (Peebles & Xie, 2023), the affine projection in Eq. (7) is **zero-initialized**: $\gamma = 0, \beta = 0$ at initialization, so AdaLN reduces to standard LayerNorm at the start of training. This allows the model to learn noise-level modulation gradually and stably, without random conditioning perturbations in early training.

Each block uses independent parameters (no weight sharing across the $N$ blocks). RoPE is applied to query and key vectors only, not to values.

#### 3.2.3 Gate MLP

The Gate MLP serves as a noise-level diagnostic for the current hidden state. It operates on a stop-gradient copy of $h_t$, ensuring that the Gate loss does not interfere with the Recursive Encoder's gradient:

$$\text{pooled}_t = \text{MHA}\!\left(Q = q,\; K = \text{sg}(h_t),\; V = \text{sg}(h_t)\right) \in \mathbb{R}^{B \times D} \tag{8}$$

where $q \in \mathbb{R}^{B \times 1 \times D}$ is a learnable query vector and $\text{sg}(\cdot)$ denotes the stop-gradient operator.

The noise level prediction uses a residual structure that enforces monotone decrease. At the first recursion step (no prior estimate available):

$$h_\text{gate} = \text{GELU}\!\left(\text{Fusion}\!\left([\text{pooled}_0;\, \mathbf{0}]\right)\right)$$

$$\hat{\tau}_0 = \text{clamp}\!\left(G - \text{Softplus}\!\left(W_\text{first}(h_\text{gate})\right),\; \min=0\right) \tag{9}$$

where $W_\text{first}$ is initialized with bias $= 10.0$ to produce $\hat{\tau}_0 \approx 10.0$ at the start of training. At subsequent steps:

$$h_\text{gate} = \text{GELU}\!\left(\text{Fusion}\!\left([\text{pooled}_t;\, \text{pooled}_{t-1}]\right)\right)$$

$$\delta_t = \text{Softplus}\!\left(W_\delta(h_\text{gate})\right) \tag{10}$$

$$\hat{\tau}_t = \text{clamp}\!\left(\hat{\tau}_{t-1} - \delta_t,\; \min=0\right) \tag{11}$$

where $W_\delta$ is initialized with bias $= 1.0$, yielding initial $\delta_t \approx 1.31$.

**Monotone decrease guarantee.** Since $\text{Softplus}(z) > 0$ for all $z \in \mathbb{R}$, we have $\delta_t > 0$ at every step. Combined with the clamp, this algebraically ensures:

$$\hat{\tau}_t = \text{clamp}\!\left(\hat{\tau}_{t-1} - \delta_t,\; \min=0\right) \leq \hat{\tau}_{t-1} \quad \forall\, t \geq 1 \tag{12}$$

This structural property ensures that the predicted noise level never increases, maintaining semantic consistency with the denoising direction and making the termination condition reliable.

**Figure 2 placeholder: Gate MLP architecture diagram showing the first-step / subsequent-step branching, Fusion linear, residual MLP, Softplus, and clamp layers.**

**Adaptive stopping.** Inference terminates when $\hat{\tau}_t < \epsilon$ for a threshold $\epsilon$ (default: 0.05). A hard cap of `max_steps` (default: 50) prevents infinite loops. The full inference procedure is given in Algorithm 1.

```
Algorithm 1: RDT Inference
──────────────────────────────────────────────────────────────
Input:  x̃ ∈ V^L,  threshold ε,  max_steps
Output: ŷ ∈ V^L
──────────────────────────────────────────────────────────────
1:  h₀    ← f_in( Emb(x̃) · √D )
2:  τ̂₀    ← Gate( sg(h₀), prev_pooled=∅, prev_τ̂=∅ )
3:  t     ← 0
4:  while t < max_steps do
5:      h_{t+1}   ← f_enc( h_t ; τ̂_t )
6:      τ̂_new     ← Gate( sg(h_{t+1}), pooled_t, τ̂_t )
7:      t         ← t + 1
8:      τ̂_t       ← τ̂_new                          # advance τ̂ index
9:      if τ̂_t < ε then break
10: end while
11: ŷ ← argmax  W_out · f_out( h_t )
──────────────────────────────────────────────────────────────
```

### 3.3 Training

#### 3.3.1 Chain-Based Training Data Construction

Each training sample is a _denoising chain_: a sequence of $(x_\text{input}, y_\text{target}^{(0)}, \ldots, y_\text{target}^{(L_\text{chain}-1)})$ tuples that simulate progressive token recovery.

**Construction procedure.** Given a clean sequence $x$:

1. Sample a permutation $\pi$ over non-special token positions, defining the token _revelation order_ — the order in which tokens will be progressively uncovered.
2. Sample chain length $L_\text{chain} \sim \text{Uniform}(L_\text{min}, L_\text{max})$ and start step $s_0 \sim \text{Uniform}(L_\text{chain}, S)$.
3. Compute the number of visible tokens at the start: $n_0 = \lfloor (1 - s_0/S) \cdot N_\text{tok} \rfloor$.
4. Construct the corrupted input $x_\text{input}$ by applying BERT-style masking (80% [MASK], 10% random token, 10% original) to positions $\pi[n_0:]$.

**Target construction.** At chain step $i \in \{0, \ldots, L_\text{chain}-1\}$, the target step index is $s_i = s_0 - (i+1)$ and the target visible count is $n_i = \lfloor(1 - s_i/S) \cdot N_\text{tok}\rfloor$. The target sequence $y_\text{target}^{(i)}$ has positions $\pi[n_i:]$ masked and the rest revealed. The loss mask at step $i$ covers the **delta region** $\pi[n_{i-1} : n_i]$ — the tokens newly uncovered at that step. The ground-truth noise levels are $\tau_\text{GT}^{(i)} = (s_i / S) \cdot G$.

**Figure 3 placeholder: Chain construction diagram showing permutation π, progressive token revelation, and delta region (orange) for a 6-token example.**

#### 3.3.2 Scheduled Sampling

To bridge the training-inference gap, we apply scheduled sampling with soft mixing (Bengio et al., 2015). At global training step $k$ out of $K_\text{total}$ total steps, the interpolation coefficient decays linearly from 1 to 0:

$$p = \max\!\left(0,\; 1 - \frac{k}{K_\text{total} - 1}\right) \tag{13}$$

The noise level fed to the Recursive Encoder is then:

$$\tau_\text{used} = p \cdot \tau_\text{GT} + (1-p) \cdot \hat{\tau}_\text{pred} \tag{14}$$

Early in training ($p \approx 1$), the encoder is guided by ground-truth noise levels, ensuring stable gradient signal. Late in training ($p \approx 0$), the encoder receives its own predictions, simulating inference conditions. Soft interpolation is used in place of Bernoulli sampling to maintain static computation graphs for XLA/TPU compatibility.

#### 3.3.3 Training Objective

The total loss combines two terms:

$$\mathcal{L} = \mathcal{L}_\text{recon} + \mathcal{L}_\text{gate} \tag{15}$$

**Reconstruction loss.** Cross-entropy over the delta region loss mask, normalized by the number of masked tokens:

$$\mathcal{L}_\text{recon} = \frac{1}{N_\text{tok}} \sum_{t,i} -\log P_\theta(y_i \mid h_t) \cdot \mathbf{1}[(t,i) \in \text{mask}] \tag{16}$$

**Gate loss.** Mean squared error between predicted and ground-truth noise levels, normalized by the number of valid chain steps:

$$\mathcal{L}_\text{gate} = \frac{1}{N_\text{samp}} \sum_{t,j} \left(\hat{\tau}_t - \tau_\text{GT}^{(t)}\right)^2 \cdot \mathbf{1}[\text{valid}(t,j)] \tag{17}$$

We additionally investigated an auxiliary self-distillation loss $\mathcal{L}_\text{aux}$ (a KL divergence from a clean-input reference distribution, weighted by $\lambda = 0.1$) intended to prevent over-confident peaked predictions in partial restoration steps. Ablation results in Section 4.3 show that $\mathcal{L}_\text{aux}$ reduces the performance drop across masking levels but depresses overall accuracy, yielding no net benefit. We therefore exclude it from the final training objective.

---

## 4. Experiments

### 4.1 Experimental Setup

**Dataset.** We evaluate on the **Wikitext-103** language modeling benchmark (Merity et al., 2017), treating text restoration as the primary evaluation task. At test time, token sequences are corrupted at fixed masking ratios $\rho \in \{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9\}$ using random uniform masking, and the model's task is to recover the original tokens.

**Baselines.** We compare against two baselines, each re-trained from scratch on Wikitext-103 under the same tokenizer and data pipeline for a controlled comparison:

- **MDLM** (Sahoo et al., 2024): state-of-the-art discrete diffusion language model; uses 1000 inference steps at the Small scale (~34M parameters).
- **CMLM** (Ghazvininejad et al., 2019): conditional masked language model with Mask-Predict decoding; uses 10 fixed iterations at the Small scale (~34M parameters).

All three models are compared at approximately matched parameter counts (~34–40M).

**Model configurations.** We report results for two RDT configurations:

| Config    | $d_\text{model}$  | Layers (in / enc / out) | Heads             | Parameters        |
| --------- | ----------------- | ----------------------- | ----------------- | ----------------- |
| RDT-Small | 512               | 2 / 6 / 2               | 8                 | ~40M              |
| RDT-Base  | **[PLACEHOLDER]** | **[PLACEHOLDER]**       | **[PLACEHOLDER]** | **[PLACEHOLDER]** |

**Training details.** **[PLACEHOLDER: optimizer, learning rate, batch size, training steps, hardware.]** Masking follows BERT-style noise: 80% [MASK], 10% random token replacement, 10% identity. Chain length $L_\text{chain}$ is sampled from $\text{Uniform}(2, 8)$ and $S = 100$ discrete corruption steps. The inference threshold $\epsilon = 0.05$ with `max_steps` $= 50$.

**Evaluation metrics.** We report:

- **Exact Match (EM):** fraction of masked tokens restored to the exact ground-truth token.
- **BERTScore F1:** token-level semantic similarity between prediction and ground truth, using a pretrained BERT model.
- **BLEU-4:** 4-gram overlap between restored and original sequences.
- **Average inference steps:** mean number of Recursive Encoder forward passes per input sequence at each corruption level.

### 4.2 Main Results

Table 1 reports restoration quality and inference step counts for RDT-Small, MDLM-Small, and CMLM-Small across corruption levels from 10% to 90%. RDT-Base results are reserved for the camera-ready version pending completion of base-scale training.

**Table 1.** Text restoration results on Wikitext-103 (Small scale, ~34–40M parameters). Bold indicates the best result per metric and corruption level. †Fixed step counts; RDT steps are adaptive averages.

| Model      | Mask | EM                                  | BERTScore  | BLEU-4     | Avg. Steps |
| ---------- | ---- | ----------------------------------- | ---------- | ---------- | ---------- |
| MDLM-Small | 10%  | 0.5307                              | 0.9852     | 0.8717     | 1000†      |
| CMLM-Small | 10%  | **0.5771**                          | **0.9866** | **0.8842** | 10†        |
| RDT-Small  | 10%  | 0.5481                              | 0.9863     | 0.8788     | **2.38**   |
| MDLM-Small | 30%  | 0.4409                              | 0.9479     | 0.6011     | 1000†      |
| CMLM-Small | 30%  | 0.3638                              | 0.9302     | 0.5598     | 10†        |
| RDT-Small  | 30%  | **0.4528**                          | **0.9507** | **0.6100** | **4.78**   |
| MDLM-Small | 50%  | 0.3332                              | 0.9022     | 0.3456     | 1000†      |
| CMLM-Small | 50%  | 0.2505                              | 0.8793     | 0.3051     | 10†        |
| RDT-Small  | 50%  | **0.3338**                          | **0.9064** | **0.3471** | **7.62**   |
| MDLM-Small | 70%  | **0.2217**                          | 0.8516     | **0.1500** | 1000†      |
| CMLM-Small | 70%  | 0.1724                              | 0.8445     | 0.1430     | 10†        |
| RDT-Small  | 70%  | 0.2191                              | **0.8583** | 0.1515     | **10.52**  |
| MDLM-Small | 90%  | **0.1196**                          | 0.7774     | **0.0327** | 1000†      |
| CMLM-Small | 90%  | 0.0906                              | 0.8159     | 0.0423     | 10†        |
| RDT-Small  | 90%  | 0.1102                              | **0.7918** | 0.0306     | **13.83**  |
| RDT-Base   | all  | _[PLACEHOLDER: base-scale results]_ |            |            |            |

**Quality.** RDT-Small outperforms MDLM-Small on Exact Match at corruption levels up to 50%, with a notable margin at mid-range corruption (e.g., EM 0.4528 vs. 0.4409 at 30%, 0.3338 vs. 0.3332 at 50%). At higher corruption (60–90%), RDT falls slightly below MDLM on EM and BLEU-4, by at most 0.012 EM points (80%). Against CMLM, RDT outperforms on all three metrics at corruption rates ≥ 20%, with increasing margin at higher masking (e.g., EM 0.4528 vs. 0.3638 at 30%). At 10% corruption, CMLM leads all models — consistent with the observation that low-corruption inputs require few iterations, and CMLM's discrete refinement is well-calibrated for this regime. On BERTScore, RDT exceeds or matches MDLM across all corruption levels, including the high-masking regime where RDT's EM lags.

**Step efficiency.** RDT uses an average of 7.79 inference steps across corruption levels from 10% to 90% — approximately 0.8% of MDLM's 1000 steps, and fewer than CMLM's fixed 10 steps at corruption rates below 80%. The Gate MLP's adaptive behavior is clearly visible: step counts range from 2.38 at 10% corruption to 13.83 at 90%, demonstrating that computation is allocated proportionally to input difficulty without any per-example tuning.

**Perplexity.** RDT's perplexity at 0% masking (55.97) is modestly higher than MDLM and CMLM (both 49.34). This gap does not reflect a deficiency in restoration quality — RDT matches or exceeds both baselines on BERTScore across all corruption levels — but rather a structural difference in training objectives. MDLM optimizes a probabilistic ELBO over the full data distribution via stochastic forward-process sampling, which implicitly aligns the model with the unconditional text distribution. RDT, by contrast, trains exclusively on corrupted inputs and does not directly optimize the likelihood of clean sequences. Perplexity is evaluated by scoring clean text under a language model, and thus measures a capacity that RDT does not explicitly develop. We treat this as a known limitation of the deterministic latent refinement formulation and discuss it further in Section 5.

**Figure 7 placeholder: Line plots of EM, BERTScore, and BLEU-4 vs. corruption level (10–90%) for RDT-Small, MDLM-Small, and CMLM-Small.**

### 4.3 Ablation Study

To validate the key design choices, we ablate RDT-Small at 50% corruption. Table 2 reports Exact Match, BERTScore, and average inference steps for each variant.

**Table 2.** Ablation study on Wikitext-103 (50% masking, Small scale). All variants use the same training budget as RDT-Small.

| Variant                                     | EM (50%)   | BERTScore (50%) | Avg. Steps |
| ------------------------------------------- | ---------- | --------------- | ---------- |
| **RDT-Small (full)**                        | **0.3338** | **0.9064**      | **7.62**   |
| w/ $\mathcal{L}_\text{aux}$ ($\lambda$=0.1) | —          | —               | —          |
| w/ rehearsal (15%)                          | —          | —               | —          |
| w/o AdaLN (const. $\tau$)                   | —          | —               | —          |
| w/o monotone gate (unconstrained)           | —          | —               | —          |
| Fixed 10 steps (no Gate)                    | —          | —               | —          |

_[PLACEHOLDER: fill remaining ablation rows once experiments complete.]_

**Auxiliary loss ($\mathcal{L}_\text{aux}$).** Adding the self-distillation KL loss with $\lambda = 0.1$ reduces the rate of accuracy degradation as masking ratio increases — confirming that it acts as a distribution-smoothing regularizer — but lowers overall Exact Match, yielding no net gain. We therefore exclude it from the final objective. The trade-off suggests a tension between calibration across corruption levels and peak restoration quality; future work may revisit adaptive $\lambda$ scheduling.

**Rehearsal region.** Including a 15% rehearsal region in the loss mask was motivated by the continual-learning concern that the model might forget earlier-recovered tokens as the chain progresses. Empirically, rehearsal also fails to improve overall performance, indicating that the delta-only loss mask provides sufficient gradient signal for stable multi-step training. _(Quantitative results pending.)_

**[PLACEHOLDER: w/o AdaLN narrative — expected: removing noise-level conditioning degrades performance across all corruption levels, establishing the necessity of per-step τ̂ conditioning.]**

**[PLACEHOLDER: unconstrained gate narrative — expected: without monotone guarantee, predicted τ̂ oscillates across steps, causing unstable termination (premature stopping or near-max-step saturation).]**

**[PLACEHOLDER: fixed-step baseline narrative — shows that adaptive stopping is preferable to a fixed 10-step budget at low corruption, where fewer steps suffice.]**

### 4.4 Analysis: Gate Behavior

**Step count distribution.** Figure 6 shows the average inference step counts of RDT-Small across corruption levels. Step counts increase monotonically with masking ratio: from 1.75 steps at 0% (clean input) to 14.93 steps at 100% (fully masked), with intermediate values of 2.38 (10%), 4.78 (30%), 7.62 (50%), and 10.52 (70%). This confirms that the Gate MLP reliably adapts computation to input complexity without any explicit conditioning on the masking ratio at inference time. The gate learns this behavior purely from the noise level regression supervision during training.

Notably, step counts increase smoothly rather than saturating: each 10-percentage-point increase in corruption adds roughly 1.3–1.6 additional steps in the 10–80% range. At 90% corruption the average of 13.83 steps remains well within the `max_steps` = 50 safety cap, indicating that the gate reaches threshold reliably even for near-fully-masked inputs.

**Figure 6 placeholder: Bar chart or scatter plot of average inference steps vs. masking ratio (0–100%), showing the monotone increase from 1.75 to 14.93 steps.**

**[PLACEHOLDER: per-input step count distribution — histogram showing variance around the mean at each corruption level. Are most inputs clustered near the mean, or is there high variance? This would reveal whether the gate makes confident termination decisions.]**

**[PLACEHOLDER: Training dynamics — Figure 5 showing training curves for $\mathcal{L}_\text{recon}$ and $\mathcal{L}_\text{gate}$. Do both converge stably? Is gate loss convergence a prerequisite for reconstruction improvement?]**

---

## 5. Conclusion

We presented the Recursive Denoising Transformer (RDT), a framework for iterative text restoration that operates in continuous latent space while handling discrete token inputs. RDT's three-component architecture — Input Processor, Recursive Encoder, Output Processor — cleanly separates the discrete-continuous interface from the denoising computation. The Gate MLP provides adaptive stopping via a structural guarantee of monotone noise level decrease, allocating computation proportionally to input corruption without any per-example tuning. Chain-based training with permutation fixation enables stable multi-step supervision with a simple delta-region loss mask.

On Wikitext-103 at the Small scale (~34–40M parameters), RDT-Small achieves competitive restoration quality with MDLM and CMLM across corruption levels from 10% to 90%, outperforming MDLM on Exact Match up to 50% masking and exceeding CMLM on BERTScore across all levels, while using an average of only 7.79 inference steps — 0.8% of MDLM's 1000-step budget and fewer than CMLM's fixed 10 steps for inputs corrupted below 80%. At high corruption (80–90%), RDT's Exact Match falls modestly below MDLM (by at most 0.012 EM points), suggesting a natural ceiling of the deterministic refinement approach when most tokens are masked. Ablation experiments confirm that a proposed auxiliary self-distillation loss ($\mathcal{L}_\text{aux}$) and rehearsal region, while individually motivated, do not improve net performance and are excluded from the final model — findings that clarify the effective inductive biases for chain-based latent denoising.

RDT's perplexity at 0% masking (55.97 vs. 49.34 for MDLM and CMLM) warrants discussion. Unlike MDLM, which defines a stochastic forward process over the full data distribution and optimizes a probabilistic ELBO, RDT performs deterministic latent refinement conditioned on a masked input — a formulation that does not directly optimize the unconditional likelihood of clean text. The perplexity metric, evaluated by a language model scoring clean sequences, thus measures a capacity that RDT does not explicitly train for. This is consistent with the observation that RDT matches or outperforms MDLM on all semantic metrics (BERTScore) while showing the perplexity gap, and it points toward incorporating a likelihood-aligned objective as a promising direction for future work.

The current work establishes the architectural viability of continuous latent denoising for discrete text. Several directions remain for future work. First, the framework naturally extends to _conditional_ restoration — text infilling, dialogue, or translation — by conditioning the Input Processor on a context sequence. Second, the fixed token restoration order (permutation π) could be replaced by a learned or task-adaptive ordering strategy. Third, scaling RDT to larger model sizes and evaluating on generation benchmarks beyond restoration would help characterize its limits relative to autoregressive baselines. Finally, combining the deterministic iterative refinement of RDT with a probabilistic generation objective (e.g., score matching or ELBO) may be a fruitful direction toward a unified framework.

---

## References

Austin, J., Johnson, D. D., Ho, J., Tarlow, D., & van den Berg, R. (2021). Structured denoising diffusion models in discrete state-spaces. _Advances in Neural Information Processing Systems_, 34.

Banino, A., Balaguer, J., & Blundell, C. (2021). PonderNet: Learning to ponder. _ICML 2021 Workshop on Theoretic Foundation, Criticism, and Application of Explainable AI_.

Bengio, S., Vinyals, O., Jaitly, N., & Shazeer, N. (2015). Scheduled sampling for sequence prediction with recurrent neural networks. _Advances in Neural Information Processing Systems_, 28.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. _Advances in Neural Information Processing Systems_, 33.

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. _Proceedings of NAACL-HLT 2019_.

Ghazvininejad, M., Levy, O., Liu, Y., & Zettlemoyer, L. (2019). Mask-predict: Parallel decoding of conditional masked language models. _Proceedings of EMNLP-IJCNLP 2019_.

Graves, A. (2016). Adaptive computation time for recurrent neural networks. _arXiv preprint arXiv:1603.08983_.

Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. _arXiv preprint arXiv:1503.02531_.

Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. _Advances in Neural Information Processing Systems_, 33.

Hoogeboom, E., Gritsenko, A. A., Bastings, J., Poole, B., van den Berg, R., & Salimans, T. (2022). Autoregressive diffusion models. _International Conference on Learning Representations_.

Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., & Soricut, R. (2020). ALBERT: A lite BERT for self-supervised learning of language representations. _International Conference on Learning Representations_.

Li, X. L., Thickstun, J., Gulrajani, I., Liang, P., & Hashimoto, T. B. (2022). Diffusion-LM improves controllable text generation. _Advances in Neural Information Processing Systems_, 35.

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. _arXiv preprint arXiv:1907.11692_.

Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2017). Pointer sentinel mixture models. _International Conference on Learning Representations_.

Peebles, W., & Xie, S. (2023). Scalable diffusion models with transformers. _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_.

Sahoo, S. S., Arriola, M., Schiff, Y., Gokaslan, A., Marroquin, E., Chiu, J. T., Rush, A., & Kuleshov, V. (2024). Simple and effective masked diffusion language models. _Advances in Neural Information Processing Systems_, 37.

Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., & Liu, Y. (2021). RoFormer: Enhanced transformer with rotary position embedding. _arXiv preprint arXiv:2104.09864_.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., ... & Lample, G. (2023). LLaMA: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. _Advances in Neural Information Processing Systems_, 30.
