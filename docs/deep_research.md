# Deep Research: Architecture-Agnostic Layer-Wise Post-Training Quantization of LLMs
### BTech Final Year Project — Guneet Toppo, DTU | Session 2026–27

---

> [!NOTE]
> This document synthesizes the full technical landscape of your project. It is organized to mirror your project phases — from foundational theory through experimental design and writing. Use it as a living reference throughout your 6-month execution period.

---

## Table of Contents

1. [The "Why": Deployment Motivation](#1-the-why-deployment-motivation)
2. [Quantization Fundamentals](#2-quantization-fundamentals)
3. [Core Method Deep-Dives](#3-core-method-deep-dives)
4. [Related Work and Positioning](#4-related-work-and-positioning)
5. [Your Three Model Families](#5-your-three-model-families)
6. [Experimental Design & Controlled Study](#6-experimental-design--controlled-study)
7. [The Novel Contribution: Hessian-Diagonal Mixed-Precision](#7-the-novel-contribution-hessian-diagonal-mixed-precision)
8. [Mathematical Foundations](#8-mathematical-foundations)
9. [Implementation Roadmap (Phase 1 → 3)](#9-implementation-roadmap-phase-1--3)
10. [Evaluation Protocol](#10-evaluation-protocol)
11. [Expected Results and Failure Modes](#11-expected-results-and-failure-modes)
12. [Writing and Publication Strategy](#12-writing-and-publication-strategy)
13. [Extended Reading List (Beyond Your 14 Core Papers)](#13-extended-reading-list)
14. [Novelty Analysis Against Prior Art](#14-novelty-analysis-against-prior-art)

---

## 1. The "Why": Deployment Motivation

### The Memory Wall Problem

Every decoder-only LLM is fundamentally memory-bandwidth-bound during inference (the *decoding* phase). This is because token generation requires loading the **entire weight matrix** from HBM (High Bandwidth Memory) to GPU compute units for every single token generated.

| Model Size | FP16 Memory | 4-bit INT4 | 2-bit INT2 |
|:---|:---|:---|:---|
| 0.5B (Qwen2.5) | ~1 GB | ~250 MB | ~125 MB |
| 1.1B (TinyLlama) | ~2.2 GB | ~550 MB | ~275 MB |
| 1B (Llama-3.2) | ~2 GB | ~500 MB | ~250 MB |
| 7B | ~14 GB | ~3.5 GB | ~1.75 GB |
| 70B | ~140 GB | ~35 GB | ~17.5 GB |

### The Roofline Model Argument

The arithmetic intensity of autoregressive decoding is ~1–2 FLOPs per byte — far below the ridge point of any modern GPU. This means:

- **Prefill (prompt processing):** Compute-bound — you're doing large matrix × matrix multiplications.
- **Decode (token generation):** Memory-bound — you're doing matrix × *vector* (a single token's KV state).

**Weight quantization directly attacks the memory bottleneck.** At 4-bit, the weight data volume drops 4×, and on memory-bound workloads, this translates almost linearly to speed. At 2-bit, an 8× reduction is theoretically possible, though accuracy degradation is the challenge your project addresses.

> [!IMPORTANT]
> This is the **core justification** for your project. Frame every motivation slide and section-1 paragraph around: *inference is memory-bandwidth bound → weight compression = direct latency reduction → PTQ is the only training-free path.*

---

## 2. Quantization Fundamentals

### 2.1 Affine Round-to-Nearest (RTN) — Your Baseline

For a weight tensor `W` with range `[w_min, w_max]`:

```
s = (w_max - w_min) / (2^b - 1)    # scale
z = round(-w_min / s)               # zero-point (for asymmetric)
W_q = clamp(round(W / s) + z, 0, 2^b - 1)
W̃   = s * (W_q - z)                 # dequantized
```

**RTN at 4-bit:** Works well because most LLM weights have roughly Gaussian distributions — most values land near zero, and 16 levels cover the range adequately.

**RTN at 2-bit:** Only 4 levels. The quantization grid error `E[δw²] ≈ s²/12` becomes enormous relative to weight magnitudes. This is why naive RTN collapses at 2-bit.

**Key invariants for RTN:**
- Zero-shot, no calibration data needed
- Per-channel quantization (`s` and `z` computed per output channel of `W`) dramatically outperforms per-tensor
- Group quantization (groups of 64–128 weights share a scale) recovers significant quality at the cost of ~1–2% overhead in model size

### 2.2 Quantization Granularity Hierarchy

```
Per-tensor  →  Per-channel  →  Per-group  →  Per-weight (unstructured)
  coarsest          ↑ your baseline               finest
                    (one scale per             (not hardware friendly)
                    output channel)
```

Your project uses **per-channel** as the standard granularity. This is important: GPTQ-style methods operate at the *column* level of `W`, which corresponds to per-input-channel Hessian rows.

---

## 3. Core Method Deep-Dives

### 3.1 GPTQ — The Workhorse [1]

**Paper:** Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers," ICLR 2023. arXiv:2210.17323.

**Core Insight:** OBQ (Optimal Brain Quantization, Frantar & Alistarh, NeurIPS 2022) showed that for a single linear layer `y = Wx`, the optimal quantization error-compensation update when quantizing weight `w_q` is:

```
δW = -E[w_q] * (H⁻¹e_q) / (H⁻¹)_qq * e_q^T
```

where `H = 2 X X^T` is the Hessian proxy (Fisher information of the layer reconstruction loss) and `e_q` is the basis vector for the quantized column `q`.

**GPTQ's three key innovations over OBQ:**

1. **Arbitrary order:** Instead of the OBQ greedy "easiest first" ordering, quantize weights in the **same left-to-right column order** for all rows. This allows vectorization across rows.

2. **Lazy batch updates:** Accumulate Hessian update contributions for blocks of 128 columns before applying them, trading memory for compute efficiency.

3. **Cholesky reformulation:** Computes `H⁻¹` column by column via a numerically stable Cholesky decomposition, avoiding the O(n³) full inversion. This is the key to scaling to billion-parameter models.

**What this gives you practically:**
- At 4-bit: near-lossless compression (perplexity increase < 0.5 points on WikiText-2 for 7B models)
- At 3-bit: usable, <2 perplexity point increase
- At 2-bit: significant degradation without further tricks (where Hadamard helps)

**Implementation considerations for Phase 1:**
```python
# Pseudocode for GPTQ-style column-wise quantization
H = 2 * X.T @ X  # [d_in × d_in] Hessian proxy
H_inv = cholesky_inverse(H + dampening * I)

for j in range(d_in):  # left to right
    w_q = quantize(W[:, j], scale_j, zero_j)  # quantize column j
    err = W[:, j] - w_q  # quantization error
    W[:, j+1:] -= err.outer(H_inv[j, j+1:]) / H_inv[j, j]  # compensate
```

### 3.2 Hadamard Incoherence Processing — The 2-bit Enabler [4, 5]

**Papers:** QuIP (Chee et al., NeurIPS 2023) and QuIP# (Tseng et al., ICML 2024).

**The Problem:** LLM weight columns are *coherent* — some columns have systematically larger magnitude than others (these are the ~1% "salient" columns that AWQ identifies). This means the quantization error is highly non-uniform: high-magnitude columns get mangled.

**The Solution — Incoherence Processing:**

Apply a random Hadamard rotation `H` to both weights and activations:
```
W' = W H^T    (right-multiply weights)
X' = H X      (left-multiply activations — happens at runtime)
```

The output `y = W X = (W H^T)(H X) = W' X'` is mathematically identical.

**Why this works:** A random Hadamard rotation approximately *equalizes* the energy across columns. The Hadamard-Walsh transform spreads the large-magnitude outlier columns across all columns. After rotation, `‖W'[:, j]‖` is approximately the same for all `j`, making quantization error uniform.

**Mathematical guarantee (the incoherence concentration bound):**
The maximum column magnitude ratio before rotation can be O(√n); after Randomized Hadamard Transform (RHT):
```
E[max_j ‖W'[:, j]‖²] ≈ ‖W‖²_F / d_in   (approximately uniform)
```
The concentration is exponentially tight — this is essentially a Johnson-Lindenstrauss-type result.

**Runtime cost:** The Hadamard rotation is O(n log n) — negligible compared to matrix multiply. QuIP# stores the rotation matrix implicitly and applies it as a fused CUDA kernel at inference time.

**What your project does:** You implement the CPU/NumPy version of the RHT and validate that it reproduces the ~13% additional error reduction at 2-bit that QuIP# reports on real models.

### 3.3 LLM.int8() — The Activation Outlier Baseline [2]

**Paper:** Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale," NeurIPS 2022. arXiv:2208.07339.

**Key Finding:** At ~6.7B+ parameters, LLMs develop "emergent" activation outliers — specific hidden dimensions consistently produce values 10–100× larger than the rest. These outliers appear in <0.01% of dimensions but cause catastrophic quantization error if treated normally.

**The Decomposition:**
```
Y = W_fp16 X_outlier + W_int8 X_normal
```

Keep the outlier columns of `X` and corresponding rows of `W` in FP16; quantize the rest to INT8.

**Why this matters for your project:** You're doing *weight-only* quantization (activations stay in FP16), so activation outliers are a less critical problem. However, *the weight columns corresponding to outlier activation dimensions are exactly the ~1% salient columns that GPTQ and AWQ try to handle carefully*. Your Hessian-diagonal sensitivity score `ŝ_j = H_jj · Var(W_{:,j})` should rank these columns high automatically — this is a testable hypothesis.

### 3.4 AWQ — Activation-Aware Salience [3]

**Paper:** Lin et al., "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration," MLSys 2024 Best Paper. arXiv:2306.00978.

**Key Finding:** 1% of weight channels are "salient" — their perturbation causes disproportionate output error. Salience is identified by the activation scale `s_x` of the corresponding input dimension (high activation magnitude = high sensitivity).

**The Scale Search:** AWQ multiplies salient weight channels by `s > 1` (increasing their effective quantization resolution) and divides activations by `s` (absorbed into the previous layer's normalization). This is a *per-channel scale search* that minimizes reconstruction error.

**Relation to your project:** AWQ uses activation magnitudes as the sensitivity proxy. Your project uses `H_jj · Var(W_{:,j})` as the sensitivity proxy. These are related but different:
- AWQ: `sensitivity(j) ∝ E[|X_j|]` (activation scale)
- Your Hessian proxy: `sensitivity(j) ∝ H_jj` where `H_jj = 2 Σ X_ij²` (proportional to activation *second moment*)

Your formula is more rigorous (second-order Taylor expansion of loss), while AWQ is a practical heuristic. This is a key differentiator to articulate in your paper.

---

## 4. Related Work and Positioning

### 4.1 The Full Landscape Map

```
                    QUANTIZATION METHODS
                           │
          ┌────────────────┼────────────────────┐
     Scalar/Uniform      Mixed-Prec        Vector/Codebook
          │                  │                  │
    RTN, GPTQ           AWQ, SqueezeLLM      QuIP#, AQLM,
    (your baselines)    LLM.int8()           VPTQ (SOTA 2-bit)
          │                  │
     ┌────┤           ┌──────┤
  per-tensor  per-channel  per-layer  per-channel
  (weak)     (your baseline) (HAWQ)   (YOUR NOVEL WORK)
```

### 4.2 Paper-by-Paper Comparison Table

| Method | Venue | Bits | Granularity | Sensitivity Metric | Architecture Coverage | Your Gap |
|:---|:---|:---|:---|:---|:---|:---|
| OBQ [pre-GPTQ] | NeurIPS'22 | 3-4 | Per-weight | Full Hessian row | Single model | Baseline |
| **GPTQ [1]** | ICLR'23 | 3-4 | Per-column | Hessian diagonal approx | Llama-centric | You implement + cross-test |
| **LLM.int8() [2]** | NeurIPS'22 | 8+16 | Per-tensor | Activation outliers | Multi-family | Weight-only baseline |
| **AWQ [3]** | MLSys'24 | 4 | Per-channel | Activation scale | Llama-centric | Heuristic vs. your 2nd-order |
| **QuIP [4]** | NeurIPS'23 | 2 | Per-matrix | Incoherence | Llama | You implement Hadamard component |
| **QuIP# [5]** | ICML'24 | 2 | Per-matrix + E8 | Incoherence + lattice | Llama | You test integer-only (no lattice) |
| **HAWQ-V2 [6]** | NeurIPS'20 | Mixed | Per-layer | Hessian trace | CNNs | Layer → **channel** (your upgrade) |
| **HAWQ-V3 [7]** | ICML'21 | Mixed | Per-layer | Hessian trace + ILP | CNNs | Closed-form vs. ILP (your upgrade) |
| **Q-BERT [8]** | AAAI'20 | 2-3 | Per-layer | Hessian | BERT only | Encoder-only, layer granularity |
| **BitMoD [9]** | IEEE TC'26 | Mixed | Sub-layer | Datatype sensitivity | Single ASIC | Hardware-coupled |
| **Cho et al. [10]** | IEEE Access'25 | Mixed | Per-group | Power-of-two | Single family | |
| SqueezeLLM | ICML'23 | 3 | Sparse+dense | Hessian diagonal | Llama | Sparse format vs. your integer |
| SpQR | ICML'23 | ~4eff | Sparse outlier | Row/col norms | Llama | |
| AQLM | ICML'24 | 2 | Codebook | Second-order | Llama | Vector quant vs. your scalar |
| SmoothQuant | ICML'23 | 8 | Per-channel | Activation scale | Multi | W+A vs. your weight-only |

### 4.3 Your Exact Research Gap (Sharpen This)

**No existing work combines ALL of:**
1. Channel-granularity bit allocation (not layer-granularity like HAWQ)
2. A closed-form O(mn) Hessian-diagonal sensitivity statistic `ŝ_j = H_jj · Var(W_{:,j})` (no eigendecomposition, no ILP solver)
3. Software-only {2, 4}-bit integer formats (no hardware co-design, no lattice codebooks)
4. A controlled equal-average-bit comparison across **multiple architecturally distinct model families**

Points 1–3 together form your technical contribution. Point 4 is your empirical contribution. Together, they produce a publishable paper.

---

## 5. Your Three Model Families

### 5.1 Why These Three?

| Property | TinyLlama-1.1B | Llama-3.2-1B | Qwen2.5-0.5B |
|:---|:---|:---|:---|
| Architecture base | Llama-2 style | Llama-3 style | Alibaba/Qwen |
| GQA | No (MHA) | Yes | Yes |
| Attention heads | 32 (KV=32) | 32 (KV=8) | 14 (KV=2) |
| FFN activation | SiLU (SwiGLU) | SiLU (SwiGLU) | SiLU (SwiGLU) |
| Norm | RMSNorm | RMSNorm | RMSNorm |
| Positional encoding | RoPE | RoPE | RoPE |
| Tie embeddings | Yes | No | Yes |
| Hidden size | 2048 | 2048 | 896 |
| Intermediate size | 5632 | 8192 | 4864 |
| # Layers | 22 | 16 | 24 |
| Parameters | 1.1B | 1.24B | 0.5B |
| License | Apache 2.0 | Llama Community | Apache 2.0 |

### 5.2 Key Architectural Divergences (This is what makes your cross-architecture study non-trivial)

**Attention type:**
- **TinyLlama:** Full multi-head attention (MHA) → 32 KV heads → large KV weight matrices
- **Llama-3.2:** Grouped-query attention (GQA, 4:1 ratio) → fewer KV params → different weight sensitivity profile
- **Qwen2.5:** Aggressive GQA (7:1 ratio) → very few KV params → extreme concentration of attention load in Q/K projections

**Implication for quantization:** The sensitivity distribution of `{q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj}` will differ substantially. Your experiment will measure whether your Hessian-diagonal metric correctly identifies these differences **without architecture-specific code**.

**FFN size:**
- Qwen2.5-0.5B has a *relatively large* intermediate size (4864) for its hidden size (896), implying a high FFN-to-attention parameter ratio. This could mean FFN layers are more resilient to quantization (more redundancy) while attention is more sensitive.

### 5.3 The "Architecture-Agnostic" Claim — How to Justify It

Your claim is that `(W, X)` is sufficient — you don't need to know whether `W` is `q_proj` or `gate_proj`. You justify this via:

1. **HuggingFace nn.Linear hooks:** `model.named_modules()` yields every `nn.Linear` regardless of its role. Your hook captures input activations `X` automatically.

2. **Identical calibration forward pass:** 128 sequences of 2048 WikiText-2 tokens, run once per model, capture all `X` tensors via hooks.

3. **The same GPTQ/Hadamard code path applies to every `nn.Linear`:** No if-statements for layer type.

**You must show in your paper:** The same code, unmodified, was applied to all three model families. Show a code snippet of the hook registration loop.

---

## 6. Experimental Design & Controlled Study

### 6.1 The Ablation Grid

Your core table is a 3×3×3 design:

```
Methods:      {RTN, GPTQ, GPTQ+Hadamard}
Bits:         {2, 3, 4}
Families:     {Qwen2.5-0.5B, Llama-3.2-1B, TinyLlama-1.1B}
```

This gives **27 quantization runs** + 3 FP16 baselines = 30 model evaluations.

**Each evaluation produces:**
- WikiText-2 perplexity (primary metric)
- ARC-Easy accuracy (zero-shot)
- HellaSwag accuracy (zero-shot)
- WinoGrande accuracy (zero-shot)

**Total result cells: 30 × 4 = 120 data points**

This is the *exact table that doesn't exist in the literature* under a single controlled protocol.

### 6.2 Calibration Protocol (Critical for Reproducibility)

```python
# Exact protocol to document in your paper
CALIBRATION_SEQLEN = 2048
CALIBRATION_NSAMPLES = 128

# WikiText-2 train split, tokenized, no padding
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokens = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")["input_ids"]

# Sample 128 contiguous chunks
samples = []
for i in range(CALIBRATION_NSAMPLES):
    start = random.randint(0, tokens.shape[1] - CALIBRATION_SEQLEN)
    samples.append(tokens[:, start:start + CALIBRATION_SEQLEN])
```

**Why this matters:** Different calibration choices (C4, The Pile, Alpaca instructions) affect GPTQ results by 0.1–0.5 perplexity points. By fixing WikiText-2 train, you ensure your 27-run table is internally comparable.

### 6.3 Compute Budget on Kaggle T4

| Model | Calibration (FP16 forward) | GPTQ per layer | # Layers | Total GPTQ time |
|:---|:---|:---|:---|:---|
| Qwen2.5-0.5B | ~2 min | ~5 sec | 24 × 7 linears | ~15 min |
| Llama-3.2-1B | ~3 min | ~8 sec | 16 × 7 linears | ~15 min |
| TinyLlama-1.1B | ~4 min | ~10 sec | 22 × 7 linears | ~25 min |

**Per run: ~30–45 minutes on T4.** 27 runs = ~15–20 GPU-hours. Well within Kaggle's 30 hr/week limit.

**Memory:** T4 has 16GB. FP16 TinyLlama-1.1B ≈ 2.2GB. Calibration activations (128 × 2048 × 2048 dims) ≈ ~2GB in FP16. Total ≈ 4–5GB — comfortable.

### 6.4 What "Transfer" Means in Your Study

The key question your ablation answers:

> "Does adding Hadamard incoherence processing (going from GPTQ to GPTQ+H) help equally across all three architectures, or does it provide architecture-specific benefit?"

**Hypothesis from theory:** The benefit of Hadamard should be related to the *kurtosis* of the weight distribution after GPTQ compensation. Architectures with more persistent outliers (heavier tails) should benefit more.

**How to test:** After your grid, compute the *relative gain* of GPTQ+H vs GPTQ at each bit width for each family. If gains are similar, the method transfers; if gains differ, you have evidence of architecture-specific behavior.

---

## 7. The Novel Contribution: Hessian-Diagonal Mixed-Precision

### 7.1 The Sensitivity Score

For a weight matrix `W ∈ R^{d_out × d_in}` with Hessian proxy `H = 2XX^T`:

```
ŝ_j = H_jj · Var(W_{:,j})   for j = 1, ..., d_in
```

Where:
- `H_jj = 2 Σ_s X_{s,j}²` — the j-th diagonal element, computable in O(mn) from calibration data (m samples, n tokens)
- `Var(W_{:,j}) = (1/d_out) Σ_i (W_{ij} - W̄_{:,j})²` — variance of column j

**Why this particular formula?**

The second-order Taylor expansion of the layer reconstruction loss after quantizing column j by error `δw_{:,j}`:

```
ΔL ≈ δw_{:,j}^T · H_{jj} · δw_{:,j}   (diagonal approximation)
```

For uniform b-bit quantization, the expected squared error per weight element is `σ²_q ≈ s²_j/12` where `s_j` (scale) is proportional to the weight range. The weight range is dominated by its standard deviation `σ_j = √Var(W_{:,j})`.

So: `E[ΔL_j] ∝ H_jj · σ²_j = H_jj · Var(W_{:,j})` ← this is `ŝ_j`.

### 7.2 The Allocation Rule (Reverse Water-Filling)

**Problem:** Given budget `b̄ = 3` bits average, assign `b_j ∈ {2, 4}` to each column j.

**Observation:** This is a 0-1 knapsack variant. For your binary {2,4}-bit case, the optimal greedy rule is:

> Assign 4-bit to the `t` columns with highest `ŝ_j`, 2-bit to the rest.

**Why "reverse water-filling"?** Standard water-filling allocates *more* bits to *stronger* channels (higher signal = more worth preserving). Here, sensitivity `ŝ_j` plays the role of "channel importance" — columns with high Hessian-weighted variance need more bits to represent accurately. You're filling the "wells" (low-sensitivity channels get fewer bits, high-sensitivity get more).

**Choosing t (the budget split):**
At equal average bits `b̄ = 3`, with `d_in` total channels:
```
t · 4 + (d_in - t) · 2 = 3 · d_in
→ t = d_in / 2   (exactly half get 4-bit, half get 2-bit)
```

**Sweep:** You'll vary t from 10% to 90% (changing average bits from 2.2 to 3.8) to trace the accuracy-vs-bits curve.

### 7.3 Baseline Comparison

| Configuration | Average bits | Method |
|:---|:---|:---|
| All 2-bit | 2.0 | RTN / GPTQ uniform |
| Mixed (top-50% → 4-bit) | 3.0 | **Your method** |
| All 3-bit | 3.0 | RTN / GPTQ uniform |
| All 4-bit | 4.0 | RTN / GPTQ uniform |

**The key comparison:** Your mixed {2,4}-bit at 3.0 bits average vs. uniform 3-bit RTN/GPTQ at 3.0 bits average.

**Expected result from theory:** Mixed-precision should outperform uniform 3-bit, because:
- The 4-bit high-sensitivity channels have much lower quantization error than if they were 3-bit
- The 2-bit low-sensitivity channels are over-quantized but this contributes little to the total loss
- Net: lower total reconstruction error than uniform 3-bit

### 7.4 Connection to Rate-Distortion Theory

Your allocation is the solution to:

```
minimize    Σ_j H_jj · σ²_qj(b_j)
subject to  (1/d_in) Σ_j b_j = b̄
            b_j ∈ {2, 3, 4}
```

Where `σ²_qj(b) ≈ Var(W_{:,j}) / (3 · 4^b)` (quantization error as function of bits and variance).

This is a discrete rate-distortion optimization. For Gaussian sources, the continuous relaxation has the water-filling solution:

```
b_j* = (1/2) log₂(H_jj · Var(W_{:,j}) / λ)   for some λ
```

Your binary {2,4} assignment is the nearest practical integer solution to this continuous optimum. **This is the theoretical backing for your allocation rule** — cite Shannon (1948) and Berger (1971) here.

---

## 8. Mathematical Foundations

### 8.1 The Grid Error Lemma (Phase-0 Verified)

For uniform b-bit affine quantization with scale `s`:
```
E[δw²] = s²/12 = Δ²/12   (Δ = quantization step size)
```

For a weight column of range `R` (approximately `4σ_j` for Gaussian):
```
s = R / (2^b - 1) ≈ 4σ_j / (2^b - 1)
E[ΔL_j] ≈ (d_out/12) · H_jj · s² ≈ (4/3) · d_out · H_jj · σ²_j / (2^b - 1)²
```

This shows the sensitivity scales as `H_jj · σ²_j / (2^b - 1)²`, confirming your score `ŝ_j`.

### 8.2 The Exactness of GPTQ's Second-Order Step

For a single column quantization: the GPTQ update `W_{:, j+1:} -= err · H⁻¹[j, j+1:] / H⁻¹[j,j]` is the exact minimizer of the residual reconstruction error under the quadratic loss approximation. This is proven via the Schur complement of `H⁻¹`.

**Key implication:** The GPTQ error is bounded by the *approximation* in the second-order Taylor expansion, not by the algorithm. At 4-bit, the Taylor approximation is tight (small perturbations). At 2-bit, it breaks down — which is why Hadamard is needed.

### 8.3 Rotation Invariance

The key property enabling Hadamard incoherence: for a random orthogonal matrix `Q`:
```
‖WX - W_q X‖² = ‖WQQ^T X - W_q QQ^T X‖² = ‖(WQ)(Q^T X) - (W_q Q)(Q^T X)‖²
```

The quantization error is identical whether you quantize `W` directly or quantize `WQ` and rotate activations by `Q^T`. But `WQ` has better statistical properties (more uniform column norms) → lower quantization error.

### 8.4 Cholesky Stability in GPTQ

**Why Cholesky instead of direct inversion?**

`H = 2XX^T` is positive semi-definite (PSD). For layers with `d_in > mn` (more input dimensions than calibration tokens), `H` is rank-deficient. The Cholesky decomposition:
1. Adds diagonal dampening: `H ← H + ε·I` to ensure PD
2. Decomposes: `H = L L^T`
3. Back-solves: avoids numerical instability from near-singular matrices

In practice: `ε = 0.01 · mean(diag(H))` is the standard dampening used by GPTQ.

---

## 9. Implementation Roadmap (Phase 1 → 3)

### 9.1 Phase 1: PyTorch Port and Full Ablation Grid (Oct–Dec 2026)

**Step 1: Hook-based calibration data collection**
```python
import torch
from collections import defaultdict

calibration_inputs = defaultdict(list)

def make_hook(name):
    def hook(module, input, output):
        calibration_inputs[name].append(input[0].detach().cpu())
    return hook

handles = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        handles.append(module.register_forward_hook(make_hook(name)))

# Run forward pass on calibration data
with torch.no_grad():
    for batch in calibration_batches:
        model(batch)

# Remove hooks
for h in handles: h.remove()
```

**Step 2: Layer-wise quantization loop**
```python
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        X = torch.cat(calibration_inputs[name], dim=0)  # [N, d_in]
        W = module.weight.data  # [d_out, d_in]
        
        if method == 'RTN':
            module.weight.data = rtn_quantize(W, bits)
        elif method == 'GPTQ':
            module.weight.data = gptq_quantize(W, X, bits)
        elif method == 'GPTQ+H':
            module.weight.data = gptq_hadamard_quantize(W, X, bits)
```

**Step 3: Evaluation via lm-evaluation-harness**
```bash
# After quantization, save model and evaluate
lm_eval --model hf \
    --model_args pretrained=./quantized_model \
    --tasks wikitext,arc_easy,hellaswag,winogrande \
    --device cuda:0 \
    --batch_size 8
```

**Key engineering considerations:**
- Store calibration activations on CPU (not GPU) to avoid OOM
- Process layers sequentially, loading each to GPU only when needed
- For TinyLlama with 22 layers × 7 linears = 154 modules, ensure your loop is correct
- Validate each quantized layer's output matches unquantized (within expected error bounds) before moving to next

### 9.2 Phase 2: Mixed-Precision Allocation (Jan–Feb 2027)

**Algorithm:**
```python
def hessian_diagonal_allocation(W, X, target_bits_avg=3.0, low_bits=2, high_bits=4):
    # 1. Compute Hessian diagonal
    H = 2 * X.T @ X  # [d_in × d_in]
    H_diag = torch.diag(H)  # [d_in]
    
    # 2. Compute sensitivity score
    col_var = W.var(dim=0)  # [d_in] — variance of each weight column
    sensitivity = H_diag * col_var  # ŝ_j
    
    # 3. Determine t (number of high-precision channels)
    d_in = W.shape[1]
    t = int(d_in * (target_bits_avg - low_bits) / (high_bits - low_bits))
    
    # 4. Rank channels
    high_precision_cols = torch.argsort(sensitivity, descending=True)[:t]
    low_precision_cols = torch.argsort(sensitivity, descending=True)[t:]
    
    # 5. Quantize with different bits
    W_q = W.clone()
    W_q[:, high_precision_cols] = gptq_quantize_cols(W, X, high_precision_cols, high_bits)
    W_q[:, low_precision_cols] = gptq_quantize_cols(W, X, low_precision_cols, low_bits)
    
    return W_q
```

**The sweep:** Vary `t` from `0.1 × d_in` to `0.9 × d_in` in 9 steps. Plot perplexity vs. average bits. Compare to the uniform-bits GPTQ curve at the same average bit widths.

### 9.3 Phase 3: Analysis and Paper Writing (Mar–May 2027)

**Key figures to generate:**
1. The 3×3×3 ablation table (main result)
2. Mixed-precision accuracy vs. average bits curve (per model family)
3. Sensitivity score distribution visualization (histogram of ŝ_j across layers)
4. Error reduction waterfall: RTN → GPTQ → GPTQ+H at 2-bit

---

## 10. Evaluation Protocol

### 10.1 WikiText-2 Perplexity (Primary Metric)

```python
# Evaluation on WikiText-2 test split
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# Concatenate, tokenize, split into 2048-token sequences
# Compute perplexity = exp(-mean(log P(token | context)))
```

**Interpretation:**
- FP16 baseline for 1B models: typically 6–9 PPL
- 4-bit RTN: +0.3–1.0 PPL degradation (acceptable)
- 4-bit GPTQ: +0.1–0.3 PPL (near-lossless)
- 3-bit GPTQ: +0.5–2.0 PPL (usable)
- 2-bit RTN: often catastrophic collapse (PPL > 100)
- 2-bit GPTQ+H: +5–20 PPL (large but structured)

### 10.2 Zero-Shot Task Benchmarks

**ARC-Easy:** 25-shot (or 0-shot) multiple choice science questions. Metric: normalized accuracy.

**HellaSwag:** 10-shot completion of incomplete sentences. Metric: accuracy. Sensitive to language modeling quality.

**WinoGrande:** 5-shot commonsense pronoun resolution. Metric: accuracy. Robust to memorization.

**Why these three?** They test complementary capabilities: factual knowledge (ARC), language coherence (HellaSwag), and reasoning (WinoGrande). Together they give a profile of model quality that PPL alone misses.

### 10.3 The lm-evaluation-harness (EleutherAI)

```bash
pip install lm_eval[hf]

lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B,dtype=float16 \
    --tasks arc_easy,hellaswag,winogrande \
    --num_fewshot 0 \
    --device cuda:0 \
    --output_path ./results/llama32_fp16.json
```

**Version note:** Pin to `lm_eval==0.4.x` for reproducibility. Major benchmark numbers change between versions.

---

## 11. Expected Results and Failure Modes

### 11.1 Expected Trends

**4-bit results (all methods):**
- RTN: slight degradation (0.3–1.0 PPL) — architecture-independent
- GPTQ: near-lossless — minor variation across families
- GPTQ+H: similar to GPTQ (rotation barely helps at 4-bit because outlier columns aren't severe)

**Hypothesis:** At 4-bit, Hadamard should provide minimal benefit. This *validates* the theory (incoherence processing is motivated by 2-bit behavior).

**3-bit results:**
- RTN: noticeable degradation (1.0–3.0 PPL), family-dependent
- GPTQ: good recovery (0.5–1.5 PPL over baseline)
- GPTQ+H: marginal improvement over GPTQ

**2-bit results — the most interesting:**
- RTN: catastrophic (PPL > 50–100 for all families)
- GPTQ: large error but structured (~10–30 PPL over baseline)
- GPTQ+H: ~13% error reduction over GPTQ (matching your Phase-0 synthetic validation)

**Cross-architecture transfer hypothesis:** 
The relative ordering of method quality should be preserved across architectures, but the absolute magnitude of degradation may differ based on:
- Weight distribution kurtosis (heavier tails → more outliers → more Hadamard benefit)
- GQA ratio (Qwen2.5's aggressive GQA may concentrate sensitive weights in Q projections)
- Hidden dimension size (smaller hidden dim → fewer parameters per layer → more sensitivity)

### 11.2 Failure Modes to Anticipate

**Numerical instability in Cholesky:**
- Symptom: `RuntimeError: linalg.cholesky: The factorization could not be completed because the input is not positive-definite`
- Fix: Increase dampening factor `ε`; try `ε = 0.1 × mean(diag(H))`

**OOM on T4 (16GB):**
- Symptom: `CUDA out of memory`
- Fix: Process calibration data in smaller sub-batches; use `torch.no_grad()` everywhere; offload activations to CPU between layers

**PPL collapse at 2-bit for one specific model:**
- Might indicate a layer with extreme outliers (could be embedding layer or lm_head)
- Fix: Skip quantization of first/last layers (standard practice)
- Note this as an interesting finding — different architectures may have different "fragile" layers

**Hadamard doesn't help at 2-bit:**
- This would be a surprising result — investigate whether your implementation applies the random rotation (use a fixed random seed for reproducibility!)
- Check: does `W' = W @ H.T` have lower column norm variance than `W`?

---

## 12. Writing and Publication Strategy

### 12.1 Paper Structure (Target: 8 pages + references)

**Title:** *"An Empirical Study of Architecture-Agnostic Components in Layer-Wise LLM Quantization"*

**Abstract (4 sentences):**
1. Problem + motivation (memory cost of LLMs)
2. What you did (unified framework + cross-family study)
3. Key finding (which components transfer, which don't)
4. Novel contribution (Hessian-diagonal allocation beats uniform at equal budget)

**Section 1: Introduction**
- Memory-bandwidth motivation (roofline argument)
- The two open questions from your synopsis
- Paper contributions bullet list (4 bullets)

**Section 2: Background**
- RTN, GPTQ, Hadamard in 1–2 paragraphs each
- Brief rate-distortion setup for water-filling

**Section 3: Framework**
- The nn.Linear hook-based quantization framework
- Code-level evidence of architecture-agnosticism

**Section 4: Controlled Experimental Study**
- Calibration protocol
- Model families (table with arch details)
- Ablation grid results table
- Analysis: which components transfer?

**Section 5: Hessian-Diagonal Mixed-Precision**
- Sensitivity score derivation
- Allocation algorithm
- Results: mixed vs. uniform at equal average bits

**Section 6: Related Work** (or fold into intro)

**Section 7: Conclusion**

### 12.2 Venue Strategy

| Venue | Deadline (approx) | Fit |
|:---|:---|:---|
| arXiv preprint | Anytime (Apr 2027) | Primary output |
| EMNLP Findings | Jun 2027 | NLP venue, efficiency track |
| ACL SRW (Student Research Workshop) | ~Feb 2027 | Perfect for BTech thesis |
| ICLR 2028 | Oct 2027 | Ambitious but aim high |
| ML for Systems (co-located ISCA/MICRO) | ~Mar 2027 | Systems angle |
| AAAI Student Program | Sep 2027 | Accessible |

**Recommendation:** Target **arXiv in April 2027** (right after Phase-3 analysis). Submit to **ACL Student Research Workshop 2027** (student-specific, reviewed but accessible). If results are strong, revise for **EMNLP 2027 Findings**.

### 12.3 Writing Tips for Technical ML Papers

1. **Lead with numbers, not prose:** "GPTQ+H reduces 2-bit perplexity by 13–18% over GPTQ across all three architectures" beats "we show that Hadamard helps."

2. **The ablation table is your paper's spine:** Make it perfectly formatted, with ±std if you run multiple seeds.

3. **Reproducibility statement:** Include exact HuggingFace model IDs, calibration seed, lm-eval version.

4. **Limitations section (important for credibility):** You didn't test >1B models (except optional 7B); you used scalar (not vector) quantization; your integer kernels don't include dequantization overhead measurements.

5. **LaTeX template:** Use the ACL 2023 style file for the report; switch to NeurIPS/ICLR template if targeting those venues.

---

## 13. Extended Reading List

### Beyond Your 14 Core Papers

| Priority | Paper | Why Read It |
|:---|:---|:---|
| **Must-Read** | SmoothQuant (Xiao et al., ICML 2023) | Activation quantization; migration trick relates to your weight sensitivity |
| **Must-Read** | OBQ/OBC (Frantar & Alistarh, NeurIPS 2022) | Mathematical foundation of GPTQ |
| **Must-Read** | AQLM (Egiazarian et al., ICML 2024) | Current 2-bit SOTA; understand why vector quant beats scalar |
| **Recommended** | SqueezeLLM (Kim et al., ICML 2023) | Sparse-quantized + Hessian diagonal for outliers |
| **Recommended** | SpQR (Dettmers et al., ICML 2023) | Near-lossless extreme compression baseline |
| **Recommended** | VPTQ (Liu et al., 2024) | 2024 vector PTQ SOTA — understand the frontier |
| **Recommended** | ZipLM (Kurtic et al., 2023) | Structured sparsity + GPTQ, Hessian reuse |
| **Background** | The Lottery Ticket Hypothesis (Frankle & Carlin, ICLR 2019) | Weight importance intuition |
| **Background** | Optimal Brain Surgeon (Hassibi & Stork, 1993) | Historical root of Hessian-based compression |
| **Background** | Information Theory (Cover & Thomas, textbook Ch. 13) | Water-filling derivation |
| **Context** | GPT-4 Technical Report (OpenAI, 2023) | Scale context for LLM deployment motivation |
| **Systems** | FlexGen (Sheng et al., ICML 2023) | LLM inference with memory constraints |
| **Systems** | Marlin (Frantar & Alistarh, 2024) | Mixed-precision CUDA kernels for GPTQ |

### Reading Schedule Suggestion

| Month | Papers to Read |
|:---|:---|
| Sep 2026 (now) | OBQ, SmoothQuant, SqueezeLLM — fill gaps in foundations |
| Oct 2026 | AQLM, VPTQ — know the 2-bit frontier before your experiments |
| Nov 2026 | SpQR, ZipLM — understand competing dense+sparse approaches |
| Jan 2027 | FlexGen, Marlin — systems context for your framework |
| Mar 2027 | Recent arXiv preprints (quantization, 2027) — related work update |

---

## 14. Novelty Analysis Against Prior Art

### 14.1 Distinguishing from HAWQ-V2/V3

| Dimension | HAWQ-V2 | HAWQ-V3 | **Your Work** |
|:---|:---|:---|:---|
| Era | CNN era (ResNet/BERT) | CNN era | LLM era |
| Granularity | **Per-layer** | **Per-layer** | **Per-channel** (finer) |
| Sensitivity | Hessian *trace* | Hessian trace + ILP | Hessian *diagonal* (O(mn)) |
| Solver | Pareto frontier | ILP | **Closed-form** (no solver) |
| Bit assignment | Multi-bit (2–8) | Dyadic (2^k) | **{2,4}-bit integer** |
| Architecture | CNNs + BERT | CNNs | **LLMs (decoder-only)** |
| Cross-family study | No | No | **Yes (3 families)** |

**The upgrade chain:** HAWQ computes layer-level Hessian trace → your work computes channel-level Hessian diagonal. HAWQ uses ILP solver → your work uses greedy ranking (O(d log d)). HAWQ was validated on CNNs → your work validates on modern LLMs.

### 14.2 Distinguishing from Q-BERT

Q-BERT uses per-layer Hessian sensitivity for BERT (encoder, classification tasks). Your work:
- Decoder-only architecture (generation, PPL, zero-shot)
- Channel-level (not layer-level) granularity
- Three architectures (not one)
- No fine-tuning after quantization

### 14.3 Distinguishing from AWQ

AWQ's sensitivity = activation *magnitude* (mean |X_j|). Your sensitivity = `H_jj · Var(W_{:,j})` (second moment of X times weight variance). The difference:

- AWQ: `sensitivity ∝ E[|X_j|]` — heuristic, 1st order
- Yours: `sensitivity ∝ E[X_j²] · σ²(W_{:,j})` — principled, 2nd order Taylor

**Claim:** Your metric is theoretically better motivated. The empirical question (does it outperform AWQ-style allocation?) is testable — add an AWQ-style baseline allocation row in your mixed-precision comparison table.

### 14.4 Distinguishing from SqueezeLLM

SqueezeLLM also uses Hessian diagonal for outlier identification (to put in sparse format). Differences:
- SqueezeLLM uses a *sparse* representation for outliers (hardware-specific) — your work uses integer formats only
- SqueezeLLM targets 3–4 bit, not 2-bit
- SqueezeLLM does not report cross-architecture comparison

### 14.5 Your Position on the Novelty Spectrum

```
Incremental ←────────────────────────────────────────────────────────────→ Novel
     │                                                                      │
  Fine-tune existing      Cross-arch evaluation      New allocation      New theory
  GPTQ for one model      under one protocol          mechanism
                                 ↑                         ↑
                         Your empirical                Your technical
                         contribution                  contribution
```

**You occupy both "new empirical study" AND "new mechanism"** — that's why this is publishable at a workshop/findings venue level as a BTech project.

---

## Summary: What Makes This a Strong 6-Month Project

| Criterion | Status |
|:---|:---|
| Clear research gap | ✅ Defined and positioned |
| Mathematical rigor | ✅ Phase-0 complete; rate-distortion foundation solid |
| Empirical novelty | ✅ 27-run grid doesn't exist in literature |
| Practical feasibility | ✅ Free Kaggle T4 GPUs, 0.5–1B models |
| Publication path | ✅ arXiv + ACL SRW target identified |
| Open-source artifact | ✅ GitHub repo plan in synopsis |
| Supervisor alignment | ✅ Report + arXiv as co-author |

---

*Document compiled: September 2026. Covers GPTQ, QuIP#, AWQ, HAWQ-V2/V3, Q-BERT, LLM.int8(), SqueezeLLM, SpQR, AQLM, VPTQ, SmoothQuant, and supporting theory. Verified against active literature through September 2026.*
