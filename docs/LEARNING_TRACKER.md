# Comprehensive Project Learning Tracker & Knowledge Compendium
## Task-Conditioned Evaluation of Quantized Small LLMs (DTU Major Project BTP/GP159)

**Department of Computer Science & Engineering, Delhi Technological University**  
**Group ID:** `DTU/CSE/BTP/GP159` | **Session:** 2026–27  
**Supervisor:** Ms. Gull Kaur, Assistant Professor, Department of CSE, DTU  
**Authors:** Ekansh Agarwal (`23/CS/149`), Faqre Alam (`23/CS/150`), Guneet Toppo (`23/CS/160`)  
**Document Purpose:** Living compendium of theoretical foundations, paper takeaways, mathematical derivations, engineering insights, and viva defense preparation.

---

## 1. Executive Research Orientation

### 1.1 The Practical Problem: The Memory Wall in Edge LLMs
Modern transformer inference is almost entirely **memory-bandwidth bound** during autoregressive token generation (arithmetic intensity $\approx 1\text{ FLOP / byte}$). While an NVIDIA H100 or A100 possesses sufficient high-bandwidth memory (HBM3), consumer devices and edge hardware (smartphones, IoT gateways, laptops with 8–16 GB shared RAM) cannot fit unquantized models:
- A $1\text{B}$ parameter FP16 model requires $\approx 2.0\text{ GB}$ of VRAM just to store static weights, plus KV-cache memory during generation.
- Quantizing to 4 bits (INT4 / NF4) compresses the model down to $\approx 500\text{ MB}$, allowing it to reside entirely in small on-chip caches and relieving the memory bus by $4\times$.

### 1.2 Why Small Sub-1B Models?
Most academic literature evaluates quantization on large models ($\ge 7\text{B}$, $13\text{B}$, $70\text{B}$). However:
1. Large models have massive redundancy and high parameter capacity; an error introduced into a single weight matrix is readily compensated for by subsequent layers.
2. Sub-1B models (`Qwen2.5-0.5B-Instruct` with $490\text{M}$ parameters and `Llama-3.2-1B-Instruct` with $1.24\text{B}$ parameters) have minimal parameter redundancy. Every layer and attention head operates near its full expressive capacity.
3. Quantization noise in sub-1B models can trigger non-linear "cliff degradation", making their empirical evaluation a critical open research problem for real-world edge deployment.

### 1.3 The Core Question: Why Test Perplexity as a Proxy?
Post-training quantization papers predominantly report **Perplexity (PPL)** on standard continuous text corpora (WikiText-2, C4) because PPL is fast, deterministic, and requires no prompt engineering:
$$\text{PPL}(W) = \exp \left( - \frac{1}{T} \sum_{t=1}^T \log P_W(x_t \mid x_{<t}) \right)$$
Jin et al. (ACL Findings 2024) tested 10 benchmarks on Qwen 7B–72B and concluded that perplexity is a usable proxy for downstream performance. 

**Our Project's Core Challenge:** Does this conclusion hold for **small models**, and does it hold **task-by-task**? A small increase in language modeling perplexity might conceal catastrophic failure in multi-step arithmetic (`gsm8k`) or strict instruction compliance (`ifeval`).

---

## 2. Theoretical Mechanics of Post-Training Quantization (PTQ)

### 2.1 Uniform Quantization Basics
Given continuous floating-point weights $W \in \mathbb{R}$, uniform integer quantization maps them to discrete integers $Q \in [-2^{b-1}, 2^{b-1}-1]$ (signed) or $[0, 2^b-1]$ (unsigned):
$$Q = \text{clamp} \left( \left\lfloor \frac{W}{S} \right\rceil + Z, \, q_{\min}, \, q_{\max} \right)$$
$$\hat{W} = S \cdot (Q - Z)$$
Where:
- $S \in \mathbb{R}^+$ is the **scale factor** ($S = \frac{W_{\max} - W_{\min}}{2^b - 1}$).
- $Z \in \mathbb{Z}$ is the **zero-point** (aligns real zero with quantized zero; $Z=0$ in symmetric quantization).
- $\lfloor \cdot \rceil$ denotes round-to-nearest integer.

**Why Naive Round-To-Nearest (RTN) Fails at 4-Bit:**
In a weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, RTN treats every weight independently. When precision drops to 4 bits ($16$ discrete levels), accumulated rounding errors distort layer outputs: $E = \| W X - \hat{W} X \|_2^2$. Without error compensation, small errors compound across 24–32 transformer layers, causing output logits to collapse into gibberish.

---

### 2.2 Second-Order Error Compensation: GPTQ (Frantar et al., ICLR 2023)
GPTQ frames post-training quantization as minimizing the layer-wise output reconstruction error over calibration activations $X$:
$$\min_{\hat{W}} \| W X - \hat{W} X \|_F^2 = \min_{\hat{W}} \text{Tr} \left( (W - \hat{W}) X X^T (W - \hat{W})^T \right)$$
Using a second-order Taylor expansion around the unquantized weights, the Hessian of the objective with respect to weights is:
$$H = 2 X X^T \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$$
When a weight column $w_q$ is rounded to its nearest discrete level $\hat{w}_q$, GPTQ uses the **Optimal Brain Surgeon (OBS)** framework to update all remaining unquantized weights $w_{-q}$ to cancel out the error:
$$w_{-q} \leftarrow w_{-q} - \frac{w_q - \hat{w}_q}{[H^{-1}]_{qq}} \cdot (H^{-1})_{:, q}$$
**Key Engineering Innovations in GPTQ:**
- **Cholesky Pre-computation:** Computes the inverse Hessian $H^{-1}$ once via Cholesky decomposition ($H^{-1} = L \cdot L^T$) in $O(d^3)$ time, enabling lazy batch updates of weight blocks.
- **Act-Order (`desc_act`):** Greedily quantizes weights corresponding to activations with the highest diagonal Hessian entries first (highest sensitivity).
- **Group-Wise Scaling:** Partitions weight columns into groups (typically $g=128$), computing an individual scale $S_g$ for each block to limit dynamic range outliers.

---

### 2.3 Activation-Aware Weight Quantization: AWQ (Lin et al., MLSys 2024)
Lin et al. demonstrated that **not all weights are equally important**:
- In LLM activations $X$, $0.5\%\text{--}1\%$ of channels exhibit extreme outlier magnitudes ($10\times\text{--}100\times$ larger than average).
- The weights corresponding to these salient activation channels carry disproportionate predictive power.
- Instead of keeping these weights in FP16 (which requires custom mixed-precision compute kernels), AWQ applies an equivalent per-channel mathematical scaling factor $s \in \mathbb{R}^{d_{\text{in}}}$:
$$Y = W X = (W \cdot \text{diag}(s)) \cdot (\text{diag}(s)^{-1} \cdot X) = W' X'$$
- By multiplying the salient weight channels by $s > 1$, their relative quantization error is reduced by $1/s$. The inverse scale $s^{-1}$ is fused into the preceding LayerNorm or linear operation, requiring zero inference overhead.
- AWQ finds optimal $s$ via a grid search on calibration data: $s^* = \arg\min_s \| W X - \text{quant}(W \cdot \text{diag}(s)) \cdot \text{diag}(s)^{-1} X \|$.

---

### 2.4 Quantile Normal Float 4: NF4 (Dettmers et al., NeurIPS 2023 / QLoRA)
Information theory dictates that uniform quantization is only optimal for uniformly distributed data. However, pre-trained neural network weights follow a bell-shaped zero-mean Gaussian distribution: $W \sim \mathcal{N}(0, \sigma^2)$.

**NF4 Mechanics:**
- NF4 constructs 16 discrete quantization levels $q_i$ ($i \in \{0, \dots, 15\}$) such that each bin has **equal probability mass**:
$$q_i = \frac{1}{2} \left( Q_X \left( \frac{i}{2^k} \right) + Q_X \left( \frac{i+1}{2^k} \right) \right)$$
where $Q_X(\cdot)$ is the quantile function of the standard normal distribution $\mathcal{N}(0, 1)$.
- Exact values are normalized to $[-1, 1]$ and adjusted to include an exact representation for $0.0$, yielding the standard 16 NF4 points.
- **Double Quantization (DQ):** Quantizes the 32-bit quantization constants (scale factors) themselves into 8-bit integers with a block size of 256, saving an additional $0.37\text{ bits/parameter}$.

---

### 2.5 Vector Outlier Mixed Precision: LLM.int8() (Dettmers et al., NeurIPS 2022)
Dettmers et al. discovered that as transformer models scale, systematic emergent activation outliers develop in specific hidden dimensions across all tokens.
- **Outlier Thresholding:** When computing $Y = X W$, LLM.int8 checks whether any activation column has magnitude $\ge \alpha$ (empirically $\alpha = 6.0$).
- **Decomposition:**
  - Outlier coordinates ($< 0.1\%$ of channels) are extracted and multiplied in standard **FP16**.
  - Non-outlier coordinates ($> 99.9\%$ of channels) are quantized to **INT8** vector-wise.
  - Results are added: $Y = Y_{\text{FP16}} + Y_{\text{INT8}}$.
- **Hardware Bottleneck:** Because of dynamic branch splitting and dequantization, LLM.int8 saves memory but is frequently **slower** in wall-clock latency than FP16 on GPUs like the T4.

---

## 3. Stanford CRFM HELM Principles & Multi-Metric Evaluation

### 3.1 What is HELM (Liang et al., TMLR 2023)?
Stanford CRFM created **HELM (Holistic Evaluation of Language Models)** to resolve the flaws of modern AI evaluation:
1. **Multi-Metric Completeness:** Models must not be judged on accuracy alone. Comprehensive benchmarking requires evaluating Accuracy, Robustness, Calibration, Fairness, Bias, Toxicity, and Efficiency simultaneously.
2. **Standardized Prompt Protocols:** All models must be evaluated under identical prompt templates, few-shot examples, and decoding strategies to eliminate harness bias.
3. **Transparency & Provenance:** Full item-level logs, prompts, and model responses must be published rather than aggregate scalar numbers.

### 3.2 Confidence Calibration: ECE & Brier Score
A quantized model might maintain reasonable top-1 accuracy while its probability distribution becomes severely distorted (overconfident or underconfident). Proskurina et al. (NAACL Findings 2024) showed that 4-bit GPTQ often depresses model confidence in true labels.

**Mathematical Definitions:**
1. **Expected Calibration Error (ECE):**
   Group predicted probabilities into $M=10$ intervals $B_m \subset (0, 1]$:
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
   where $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbb{I}(\hat{y}_i = y_i)$ and $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$.
2. **Brier Score:**
   Measures mean squared error of probability forecasts against one-hot true labels:
   $$\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (p_{ik} - y_{ik})^2$$

### 3.3 Hardware Efficiency Dimensions
Real-world bit budgets require evaluating the trade-off between accuracy and resource cost:
- **Disk Footprint (MB):** Physical storage required for deployment.
- **Peak VRAM (MB):** Measured via `torch.cuda.max_memory_allocated()`.
- **Latency & Throughput:** Inference latency (ms) and throughput (tokens/second) during greedy decoding with standardized prompt and generation lengths.

---

## 4. The 15 Primary Literature Papers & Key Takeaways

| # | Paper Title | Authors / Venue | Core Insight / Method | Connection to BTP GP159 |
|:---:|:---|:---|:---|:---|
| 1 | **GPTQ** | Frantar et al. (ICLR 2023) | Inverse Hessian OBS error compensation; near-lossless 4-bit weights | Primary 8-bit & 4-bit baseline format |
| 2 | **Benchmarking PTQ** | Zhao et al. (2025) | Taxonomizes PTQ into Compensation, Rotation, Salience, and Optimization | Theoretical foundation for method comparisons |
| 3 | **APTQ** | Guan et al. (DAC 2024) | Attention-aware mixed precision using Hessian trace | Structural explanation for small LM sensitivity |
| 4 | **GAMMA** | Yao et al. (2026) | Global bit allocation via teacher-forced hidden reconstruction | Context for task-conditioned bit budgets |
| 5 | **KRONQ** | Lee et al. (2026) | Kronecker-factored Hessian approximation | Higher-order curvature preserving reasoning |
| 6 | **D²Quant** | Yan et al. (2026) | Deviation-aware correction for sub-4-bit PTQ | Identifies non-linear degradation mechanisms |
| 7 | **Knowledge Scaling PTQ** | Zhou et al. (2026) | Quantization hurts reasoning far more than factual memorization | Directly motivates Hypothesis H1 (GSM8K sensitivity) |
| 8 | **CrossQuant** | Liu et al. (2024) | Row/column activation bounds to prevent truncation | Mathematical insight into activation distortion |
| 9 | **Japanese Small LMs** | Yasuno (2026) | Empirical PTQ evaluation on sub-1B models | Empirical proof that sub-1B LMs behave uniquely |
| 10 | **HELM Core** | Liang et al. (TMLR 2023) | Holistic evaluation taxonomy, multi-metric measurement | Conceptual evaluation framework of our project |
| 11 | **DecodingTrust** | Wang et al. (NeurIPS 2023) | Trustworthiness & safety across 8 dimensions | Demonstrates vulnerability of compressed models |
| 12 | **VHELM** | Hasson et al. (2024) | Multimodal holistic evaluation extension | Extension of holistic principles to vision-language |
| 13 | **HEIM** | Lee et al. (ICML 2024) | Multi-metric evaluation for generative text-to-image | Standardized holistic metrics across modalities |
| 14 | **AIR-Bench 2024** | Stanford CRFM (2024) | Automated, safety-oriented holistic benchmark | Rigorous constraint-following evaluation |
| 15 | **Foundation Models** | Bommasani et al. (2021) | Foundational charter on risks, costs, and scale | Context on environmental and edge feasibility |

### Deep Dive: Jin et al. (ACL Findings 2024) vs. Our Work
**What Jin et al. Did:**
- Evaluated Qwen-Chat (7B, 14B, 72B) across 10 benchmarks.
- Included GSM8K and computed Pearson correlation between PPL and benchmark performance.
- Concluded that perplexity is an adequate proxy for general downstream quality.

**Why Our Work is Essential & Distinct:**
1. **Model Regime:** Jin et al. evaluated $\ge 7\text{B}$ models. We evaluate **sub-1B models** ($0.5\text{B}$ and $1\text{B}$) where parameter capacity is constrained.
2. **Task-Conditioned Decoupling:** Jin et al. pooled tasks into aggregate correlations. We test whether PPL decouples **specifically on multi-step reasoning (GSM8K) and instruction following (IFEval)** while holding for factual recall (ARC-Easy).
3. **Statistical Metric:** Pearson correlation measures linear association and is vulnerable to scale non-linearities; we utilize **Spearman's rank correlation ($\rho$)** with paired bootstrap confidence intervals.
4. **Actionable Decision Rule:** We translate findings into practical bit-budget guidelines with confidence intervals ($\le 3.0\text{ pp}$ bound).

---

## 5. Statistical Methodology & Rigor

### 5.1 Why "3 Random Seeds" is Invalid for Greedy Evaluation
In earlier project drafts, running "3 random seeds" was proposed to quantify uncertainty.
- In `lm-evaluation-harness`, multiple-choice benchmarks compute unnormalized log-likelihoods: $\log P(Y \mid X)$. This is **100% deterministic**.
- Generative tasks (`gsm8k`, `ifeval`) under greedy decoding ($\text{temperature} = 0$, `top_p = 1.0`) are also **100% deterministic**.
- Running 3 seeds yields identical numbers (zero variance), giving a false impression of reproducibility while ignoring real uncertainty.

### 5.2 Paired Bootstrap Confidence Intervals (The Real Uncertainty)
The genuine source of statistical uncertainty is **finite item sampling** from the underlying task distribution:
- Test datasets have finite size: ARC-Easy ($N=2,376$), GSM8K ($N=1,319$), IFEval ($N=541$).
- Because both FP16 baseline and quantized variants are evaluated on the **exact same test items**, their outcomes are paired: $(y_{\text{quant}}^{(i)}, y_{\text{FP16}}^{(i)})$.
- We perform **Paired Bootstrap Resampling** ($B=10,000$ iterations):
  $$\Delta^{(b)} = \frac{100}{N} \sum_{i=1}^N \left( y_{\text{quant}}^{(i*)} - y_{\text{FP16}}^{(i*)} \right)$$
- The 95% Confidence Interval is given by the empirical percentiles: $[\Delta_{2.5\%}, \Delta_{97.5\%}]$.
- **Why this matters:** Paired bootstrap isolates the true difference variance from item difficulty variance, producing significantly tighter error bounds than independent two-sample tests.

### 5.3 Spearman Rank Correlation ($\rho$) & Leave-One-Out (LOO)
To determine if perplexity is a faithful proxy for task accuracy:
$$\rho = 1 - \frac{6 \sum d_i^2}{n (n^2 - 1)}$$
where $d_i = \text{rank}(\Delta\text{PPL}_i) - \text{rank}(\Delta\text{Acc}_i)$.
- **Bootstrap CIs on $\rho$:** 2,000 iterations to establish confidence bounds.
- **Leave-One-Out Diagnostic:** Recomputes $\rho$ excluding one variant at a time:
  $$\rho_{-k} = \text{SpearmanCorr}(\Delta\text{PPL}_{-k}, \Delta\text{Acc}_{-k})$$
  Identifies whether an extreme outlier (e.g. an unstable 4-bit community variant) is single-handedly driving the correlation.
- **Dual Corpus:** Compares WikiText-2 PPL with out-of-domain C4 PPL to ensure calibration dataset overlap does not artificially deflate perplexity degradation.

### 5.4 Noise-Aware Bit-Budget Decision Rule ($\le 3.0\text{ pp}$)
Because IFEval has only $N=541$ prompts, the standard error of a point estimate is $\text{SE} \approx \sqrt{\frac{p(1-p)}{N}} \approx 2.1\text{ pp}$. A raw point drop of $3.1\text{ pp}$ could be statistical noise, while a point drop of $2.8\text{ pp}$ might have an upper bound of $6.0\text{ pp}$.
- **Acceptable ($\checkmark$):** Upper 95% bound of accuracy drop $\le 3.0\text{ pp}$ ($-\Delta\text{Acc}_{\text{low}} \le 3.0\text{ pp}$).
- **Marginal ($!$):** Point drop $\le 3.0\text{ pp}$, but upper bound $> 3.0\text{ pp}$.
- **Rejected ($\times$):** Point drop strictly exceeds $3.0\text{ pp}$.
- **Sensitivity Sweep:** Repeated at $\{1.0, 2.0, 3.0, 5.0\}\text{ pp}$ to verify threshold robustness.

---

## 6. Viva Voce Defense FAQ & Model Answers

### Q1: Why do you evaluate pre-quantized released checkpoints instead of writing your own quantization code?
**Model Answer:**  
> "Our approved departmental problem statement explicitly defines this as an empirical evaluation of production-ready, released checkpoints. In the real world, developers and edge deployment engineers do not write low-level CUDA quantization kernels from scratch; they deploy official vendor models (such as Qwen's official GPTQ-Int4/Int8) or widely used community checkpoints from Hugging Face. Furthermore, this isolates software engineering kernel bugs from the underlying empirical research questions regarding accuracy degradation, perplexity reliability, and bit budgets."

### Q2: Why did you replace the '3 random seeds' in Phase F with paired bootstrap confidence intervals?
**Model Answer:**  
> "Under greedy decoding (temperature 0) and log-likelihood multiple-choice evaluation, model outputs are completely deterministic. Running multiple seeds produces identical scalar values, measuring zero variance. The genuine statistical uncertainty comes from finite sample size—for example, IFEval has only 541 prompts. By performing a 10,000-iteration paired bootstrap over individual item differences between the quantized model and the FP16 baseline, we obtain rigorous 95% confidence intervals that accurately capture sampling error."

### Q3: Jin et al. (ACL Findings 2024) already evaluated GSM8K and concluded perplexity is a usable proxy. What is your actual research contribution?
**Model Answer:**  
> "Jin et al. evaluated models starting at 7B parameters up to 72B parameters and pooled correlations across multiple tasks. Our study addresses three distinct gaps: first, we focus on the sub-1B parameter regime (0.5B and 1B), where models have far less parameter redundancy and are prone to non-linear degradation. Second, we test the perplexity proxy hypothesis task-by-task rather than pooled, specifically examining whether multi-step arithmetic (GSM8K) and instruction following (IFEval) decouple from perplexity even when factual recall holds. Third, we establish a noise-aware, confidence-bounded bit-budget guideline matrix."

### Q4: Why is BitsAndBytes LLM.int8 often slower than FP16 despite using 8-bit integers?
**Model Answer:**  
> "BitsAndBytes LLM.int8 dynamically inspects activation vectors during the forward pass to isolate outlier dimensions ($\ge 6.0$). It separates these channels into FP16 matrix multiplication while quantizing the remainder to INT8. On GPUs like the NVIDIA T4, the overhead of dynamic thresholding, memory dequantization, and mixed-precision accumulation can exceed the memory-bandwidth savings, causing runtime latency to be equal to or slower than pure FP16."

### Q5: How does Stanford CRFM HELM influence your methodology if you aren't using the HELM codebase directly?
**Model Answer:**  
> "HELM is an evaluation philosophy, not just a software package. We adopt three core HELM principles: first, a multi-axial scenario taxonomy evaluating distinct capabilities rather than an aggregate score; second, standardized prompt templates and decoding parameters across all variants; and third, multi-metric measurement—we evaluate confidence calibration via ECE and Brier score alongside operational efficiency (VRAM, disk, throughput) rather than evaluating accuracy in isolation."

