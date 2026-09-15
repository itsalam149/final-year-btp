# Deep Dive: Literature Review of Reference Papers
**Project Context:** Architecture-Agnostic Layer-Wise PTQ of LLMs (with a focus on Hessian-Diagonal Mixed-Precision Allocation).

This document summarizes the 9 research papers located in the `papers/` directory, detailing their core contributions and how they directly relate to our project's methodology.

---

## 1. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
**File:** `1` (Frantar et al., ICLR 2023)
- **What it is:** The foundational paper that introduced GPTQ. It leverages approximate second-order information (the inverse Hessian matrix) to perform one-shot weight quantization, processing weights column-by-column using the Optimal Brain Surgeon (OBS) update rule.
- **Connection to our project:** This is the baseline algorithm we are expanding upon. While GPTQ assumes a uniform bit-width across the model, our Phase 2 introduces "Hessian-Diagonal Mixed-Precision" to dynamically allocate bits based on the very Hessian matrices GPTQ computes.

## 2. Benchmarking Post-Training Quantization in LLMs
**File:** `2.pdf` (Zhao et al., 2025)
- **What it is:** A massive benchmarking study that categorizes PTQ into four strategies: Compensation (e.g., GPTQ), Rotation (e.g., QuIP), Salience (e.g., AWQ), and Optimization. It evaluates their cross-structure (Mamba, MoE) and cross-bitwidth robustness.
- **Connection to our project:** Directly aligns with our **Phase 1**. We are also building an ablation grid across different architectures (Qwen, LLaMA, Pythia) and methods (RTN, GPTQ, Hadamard). Their findings validate our approach of evaluating cross-architecture PTQ resilience.

## 3. APTQ: Attention-aware Post-Training Mixed-Precision Quantization
**File:** `3.pdf` (Guan et al., DAC 2024)
- **What it is:** Introduces APTQ, which expands second-order quantization to consider the entire attention block (including non-linear softmax). Crucially, it proposes a **Hessian trace-driven mixed-precision quantization scheme**.
- **Connection to our project:** Highly relevant to **Phase 2**. They use the *trace* of the Hessian to assign 2-bit or 4-bit precision to layers. We are doing something very similar, but using the *diagonal* of the Hessian ($H_{jj}$) multiplied by weight variance as our sensitivity metric.

## 4. GAMMA: Global Bit Allocation for Mixed-Precision Models
**File:** `4.pdf` (Yao et al., 2026)
- **What it is:** Proposes a quantizer-agnostic framework for mixed-precision allocation that avoids the massive compute costs of search-based methods or the inaccuracy of static proxy metrics. It learns module-wise bit preferences via a teacher-forced hidden-state reconstruction objective.
- **Connection to our project:** Represents the state-of-the-art alternative to Hessian-based mixed precision. While we use a closed-form analytical score (Hessian-diagonal), GAMMA uses a differentiable learning pipeline. It is an excellent comparison point for our Phase 2 literature review.

## 5. KRONQ: LLM Quantization via Kronecker-Factored Hessian
**File:** `5.pdf` (Lee et al., 2026)
- **What it is:** Challenges the standard GPTQ assumption that only input activation statistics matter. KRONQ uses Kronecker-factored approximations to include **gradient covariance** (output-side statistics) in the quantization objective and mixed-precision allocation.
- **Connection to our project:** Extremely relevant mathematical extension of GPTQ. While we use the activation Hessian ($H = X^T X$), they prove that incorporating the gradient Hessian ($H_G$) improves 2-bit performance on massive models. 

## 6. D²Quant: Accurate Low-bit Post-Training Weight Quantization
**File:** `6.pdf` (Yan et al., 2026)
- **What it is:** Identifies that sub-4-bit PTQ degrades due to down-projection matrices and activation mean-shifts. It introduces a Dual-Scale Quantizer (DSQ) for weights and a Deviation-Aware Correction (DAC) that adjusts biases in LayerNorm to fix activation drift.
- **Connection to our project:** Provides a structural explanation for *why* GPTQ fails at 2-bit. In our project, if we notice severe degradation at 2-bit (e.g., in Qwen2.5), we can cite D²Quant to explain that LayerNorm activation drift is the likely culprit.

## 7. Task-Stratified Knowledge Scaling Laws for PTQ
**File:** `7.pdf` (Zhou et al., 2026)
- **What it is:** Investigates how fine-grained PTQ factors (group size and calibration set size) impact different types of LLM knowledge (memorization, application, reasoning). Finds that reasoning is highly sensitive to bit-width, while memorization is sensitive to calibration data.
- **Connection to our project:** Validates our use of WikiText-2 as a calibration set. If we see that our 2-bit models fail on reasoning tasks (like WinoGrande) but survive on easier tasks, this paper provides the theoretical backing for that phenomenon.

## 8. CrossQuant: A PTQ Method with Smaller Quantization Kernel
**File:** `8.pdf` (Liu et al., 2024)
- **What it is:** Defines the "quantization kernel" as the set of elements in an activation matrix that get quantized to exactly zero. Proves that rounding small values to zero is the root cause of precision loss, and proposes cross-quantizing using both row and column maximums.
- **Connection to our project:** Provides a deep mathematical look at activation quantization. While our project is primarily focused on *weight-only* quantization, this paper is useful background reading on why outlier features distort LLM mathematics.

## 9. Adapting Methods for Domain-Specific Japanese Small LMs
**File:** `9.pdf` (Yasuno, 2026)
- **What it is:** A practical engineering paper about fine-tuning Japanese LLMs via QLoRA, and evaluating how 4-bit quantization impacts them. It makes a notable finding that GQA (Grouped-Query Attention) architectures degrade severely under 4-bit quantization compared to standard MHA.
- **Connection to our project:** Directly supports our cross-architecture hypothesis in **Phase 1**! In our ablation grid, Qwen2.5 uses GQA while Pythia uses MHA. If our grid shows Qwen2.5 struggling at 2/3-bit compared to Pythia, we can use Yasuno's findings on GQA vulnerability to support our conclusions!
