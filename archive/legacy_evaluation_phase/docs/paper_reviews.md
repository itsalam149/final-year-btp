# Deep Dive: Literature Review of Reference Papers
**Project:** Task-Conditioned Evaluation of Quantized Large Language Models  
**Group ID:** `DTU/CSE/BTP/GP159` | Delhi Technological University 2026–27

This document summarizes the core reference research papers located in the [`papers/`](../papers/) directory, detailing their key contributions and how they directly map to our project's methodology.

---

## 1. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
**File:** [`papers/1_GPTQ.pdf`](../papers/1_GPTQ.pdf) (Frantar et al., ICLR 2023)  
- **What it is:** The foundational paper introducing GPTQ. It leverages approximate second-order inverse Hessian information to perform one-shot weight quantization, processing weights column-by-column using the Optimal Brain Surgeon (OBS) update rule.
- **Connection to our project:** Evaluates official GPTQ-Int4 and GPTQ-Int8 releases for Qwen2.5 and Llama-3.2 to benchmark their degradation on multi-step reasoning.

## 2. Benchmarking Post-Training Quantization in LLMs
**File:** [`papers/2_Benchmarking_PTQ.pdf`](../papers/2_Benchmarking_PTQ.pdf) (Zhao et al., 2025)  
- **What it is:** A comprehensive benchmarking study that categorizes PTQ into four strategies: Compensation (GPTQ), Rotation (QuIP), Salience (AWQ), and Optimization.
- **Connection to our project:** Validates our cross-method comparison (GPTQ vs. AWQ vs. BitsAndBytes) at equal bit-widths.

## 3. APTQ: Attention-aware Post-Training Mixed-Precision Quantization
**File:** [`papers/3_APTQ_Mixed_Precision.pdf`](../papers/3_APTQ_Mixed_Precision.pdf) (Guan et al., DAC 2024)  
- **What it is:** Expands second-order quantization to consider the entire attention block (including non-linear softmax) and proposes a Hessian trace-driven mixed-precision scheme.
- **Connection to our project:** Provides structural insight into why attention layers in small models are sensitive to sub-4-bit compression.

## 4. GAMMA: Global Bit Allocation for Mixed-Precision Models
**File:** [`papers/4_GAMMA_Bit_Allocation.pdf`](../papers/4_GAMMA_Bit_Allocation.pdf) (Yao et al., 2026)  
- **What it is:** A quantizer-agnostic framework for mixed-precision allocation that learns module-wise bit preferences via a teacher-forced hidden-state reconstruction objective.
- **Connection to our project:** Serves as a reference for bit-budget allocation thresholds.

## 5. KRONQ: LLM Quantization via Kronecker-Factored Hessian
**File:** [`papers/5_KRONQ_Hessian.pdf`](../papers/5_KRONQ_Hessian.pdf) (Lee et al., 2026)  
- **What it is:** Incorporates gradient covariance into the quantization objective using Kronecker-factored approximations.
- **Connection to our project:** Theoretical backing on why higher-order loss approximations matter for preserving reasoning capacity.

## 6. D²Quant: Accurate Low-bit Post-Training Weight Quantization
**File:** [`papers/6_D2Quant_Low_Bit.pdf`](../papers/6_D2Quant_Low_Bit.pdf) (Yan et al., 2026)  
- **What it is:** Identifies that sub-4-bit PTQ degrades due to down-projection matrices and activation mean-shifts, proposing Dual-Scale Quantization and Deviation-Aware Correction.
- **Connection to our project:** Explains potential failure modes if 4-bit models experience non-linear degradation on reasoning tasks.

## 7. Task-Stratified Knowledge Scaling Laws for PTQ
**File:** [`papers/7_Knowledge_Scaling_PTQ.pdf`](../papers/7_Knowledge_Scaling_PTQ.pdf) (Zhou et al., 2026)  
- **What it is:** Investigates how quantization affects different knowledge types (memorization vs. application vs. reasoning). Crucially finds that reasoning is hypersensitive to bit-width.
- **Connection to our project:** Directly supports our **RQ1** and the hypothesis that GSM8K multi-step reasoning may experience distinct degradation curves compared to factual recall.

## 8. CrossQuant: A PTQ Method with Smaller Quantization Kernel
**File:** [`papers/8_CrossQuant.pdf`](../papers/8_CrossQuant.pdf) (Liu et al., 2024)  
- **What it is:** Analyzes precision loss from rounding small activation values to zero and cross-quantizes using row and column bounds.
- **Connection to our project:** Mathematical context on activation distortion in small language models.

## 9. Adapting Methods for Domain-Specific Japanese Small LMs
**File:** [`papers/9_Japanese_Small_LMs.pdf`](../papers/9_Japanese_Small_LMs.pdf) (Yasuno, 2026)  
- **What it is:** Analyzes how small Japanese language models respond to various PTQ strategies across general and domain-specific benchmarks.
- **Connection to our project:** Demonstrates that sub-1B parameter models exhibit unique quantization sensitivities compared to 7B+ architectures.
