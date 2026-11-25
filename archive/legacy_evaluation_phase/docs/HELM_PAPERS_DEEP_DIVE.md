# Deep Dive: Stanford CRFM HELM Literature Review & Subtopic Analysis
**Project:** Architecture-Agnostic Layer-Wise Post-Training Quantization of LLMs  
**Delhi Technological University | Session 2026–27**  
**Team Members:** Faqre Alam (23/CS/150), Ekansh Agrawal (23/CS/149), Guneet Toppo (23/CS/160)  
**Supervisor:** Ms. Gull Kaur (Assistant Professor, Department of Computer Engineering)  
**Document Context:** Detailed review of Stanford CRFM HELM papers and specialized subtopics downloaded in `papers/helm/`, establishing their technical relevance to our PTQ research.

---

## Overview

Following the guidance of **Ms. Gull Kaur**, this document analyzes the key research publications from **Stanford Center for Research on Foundation Models (CRFM)** associated with the **Holistic Evaluation of Language Models (HELM)** framework ([crfm.stanford.edu/helm](https://crfm.stanford.edu/helm/)). 

Traditional Post-Training Quantization (PTQ) literature has operated under a narrow evaluation regime (reporting solely perplexity on WikiText-2/C4 and accuracy on a few multiple-choice datasets). HELM provides a rigorous, multi-metric taxonomy that exposes the hidden trade-offs of low-bit model compression across **Uncertainty Calibration**, **Perturbation Robustness**, **Fairness & Bias**, **Safety & Toxicity**, and **Real-World Deployment Efficiency**.

The papers cataloged and analyzed below are stored in the local repository at [`papers/helm/`](file:///Users/faqrealam149/Desktop/final-year-btp/papers/helm).

---

## 1. Paper 1: Holistic Evaluation of Language Models (HELM Core Landmark Paper)
- **File:** [`papers/helm/1_HELM_Holistic_Evaluation_of_Language_Models.pdf`](file:///Users/faqrealam149/Desktop/final-year-btp/papers/helm/1_HELM_Holistic_Evaluation_of_Language_Models.pdf)
- **Authors:** Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, et al. (Stanford CRFM)
- **Publication:** Annals of the New York Academy of Sciences / Transactions on Machine Learning Research (TMLR) / arXiv:2211.09110 (250+ pages).

### Core Contributions
1. **The 2D Evaluation Taxonomy (Scenarios $\times$ Metrics):** 
   HELM establishes that evaluating language models requires mapping explicit task scenarios against 7 multi-dimensional metric categories rather than a single aggregated accuracy score.
2. **Standardization & Comparability:**
   HELM densely benchmarks 30+ major models across 42 scenarios under completely identical prompting, few-shot demonstration selection, and decoding configurations, resolving cross-paper benchmarking discrepancies.
3. **The 7 Evaluation Pillars:**
   - *Accuracy:* Exact Match, F1, Perplexity, Normalized Multiple-Choice Likelihood.
   - *Calibration:* Expected Calibration Error (ECE), Selective Classification.
   - *Robustness:* Sensitivity to semantic-preserving input noise (typos, casing, contrast sets).
   - *Fairness:* Demographic parity across protected identity groups.
   - *Bias:* Representational harms and stereotyping (StereoSet, CrowS-Pairs).
   - *Toxicity:* Generation of offensive or harmful completions (RealToxicityPrompts).
   - *Efficiency:* Hardware inference latency, throughput, and memory footprint.

### 1.1 Mathematical Formulation of HELM's Metric Pillars (Extracted from Paper Text)

From Section 4 of Liang et al. (TMLR 08/2023, pages 25–35), we extract the exact mathematical definitions of HELM's non-accuracy evaluation dimensions:

#### A. Calibration and Uncertainty (Section 4.4, Page 27–28)
HELM partitions model predictions across instances $i = 1, \dots, N$ into $M = 10$ confidence bins $B_1, \dots, B_{10}$ based on the model's confidence $p_i = \max_k P(y_i = k \mid x_i)$:
$$\mathrm{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} p_i, \qquad \mathrm{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(\hat{y}_i = y_i)$$
The **Expected Calibration Error (ECE)** is strictly defined as:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \mathrm{acc}(B_m) - \mathrm{conf}(B_m) \right|$$
*Key Insight from HELM:* Many models that achieve high raw task accuracy are severely miscalibrated. In our PTQ context, quantizing weights to 2-bit or 3-bit scales destroys the logit margin, causing ECE to spike even when multiple-choice accuracy drops only slightly.

#### B. Robustness via Input Perturbations (Section 4.5, Page 28–30)
HELM defines local robustness by measuring performance invariance under semantic-preserving input transformations $\tau(x)$:
- **Typographical noise:** Character swaps, omissions, and common keyboard errors.
- **Dialect / formatting invariance:** Case alterations and whitespace modifications.
The **Robust Accuracy** is measured as:
$$\mathrm{Acc}_{\mathrm{robust}} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}(\hat{y}(\tau(x_i)) = y_i)$$
*Key Insight from HELM:* In empirical testing (Finding 12, Page 8), models like OPT-175B plummeted from $51.3\%$ standard accuracy down to $8.8\%$ robust accuracy under input perturbations. For our BTP, testing whether randomized Hadamard rotation (QuIP) restores robust accuracy at 2-bit is a breakthrough research contribution.

#### C. Inference Efficiency (Section 4.9, Page 33–35)
HELM differentiates between:
1. **Denoised Inference Runtime:** Real hardware execution time after filtering out system-level latency jitter (measured in ms/query or ms/token).
2. **Idealized Inference Runtime / Arithmetic Intensity:** Theoretical compute and memory bandwidth requirements based on model parameter count, FLOPs, and activation memory.
*Key Insight from HELM:* Evaluating efficiency alongside accuracy prevents researchers from proposing over-engineered methods that provide negligible real-world speedup despite theoretical compression.

### 1.2 Direct Connection to Our BTP Quantization Project
- **The Perplexity Illusion:** In our Phase 1 ablation grid, models quantized to 4-bit (e.g., Qwen2.5-0.5B via GPTQ) achieve near-lossless WikiText-2 perplexity. However, HELM proves that models with identical perplexity can exhibit radically divergent robustness and calibration.
- **Methodological Upgrade:** In our Phase 2 and 3, we adopt HELM's multi-metric paradigm to demonstrate that our novel **Hessian-Diagonal Mixed-Precision Allocation** ($\hat{s}_j = \mathbf{H}_{jj} \cdot \mathrm{Var}(\mathbf{W}_{:,j})$) does not just retain perplexity—it preserves calibration and robustness better than uniform 3-bit baselines.

---

## 2. Paper 2: DecodingTrust — Comprehensive Trustworthiness Assessment
- **File:** [`papers/helm/2_DecodingTrust_Trustworthiness_and_Safety.pdf`](file:///Users/faqrealam149/Desktop/final-year-btp/papers/helm/2_DecodingTrust_Trustworthiness_and_Safety.pdf)
- **Authors:** Boxin Wang, Weixin Chen, Hengzhi Pei, Chulin Xie, Mintong Kang, et al. (Stanford / UIUC collaboration)
- **Publication:** NeurIPS 2023 Outstanding Paper Award / arXiv:2306.11698.

### Core Contributions
1. **Multi-Faceted Trustworthiness Benchmark:**
   DecodingTrust operationalizes safety and trustworthiness into 8 distinct dimensions:
   - Toxicity
   - Stereotype Bias
   - Adversarial Robustness
   - Out-of-Distribution Robustness
   - Robustness to Adversarial Demonstrations
   - Privacy Leakage (PII extraction)
   - Machine Ethics
   - Calibration & Uncertainty
2. **Vulnerability of Aligned Models:**
   The paper proves that even models rigorously aligned via RLHF (such as GPT-4 and GPT-3.5) harbor severe vulnerabilities when evaluated under targeted adversarial perturbations and jailbreaks.

### 1.1 Mathematical Formulation of HELM's Metric Pillars (Extracted from Paper Text)

From Section 4 of Liang et al. (TMLR 08/2023, pages 25–35), we extract the exact mathematical definitions of HELM's non-accuracy evaluation dimensions:

#### A. Calibration and Uncertainty (Section 4.4, Page 27–28)
HELM partitions model predictions across instances $i = 1, \dots, N$ into $M = 10$ confidence bins $B_1, \dots, B_{10}$ based on the model's confidence $p_i = \max_k P(y_i = k \mid x_i)$:
$$\mathrm{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} p_i, \qquad \mathrm{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(\hat{y}_i = y_i)$$
The **Expected Calibration Error (ECE)** is strictly defined as:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \mathrm{acc}(B_m) - \mathrm{conf}(B_m) \right|$$
*Key Insight from HELM:* Many models that achieve high raw task accuracy are severely miscalibrated. In our PTQ context, quantizing weights to 2-bit or 3-bit scales destroys the logit margin, causing ECE to spike even when multiple-choice accuracy drops only slightly.

#### B. Robustness via Input Perturbations (Section 4.5, Page 28–30)
HELM defines local robustness by measuring performance invariance under semantic-preserving input transformations $\tau(x)$:
- **Typographical noise:** Character swaps, omissions, and common keyboard errors.
- **Dialect / formatting invariance:** Case alterations and whitespace modifications.
The **Robust Accuracy** is measured as:
$$\mathrm{Acc}_{\mathrm{robust}} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}(\hat{y}(\tau(x_i)) = y_i)$$
*Key Insight from HELM:* In empirical testing (Finding 12, Page 8), models like OPT-175B plummeted from $51.3\%$ standard accuracy down to $8.8\%$ robust accuracy under input perturbations. For our BTP, testing whether randomized Hadamard rotation (QuIP) restores robust accuracy at 2-bit is a breakthrough research contribution.

#### C. Inference Efficiency (Section 4.9, Page 33–35)
HELM differentiates between:
1. **Denoised Inference Runtime:** Real hardware execution time after filtering out system-level latency jitter (measured in ms/query or ms/token).
2. **Idealized Inference Runtime / Arithmetic Intensity:** Theoretical compute and memory bandwidth requirements based on model parameter count, FLOPs, and activation memory.
*Key Insight from HELM:* Evaluating efficiency alongside accuracy prevents researchers from proposing over-engineered methods that provide negligible real-world speedup despite theoretical compression.

### 1.2 Direct Connection to Our BTP Quantization Project
- **Safety Degradation Under Low-Bit PTQ:**
  Safety alignment vectors reside in subtle, low-magnitude weight directions. Coarse quantization (especially uniform 2-bit or aggressive RTN) acts as a mathematical truncation that strips these fine-grained refusal boundaries.
- **Empirical Hypothesis for Our Research:**
  By testing quantized checkpoints against DecodingTrust/HELM safety subsets (e.g., toxicity generation rate), we can evaluate whether our Hessian-diagonal allocator protects safety-critical weights compared to uniform rounding.

---

## 3. Paper 3: VHELM — Holistic Evaluation of Vision-Language Models
- **File:** [`papers/helm/3_VHELM_Holistic_Evaluation_Vision_Language.pdf`](file:///Users/faqrealam149/Desktop/final-year-btp/papers/helm/3_VHELM_Holistic_Evaluation_Vision_Language.pdf)
- **Authors:** Yana Hasson et al. (Stanford CRFM)
- **Publication:** arXiv:2410.07112.

### Core Contributions
1. **Extending HELM to Multimodal Architectures:**
   Adapts the HELM holistic philosophy to Vision-Language Models (VLMs), recognizing that multimodal models face compound vulnerabilities combining visual artifacts and linguistic biases.
2. **The 9 Evaluation Dimensions:**
   Evaluates VLMs across Visual Perception, Knowledge, Reasoning, Bias, Fairness, Multilinguality, Robustness, Safety, and Toxicity.
3. **Perception vs. Reasoning Trade-off:**
   Demonstrates that multimodal compression or architectural shifts frequently compromise visual reasoning far more rapidly than basic visual recognition.

### 1.1 Mathematical Formulation of HELM's Metric Pillars (Extracted from Paper Text)

From Section 4 of Liang et al. (TMLR 08/2023, pages 25–35), we extract the exact mathematical definitions of HELM's non-accuracy evaluation dimensions:

#### A. Calibration and Uncertainty (Section 4.4, Page 27–28)
HELM partitions model predictions across instances $i = 1, \dots, N$ into $M = 10$ confidence bins $B_1, \dots, B_{10}$ based on the model's confidence $p_i = \max_k P(y_i = k \mid x_i)$:
$$\mathrm{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} p_i, \qquad \mathrm{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(\hat{y}_i = y_i)$$
The **Expected Calibration Error (ECE)** is strictly defined as:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \mathrm{acc}(B_m) - \mathrm{conf}(B_m) \right|$$
*Key Insight from HELM:* Many models that achieve high raw task accuracy are severely miscalibrated. In our PTQ context, quantizing weights to 2-bit or 3-bit scales destroys the logit margin, causing ECE to spike even when multiple-choice accuracy drops only slightly.

#### B. Robustness via Input Perturbations (Section 4.5, Page 28–30)
HELM defines local robustness by measuring performance invariance under semantic-preserving input transformations $\tau(x)$:
- **Typographical noise:** Character swaps, omissions, and common keyboard errors.
- **Dialect / formatting invariance:** Case alterations and whitespace modifications.
The **Robust Accuracy** is measured as:
$$\mathrm{Acc}_{\mathrm{robust}} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}(\hat{y}(\tau(x_i)) = y_i)$$
*Key Insight from HELM:* In empirical testing (Finding 12, Page 8), models like OPT-175B plummeted from $51.3\%$ standard accuracy down to $8.8\%$ robust accuracy under input perturbations. For our BTP, testing whether randomized Hadamard rotation (QuIP) restores robust accuracy at 2-bit is a breakthrough research contribution.

#### C. Inference Efficiency (Section 4.9, Page 33–35)
HELM differentiates between:
1. **Denoised Inference Runtime:** Real hardware execution time after filtering out system-level latency jitter (measured in ms/query or ms/token).
2. **Idealized Inference Runtime / Arithmetic Intensity:** Theoretical compute and memory bandwidth requirements based on model parameter count, FLOPs, and activation memory.
*Key Insight from HELM:* Evaluating efficiency alongside accuracy prevents researchers from proposing over-engineered methods that provide negligible real-world speedup despite theoretical compression.

### 1.2 Direct Connection to Our BTP Quantization Project
- **Cross-Architecture Parallel:**
  Just as VHELM analyzes how multimodal fusion layers degrade differently than text encoders, our project investigates how attention variations (7:1 GQA in Qwen2.5 vs. 4:1 GQA in Llama-3.2 vs. Full MHA in Pythia) dictate quantization resilience.
- **Future Work Projection:**
  In our final dissertation (Phase 3), we can cite VHELM to outline how our architecture-agnostic layer-wise Hessian framework can be naturally extended to multimodal vision transformers.

---

## 4. Paper 4: HEIM — Holistic Evaluation of Text-to-Image Models
- **File:** [`papers/helm/4_HEIM_Holistic_Evaluation_Text_to_Image.pdf`](file:///Users/faqrealam149/Desktop/final-year-btp/papers/helm/4_HEIM_Holistic_Evaluation_Text_to_Image.pdf)
- **Authors:** Tony Lee, Michihiro Yasunaga, Chenlin Meng, Yifan Mai, Joon Sung Park, et al. (Stanford CRFM)
- **Publication:** ICML 2024 / arXiv:2311.04287.

### Core Contributions
1. **Holistic Generation Evaluation:**
   Establishes a 12-aspect evaluation framework for generative text-to-image models (including text-image alignment, image quality, aesthetics, originality, reasoning, bias, and safety/NSFW filtering).
2. **Failure of Single Metrics:**
   Reveals that standard image generation metrics (like FID - Fréchet Inception Distance) fail to capture crucial aspects such as compositional alignment and demographic representational harms.

### 1.1 Mathematical Formulation of HELM's Metric Pillars (Extracted from Paper Text)

From Section 4 of Liang et al. (TMLR 08/2023, pages 25–35), we extract the exact mathematical definitions of HELM's non-accuracy evaluation dimensions:

#### A. Calibration and Uncertainty (Section 4.4, Page 27–28)
HELM partitions model predictions across instances $i = 1, \dots, N$ into $M = 10$ confidence bins $B_1, \dots, B_{10}$ based on the model's confidence $p_i = \max_k P(y_i = k \mid x_i)$:
$$\mathrm{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} p_i, \qquad \mathrm{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(\hat{y}_i = y_i)$$
The **Expected Calibration Error (ECE)** is strictly defined as:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \mathrm{acc}(B_m) - \mathrm{conf}(B_m) \right|$$
*Key Insight from HELM:* Many models that achieve high raw task accuracy are severely miscalibrated. In our PTQ context, quantizing weights to 2-bit or 3-bit scales destroys the logit margin, causing ECE to spike even when multiple-choice accuracy drops only slightly.

#### B. Robustness via Input Perturbations (Section 4.5, Page 28–30)
HELM defines local robustness by measuring performance invariance under semantic-preserving input transformations $\tau(x)$:
- **Typographical noise:** Character swaps, omissions, and common keyboard errors.
- **Dialect / formatting invariance:** Case alterations and whitespace modifications.
The **Robust Accuracy** is measured as:
$$\mathrm{Acc}_{\mathrm{robust}} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}(\hat{y}(\tau(x_i)) = y_i)$$
*Key Insight from HELM:* In empirical testing (Finding 12, Page 8), models like OPT-175B plummeted from $51.3\%$ standard accuracy down to $8.8\%$ robust accuracy under input perturbations. For our BTP, testing whether randomized Hadamard rotation (QuIP) restores robust accuracy at 2-bit is a breakthrough research contribution.

#### C. Inference Efficiency (Section 4.9, Page 33–35)
HELM differentiates between:
1. **Denoised Inference Runtime:** Real hardware execution time after filtering out system-level latency jitter (measured in ms/query or ms/token).
2. **Idealized Inference Runtime / Arithmetic Intensity:** Theoretical compute and memory bandwidth requirements based on model parameter count, FLOPs, and activation memory.
*Key Insight from HELM:* Evaluating efficiency alongside accuracy prevents researchers from proposing over-engineered methods that provide negligible real-world speedup despite theoretical compression.

### 1.2 Direct Connection to Our BTP Quantization Project
- **Broader Theoretical Framing:**
  Reinforces the foundational lesson that in generative AI, evaluating compressed or quantized models using a single loss metric (Perplexity for LLMs, FID for Diffusion Models) provides a fundamentally misleading assessment of real deployment viability.

---

## 5. Paper 5: AIR-Bench 2024 — Safety & Risk Assessment Benchmark
- **File:** [`papers/helm/5_AIR_Bench_2024_Safety_Benchmark.pdf`](file:///Users/faqrealam149/Desktop/final-year-btp/papers/helm/5_AIR_Bench_2024_Safety_Benchmark.pdf)
- **Authors:** AIR-Bench Team (Integrated with Stanford HELM Safety)
- **Publication:** arXiv:2407.17436.

### Core Contributions
1. **Autonomous Risk Assessment:**
   Structures LLM safety evaluation into standard regulatory and operational risk categories: violent crimes, non-violent illegal acts, cyberattacks, sexual offenses, hate speech, and suicide/self-harm.
2. **Automated Dynamic Auditing:**
   Provides standardized test sets designed to detect safety boundary degradation with minimal prompt counts.

### 1.1 Mathematical Formulation of HELM's Metric Pillars (Extracted from Paper Text)

From Section 4 of Liang et al. (TMLR 08/2023, pages 25–35), we extract the exact mathematical definitions of HELM's non-accuracy evaluation dimensions:

#### A. Calibration and Uncertainty (Section 4.4, Page 27–28)
HELM partitions model predictions across instances $i = 1, \dots, N$ into $M = 10$ confidence bins $B_1, \dots, B_{10}$ based on the model's confidence $p_i = \max_k P(y_i = k \mid x_i)$:
$$\mathrm{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} p_i, \qquad \mathrm{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(\hat{y}_i = y_i)$$
The **Expected Calibration Error (ECE)** is strictly defined as:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \mathrm{acc}(B_m) - \mathrm{conf}(B_m) \right|$$
*Key Insight from HELM:* Many models that achieve high raw task accuracy are severely miscalibrated. In our PTQ context, quantizing weights to 2-bit or 3-bit scales destroys the logit margin, causing ECE to spike even when multiple-choice accuracy drops only slightly.

#### B. Robustness via Input Perturbations (Section 4.5, Page 28–30)
HELM defines local robustness by measuring performance invariance under semantic-preserving input transformations $\tau(x)$:
- **Typographical noise:** Character swaps, omissions, and common keyboard errors.
- **Dialect / formatting invariance:** Case alterations and whitespace modifications.
The **Robust Accuracy** is measured as:
$$\mathrm{Acc}_{\mathrm{robust}} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}(\hat{y}(\tau(x_i)) = y_i)$$
*Key Insight from HELM:* In empirical testing (Finding 12, Page 8), models like OPT-175B plummeted from $51.3\%$ standard accuracy down to $8.8\%$ robust accuracy under input perturbations. For our BTP, testing whether randomized Hadamard rotation (QuIP) restores robust accuracy at 2-bit is a breakthrough research contribution.

#### C. Inference Efficiency (Section 4.9, Page 33–35)
HELM differentiates between:
1. **Denoised Inference Runtime:** Real hardware execution time after filtering out system-level latency jitter (measured in ms/query or ms/token).
2. **Idealized Inference Runtime / Arithmetic Intensity:** Theoretical compute and memory bandwidth requirements based on model parameter count, FLOPs, and activation memory.
*Key Insight from HELM:* Evaluating efficiency alongside accuracy prevents researchers from proposing over-engineered methods that provide negligible real-world speedup despite theoretical compression.

### 1.2 Direct Connection to Our BTP Quantization Project
- **Safety Benchmarking for BTP:**
  Provides the structured categories needed for our Phase 2 safety audit, allowing us to test whether 2-bit and 3-bit quantized models experience increased willingness to comply with prohibited prompts compared to their FP16 baseline.

---

## 6. Paper 6: On the Opportunities and Risks of Foundation Models
- **File:** [`papers/helm/6_Opportunities_and_Risks_of_Foundation_Models.pdf`](file:///Users/faqrealam149/Desktop/final-year-btp/papers/helm/6_Opportunities_and_Risks_of_Foundation_Models.pdf)
- **Authors:** Rishi Bommasani, Drew A. Hudson, Ehsan Adeli, Russ Altman, et al. (Stanford CRFM)
- **Publication:** arXiv:2108.07258 (200+ page foundational report).

### Core Contributions
1. **Coined "Foundation Models":**
   The defining academic manifesto that introduced the term "Foundation Models" and unified the study of large-scale pretrained models across modalities.
2. **Centrality of Resource Constraints:**
   Section 4.5 of the report specifically highlights the **computational and memory bottlenecks** of deploying foundation models, explicitly calling for advances in **pruning, quantization, and architectural efficiency** to democratize access.

### 1.1 Mathematical Formulation of HELM's Metric Pillars (Extracted from Paper Text)

From Section 4 of Liang et al. (TMLR 08/2023, pages 25–35), we extract the exact mathematical definitions of HELM's non-accuracy evaluation dimensions:

#### A. Calibration and Uncertainty (Section 4.4, Page 27–28)
HELM partitions model predictions across instances $i = 1, \dots, N$ into $M = 10$ confidence bins $B_1, \dots, B_{10}$ based on the model's confidence $p_i = \max_k P(y_i = k \mid x_i)$:
$$\mathrm{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} p_i, \qquad \mathrm{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(\hat{y}_i = y_i)$$
The **Expected Calibration Error (ECE)** is strictly defined as:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \mathrm{acc}(B_m) - \mathrm{conf}(B_m) \right|$$
*Key Insight from HELM:* Many models that achieve high raw task accuracy are severely miscalibrated. In our PTQ context, quantizing weights to 2-bit or 3-bit scales destroys the logit margin, causing ECE to spike even when multiple-choice accuracy drops only slightly.

#### B. Robustness via Input Perturbations (Section 4.5, Page 28–30)
HELM defines local robustness by measuring performance invariance under semantic-preserving input transformations $\tau(x)$:
- **Typographical noise:** Character swaps, omissions, and common keyboard errors.
- **Dialect / formatting invariance:** Case alterations and whitespace modifications.
The **Robust Accuracy** is measured as:
$$\mathrm{Acc}_{\mathrm{robust}} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}(\hat{y}(\tau(x_i)) = y_i)$$
*Key Insight from HELM:* In empirical testing (Finding 12, Page 8), models like OPT-175B plummeted from $51.3\%$ standard accuracy down to $8.8\%$ robust accuracy under input perturbations. For our BTP, testing whether randomized Hadamard rotation (QuIP) restores robust accuracy at 2-bit is a breakthrough research contribution.

#### C. Inference Efficiency (Section 4.9, Page 33–35)
HELM differentiates between:
1. **Denoised Inference Runtime:** Real hardware execution time after filtering out system-level latency jitter (measured in ms/query or ms/token).
2. **Idealized Inference Runtime / Arithmetic Intensity:** Theoretical compute and memory bandwidth requirements based on model parameter count, FLOPs, and activation memory.
*Key Insight from HELM:* Evaluating efficiency alongside accuracy prevents researchers from proposing over-engineered methods that provide negligible real-world speedup despite theoretical compression.

### 1.2 Direct Connection to Our BTP Quantization Project
- **Direct Motivation for Our Thesis:**
  Our BTP directly answers the challenge framed in this report: solving the memory-bandwidth bottleneck to enable local, edge deployment of foundation models without catastrophic capability collapse.

---

## Summary Matrix: HELM Papers vs. Our PTQ Research

| Paper | Primary Focus | Key Metrics / Concepts | Direct Application to Our BTP |
|:---|:---|:---|:---|
| **HELM Core** | General LLM Evaluation | 7 Metric Pillars, Scenarios $\times$ Metrics, Standardization | Establishes our multi-metric evaluation suite (ECE, Robustness, Efficiency). |
| **DecodingTrust** | Trustworthiness & Safety | 8 Trust Dimensions, Adversarial Robustness, Calibration | Provides methodology to test safety guardrail erosion under low-bit PTQ. |
| **VHELM** | Multimodal Evaluation | 9 Vision-Language Dimensions, Visual Reasoning | Future-work roadmap extending our layer-wise Hessian allocator to VLMs. |
| **HEIM** | Generative Image Models | 12 Aspect Generation Evaluation | Theoretical justification for rejecting single-metric (PPL) evaluations. |
| **AIR-Bench** | Safety & Operational Risk | Standardized Risk Taxonomies (Hate, Cyber, Violence) | Structured test harness for evaluating model refusal rates across bit-widths. |
| **Foundation Models**| Paradigm & Ecosystem | Emergence, Homogenization, Computational Efficiency | Core academic citation motivating our memory-bandwidth PTQ research. |

---

*Authored by Antigravity for the BTP Project Team: Faqre Alam, Ekansh Agrawal, and Guneet Toppo. Delhi Technological University, 2026–27.*

