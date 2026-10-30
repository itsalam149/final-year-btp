# Master Build & Execution Plan (v2, reviewed and enhanced)
## Task-Conditioned Evaluation of Quantized Small LLMs: Accuracy Degradation, Reliability of Perplexity as a Proxy, and Practical Bit-Budget Guidelines

**Department of Computer Science & Engineering, Delhi Technological University**  
**B.Tech Major Project | Session 2026–27 | Group ID: `DTU/CSE/BTP/GP159`**

- **Team Members:**
  - Ekansh Agarwal (`23/CS/149`)
  - Faqre Alam (`23/CS/150`)
  - Guneet Toppo (`23/CS/160`)
- **Supervisor:** Ms. Gull Kaur, Assistant Professor, Department of CSE, DTU
- **Plan Reviewed:** 30 Oct 2026

> This file replaces `BUILD_PLAN.md` (v1). Section 0 lists what changed and why. Everything marked **[VERIFIED]** was checked against active literature and Hugging Face repositories on 30 Oct 2026; **[UNVERIFIED]** means it requires an empirical runtime check.

---

## 0. Review Summary

### Verdict
The plan is **technically sound, methodologically rigorous, and highly feasible**. Evaluating pre-quantized released checkpoints requires only free Kaggle T4 GPU time, and the research questions address an open problem in efficient deployment. The updates below resolve the literature gap positioning, remove pseudo-repetition ("3 seeds" on deterministic greedy decoding), introduce paired-bootstrap uncertainty quantification, and incorporate HELM-style calibration and efficiency measurement.

### What Changed from v1, and Why

| # | Component | Change Made | Rationale / Academic Need |
|---|---|---|---|
| 1 | **Literature Gap** (Section 1) | **Gap claim rewritten honestly:** Acknowledges Jin et al. (ACL Findings 2024), which already evaluated GSM8K on 7B–72B models and reported PPL as a usable proxy. | Avoids false novelty claims; accurately positions our study on **small sub-1B models**, **IFEval**, **task-conditioned proxy decoupling**, and a **noise-aware bit-budget rule**. |
| 2 | **Checkpoint Audit** (Section 2.1) | **Provenance audit table added** (`configs/checkpoints.yaml`). | Pre-quantized checkpoints vary by publisher, group size, and calibration corpus. Disentangles algorithm effects from publisher choices and records them transparently. |
| 3 | **Uncertainty Quantification** (Section 6.1) | **Replaced "3 random seeds" with Paired Bootstrap CIs (10,000 iterations).** | Loglikelihood tasks and greedy decoding are fully deterministic; random seeds produce identical outputs. Real statistical uncertainty stems from finite sample sizes ($N$). |
| 4 | **Bit-Budget Criterion** (Section 6.3) | **Made the $\le 3\text{ pp}$ rule noise-aware.** | IFEval ($N=541$) has a standard error of $\sim 2\text{ pp}$. A point-estimate drop of 3 pp is indistinguishable from noise without confidence bounds. |
| 5 | **Perplexity Proxy Test** (Section 6.2) | **Strengthened with bootstrap CIs, leave-one-out tests, and a second PPL corpus.** | Spearman $\rho$ on $N \approx 5\text{--}9$ points is sensitive to leverage points. Adding a second corpus controls for calibration-data contamination. |
| 6 | **HELM Alignment** (Section 5) | **Added real multi-metric measurement:** Calibration (ECE, Brier) and Efficiency (VRAM, disk size, tokens/s). | HELM requires holistic, multi-axial evaluation. Real packed checkpoints enable measuring actual inference speedup and confidence shifts at near-zero extra cost. |
| 7 | **Environment & Riks** (Section 9) | **Documented hardware and library constraints:** AutoAWQ archive status, T4 lacking bfloat16, gated Llama licence, and `lm-eval` version caps. | Prevents silent runtime failures during GPU runs. |
| 8 | **Compute Estimates** (Section 8) | **Calibrated execution time:** Adjusted from 10–40 min to realistic 1–2 GPU-hours per variant across the full 7-task suite. | MMLU ($14\text{k}$ items) and generative tasks (GSM8K, IFEval) dominate runtime. |
| 9 | **Roadmap & Priorities** (Section 7) | **Re-based from 30 Oct 2026 with Must / Should / Could tiers.** | Aligns deliverables with DTU mid-semester and end-semester deadlines. |
| 10 | **Housekeeping** (Section 14) | **Standardized relative links, updated README, and established `PROJECT_LOG.md`.** | Ensures repository portability and clean peer-review standards. |

---

## 1. Project Overview & Research Context

### 1.1 The Problem
Post-Training Quantization (PTQ) compresses transformer weights from 16-bit floating point down to 8-bit or 4-bit integers to alleviate memory-bandwidth bottlenecks. Model compression literature frequently reports **Perplexity (PPL)** on standard corpora (e.g., WikiText-2) as a primary evaluation metric because it is fast and cheap to compute. Benchmarking surveys (e.g., Jin et al., 2024) have argued that perplexity correlates well with general downstream benchmark scores.

However, in production deployments—especially on resource-constrained edge devices—practitioners require specific task capabilities: exact arithmetic computation, multi-step logical deduction, and strict constraint adherence. Relying on an aggregate, language-modeling loss metric may obscure localized catastrophic failures in reasoning or instruction compliance.

### 1.2 Prior Literature & Ground Truth
- **Jin et al., "A Comprehensive Evaluation of Quantization Strategies for Large Language Models" (ACL Findings 2024, arXiv:2402.16775) [VERIFIED]:** Evaluated 10 benchmarks across knowledge, alignment, and efficiency on Qwen-Chat (7B, 14B, 72B). Concluded that 4-bit quantization retains most capabilities and that perplexity can serve as an adequate proxy for most downstream tasks, using Pearson correlation and including GSM8K.
- **Quantization and Reasoning Dynamics (2025–2026) [VERIFIED]:** Recent studies—including *Silent Failures in Quantized LLM Reasoning* (Oladri et al., 2026), *Extreme Low-Bit Inference in Reasoning Models* (2026), and *When Quantization Affects Confidence of LLMs?* (Proskurina et al., NAACL Findings 2024)—demonstrate that quantization can disproportionately alter confidence calibration and multi-step reasoning trajectories. However, these papers focus primarily on large models ($\ge 3\text{B}$) or single quantization formats.
- **Literature Correction:** The citation "Twagirayezu & Mitra (2026)" from early informal notes could not be verified in arXiv/ACL databases and is flagged for exclusion or replacement with verified 2025–2026 citations.

### 1.3 The Honest Research Gap
1. **Small-Scale Regime (0.5B to 1B):** Real-world edge and mobile deployments depend on sub-1B models (`Qwen2.5-0.5B-Instruct`, `Llama-3.2-1B-Instruct`), which operate near capability thresholds where quantization noise may induce non-linear degradation.
2. **Task-Conditioned Proxy Decoupling:** Instead of pooling all benchmarks into a single correlation coefficient, we test whether $\Delta\text{PPL}$ tracks $\Delta\text{Acc}$ for **each distinct capability axis separately** (factual recall, commonsense, domain knowledge, multi-step math, and instruction following).
3. **Instruction Following (IFEval):** Testing programmatic constraint satisfaction under quantization and verifying whether perplexity predicts instruction-following degradation.
4. **Noise-Aware Bit-Budget Decision Rules:** Establishing empirical bit-budget guidance using paired-bootstrap confidence intervals ($\le 3\text{ pp}$ bound) rather than arbitrary point cutoffs.
5. **Real-Checkpoint Multi-Metric Evaluation:** Measuring actual VRAM allocation, disk size, inference throughput, and confidence calibration alongside accuracy in the spirit of HELM.

### 1.4 Research Questions (Operationalized)
- **RQ1 (Task-Conditioned Degradation):** For each (model, variant, task), what is $\Delta\text{Acc} = \text{Acc}_{\text{quant}} - \text{Acc}_{\text{FP16}}$ in percentage points, accompanied by a 95% paired-bootstrap confidence interval?
- **RQ2 (Reliability of Perplexity as a Proxy):** For each task, does Spearman rank correlation $\rho(\Delta\text{PPL}, \Delta\text{Acc})$ indicate a faithful proxy ($\rho \le -0.7$ with CI excluding 0) or decoupling ($|\rho| < 0.4$ or CI spanning 0)?
- **RQ3 (Practical Bit-Budget Guidelines):** Under a noise-aware acceptance criterion, what is the minimum acceptable bit precision for each task category?
- **Hypothesis H1:** Multi-step arithmetic reasoning (GSM8K) experiences greater relative degradation under 4-bit quantization than surface factual recall (ARC-Easy) or commonsense knowledge (HellaSwag).

### 1.5 HELM Alignment
We adopt the foundational principles of the **Stanford CRFM Holistic Evaluation of Language Models (HELM)** (Liang et al., TMLR 2023):
- Multi-metric capability evaluation across distinct scenario axes.
- Standardized, reproducible prompt formatting and fixed decoding parameters.
- Incorporating calibration (Expected Calibration Error, Brier score) and efficiency (throughput, memory) to assess the holistic impact of compression.

---

## 2. Models and Quantized Variants

Per specification: **We do not write custom quantization kernels.** We evaluate official and community Hugging Face released checkpoints:

| Model Family | Variant | Target Precision | Method | Hugging Face Checkpoint | Status / Verification |
|:---|:---|:---:|:---|:---|:---|
| **Qwen2.5-0.5B-Instruct** | `fp16` | 16-bit | Baseline | `Qwen/Qwen2.5-0.5B-Instruct` | Verified official baseline |
| | `bnb_int8` | 8-bit | BitsAndBytes | `Qwen/Qwen2.5-0.5B-Instruct` (`load_in_8bit=True`) | Dynamic LLM.int8 |
| | `gptq_int8` | 8-bit | GPTQ | `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8` | **[VERIFIED]** Official, Apache-2.0, ~640 MB |
| | `bnb_nf4` | 4-bit | BitsAndBytes NF4 | `Qwen/Qwen2.5-0.5B-Instruct` (`bnb_4bit_quant_type="nf4"`) | NormalFloat4 |
| | `gptq_int4` | 4-bit | GPTQ | `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4` | **[VERIFIED]** Official, Apache-2.0, ~460 MB |
| | `awq_int4` | 4-bit | AWQ | `Qwen/Qwen2.5-0.5B-Instruct-AWQ` | **[VERIFIED]** Official, Apache-2.0 |
| **Llama-3.2-1B-Instruct** | `fp16` | 16-bit | Baseline | `meta-llama/Llama-3.2-1B-Instruct` | Gated (HF token required) |
| | `bnb_int8` | 8-bit | BitsAndBytes | `meta-llama/Llama-3.2-1B-Instruct` (`load_in_8bit=True`) | Dynamic LLM.int8 |
| | `bnb_nf4` | 4-bit | BitsAndBytes NF4 | `meta-llama/Llama-3.2-1B-Instruct` (`bnb_4bit_quant_type="nf4"`) | NormalFloat4 |
| | `gptq_int4` | 4-bit | GPTQ | `shuyuej/Llama-3.2-1B-Instruct-GPTQ` | **[VERIFIED]** Community release |
| | `awq_int4` | 4-bit | AWQ | `ciCic/llama-3.2-1B-Instruct-AWQ` | **[VERIFIED]** Community release |
| **TinyLlama-1.1B** *(Optional)* | `fp16`, `gptq` | 16 / 4 | Baseline & GPTQ | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Optional third family (Tier C) |

### 2.1 Checkpoint Provenance Audit
Because pre-quantized checkpoints originate from different publishers, differences between GPTQ, AWQ, and NF4 conflate algorithmic properties with publisher choices (calibration set, group size, act-order). We explicitly log and audit these parameters in `configs/checkpoints.yaml`:
- Repository ID and pinned commit hash
- Publisher identity (official vendor vs. community contributor)
- Bit-width and quantization format
- Group size ($128$ vs $32$) and symmetric/asymmetric scheme
- Act-order (`desc_act: true/false`)
- Calibration dataset and sample sequence length
- License and disk footprint

---

## 3. Benchmarks & Evaluation Protocol

Evaluations are conducted via EleutherAI's pinned `lm-evaluation-harness` wrapper:

| Task Name | Capability Axis | Primary Metric | Few-Shot | Sample Count ($N$) | Approx. 1-SE ($p \approx 0.4$) |
|:---|:---|:---|:---:|---:|---:|
| `wikitext` | Language Modeling (Proxy) | `word_perplexity` | 0-shot | Document-level | — |
| `c4` *(Control)* | Out-of-Domain PPL | `word_perplexity` | 0-shot | 1,000 samples | — |
| `arc_easy` | Factual Recall | `acc_norm` | 0-shot | 2,376 | $\sim 1.0\text{ pp}$ |
| `hellaswag` | Commonsense Reasoning | `acc_norm` | 0-shot | 10,042 | $\sim 0.5\text{ pp}$ |
| `winogrande` | Coreference Resolution | `acc` | 0-shot | 1,267 | $\sim 1.4\text{ pp}$ |
| `mmlu` | Broad Domain Knowledge | `acc` | 5-shot | 14,042 | $\sim 0.4\text{ pp}$ |
| `gsm8k` | Multi-step Arithmetic | `exact_match` | 5-shot | 1,319 | $\sim 1.3\text{ pp}$ |
| `ifeval` | Instruction Following | `prompt_level_strict_acc` | 0-shot | 541 | $\sim 2.1\text{ pp}$ |

### 3.1 Pinned Protocol Rules
1. **Chat Template Handling:** Both models are *Instruct* checkpoints. Multiple-choice log-likelihood tasks (`arc_easy`, `hellaswag`, `winogrande`, `mmlu`) evaluate unprompted raw option likelihoods. Generative tasks (`gsm8k`, `ifeval`) apply the official chat template.
2. **Decoding Parameters:** Greedy decoding ($\text{temperature} = 0$, `top_p = 1.0`) with fixed `max_new_tokens` (GSM8K: 512, IFEval: 1024).
3. **GSM8K Filter Reporting:** Report both `strict-match` (verbatim `#### [number]`) and `flexible-extract` (regex match). The divergence isolates syntax/format compliance from arithmetic ability.
4. **IFEval Reporting:** Primary is `prompt-level strict accuracy`. Secondary metrics logged are `instruction-level strict` and `prompt-level loose`.
5. **Sample Logging:** All runs execute with `--log_samples` to save item-level records (`results/raw/`) enabling paired-sample statistical tests and failure-mode analysis.
6. **Precision & Compute:** Use `fp16` across all baselines and BitsAndBytes compute dtypes on NVIDIA T4 GPU.

---

## 4. Repository Architecture

```
final-year-btp/
├── BUILD_PLAN.md               ← Master execution blueprint (v2)
├── README.md                   ← Project overview, setup, and research framing
├── PROJECT_LOG.md              ← Running chronological engineering log
├── requirements.txt            ← Pinned dependencies and backend notes
├── run_experiments.py          ← Primary evaluation pipeline CLI
├── run_analysis.py             ← Statistical analysis CLI (bootstrap, proxy, budget)
├── configs/
│   ├── eval_config.yaml        ← Pinned evaluation matrix and task settings
│   └── checkpoints.yaml        ← Provenance audit table with commit hashes
├── src/
│   ├── model_loader.py         ← Hugging Face, BnB, GPTQ, AWQ loader
│   ├── evaluator.py            ← lm-eval wrapper with sample logging
│   ├── metrics/
│   │   ├── calibration.py      ← Expected Calibration Error (ECE) and Brier Score
│   │   └── efficiency.py       ← Peak VRAM, disk footprint, and token latency
│   └── analysis/
│       ├── degradation.py      ← Paired bootstrap CIs (10k iterations) on ΔAcc
│       ├── spearman_proxy.py   ← Spearman ρ with bootstrap CI and LOO analysis
│       ├── bit_budget.py       ← Noise-aware acceptance rule and sensitivity
│       └── failure_modes.py    ← Item-level flips, GSM8K syntax vs logic, IFEval
├── results/
│   ├── raw/                    ← Untouched JSON outputs and per-sample logs
│   ├── results_master.csv      ← Master aggregate tabular results
│   └── *.md                    ← Generated reports (degradation, proxy, budget)
├── docs/                       ← Research specifications and literature surveys
├── papers/                     ← 9 PTQ papers + 6 Stanford CRFM HELM papers
└── synopsis/                   ← LaTeX synopsis for DTU academic submission
```

---

## 5. HELM-Inspired Multi-Metric Layer

In alignment with Stanford CRFM HELM, we supplement pure task accuracy with two cheap, high-value metrics enabled by real packed checkpoints:

### 5.1 Calibration (Confidence Reliability)
- **Expected Calibration Error (ECE):** Evaluated on multiple-choice benchmarks (`arc_easy`, `hellaswag`, `winogrande`, `mmlu`). Given predicted option probabilities $P(\hat{Y} = y | X)$, predictions are partitioned into $M=10$ equal bins $B_m \subset (0, 1]$:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- **Brier Score:** Mean squared error between predicted probabilities and one-hot true outcomes:
  $$\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (p_{ik} - y_{ik})^2$$

### 5.2 Efficiency (Deployment Feasibility)
- **Memory Footprint:** Pinned checkpoint size on disk (MB) and peak GPU allocation via `torch.cuda.max_memory_allocated()`.
- **Inference Throughput:** Latency (ms) and throughput (tokens/second) measured during greedy decoding under a standardized 128-token prompt generating 64 tokens.

---

## 6. Statistical Analysis Plan

### 6.1 Task Degradation with Paired Bootstrap (RQ1)
- For each variant $v$, baseline $b$, and task $t$:
  $$\Delta\text{Acc}_{v, t} = \text{Acc}_{v, t} - \text{Acc}_{b, t}$$
- Since both models are evaluated on the exact same test items $i \in \{1, \dots, N\}$, we compute **Paired Bootstrap Confidence Intervals** (10,000 resamples):
  $$\delta^{(b)} = \frac{1}{N} \sum_{i=1}^N \left( \mathbb{I}(y_{v}^{(i)} = y^{*(i)}) - \mathbb{I}(y_{b}^{(i)} = y^{*(i)}) \right)$$
- Percentile bounds yield the empirical 95% CI: $[\Delta\text{Acc}_{0.025}, \Delta\text{Acc}_{0.975}]$.
- Relative drop is reported alongside absolute drop: $\Delta_{\text{rel}} = \frac{\Delta\text{Acc}}{\text{Acc}_{\text{FP16}}}$.

### 6.2 Perplexity Proxy Reliability (RQ2)
- Spearman rank correlation $\rho(\Delta\text{PPL}, \Delta\text{Acc})$ is computed across variants per task.
- **Bootstrap CIs:** 95% bootstrap confidence interval on $\rho$ using 2,000 resamples.
- **Leave-One-Out (LOO) Diagnostic:** Systematically drops each variant to verify whether an extreme point drives the correlation.
- **Pre-Registered Classification Criteria:**
  - **Faithful Proxy:** $\rho \le -0.70$ with 95% CI upper bound $< 0$.
  - **Decoupled:** $|\rho| < 0.40$ or 95% CI spanning zero.
  - **Intermediate:** $-0.70 < \rho \le -0.40$.
- **Dual-Corpus Control:** Compare WikiText-2 PPL correlations with an uncalibrated C4 corpus subset to detect calibration contamination.

### 6.3 Noise-Aware Bit-Budget Rule (RQ3)
Given finite sample sizes (especially IFEval with $N=541$ and $\text{SE} \approx 2.1\text{ pp}$), acceptance cannot depend on a noisy raw point estimate:
- **Acceptable ($\checkmark$):** The upper bound of accuracy loss is $\le 3.0\text{ pp}$ ($-\Delta\text{Acc}_{0.025} \le 3.0$).
- **Marginal ($!$):** The point estimate drop is $\le 3.0\text{ pp}$, but the 95% CI upper bound exceeds $3.0\text{ pp}$ ($-\Delta\text{Acc}_{0.025} > 3.0$).
- **Unacceptable ($\times$):** The point estimate drop strictly exceeds $3.0\text{ pp}$.
- **Sensitivity Sweep:** Repeat evaluation across thresholds $\{1.0, 2.0, 3.0, 5.0\}\text{ pp}$.

### 6.4 Failure Mode Analysis
- **Flip Dynamics:** Track items correct in FP16 but corrupted in quantized form ($1 \to 0$, regression) vs. items corrupted in FP16 but correct in quantized form ($0 \to 1$, spurious gain).
- **GSM8K Error Disentanglement:** Calculate gap between strict exact-match and flexible regex extraction to quantify formatting deterioration vs. arithmetic reasoning corruption.
- **IFEval Constraint Breakdown:** Compute error rates across instruction rule types (e.g., word count, punctuation, casing, response format).

---

## 7. Roadmap & Deliverables (Re-based 30 Oct 2026)

| Phase | Window | Focus / Deliverable | Priority Tier |
|:---|:---|:---|:---:|
| **Phase A** | Completed | Cleaned architecture, modular runner, test harnesses, HELM papers | **Must** |
| **Phase B+** | 30 Oct – 3 Nov | Complete `checkpoints.yaml` audit; load validation on T4; benchmark timing test | **Must** |
| **Phase B** | 30 Oct – 6 Nov | Run FP16 baselines for Qwen2.5-0.5B and Llama-3.2-1B; validate against model cards | **Must** |
| **Phase C** | 3 Nov – 12 Nov | Core runs: 8-bit & 4-bit variants on `wikitext`, `arc_easy`, `hellaswag`, `winogrande`, `mmlu`. Deliverable for **Mid-Semester Review**. | **Must** |
| **Phase D** | 9 Nov – 29 Nov | Generative runs: `gsm8k` and `ifeval` across all variants. Full results matrix. | **Must** |
| **Phase D2** | 16 Nov – 29 Nov | Multi-metric execution: Calibration (ECE/Brier) and Efficiency (VRAM/throughput). | **Should** |
| **Phase E** | 16 Nov – 6 Dec | Statistical analysis: Paired bootstrap CIs, Spearman $\rho$ with CIs, failure flips. | **Must** |
| **Phase F′** | 23 Nov – 6 Dec | Uncertainty & reproducibility verification: few-shot exemplar sensitivity, batch tests. | **Should** |
| **Phase G** | 30 Nov – 13 Dec | Decision matrix: Noise-aware bit-budget guidelines and sensitivity sweep. | **Must** |
| **Phase H** | 30 Nov – 18 Dec | Final BTP Dissertation write-up and arXiv-style research paper submission. | **Must** |
| **Phase I** | 7 Dec – 18 Dec | Viva defense demonstration, slide deck, and repository archiving. | **Must** |

---

## 8. Compute Budget & Execution Strategy

- **Primary Resource:** Kaggle GPU Free Tier (NVIDIA T4 16GB, ~30 GPU-hours/week quota).
- **Backup Resource:** Google Colab Free Tier (T4 GPU).
- **Local Mac (Apple Silicon / CPU):** Development, offline analysis, unit testing, and `--limit 50` FP16 smoke tests.

### Realistic Runtime Estimates
- Fast loglikelihood tasks (`wikitext`, `arc_easy`, `hellaswag`, `winogrande`): $\sim 10\text{--}15$ minutes per model variant.
- Domain knowledge (`mmlu`, 14k questions, 5-shot): $\sim 30\text{--}40$ minutes per variant.
- Generative reasoning (`gsm8k`, `ifeval`, greedy decoding): $\sim 25\text{--}40$ minutes per variant.
- **Total per variant:** $\sim 1.2\text{--}1.8$ GPU-hours.
- **Full 11-variant grid:** $\approx 15\text{--}20$ total GPU-hours, easily achievable within 1–2 Kaggle weekly allocations.

---

## 9. Environment, Dependencies & Known Risks

### Environment Notes
- Pinned `lm-evaluation-harness==0.4.3`.
- NVIDIA T4 lacks hardware `bfloat16`; all compute and loading operations use `torch.float16`.
- AutoAWQ has entered maintenance/deprecation; fallback loader relies on Hugging Face native integration or pinned backend.
- `meta-llama/Llama-3.2-1B-Instruct` is gated; requires Hugging Face user access token (`HF_TOKEN`).

### Risk Register

| Risk | Probability | Impact | Mitigation Strategy |
|:---|:---:|:---:|:---|
| Dependency mismatch between `lm-eval` and `transformers` | Medium | High | Pin compatible transformer runtime; lock versions in `requirements.txt`. |
| Floor effects on GSM8K for 0.5B model | High | Medium | Report relative drops and strict vs. flexible extraction metrics; analyze failure flips. |
| Checkpoint publisher differences confound algorithms | High | Medium | Explicit provenance audit in `checkpoints.yaml`; frame findings as empirical checkpoint evaluations. |
| Spearman $\rho$ instability over small $N$ | High | Medium | Pair with bootstrap CIs, leave-one-out sensitivity, and dual PPL corpora. |

---

## 10. Supervisor Alignment Questions

1. **Controlled Quantization Supplement:** If time permits, does the supervisor approve running a single controlled quantization sweep on `Qwen2.5-0.5B` using `optimum` to control for publisher differences?
2. **HELM Metric Scope:** Confirm agreement on reporting ECE calibration and throughput efficiency alongside standard accuracy.
3. **Threshold Flexibility:** Confirm whether the $3\text{ pp}$ acceptable threshold is a fixed departmental criterion or open to sensitivity sweeps ($1\text{--}5\text{ pp}$).
