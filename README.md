# Task-Conditioned Evaluation of Quantized Small LLMs: Accuracy Degradation, the Reliability of Perplexity as a Proxy, and Practical Bit-Budget Guidelines

### Delhi Technological University | B.Tech Major Project (Session 2026–27)
**Group ID:** `DTU/CSE/BTP/GP159`  
**Supervising Faculty:** Ms. Gull Kaur, Assistant Professor, Department of Computer Science & Engineering  
**Team Members:**
- Ekansh Agarwal (`23/CS/149`)
- Faqre Alam (`23/CS/150`)
- Guneet Toppo (`23/CS/160`)

---

## 1. Project Overview & Research Problem

Post-training quantization (PTQ) compresses large language model weights from 16-bit floating point down to 8-bit or 4-bit integers to reduce memory-bandwidth pressure on consumer and edge hardware. In compression literature, **Perplexity (PPL)** is frequently reported as a primary proxy metric because it is cheap and fast to evaluate. Survey papers (e.g., Jin et al., ACL Findings 2024) have suggested that perplexity correlates well with general benchmark performance.

However, modern edge deployments demand specialized capabilities: exact arithmetic computation, multi-step logical deduction, and strict constraint adherence. Relying on an aggregate language-modeling loss may mask severe, localized capability degradation in reasoning or instruction compliance. Furthermore, prior empirical studies focus predominantly on models $\ge 3\text{B}$ parameters. Small sub-1B models (`Qwen2.5-0.5B-Instruct` and `Llama-3.2-1B-Instruct`) that realistically run on edge devices remain under-studied.

### The Three Core Research Questions (RQs)
1. **RQ1 (Task-Conditioned Degradation):** How much does post-training quantization degrade distinct capability axes (factual recall, commonsense reasoning, domain knowledge, multi-step arithmetic, and instruction following), quantified via **Paired Bootstrap 95% Confidence Intervals (10,000 iterations)**?
2. **RQ2 (Reliability of Perplexity as a Proxy):** Does perplexity track accuracy degradation across tasks, or does it hide localized damage? Evaluated using **Spearman's rank correlation ($\rho$)** with bootstrap CIs, leave-one-out sensitivity diagnostics, and a secondary out-of-domain control corpus (C4).
3. **RQ3 (Practical Bit-Budget Guidelines):** What is the minimum acceptable bit-width for each task type under a noise-aware acceptance criterion ($\le 3.0\text{ pp}$ loss bound)?

---

## 2. Models & Quantization Formats

> **Specification Rule:** Per the approved departmental problem statement, we write **no custom quantization kernels**. We empirically evaluate official vendor and community released Hugging Face checkpoints.

| Model Family | Variant | Target Bits | Method | Checkpoint Repository ID | Publisher |
|:---|:---|:---:|:---|:---|:---:|
| **Qwen2.5-0.5B-Instruct** | `fp16` | 16 | Baseline | `Qwen/Qwen2.5-0.5B-Instruct` | Official |
| | `bnb_int8` | 8 | BitsAndBytes | `Qwen/Qwen2.5-0.5B-Instruct` (`load_in_8bit`) | Official |
| | `gptq_int8` | 8 | GPTQ | `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8` | Official |
| | `bnb_nf4` | 4 | BitsAndBytes | `Qwen/Qwen2.5-0.5B-Instruct` (NF4) | Official |
| | `gptq_int4` | 4 | GPTQ | `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4` | Official |
| | `awq_int4` | 4 | AWQ | `Qwen/Qwen2.5-0.5B-Instruct-AWQ` | Official |
| **Llama-3.2-1B-Instruct** | `fp16` | 16 | Baseline | `meta-llama/Llama-3.2-1B-Instruct` | Official (Gated) |
| | `bnb_int8` | 8 | BitsAndBytes | `meta-llama/Llama-3.2-1B-Instruct` (`load_in_8bit`) | Official |
| | `bnb_nf4` | 4 | BitsAndBytes | `meta-llama/Llama-3.2-1B-Instruct` (NF4) | Official |
| | `gptq_int4` | 4 | GPTQ | `shuyuej/Llama-3.2-1B-Instruct-GPTQ` | Community |
| | `awq_int4` | 4 | AWQ | `ciCic/llama-3.2-1B-Instruct-AWQ` | Community |

All checkpoint provenance details (group size, symmetric/asymmetric scheme, act-order, calibration sets, commit hashes) are formally audited in [`configs/checkpoints.yaml`](configs/checkpoints.yaml).

---

## 3. Benchmark Suite & Evaluation Protocol

Evaluations are conducted through a single pinned harness (`lm-evaluation-harness==0.4.3`) to guarantee identical prompts, demonstrator shots, and scoring rules:

| Task Name | Capability Axis | Primary Metric | Few-Shot | Sample Size ($N$) | Approx. 1-SE |
|:---|:---|:---|:---:|---:|---:|
| `wikitext` | Language Modeling (Primary Proxy) | `word_perplexity` | 0-shot | Document-level | — |
| `c4` | Language Modeling (Control Proxy) | `word_perplexity` | 0-shot | 1,000 samples | — |
| `arc_easy` | Factual Recall | `acc_norm` | 0-shot | 2,376 | $\sim 1.0\text{ pp}$ |
| `hellaswag` | Commonsense Reasoning | `acc_norm` | 0-shot | 10,042 | $\sim 0.5\text{ pp}$ |
| `winogrande` | Coreference Resolution | `acc` | 0-shot | 1,267 | $\sim 1.4\text{ pp}$ |
| `mmlu` | Broad Domain Knowledge | `acc` | 5-shot | 14,042 | $\sim 0.4\text{ pp}$ |
| `gsm8k` | Multi-step Arithmetic | `exact_match` (strict & flexible) | 5-shot | 1,319 | $\sim 1.3\text{ pp}$ |
| `ifeval` | Instruction Following | `prompt_level_strict_acc` | 0-shot | 541 | $\sim 2.1\text{ pp}$ |

### Stanford CRFM HELM-Inspired Multi-Metric Layer
In alignment with Stanford CRFM HELM principles (Liang et al., TMLR 2023), we augment accuracy with two empirical dimensions:
1. **Calibration (ECE & Brier Score):** Evaluated over multiple-choice option likelihoods to detect whether 4-bit quantization damages confidence reliability (`src/metrics/calibration.py`).
2. **Operational Efficiency:** Measures disk footprint (MB), peak VRAM allocated (`torch.cuda.max_memory_allocated`), inference latency, and tokens/sec throughput (`src/metrics/efficiency.py`).

---

## 4. Repository Structure

```
final-year-btp/
├── BUILD_PLAN.md               # Master Build & Execution Plan v2
├── README.md                   # Project overview, specification & usage guide
├── PROJECT_LOG.md              # Running chronological engineering log
├── requirements.txt            # Pinned reproducible dependencies
├── run_experiments.py          # Primary evaluation orchestrator CLI
├── run_analysis.py             # Statistical analysis CLI (bootstrap CIs, Spearman, budget)
│
├── configs/
│   ├── eval_config.yaml        # Evaluation grid, tasks, and runtime parameters
│   └── checkpoints.yaml        # Checkpoint provenance audit table
│
├── src/
│   ├── model_loader.py         # Universal model loader (FP16, BnB, GPTQ, AWQ)
│   ├── evaluator.py            # lm-evaluation-harness wrapper with sample logging
│   ├── metrics/
│   │   ├── calibration.py      # ECE and Brier score calculators
│   │   └── efficiency.py       # VRAM, disk footprint, and token latency profiler
│   └── analysis/
│       ├── degradation.py      # Paired bootstrap CIs (10k iterations) on ΔAcc
│       ├── spearman_proxy.py   # Spearman ρ with bootstrap CI and LOO diagnostic
│       ├── bit_budget.py       # Noise-aware acceptance rule and sensitivity sweep
│       └── failure_modes.py    # Item-level flip dynamics and GSM8K/IFEval error analysis
│
├── results/
│   ├── raw/                    # Untouched JSON outputs and per-sample logs
│   ├── results_master.csv      # Primary experimental results table
│   └── *.md                    # Generated analytical markdown reports
│
├── docs/                       # Research specifications, HELM deep-dive, and notes
├── papers/
│   ├── 1_GPTQ.pdf ...          # 9 Foundational PTQ literature papers
│   └── helm/                   # 6 Stanford CRFM HELM benchmark papers
│
└── synopsis/
    └── main.tex                # Formal DTU synopsis LaTeX source
```

---

## 5. Quick Start & Execution Commands

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Verify planned grid without compute (Dry Run)
```bash
python run_experiments.py --dry-run
```

### 3. Local smoke test (Mac / CPU, 50 samples per task)
```bash
python run_experiments.py --model qwen2.5-0.5b --variant fp16 --tasks wikitext,arc_easy --limit 50
```

### 4. GPU execution runs (Kaggle T4 / Google Colab)

> **Hugging Face Authentication:** Evaluating `Llama-3.2-1B-Instruct` requires a Hugging Face User Access Token (due to Meta's gated model agreement).
> - **In Kaggle:** Go to **Add-ons** ➔ **Secrets**, add `HF_TOKEN` with your read access token, and attach it to the notebook.
> - **In Terminal / Colab:** Run `export HF_TOKEN="hf_..."` or `huggingface-cli login`.
> - See [kaggle_evaluation_runner.ipynb](notebooks/kaggle_evaluation_runner.ipynb) for the complete visual step-by-step token setup guide.

```bash
# Run baseline models
python run_experiments.py --variant fp16

# Run complete Qwen2.5-0.5B-Instruct grid
python run_experiments.py --model qwen2.5-0.5b

# Run complete Llama-3.2-1B-Instruct grid
python run_experiments.py --model llama-3.2-1b
```

### 5. Statistical Analysis & Decision Reports
```bash
python run_analysis.py --csv results/results_master.csv --bootstrap 10000
```
This generates:
- `results/degradation_report.md` (ΔAcc with 95% paired bootstrap CIs)
- `results/spearman_proxy_report.md` (Spearman rank correlation, bootstrap CIs, LOO range)
- `results/bit_budget_guidelines.md` (Noise-aware minimum precision matrix and sensitivity sweep)

---

## 6. Academic References & Literature Base

The repository includes a curated library of **15 peer-reviewed papers** in `papers/`:
- **Quantization:** GPTQ (Frantar et al., ICLR 2023), AWQ (Lin et al., MLSys 2024), QLoRA / NF4 (Dettmers et al., NeurIPS 2023), Jin et al. (ACL Findings 2024), Li et al. (ICML 2024).
- **Evaluation & HELM:** HELM (Liang et al., TMLR 2023), DecodingTrust (Wang et al., NeurIPS 2023 Outstanding Paper), VHELM (Hasson et al., 2024), HEIM (Lee et al., ICML 2024), AIR-Bench (2024), Foundation Models (Bommasani et al., 2021).
