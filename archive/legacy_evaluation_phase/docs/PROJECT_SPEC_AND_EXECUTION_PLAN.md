# Official Project Specification & Execution Blueprint
**Department of Computer Science & Engineering, Delhi Technological University**  
**B.Tech Major Project, Session 2026–27**

---

## 1. Project Identity & Group Details

- **Project Title:** Task-Conditioned Evaluation of Quantized Large Language Models: Accuracy Degradation, the Reliability of Perplexity as a Proxy, and Practical Bit-Budget Guidelines
- **Group ID:** `DTU/CSE/BTP/GP159`
- **Supervisor:** Ms. Gull Kaur, Assistant Professor, Department of CSE, DTU
- **Group Members:**
  1. Ekansh Agarwal (Roll No: 23/CS/149)
  2. Faqre Alam (Roll No: 23/CS/150)
  3. Guneet Toppo (Roll No: 23/CS/160)

---

## 2. Core Problem Statement & Research Questions

Post-Training Quantization (PTQ) down to 8-bit and 4-bit enables LLMs to run on edge and consumer hardware. Prior foundational studies (e.g., Jin et al., ACL 2024 Findings) concluded that **Perplexity (PPL)** is a cheap, reliable proxy for general model performance. However:
1. That evidence was collected before multi-step reasoning benchmarks (**GSM8K**) and instruction-following benchmarks (**IFEval**) became standard.
2. Prior studies focus predominantly on large models ($\ge 3\text{B}$) and single quantization formats, overlooking edge-viable models in the **$0.5\text{B}$ to $1\text{B}$ parameter range**.

### The Three Research Questions (RQs):
- **RQ1 (Task-Conditioned Degradation):** How much does quantization hurt each specific capability (factual recall, commonsense reasoning, multi-step arithmetic, and instruction following) across each model and method?
- **RQ2 (Reliability of Perplexity as a Proxy):** Does perplexity track accuracy after quantization, or does it mask severe localized degradation? Evaluated via **Spearman's rank correlation ($\rho$)** between $\Delta\text{PPL}$ and $\Delta\text{Acc}_t$.
- **RQ3 (Practical Bit-Budget Guidelines):** How many bits are sufficient for each task type? Evaluated using our **$\le 3\text{ percentage points}$ degradation criterion** compared to FP16.
- **Specific Hypothesis:** Does GSM8K hold up better under subtle quantization changes than tasks requiring broad reading context or commonsense reasoning?

---

## 3. Experimental Design

### 3.1 Models & Quantization Variants Evaluated
As explicitly mandated by the approved Problem Statement:
> *"We do not write any quantization code. We use official or community checkpoints from Hugging Face: Qwen2.5-0.5B-Instruct with the official GPTQ-Int4 and GPTQ-Int8 releases, Llama-3.2-1B-Instruct with community GPTQ-4bit and AWQ-4bit versions, NF4 and Int8 through bitsandbytes flags, and optionally TinyLlama-1.1B GPTQ."*

| Model Family | Variant Name | Precision | Quantization Method | Loader / Checkpoint Source |
|:---|:---|:---:|:---:|:---|
| **Qwen2.5-0.5B-Instruct** | `fp16` | 16-bit | Baseline FP16 | `Qwen/Qwen2.5-0.5B-Instruct` |
| | `bnb_int8` | 8-bit | BitsAndBytes Int8 | `Qwen/Qwen2.5-0.5B-Instruct` (`load_in_8bit=True`) |
| | `gptq_int8` | 8-bit | GPTQ | `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8` (Official) |
| | `bnb_nf4` | 4-bit | BitsAndBytes NF4 | `Qwen/Qwen2.5-0.5B-Instruct` (`bnb_4bit_quant_type="nf4"`) |
| | `gptq_int4` | 4-bit | GPTQ | `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4` (Official) |
| | `awq_int4` | 4-bit | AWQ | `Qwen/Qwen2.5-0.5B-Instruct-AWQ` |
| **Llama-3.2-1B-Instruct** | `fp16` | 16-bit | Baseline FP16 | `meta-llama/Llama-3.2-1B-Instruct` |
| | `bnb_int8` | 8-bit | BitsAndBytes Int8 | `meta-llama/Llama-3.2-1B-Instruct` (`load_in_8bit=True`) |
| | `bnb_nf4` | 4-bit | BitsAndBytes NF4 | `meta-llama/Llama-3.2-1B-Instruct` (`bnb_4bit_quant_type="nf4"`) |
| | `gptq_int4` | 4-bit | GPTQ | `shuyuej/Llama-3.2-1B-Instruct-GPTQ` |
| | `awq_int4` | 4-bit | AWQ | `ciCic/llama-3.2-1B-Instruct-AWQ` |
| **TinyLlama-1.1B** *(Optional)* | `fp16`, `gptq` | 16 / 4 bit | Baseline / GPTQ | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |

### 3.2 Evaluation Benchmark Suite (6 Tasks + Perplexity)
All evaluations are executed using a single pinned version of **`lm-evaluation-harness==0.4.3`**:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        EVALUATION BENCHMARK SUITE                      │
├────────────────────┬────────────────────────────────┬──────────────────┤
│ Task ID            │ Capability Category            │ Metric Measured  │
├────────────────────┼────────────────────────────────┼──────────────────┤
│ 1. wikitext        │ Language Modeling (Proxy)      │ word_perplexity  │
│ 2. arc_easy        │ Factual Knowledge / Recall     │ acc_norm (0-shot)│
│ 3. hellaswag       │ Commonsense Reasoning          │ acc_norm (0-shot)│
│ 4. winogrande      │ Coreference / Reasoning        │ acc (0-shot)     │
│ 5. mmlu            │ Academic / Domain Knowledge    │ acc (5-shot)     │
│ 6. gsm8k           │ Multi-Step Arithmetic (Math)   │ exact_match (CoT)│
│ 7. ifeval          │ Instruction Following          │ strict_acc       │
└────────────────────┴────────────────────────────────┴──────────────────┘
```

---

## 4. Software Architecture & Implementation

The project code is organized cleanly into modular components:

```
final-year-btp/
├── configs/
│   └── eval_config.yaml         ← Central config for all models, methods, and tasks
├── src/
│   ├── model_loader.py          ← Universal loader for FP16, BnB (Int8/NF4), GPTQ, AWQ
│   ├── evaluator.py             ← Pinned lm-eval interface & metric extraction
│   └── analysis/
│       ├── degradation.py       ← Task degradation calculator (ΔAcc, ΔPPL)
│       ├── spearman_proxy.py    ← Spearman correlation (ρ) analysis for RQ2
│       └── bit_budget.py        ← Practical bit-budget guidelines (<= 3 pp rule for RQ3)
├── run_experiments.py           ← Master CLI execution script
├── run_analysis.py              ← Master analysis script generating markdown reports
├── requirements.txt             ← Pinned dependencies
├── synopsis/main.tex            ← Official DTU synopsis LaTeX document
└── results/
    ├── results_master.csv       ← Primary experimental results log
    ├── degradation_report.md    ← Formatted degradation tables
    ├── spearman_proxy_report.md ← Spearman correlation findings
    └── bit_budget_guidelines.md ← Bit-budget guidelines matrix
```

---

## 5. Execution Commands & Quickstart

### Step 1: Inspect Planned Grid (Dry Run)
```bash
python run_experiments.py --dry-run
```

### Step 2: Quick Smoke Test (50 samples per task on CPU or Mac MPS)
```bash
python run_experiments.py --model qwen2.5-0.5b --variant fp16 --tasks wikitext,arc_easy --limit 50
```

### Step 3: Run Full Evaluation Grid on GPU (Kaggle T4 / Colab)
```bash
python run_experiments.py
```

### Step 4: Run Statistical Analysis & Generate Deliverables
```bash
python run_analysis.py --csv results/results_master.csv
```

---

## 6. Academic Calendar & Milestones (Gantt Alignment)

- **Weeks 1–2 (Oct 12–25):** Setup Kaggle/Colab, pin `lm-evaluation-harness`, author CLI scripts, dry-run on local machine. *(Completed)*
- **Weeks 2–3 (Oct 19–Nov 1):** Execute FP16 baselines for Qwen and Llama-3.2; verify numbers against model cards.
- **Weeks 3–4 (Oct 26–Nov 8):** Core benchmark runs (MMLU, ARC-Easy, HellaSwag, WinoGrande, WikiText-2); prepare mid-semester report.
- **Weeks 5–7 (Nov 9–Nov 29):** Full grid expansion adding GSM8K and IFEval across all bit-widths.
- **Weeks 6–8 (Nov 16–Dec 6):** Statistical analysis (Spearman $\rho$ matrix, degradation curves).
- **Weeks 7–8 (Nov 23–Dec 6):** 3-seed variability runs for selected configurations ($\text{mean} \pm \text{std}$).
- **Weeks 8–9 (Nov 30–Dec 13):** Apply $\le 3\text{ pp}$ rule to draft final Bit-Budget Guidelines.
- **Weeks 9–10 (Dec 7–Dec 18):** Final BTech Dissertation authoring, viva preparation, and submission.

---

*Compiled by Antigravity on behalf of DTU Group GP159: Ekansh Agarwal, Faqre Alam, and Guneet Toppo. Guide: Ms. Gull Kaur.*

