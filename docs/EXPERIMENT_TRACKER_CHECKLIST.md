# Project Tracking & Experiment Execution Checklist
## Task-Conditioned Evaluation of Quantized Small LLMs (DTU Major Project BTP/GP159)

**Group ID:** `DTU/CSE/BTP/GP159`  
**Supervising Faculty:** Ms. Gull Kaur, Assistant Professor, Department of CSE, DTU  
**Team Members:** Ekansh Agarwal (`23/CS/149`), Faqre Alam (`23/CS/150`), Guneet Toppo (`23/CS/160`)  
**Last Updated:** 30 Oct 2026

---

## 1. High-Level Phase Roadmap

| Phase | Description | Window | Priority | Status |
|:---:|:---|:---:|:---:|:---:|
| **Phase A** | Setup, clean architecture, unified CLIs, HELM papers integration | Prior – 30 Oct | Must | **[COMPLETED]** |
| **Phase B+** | Checkpoint provenance audit, HF commit pinning, T4 load validation | 30 Oct – 3 Nov | Must | **[COMPLETED]** |
| **Phase B** | FP16 baselines for Qwen2.5-0.5B & Llama-3.2-1B; model-card cross-check | 30 Oct – 6 Nov | Must | **[READY TO RUN]** |
| **Phase C** | Core runs: 8-bit & 4-bit on `wikitext`, `c4`, `arc_easy`, `hellaswag`, `winogrande`, `mmlu` | 3 Nov – 12 Nov | Must | **[PENDING]** |
| **Phase D** | Generative runs: `gsm8k` & `ifeval` across all 11 model configurations | 9 Nov – 29 Nov | Must | **[PENDING]** |
| **Phase D2** | HELM multi-metric layer: Confidence Calibration (ECE/Brier) & Efficiency (VRAM/tokens/s) | 16 Nov – 29 Nov | Should | **[CODE READY]** |
| **Phase E** | Statistical analysis: 10,000 paired bootstrap CIs, Spearman $\rho$, failure-mode flips | 16 Nov – 6 Dec | Must | **[CODE READY]** |
| **Phase F′** | Uncertainty & reproducibility verification: few-shot exemplar sensitivity, batch tests | 23 Nov – 6 Dec | Should | **[PENDING]** |
| **Phase G** | Noise-aware bit-budget guidelines ($\le 3.0\text{ pp}$ bound) & sensitivity sweep | 30 Nov – 13 Dec | Must | **[CODE READY]** |
| **Phase H** | Major Project Dissertation write-up & arXiv-style research draft | 30 Nov – 18 Dec | Must | **[PENDING]** |
| **Phase I** | Viva defense demonstration, slide deck, and repository archiving | 7 Dec – 18 Dec | Must | **[PENDING]** |

---

## 2. Model & Checkpoint Provenance Audit (Phase B+)

Verify each repository ID, access permissions, commit hash, group size, and calibration data:

- [x] **Qwen2.5-0.5B-Instruct (FP16 Baseline)** — `Qwen/Qwen2.5-0.5B-Instruct` (Official, Commit: `7ae5576`, 953.3 MB)
- [x] **Qwen2.5-0.5B-Instruct (BnB Int8)** — `Qwen/Qwen2.5-0.5B-Instruct` (`load_in_8bit=True`, Commit: `7ae5576`)
- [x] **Qwen2.5-0.5B-Instruct (GPTQ Int8)** — `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8` (Official, Commit: `c68601e`, 621.0 MB, group size 128, sym: True)
- [x] **Qwen2.5-0.5B-Instruct (BnB NF4)** — `Qwen/Qwen2.5-0.5B-Instruct` (`bnb_4bit_quant_type="nf4"`, Commit: `7ae5576`)
- [x] **Qwen2.5-0.5B-Instruct (GPTQ Int4)** — `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4` (Official, Commit: `c34a4a9`, 449.1 MB, group size 128, sym: True)
- [x] **Qwen2.5-0.5B-Instruct (AWQ Int4)** — `Qwen/Qwen2.5-0.5B-Instruct-AWQ` (Official, Commit: `db09cd2`, 707.8 MB, group size 128, zero_point: True)
- [x] **Llama-3.2-1B-Instruct (FP16 Baseline)** — `meta-llama/Llama-3.2-1B-Instruct` (Official, Commit: `9213176`, 4,725.2 MB)
- [x] **Llama-3.2-1B-Instruct (BnB Int8)** — `meta-llama/Llama-3.2-1B-Instruct` (`load_in_8bit=True`, Commit: `9213176`)
- [x] **Llama-3.2-1B-Instruct (BnB NF4)** — `meta-llama/Llama-3.2-1B-Instruct` (`bnb_4bit_quant_type="nf4"`, Commit: `9213176`)
- [x] **Llama-3.2-1B-Instruct (GPTQ Int4)** — `shuyuej/Llama-3.2-1B-Instruct-GPTQ` (Community, Commit: `4024c85`, 993.3 MB, group size 128, `dataset: null` noted)
- [x] **Llama-3.2-1B-Instruct (AWQ Int4)** — `ciCic/llama-3.2-1B-Instruct-AWQ` (Community, Commit: `0412140`, 1,500.8 MB, group size 128)
- [x] *(Optional Tier C)* **TinyLlama-1.1B-Chat (FP16 / GPTQ)** — `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (Commit: `fe8a4ea`, 2,100.4 MB)

---

## 3. Evaluation Execution Grid ($11 \text{ Configurations} \times 8 \text{ Tasks}$)

### Model Family 1: `Qwen2.5-0.5B-Instruct`
| Variant | Precision | Wikitext (PPL) | C4 (Control PPL) | ARC-Easy (Recall) | HellaSwag (Common) | WinoGrande (Coref) | MMLU (Domain) | GSM8K (Math) | IFEval (Instruct) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `fp16` | 16-bit | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| `bnb_int8` | 8-bit | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| `gptq_int8`| 8-bit | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| `bnb_nf4`  | 4-bit | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| `gptq_int4`| 4-bit | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| `awq_int4` | 4-bit | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

### Model Family 2: `Llama-3.2-1B-Instruct`
| Variant | Precision | Wikitext (PPL) | C4 (Control PPL) | ARC-Easy (Recall) | HellaSwag (Common) | WinoGrande (Coref) | MMLU (Domain) | GSM8K (Math) | IFEval (Instruct) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `fp16` | 16-bit | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| `bnb_int8` | 8-bit | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| `bnb_nf4`  | 4-bit | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| `gptq_int4`| 4-bit | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| `awq_int4` | 4-bit | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

### Model Family 3: `TinyLlama-1.1B-Chat` (DTU Synopsis 3rd Baseline)
| Variant | Precision | Wikitext (PPL) | C4 (Control PPL) | ARC-Easy (Recall) | HellaSwag (Common) | WinoGrande (Coref) | MMLU (Domain) | GSM8K (Math) | IFEval (Instruct) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `fp16` | 16-bit | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| `bnb_int8` | 8-bit | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| `bnb_nf4`  | 4-bit | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| `gptq_int4`| 4-bit | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| `awq_int4` | 4-bit | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

---

## 4. Multi-Metric Evaluation & Profiling (Phase D2)

### Confidence Calibration (HELM Alignment)
- [ ] Extract multiple-choice option log-likelihoods for `arc_easy`
- [ ] Extract multiple-choice option log-likelihoods for `hellaswag`
- [ ] Extract multiple-choice option log-likelihoods for `winogrande`
- [ ] Extract multiple-choice option log-likelihoods for `mmlu`
- [ ] Compute Expected Calibration Error (ECE, $M=10$ bins) across all 11 variants
- [ ] Compute Brier Score across all 11 variants
- [ ] Assess whether 4-bit quantization damages confidence calibration relative to FP16

### Operational Efficiency Profiling (Hardware Reality)
- [ ] Measure exact on-disk checkpoint footprint (MB) for all 11 variants
- [ ] Profile peak GPU VRAM allocation (`torch.cuda.max_memory_allocated`) on NVIDIA T4
- [ ] Measure greedy generation latency (ms) under standardized 128-token prompt
- [ ] Calculate decoding throughput (tokens/sec) across batch sizes
- [ ] Empirically evaluate whether BitsAndBytes LLM.int8 introduces latency slowdown compared to FP16

---

## 5. Statistical Analysis & Decision Rules (Phase E & G)

- [ ] **Paired Bootstrap Confidence Intervals (RQ1):**
  - [ ] Resample 10,000 iterations over paired item differences
  - [ ] Compute 95% CIs: $[\Delta\text{Acc}_{0.025}, \Delta\text{Acc}_{0.975}]$
  - [ ] Calculate relative degradation $\Delta_{\text{rel}} = \Delta\text{Acc} / \text{Acc}_{\text{FP16}}$
  - [ ] Flag floor artifacts (e.g., baseline near chance)
- [ ] **Perplexity Proxy Reliability (RQ2):**
  - [ ] Compute Spearman rank correlation $\rho(\Delta\text{PPL}, \Delta\text{Acc})$ per capability axis
  - [ ] Calculate 2,000-iteration bootstrap CIs on $\rho$
  - [ ] Perform Leave-One-Out (LOO) influence diagnostics to detect single-point distortion
  - [ ] Contrast WikiText-2 PPL with out-of-domain C4 PPL to check calibration contamination
  - [ ] Categorize each task as **Faithful Proxy**, **Decoupled**, or **Intermediate**
- [ ] **Failure Mode Disentanglement:**
  - [ ] Quantify item-level flip rates ($1 \to 0$ regression vs $0 \to 1$ spurious gain)
  - [ ] Disentangle GSM8K formatting failures (no `####`) from arithmetic reasoning failures
  - [ ] Dissect IFEval constraint compliance across instruction categories
- [ ] **Noise-Aware Bit-Budget Decision Matrix (RQ3):**
  - [ ] Classify variants into **Acceptable** ($\le 3.0\text{ pp}$ bound), **Marginal**, or **Rejected**
  - [ ] Execute sensitivity analysis sweep across $\{1.0, 2.0, 3.0, 5.0\}\text{ pp}$
  - [ ] Produce final recommended bit-precision decision matrix per task

---

## 6. Academic Deliverables & Viva Milestones (Phase H & I)

- [ ] Complete Mid-Semester Review slide deck with Phase B & C results (November)
- [ ] Verify all citations against primary literature (eliminate unverified citations)
- [ ] Format LaTeX Major Project Dissertation following official DTU CSE department guidelines
- [ ] Produce publication-quality figures:
  - [ ] Task degradation bar charts with bootstrap error bars
  - [ ] Scatter plots of $\Delta\text{PPL}$ vs $\Delta\text{Acc}$ with regression lines
  - [ ] Calibration reliability diagrams (ECE confidence vs accuracy curves)
  - [ ] Efficiency Pareto frontier (Accuracy vs Memory/Latency)
- [ ] Prepare live evaluation demonstration script for external viva defense
- [ ] Archive repository with frozen dependencies, logs, and artifacts
