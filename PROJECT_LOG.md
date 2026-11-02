# Engineering Project Log
## Task-Conditioned Evaluation of Quantized Small LLMs (DTU Major Project BTP/GP159)

**Group ID:** `DTU/CSE/BTP/GP159`  
**Supervising Faculty:** Ms. Gull Kaur, Assistant Professor, Department of CSE, DTU  
**Team Members:** Ekansh Agarwal (`23/CS/149`), Faqre Alam (`23/CS/150`), Guneet Toppo (`23/CS/160`)

---

### [2026-10-30] — Major Project Initialization & Plan Enhancement (v2)
- **Reviewed DTU Approved Problem Statement:** Confirmed research scope focusing on empirical evaluation of released checkpoints (`Qwen2.5-0.5B-Instruct`, `Llama-3.2-1B-Instruct`) across FP16, 8-bit (GPTQ, BnB), and 4-bit (GPTQ, AWQ, BnB-NF4) without custom quantization code.
- **HELM Papers Integration:** Successfully acquired, verified, and indexed 6 Stanford CRFM HELM foundation papers in `papers/helm/`:
  1. `1_HELM_Holistic_Evaluation_of_Language_Models.pdf` (Liang et al., TMLR 2023)
  2. `2_DecodingTrust_Trustworthiness_and_Safety.pdf` (Wang et al., NeurIPS 2023 Outstanding Paper)
  3. `3_VHELM_Holistic_Evaluation_Vision_Language.pdf` (Hasson et al., Stanford CRFM 2024)
  4. `4_HEIM_Holistic_Evaluation_Text_to_Image.pdf` (Lee et al., ICML 2024)
  5. `5_AIR_Bench_2024_Safety_Benchmark.pdf` (Stanford CRFM 2024)
  6. `6_Opportunities_and_Risks_of_Foundation_Models.pdf` (Bommasani et al., Stanford CRFM 2021)
- **Repository Architecture Cleanup:** Permanently removed legacy hand-rolled prototype directory (`phase1/`), obsolete logs, and temporary scripts. Consolidated project around unified `src/`, `configs/`, `results/`, and `papers/`.
- **Master Execution Plan v2:**
  - Upgraded literature gap positioning with respect to Jin et al. (ACL Findings 2024).
  - Designed `configs/checkpoints.yaml` for pre-quantization provenance auditing.
  - Replaced deterministic 3-seed evaluation with 10,000-iteration Paired Bootstrap Confidence Intervals for accurate sampling error measurement.
  - Formulated noise-aware Bit-Budget acceptance rule ($\le 3\text{ pp}$ bound) with sensitivity analysis sweeps ($1, 2, 3, 5\text{ pp}$).
  - Introduced HELM-inspired multi-metric modules for Confidence Calibration (ECE, Brier Score) and Efficiency Profiling (disk, peak VRAM, throughput).
  - Formalized failure mode analysis for GSM8K formatting vs arithmetic errors and IFEval constraint categories.

---

### [2026-10-30] — Phase B+ Checkpoint Audit, Tracking Infrastructure & Pipeline Verification
- **Track Checklist Created:** Published [`docs/EXPERIMENT_TRACKER_CHECKLIST.md`](docs/EXPERIMENT_TRACKER_CHECKLIST.md) providing an 88-cell execution matrix, phase milestones, and viva deliverables.
- **Comprehensive Learning Compendium:** Authored [`docs/LEARNING_TRACKER.md`](docs/LEARNING_TRACKER.md), detailing PTQ mathematical derivations (GPTQ OBS update, AWQ channel scaling, NF4 normal quantiles, LLM.int8 mixed precision), HELM multi-metric theory (ECE, Brier, VRAM), 15-paper literature synthesis, and viva voce defense FAQs.
- **Automated Checkpoint Provenance Audit:**
  - Implemented and executed [`scripts/audit_checkpoints.py`](scripts/audit_checkpoints.py) against live Hugging Face Hub.
  - Successfully verified all checkpoints.
  - Pinned exact commit SHAs, file footprints, group sizes ($128$), and symmetry flags into [`configs/checkpoints.yaml`](configs/checkpoints.yaml).
- **Turnkey Kaggle T4 Runner Notebook:** Created [`notebooks/kaggle_evaluation_runner.ipynb`](notebooks/kaggle_evaluation_runner.ipynb) with pre-configured T4 GPU checks, dependency installations, phased execution commands, and inline Markdown analytical reporting.
- **End-to-End Integration Smoke Test:** Created and executed [`scripts/run_smoke_test.py`](scripts/run_smoke_test.py), verifying calibration (ECE/Brier), failure modes (flip dynamics, GSM8K format gaps), 10,000 paired bootstrap CIs, Spearman proxy analysis, and noise-aware bit-budget guidelines. All 6 test suites passed cleanly with exit code 0.

---

### [2026-10-30] — 3-Model Scope Expansion & Hybrid Compute Orchestration
- **Integrated Model Family 3 (`TinyLlama-1.1B-Chat`):** Added the 3rd comparative model specified in the approved DTU BTP Synopsis (`synopsis/main.tex`) across `configs/eval_config.yaml`, `configs/checkpoints.yaml`, and `docs/EXPERIMENT_TRACKER_CHECKLIST.md`. Evaluation grid expanded to 16 configurations ($3\text{ model families} \times 5\text{--}6\text{ variants}$).
- **TinyLlama Local Baseline Verified on Apple Silicon (MPS):**
  - Downloaded `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (2,100.4 MB, commit `fe8a4ea`).
  - Successfully evaluated `arc_easy` on Mac MPS (80 loglikelihood requests processed in 4 seconds at 20 req/s).
  - Logged verified baseline results to `results/results_master.csv`: `acc: 0.70`, `acc_norm: 0.45`.
- **Hybrid Workflow Orchestrated:**
  - Mac M4 configured for baseline loglikelihood verification, script testing, and interactive reporting.
  - Turnkey runner [`notebooks/kaggle_evaluation_runner.ipynb`](notebooks/kaggle_evaluation_runner.ipynb) updated with automated repository cloning and pull handling for 4-bit / 8-bit CUDA execution on Kaggle T4.

---

### [2026-11-03] — Cloud GPU Hardening, OOM Prevention & Dataset Parser Compatibility
- **Hugging Face Hub & Datasets Compatibility Patches:**
  - Solved `huggingface_hub` `HfUriError` on bare repo IDs (`gsm8k`, `hellaswag`, `winogrande`, `arc_easy`) via universal `_parse_repo_body` and `HfUri.__post_init__` patch in `src/harness_runner.py` and `src/__init__.py`.
  - Intercepted strict `datasets.builder.DatasetBuilder` repo-id path checks preventing `Data files don't belong to` crashes on relocated datasets.
  - Suppressed deprecated `trust_remote_code` notices, eliminating redundant 404 HTTP script lookups across 57 MMLU subcategories.
  - Verified live downloads and sample parsing across all 7 evaluation benchmarks (`wikitext`, `arc_easy`, `hellaswag`, `winogrande`, `mmlu`, `gsm8k`, `ifeval`).
- **NVIDIA T4 CUDA OOM Elimination:**
  - Replaced aggressive `lm-eval` `--batch_size auto` doubling with calibrated fixed batch sizes: `8` for loglikelihood tasks, `4` for generative tasks.
  - Enabled PyTorch virtual memory allocator config: `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`.
  - Added inter-run garbage collection (`gc.collect()`) and CUDA cache flushing (`torch.cuda.empty_cache()`).
- **Execution Resilience & Resume Architecture:**
  - Integrated `--skip-completed` flag across `run_experiments.py` and `kaggle_evaluation_runner.ipynb`.
  - Secured Kaggle repo workspace setup (Cell 1b) to preserve uncommitted result CSVs across session restarts.
  - Added multi-tier `HF_TOKEN` auto-resolution across Kaggle Secrets, environment variables, and local cache.
- **Verification:** Full end-to-end smoke test suite passed with 100% success; 16-run experimental dry-run verified.

---

### [2026-11-03] — Context-Window Capping (max_length=2048) & Rolling Perplexity Stabilization
- **Root-Cause Diagnosis for WikiText Perplexity OOM:**
  - Identified that models with massive context windows (Qwen2.5: 32,768 tokens, Llama-3.2: 131,072 tokens) cause `lm_eval`'s `loglikelihood_rolling` to feed the entire unconstrained sequence into PyTorch SDPA attention, demanding a single 7.51 GiB buffer and triggering `torch.OutOfMemoryError` on 16GB GPUs.
- **Double-Layered Memory Hardening:**
  - Standardized `max_length=2048` in `build_lm_eval_model_args` in `src/model_loader.py` for all models and quantizations, aligning with standard literature benchmarking on WikiText-2.
  - Implemented monkey-patch in `src/harness_runner.py` intercepting `HFLM.__init__` to cap any unconstrained or oversized `max_length` at 2048.
  - Capped `effective_batch_size` to 4 whenever `wikitext` is evaluated in `src/evaluator.py`, keeping peak attention allocations under 50 MB and total VRAM well under 4 GB.
- **Verification:** Verified across all models (Qwen2.5-0.5B, Llama-3.2-1B, TinyLlama-1.1B) and quantization formats. All 6 smoke test suites passed cleanly with exit code 0.

---
