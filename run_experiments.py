#!/usr/bin/env python3
"""
run_experiments.py
─────────────────────────────────────────────────────────────────────────────
Master Evaluation Grid Orchestrator for BTP Project GP159.
Implements the full experimental matrix specified in the approved DTU problem statement:
  - Models: Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct
  - Precisions: FP16, 8-bit (GPTQ, BnB), 4-bit (GPTQ, AWQ, BnB-NF4)
  - Tasks: wikitext, arc_easy, hellaswag, winogrande, mmlu, gsm8k, ifeval

Usage:
    # 1. Dry run (inspect all planned runs without execution):
    python run_experiments.py --dry-run

    # 2. Quick smoke test on 50 samples per task:
    python run_experiments.py --model qwen2.5-0.5b --variant fp16 --limit 50

    # 3. Run specific model:
    python run_experiments.py --model qwen2.5-0.5b

    # 4. Full experimental grid:
    python run_experiments.py

Group ID: DTU/CSE/BTP/GP159 | Delhi Technological University 2026-27
"""

import os
import sys
import gc
import time
import yaml
import torch
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

# Prevent PyTorch CUDA memory fragmentation on 16GB GPUs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Local imports
from src.model_loader import detect_device, build_lm_eval_model_args
from src.evaluator import run_evaluation

# Automatic Hugging Face Token resolution (Kaggle Secrets -> Environment -> Local Cache)
if "HF_TOKEN" not in os.environ:
    try:
        from kaggle_secrets import UserSecretsClient
        _token = UserSecretsClient().get_secret("HF_TOKEN")
        if _token:
            os.environ["HF_TOKEN"] = _token
            from huggingface_hub import login
            login(token=_token, add_to_git_credential=False)
    except Exception:
        pass

def load_config(config_path: str = "configs/eval_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def ensure_results_csv(csv_path: str) -> None:
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        columns = [
            "timestamp", "model", "variant", "method", "bits", "loader", "hf_id",
            "status", "eval_time_s", "seed",
            "metric_wikitext", "metric_arc_easy", "metric_hellaswag",
            "metric_winogrande", "metric_mmlu", "metric_gsm8k", "metric_ifeval",
            "error_message"
        ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(csv_path, index=False)
        print(f"[runner] Initialized results CSV: {csv_path}")

def log_result_row(csv_path: str, row_dict: dict) -> None:
    ensure_results_csv(csv_path)
    df_existing = pd.read_csv(csv_path)
    df_new = pd.DataFrame([row_dict])
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(csv_path, index=False)
    print(f"[runner] Logged result to {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Run Task-Conditioned Evaluation Grid (BTP GP159)")
    parser.add_argument("--config", default="configs/eval_config.yaml", help="Path to config YAML")
    parser.add_argument("--model", default=None, help="Filter by model ID (e.g. qwen2.5-0.5b)")
    parser.add_argument("--variant", default=None, help="Filter by variant name (e.g. fp16, gptq_int4)")
    parser.add_argument("--tasks", default=None, help="Comma-separated tasks to run (default: all from config)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of eval samples (for fast testing)")
    parser.add_argument("--batch-size", default="auto", help="Batch size for lm-eval (default: auto)")
    parser.add_argument("--device", default=None, help="Device to use: cuda | mps | cpu (default: auto)")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing")
    parser.add_argument("--skip-completed", action="store_true", help="Skip runs that have already succeeded for all requested tasks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--chat-template", action="store_true", help="Apply official model chat template")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device or detect_device()
    output_dir = cfg.get("output_dir", "results")
    results_csv = cfg.get("results_csv", "results/results_master.csv")
    ensure_results_csv(results_csv)

    # Determine tasks to evaluate
    if args.tasks:
        selected_tasks = [t.strip() for t in args.tasks.split(",")]
    else:
        selected_tasks = [t["id"] for t in cfg["tasks"]]

    # Build list of planned experiment runs
    planned_runs = []
    for model_cfg in cfg["models"]:
        if args.model and model_cfg["id"] != args.model:
            continue
        for var_cfg in model_cfg["variants"]:
            if args.variant and var_cfg["name"] != args.variant:
                continue
            planned_runs.append((model_cfg, var_cfg))

    print("\n" + "═" * 75)
    print("  TASK-CONDITIONED QUANTIZATION EVALUATION GRID (BTP GP159)")
    print(f"  DTU CSE | Session 2026-27 | Guide: Ms. Gull Kaur")
    print("═" * 75)
    print(f"  Target Device : {device}")
    print(f"  Tasks ({len(selected_tasks)})  : {', '.join(selected_tasks)}")
    print(f"  Planned Runs  : {len(planned_runs)}")
    print(f"  Results CSV   : {results_csv}\n")

    for i, (m_cfg, v_cfg) in enumerate(planned_runs, 1):
        print(f"  [{i:02d}/{len(planned_runs):02d}] Model: {m_cfg['name']:<24} Variant: {v_cfg['name']:<12} "
              f"Bits: {v_cfg['bits']:<2} Method: {v_cfg['method']}")

    if args.dry_run:
        print("\n[runner] Dry-run requested. Exiting without execution.")
        return

    # Execute grid
    total_runs = len(planned_runs)
    t_start_grid = time.time()
    completed = 0
    failed = 0

    for i, (m_cfg, v_cfg) in enumerate(planned_runs, 1):
        run_id = f"{m_cfg['id']}_{v_cfg['name']}_s{args.seed}"

        # Resume support: skip if already successfully evaluated for all requested tasks
        if args.skip_completed and Path(results_csv).exists():
            try:
                df_curr = pd.read_csv(results_csv)
                matching = df_curr[
                    (df_curr["model"] == m_cfg["id"]) &
                    (df_curr["variant"] == v_cfg["name"]) &
                    (df_curr["status"] == "SUCCESS")
                ]
                if not matching.empty:
                    needed_metrics = [f"metric_{t}" for t in selected_tasks]
                    has_all = True
                    for met in needed_metrics:
                        if met not in matching.columns or matching[met].dropna().empty:
                            has_all = False
                            break
                    if has_all:
                        print(f"  [{i:02d}/{total_runs:02d}] ⏭️  Skipping {run_id} (already completed successfully in {results_csv})")
                        completed += 1
                        continue
            except Exception:
                pass

        print("\n" + "─" * 75)
        print(f"  EXECUTION [{i:02d}/{total_runs:02d}]: {run_id}")
        print("─" * 75)

        model_args = build_lm_eval_model_args(v_cfg, device=device)
        
        try:
            combined_eval_results = {}
            total_eval_time = 0.0
            overall_status = "SUCCESS"
            error_msg = ""

            # Intelligent Task Partitioning:
            # - Standard loglikelihood tasks (wikitext, arc_easy, hellaswag, winogrande, mmlu) run without chat template
            # - Generative reasoning & instruction following (gsm8k, ifeval) run with chat template
            generative_tasks = {"gsm8k", "ifeval"}

            if args.chat_template:
                sub_batches = [(selected_tasks, True)]
            else:
                std_tasks = [t for t in selected_tasks if t not in generative_tasks]
                gen_tasks = [t for t in selected_tasks if t in generative_tasks]
                sub_batches = []
                if std_tasks:
                    sub_batches.append((std_tasks, False))
                if gen_tasks:
                    sub_batches.append((gen_tasks, True))

            for sub_tasks, use_chat in sub_batches:
                if not sub_tasks:
                    continue
                tag = "Chat-Template" if use_chat else "Standard"
                print(f"[runner] Running {tag} batch: {', '.join(sub_tasks)}")
                sub_results = run_evaluation(
                    model_args=model_args,
                    tasks=sub_tasks,
                    output_dir=output_dir,
                    run_id=run_id,
                    batch_size=args.batch_size,
                    device=device,
                    limit=args.limit,
                    apply_chat_template=use_chat,
                )
                total_eval_time += sub_results.get("eval_time_s", 0.0)
                if sub_results.get("status") != "SUCCESS":
                    overall_status = "FAILED"
                    error_msg = sub_results.get("error_message", "")
                combined_eval_results.update(sub_results)

            combined_eval_results["eval_time_s"] = round(total_eval_time, 2)
            combined_eval_results["status"] = overall_status
            if error_msg:
                combined_eval_results["error_message"] = error_msg

            row = {
                "timestamp": datetime.now().isoformat(),
                "model": m_cfg["id"],
                "variant": v_cfg["name"],
                "method": v_cfg["method"],
                "bits": v_cfg["bits"],
                "loader": v_cfg["loader"],
                "hf_id": v_cfg["hf_id"],
                "seed": args.seed,
                **combined_eval_results
            }
            log_result_row(results_csv, row)
            if overall_status == "SUCCESS":
                completed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[runner] ❌ Run failed with error: {e}")
            row = {
                "timestamp": datetime.now().isoformat(),
                "model": m_cfg["id"],
                "variant": v_cfg["name"],
                "method": v_cfg["method"],
                "bits": v_cfg["bits"],
                "loader": v_cfg["loader"],
                "hf_id": v_cfg["hf_id"],
                "seed": args.seed,
                "status": "FAILED",
                "eval_time_s": 0.0,
                "error_message": str(e)
            }
            log_result_row(results_csv, row)
            failed += 1
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_time_h = (time.time() - t_start_grid) / 3600
    print("\n" + "═" * 75)
    print(f"  GRID FINISHED: {completed} Succeeded, {failed} Failed. Total time: {total_time_h:.2f} hrs.")
    print(f"  Master Results Saved to: {results_csv}")
    print("═" * 75 + "\n")

if __name__ == "__main__":
    main()

