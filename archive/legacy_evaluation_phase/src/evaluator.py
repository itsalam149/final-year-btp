"""
evaluator.py
─────────────────────────────────────────────────────────────────────────────
Standardized evaluation harness interface for BTP Project GP159 (v2 Plan).
Wraps EleutherAI lm-evaluation-harness to measure:
  1. WikiText-2 & C4 Perplexity (Language modeling proxy & control)
  2. ARC-Easy (Factual recall)
  3. HellaSwag (Commonsense reasoning)
  4. WinoGrande (Coreference / reasoning)
  5. MMLU (Academic knowledge, 5-shot)
  6. GSM8K (Multi-step math reasoning, 5-shot: strict and flexible)
  7. IFEval (Instruction following: prompt-level strict & instruction-level)

Preserves untouched raw JSON outputs in results/raw/ and logs sample-level outcomes.

Group ID: DTU/CSE/BTP/GP159 | DTU 2026-27
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

PRIMARY_METRICS = {
    "wikitext": ["word_perplexity", "byte_perplexity"],
    "arc_easy": ["acc_norm,none", "acc,none"],
    "hellaswag": ["acc_norm,none", "acc_none"],
    "winogrande": ["acc,none"],
    "mmlu": ["acc,none", "acc_norm,none"],
    "gsm8k": ["exact_match,strict-match", "exact_match,flexible-extract", "exact_match"],
    "ifeval": ["prompt_level_strict_acc,none", "inst_level_strict_acc,none"],
}

SECONDARY_METRICS = {
    "gsm8k": ["exact_match,flexible-extract"],
    "ifeval": ["inst_level_strict_acc,none", "prompt_level_loose_acc,none"],
}


def extract_metric_from_results(results_dict: Dict[str, Any], task_name: str) -> Optional[float]:
    """
    Search lm-eval results dictionary for the primary metric of a given task.
    Handles hierarchy in both lm-eval v0.4.x formats.
    """
    results_block = results_dict.get("results", {})
    if task_name not in results_block:
        return None

    task_res = results_block[task_name]
    candidate_keys = PRIMARY_METRICS.get(task_name, ["acc", "acc_norm"])

    for k in candidate_keys:
        if k in task_res:
            val = task_res[k]
            try:
                return float(val)
            except (ValueError, TypeError):
                continue

    # Fallback search
    for k, v in task_res.items():
        if ("acc" in k or "perplexity" in k or "exact_match" in k) and not k.endswith("_stderr"):
            try:
                return float(v)
            except (ValueError, TypeError):
                continue

    return None


def extract_secondary_metrics(results_dict: Dict[str, Any], task_name: str) -> Dict[str, float]:
    """
    Extract secondary metrics for a task (e.g. flexible-extract on GSM8K, instruction-level on IFEval).
    """
    secondary = {}
    results_block = results_dict.get("results", {})
    if task_name not in results_block:
        return secondary

    task_res = results_block[task_name]
    candidates = SECONDARY_METRICS.get(task_name, [])

    for k in candidates:
        if k in task_res:
            try:
                clean_name = k.replace(",", "_").replace("-", "_")
                secondary[f"metric_{task_name}_{clean_name}"] = float(task_res[k])
            except (ValueError, TypeError):
                continue

    return secondary


def run_evaluation(
    model_args: str,
    tasks: List[str],
    output_dir: str,
    run_id: str,
    batch_size: Any = "auto",
    device: str = "auto",
    limit: Optional[int] = None,
    num_fewshot: Optional[int] = None,
    apply_chat_template: bool = False,
) -> Dict[str, Any]:
    """
    Execute lm-evaluation-harness with raw sample logging.
    """
    raw_dir = Path(output_dir) / "raw" / run_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    results_json = raw_dir / "lm_eval_output.json"

    tasks_arg = ",".join(tasks)

    # Auto or custom batch-size safety bounds
    effective_batch_size = batch_size
    if str(batch_size).lower() == "auto":
        if apply_chat_template or any(t in ["gsm8k", "ifeval"] for t in tasks):
            effective_batch_size = 4
        elif "wikitext" in tasks:
            effective_batch_size = 4
        elif device in ["mps", "cpu"]:
            effective_batch_size = 4
        else:
            effective_batch_size = 8
        print(f"[evaluator] Note: Using stable batch_size={effective_batch_size} for device '{device}' (prevents CUDA OOM).")
    else:
        if apply_chat_template or any(t in ["gsm8k", "ifeval"] for t in tasks) or ("wikitext" in tasks):
            try:
                if int(effective_batch_size) > 4:
                    effective_batch_size = 4
                    print(f"[evaluator] Note: Capped batch_size to 4 to prevent CUDA OOM.")
            except (ValueError, TypeError):
                effective_batch_size = 4

    cmd = [
        sys.executable, "-m", "src.harness_runner",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", tasks_arg,
        "--batch_size", str(effective_batch_size),
        "--output_path", str(results_json),
        "--log_samples",
    ]

    if apply_chat_template:
        cmd.append("--apply_chat_template")

    if device != "auto":
        cmd.extend(["--device", device])

    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])

    print(f"\n[evaluator] Launching lm-eval for run: {run_id}")
    print(f"[evaluator] Tasks: {tasks_arg}")
    print(f"[evaluator] Command: {' '.join(cmd)}\n")

    t0 = time.time()
    log_lines = []
    # Stream stdout/stderr live to console so progress bars (tqdm) and downloads are visible
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    is_prev_progress = False
    for raw_line in iter(proc.stdout.readline, ""):
        log_lines.append(raw_line)
        line = raw_line.rstrip("\r\n")
        
        # Detect tqdm progress bar updates to keep them on a single line
        is_progress = ("%|" in line) or ("it/s" in line and "/" in line) or ("s/it" in line and "/" in line)
        if is_progress:
            # Overwrite line in-place in Jupyter notebook / console
            sys.stdout.write(f"\r{line[:115]:<115}")
            sys.stdout.flush()
            is_prev_progress = True
        else:
            if is_prev_progress:
                sys.stdout.write("\n")
                is_prev_progress = False
            sys.stdout.write(f"{line}\n")
            sys.stdout.flush()

    if is_prev_progress:
        sys.stdout.write("\n")

    proc.stdout.close()
    returncode = proc.wait()
    eval_time_s = time.time() - t0
    full_log = "".join(log_lines)

    metrics = {
        "eval_time_s": round(eval_time_s, 2),
        "status": "SUCCESS" if returncode == 0 else "FAILED",
        "error_message": "",
    }

    if returncode != 0:
        print(f"\n[evaluator] ❌ lm-eval failed with return code {returncode}!")
        metrics["error_message"] = full_log[-1000:]
        return metrics

    # Parse output JSON - pick the freshest results JSON generated by lm-eval
    found_json = None
    if results_json.is_file():
        found_json = results_json
    else:
        results_files = list(raw_dir.glob("**/results*.json"))
        if results_files:
            results_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            found_json = results_files[0]

    if found_json and found_json.exists():
        try:
            with open(found_json) as f:
                data = json.load(f)

            for task in tasks:
                val = extract_metric_from_results(data, task)
                metrics[f"metric_{task}"] = val
                print(f"  -> Task [{task:12s}]: {val}")

                # Secondary metrics
                sec_dict = extract_secondary_metrics(data, task)
                for sk, sv in sec_dict.items():
                    metrics[sk] = sv
                    print(f"     [Secondary] {sk}: {sv}")

        except Exception as e:
            print(f"[evaluator] Error parsing output JSON: {e}")
            metrics["error_message"] = str(e)
    else:
        print(f"[evaluator] Warning: results JSON not found at {results_json}")

    return metrics
