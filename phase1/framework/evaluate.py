"""
evaluate.py
─────────────────────────────────────────────────────────────────────────────
Evaluation wrapper for the PTQ ablation grid.

- Runs EleutherAI lm-evaluation-harness on a saved quantized model.
- Extracts WikiText-2 perplexity + ARC-Easy / HellaSwag / WinoGrande accuracy.
- Logs one result row to ablation_table.csv.

Authors: Faqre Alam · Guneet Toppo · Ekansh Agrawal
Project: BTech Final Year Project, DTU 2026-27
"""

import os
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# CSV Schema
# ─────────────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "timestamp",
    "model",
    "method",
    "bits",
    "ppl_wikitext2",     # WikiText-2 test perplexity  (lower = better)
    "acc_arc_easy",      # ARC-Easy 0-shot accuracy    (higher = better)
    "acc_hellaswag",     # HellaSwag 0-shot accuracy   (higher = better)
    "acc_winogrande",    # WinoGrande 0-shot accuracy  (higher = better)
    "quant_time_s",      # wall-clock time for quantization (seconds)
    "eval_time_s",       # wall-clock time for evaluation   (seconds)
    "model_path",        # path to saved quantized model
]


def ensure_csv(csv_path: str) -> None:
    """Create CSV with header if it doesn't exist."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
        print(f"[evaluate] Created results CSV: {csv_path}")


def log_result(row: Dict, csv_path: str) -> None:
    """
    Append one result row to the ablation CSV.

    Parameters
    ----------
    row      : dict with keys from CSV_COLUMNS (missing keys → empty string)
    csv_path : path to the output CSV file
    """
    ensure_csv(csv_path)
    row["timestamp"] = row.get("timestamp", datetime.now().isoformat())

    # Fill missing columns with empty string
    full_row = {col: row.get(col, "") for col in CSV_COLUMNS}

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(full_row)

    print(f"[evaluate] Logged: model={full_row['model']}  "
          f"method={full_row['method']}  bits={full_row['bits']}  "
          f"ppl={full_row['ppl_wikitext2']}")


# ─────────────────────────────────────────────────────────────────────────────
# lm-evaluation-harness runner
# ─────────────────────────────────────────────────────────────────────────────

def run_lm_eval(
    model_path: str,
    tasks: list = None,
    batch_size: int = 8,
    device: str = "cuda",
    output_json: Optional[str] = None,
) -> Dict:
    """
    Run lm-evaluation-harness on a saved model and return parsed metrics.

    Uses the CLI (`lm_eval`) so it works with any lm-eval v0.4.x installation.
    Falls back to the Python API if the CLI is unavailable.

    Parameters
    ----------
    model_path  : path to the saved HuggingFace model directory
    tasks       : list of lm-eval task names
                  default: ["wikitext", "arc_easy", "hellaswag", "winogrande"]
    batch_size  : evaluation batch size
    device      : "cuda" or "cpu"
    output_json : where to save the raw lm-eval JSON output (optional)

    Returns
    -------
    Dict with keys: ppl_wikitext2, acc_arc_easy, acc_hellaswag, acc_winogrande
    """
    if tasks is None:
        tasks = ["wikitext", "arc_easy", "hellaswag", "winogrande"]

    # Create a temp output path if not provided
    if output_json is None:
        safe_name = Path(model_path).name.replace("/", "_")
        output_json = str(Path(model_path) / "lm_eval_results.json")

    tasks_str = ",".join(tasks)

    # ── Try CLI approach first ────────────────────────────────────────────
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype=float16",
        "--tasks", tasks_str,
        "--num_fewshot", "0",
        "--batch_size", str(batch_size),
        "--device", device,
        "--output_path", output_json,
        "--log_samples",
    ]

    print(f"\n[evaluate] Running lm-eval on: {model_path}")
    print(f"[evaluate] Tasks: {tasks_str}")
    print(f"[evaluate] Command: {' '.join(cmd)}\n")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    eval_time = time.time() - t0

    if result.returncode != 0:
        print(f"[evaluate] lm-eval stderr:\n{result.stderr}")
        # Try Python API fallback
        return _run_lm_eval_python(
            model_path, tasks, batch_size, device, output_json, eval_time
        )

    # ── Parse the JSON output ─────────────────────────────────────────────
    return _parse_lm_eval_json(output_json, eval_time)


def _parse_lm_eval_json(json_path: str, eval_time: float) -> Dict:
    """
    Parse lm-eval JSON output and extract our four metrics.
    Handles both v0.4.x formats.
    """
    metrics = {
        "ppl_wikitext2": "",
        "acc_arc_easy": "",
        "acc_hellaswag": "",
        "acc_winogrande": "",
        "eval_time_s": round(eval_time, 1),
    }

    try:
        # lm-eval v0.4.x writes results into a subfolder
        json_path_obj = Path(json_path)
        if json_path_obj.is_dir():
            json_files = list(json_path_obj.glob("*.json"))
            if json_files:
                json_path_obj = json_files[0]

        with open(json_path_obj) as f:
            data = json.load(f)

        results = data.get("results", {})

        # WikiText-2 perplexity
        for key in ["wikitext", "wikitext2"]:
            if key in results:
                ppl = results[key].get("word_perplexity,none",
                      results[key].get("perplexity,none",
                      results[key].get("bits_per_byte,none", "")))
                if ppl:
                    metrics["ppl_wikitext2"] = round(float(ppl), 3)
                break

        # ARC-Easy accuracy (normalized)
        for key in ["arc_easy"]:
            if key in results:
                acc = results[key].get("acc_norm,none",
                      results[key].get("acc,none", ""))
                if acc:
                    metrics["acc_arc_easy"] = round(float(acc) * 100, 2)

        # HellaSwag accuracy
        if "hellaswag" in results:
            acc = results["hellaswag"].get("acc_norm,none",
                  results["hellaswag"].get("acc,none", ""))
            if acc:
                metrics["acc_hellaswag"] = round(float(acc) * 100, 2)

        # WinoGrande accuracy
        if "winogrande" in results:
            acc = results["winogrande"].get("acc,none", "")
            if acc:
                metrics["acc_winogrande"] = round(float(acc) * 100, 2)

    except Exception as e:
        print(f"[evaluate] WARNING: Could not parse lm-eval JSON: {e}")

    return metrics


def _run_lm_eval_python(
    model_path: str,
    tasks: list,
    batch_size: int,
    device: str,
    output_json: str,
    eval_time_so_far: float,
) -> Dict:
    """
    Python API fallback for lm-eval (in case CLI path fails).
    """
    try:
        import lm_eval
        from lm_eval import evaluator, utils

        print("[evaluate] Using lm_eval Python API (fallback)...")
        t0 = time.time()

        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_path},dtype=float16",
            tasks=tasks,
            num_fewshot=0,
            batch_size=batch_size,
            device=device,
        )

        eval_time = time.time() - t0

        # Save results to JSON
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)

        return _parse_lm_eval_json(output_json, eval_time)

    except Exception as e:
        print(f"[evaluate] ERROR: lm-eval Python API also failed: {e}")
        return {
            "ppl_wikitext2": "ERROR",
            "acc_arc_easy": "ERROR",
            "acc_hellaswag": "ERROR",
            "acc_winogrande": "ERROR",
            "eval_time_s": round(eval_time_so_far, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print results table to terminal
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(csv_path: str) -> None:
    """Print the current state of the ablation CSV as a formatted table."""
    if not Path(csv_path).exists():
        print("[evaluate] No results file yet.")
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("[evaluate] Results file is empty.")
        return

    header = f"{'Model':<22} {'Method':<16} {'Bits':>4}  {'PPL':>8}  {'ARC':>6}  {'HS':>6}  {'WG':>6}"
    print("\n" + "═" * len(header))
    print(header)
    print("─" * len(header))
    for r in rows:
        print(f"{r['model']:<22} {r['method']:<16} {r['bits']:>4}  "
              f"{r['ppl_wikitext2']:>8}  {r['acc_arc_easy']:>6}  "
              f"{r['acc_hellaswag']:>6}  {r['acc_winogrande']:>6}")
    print("═" * len(header) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Quick test (run: python -m phase1.framework.evaluate)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, os

    print("\n=== evaluate.py self-test ===\n")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        tmppath = f.name

    # Test CSV logging
    ensure_csv(tmppath)
    test_row = {
        "model": "qwen2.5-0.5b",
        "method": "gptq",
        "bits": 4,
        "ppl_wikitext2": 13.42,
        "acc_arc_easy": 61.5,
        "acc_hellaswag": 58.2,
        "acc_winogrande": 55.1,
        "quant_time_s": 900,
        "eval_time_s": 1200,
        "model_path": "/tmp/fake_model",
    }
    log_result(test_row, tmppath)
    log_result({**test_row, "method": "rtn", "ppl_wikitext2": 18.91}, tmppath)
    print_results_table(tmppath)
    os.unlink(tmppath)
    print("evaluate.py self-test passed ✅")
