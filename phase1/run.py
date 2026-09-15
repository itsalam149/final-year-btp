"""
run.py
─────────────────────────────────────────────────────────────────────────────
Main orchestrator for the Phase-1 ablation grid.

Usage
-----
    # Run the full 27-run grid (from repo root):
    python phase1/run.py

    # Run a single experiment (quick test):
    python phase1/run.py --model qwen2.5-0.5b --method rtn --bits 4

    # Run all methods for one model:
    python phase1/run.py --model pythia-1.4b

    # Dry-run (list planned experiments without executing):
    python phase1/run.py --dry-run

Options
-------
    --config    Path to YAML config  (default: phase1/configs/run_config.yaml)
    --model     Short model name to filter (e.g. qwen2.5-0.5b)
    --method    Method to filter (rtn | gptq | gptq_hadamard)
    --bits      Bit-width to filter (2 | 3 | 4)
    --dry-run   Print planned runs without executing
    --skip-eval Skip lm-eval (quantize only, useful for debugging)
    --device    cuda | cpu  (default: auto-detect)

Authors: Faqre Alam · Guneet Toppo · Ekansh Agrawal
Project: BTech Final Year Project, DTU 2026-27
"""

import argparse
import os
import sys
import time
import shutil
import yaml
import torch
from pathlib import Path
from datetime import datetime

# Make sure the repo root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM

from phase1.framework.calibrate import get_calibration_data, capture_layer_inputs
from phase1.framework.quantize  import quantize_model
from phase1.framework.evaluate  import run_lm_eval, log_result, print_results_table


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def detect_device() -> str:
    """Auto-detect best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_info(device: str) -> str:
    if device == "cuda":
        return (f"GPU: {torch.cuda.get_device_name(0)}  "
                f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    if device == "mps":
        return "Apple Silicon MPS (Metal Performance Shaders)"
    return "CPU only"


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def apply_profile(cfg: dict, profile: str) -> dict:
    """Override top-level config keys with values from a named profile."""
    if profile and profile in cfg:
        overrides = cfg[profile]
        for key, val in overrides.items():
            cfg[key] = val
        print(f"[run] Profile '{profile}' applied.")
    return cfg


def build_run_list(cfg: dict, filter_model=None, filter_method=None, filter_bits=None):
    """Return list of (model_dict, method, bits) tuples to run."""
    runs = []
    for model_dict in cfg["models"]:
        if filter_model and model_dict["short_name"] != filter_model:
            continue
        for method in cfg["methods"]:
            if filter_method and method != filter_method:
                continue
            for bits in cfg["bits"]:
                if filter_bits and bits != filter_bits:
                    continue
                runs.append((model_dict, method, bits))

    return runs


def model_save_path(output_dir: str, model_short: str, method: str, bits: int) -> str:
    return str(Path(output_dir) / f"{model_short}_{method}_{bits}b")


def banner(text: str):
    width = min(70, len(text) + 6)
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


# ─────────────────────────────────────────────────────────────────────────────
# Main run function for a single (model, method, bits) configuration
# ─────────────────────────────────────────────────────────────────────────────

def run_one(
    model_dict:  dict,
    method:      str,
    bits:        int,
    cfg:         dict,
    device:      str,
    skip_eval:   bool,
    output_dir:  str,
    csv_path:    str,
):
    model_id    = model_dict["id"]
    model_short = model_dict["short_name"]
    cal_cfg     = cfg["calibration"]
    gptq_cfg    = cfg.get("gptq", {})
    had_cfg     = cfg.get("hadamard", {})
    eval_cfg    = cfg.get("eval", {})

    banner(f"RUN:  {model_short}  |  {method}  |  {bits}-bit")

    # ── 1. Load tokenizer + model ────────────────────────────────────────
    print(f"\n[run] Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Set pad token if missing (common for decoder-only models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "mps":
        # Apple Silicon MPS crashes with device_map="auto" during loading
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model = model.to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    model.eval()
    print(f"[run] Model loaded. Params: "
          f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # ── 2. Calibration (skip for RTN — no calibration data needed) ───────
    activations = {}
    if method != "rtn":
        print(f"\n[run] Collecting calibration activations...")
        samples = get_calibration_data(
            tokenizer,
            n_samples=cal_cfg["n_samples"],
            seq_len=cal_cfg["seq_len"],
            seed=cal_cfg["seed"],
            dataset_name=cal_cfg["dataset"],
            dataset_config=cal_cfg["dataset_config"],
            split=cal_cfg["split"],
        )
        activations = capture_layer_inputs(
            model,
            samples,
            skip_layer_names=cfg.get("skip_layer_names", []),
            device=device,
        )
        # Move model back to GPU (capture_layer_inputs may shift it)
        model.to(device)

    # ── 3. Quantize ──────────────────────────────────────────────────────
    print(f"\n[run] Quantizing ({method}, {bits}-bit)...")
    t_quant_start = time.time()

    model = quantize_model(
        model,
        activations=activations,
        method=method,
        bits=bits,
        dampening=gptq_cfg.get("dampening_factor", 0.01),
        block_size=gptq_cfg.get("block_size", 128),
        hadamard_seed=had_cfg.get("seed", 1337),
        skip_layer_names=cfg.get("skip_layer_names", []),
        device=device,
    )

    quant_time = time.time() - t_quant_start
    print(f"[run] Quantization done in {quant_time:.1f}s ({quant_time/60:.1f} min)")

    # ── 4. Save quantized model ──────────────────────────────────────────
    save_path = model_save_path(output_dir, model_short, method, bits)
    print(f"\n[run] Saving quantized model to: {save_path}")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # ── 5. Evaluate ──────────────────────────────────────────────────────
    eval_metrics = {}
    eval_time    = 0.0

    if not skip_eval:
        print(f"\n[run] Starting lm-eval...")
        t_eval_start = time.time()
        eval_metrics = run_lm_eval(
            model_path=save_path,
            tasks=eval_cfg.get("tasks", ["wikitext", "arc_easy", "hellaswag", "winogrande"]),
            batch_size=eval_cfg.get("batch_size", 8),
            device=eval_cfg.get("device", device),
        )
        eval_time = time.time() - t_eval_start
        print(f"[run] Evaluation done in {eval_time:.1f}s ({eval_time/60:.1f} min)")
    else:
        print("[run] Skipping evaluation (--skip-eval flag set).")

    # ── 6. Log result ────────────────────────────────────────────────────
    row = {
        "model":          model_short,
        "method":         method,
        "bits":           bits,
        "quant_time_s":   round(quant_time, 1),
        "eval_time_s":    round(eval_time, 1),
        "model_path":     save_path,
        **eval_metrics,
    }
    log_result(row, csv_path)

    # ── 7. Free memory ───────────────────────────────────────────────────
    del model, activations
    torch.cuda.empty_cache()

    total_time = quant_time + eval_time
    print(f"\n[run] ✅ Completed in {total_time:.1f}s ({total_time/60:.1f} min)\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase-1 ablation grid runner — LLM PTQ study"
    )
    parser.add_argument("--config",    default="phase1/configs/run_config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--model",     default=None,
                        help="Filter: short model name (e.g. qwen2.5-0.5b)")
    parser.add_argument("--method",    default=None,
                        help="Filter: rtn | gptq | gptq_hadamard")
    parser.add_argument("--bits",      default=None, type=int,
                        help="Filter: 2 | 3 | 4")
    parser.add_argument("--dry-run",   action="store_true",
                        help="List planned runs without executing")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip lm-eval (quantize only)")
    parser.add_argument("--device",    default=None,
                        help="cuda | mps | cpu  (default: auto-detect)")
    parser.add_argument("--profile",   default=None,
                        help="Config profile to apply, e.g. local_test")
    args = parser.parse_args()

    # ── Load config + apply profile ───────────────────────────────────────
    cfg = load_config(args.config)
    if args.profile:
        cfg = apply_profile(cfg, args.profile)

    # ── Device ────────────────────────────────────────────────────────────
    device = args.device or ("auto" if not args.device else args.device)
    if device == "auto" or device is None:
        device = detect_device()
    print(f"\n[run] Device: {device}  —  {device_info(device)}")

    # ── Output paths ──────────────────────────────────────────────────────
    output_dir = cfg.get("output_dir", "phase1/results")
    csv_path   = str(Path(output_dir) / "ablation_table.csv")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Build run list ────────────────────────────────────────────────────
    runs = build_run_list(cfg,
                          filter_model=args.model,
                          filter_method=args.method,
                          filter_bits=args.bits)

    if not runs:
        print("[run] No runs matched the given filters. Exiting.")
        return

    banner(f"PHASE-1 ABLATION GRID  —  {len(runs)} runs planned")
    print(f"\n  Output dir : {output_dir}")
    print(f"  CSV path   : {csv_path}")
    print(f"  Device     : {device}\n")

    for i, (model_dict, method, bits) in enumerate(runs, 1):
        print(f"  [{i:02d}/{len(runs):02d}]  {model_dict['short_name']:<22}  "
              f"{method:<16}  {bits}-bit")

    if args.dry_run:
        print("\n[run] Dry-run mode — exiting without executing.")
        return

    print(f"\n[run] Starting grid at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Execute runs ──────────────────────────────────────────────────────
    completed = 0
    failed    = 0
    t_grid_start = time.time()

    for i, (model_dict, method, bits) in enumerate(runs, 1):
        try:
            run_one(
                model_dict=model_dict,
                method=method,
                bits=bits,
                cfg=cfg,
                device=device,
                skip_eval=args.skip_eval,
                output_dir=output_dir,
                csv_path=csv_path,
            )
            completed += 1
        except KeyboardInterrupt:
            print("\n[run] Interrupted by user. Partial results saved.")
            break
        except Exception as e:
            import traceback
            print(f"\n[run] ❌ ERROR in run {i}: {e}")
            traceback.print_exc()
            # Log error row so we know which run failed
            log_result({
                "model": model_dict["short_name"],
                "method": method,
                "bits": bits,
                "ppl_wikitext2": f"ERROR: {str(e)[:80]}",
            }, csv_path)
            failed += 1
            continue

    # ── Final summary ─────────────────────────────────────────────────────
    total_time = time.time() - t_grid_start
    banner(f"GRID COMPLETE — {completed}/{len(runs)} succeeded  "
           f"({total_time/3600:.1f} GPU-hours)")

    print_results_table(csv_path)
    print(f"\n📄 Full results: {csv_path}\n")


if __name__ == "__main__":
    main()
