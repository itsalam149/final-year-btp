#!/usr/bin/env python3
"""
run_analysis.py
─────────────────────────────────────────────────────────────────────────────
Statistical Analysis Pipeline for BTP Project GP159 (v2 Execution Plan).
Takes results_master.csv and computes:
  1. Task-by-task accuracy degradation against FP16 baselines (RQ1),
     with 10,000-iteration Paired Bootstrap Confidence Intervals (95% CI).
  2. Spearman rank correlation between ΔPPL and ΔAcc with bootstrap CIs
     and Leave-One-Out (LOO) sensitivity checks (RQ2).
  3. Noise-Aware Practical Bit-Budget Guidelines based on the <= 3 pp criterion
     and sensitivity analysis across [1.0, 2.0, 3.0, 5.0] pp (RQ3).
  4. Generates publication-ready Markdown and CSV summaries in results/.

Usage:
    python run_analysis.py
    python run_analysis.py --csv results/results_master.csv --bootstrap 10000

Group ID: DTU/CSE/BTP/GP159 | DTU 2026-27
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

from src.analysis.degradation import compute_degradation, format_degradation_summary
from src.analysis.spearman_proxy import compute_spearman_proxy_analysis, format_spearman_table
from src.analysis.bit_budget import (
    evaluate_bit_budget_guidelines,
    generate_task_budget_matrix,
    format_budget_table,
    run_sensitivity_sweep,
)


def main():
    parser = argparse.ArgumentParser(description="Run statistical analysis on quantization results (v2)")
    parser.add_argument("--csv", default="results/results_master.csv", help="Path to input results CSV")
    parser.add_argument("--outdir", default="results", help="Directory to save analysis reports")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Iterations for bootstrap CIs (default: 10000)")
    parser.add_argument("--ppl-corpus", default="delta_wikitext", help="Perplexity column to correlate (default: delta_wikitext)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[analysis] ❌ Results CSV not found at {csv_path}. Run experiments first!")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[analysis] ❌ Results CSV is empty.")
        sys.exit(1)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 78)
    print("  TASK-CONDITIONED QUANTIZATION STATISTICAL ANALYSIS (BTP GP159, v2)")
    print("═" * 78 + "\n")

    # Detect metric columns
    metric_cols = [c for c in df.columns if c.startswith("metric_")]
    task_names = [c.replace("metric_", "") for c in metric_cols]

    # -----------------------------------------------------------------
    # 1. Degradation Analysis (RQ1)
    # -----------------------------------------------------------------
    print(">>> 1. Computing Task-by-Task Degradation (ΔAcc & ΔPPL with CIs) ...")
    deg_df = compute_degradation(df, metric_cols, n_bootstrap=args.bootstrap)
    deg_csv = out_dir / "degradation_master.csv"
    deg_df.to_csv(deg_csv, index=False)

    deg_summary = format_degradation_summary(deg_df, task_names)
    print(deg_summary)

    deg_md = out_dir / "degradation_report.md"
    with open(deg_md, "w") as f:
        f.write("# Task-Conditioned Degradation Report (RQ1)\n\n")
        f.write("Degradation measured in percentage points (pp) against unquantized FP16 baseline.\n")
        f.write("Includes 95% paired bootstrap confidence intervals and relative drops.\n\n")
        f.write(deg_summary)
        f.write("\n")
    print(f"  -> Saved to: {deg_md}\n")

    # -----------------------------------------------------------------
    # 2. Spearman Correlation Analysis: Perplexity as Proxy (RQ2)
    # -----------------------------------------------------------------
    ppl_col = args.ppl_corpus
    print(f">>> 2. Evaluating Perplexity as a Proxy via Spearman Correlation (RQ2) [{ppl_col}] ...")
    if ppl_col in deg_df.columns:
        spearman_df = compute_spearman_proxy_analysis(deg_df, ppl_col=ppl_col)
        spearman_summary = format_spearman_table(spearman_df)
        print(spearman_summary)

        spearman_csv = out_dir / "spearman_proxy_analysis.csv"
        spearman_df.to_csv(spearman_csv, index=False)

        spearman_md = out_dir / "spearman_proxy_report.md"
        with open(spearman_md, "w") as f:
            f.write("# Perplexity as a Proxy Reliability Report (RQ2)\n\n")
            f.write(f"Spearman Rank Correlation between {ppl_col} and downstream ΔAcc across tasks.\n")
            f.write("Evaluates whether perplexity is a faithful proxy or hides localized capability damage.\n\n")
            f.write(spearman_summary)
            f.write("\n\n### Decision Criteria:\n")
            f.write("- **Faithful Proxy**: ρ <= -0.70 AND 95% CI upper bound < 0 (worsening PPL reliably tracks accuracy loss).\n")
            f.write("- **Decoupled**: |ρ| < 0.40 OR 95% CI spans 0 (PPL hides damage; task degradation decoupled from PPL).\n")
            f.write("- **Intermediate**: -0.70 < ρ <= -0.40.\n")
        print(f"  -> Saved to: {spearman_md}\n")
    else:
        print(f"  -> Warning: {ppl_col} not found in data. Skipping Spearman proxy analysis.")

    # -----------------------------------------------------------------
    # 3. Practical Bit-Budget Guidelines (RQ3: Noise-Aware Rule)
    # -----------------------------------------------------------------
    print(">>> 3. Generating Noise-Aware Bit-Budget Guidelines (RQ3) ...")
    budget_eval_df = evaluate_bit_budget_guidelines(deg_df, threshold_pp=3.0)
    budget_matrix_df = generate_task_budget_matrix(budget_eval_df)

    budget_summary = format_budget_table(budget_matrix_df)
    print(budget_summary)

    budget_csv = out_dir / "bit_budget_guidelines.csv"
    budget_matrix_df.to_csv(budget_csv, index=False)

    # Sensitivity Analysis Sweep
    sensitivity_df = run_sensitivity_sweep(deg_df, thresholds=[1.0, 2.0, 3.0, 5.0])
    sens_md_table = format_budget_table(sensitivity_df)
    sensitivity_csv = out_dir / "bit_budget_sensitivity.csv"
    sensitivity_df.to_csv(sensitivity_csv, index=False)

    budget_md = out_dir / "bit_budget_guidelines.md"
    with open(budget_md, "w") as f:
        f.write("# Practical Bit-Budget Guidelines (RQ3)\n\n")
        f.write("Evaluated under DTU Working Acceptance Rule: **ΔAcc loss <= 3.0 percentage points with 95% CI**.\n\n")
        f.write(budget_summary)
        f.write("\n\n## Sensitivity Analysis Across Thresholds\n\n")
        f.write(sens_md_table)
        f.write("\n")
    print(f"  -> Saved to: {budget_md}\n")

    print("═" * 78)
    print("  STATISTICAL ANALYSIS COMPLETE — ALL REPORTS COMPILED IN results/")
    print("═" * 78 + "\n")


if __name__ == "__main__":
    main()
