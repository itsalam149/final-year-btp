"""
bit_budget.py
─────────────────────────────────────────────────────────────────────────────
Direct implementation for Research Question 3 (RQ3):
Practical Bit-Budget Guidelines using a Noise-Aware Acceptance Criterion.

Decision Rule (v2 Plan):
  - ACCEPTABLE (✓)  : Upper bound of accuracy loss is <= 3.0 pp (-ci_low <= 3.0 pp)
  - MARGINAL   (!)  : Point estimate loss <= 3.0 pp, but upper bound > 3.0 pp
  - REJECTED   (✗)  : Point estimate loss > 3.0 pp (-delta > 3.0 pp)

Features:
  - Noise-aware classification using paired bootstrap confidence intervals
  - Sensitivity analysis sweep across thresholds: [1.0, 2.0, 3.0, 5.0] pp
  - Generates Task-Conditioned Minimum Bit-Width Decision Matrix

Group ID: DTU/CSE/BTP/GP159 | DTU 2026-27
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from tabulate import tabulate

TASK_CATEGORIES = {
    "arc_easy": "Factual Recall",
    "mmlu": "Academic / Domain Knowledge",
    "hellaswag": "Commonsense Reasoning",
    "winogrande": "Coreference / Commonsense",
    "gsm8k": "Multi-Step Mathematical Reasoning",
    "ifeval": "Instruction Following",
}


def evaluate_bit_budget_guidelines(
    df: pd.DataFrame,
    threshold_pp: float = 3.0,
    task_delta_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Apply the noise-aware acceptance criterion across all models, bit-widths, and tasks.

    Parameters
    ----------
    df : pd.DataFrame with delta_* and optional delta_*_ci_low columns
    threshold_pp : maximum allowable degradation in percentage points (default: 3.0)
    task_delta_cols : list of delta columns to evaluate

    Returns
    -------
    pd.DataFrame with noise-aware status per configuration and task
    """
    if task_delta_cols is None:
        task_delta_cols = [
            c for c in df.columns
            if c.startswith("delta_")
            and not c.endswith("_ci_low")
            and not c.endswith("_ci_high")
            and "wikitext" not in c
            and "c4" not in c
        ]

    results = []

    for _, row in df.iterrows():
        model = row["model"]
        method = row["method"]
        bits = row["bits"]

        # Skip FP16 baseline
        if bits == 16:
            continue

        for col in task_delta_cols:
            task_name = col.replace("delta_", "")
            delta_val = row.get(col, np.nan)
            ci_low_col = f"{col}_ci_low"
            ci_low_val = row.get(ci_low_col, np.nan)

            if pd.isna(delta_val):
                status = "NO_DATA"
                verdict = "UNKNOWN"
            else:
                loss_pt = -delta_val
                loss_upper_bound = -ci_low_val if not pd.isna(ci_low_val) else loss_pt

                if loss_upper_bound <= threshold_pp:
                    status = f"ACCEPTABLE (loss <= {threshold_pp} pp with 95% CI)"
                    verdict = "ACCEPTABLE"
                elif loss_pt <= threshold_pp and loss_upper_bound > threshold_pp:
                    status = f"MARGINAL (point loss {loss_pt:.2f} <= {threshold_pp} pp, but CI upper bound {loss_upper_bound:.2f} > {threshold_pp} pp)"
                    verdict = "MARGINAL"
                else:
                    status = f"REJECTED (loss {loss_pt:.2f} pp > {threshold_pp} pp)"
                    verdict = "REJECTED"

            results.append({
                "model": model,
                "bits": bits,
                "method": method,
                "task": task_name,
                "category": TASK_CATEGORIES.get(task_name, "General"),
                "delta_pp": delta_val,
                "ci_low_pp": ci_low_val,
                "status": status,
                "verdict": verdict,
                "is_acceptable": (verdict == "ACCEPTABLE"),
            })

    if not results:
        return pd.DataFrame(columns=[
            "model", "bits", "method", "task", "category",
            "delta_pp", "ci_low_pp", "status", "verdict", "is_acceptable"
        ])

    return pd.DataFrame(results)


def run_sensitivity_sweep(
    df: pd.DataFrame,
    thresholds: List[float] = [1.0, 2.0, 3.0, 5.0],
    task_delta_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Evaluates acceptance rate across multiple degradation thresholds (pp).
    """
    records = []
    for th in thresholds:
        eval_df = evaluate_bit_budget_guidelines(df, threshold_pp=th, task_delta_cols=task_delta_cols)
        total_evals = len(eval_df[eval_df["verdict"] != "NO_DATA"])
        n_acc = len(eval_df[eval_df["verdict"] == "ACCEPTABLE"])
        n_mar = len(eval_df[eval_df["verdict"] == "MARGINAL"])
        n_rej = len(eval_df[eval_df["verdict"] == "REJECTED"])

        records.append({
            "threshold_pp": th,
            "total_tested": total_evals,
            "acceptable_count": n_acc,
            "acceptable_pct": round(n_acc / total_evals * 100.0, 1) if total_evals > 0 else 0,
            "marginal_count": n_mar,
            "rejected_count": n_rej,
        })

    return pd.DataFrame(records)


def generate_task_budget_matrix(guidelines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes the minimum bit-width required to achieve acceptable degradation for each task category.
    """
    columns = ["Task", "Capability Category", "Min Acceptable Bits", "Confidence Tier", "Best 4-Bit Format"]
    if guidelines_df.empty or "task" not in guidelines_df.columns:
        return pd.DataFrame(columns=columns)

    matrix = []

    for task in guidelines_df["task"].unique():
        sub = guidelines_df[guidelines_df["task"] == task]
        cat = TASK_CATEGORIES.get(task, "General")

        # 4-bit check
        four_bit = sub[sub["bits"] == 4]
        four_bit_acc = any(four_bit["verdict"] == "ACCEPTABLE")
        four_bit_mar = any(four_bit["verdict"] == "MARGINAL")

        # 8-bit check
        eight_bit = sub[sub["bits"] == 8]
        eight_bit_acc = any(eight_bit["verdict"] == "ACCEPTABLE")

        if four_bit_acc:
            rec_bits = 4
            confidence = "High (4-bit Acceptable with 95% CI)"
        elif four_bit_mar:
            rec_bits = 4
            confidence = "Marginal (4-bit Point <= 3 pp, CI spans threshold)"
        elif eight_bit_acc:
            rec_bits = 8
            confidence = "High (8-bit Required, 4-bit Fails)"
        else:
            rec_bits = 16
            confidence = "FP16 Required (Quantization Exceeds Budget)"

        # Identify best 4-bit method
        best_4bit_method = "n/a"
        if not four_bit.empty and four_bit["delta_pp"].notna().any():
            best_row = four_bit.sort_values(by="delta_pp", ascending=False).iloc[0]
            best_4bit_method = f"{best_row['method']} ({best_row['delta_pp']:+.2f} pp)"

        matrix.append({
            "Task": task,
            "Capability Category": cat,
            "Min Acceptable Bits": rec_bits,
            "Confidence Tier": confidence,
            "Best 4-Bit Format": best_4bit_method,
        })

    return pd.DataFrame(matrix)


def format_budget_table(df: pd.DataFrame) -> str:
    """
    Format decision matrix into a clean Markdown table.
    """
    return tabulate(df, headers="keys", tablefmt="github", showindex=False)


format_guidelines_table = format_budget_table
