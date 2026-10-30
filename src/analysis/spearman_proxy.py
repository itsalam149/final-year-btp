"""
spearman_proxy.py
─────────────────────────────────────────────────────────────────────────────
Direct implementation for Research Question 2 (RQ2):
Evaluates the reliability of perplexity as a proxy for downstream
task accuracy degradation under post-training quantization.

Methodology (v2 Plan):
  - Computes Spearman's rank correlation coefficient (ρ)
  - Computes Bootstrap 95% Confidence Interval on ρ (2,000 resamples)
  - Conducts Leave-One-Out (LOO) sensitivity check to identify leverage points
  - Supports dual corpora comparison: WikiText-2 (primary) vs C4 (control)

Pre-registered Interpretation:
  - FAITHFUL PROXY : ρ <= -0.70 AND 95% CI upper bound < 0
  - DECOUPLED      : |ρ| < 0.40 OR 95% CI spans 0 (PPL hides damage)
  - INTERMEDIATE   : -0.70 < ρ <= -0.40

Group ID: DTU/CSE/BTP/GP159 | DTU 2026-27
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate


def bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes bootstrap confidence interval for Spearman's rho.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)

    if n < 4:
        res = stats.spearmanr(x, y)
        rho = float(res.statistic if hasattr(res, "statistic") else res[0])
        return rho, np.nan, np.nan

    res = stats.spearmanr(x, y)
    rho_point = float(res.statistic if hasattr(res, "statistic") else res[0])

    rng = np.random.default_rng(seed)
    boot_rhos = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bx, by = x[idx], y[idx]
        # Avoid rank ties collapsing when all resampled points are identical
        if len(np.unique(bx)) < 2 or len(np.unique(by)) < 2:
            continue
        r = stats.spearmanr(bx, by)
        val = r.statistic if hasattr(r, "statistic") else r[0]
        if not np.isnan(val):
            boot_rhos.append(val)

    if len(boot_rhos) < 100:
        return rho_point, np.nan, np.nan

    alpha = 1.0 - confidence_level
    ci_low = float(np.percentile(boot_rhos, 100.0 * (alpha / 2.0)))
    ci_high = float(np.percentile(boot_rhos, 100.0 * (1.0 - alpha / 2.0)))

    return rho_point, ci_low, ci_high


def leave_one_out_spearman(
    x: np.ndarray,
    y: np.ndarray,
    variant_labels: Optional[List[str]] = None,
) -> Tuple[float, float, Optional[str]]:
    """
    Computes Leave-One-Out (LOO) rho range to check if a single variant drives the result.
    Returns (min_rho_loo, max_rho_loo, most_influential_variant).
    """
    n = len(x)
    if n <= 3:
        return np.nan, np.nan, None

    loo_rhos = []
    for i in range(n):
        sub_x = np.delete(x, i)
        sub_y = np.delete(y, i)
        r = stats.spearmanr(sub_x, sub_y)
        val = float(r.statistic if hasattr(r, "statistic") else r[0])
        loo_rhos.append(val)

    base_r = stats.spearmanr(x, y)
    base_val = float(base_r.statistic if hasattr(base_r, "statistic") else base_r[0])

    diffs = [abs(val - base_val) for val in loo_rhos]
    max_idx = int(np.argmax(diffs))
    influential = variant_labels[max_idx] if variant_labels else f"index_{max_idx}"

    return float(min(loo_rhos)), float(max(loo_rhos)), influential


def compute_spearman_proxy_analysis(
    df: pd.DataFrame,
    ppl_col: str = "delta_wikitext",
    task_delta_cols: Optional[List[str]] = None,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Compute rigorous Spearman correlation for each downstream task against perplexity delta.

    Parameters
    ----------
    df : pd.DataFrame with delta columns computed by degradation.py
    ppl_col : column name representing change in perplexity (ΔPPL)
    task_delta_cols : list of task accuracy delta column names
    n_bootstrap : iterations for bootstrap CI
    confidence_level : confidence level for CI

    Returns
    -------
    pd.DataFrame containing: task, spearman_rho, ci_low, ci_high, loo_min, loo_max,
                            influential_variant, p_value, sample_size, proxy_status
    """
    if task_delta_cols is None:
        task_delta_cols = [
            c for c in df.columns
            if c.startswith("delta_")
            and not c.endswith("_ci_low")
            and not c.endswith("_ci_high")
            and c != ppl_col
            and "c4" not in c
        ]

    records = []

    for col in task_delta_cols:
        task_name = col.replace("delta_", "")
        required_cols = [ppl_col, col]
        has_variant = "method" in df.columns

        cols_to_extract = required_cols + (["method"] if has_variant else [])
        valid_data = df[cols_to_extract].dropna(subset=required_cols)
        n = len(valid_data)

        if n < 3:
            records.append({
                "task": task_name,
                "spearman_rho": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "loo_min": np.nan,
                "loo_max": np.nan,
                "influential_variant": None,
                "p_value": np.nan,
                "sample_size": n,
                "proxy_status": "INSUFFICIENT_DATA",
            })
            continue

        x = valid_data[ppl_col].values
        y = valid_data[col].values
        var_labels = valid_data["method"].tolist() if has_variant else None

        res = stats.spearmanr(x, y)
        rho = float(res.statistic if hasattr(res, "statistic") else res[0])
        p_val = float(res.pvalue if hasattr(res, "pvalue") else res[1])

        rho_pt, ci_lo, ci_hi = bootstrap_spearman_ci(
            x, y, n_bootstrap=n_bootstrap, confidence_level=confidence_level
        )
        loo_min, loo_max, inf_var = leave_one_out_spearman(x, y, variant_labels=var_labels)

        # Pre-registered decision rule (v2 plan Section 6.2)
        if rho <= -0.70 and (np.isnan(ci_hi) or ci_hi < 0.0):
            status = "FAITHFUL_PROXY"
        elif abs(rho) < 0.40 or (not np.isnan(ci_lo) and not np.isnan(ci_hi) and ci_lo <= 0.0 <= ci_hi):
            status = "DECOUPLED (PPL HIDES DAMAGE)"
        elif -0.70 < rho <= -0.40:
            status = "INTERMEDIATE"
        else:
            status = "UNRELIABLE / INVERSE"

        records.append({
            "task": task_name,
            "spearman_rho": round(rho, 4),
            "ci_95": f"[{ci_lo:.2f}, {ci_hi:.2f}]" if not np.isnan(ci_lo) else "n/a",
            "loo_range": f"[{loo_min:.2f}, {loo_max:.2f}]" if not np.isnan(loo_min) else "n/a",
            "influential_variant": inf_var,
            "p_value": round(p_val, 4),
            "sample_size": n,
            "proxy_status": status,
        })

    return pd.DataFrame(records)


def format_spearman_table(spearman_df: pd.DataFrame) -> str:
    """
    Format Spearman proxy results as a publication-ready Markdown table.
    """
    return tabulate(spearman_df, headers="keys", tablefmt="github", showindex=False)
