"""
Analysis subpackage for task degradation, proxy evaluation, and bit-budget guidelines.
"""

from .degradation import compute_degradation, paired_bootstrap_ci, format_degradation_summary
from .spearman_proxy import (
    compute_spearman_proxy_analysis,
    bootstrap_spearman_ci,
    leave_one_out_spearman,
    format_spearman_table,
)
from .bit_budget import (
    evaluate_bit_budget_guidelines,
    run_sensitivity_sweep,
    generate_task_budget_matrix,
    format_budget_table,
)
from .failure_modes import (
    analyze_flip_dynamics,
    analyze_gsm8k_failure_modes,
    analyze_ifeval_categories,
)

__all__ = [
    "compute_degradation",
    "paired_bootstrap_ci",
    "format_degradation_summary",
    "compute_spearman_proxy_analysis",
    "bootstrap_spearman_ci",
    "leave_one_out_spearman",
    "format_spearman_table",
    "evaluate_bit_budget_guidelines",
    "run_sensitivity_sweep",
    "generate_task_budget_matrix",
    "format_budget_table",
    "analyze_flip_dynamics",
    "analyze_gsm8k_failure_modes",
    "analyze_ifeval_categories",
]
