"""
calibration.py
HELM-aligned confidence calibration measurement for LLM evaluations.
Computes:
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier Score (multi-class & binary)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10,
) -> Tuple[float, float, List[Dict[str, float]]]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        confidences: 1D array of predicted class probabilities for top predicted class (in [0, 1]).
        accuracies: 1D binary array where 1 = correct prediction, 0 = incorrect prediction.
        num_bins: Number of equal-width bins between 0 and 1 (default: 10).

    Returns:
        (ece, mce, bin_statistics)
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    accuracies = np.asarray(accuracies, dtype=np.float64)

    assert len(confidences) == len(accuracies), "Confidences and accuracies must have same length"
    n_samples = len(confidences)
    if n_samples == 0:
        return 0.0, 0.0, []

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_stats = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_count = int(np.sum(in_bin))

        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_error = abs(bin_acc - bin_conf)

            ece += (bin_count / n_samples) * bin_error
            mce = max(mce, bin_error)

            bin_stats.append({
                "bin_idx": i,
                "range": f"[{bin_lower:.2f}, {bin_upper:.2f}]",
                "count": bin_count,
                "accuracy": bin_acc,
                "confidence": bin_conf,
                "calibration_gap": bin_error,
            })
        else:
            bin_stats.append({
                "bin_idx": i,
                "range": f"[{bin_lower:.2f}, {bin_upper:.2f}]",
                "count": 0,
                "accuracy": 0.0,
                "confidence": 0.0,
                "calibration_gap": 0.0,
            })

    return float(ece), float(mce), bin_stats


def compute_brier_score(
    probs: np.ndarray,
    targets: np.ndarray,
) -> float:
    """
    Computes standard Brier Score:
    BS = (1 / N) * sum_{i=1}^N sum_{k=1}^K (p_{ik} - y_{ik})^2

    Args:
        probs: 2D array of shape (N, K) representing predicted probability distribution over K classes.
        targets: 1D array of ground truth class indices (0 to K-1).

    Returns:
        Scalar Brier score in [0, 2] (lower is better; 0 = perfect probability assignment).
    """
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)

    n_samples, n_classes = probs.shape
    if n_samples == 0:
        return 0.0

    # Create one-hot encoding for targets
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n_samples), targets] = 1.0

    # Mean squared error across all classes
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return float(brier)


def evaluate_calibration(
    option_loglikelihoods: List[List[float]],
    gold_indices: List[int],
    num_bins: int = 10,
) -> Dict[str, Union[float, List[Dict[str, float]]]]:
    """
    Computes calibration metrics directly from multiple-choice loglikelihood scores
    logged by lm-evaluation-harness.

    Args:
        option_loglikelihoods: List of N items, each containing loglikelihoods for K options.
        gold_indices: List of N integers representing the true option index.
        num_bins: Number of confidence bins (default 10).

    Returns:
        Dictionary with ece, mce, brier_score, and bin_stats.
    """
    n_samples = len(option_loglikelihoods)
    if n_samples == 0:
        return {"ece": 0.0, "mce": 0.0, "brier_score": 0.0, "bin_stats": []}

    # Convert log-likelihoods to probabilities via softmax
    all_probs = []
    top_confs = []
    top_correct = []

    for lls, gold in zip(option_loglikelihoods, gold_indices):
        lls_arr = np.array(lls, dtype=np.float64)
        # Numerically stable softmax
        exp_lls = np.exp(lls_arr - np.max(lls_arr))
        probs = exp_lls / np.sum(exp_lls)
        all_probs.append(probs)

        pred_idx = int(np.argmax(probs))
        top_confs.append(probs[pred_idx])
        top_correct.append(1 if pred_idx == gold else 0)

    all_probs = np.array(all_probs)
    gold_indices = np.array(gold_indices)
    top_confs = np.array(top_confs)
    top_correct = np.array(top_correct)

    ece, mce, bin_stats = compute_ece(top_confs, top_correct, num_bins=num_bins)
    brier = compute_brier_score(all_probs, gold_indices)

    return {
        "ece": ece,
        "mce": mce,
        "brier_score": brier,
        "mean_confidence": float(np.mean(top_confs)),
        "accuracy": float(np.mean(top_correct)),
        "bin_stats": bin_stats,
    }

