"""Clinical model evaluation utilities."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


def threshold_metrics(y_true: pd.Series, y_prob: pd.Series, threshold: float) -> dict[str, float]:
    """Calculate binary classification metrics at a decision threshold.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Positive-class probabilities.
        threshold: Decision threshold.

    Returns:
        Dictionary with confusion-matrix-derived metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    return {
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


def bootstrap_auc_ci(
    y_true: pd.Series,
    y_prob: pd.Series,
    iterations: int = 300,
    random_state: int = 42,
) -> tuple[float, float]:
    """Estimate a nonparametric bootstrap confidence interval for AUROC.

    Args:
        y_true: Ground-truth labels.
        y_prob: Predicted probabilities.
        iterations: Number of bootstrap resamples.
        random_state: Random seed.

    Returns:
        Lower and upper 95% confidence bounds.
    """
    rng = np.random.default_rng(random_state)
    aucs: list[float] = []
    y_true_array = np.asarray(y_true)
    y_prob_array = np.asarray(y_prob)
    for _ in range(iterations):
        sample_indices = rng.integers(0, len(y_true_array), len(y_true_array))
        sample_true = y_true_array[sample_indices]
        if len(np.unique(sample_true)) < 2:
            continue
        aucs.append(roc_auc_score(sample_true, y_prob_array[sample_indices]))
    lower, upper = np.percentile(aucs, [2.5, 97.5])
    return float(lower), float(upper)


def evaluate_predictions(
    y_true: pd.Series,
    y_prob: pd.Series,
    threshold: float,
    bootstrap_iterations: int,
) -> dict[str, float]:
    """Calculate discrimination, calibration, and threshold metrics.

    Args:
        y_true: Ground-truth labels.
        y_prob: Positive-class probabilities.
        threshold: Decision threshold.
        bootstrap_iterations: Bootstrap iterations for confidence interval.

    Returns:
        Metrics dictionary.
    """
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    lower, upper = bootstrap_auc_ci(y_true, y_prob, bootstrap_iterations)
    metrics = {
        "auroc": auroc,
        "auroc_ci_lower": lower,
        "auroc_ci_upper": upper,
        "auprc": auprc,
        "brier_score": brier,
    }
    metrics.update(threshold_metrics(y_true, y_prob, threshold))
    return {key: float(value) if not isinstance(value, int) else value for key, value in metrics.items()}


def write_metrics(metrics: dict[str, float], output_path: Path) -> None:
    """Write metrics dictionary as formatted JSON.

    Args:
        metrics: Evaluation metrics.
        output_path: JSON output path.
    """
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2)


def plot_calibration(y_true: pd.Series, y_prob: pd.Series, output_path: Path) -> None:
    """Write a calibration plot to disk.

    Args:
        y_true: Ground-truth labels.
        y_prob: Predicted probabilities.
        output_path: PNG output path.
    """
    fraction_positive, mean_predicted = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(mean_predicted, fraction_positive, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event rate")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
