"""Subgroup fairness and safety audit utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score


def subgroup_report(
    frame: pd.DataFrame,
    y_true_col: str,
    y_prob_col: str,
    threshold: float,
    subgroup_cols: list[str],
) -> pd.DataFrame:
    """Create a subgroup performance report.

    Args:
        frame: DataFrame containing labels, probabilities, and subgroup columns.
        y_true_col: Column containing ground-truth binary labels.
        y_prob_col: Column containing model probabilities.
        threshold: Binary decision threshold.
        subgroup_cols: Columns to audit.

    Returns:
        Long-format subgroup metrics DataFrame.
    """
    records: list[dict[str, object]] = []
    for column in subgroup_cols:
        for value, group in frame.groupby(column, dropna=False):
            if len(group) < 20 or group[y_true_col].nunique() < 2:
                continue
            y_true = group[y_true_col]
            y_prob = group[y_prob_col]
            y_pred = (y_prob >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            records.append(
                {
                    "subgroup_column": column,
                    "subgroup_value": value,
                    "n": len(group),
                    "event_rate": y_true.mean(),
                    "auroc": roc_auc_score(y_true, y_prob),
                    "brier_score": brier_score_loss(y_true, y_prob),
                    "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
                    "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
                    "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
                    "specificity": tn / (tn + fp) if tn + fp else 0.0,
                }
            )
    return pd.DataFrame.from_records(records)


def fairness_summary(report: pd.DataFrame) -> dict[str, float]:
    """Summarize max subgroup metric gaps.

    Args:
        report: Subgroup metrics DataFrame.

    Returns:
        Dictionary with maximum AUROC and FNR gaps.
    """
    summaries: dict[str, float] = {}
    for column in report["subgroup_column"].unique():
        subset = report[report["subgroup_column"] == column]
        summaries[f"{column}_auroc_gap"] = float(subset["auroc"].max() - subset["auroc"].min())
        summaries[f"{column}_fnr_gap"] = float(
            subset["false_negative_rate"].max() - subset["false_negative_rate"].min()
        )
    return summaries
