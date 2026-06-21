"""Feature engineering for emergency medicine triage risk models."""

from __future__ import annotations

import pandas as pd

VITAL_COLUMNS = [
    "age",
    "systolic_bp",
    "heart_rate",
    "respiratory_rate",
    "oxygen_saturation",
    "temperature_c",
    "comorbidity_count",
]
CATEGORICAL_COLUMNS = ["sex", "race_ethnicity", "chief_complaint"]
TARGET_COLUMN = "critical_deterioration_24h"


def add_clinical_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable clinical vital sign flags.

    Args:
        frame: Raw ED encounter DataFrame.

    Returns:
        DataFrame with additional binary clinical flags.
    """
    enriched = frame.copy()
    enriched["flag_hypotension"] = (enriched["systolic_bp"] < 90).astype(int)
    enriched["flag_tachycardia"] = (enriched["heart_rate"] > 120).astype(int)
    enriched["flag_tachypnea"] = (enriched["respiratory_rate"] > 24).astype(int)
    enriched["flag_hypoxia"] = (enriched["oxygen_saturation"] < 90).astype(int)
    enriched["flag_fever"] = (enriched["temperature_c"] >= 38.3).astype(int)
    enriched["flag_elderly"] = (enriched["age"] >= 75).astype(int)
    return enriched


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into model features and target.

    Args:
        frame: ED encounter DataFrame containing the target column.

    Returns:
        Tuple of feature DataFrame and target Series.
    """
    enriched = add_clinical_flags(frame)
    feature_columns = [
        *VITAL_COLUMNS,
        *CATEGORICAL_COLUMNS,
        "flag_hypotension",
        "flag_tachycardia",
        "flag_tachypnea",
        "flag_hypoxia",
        "flag_fever",
        "flag_elderly",
    ]
    return enriched[feature_columns], enriched[TARGET_COLUMN]


def derive_triage_recommendation(probability: float) -> str:
    """Map deterioration probability to a clinical support recommendation.

    Args:
        probability: Estimated probability of critical deterioration.

    Returns:
        Human-readable triage support band.
    """
    if probability >= 0.65:
        return "urgent: immediate clinician evaluation recommended"
    if probability >= 0.35:
        return "semi-urgent: accelerated triage review recommended"
    if probability >= 0.15:
        return "routine: standard triage with reassessment"
    return "low acuity: standard triage pathway"
