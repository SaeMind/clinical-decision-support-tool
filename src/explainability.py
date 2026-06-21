"""Lightweight rule-based explanation layer for triage recommendations."""

from __future__ import annotations

import pandas as pd


def top_risk_factors(row: pd.Series, max_factors: int = 5) -> list[str]:
    """Return interpretable risk drivers for a single encounter.

    Args:
        row: Encounter row containing vital signs and complaint fields.
        max_factors: Maximum number of explanation factors to return.

    Returns:
        Ordered list of human-readable risk factors.
    """
    factors: list[str] = []
    if row["oxygen_saturation"] < 90:
        factors.append("oxygen saturation below 90%")
    if row["systolic_bp"] < 90:
        factors.append("systolic blood pressure below 90 mmHg")
    if row["heart_rate"] > 120:
        factors.append("heart rate above 120 bpm")
    if row["respiratory_rate"] > 24:
        factors.append("respiratory rate above 24 breaths/min")
    if row["temperature_c"] >= 38.3:
        factors.append("temperature consistent with fever")
    if row["chief_complaint"] in {"chest pain", "shortness of breath", "stroke symptoms", "syncope"}:
        factors.append(f"high-risk chief complaint: {row['chief_complaint']}")
    if row["age"] >= 75:
        factors.append("age 75 years or older")
    if row["comorbidity_count"] >= 3:
        factors.append("three or more documented comorbidities")
    return factors[:max_factors]
