"""Tests for clinical feature engineering utilities."""

import pandas as pd

from src.features import add_clinical_flags, derive_triage_recommendation


def test_add_clinical_flags_identifies_abnormal_vitals():
    """Clinical flags should identify severe abnormal vital signs."""
    frame = pd.DataFrame(
        {
            "systolic_bp": [88],
            "heart_rate": [130],
            "respiratory_rate": [28],
            "oxygen_saturation": [86],
            "temperature_c": [38.8],
            "age": [79],
        }
    )
    enriched = add_clinical_flags(frame)
    assert enriched.loc[0, "flag_hypotension"] == 1
    assert enriched.loc[0, "flag_tachycardia"] == 1
    assert enriched.loc[0, "flag_tachypnea"] == 1
    assert enriched.loc[0, "flag_hypoxia"] == 1
    assert enriched.loc[0, "flag_fever"] == 1
    assert enriched.loc[0, "flag_elderly"] == 1


def test_derive_triage_recommendation_returns_urgent_for_high_probability():
    """High risk probabilities should map to urgent recommendations."""
    assert derive_triage_recommendation(0.8).startswith("urgent")
