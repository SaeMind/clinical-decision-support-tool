"""Synthetic emergency department triage data generator.

The generated data are intentionally synthetic and are used only for portfolio
workflow demonstration. They approximate plausible relationships between vital
signs, chief complaints, demographics, and short-horizon clinical risk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve_path


COMPLAINTS = [
    "chest pain",
    "shortness of breath",
    "abdominal pain",
    "fever",
    "headache",
    "weakness",
    "fall",
    "syncope",
    "stroke symptoms",
    "wound check",
    "medication refill",
]

RACES = ["Asian", "Black", "Hispanic", "White", "Other"]
SEXES = ["Female", "Male"]


def sigmoid(value: np.ndarray) -> np.ndarray:
    """Apply the logistic sigmoid transformation.

    Args:
        value: Numeric input array.

    Returns:
        Array transformed to the interval [0, 1].
    """
    return 1 / (1 + np.exp(-value))


def generate_synthetic_ed_data(
    rows: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic ED triage observations.

    Args:
        rows: Number of synthetic encounters to generate.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with demographics, vital signs, complaints, outcomes, and
        nurse-assigned ESI labels.
    """
    rng = np.random.default_rng(random_state)
    age = np.clip(rng.normal(49, 22, rows), 0, 98).round().astype(int)
    sex = rng.choice(SEXES, size=rows, p=[0.53, 0.47])
    race_ethnicity = rng.choice(RACES, size=rows, p=[0.08, 0.22, 0.25, 0.38, 0.07])
    chief_complaint = rng.choice(
        COMPLAINTS,
        size=rows,
        p=[0.13, 0.12, 0.14, 0.12, 0.10, 0.09, 0.08, 0.07, 0.05, 0.06, 0.04],
    )

    sbp = rng.normal(128, 24, rows).round().astype(int)
    heart_rate = rng.normal(88, 22, rows).round().astype(int)
    respiratory_rate = rng.normal(18, 5, rows).round().astype(int)
    oxygen_saturation = np.clip(rng.normal(96, 4, rows), 70, 100).round().astype(int)
    temperature_c = np.round(rng.normal(37.0, 0.8, rows), 1)
    comorbidity_count = rng.poisson(np.clip(age / 35, 0.1, 3.5), rows)

    high_risk_complaint = np.isin(
        chief_complaint,
        ["chest pain", "shortness of breath", "syncope", "stroke symptoms"],
    ).astype(int)
    infection_complaint = np.isin(chief_complaint, ["fever", "abdominal pain"]).astype(int)

    risk_logit = (
        -5.2
        + 0.030 * age
        + 1.05 * high_risk_complaint
        + 0.65 * infection_complaint
        + 0.55 * comorbidity_count
        + 1.10 * (heart_rate > 120)
        + 0.80 * (respiratory_rate > 24)
        + 1.35 * (oxygen_saturation < 90)
        + 1.25 * (sbp < 90)
        + 0.80 * (temperature_c >= 38.3)
        + 0.018 * np.maximum(heart_rate - 90, 0)
        + 0.030 * np.maximum(95 - oxygen_saturation, 0)
    )
    deterioration_probability = sigmoid(risk_logit)
    critical_deterioration_24h = rng.binomial(1, deterioration_probability)
    icu_admission = rng.binomial(1, np.clip(deterioration_probability * 0.55, 0, 0.95))
    hospital_admission = rng.binomial(1, np.clip(deterioration_probability * 1.75, 0, 0.95))

    esi_score = np.select(
        [
            deterioration_probability >= 0.70,
            deterioration_probability >= 0.42,
            deterioration_probability >= 0.20,
            deterioration_probability >= 0.08,
        ],
        [1, 2, 3, 4],
        default=5,
    )
    label_noise = rng.choice([-1, 0, 1], size=rows, p=[0.06, 0.86, 0.08])
    nurse_esi = np.clip(esi_score + label_noise, 1, 5).astype(int)

    return pd.DataFrame(
        {
            "encounter_id": [f"ED{i:06d}" for i in range(rows)],
            "age": age,
            "sex": sex,
            "race_ethnicity": race_ethnicity,
            "chief_complaint": chief_complaint,
            "systolic_bp": sbp,
            "heart_rate": heart_rate,
            "respiratory_rate": respiratory_rate,
            "oxygen_saturation": oxygen_saturation,
            "temperature_c": temperature_c,
            "comorbidity_count": comorbidity_count,
            "nurse_esi": nurse_esi,
            "critical_deterioration_24h": critical_deterioration_24h,
            "icu_admission": icu_admission,
            "hospital_admission": hospital_admission,
        }
    )


def write_synthetic_data(path: str | Path, rows: int = 5000) -> Path:
    """Generate and write synthetic ED data to CSV.

    Args:
        path: Output CSV path relative to the project root or absolute path.
        rows: Number of synthetic rows to generate.

    Returns:
        Resolved output path.
    """
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_synthetic_ed_data(rows=rows)
    frame.to_csv(output_path, index=False)
    return output_path
