"""Model training utilities for triage deterioration prediction."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_COLUMNS, VITAL_COLUMNS


def build_model(random_state: int = 42) -> Pipeline:
    """Build a triage risk model pipeline.

    Args:
        random_state: Random seed for deterministic training behavior.

    Returns:
        Scikit-learn Pipeline including preprocessing and classifier.
    """
    numeric_features = [
        *VITAL_COLUMNS,
        "flag_hypotension",
        "flag_tachycardia",
        "flag_tachypnea",
        "flag_hypoxia",
        "flag_fever",
        "flag_elderly",
    ]
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def train_model(features: pd.DataFrame, target: pd.Series, random_state: int = 42) -> Pipeline:
    """Train a triage risk prediction model.

    Args:
        features: Feature matrix.
        target: Binary target indicating critical deterioration within 24 hours.
        random_state: Random seed.

    Returns:
        Fitted scikit-learn Pipeline.
    """
    model = build_model(random_state=random_state)
    model.fit(features, target)
    return model


def predict_risk(model: Pipeline, features: pd.DataFrame) -> pd.Series:
    """Predict deterioration probability for ED encounters.

    Args:
        model: Fitted model pipeline.
        features: Feature matrix.

    Returns:
        Probability Series for the positive class.
    """
    probabilities = model.predict_proba(features)[:, 1]
    return pd.Series(probabilities, index=features.index, name="risk_probability")


def model_metadata(model: Pipeline) -> dict[str, Any]:
    """Return lightweight model metadata for governance logging.

    Args:
        model: Fitted model pipeline.

    Returns:
        Dictionary with model class and preprocessing information.
    """
    return {
        "model_type": model.named_steps["classifier"].__class__.__name__,
        "preprocessor_type": model.named_steps["preprocessor"].__class__.__name__,
        "supports_predict_proba": hasattr(model, "predict_proba"),
    }
