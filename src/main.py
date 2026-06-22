"""Command-line entrypoint for the clinical triage CDS portfolio artifact."""

from __future__ import annotations

import argparse
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ensure_output_dir, load_settings, resolve_path
from .data_generator import write_synthetic_data
from .economics import economic_impact
from .evaluation import evaluate_predictions, plot_calibration, write_metrics
from .explainability import top_risk_factors
from .fairness import fairness_summary, subgroup_report
from .features import TARGET_COLUMN, derive_triage_recommendation, split_features_target
from .model import model_metadata, predict_risk, train_model


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Run ED triage CDS demonstration pipeline.")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic ED data.")
    parser.add_argument("--train", action="store_true", help="Train model.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model and write outputs.")
    parser.add_argument("--rows", type=int, default=5000, help="Synthetic rows to generate.")
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    """Run data generation, training, evaluation, and reporting.

    Args:
        args: Parsed CLI arguments.
    """
    settings = load_settings()
    output_dir = ensure_output_dir(settings)
    data_path = resolve_path(settings["paths"]["data"])

    if args.generate_data or not data_path.exists():
        write_synthetic_data(data_path, rows=args.rows)

    frame = pd.read_csv(data_path)
    features, target = split_features_target(frame)
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        target,
        test_size=settings["model"]["test_size"],
        stratify=target,
        random_state=settings["model"]["random_state"],
    )

    if not args.train:
        return

    model = train_model(train_x, train_y, random_state=settings["model"]["random_state"])

    if not args.evaluate:
        return

    risk_prob = predict_risk(model, test_x)
    metrics = evaluate_predictions(
        test_y,
        risk_prob,
        threshold=settings["model"]["risk_threshold"],
        bootstrap_iterations=settings["validation"]["bootstrap_iterations"],
    )
    metrics.update(model_metadata(model))
    write_metrics(metrics, output_dir / "metrics.json")
    plot_calibration(test_y, risk_prob, output_dir / "calibration_plot.png")

    audit_frame = frame.loc[test_x.index].copy()
    audit_frame["risk_probability"] = risk_prob
    audit_frame[TARGET_COLUMN] = test_y
    fair_report = subgroup_report(
        audit_frame,
        y_true_col=TARGET_COLUMN,
        y_prob_col="risk_probability",
        threshold=settings["model"]["risk_threshold"],
        subgroup_cols=["sex", "race_ethnicity"],
    )
    fair_report.to_csv(output_dir / "fairness_report.csv", index=False)
    with (output_dir / "fairness_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(fairness_summary(fair_report), file_obj, indent=2)

    impact = economic_impact(settings)
    with (output_dir / "economic_impact.json").open("w", encoding="utf-8") as file_obj:
        json.dump(impact, file_obj, indent=2)

    samples = audit_frame.head(25).copy()
    samples["recommendation"] = samples["risk_probability"].apply(derive_triage_recommendation)
    samples["explanation"] = samples.apply(lambda row: "; ".join(top_risk_factors(row)), axis=1)
    samples.to_csv(output_dir / "recommendations_sample.csv", index=False)


def main() -> None:
    """Execute the pipeline from CLI arguments."""
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
