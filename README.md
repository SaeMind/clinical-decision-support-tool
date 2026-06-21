# Clinical Decision Support Tool for Emergency Medicine Triage

A portfolio-grade Clinical Data Science artifact specifying and prototyping an AI-augmented emergency department triage decision support system. The system estimates risk of critical deterioration and recommends triage priority while preserving nurse judgment through human-in-the-loop override workflows.

## Portfolio Objective

This repository demonstrates clinical AI product design, healthcare ML validation, MLOps architecture, fairness auditing, regulatory framing, and health-economic impact modeling for emergency medicine triage.

## Repository Structure

```text
clinical-triage-cds-tool/
├── configs/
│   └── settings.yaml
├── data/
│   └── synthetic_ed_triage.csv
├── docs/
│   ├── PRD.md
│   ├── regulatory_compliance.md
│   ├── statistical_analysis_plan.md
│   ├── mlops_architecture.md
│   └── bibliography.md
├── outputs/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_generator.py
│   ├── features.py
│   ├── model.py
│   ├── evaluation.py
│   ├── fairness.py
│   ├── economics.py
│   ├── explainability.py
│   └── main.py
├── tests/
│   ├── test_features.py
│   └── test_economics.py
├── .env.example
├── requirements.txt
└── README.md
```

## Core Clinical Framing

The stronger endpoint is not direct ESI imitation. The model estimates short-horizon clinical risk:

- Critical deterioration within 24 hours
- ICU admission
- Sepsis/stroke/MI flags
- Hospital admission likelihood
- Human-readable triage recommendation derived from risk bands

## How to Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main --generate-data --train --evaluate
```

Outputs are written to `/outputs`:

- `metrics.json`
- `fairness_report.csv`
- `economic_impact.json`
- `recommendations_sample.csv`
- `calibration_plot.png`

## Validation Focus

The evaluation pipeline reports:

- AUROC and AUPRC
- Sensitivity/specificity at configured threshold
- Bootstrap confidence intervals
- Calibration slope/intercept and Brier score
- Subgroup AUROC, false negative rate, and calibration by age, sex, and race/ethnicity
- Decision-support recommendation bands

## Regulatory Positioning

This artifact frames the tool as human-in-the-loop clinical decision support. It includes SaMD/CDS boundary analysis, HIPAA controls, auditability, model governance, rollback criteria, and post-deployment monitoring.

## Disclaimer

This repository is for portfolio demonstration only. It is not a medical device, not validated for patient care, and must not be used for clinical decision-making.
