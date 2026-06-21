# Statistical Analysis Plan

## Objective

Estimate and validate the performance of an ED triage CDS model for predicting critical deterioration within 24 hours of arrival.

## Primary Endpoint

Binary critical deterioration endpoint within 24 hours.

## Primary Metrics

- AUROC with 95% bootstrap confidence interval
- AUPRC
- Sensitivity at locked threshold
- Specificity at locked threshold
- Brier score
- Calibration curve

## Threshold Selection

Thresholds are selected on validation data to prioritize sensitivity. The operating threshold should be locked before prospective testing.

## Subgroup Analyses

Report metrics by:

- Sex
- Race/ethnicity
- Age band
- Language
- Arrival mode
- Insurance type, if available

## Prospective Silent Trial

Before live use, run the model silently for at least four weeks. Compare predictions to observed deterioration outcomes without influencing clinician workflow.
