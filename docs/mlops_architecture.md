# MLOps Architecture

```text
ED EHR Source Systems
  -> HL7/FHIR ingestion
  -> schema validation
  -> feature store
  -> model registry
  -> inference service
  -> explanation service
  -> nurse triage UI
  -> EHR writeback
  -> audit log warehouse
  -> monitoring dashboard
```

## Training Pipeline

1. Extract historical ED encounters.
2. Apply cohort definition and endpoint labeling.
3. Validate missingness and leakage.
4. Train model.
5. Evaluate discrimination, calibration, safety, and fairness.
6. Register approved model with version metadata.

## Deployment Pipeline

1. Deploy model to inference service.
2. Run synthetic transaction tests.
3. Run silent prospective trial.
4. Enable limited user cohort.
5. Monitor live metrics.
6. Roll back if safety thresholds are breached.
