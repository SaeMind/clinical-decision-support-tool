# Product Requirements Document
## Clinical Decision Support Tool for Emergency Medicine Triage

**Version:** 2.0  
**Date:** June 2026  
**Status:** Portfolio Review Artifact  
**Audience:** Clinical informaticists, emergency medicine leadership, ML engineering teams, healthcare AI reviewers

---

## Executive Summary

This PRD specifies a human-in-the-loop clinical decision support system for emergency department triage. The system estimates short-horizon risk of critical deterioration and converts that probability into triage support recommendations for nurses and physicians. The system is intentionally designed to augment, not replace, clinical judgment.

The updated clinical target is superior to direct Emergency Severity Index imitation. Rather than only predicting another nurse's label, the tool estimates clinically meaningful outcomes such as critical deterioration within 24 hours, ICU admission, hospital admission, and time-sensitive syndromes such as sepsis, stroke, and myocardial infarction.

### Primary Product Goal

Reduce under-recognition of high-risk ED presentations while improving triage throughput and preserving clinician authority.

### Success Targets

| Domain | Target |
|---|---:|
| Inference latency | p95 < 2 seconds |
| AUROC for critical deterioration | >= 0.88 |
| Sensitivity at operating threshold | >= 0.95 for high-risk cases |
| Calibration | Brier score monitored monthly; reliability curve reviewed quarterly |
| Clinician agreement | >= 80% |
| Override rate | 10% to 20% expected operating range |
| Fairness | Max subgroup AUROC gap <= 0.05; max FNR gap <= 5 percentage points |
| Safety | No increase in preventable adverse events attributable to CDS use |

---

## 1. Product Overview

### 1.1 Product Name

**TriageAI Clinical Decision Support System**

### 1.2 Clinical Context

Emergency department triage requires rapid identification of unstable or potentially unstable patients. Triage nurses operate under high cognitive load, incomplete information, crowding pressure, and time constraints. A decision support system can standardize risk recognition, surface abnormal vital sign patterns, and provide explainable risk bands.

### 1.3 Primary Users

#### ED Triage Nurse

Uses the tool during initial intake. Inputs chief complaint, vitals, demographics, and brief history. Reviews risk recommendation and either accepts or overrides the suggestion.

#### ED Physician or APP

Reviews the recommendation in the patient summary, particularly for high-risk or discordant cases.

#### ED Leadership

Monitors throughput, override patterns, safety events, model drift, and subgroup performance.

---

## 2. Clinical Problem

### 2.1 Triage Variability

Manual triage can vary between nurses because acuity assignment depends on incomplete early information and subjective risk interpretation. This variability can lead to undertriage of unstable patients and overtriage of low-acuity patients.

### 2.2 Operational Bottleneck

Initial triage can consume 8 to 12 minutes per patient in high-volume settings. If a system reduces triage time to 3 to 4 minutes without degrading safety, annual capacity gains can be substantial.

### 2.3 Safety Risk

Undertriage can delay recognition of sepsis, myocardial infarction, stroke, respiratory failure, occult shock, or neurologic deterioration. The model's highest-priority objective is minimizing false negatives among high-risk patients.

---

## 3. Clinical Target Definition

### 3.1 Primary Prediction Target

**Critical deterioration within 24 hours of ED arrival**, defined as any of the following:

- ICU admission
- Intubation or non-invasive ventilation initiation
- Vasopressor initiation
- Cardiac arrest
- Death
- Rapid response or equivalent escalation event

### 3.2 Secondary Targets

- Hospital admission
- ICU admission
- Sepsis within 6 hours
- Stroke activation
- Myocardial infarction pathway activation
- ESI/nurse triage label as a secondary comparator, not the primary endpoint

### 3.3 Recommendation Mapping

| Estimated Risk | Recommendation |
|---:|---|
| >= 0.65 | Urgent: immediate clinician evaluation recommended |
| 0.35 to 0.64 | Semi-urgent: accelerated triage review recommended |
| 0.15 to 0.34 | Routine: standard triage with reassessment |
| < 0.15 | Low acuity: standard triage pathway |

---

## 4. Data Requirements

### 4.1 Candidate Datasets

The preferred public-data foundation is **MIMIC-IV-ED**, not MIMIC-III, because it is more appropriate for ED-specific workflows. If ESI labels or desired deterioration targets are incomplete, labels should be constructed from downstream events using reproducible rules.

### 4.2 Required Fields

| Category | Fields |
|---|---|
| Demographics | Age, sex, race/ethnicity, language when available |
| Presentation | Chief complaint, arrival mode, pain score |
| Vitals | SBP, DBP, HR, RR, SpO2, temperature |
| History | Problem list, comorbidity count, medication flags |
| Outcomes | ICU admission, hospital admission, death, escalation events |
| Workflow | Nurse triage label, timestamps, override logs |

### 4.3 Data Quality Requirements

- Missingness quantified for every feature
- Imputation strategy documented
- Outcome definitions versioned
- Data leakage audit before model training
- Site-specific representativeness assessment prior to deployment

---

## 5. Technical Architecture

### 5.1 Model Components

1. **Chief Complaint NLP**
   - Baseline: bag-of-complaints one-hot encoding
   - Advanced: ClinicalBERT/BioClinicalBERT text embeddings

2. **Structured Vital Sign Model**
   - Gradient boosted trees for structured features
   - Rule-based flags for severe vital abnormalities

3. **Risk Aggregation**
   - Calibrated probability model
   - Recommendation band mapping
   - Confidence and uncertainty outputs

4. **Explanation Layer**
   - Top risk drivers
   - SHAP for structured features
   - Calibration plot
   - Counterfactual examples for review-only workflows

### 5.2 MLOps Architecture

```text
FHIR/HL7 feed
    -> ingestion service
    -> feature validation
    -> feature store
    -> model inference service
    -> explanation service
    -> nurse-facing UI
    -> EHR writeback
    -> monitoring + audit logs
```

### 5.3 Production Monitoring

- AUROC and AUPRC on adjudicated outcome windows
- Sensitivity and false negative rate at operating threshold
- Calibration drift
- Subgroup AUROC and FNR drift
- Override rate by nurse, shift, site, and chief complaint
- Latency p50/p95/p99
- Downtime and fallback activation rate

---

## 6. Functional Requirements

| ID | Requirement | Acceptance Criteria |
|---|---|---|
| F1 | Generate risk probability | p95 latency < 2 seconds |
| F2 | Generate triage support band | Deterministic mapping from calibrated risk probability |
| F3 | Explain risk drivers | Display 3 to 5 human-readable risk factors |
| F4 | Support override | Nurse can override in < 10 seconds with reason capture |
| F5 | Write audit log | All recommendations, user actions, timestamps, and model versions stored |
| F6 | Fallback mode | If model unavailable, interface defaults to manual triage required |

---

## 7. Non-Functional Requirements

### Performance

- p95 inference latency < 2 seconds
- p99 latency < 5 seconds
- 99.5% availability minimum during pilot

### Security

- TLS 1.3 in transit
- AES-256 at rest
- Role-based access control
- PHI minimization
- Full audit trail

### Reliability

- Model versioning
- Reproducible training pipeline
- Automated rollback criteria
- Drift alerts

---

## 8. Statistical Analysis Plan

### Primary Metrics

- AUROC with 95% bootstrap confidence interval
- AUPRC
- Sensitivity and specificity at locked operating threshold
- PPV and NPV
- Brier score
- Calibration curve
- Decision curve analysis

### Prospective Silent Trial

The first deployment phase must run silently. Recommendations are logged but not shown to nurses. This estimates performance under real site-specific data distribution before clinical influence.

### Go/No-Go Criteria

- AUROC >= 0.85 in silent trial
- Sensitivity >= 0.92 for deterioration endpoint
- No subgroup FNR gap > 5 percentage points
- No data leakage or systematic missingness issue

---

## 9. Fairness and Equity

The system must report performance across race/ethnicity, sex, age group, language, insurance type, and arrival mode when available.

Required audits:

- Subgroup AUROC
- Subgroup calibration
- False negative rate gap
- False positive rate gap
- Equalized odds review
- Calibration within groups

---

## 10. Regulatory and Compliance Strategy

The system is positioned as human-in-the-loop clinical decision support. Because it influences triage prioritization, regulatory classification must be evaluated before production deployment.

### Key Questions

- Does the clinician have independent basis to review the recommendation?
- Are explanations sufficiently transparent?
- Is the output merely informational or does it drive time-sensitive clinical action?
- Does the tool meet the definition of Software as a Medical Device?

### Required Controls

- HIPAA-compliant deployment environment
- Business associate agreements for vendors
- Model card and data sheet
- Audit logs retained according to institutional policy
- Change-control process for model updates

---

## 11. Health-Economic Impact

### Example Annual Impact Model

For a 100,000-visit ED:

- Reducing LWBS from 4.8% to 3.5% prevents 1,300 incomplete visits.
- At $275 net revenue per completed visit, estimated retained revenue is $357,500.
- Reducing triage time from 10 to 4 minutes saves 600,000 nurse minutes, equivalent to 10,000 hours.
- At $62 loaded hourly nursing cost, labor capacity value is $620,000.

Estimated total annual operational value: **$977,500** before implementation costs.

---

## 12. Rollout Plan

| Phase | Duration | Description |
|---|---:|---|
| Retrospective validation | 8 weeks | Train and validate on historical data |
| Silent prospective trial | 4 weeks | Log predictions without surfacing recommendations |
| Limited pilot | 4 weeks | Display recommendations to selected nurse cohort |
| Expanded pilot | 8 weeks | Full triage area deployment with monitoring |
| Governance review | Ongoing | Quarterly model, fairness, and safety review |

---

## 13. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Automation bias | Medium | High | Human override, confidence bands, training, audit |
| Dataset shift | High | High | Silent trial and drift monitoring |
| Subgroup underperformance | Medium | High | Fairness constraints and subgroup audits |
| EHR integration failure | Medium | Medium | Early HL7/FHIR testing |
| Poor calibration | Medium | High | Calibration monitoring and threshold locking |
| Alert fatigue | Medium | Medium | Recommendation band tuning and override analysis |

---

## 14. Portfolio Implementation Notes

This repository includes a synthetic data generator, model training pipeline, evaluation scripts, fairness auditing, economic impact modeling, and governance documentation. The code is not clinical-grade but demonstrates the architecture and validation mindset expected for a Clinical Data Science portfolio artifact.
