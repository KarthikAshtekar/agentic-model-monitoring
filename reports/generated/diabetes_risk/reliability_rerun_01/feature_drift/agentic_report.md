# Agentic monitoring report: feature_drift

## Execution provenance

- Provider: `groq`
- Model: `openai/gpt-oss-20b`
- Execution mode: `live`
- Fake LLM: `false`
- Run ID: `MON-DIABETES-RISK-FEATURE-DRIFT-44994E7FD981`
- Thread ID: `agentic-mon-diabetes-risk-feature-drift-44994e7fd981`
- Created at UTC: `2026-07-25T11:31:38.044271+00:00`
- Registered model: `diabetes_risk`
- Model: `BRFSS Diabetes Risk XGBoost`
- Domain: `diabetes_screening`
- Task: `binary_classification`
- Bundle mode: `live_inference`

## Repeat reliability provenance

- Evaluation type: `repeat_reliability_run`
- Replaces original evaluation: `false`
- Original result preserved: `true`
- Model ID: `diabetes_risk`
- Scenario: `feature_drift`
- Provider: `groq`
- Provider model: `openai/gpt-oss-20b`
- Execution timestamp: `2026-07-25T11:31:38.044271+00:00`
- Original result path: `reports/generated/diabetes_risk/feature_drift/live_groq/agentic_result.json`
- Repeat result path: `reports/generated/diabetes_risk/reliability_rerun_01/feature_drift/agentic_result.json`

## Scenario and run

- Run ID: `MON-DIABETES-RISK-FEATURE-DRIFT-44994E7FD981`
- Thread ID: `agentic-mon-diabetes-risk-feature-drift-44994e7fd981`
- Final status: `approved`
- Deterministic candidates: `mixed_incident`, `performance_degradation`, `feature_drift`, `prediction_drift`

## Workflow persistence

- Backend: `memory`
- Checkpoint database: `not applicable`
- Thread ID: `agentic-mon-diabetes-risk-feature-drift-44994e7fd981`
- Resumed from checkpoint: `false`
- Pause count: `1`
- Resume count: `1`

## Agent triage

- Incident: `mixed_incident`
- Severity: `high`
- Reason: High severity evidence for feature drift (BMI, GenHlth, HighBP), prediction drift (probability PSI, positive rate changes), and performance (Brier, F1, precision) indicates a mixed incident.
- Evidence: `DRIFT-PSI-BMI`, `DRIFT-PSI-GENHLTH`, `DRIFT-PSI-HIGHBP`, `PERF-BRIER`, `PERF-F1-OPERATING`, `PERF-PRECISION-OPERATING`, `PRED-PSI-PROBABILITY`, `PRED-RATE-DEFAULT`, `PRED-RATE-OPERATING`

## Diagnostic route

`mixed_diagnostics` using `13` already-calculated evidence records.

## Evidence-backed claims

- Feature drift detected in BMI, GenHlth, and HighBP. (`DRIFT-PSI-BMI`, `DRIFT-PSI-GENHLTH`, `DRIFT-PSI-HIGHBP`)
- Prediction drift observed via high probability PSI and increased positive rates. (`PRED-PSI-PROBABILITY`, `PRED-RATE-DEFAULT`, `PRED-RATE-OPERATING`)
- Performance metrics degraded: Brier score, F1, and precision have fallen below thresholds. (`PERF-BRIER`, `PERF-F1-OPERATING`, `PERF-PRECISION-OPERATING`)

## Root-cause hypothesis

Hypothesis: Feature drift in BMI, GenHlth, and HighBP is contributing to prediction drift and performance degradation.

## Recommended actions

- `investigate_feature_drift` [high]: Investigate the source and extent of feature drift in BMI, GenHlth, and HighBP. Evidence: `DRIFT-PSI-BMI`, `DRIFT-PSI-GENHLTH`, `DRIFT-PSI-HIGHBP`. Human approval: `true`.
- `investigate_data_pipeline` [high]: Examine the data ingestion and preprocessing pipeline for recent changes that could explain the observed drift. Evidence: `DRIFT-PSI-BMI`, `DRIFT-PSI-GENHLTH`, `DRIFT-PSI-HIGHBP`, `PRED-PSI-PROBABILITY`, `PRED-RATE-DEFAULT`, `PRED-RATE-OPERATING`. Human approval: `true`.
- `evaluate_threshold` [medium]: Assess whether adjusting the operating threshold could mitigate performance drops. Evidence: `PERF-F1-OPERATING`, `PERF-PRECISION-OPERATING`. Human approval: `true`.

No action was automatically executed.

## Uncertainties

- Uncertainty whether drift originates from data collection changes or external population shifts; threshold adjustment may not fully restore performance.

## Verification

- Status before finalisation: `pass`
- Violations: []
- Unsupported evidence IDs: []
- Policy checks: `citation_existence_and_uniqueness`, `claim_and_action_grounding`, `overall_evidence_coverage`, `severity_compatibility`, `confidence_schema_bounds`, `uncertainty_presence`, `incident_candidate_compatibility`, `batch_blocking_policy`, `label_and_performance_availability_policy`, `normal_operation_action_policy`, `retraining_eligibility_policy`, `feature_drift_interpretation_policy`, `human_approval_policy`, `no_automatic_action_policy`

## Fallback

- Used: `false`
- Revision attempts: `0`

## Human decision

- Decision: `approve` by `reliability-rerun-01` at `2026-07-25T11:31:38.035276Z`. Comment: _None_

## Structured LLM calls

- `groq` / `openai/gpt-oss-20b` / `AgentTriage`: latency `1891.81` ms, parse success `True`, tokens `{"input_tokens": 3520, "output_token_details": {"reasoning": 946}, "output_tokens": 1112, "total_tokens": 4632}`, fake `false`
- `groq` / `openai/gpt-oss-20b` / `AgentRecommendation`: latency `7491.26` ms, parse success `True`, tokens `{"input_tokens": 4161, "output_token_details": {"reasoning": 779}, "output_tokens": 1465, "total_tokens": 5626}`, fake `false`

## Limitations

- This is a controlled replay, not a live-production observation.
- Deterministic monitoring evidence remains authoritative; the LLM did not recalculate it.
- Evidence associations do not establish causality.
- The selected checkpoint backend supports controlled local workflows only; SQLite is
  not presented as production-grade durability.
- Recommendations require external human execution; this workflow performs no remediation.
