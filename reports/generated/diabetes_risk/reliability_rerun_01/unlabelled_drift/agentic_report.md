# Agentic monitoring report: unlabelled_drift

## Execution provenance

- Provider: `groq`
- Model: `openai/gpt-oss-20b`
- Execution mode: `live`
- Fake LLM: `false`
- Run ID: `MON-DIABETES-RISK-UNLABELLED-DRIFT-173A5F3AE6C8`
- Thread ID: `agentic-mon-diabetes-risk-unlabelled-drift-173a5f3ae6c8`
- Created at UTC: `2026-07-25T11:33:24.977561+00:00`
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
- Scenario: `unlabelled_drift`
- Provider: `groq`
- Provider model: `openai/gpt-oss-20b`
- Execution timestamp: `2026-07-25T11:33:24.977561+00:00`
- Original result path: `reports/generated/diabetes_risk/unlabelled_drift/live_groq/agentic_result.json`
- Repeat result path: `reports/generated/diabetes_risk/reliability_rerun_01/unlabelled_drift/agentic_result.json`

## Scenario and run

- Run ID: `MON-DIABETES-RISK-UNLABELLED-DRIFT-173A5F3AE6C8`
- Thread ID: `agentic-mon-diabetes-risk-unlabelled-drift-173a5f3ae6c8`
- Final status: `approved`
- Deterministic candidates: `feature_drift`, `prediction_drift`, `insufficient_evidence`

## Workflow persistence

- Backend: `memory`
- Checkpoint database: `not applicable`
- Thread ID: `agentic-mon-diabetes-risk-unlabelled-drift-173a5f3ae6c8`
- Resumed from checkpoint: `false`
- Pause count: `1`
- Resume count: `1`

## Agent triage

- Incident: `feature_drift`
- Severity: `high`
- Reason: Evidence shows high‑severity drift in BMI, DiffWalk, and GenHlth PSI values, exceeding critical thresholds, indicating feature drift. Prediction drift evidence also present but the primary incident type is feature drift.
- Evidence: `DRIFT-PSI-BMI`, `DRIFT-PSI-DIFFWALK`, `DRIFT-PSI-GENHLTH`, `PRED-PSI-PROBABILITY`, `PRED-RATE-DEFAULT`, `PRED-RATE-OPERATING`

## Diagnostic route

`drift_diagnostics` using `9` already-calculated evidence records.

## Evidence-backed claims

- Feature drift observed in BMI, DiffWalk, and GenHlth with PSI values exceeding critical thresholds. (`DRIFT-PSI-BMI`, `DRIFT-PSI-DIFFWALK`, `DRIFT-PSI-GENHLTH`)
- Prediction probability PSI and positive rate changes exceed critical thresholds. (`PRED-PSI-PROBABILITY`, `PRED-RATE-DEFAULT`, `PRED-RATE-OPERATING`)
- Labels are not available for this batch, preventing performance evaluation. (`SYSTEM-LABELS-UNAVAILABLE`)

## Root-cause hypothesis

Hypothesis: The observed feature drift may be due to changes in population characteristics or survey response patterns over time.

## Recommended actions

- `investigate_feature_drift` [high]: Investigate feature drift Evidence: `DRIFT-PSI-BMI`, `DRIFT-PSI-DIFFWALK`, `DRIFT-PSI-GENHLTH`. Human approval: `true`.
- `collect_more_labels` [medium]: Collect more labels Evidence: `SYSTEM-LABELS-UNAVAILABLE`. Human approval: `true`.

No action was automatically executed.

## Uncertainties

- Uncertainty about whether drift is due to data distribution shift or model degradation.
- Uncertainty about impact on predictions due to lack of labels.

## Verification

- Status before finalisation: `pass`
- Violations: []
- Unsupported evidence IDs: []
- Policy checks: `citation_existence_and_uniqueness`, `claim_and_action_grounding`, `overall_evidence_coverage`, `severity_compatibility`, `confidence_schema_bounds`, `uncertainty_presence`, `incident_candidate_compatibility`, `batch_blocking_policy`, `label_and_performance_availability_policy`, `normal_operation_action_policy`, `retraining_eligibility_policy`, `feature_drift_interpretation_policy`, `human_approval_policy`, `no_automatic_action_policy`

## Fallback

- Used: `false`
- Revision attempts: `1`

## Human decision

- Decision: `approve` by `reliability-rerun-01` at `2026-07-25T11:33:24.971324Z`. Comment: _None_

## Structured LLM calls

- `groq` / `openai/gpt-oss-20b` / `AgentTriage`: latency `6086.53` ms, parse success `True`, tokens `{"input_tokens": 2868, "output_token_details": {"reasoning": 1105}, "output_tokens": 1245, "total_tokens": 4113}`, fake `false`
- `groq` / `openai/gpt-oss-20b` / `AgentRecommendation`: latency `36077.03` ms, parse success `True`, tokens `{"input_tokens": 3486, "output_token_details": {"reasoning": 846}, "output_tokens": 1336, "total_tokens": 4822}`, fake `false`
- `groq` / `openai/gpt-oss-20b` / `AgentRecommendation`: latency `36094.8` ms, parse success `True`, tokens `{"input_tokens": 3506, "output_token_details": {"reasoning": 796}, "output_tokens": 1321, "total_tokens": 4827}`, fake `false`

## Limitations

- This is a controlled replay, not a live-production observation.
- Deterministic monitoring evidence remains authoritative; the LLM did not recalculate it.
- Evidence associations do not establish causality.
- The selected checkpoint backend supports controlled local workflows only; SQLite is
  not presented as production-grade durability.
- Recommendations require external human execution; this workflow performs no remediation.
