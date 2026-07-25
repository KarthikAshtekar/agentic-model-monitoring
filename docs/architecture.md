# Architecture

This repository is the future monitoring and investigation layer for an existing
credit-default model. It is intentionally separate from model development so that
monitoring can depend on a versioned, stable export rather than on the internal source
layout of another repository.

## Implemented system flow

```text
Validated model bundle
        ↓
Deterministic monitoring engine
        ↓
Structured evidence packet
        ↓
LangGraph triage
        ↓
Conditional diagnostic route
        ↓
Structured LLM recommendation
        ↓
Deterministic evidence/policy verifier
   ↙ revision     ↓ pass       ↘ fallback
                         human approval interrupt
                                  ↓
                           agentic incident report
```

This is one orchestrator. Data-quality, drift, and performance calculations are
deterministic functions and are not described as agents.

## Implemented monitoring flow

```text
Validated credit-default bundle
        ↓
Scenario replay batch
        ↓
Data-quality validation
        ↓
Feature and prediction drift
        ↓
Label availability gate and labelled performance evaluation
        ↓
Structured evidence records
        ↓
Deterministic incident candidates
        ↓
Deterministic JSON and Markdown reports
        ↓
Compact evidence selection (maximum 30 records)
        ↓
Bounded LangGraph workflow and agentic reports
```

Blocked data-quality batches stop before model inference, drift, or performance
evaluation. Valid batches use the exported pipeline directly. Incident candidates are
ordered rule outputs, not final recommendations.

## Implemented model-bundle contract

The source repository exports a self-contained versioned bundle with this contract:

```text
artifacts/
├── models/
│   └── credit_default_pipeline.joblib
├── metadata/
│   ├── model_metadata.json
│   ├── feature_schema.json
│   └── bundle_manifest.json
└── baselines/
    ├── reference_metrics.json
    └── reference_feature_summary.parquet
data/
└── reference/
    ├── reference_features.parquet
    ├── reference_labels.parquet
    └── reference_predictions.parquet
```

`credit_default_pipeline.joblib` is the selected `xgboost_public` fitted scikit-learn
`Pipeline`. Its `ColumnTransformer` preprocessing and `XGBClassifier` remain together, so
the monitoring runtime does not reconstruct preprocessing or import source modules.

`feature_schema.json` records the exact 36-feature order and observed held-out ranges.
`model_metadata.json` records model, threshold, dataset, split, compatibility, and
provenance facts. `bundle_manifest.json` covers every non-manifest payload with size and
SHA-256. A manifest cannot contain its own stable checksum, so that self-referential
exception is explicit.

## Reference contract

The reference is the source project's group-aware stratified random held-out test split
(`test_size=0.20`, `random_state=42`) with 6,002 samples. The UCI dataset has no true
application timestamp, so this is not presented as a temporal validation split.

- `reference_features.parquet`: `record_id` plus the 36 ordered predictors; no target.
- `reference_labels.parquet`: `record_id`, `actual_label`.
- `reference_predictions.parquet`: `record_id`, probability, decision at `0.50`, and
  decision at the selected operating threshold `0.25`.

The authoritative processed CSV does not retain the original UCI identifier. `record_id`
is therefore its stable zero-based row position, not a customer identifier.

## Runtime and export boundary

Normal monitoring runtime uses only `monitoring_agent.bundle` and the exported files.
It does not import from `credit-default-xai`. The one-time exporter may use the sibling
split function to reproduce the source split, which is allowed only at export time.

Validation checks required files, Pydantic JSON contracts, record alignment, target
separation, feature order, probability bounds, threshold decisions, metadata counts,
manifest checksums, and complete-pipeline probability reproduction.

## Planned monitoring domains

- **Data quality:** schema, types, ranges, missingness, and invalid values.
- **Feature drift:** changes in individual input distributions.
- **Prediction drift:** changes in score and decision distributions.
- **Performance:** labelled-batch predictive quality after sufficient outcomes arrive.
- **Calibration:** agreement between predicted risk and observed event rates.
- **Fairness:** governed group-level performance and outcome comparisons.
- **Model-version comparison:** regression checks between approved and candidate bundles.

The model's `0.25` operating threshold is source-authoritative: it maximized validation
recall subject to validation precision of at least `0.50`. Other monitoring thresholds in
`configs/monitoring.yaml` remain unvalidated placeholders.

## Runtime boundary

The development environment replays six deterministic simulated production batches from
local files against this bundle. They cover stable operation, covariate shift, blocking
data faults, synthetic outcome shift, absent outcomes, and only 100/1,000 aligned outcomes.
It is not connected to live scoring, event streaming,
alerting, case management, or production data stores. The LangGraph workflow can
synthesise and verify recommendations from the replay evidence, but remediation remains
external and subject to explicit human approval.

Checkpoint construction is dependency-injected. Memory remains the default. The optional
official `SqliteSaver` stores local checkpoints under
`artifacts/checkpoints/agent_checkpoints.sqlite`, allowing an approval interrupt to be
resumed by a later Python process. Connections have a scoped lifecycle, and state contains
only JSON-like monitoring/recommendation values—never keys, raw datasets, fitted models,
or connections. This local SQLite backend demonstrates process-restart behavior; it is not
production-grade persistence.

Agentic reports are separated by execution provenance under each scenario:
`fake/` contains offline test-provider outputs and `live_groq/` contains real-provider
outputs. The deterministic evaluation layer reads only `live_groq/agentic_result.json`
plus the corresponding `monitoring_result.json`, then writes JSON, Markdown, and CSV
artifacts under `reports/evaluations/live_groq/`. The preserved core evaluation remains
there; the six-scenario extension writes under
`reports/evaluations/live_groq_six_scenarios/`. SQLite demonstration reports use a
separate `live_groq_persistent/<thread-id>/` namespace. Evaluation performs no network
call and uses no LLM judge.
