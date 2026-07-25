# Deterministic monitoring methodology

## Scope

The monitoring engine compares replay batches with the selected registered model's
reference split. Credit default uses 6,002 rows and 36 features; diabetes risk uses
50,736 rows and 21 raw BRFSS features. All metrics, evidence IDs, incident candidates,
and reports are produced by deterministic Python functions. No LLM participates in
calculation or classification.

Thresholds are initial portfolio-MVP values chosen to distinguish controlled scenarios.
They are not production-validated limits.

`model_id` selects the strict manifest, bundle, binary adapter, scenario config, and
domain policy. The monitoring result carries model/domain identity, business context,
operating threshold, allowed actions, prohibited claims, and domain limitations so the
agent never has to infer those facts.

## Data quality

The schema gate checks non-empty batches, `record_id` presence and uniqueness, required
features, target exclusion, unexpected columns, exact feature order, compatible numeric
dtypes, per-feature missingness, infinite values, complete-column emptiness, observed
reference-range violations, integer constraints, and minimum batch size.

Missing required columns, target leakage, duplicate IDs, unexpected columns, and order
mismatch block inference. Critical missingness or range violations are evidence but do
not independently block unless configuration is changed. Observed ranges are descriptive
reference bounds, not universal business rules.

## Feature drift

Continuous/high-cardinality numeric PSI uses ten reference quantile bins. Repeated edges
are removed, the outer bins include underflow and overflow, and a small epsilon prevents
zero-frequency division. Constant features receive a safe three-edge representation.

Numeric evidence also includes the two-sample Kolmogorov-Smirnov statistic and p-value,
mean shift, median shift, and missing-rate change. PSI is the primary severity signal.
KS is supporting evidence: a small p-value in a large batch is not by itself treated as
material or causal drift.

Features explicitly categorical in the schema, or integer-constrained with at most 20
reference values, use category-frequency PSI, Jensen-Shannon divergence, unseen-category
rate, and missing-rate change. PSI or Jensen-Shannon divergence drives primary severity.

## Prediction drift

The exported pipeline generates positive-class probabilities. Monitoring compares their
mean, median, PSI, KS statistic, risk-decile distribution, and predicted-positive rates at
the default threshold `0.50` and operating threshold `0.25`. Rate evidence uses absolute
change from the full reference split.

## Labelled performance

Performance is evaluated only when labels align one-to-one by `record_id`, are binary,
meet the 200-row minimum, and contain both classes. Both thresholds receive accuracy,
precision, recall, F1, predicted-positive counts/rates, and confusion-matrix counts.
ROC-AUC, PR-AUC, and Brier score are threshold-independent.

Drops are calculated as reference minus observed performance; Brier change is observed
minus reference. The operating threshold is the primary policy threshold. Undefined
metrics remain undefined rather than being silently converted to zero.

## Replay scenarios

- **Normal operation:** unchanged stratified 1,000-row reference sample.
- **Feature drift:** scales `LIMIT_BAL` by `0.35` with observed-range clipping, then sets
  `PAY_0` and `RecentPaymentDelay` to `2` for 75% of rows. Labels are unchanged.
- **Data-quality failure:** duplicates 20 record IDs, sets 15% of `PAY_AMT1` to missing,
  and places 8% of `LIMIT_BAL` above the observed reference maximum. Duplicate IDs block
  inference.
- **Performance degradation:** leaves features and probabilities unchanged, then flips
  180 observed negative labels with the lowest model probabilities to positive. This is
  transparent synthetic concept/outcome drift, not an observed event.

Every scenario uses seed `20260725`, preserves feature/label separation, and records exact
transformations in `scenario_manifest.json`.

The diabetes replay uses model-specific features: BMI/HighBP/GenHlth for labelled feature
drift; duplicate IDs, BMI missingness, and PhysHlth range faults for data quality;
low-score negative label flips for synthetic performance degradation; BMI/DiffWalk/
GenHlth shifts with no labels; and only 100 aligned labels for evidence insufficiency.
The same incident expectations and safety boundaries apply, but evidence and wording
remain screening-specific.

## Incident candidates and limitations

Rule-based candidates distinguish data-quality failure, feature drift, prediction drift,
performance degradation, mixed incidents, insufficient evidence, and normal operation.
They are preliminary evidence labels for a future agent, not remediation advice.

Replay samples inherit the limitations of an academic random held-out split. Synthetic
transformations do not reproduce causal production mechanisms, operational latency,
label delay, reject inference, policy changes, seasonality, or live data contracts.
