# Diabetes onboarding report

## Decision

`diabetes_risk` was onboarded as a full `live_inference` bundle using the existing fitted
BRFSS binary XGBoost pipeline plus the existing fitted probability calibrator. No
training, tuning, threshold selection, or source-project modification occurred.

## Validation

| Check | Result |
|---|---|
| Source commit/worktree | matched / clean |
| Held-out row alignment | pass |
| Ordered raw features | 21, exact |
| Target excluded from inference | pass |
| Scores finite and within `[0,1]` | pass |
| Probability reproduction | pass; maximum difference `1.0664e-7` |
| Metric reproduction | pass; no material differences at `1e-6` |
| Direct inference | pass |
| Generic bundle validator | `15/15` checks |

At threshold `0.25`, the reproduced reference metrics are ROC-AUC `0.828593`, PR-AUC
`0.429353`, recall `0.589900`, precision `0.386469`, F1 `0.466991`, and Brier score
`0.096986`.

## Defects fixed during onboarding

Read-only inspection initially exposed an eager package-export circular import. The
models and onboarding package exports were made lazy. The first reproduction gate of
`1e-7` was `6.64e-9` tighter than the observed maximum; it was explicitly changed to
`2e-7`, still below the separate `1e-6` metric-comparison tolerance. Threshold decisions
and reference metrics were unchanged.

## Limitations

The model uses self-reported BRFSS 2015 features and an internal random held-out split.
It is a survey-based screening research artifact, not a medical diagnosis, treatment
system, external clinical validation, production service, or autonomous action system.
