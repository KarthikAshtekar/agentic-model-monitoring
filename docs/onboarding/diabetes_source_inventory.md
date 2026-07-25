# Diabetes source inventory

## Read-only source

- Project: `25BM6JP22`
- Verified commit: `7016f3fd7e5e33a5884c2bdf4c05c98d2b21e0e5`
- Worktree at onboarding: clean
- Candidate files discovered: `94`
- Source files modified: none
- Model retraining or notebook execution: none

The user supplied and confirmed the local source folder. Its machine-specific absolute
path was used only at command time and is deliberately not persisted in portable
manifests or bundle metadata.

## Candidate selection

No alternative candidate was used. The confirmed project contained the strongest
internally consistent evidence set: fitted binary estimator, fitted calibrator,
held-out split, row-aligned probabilities, threshold analysis, final metrics, model card,
source code, and environment declarations. Three-class artifacts were excluded.

## Selected authoritative assets

- `models/brfss_final/final_binary_xgboost.joblib`
- `models/brfss_final/binary_probability_calibrator.joblib`
- `data/processed/brfss_test_set.csv.gz`
- `reports/brfss_final/tables/binary_test_predictions.csv`
- `reports/brfss_final/tables/final_test_metrics_binary.csv`
- `reports/brfss_final/tables/binary_threshold_analysis.csv`
- `reports/brfss_final/run_summary.json`
- `reports/brfss_final/MODEL_CARD.md`

## Verified contract

The source dataset contains `253,680` BRFSS 2015 survey rows. The maintained binary
target is `Diabetes_binary`, where `1` is the diabetes-positive class and `0` combines
no-diabetes and prediabetes records. The fitted pipeline accepts 21 ordered raw survey
features and its deterministic feature engineer expands them to 52 model inputs.

The preprocessing contract is the fitted pipeline's source feature-engineering step
followed by its fitted scikit-learn/XGBoost transformations. The portable bundle replaces
only the stateless source-module class reference; fitted state is unchanged.

The authoritative held-out split contains `50,736` rows and is described as a stratified
20% split with random state `42`. Its operating threshold is `0.25`.

## Compatibility

The source artifacts record XGBoost `3.3.0`; the monitoring runtime pins XGBoost `3.2.0`.
Loading emits the upstream serialized-model portability warning. Despite that warning,
the portable bundle reproduced all probabilities within `1.0664e-7` and all six reported
metrics within `1e-6`. This compatibility evidence supports controlled replay only and
does not remove the need for a stable neutral-format export before production use.
