# Diabetes source inventory

- Source project: `25BM6JP22` (external read-only path not persisted)
- Source commit: `7016f3fd7e5e33a5884c2bdf4c05c98d2b21e0e5`
- Source worktree clean: `true`
- Dataset: CDC BRFSS 2015, `253680` rows in the maintained source project
- Task: binary diabetes-positive versus no-diabetes/prediabetes screening outcome
- Target: `Diabetes_binary`; positive class `1`
- Selected model: fitted XGBoost pipeline plus fitted sigmoid calibrator
- Raw inference features: `21`
- Preprocessed representation: `52`
- Split: stratified held-out test split (test_size=0.20, random_state=42); `50736` rows
- Operating threshold: `0.25`
- Bundle mode: `live_inference`
- Prediction reproduction maximum absolute difference: `1.06640290487e-07`
- Source environment: Python 3.13.7, NumPy 2.5.1, pandas 3.0.5,
  scikit-learn 1.9.0, XGBoost 3.3.0, joblib 1.5.3
- Monitoring environment XGBoost: `3.2.0`

## Authoritative source artifacts

- `models/brfss_final/final_binary_xgboost.joblib`
- `models/brfss_final/binary_probability_calibrator.joblib`
- `data/processed/brfss_test_set.csv.gz`
- `reports/brfss_final/tables/binary_test_predictions.csv`
- `reports/brfss_final/tables/final_test_metrics_binary.csv`
- `reports/brfss_final/tables/binary_threshold_analysis.csv`
- `reports/brfss_final/run_summary.json`
- `reports/brfss_final/MODEL_CARD.md`

## Strategy and limitations

Full inference is reproducible without retraining. The export replaces only the source
module references for stateless feature engineering and calibration composition; fitted
scikit-learn and XGBoost state remains unchanged. The source XGBoost joblib emits a
cross-version portability warning under the pinned monitoring environment, but reference
scores reproduce within tolerance. This is survey-based risk screening, not diagnosis,
treatment, individual patient care, or external clinical validation.
