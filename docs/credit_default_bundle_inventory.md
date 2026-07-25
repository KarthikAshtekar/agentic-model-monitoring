# Credit-default bundle inventory

## Source inspected

- Repository: external `credit-default-xai` source project at the preserved source commit
- Dataset: UCI Default of Credit Card Clients / Taiwan credit-card default (UCI ID 350)
- Target: `Default_Flag`; positive class `1` means next-month default
- Selected model: `xgboost_public`
- Selected feature set: `application_public`
- Operating threshold: `0.25` from
  `reports/model_validation/selected_recall_policy.json`

## Authoritative source artifacts

- `model`: `models/xgboost_public.pkl`
- `processed_dataset`: `data/processed/uci_taiwan_credit_default_processed.csv`
- `stored_predictions`: `reports/model_validation/xgboost_test_predictions.csv`
- `model_metrics`: `reports/model_validation/xgboost_public_model_metrics.json`
- `selected_policy`: `reports/model_validation/selected_recall_policy.json`
- `training_summary`: `reports/model_validation/xgboost_training_summary.json`
- `model_card`: `docs/model_card.md`

The training summary selects `models/xgboost_public.pkl`; the selected-policy JSON is
authoritative for the operating threshold; the processed UCI table and source split code
reconstruct the held-out rows; the stored prediction CSV and model reports provide
independent reproduction checks.

## Inference features (36)

1. `LIMIT_BAL`
2. `EDUCATION`
3. `MARRIAGE`
4. `AGE`
5. `PAY_0`
6. `PAY_2`
7. `PAY_3`
8. `PAY_4`
9. `PAY_5`
10. `PAY_6`
11. `BILL_AMT1`
12. `BILL_AMT2`
13. `BILL_AMT3`
14. `BILL_AMT4`
15. `BILL_AMT5`
16. `BILL_AMT6`
17. `PAY_AMT1`
18. `PAY_AMT2`
19. `PAY_AMT3`
20. `PAY_AMT4`
21. `PAY_AMT5`
22. `PAY_AMT6`
23. `BillToLimitRatio_1`
24. `BillToLimitRatio_2`
25. `BillToLimitRatio_3`
26. `BillToLimitRatio_4`
27. `BillToLimitRatio_5`
28. `BillToLimitRatio_6`
29. `AvgBillToLimitRatio`
30. `AvgPaymentToBillRatio`
31. `RecentPaymentDelay`
32. `MaxPaymentDelay`
33. `NumDelayedMonths`
34. `AvgBillAmount`
35. `AvgPaymentAmount`
36. `PaymentToLimitRatio`

## Reference selection

The bundle uses the untouched group-aware stratified random held-out test split
(`test_size=0.20`, `random_state=42`) reconstructed by the source repository's split
function. It contains `6002` rows. `record_id` is the zero-based row position
from the authoritative processed CSV because that CSV does not retain the original UCI
identifier. The source stored labels match exactly and the maximum absolute difference
against its stored probabilities is `2.97721862719e-08`.

## Serialization and compatibility

The source artifact is a complete fitted scikit-learn `Pipeline` containing a
`ColumnTransformer` preprocessor and `XGBClassifier`. It is exported as
`artifacts/models/credit_default_pipeline.joblib`; normal monitoring runtime does not
import sibling-repository modules.

- Source environment at export: `{"joblib": "1.5.3", "numpy": "2.4.6", "pandas": "3.0.3", "python": "3.13.7", "scikit-learn": "1.9.0", "scipy": "1.17.1", "xgboost": "3.2.0"}`
- Monitoring export environment: `{"joblib": "1.5.3", "numpy": "2.5.1", "pandas": "3.0.5", "python": "3.13.7", "scikit-learn": "1.9.0", "scipy": "1.18.0", "xgboost": "3.2.0"}`

Compatibility findings:

- DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
As an alternative, you can create a new view using np.reshape (with copy=False if needed).
- DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
As an alternative, you can create a new view using np.reshape (with copy=False if needed).
- DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
As an alternative, you can create a new view using np.reshape (with copy=False if needed).
- DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
As an alternative, you can create a new view using np.reshape (with copy=False if needed).
- Source library versions are reconstructed from the current source .venv because no immutable training lockfile is present.
- numpy version differs: source environment 2.4.6, export environment 2.5.1.
- pandas version differs: source environment 3.0.3, export environment 3.0.5.
- scipy version differs: source environment 1.17.1, export environment 1.18.0.

The source repository does not contain an immutable training lockfile. The source
environment versions above are the current source `.venv` versions, not independently
provable historical training versions. Usability is therefore established by reproducing
all exported reference probabilities, not by version strings alone.

## Metric comparison

- None above the `1e-6` comparison tolerance.

Metrics reproduced from this bundle are authoritative for future monitoring baselines.

## Bundle strategy

Priority 1 was selected: reuse the existing complete fitted inference pipeline. No model
was retrained. No DNN, fairness, explainability, dashboard, or scenario workflow was run.
The bundle is for replay simulation and is not evidence of live production readiness.
