# Binary-classification adapter

`BinaryClassificationAdapter` separates monitoring from estimator-specific details while
keeping model semantics explicit in a strict registered manifest.

## Supported contract

- Task: binary classification
- Adapter: `sklearn_binary_classifier`
- Score paths: `predict_proba`, sigmoid-transformed `decision_function`, or explicitly
  scored-reference-only bundles
- Positive class: explicit value and output-column index
- Thresholds: explicit default and operating values
- Input: exact ordered raw feature frame
- Labels: non-null binary `0/1` values when performance is evaluated

The adapter rejects reordered, missing, extra, or target-bearing inference columns;
missing/non-binary labels; row-count mismatch; missing model artifacts; invalid score
shape; and non-finite or out-of-range scores.

## Deterministic outputs

For every requested threshold it returns sample and positive counts, predicted-positive
rate, accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, confusion-matrix
counts, and an explicit list of undefined metrics. Ranking metrics remain `null` for
single-class labels rather than being manufactured.

The monitoring engine obtains feature/label contracts, thresholds, reference data, and
the fitted estimator through `RegisteredModelBundle`. Credit-default behavior remains
available through the backward-compatible `CreditDefaultBundle` wrapper.

## Domain separation

The adapter does not contain credit or healthcare language. Business terminology,
allowed actions, prohibited claims, uncertainty wording, and limitations come from the
registered domain policy pack. This prevents the diabetes agent from using borrower/loan
language and prevents either model from converting monitoring evidence into an automatic
business or clinical action.
