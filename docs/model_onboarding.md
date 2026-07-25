# Model onboarding

## Boundary

Onboarding adds an already-fitted binary classifier to the monitoring runtime. It does
not train, tune, select, deploy, or remediate a model. The source project is treated as
read-only, and the runtime consumes only a repository-local registered bundle.

The generic inspection step examines filenames and candidate tabular/report artifacts
without deserializing models or executing notebooks. It deliberately leaves business
use, positive-outcome meaning, operating threshold, error costs, protected attributes,
permitted remediation, and approval rules unresolved unless supplied explicitly.

## Workflow

1. Run `scripts/inspect_model_project.py` against the candidate project.
2. Review the JSON, Markdown, and candidate YAML outputs.
3. Supply a strict manifest under `configs/models/`.
4. Add a unique entry to `configs/model_registry.yaml`.
5. Run `scripts/onboard_binary_classifier.py` for a supported deterministic exporter.
6. Validate checksums, feature order, target separation, row alignment, score bounds,
   thresholds, reference metrics, and direct inference.
7. Generate model-namespaced scenarios and run monitoring.

For the BRFSS model:

```powershell
python scripts\inspect_model_project.py `
  --source-project "<path-to-25BM6JP22>" `
  --output reports\onboarding\diabetes_risk

python scripts\onboard_binary_classifier.py `
  --source-project "<path-to-25BM6JP22>" `
  --manifest configs\models\diabetes_risk.yaml `
  --model-id diabetes_risk
```

The exporter rejects a dirty source worktree, a mismatched source commit, missing
authoritative files, target leakage, row misalignment, non-finite probabilities,
probabilities outside `[0,1]`, material metric differences, and unreproducible inference.
Existing bundle files require the explicit `--overwrite` flag.

## Manifest contract

Each strict manifest records:

- stable model/display/version/task/adapter/domain identity;
- nullable business context without invented values;
- repository-relative bundle paths;
- score method, positive class, class index, and thresholds;
- exact ordered raw features, target, identifier, feature groups, and preprocessing
  representation;
- monitoring thresholds and minimum sample sizes;
- allowed actions, prohibited automatic actions, and protected attributes; and
- source-relative provenance, commit, export mode, compatibility warnings, and
  limitations.

Duplicate YAML keys, unknown fields, absolute runtime paths, path traversal, registry and
manifest ID mismatch, duplicate manifest paths, unsupported tasks/adapters, and unknown
model IDs are rejected.

## BRFSS onboarding result

The `diabetes_risk` bundle uses the existing fitted XGBoost pipeline and fitted sigmoid
calibrator. A source-equivalent stateless feature engineer and a small calibration wrapper
remove source-module imports; fitted estimator state is unchanged. No retraining occurred.

- Reference rows: `50,736`
- Raw inference features: `21`
- Transformed representation: `52`
- Target: `Diabetes_binary`; positive class `1`
- Operating threshold: `0.25`
- Bundle mode: `live_inference`
- Maximum source-score difference: `1.0664e-7`
- Validation tolerance: `2e-7`
- Bundle validation: `15/15` checks passed

See [the source inventory](onboarding/diabetes_source_inventory.md) and
[onboarding report](onboarding/diabetes_onboarding_report.md).

## Safety

Source absolute paths exist only in local inspection outputs where provenance requires
them; registry manifests and portable model metadata use source-relative paths or a
neutral external-source description. No API key, raw provider reasoning, diagnosis,
treatment recommendation, lending decision, deployment, or automated remediation is part
of onboarding.
