# Reproducibility Guide

## Registered-model commands

List the strict registry:

```powershell
python scripts\list_registered_models.py
```

Inspect and onboard the existing fitted BRFSS binary model without retraining:

```powershell
python scripts\inspect_model_project.py `
  --source-project "<path-to-25BM6JP22>" `
  --output reports\onboarding\diabetes_risk

python scripts\onboard_binary_classifier.py `
  --source-project "<path-to-25BM6JP22>" `
  --manifest configs\models\diabetes_risk.yaml `
  --model-id diabetes_risk
```

Generate and monitor only diabetes namespaced scenarios:

```powershell
python scripts\generate_monitoring_scenarios.py --all --model-id diabetes_risk
python scripts\run_monitoring.py --all --model-id diabetes_risk
```

Run one live scenario and evaluate already-completed live reports:

```powershell
python scripts\run_agentic_monitoring.py `
  --model-id diabetes_risk `
  --scenario performance_degradation `
  --decision approve `
  --reviewer "reproduction-reviewer"

python scripts\evaluate_live_agent.py --model-id diabetes_risk
python scripts\evaluate_cross_model_agent.py
```

The last two commands make no provider calls. Diabetes generated outputs use
`data/scenarios/diabetes_risk/<scenario>/`,
`reports/generated/diabetes_risk/<scenario>/`, and
`reports/evaluations/diabetes_risk/live_groq_six_scenarios/`. Re-running a live scenario
replaces that model/scenario/mode report, so preserve reviewed results before intentional
reproduction.

## Reproduced environment

The final local evaluation used:

- Python `3.13.7` (project supports Python 3.11–3.13)
- LangGraph `1.2.9`
- LangGraph SQLite checkpointer `3.1.0`
- LangChain Core `1.5.1`
- LangChain Groq `1.1.3`
- Pydantic `2.13.4`
- Pydantic Settings `2.14.2`
- pandas `3.0.5`
- NumPy `2.5.1`
- SciPy `1.18.0`
- scikit-learn `1.9.0`
- XGBoost `3.2.0`
- PyArrow `25.0.0`
- joblib `1.5.3`
- pytest and Ruff from the development extras

`xgboost==3.2.0` is pinned because the validated fitted pipeline depends on that
compatibility setup. The source repository did not preserve an immutable historical
training lockfile; exact stored-reference probability reproduction is the operational
bundle compatibility check.

## 1. Create the environment

From this repository root in Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## 2. Configure the live provider

Create a repository-root `.env`, which is Git-ignored:

```env
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-20b
LLM_TEMPERATURE=0
LLM_TIMEOUT_SECONDS=30
LLM_MAX_RETRIES=1
LLM_MAX_REVISION_ATTEMPTS=1
GROQ_API_KEY=
LANGGRAPH_STRICT_MSGPACK=true
AGENT_CHECKPOINT_BACKEND=memory
AGENT_CHECKPOINT_DB=artifacts/checkpoints/agent_checkpoints.sqlite
```

Never put the actual key in a committed file, command transcript, report, or screenshot.
Offline graph tests and demos use `--use-fake-llm` and require no provider key.

## 3. Generate the model bundle when needed

The model is exported from a separate `credit-default-xai` checkout. On the original
machine, this local checkout was:

```text
<path-to-credit-default-xai>
```

That is a provenance example, not a universal required path. Set the variable to the
location of your own verified source checkout:

```powershell
$SourceRepo = "<path-to-credit-default-xai>"
python scripts\export_credit_default_bundle.py `
  --source-repo $SourceRepo `
  --overwrite
python scripts\validate_credit_default_bundle.py
```

The exporter writes:

```text
artifacts/models/credit_default_pipeline.joblib
artifacts/metadata/model_metadata.json
artifacts/metadata/feature_schema.json
artifacts/metadata/bundle_manifest.json
artifacts/baselines/reference_metrics.json
artifacts/baselines/reference_feature_summary.parquet
data/reference/reference_features.parquet
data/reference/reference_labels.parquet
data/reference/reference_predictions.parquet
```

The resulting bundle contains 36 ordered predictors and a 6,002-row held-out reference
set. Provenance, source paths, thresholds, versions, limitations, and compatibility
warnings are in `artifacts/metadata/model_metadata.json`.

## 4. Recreate the controlled scenarios

The shared scenario seed is:

```text
20260725
```

Generate and monitor all scenarios:

```powershell
python scripts\generate_monitoring_scenarios.py --all --overwrite
python scripts\run_monitoring.py --all
```

Expected deterministic output for each scenario:

```text
data/scenarios/<scenario>/features.parquet
data/scenarios/<scenario>/labels.parquet        # when labels are present
data/scenarios/<scenario>/scenario_manifest.json
reports/generated/<scenario>/monitoring_result.json
reports/generated/<scenario>/monitoring_report.md
```

Do not regenerate these artifacts merely to view an existing reviewed demo.

## 5. Reproduce agentic outputs

Run one live case:

```powershell
python scripts\run_agentic_monitoring.py `
  --scenario performance_degradation `
  --decision approve `
  --reviewer "reproduction-reviewer"
```

Run all six live cases only when intentionally reproducing the complete evaluator:

```powershell
python scripts\run_agentic_monitoring.py `
  --all `
  --decision approve `
  --reviewer "reproduction-reviewer"
```

The normal case does not use the supplied decision because it requires no interrupt.
Non-normal cases pause and resume as approved. These commands call the live Groq API and
replace the corresponding `live_groq/` reports.

For the reviewed robustness extension, the original four live reports were preserved and
Groq was called only for `unlabelled_drift` and `insufficient_labels`.

For an offline graph demonstration:

```powershell
python scripts\run_agentic_monitoring.py `
  --all `
  --decision approve `
  --reviewer "offline-reproduction" `
  --use-fake-llm
```

Expected agentic output:

```text
reports/generated/<scenario>/fake/agentic_result.json
reports/generated/<scenario>/fake/agentic_report.md
reports/generated/<scenario>/live_groq/agentic_result.json
reports/generated/<scenario>/live_groq/agentic_report.md
```

Each report contains provider, model, execution mode, fake/live flag, run ID, thread ID,
and UTC timestamp. It does not contain the API key or raw provider reasoning.

## 6. Recreate the deterministic live evaluation

The preserved four-scenario core evaluation uses:

```powershell
python scripts\evaluate_live_agent.py
```

The six-scenario extension uses:

```powershell
python scripts\evaluate_live_agent.py --extended
```

Expected extended outputs:

```text
reports/evaluations/live_groq_six_scenarios/live_evaluation_summary.json
reports/evaluations/live_groq_six_scenarios/live_evaluation_report.md
reports/evaluations/live_groq_six_scenarios/live_scenario_comparison.csv
```

The evaluator performs no network call and uses no LLM judge. It reads live agentic and
deterministic monitoring JSON, then checks structured parsing, incident/route
compatibility, citation grounding, policy, verification, fallback, approval, latency, and
tokens.

### Repeat reliability check

The reviewed repeat used only the two first-run diabetes fallback cases:

```powershell
python scripts\run_agentic_monitoring.py `
  --model-id diabetes_risk `
  --scenario feature_drift `
  --decision approve `
  --reviewer "reliability-rerun-01" `
  --run-label reliability_rerun_01

python scripts\run_agentic_monitoring.py `
  --model-id diabetes_risk `
  --scenario unlabelled_drift `
  --decision approve `
  --reviewer "reliability-rerun-01" `
  --run-label reliability_rerun_01

python scripts\evaluate_reliability_rerun.py
```

`--run-label` preserves the default `live_groq/` result and writes the repeat under
`reports/generated/diabetes_risk/reliability_rerun_01/<scenario>/`. The comparison is
under `reports/evaluations/diabetes_risk/reliability_rerun/`. Do not rerun the reviewed
`reliability_rerun_01` label in place; use a new unique label for any future experiment
and treat it as a separate comparison.

The original 12-case evaluation remains authoritative. The repeat used unchanged model
assets, scenario artifacts, prompts, evidence, structured schemas, routing, verification,
revision, and approval policies. The two repeat observations do not estimate production
or long-run reliability.

## 7. Reproduce local cross-process approval resume

Start and exit at an approval interrupt:

```powershell
python scripts\run_agentic_monitoring.py `
  --scenario performance_degradation `
  --checkpoint-backend sqlite `
  --thread-id "persistent-demo-001" `
  --pause-only
```

In a separate Python invocation:

```powershell
python scripts\run_agentic_monitoring.py `
  --resume-thread "persistent-demo-001" `
  --checkpoint-backend sqlite `
  --decision approve `
  --reviewer "persistent-demo"
```

The SQLite file is local and Git-ignored. Persistence reports write under
`live_groq_persistent/<thread-id>/`, preserving the ordinary `live_groq/` report.
Approval-only resume uses the saved recommendation and makes no additional Groq call.
This is not a production-grade state backend.

## 8. Focused validation

The final packaging validation scope is:

```powershell
python -m pytest tests\agent tests\evaluation tests\scenarios tests\monitoring -q
python -m ruff check `
  src\monitoring_agent\agent `
  src\monitoring_agent\evaluation `
  src\monitoring_agent\scenarios `
  src\monitoring_agent\monitoring `
  scripts\run_agentic_monitoring.py `
  scripts\evaluate_live_agent.py `
  scripts\generate_monitoring_scenarios.py `
  scripts\run_monitoring.py `
  tests\agent `
  tests\evaluation `
  tests\scenarios `
  tests\monitoring `
  tests\fakes
```

The reviewed run found two isolated test/support failures, fixed them, and passed their
targeted reruns; every other focused case passed. The exact requested Ruff scope passed.
This is not a claim that the full repository suite was run.

## Known NumPy/joblib compatibility warning

Loading the fitted pipeline through joblib may capture this NumPy 2.5 deprecation warning:

```text
DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
Use np.reshape with copy=False where appropriate.
```

The bundle metadata also records source/export NumPy, pandas, and SciPy version
differences. The warning did not prevent exact reference-prediction reproduction, but a
future model re-export should remove the deprecated behavior in the originating
serialization/library path rather than suppress it blindly.

## Artifact and Git provenance

The following generated or sensitive contents are ignored by Git:

- `.env`, virtual environments, caches, and coverage outputs
- `data/reference/*`, `data/production/*`, and `data/scenarios/*`
- `artifacts/models/*`, `artifacts/baselines/*`, and `artifacts/metadata/*`
- `artifacts/checkpoints/*`
- `reports/generated/*`, `reports/evaluations/*`, and `reports/figures/*`

Each ignored directory keeps only a versionable `.gitkeep` placeholder where applicable.
Another user must therefore export/obtain the validated bundle and regenerate scenarios
and reports locally. Documentation can describe reviewed generated results even though
the large/sensitive runtime artifacts are intentionally not committed.
