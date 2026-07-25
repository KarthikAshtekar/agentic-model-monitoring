# Agentic Model Risk & Monitoring Copilot

`agentic-model-monitoring` is a production-oriented replay MVP that monitors an existing
credit-default model, investigates model-risk incidents with one bounded LangGraph
orchestrator, and proposes evidence-backed actions for human approval.

## Why this is a separate repository

Model monitoring has a different lifecycle and operational boundary from model training.
This project will consume a stable, versioned model-monitoring bundle exported by the
existing `credit-default-xai` project. It must not directly import source code from that
sibling repository. This separation makes the monitoring contract explicit and avoids
coupling runtime checks to model-development internals.

## One-agent architecture

The implemented flow is:

```text
validated model bundle
        ↓
deterministic monitoring and evidence IDs
        ↓
bounded structured evidence packet
        ↓
one LangGraph triage and diagnostic route
        ↓
structured Groq recommendation
        ↓
deterministic evidence and policy verifier
        ↓
one revision at most, or deterministic fallback
        ↓
human approval interrupt for non-normal incidents
        ↓
agentic JSON and Markdown report
```

Python remains authoritative for data quality, drift, performance, incident candidates,
severity, and evidence values. The LLM does not recalculate metrics or execute an action.
See [`docs/architecture.md`](docs/architecture.md) and
[`docs/agentic_methodology.md`](docs/agentic_methodology.md).

## Repository structure

```text
artifacts/               Model bundles, metadata, and monitoring baselines
configs/                 Initial monitoring and scenario configuration
data/                    Reference, replayed production, and scenario batches
docs/                    Architecture and contract documentation
notebooks/               Future reviewable exploration
reports/                 Generated reports, evaluations, and figures
scripts/                 Monitoring, agent, bundle-export, and validation utilities
src/monitoring_agent/    Bundle, monitoring, scenario, and agent packages
tests/                   Focused deterministic and agent workflow tests
```

Generated and potentially sensitive contents under `data/`, `artifacts/`, and `reports/`
are ignored by Git; only their `.gitkeep` placeholders are versionable.

## Current status

The stable model-bundle integration phase is complete. The exported bundle contains:

- The selected `xgboost_public` complete fitted inference pipeline.
- The 36 ordered `application_public` predictors.
- The UCI Default of Credit Card Clients / Taiwan credit-card default held-out reference
  split with 6,002 samples.
- Separate reference features, `Default_Flag` labels, probabilities, and decisions.
- The validation-selected operating threshold `0.25` and default threshold `0.50`.
- Reproduced metrics, feature summaries, provenance, checksums, and compatibility notes.

The monitoring package loads this bundle without importing sibling-repository code during
normal runtime. The export script alone uses the source split function to reconstruct the
authoritative held-out rows.

The deterministic monitoring phase is also complete. It provides:

- Data-quality checks covering schema, identifiers, order, dtype, missingness, finite
  values, observed reference ranges, integer constraints, and batch size.
- Numeric and categorical feature drift using PSI, KS, Jensen-Shannon divergence, unseen
  categories, location shifts, and missing-rate changes.
- Prediction drift using probability PSI and KS, decision-rate changes, and risk deciles.
- Labelled performance at thresholds `0.50` and `0.25`, including ranking and calibration
  metrics compared with exported reference baselines.
- Structured evidence IDs, deterministic incident candidates, and JSON/Markdown reports.
- Four reproducible 1,000-row replay scenarios: `normal_operation`, `feature_drift`,
  `data_quality_failure`, and `performance_degradation`.

The bounded agent phase is complete. It provides strict `AgentTriage` and
`AgentRecommendation` schemas, Groq provider configuration, compact evidence selection,
conditional diagnostics over already-calculated evidence, deterministic citation and
policy verification, one optional LLM revision, deterministic fallback, and an in-memory
human approval interrupt. Agentic reports are written beside—but do not overwrite—the
deterministic reports.

The environment remains limited to replay-based simulated production batches. The
performance-degradation scenario deliberately changes labels in a low-predicted-risk
segment to represent synthetic concept/outcome drift; it is not an observed real-world
incident. The project is not connected to a live production model, data source, alerting
service, case-management system, dashboard, or automated retraining system.

## Local setup

From the repository root in Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Python 3.11 through 3.13 is supported. If Python 3.12 is unavailable, select another
installed interpreter within that range explicitly when creating `.venv`.

For live Groq use, copy only the variable names from `.env.example` into your own
untracked `.env` or export them in the shell:

```env
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-20b
LLM_TEMPERATURE=0
LLM_TIMEOUT_SECONDS=30
LLM_MAX_RETRIES=1
LLM_MAX_REVISION_ATTEMPTS=1
GROQ_API_KEY=
```

The live client stops with a concise configuration error when `GROQ_API_KEY` is absent.
It does not silently select another provider.

## Verification

With the virtual environment active:

```powershell
python scripts\verify_setup.py
python -m pytest
python -m ruff check .
python -m compileall src tests scripts
python -c "import monitoring_agent; print(monitoring_agent.__version__)"
monitoring-verify
```

Export the bundle from the verified source repository and validate it:

```powershell
python scripts\export_credit_default_bundle.py `
  --source-repo "D:\PGDBA\Projects\Credit Default Risk\credit-default-xai" `
  --overwrite
python scripts\validate_credit_default_bundle.py
python -m pytest tests\bundle -q
python -m ruff check src\monitoring_agent\bundle `
  scripts\export_credit_default_bundle.py `
  scripts\validate_credit_default_bundle.py `
  tests\bundle
```

The path originally expected at `D:\PGDBA\Projects\credit-default-xai` was not present
during integration. Pass the verified location above explicitly. The exporter refuses to
replace an existing bundle unless `--overwrite` is supplied.

## Bundle compatibility

The source artifact is a fitted scikit-learn `Pipeline` containing its
`ColumnTransformer` preprocessing and an `XGBClassifier`. The bundle validator loads that
pipeline and checks that it reproduces every stored reference probability within a small
numerical tolerance.

The source repository does not contain an immutable historical training lockfile.
Recorded training-library versions therefore come from its current `.venv` and are marked
accordingly. Exact reference-prediction reproduction is the operational compatibility
check. See [`docs/credit_default_bundle_inventory.md`](docs/credit_default_bundle_inventory.md)
for the factual inventory.

## Deterministic scenario monitoring

Generate all four scenarios and run the monitoring engine:

```powershell
python scripts\generate_monitoring_scenarios.py --all --overwrite
python scripts\run_monitoring.py --all
```

Generate or monitor one scenario:

```powershell
python scripts\generate_monitoring_scenarios.py `
  --scenario feature_drift `
  --overwrite
python scripts\run_monitoring.py --scenario feature_drift
```

Scenario data is written under `data/scenarios/<scenario>/`. Evidence reports are written
under `reports/generated/<scenario>/monitoring_result.json` and
`monitoring_report.md`. Calculations are deterministic Python functions; no LLM is used.
See [`docs/monitoring_methodology.md`](docs/monitoring_methodology.md) for definitions and
threshold caveats.

## Agentic monitoring

Run one live Groq workflow:

```powershell
python scripts\run_agentic_monitoring.py `
  --scenario performance_degradation `
  --decision approve `
  --reviewer "model-risk-reviewer"
```

Run the fully offline, clearly labelled fake-LLM demonstration:

```powershell
python scripts\run_agentic_monitoring.py `
  --all `
  --decision approve `
  --reviewer "demo-reviewer" `
  --use-fake-llm
```

Without `--decision`, every non-normal run pauses and asks for `approve`, `reject`, or
`request_revision`. Normal operation finishes without an interrupt. A reviewer request
for revision is recorded as a terminal decision in this MVP and does not make another
LLM call.

Every claim and action must cite selected evidence IDs. Deterministic verification rejects
unknown IDs, policy-incompatible incidents or actions, unsupported retraining, blocked
batch performance claims, unsafe normal-operation actions, missing approval, and other
hard-rule violations. The model gets one concise revision attempt. Provider, parsing, or
repeated verification failure produces a conservative deterministic recommendation.
Fallback and fake-provider use are explicit in both generated reports:

```text
reports/generated/<scenario>/agentic_result.json
reports/generated/<scenario>/agentic_report.md
```

No recommended action is automatically executed.

## Development phases

1. **Completed:** inspect `credit-default-xai` and finalize the versioned bundle contract.
2. **Completed:** export and validate the model, schema, metadata, and reference data.
3. **Completed:** implement deterministic monitoring metrics and focused tests.
4. **Completed:** implement four replay scenarios, evidence, incident candidates, and reports.
5. **Completed:** add one bounded LangGraph orchestrator, strict Groq outputs, verification,
   one revision, fallback, approval, and agentic reports.
6. Validate behavior, thresholds, durable checkpointing, governance controls, and failure
   recovery before any production integration.

## Scope and limitations

The package can load and score the exported model, but that technical compatibility does
not establish model validity, production readiness, regulatory compliance, fairness, or
scientifically justified monitoring thresholds. The source model uses a public academic
dataset and a random held-out split because no true application timestamp exists. Any
future agent recommendation must remain reviewable and subject to human approval.
This MVP uses an in-memory LangGraph checkpointer, local replay artifacts, and one
structured model provider. It does not provide durable cases, production authentication,
external actions, fairness monitoring, or causal diagnosis.
