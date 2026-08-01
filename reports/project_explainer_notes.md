# Project explainer source notes

## Delivery contract

- Audience: technical data scientist first; business reader second.
- Delivery mode: one self-contained portable HTML report.
- Canonical output: `project_explainer.html`.
- Authoritative report source: `project_explainer_artifact.json`.
- Reproduced evidence snapshot: `reports/project_explainer_evidence.json`.
- Generator: `scripts/build_project_explainer.py`.
- Existing explainer at task start: none; no archive was required for the first build.

## Detected analytical task

The project combines operational model monitoring, binary-classification evaluation,
anomaly/drift detection, and governed agentic decision support. It loads two fitted
classifiers from self-contained bundles and does not fit a new model. The primary unit is
one registered model plus one current batch. The decision moment is after a feature batch
arrives, when labels may be complete, partial, or absent.

Regression, time-series, clustering, causal, and optimization diagnostics are omitted
because they do not match the executable task. There is no notebook, so the report uses a
complete module-by-module walkthrough instead of cell commentary.

## Technical-report section mapping

| Required role | Report section |
|---|---|
| Title | Agentic Model Risk & Monitoring Copilot |
| Technical summary | Technical summary |
| Key findings with visual evidence | Threshold trade-offs; live-agent rates |
| Scope, data, and metric definitions | Detected task; reference profiles; metric and feature dictionaries |
| Methodology | Architecture; model boundary; scenario design; deterministic results |
| Model/validation details | Fitted specifications; threshold charts; live evaluation |
| Limitations, uncertainty, robustness | Failure analysis; protected-file mismatch; known limitations; reproduced dtype-gate defect |
| Recommended next steps | Project-specific future-scope table |
| Further questions | Operational-use questions |

## Chart map

| Section | Analytical question | Family / type | Fields | Supported claim | Palette policy |
|---|---|---|---|---|---|
| Credit threshold | What changes when the source threshold moves from 0.50 to 0.25? | Comparison / grouped bar | metric, value, threshold | Recall rises while precision and specificity fall on the same held-out population | Shared reader, hard two-root cap, threshold shown by legend |
| Diabetes threshold | What changes when the source threshold moves from 0.50 to 0.25? | Comparison / grouped bar | metric, value, threshold | Screening recall rises materially at a precision/specificity cost | Shared reader, hard two-root cap, threshold shown by legend |
| Agent evaluation | Where did live-provider control outcomes differ by model? | Comparison / grouped bar | metric, rate, model_id | Both models preserve grounding/policy/approval; diabetes uses fallback in two cases | Shared reader, hard two-root cap, model shown by legend |

All charts use rates on a zero-to-one scale. Threshold charts compare values only within
the same model population. The cross-model agent chart compares the same six controlled
scenario definitions per model. No time-series chart is used because the evidence has no
meaningful temporal sequence.

## Evidence boundary and omissions

- Reproduced now: bundle validation, exact local inference reproduction, reference-data
  profile, threshold metrics, twelve deterministic scenario outcomes, code/test inventory,
  full pytest receipt, protected-file SHA checks, and the malformed-dtype gate probe.
- Preserved historical evidence: first-run live Groq evaluation and isolated two-case
  diabetes reliability repeat.
- Not rerun: live Groq calls, model training, model tuning, calibration fitting, threshold
  selection, source-repository export, or human acceptance testing.
- Not implemented by the project: fairness/subgroup monitoring, calibration curves or ECE,
  temporal/external validation, production ingestion, production case management, and
  automatic remediation.
- The generator never reads `.env`, calls a provider, regenerates scenario files, or
  overwrites stored monitoring/evaluation reports.

## Validation commands

```powershell
.\.venv\Scripts\python.exe -m py_compile scripts\build_project_explainer.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
node --check scripts\package_project_explainer.mjs
.\.venv\Scripts\python.exe scripts\build_project_explainer.py
```

The packaged report builder validates the canonical artifact, embeds the exact payload,
and checks an installed Chromium at 1440-pixel desktop and 390-pixel mobile widths for
rendered counts, geometry, overflow, external requests, browser errors, and source-dialog
interaction.
