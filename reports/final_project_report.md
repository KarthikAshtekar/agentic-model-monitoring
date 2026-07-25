# Final Project Report: Agentic Model Risk & Monitoring Copilot

## 1. Executive summary

The project packages a validated credit-default model and its reference contract into a
local replay-based monitoring system. Deterministic Python components calculate
data-quality, feature/prediction-drift, and labelled-performance evidence; one LangGraph
orchestrator asks Groq `openai/gpt-oss-20b` for structured triage and recommendations. A
deterministic verifier rejects unsupported citations and policy violations, permits one
revision, and requires approval for non-normal actions.

Six controlled live-provider evaluations completed with compatible incident and route
selection, `100%` evidence grounding, `100%` policy compliance, `83.33%` first-pass
verification, and `0%` fallback. This is a validated portfolio MVP for controlled replay,
not a production deployment or production-readiness result. The authoritative evaluation
is
[`evaluations/live_groq_six_scenarios/live_evaluation_report.md`](evaluations/live_groq_six_scenarios/live_evaluation_report.md).
The original four-scenario evaluation remains preserved under `evaluations/live_groq/`.

## 2. Business and technical objective

Model-risk teams need more than metric alarms: they need traceable evidence, an
operational interpretation, controlled next steps, and a human decision record. The
technical objective was to demonstrate that separation without allowing the LLM to
recalculate statistics or execute remediation. The workflow therefore treats monitoring
calculations as authoritative and LLM output as a proposal subject to deterministic
validation.

## 3. Why agentic orchestration was used

The workflow contains stateful conditional behavior: select evidence, classify an
incident, choose one diagnostic route, request a recommendation, verify it, optionally
revise once, fall back safely on failure, and pause for approval. LangGraph makes these
states and transitions explicit and checkpointable. A simple linear script could call the
same functions, but it would represent interruption, resume, bounded revision, and
conditional recovery less clearly.

The MVP deliberately uses one orchestrator rather than multiple agents. The decision and
its trade-offs are recorded in
[`../docs/adr/001-single-orchestrator-agent.md`](../docs/adr/001-single-orchestrator-agent.md).

## 4. Source model and monitoring bundle

The source is the selected `xgboost_public` binary classifier packaged as a complete
scikit-learn `Pipeline` with preprocessing and `XGBClassifier`. The monitoring bundle
contains 36 ordered predictors, model/version metadata, the operating threshold `0.25`,
and a 6,002-row group-aware stratified held-out reference set. It also stores reference
features, labels, probabilities, metrics, feature summaries, checksums, and provenance.

The local source-repository path in
[`../artifacts/metadata/model_metadata.json`](../artifacts/metadata/model_metadata.json)
is machine-specific provenance, not a universal installation requirement. Runtime
monitoring consumes only the exported bundle and does not import source-model code.

## 5. Monitoring methodology

The deterministic engine in [`../src/monitoring_agent/monitoring/`](../src/monitoring_agent/monitoring/)
implements:

- structural and value-level data-quality gates;
- numeric and categorical drift using PSI, KS, Jensen-Shannon divergence, unseen
  categories, location shifts, and missing-rate movement;
- prediction-distribution and decision-rate drift; and
- labelled performance at default and operating thresholds, including accuracy,
  precision, recall, F1, ROC-AUC, PR-AUC, and Brier score.

Every conclusion has a stable evidence ID, observed/reference value, threshold, status,
severity, source, and message. Incident candidates and overall deterministic severity are
rule outputs, not LLM conclusions. Thresholds in
[`../configs/monitoring.yaml`](../configs/monitoring.yaml) are replay-MVP settings, not
production-governed limits.

## 6. Scenario methodology

All six scenarios use a deterministic seed of `20260725` and 1,000 replay rows:

1. **Normal operation:** unchanged stratified reference replay.
2. **Feature drift:** controlled changes to `LIMIT_BAL`, `PAY_0`, and
   `RecentPaymentDelay`.
3. **Data-quality failure:** 20 duplicate IDs, 15% missing `PAY_AMT1`, and 8%
   out-of-range `LIMIT_BAL`.
4. **Performance degradation:** flip 180 lowest-risk observed negatives to positive
   labels.
5. **Unlabelled drift:** shift `LIMIT_BAL`, `PAY_2`, and `MaxPaymentDelay` while
   intentionally omitting outcomes.
6. **Insufficient labels:** retain only 100 aligned labels (10% coverage), below the
   200-label performance minimum.

The fourth case is a synthetic outcome/concept-drift demonstration. It does not represent
observed production behavior. Definitions and limitations are in
[`../configs/scenarios.yaml`](../configs/scenarios.yaml).

## 7. LangGraph architecture

The graph loads the authoritative monitoring result, selects at most 30 evidence records,
requests strict triage, prepares route-specific context, requests a recommendation, and
passes it to the verifier. Conditional edges permit a single revision, deterministic
fallback, direct completion for normal operation, or an approval interrupt for non-normal
recommendations. The serializable state contains no raw dataset, fitted model, API key, or
hidden reasoning.

Implementation: [`../src/monitoring_agent/agent/graph.py`](../src/monitoring_agent/agent/graph.py)
and [`../src/monitoring_agent/agent/nodes.py`](../src/monitoring_agent/agent/nodes.py).

## 8. LLM integration

The live adapter uses `ChatGroq` with model `openai/gpt-oss-20b`, temperature `0`, a
30-second timeout, one provider retry, provider-native `json_schema`, strict mode, and
structured Pydantic outputs. It records parsed outputs, safe error categories, model,
latency, and token metadata when exposed. It does not retain raw model reasoning.

Two schemas separate triage from recommendation. Controlled literals restrict incident,
severity, route, action, priority, verification, and approval values. See
[`../src/monitoring_agent/agent/schemas.py`](../src/monitoring_agent/agent/schemas.py) and
[`../src/monitoring_agent/agent/llm.py`](../src/monitoring_agent/agent/llm.py).

## 9. Evidence-grounding design

The LLM receives a compact JSON case summary and selected deterministic evidence—not raw
feature rows or the full Markdown report. Every claim, root-cause hypothesis, and action
must cite evidence IDs. The verifier confirms those IDs exist, each claim/action has
support, and `overall_evidence_ids` covers the supporting citations.

Run-level facts such as batch blocking and label availability remain structured context.
Only IDs in the authoritative monitoring evidence registry are citable. The final live
evaluation measured `100%` grounding across the six cases.

## 10. Policy verifier

Hard policy is implemented as Python functions, not prompt-only guidance. A blocked batch
must remain a data-quality failure and cannot support performance or model-change claims.
Unevaluated labels cannot justify degradation, threshold, recalibration, or retraining
claims. Normal operation allows only no action or continued monitoring; non-normal
recommendations and actions require approval.

Retraining is only an evaluation option when performance was evaluated on enough labels,
material degradation exists, and critical performance evidence is cited. Feature drift
alone cannot be described as proven model failure. No automatic deployment, rollback,
retraining, or external action is allowed.

## 11. Revision, fallback, and human approval

If the first recommendation violates evidence or policy, the verifier returns concise
feedback for one more LLM recommendation. If provider access fails, parsing fails, or the
revised result remains invalid, a conservative deterministic fallback uses the leading
incident candidate and existing evidence IDs. The fallback never asserts a detailed root
cause and still requires approval when non-normal.

LangGraph `interrupt()` pauses non-normal cases with a serializable recommendation packet.
`Command(resume=...)` records approve, reject, or request-revision on the same thread.
Approval records a decision; it does not execute the proposed action.

Memory remains the default checkpointer. The optional official SQLite backend writes to
`artifacts/checkpoints/agent_checkpoints.sqlite`. In the reviewed demonstration, one
process paused `persistent-demo-001` and exited; a separate process resumed the stored
approval, preserved the original run/evidence/recommendation, made no additional Groq
call, and completed approved. The SQLite report is stored separately under
`live_groq_persistent/persistent-demo-001/`. This local backend is not production-grade
persistence.

## 12. Deterministic scenario results

The monitoring engine produced:

| Scenario | Deterministic incident candidates | Batch blocked | Performance evaluated |
|---|---|---:|---:|
| Normal operation | `normal_operation` | No | Yes |
| Feature drift | `mixed_incident`, `performance_degradation`, `feature_drift`, `prediction_drift` | No | Yes |
| Data-quality failure | `data_quality_failure` | Yes | No |
| Performance degradation | `performance_degradation` | No | Yes |
| Unlabelled drift | `feature_drift`, `prediction_drift`, `insufficient_evidence` | No | No |
| Insufficient labels | `insufficient_evidence` | No | No |

Supporting artifacts are under [`generated/`](generated/), with deterministic
`monitoring_result.json` and `monitoring_report.md` for each scenario.

## 13. Live Groq evaluation

| Scenario | Incident | Route | Revisions | Fallback | Latency | Tokens |
|---|---|---|---:|---:|---:|---:|
| Normal operation | `normal_operation` | `no_additional_diagnostics` | 0 | No | 2,673.26 ms | 5,811 |
| Feature drift | `mixed_incident` | `mixed_diagnostics` | 0 | No | 4,457.58 ms | 7,959 |
| Data-quality failure | `data_quality_failure` | `data_quality_diagnostics` | 1 | No | 4,418.93 ms | 7,224 |
| Performance degradation | `performance_degradation` | `performance_diagnostics` | 0 | No | 2,604.25 ms | 4,823 |
| Unlabelled drift | `feature_drift` | `drift_diagnostics` | 0 | No | 4,137.00 ms | 7,504 |
| Insufficient labels | `insufficient_evidence` | `evidence_sufficiency_review` | 0 | No | 2,230.44 ms | 3,715 |

All six final outputs parsed, matched allowed incidents/routes, cited valid evidence, and
passed policy. Five passed the first verifier check; the preserved data-quality run used
one revision. Aggregate mean/median scenario LLM latency was 3,420.24/3,405.13 ms, with
37,036 recorded tokens (25,316 input and 11,720 output). The deterministic evaluator
uses no second LLM.

## 14. Defects found during live evaluation

Live validation exposed implementation defects that were fixed and regression-tested:

- **Synthetic citable system IDs:** removed; system facts remain context while citations
  resolve only to authoritative evidence.
- **Semantic checks at parse time:** moved to deterministic verification so invalid
  semantics can use the bounded revision path.
- **Pass-evidence padding:** restricted to normal operation, reducing irrelevant
  non-normal tokens and avoiding provider budget pressure.
- **Windows console characters:** approval payloads now use ASCII-safe escaped display
  without changing structured content.
- **Approval-completion accounting:** normal operation can only pass when it avoids the
  approval path.
- **Report provenance:** legacy fake outputs moved under `fake/`; live outputs write under
  `live_groq/`, both with explicit identity fields.
- **Insufficient-label fallback wording:** made the evidence-sufficiency uncertainty
  explicit so deterministic policy and fallback agree.
- **Legacy reproducibility test scope:** moved byte-level regeneration to a temporary
  output directory so reviewed scenario artifacts are not overwritten by tests.
- **Unlabelled manifest provenance:** replaced a reused hard-coded `PAY_0` description
  with the actual `PAY_2` link; scenario data and monitoring evidence were unchanged.

The two new live Groq runs exposed no live implementation defect; both passed structured
parsing, grounding, policy, and first-pass verification without fallback.

The final outputs contain no fallback. Initial provider/parse failures remain documented
in the [evaluation report](evaluations/live_groq/live_evaluation_report.md).

## 15. Safety and governance controls

- Project-root `.env` loading with a secret field excluded from representations
- `.env`, generated data, model binaries, and generated reports ignored by Git
- Safe provider error categories without request details or credentials
- No raw LLM reasoning stored
- Strict controlled schemas and exact evidence-ID checks
- Deterministic policy enforcement
- One revision maximum in the graph
- Conservative fallback
- Human approval for non-normal recommendations
- No external remediation execution

## 16. Limitations

The evidence comes from one public academic model, a random held-out reference split, and
six controlled replays. Scenario transformations and thresholds are demonstrations, not
estimates of production incident frequency or business risk. The evaluation does not
measure rare provider failures, repeated-run variance, prose quality, fairness, production
latency, or operational recovery.

Memory is the default checkpointer and SQLite is only a local process-restart
demonstration; there is no production-grade case database, real-time ingestion,
authentication layer, alerting integration, or production deployment. Full limitations
are documented in [`../docs/limitations.md`](../docs/limitations.md).

## 17. Conclusion

The project demonstrates a defensible boundary between deterministic model monitoring and
LLM-assisted operational synthesis. Within six controlled replays, the live agent was
compatible, grounded, policy-compliant, and approval-controlled, and its revision path
corrected one recommendation without fallback. The appropriate verdict is
`live_agent_validated_with_limitations`: strong portfolio evidence for the architecture,
not a claim of production readiness.
