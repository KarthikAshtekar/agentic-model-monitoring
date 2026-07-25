# Agentic Monitoring Methodology

## Why one orchestrator

The MVP uses one LangGraph orchestrator because the current decision boundary is small
and explicit: triage one deterministic monitoring result, select one diagnostic view,
propose controlled actions, verify them, and request approval. Multiple autonomous
specialists would add coordination state and failure modes without adding new evidence.
Data-quality, drift, and performance functions remain deterministic components, not
agents.

## Deterministic analytics and LLM reasoning

The deterministic monitoring engine is authoritative for batch validity and blocking,
feature and prediction drift, labelled performance, evidence values and IDs, incident
candidates, and overall severity. The LLM receives no raw feature rows and does not
recalculate these metrics.

The structured LLM performs only bounded synthesis:

- select a candidate-compatible incident and diagnostic route;
- express observed claims using exact evidence citations;
- clearly label a possible root cause as a hypothesis;
- propose actions from a controlled vocabulary; and
- state uncertainties and the need for human approval.

Evidence messages are passed as JSON data and prompts explicitly say they are not
instructions.

Before the evidence packet is built, the selected registered manifest and domain pack add
model identity, use case, positive-outcome wording, prediction unit, threshold, allowed
actions, prohibited claims, safe terminology, uncertainty wording, and limitations. The
same graph therefore handles credit and diabetes evidence without embedding either
domain's language in the adapter.

## State, nodes, and conditional edges

The serializable state contains the run and thread IDs, scenario, deterministic monitoring
result, available and selected evidence, triage, diagnostic context, recommendation,
verification, revision feedback and count, approval fields, LLM call metadata, execution
errors, checkpoint backend/database, pause/resume counts, cross-process resume flag, final
status, and report paths. It contains no API key, fitted model, raw dataset, open
connection, or hidden model reasoning.

The graph runs:

```text
START
  → load_monitoring_case
  → select_evidence
  → llm_triage
  → route_diagnostics
  → prepare_diagnostic_context
  → llm_recommendation
  → verify_recommendation
       ├─ revise_recommendation → llm_recommendation (at most once)
       ├─ deterministic_fallback
       └─ pass
  → prepare_human_approval → human_approval when non-normal
  → finalize
  → END
```

Diagnostic routing only filters already-calculated evidence:

- data-quality diagnostics emphasize blocking, schema, missingness, duplicates, and range
  findings;
- drift diagnostics emphasize feature, categorical, prediction, rate, and KS evidence;
- performance diagnostics emphasize threshold metrics, ranking/calibration metrics,
  deltas, and labelled sample size;
- mixed diagnostics combine material drift and performance evidence;
- evidence-sufficiency review emphasizes why a stronger conclusion is unavailable; and
- no-additional-diagnostics preserves the normal-operation packet.

## Structured outputs and Groq

The live provider is `ChatGroq`, defaulting to `openai/gpt-oss-20b`, temperature zero,
30-second timeout, and one provider retry. Both calls use provider-native JSON Schema
with strict validation and raw-response inclusion for metadata extraction. Raw hidden
reasoning is not stored. Reports retain only parsed schemas, concise rationale fields,
model/provider identity, latency, token usage when available, and errors.

All schema properties are required. Nullable fields remain required when absence is
valid. Extra properties are rejected. Incident, severity, route, action, priority,
verification, and approval values are controlled literals.

## Evidence selection and grounding

The selector keeps every critical, warning, and authoritative system record, then adds a
small set of relevant passes for normal operation. Run-level label counts, coverage, and
the minimum sample requirement are structured case context. The packet is capped at 30
unique records; original evidence IDs and calculated values are preserved exactly.

The deterministic verifier checks that every claim and action cites available IDs and
that the overall citation list covers all supporting citations. It also checks incident
compatibility, severity, uncertainty, approval, and the hard action policy.

## Hard policy verification

The verifier enforces these boundaries:

- a blocked batch is a data-quality failure, cannot support performance claims, and can
  only propose quarantine, pipeline investigation, or governance escalation;
- missing or unevaluated labels cannot validate performance, concept drift, retraining,
  threshold, or recalibration changes;
- normal operation allows only no action or continued monitoring, with info/low severity
  and no approval;
- retraining is only an evaluation proposal when labelled sample size is sufficient,
  material performance degradation exists, and critical performance evidence is cited;
- feature drift alone is not proven model failure;
- all non-normal recommendations and their actions require human approval; and
- no automatic deployment, rollback, retraining, or external action is allowed.

The verifier avoids pretending to solve general natural-language fact checking. It
validates structured citations and policy conditions and uses only limited prohibited
phrases for explicitly unsafe claims.

## Revision and fallback

If the first structured recommendation violates evidence or policy, the LLM receives one
concise verifier feedback string. The revised recommendation is verified again. There is
no open-ended cycle.

API failure, parse failure, a disabled revision, or a second failed verification produces
a deterministic fallback. The fallback uses the leading deterministic incident
candidate, existing evidence IDs, and conservative controlled actions. It asserts no
detailed root cause, explicitly records that the LLM result was unavailable or rejected,
and still requires approval for every non-normal incident. All provider and verification
errors remain in run metadata.

## Human approval

Non-normal recommendations pause with LangGraph `interrupt()` and a serializable packet
containing scenario, incident, severity, summary, actions, evidence IDs, and allowed
decisions. `Command(resume=...)` continues the same stable thread with an `approve`,
`reject`, or `request_revision` decision. The last choice is terminal in this MVP; it
does not trigger another LLM call. Normal operation finalizes without an interrupt.

Memory is the default checkpointer. Optional local SQLite uses the official
`langgraph-checkpoint-sqlite` backend and a scoped connection. A demonstrated first
process persisted `persistent-demo-001` at approval; a separate process resumed it,
preserved the original run/evidence/recommendation, made no additional LLM call, and
finished approved. SQLite is a local-development backend, not production-grade durability.

## Live evaluation methodology

Each registered model is evaluated with the real Groq provider
`openai/gpt-oss-20b` across six controlled replay scenarios. The evaluator does not use
an LLM-as-judge because a second model would add stochastic interpretation and would not
be authoritative for evidence IDs or hard governance rules. Instead, it reuses Pydantic
contracts, deterministic monitoring results, evidence registries, and hard policy.

Scenario expectations are fixed before evaluation:

- normal operation must remain normal with no additional diagnostics or approval;
- feature drift may be mixed or drift-only and must use mixed or drift diagnostics;
- blocked data quality must remain a data-quality failure with data-quality diagnostics;
  and
- performance degradation must remain performance degradation with performance
  diagnostics;
- unlabelled drift must use a compatible drift/evidence-sufficiency route without a
  performance conclusion or immediate model-change action; and
- insufficient labels must route to evidence-sufficiency review, prioritize
  `collect_more_labels`, and state the label-coverage limitation.

For every completed result, the evaluator checks live provider provenance, triage and
recommendation parse success, incident and route compatibility, validity of every cited
deterministic evidence ID, claim and action citations, final verifier status, hard policy,
approval behavior, revision count, fallback usage, latency, and available token metadata.
Structured-output success requires both live structured calls to parse; fallback is never
counted as structured-output success. Evidence-grounding success requires valid citations
for the overall result, every claim, and every action.

The preserved credit evaluation completed six scenarios with `100%` structured output,
incident/routing compatibility, grounding, policy compliance, and approval completion;
first-pass verification was `83.33%` and fallback was `0%`.

The diabetes evaluation completed six scenarios. Structured-output and first-pass rates
were `66.67%`; incident/routing compatibility, grounding, policy, and approval were
`100%`; fallback was `33.33%`. One feature-drift recommendation and one unlabelled-drift
recommendation were rate-limited. The unlabelled case also produced an incompatible live
triage that deterministic verification replaced. These first-run outcomes were preserved,
not rerun. Mean/median diabetes scenario LLM latency was
`14,532.31 / 10,074.59 ms`, with `25,496 / 6,999 / 32,495`
input/output/total tokens.

Across both models, 12 controlled replays completed with `83.33%` structured output,
`100%` incident/routing/grounding/policy/approval, `75%` first-pass verification, and
`8.33%` revision and `16.66%` fallback. Predictive metrics are not averaged across domains. The current
cross-model verdict is `cross_model_agent_requires_revision`, driven by the preserved
diabetes structured-output/fallback results.

These rates describe only 12 deliberately constructed examples (six per model). They do not estimate
rare failure frequency, prose quality, long-run reliability, threshold validity,
provider availability, or production behavior. The diabetes factual verdict is
`live_agent_requires_revision`; neither the credit nor cross-model result is a
production-readiness claim.

## Current limitations and later extension

This workflow uses synthetic local replay scenarios and placeholder monitoring thresholds.
It is not live monitoring, causal analysis, regulatory validation, production-grade case
management, or automatic remediation. The fake structured provider is for offline testing
only and is clearly labelled in reports.

Future specialist subgraphs could be added behind the same evidence and policy contracts
when new governed data sources justify them. Candidates include fairness review,
calibration investigation, challenger comparison, and durable incident integration. They
should remain bounded, evidence-grounded, and subordinate to the deterministic verifier
and human approval boundary.
