# Interview Defense

## 1. What problem does this project solve?

It turns deterministic model-monitoring signals into a traceable incident recommendation
without letting the LLM become the source of truth. Python calculates batch validity,
drift, and labelled performance; the LLM organizes those findings into controlled claims
and actions. A verifier and human-approval boundary make the result reviewable for a
model-risk workflow.

## 2. Why is this an agentic workflow?

The workflow maintains state and chooses among conditional paths rather than making one
prompt call. It triages an incident, selects a diagnostic route, proposes an action,
verifies the result, optionally revises once, falls back on failure, and pauses for human
approval. Those bounded decisions and recovery paths are the agentic part.

## 3. Why use LangGraph instead of a normal Python pipeline?

A linear Python pipeline could perform the calculations, but the operational workflow is
not strictly linear. LangGraph represents conditional routes, interruption, resume,
revision, fallback, and final state explicitly. It also provides a clear boundary for
switching between memory and local SQLite checkpointing without changing deterministic
monitoring logic.

## 4. How many agents are used?

One. The project has a single LangGraph monitoring orchestrator. Data-quality, drift,
performance, verification, and fallback are deterministic tools or nodes—not separate
agents.

## 5. Why did you choose one orchestrator rather than multiple agents?

All current decisions use the same compact monitoring result and hard policy. Multiple
agents would add coordination state, duplicated prompts, token cost, and more failure
modes without introducing distinct evidence sources or permissions. Separate specialist
subgraphs would only be justified if future domains had materially different tools,
owners, data access, or review rules.

## 6. What does the LLM do?

It produces structured triage and an operational recommendation from supplied evidence.
It selects an allowed incident and route, expresses evidence-backed claims, labels any
root-cause statement as a hypothesis, proposes controlled action types, and states
uncertainty. It does not calculate PSI, performance metrics, thresholds, or batch
validity.

## 7. What remains deterministic?

Model loading and scoring, data-quality validation, feature/prediction drift,
labelled-performance measurement, evidence IDs, incident candidates, and overall severity
are deterministic. Evidence selection, diagnostic filtering, policy enforcement,
recommendation verification, fallback construction, evaluation, and final status are also
Python-controlled. The LLM only synthesizes within those boundaries.

## 8. How are recommendations evidence-backed?

Every claim, root-cause hypothesis, and action contains explicit evidence IDs. Those IDs
refer to records containing metric, observed/reference values, threshold, severity,
message, and source. The verifier rejects missing or unsupported IDs and requires the
overall evidence list to cover all supporting citations.

## 9. How do you prevent hallucinated evidence?

The prompt tells the model to use only supplied IDs, but prompt instructions are not the
actual control. A deterministic verifier compares every cited ID with the authoritative
monitoring evidence registry. An unsupported ID triggers revision or fallback, so it
cannot silently enter a verified final recommendation.

## 10. Why use strict structured output?

Strict JSON Schema makes incident types, routes, actions, severities, priorities, and
approval fields predictable. Pydantic then rejects extra fields or invalid controlled
values. This makes downstream verification and reporting safer than parsing free-form
prose.

## 11. What happens when the LLM violates policy?

The verifier returns a structured violation list and concise revision feedback. The model
gets at most one additional recommendation call. If that result still fails—or the
provider/parsing path fails—the graph replaces it with a conservative deterministic
fallback.

## 12. Why did one scenario require revision?

The data-quality scenario's first structured recommendation did not pass the deterministic
verifier, so the graph used its single allowed revision. The corrected recommendation
preserved batch blocking, cited critical data-quality evidence, used safe actions, and
passed verification. The final artifact retains the revision count but not the rejected
payload, so I do not claim a more specific first-pass violation than the stored evidence
supports.

## 13. What does 83.33% first-pass verification mean?

Five of six final live scenarios passed verification on their first recommendation.
The remaining data-quality case passed after one bounded revision. This rate therefore
shows both first-pass quality and that the verifier/revision control worked; it is not a
failure rate because all six final results passed.

## 14. What happens if the LLM API fails?

The adapter captures a safe error category without persisting credentials or raw provider
reasoning. The graph continues with deterministic triage where necessary and produces a
conservative fallback recommendation. A non-normal fallback still requires human
approval, so provider failure does not authorize an action.

## 15. How does human approval work?

Every non-normal recommendation reaches a LangGraph `interrupt()` containing a
JSON-serializable review packet. The same thread resumes through `Command(resume=...)`
with approve, reject, or request-revision plus reviewer metadata. In this MVP,
request-revision is a terminal human decision and no remediation is executed.

Memory is the default backend. For the durable-resume proof, one process persisted
`persistent-demo-001` with SQLite and exited at the interrupt; a separate process loaded
that thread and approved it without another Groq call. The original run ID, evidence, and
recommendation were preserved. SQLite remains a local-development backend.

## 16. What are the six scenarios?

They are normal operation, feature drift, data-quality failure, and synthetic performance
degradation, plus unlabelled drift and insufficient labels. Each uses a 1,000-row
controlled replay generated with seed `20260725`; the insufficient-label case retains
only 100 aligned outcomes. Together they exercise safe completion, mixed drift analysis,
inference blocking, labelled degradation, pre-outcome monitoring, and safe abstention.

## 17. How was performance degradation simulated?

The generator selects 180 observed negatives with the lowest predicted risk and flips
their labels to positive. Features and model probabilities remain unchanged, so labelled
performance deteriorates without introducing feature drift. This is a deterministic
outcome/concept-drift demonstration, not a real production event.

## 18. Why is feature drift not automatically model failure?

Feature drift measures distribution change, not business harm or predictive failure by
itself. It can arise from population change, data-pipeline behavior, seasonality, or
legitimate product changes. The workflow therefore asks for investigation and supporting
performance/label evidence rather than declaring causality.

## 19. Why not automatically retrain?

Retraining changes a governed model and can introduce new bias, instability, leakage, or
performance regressions. The policy permits only an evaluation recommendation when
performance is measured on enough labels and critical performance evidence is cited.
Training, validation, approval, and deployment remain outside this workflow.

## 20. How was the project evaluated?

The final agent ran through the real Groq API on all six controlled scenarios. The
extension reused the original four results and called Groq only for the two delayed-label
cases. A
deterministic evaluator compared incidents and routes with fixed expectations, checked
every citation against monitoring evidence, reran hard policy, and measured revision,
fallback, approval, latency, and tokens. No second LLM judged the answers.

## 21. What do the 100% evaluation rates mean?

In these six final runs, all incidents and routes were compatible, every claim/action
was grounded, hard policy passed, and approval behavior completed correctly. They do not
mean the LLM is universally accurate or the system covers every possible incident. The
denominator is six deliberately constructed replay cases.

## 22. Why are six scenarios not enough for production readiness?

Six cases cannot estimate rare failure rates, repeated-run variance, provider
availability, unseen data corruption, or performance across real populations. The
thresholds are provisional and the data is a public academic dataset rather than
production traffic. Production readiness would also require security, authentication,
durable state, observability, incident operations, governance sign-off, and a broader
benchmark.

## 23. What would you implement next?

I would replace local SQLite with a governed production-grade checkpoint/incident
backend, then expand the benchmark with repeated and edge-case scenarios. I would govern thresholds against
operating costs and test another approved model bundle through the same contract. Only
after those controls would I consider deployment and operational integrations.

## 24. How can this become multi-agent later?

I would not split it merely to increase the agent count. Specialist subgraphs could be
justified for fairness review, challenger comparison, or causal data-pipeline
investigation if each domain had distinct evidence, tools, permissions, and owners. The
main orchestrator would still enforce shared evidence and approval contracts.

## 25. What are the largest limitations?

The largest limitations are the six synthetic replays, provisional thresholds, one
public dataset/model, one live provider/model, and a small evaluation sample. Operationally
SQLite is only a local checkpoint backend; there is no production-grade database,
streaming ingestion, authentication, or external incident system. The project demonstrates
architecture and controls rather than production behavior.

## 26. What did you personally learn from the project?

I learned that the most important GenAI control is often a strong deterministic boundary,
not a longer prompt. Live evaluation exposed practical issues—citation provenance,
parse-time semantics, provider token budgets, Windows console encoding, and approval
accounting—that offline happy paths did not reveal. I also learned to treat the revision
and fallback paths as measurable product behavior rather than decorative architecture.
