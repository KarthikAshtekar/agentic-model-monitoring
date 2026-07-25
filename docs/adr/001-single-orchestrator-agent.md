# ADR 001: Use One Bounded LangGraph Orchestrator

## Status

Accepted for the replay MVP.

## Context

The workflow consumes one authoritative monitoring result containing batch validity,
drift, performance, evidence IDs, incident candidates, and deterministic severity. It must
triage the case, select a diagnostic view, request a recommendation, enforce evidence and
policy rules, optionally revise once, fall back safely, and pause for human approval.

The analytical domains are implemented as deterministic Python functions. They do not
need independent goals, memory, permissions, or external tools. Introducing several
agents would therefore create coordination overhead without adding a distinct source of
truth.

## Decision

Use one LangGraph monitoring orchestrator. Keep data-quality, drift, performance,
evidence selection, policy verification, fallback, evaluation, and reporting
deterministic. Use one structured LLM for bounded triage and recommendation synthesis.
Do not decompose the MVP into multiple agents.

The graph explicitly limits autonomy:

- controlled incident, route, severity, and action values;
- exact evidence-ID citations;
- one recommendation revision maximum;
- deterministic fallback after provider, parse, or repeated-verification failure; and
- human approval before finalizing every non-normal recommendation.

## Rationale

1. **One evidence contract:** all decisions use the same monitoring result, so specialist
   agent memories would duplicate state.
2. **Lower failure surface:** one provider boundary reduces prompt drift, token cost,
   coordination errors, and ambiguous ownership.
3. **Clear governance:** deterministic nodes remain authoritative and the LLM has a
   narrow, auditable role.
4. **Testability:** routing, revision, fallback, interruption, and resume can be tested
   through one serializable graph state.
5. **Portfolio clarity:** the architecture demonstrates agentic control flow without
   implying that agent count is a quality metric.

## Consequences

### Positive

- Simpler state and checkpoint model
- Explicit conditional edges and terminal statuses
- Easier evidence and policy auditing
- Lower live-provider token usage
- Straightforward fake-provider injection for offline tests
- Clear statement that deterministic tools are not agents

### Negative

- One prompt/schema pair must cover several incident types
- The orchestrator may become crowded if materially different governed domains are added
- In-memory checkpointing is not sufficient for durable production cases
- A single provider/model remains a shared availability and behavior dependency

## Alternatives considered

### Linear Python pipeline

Rejected as the primary orchestration representation because interruption/resume,
conditional routing, bounded revision, and fallback are first-class workflow states.
The underlying analytics remain normal Python functions.

### One agent per monitoring domain

Rejected for the MVP. Data quality, drift, and performance do not have distinct
permissions or evidence stores, and their calculations are already deterministic.
Separate agents would duplicate context and require another mechanism to reconcile
conflicting answers.

### Free-form autonomous tool use

Rejected because the project needs traceable claims and policy-constrained actions.
Arbitrary tool loops would expand scope and make termination, grounding, and governance
harder to verify.

### LLM-as-judge

Rejected for final evaluation. Evidence-ID validity, incident expectations, and hard
policy are deterministic questions; a second LLM would add stochasticity without becoming
authoritative.

## Future evolution

Specialist subgraphs become justified only when a domain has distinct evidence, tools,
permissions, owners, or approval rules. Examples could include fairness assessment using
governed group attributes, challenger-model comparison using a separate approved bundle,
or data-pipeline investigation with lineage access.

Those subgraphs should remain subordinate to a shared evidence contract, deterministic
policy verification, bounded execution, and human approval. Future evolution should not
be described as implemented until those capabilities and their evaluations exist.

