# Limitations

The project demonstrates a controlled model-monitoring and recommendation architecture.
These limitations define what the generated evidence does—and does not—support.

## Data limitations

- The model uses the public UCI Default of Credit Card Clients / Taiwan credit-card
  default dataset, not bank production data.
- The 6,002-row reference is a group-aware stratified random held-out split because the
  processed source has no trustworthy application timestamp.
- `record_id` is the processed CSV row position, not an original customer identifier.
- Public academic data may not represent current products, populations, underwriting
  policies, reporting practices, or regulatory contexts.
- Sensitive-group fairness analysis is not implemented because governed group definitions
  and a valid fairness protocol are outside the current bundle contract.

## Simulation limitations

- The benchmark contains six controlled 1,000-row scenarios, not an open-ended incident
  distribution.
- Feature drift changes selected predictors deterministically and does not preserve every
  real-world dependency.
- Data-quality faults are deliberately concentrated and easier to interpret than many
  organic pipeline failures.
- Performance degradation is created by flipping 180 low-risk negative labels to
  positive; it is a synthetic outcome/concept-drift demonstration.
- The unlabelled case deliberately removes outcomes; the partial-label case exposes only
  100 aligned outcomes (10% coverage), below the 200-label evaluation minimum.
- Scenario success does not estimate production incident prevalence, recurrence, or
  detection power.

## Monitoring limitations

- PSI, KS, Jensen-Shannon divergence, missingness, and performance thresholds are
  replay-MVP settings, not production-governed limits.
- Drift indicates association/distribution change, not causality or model failure.
- Labelled performance can only be evaluated after enough reliable outcomes arrive.
- The MVP does not implement fairness monitoring, real-time calibration surveillance,
  delayed-label correction, seasonality models, or champion/challenger comparison.
- Reference ranges come from one held-out sample and are not universal validity bounds.

## LLM limitations

- Live evaluation uses one provider/model: Groq `openai/gpt-oss-20b`.
- Temperature zero reduces but does not eliminate output variance.
- Strict schemas constrain shape and vocabulary, not all semantic quality.
- Provider rate limits, timeouts, outages, or schema behavior can still cause fallback.
- The prompt cannot guarantee grounding; safety depends on deterministic verification.
- The system does not evaluate long-form prose quality or compare multiple LLMs.

## Evaluation limitations

- All `100%` rates use a denominator of six final controlled live runs.
- First-pass verification of `83.33%` means five passed immediately and one passed after a
  bounded revision; it does not estimate a population rate.
- The final fallback rate was `0%`, although fallback behavior is covered by focused fake
  tests rather than needed by the six final live outputs.
- One successful run per final scenario does not measure repeated-run stability or rare
  provider failures.
- Deterministic evaluation checks contracts, citations, routing, and policy; it does not
  prove causal correctness, usefulness to reviewers, or regulatory sufficiency.
- The benchmark was not run on production traffic.

## Operational limitations

- Memory is the default checkpointer. Optional SQLite survives a local process restart,
  but it is not production-grade persistence or a governed incident database.
- There is no production-grade case database, authentication, authorization, encryption design,
  audit-log service, secrets manager, or retention policy.
- There is no streaming ingestion, scheduled execution, alerting, ticketing, email, or
  production observability integration.
- The CLI can replace the same scenario/mode report; it is not an immutable incident
  archive.
- Runtime artifacts are Git-ignored and must be reproduced or transferred through a
  governed artifact process.
- The project is not deployed and makes no service-level availability claim.

## Ethical and governance limitations

- Recommendations are decision support, not credit decisions or customer-level actions.
- Approval records a reviewer decision but does not establish that an organization has
  completed its required model-risk governance.
- The source model and public dataset may contain historical or sampling biases that this
  project does not resolve.
- Automatic retraining or threshold changes could create material harm; the workflow only
  proposes evaluation and never executes them.
- Production use would require accountable owners, documented thresholds, model and data
  governance, fairness assessment, validation independence, incident procedures, and
  regulatory review.

## Appropriate readiness statement

The defensible conclusion is `live_agent_validated_with_limitations` for six controlled
replay scenarios. It is not “production-ready,” “validated on production traffic,”
“real-time,” “autonomously remediating,” or a deployed multi-agent system.
