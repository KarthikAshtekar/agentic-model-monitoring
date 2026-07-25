# Limitations

The project demonstrates a controlled model-monitoring and recommendation architecture.
These limitations define what the generated evidence does—and does not—support.

The registry contains two academic binary classifiers and two domain packs. The credit
six-run evaluation is preserved; the diabetes six-run evaluation has the factual verdict
`live_agent_requires_revision` because two recommendation calls used fallback. The
combined 12-run verdict is `cross_model_agent_requires_revision`.

## Data limitations

- The model uses the public UCI Default of Credit Card Clients / Taiwan credit-card
  default dataset, not bank production data.
- The diabetes model uses self-reported CDC BRFSS 2015 indicators and outcomes. It is
  neither current clinical data nor external medical validation.
- Both references are internal random held-out splits rather than temporal, prospective,
  or externally governed production baselines.
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

- Model-specific percentages use six controlled live runs; cross-model percentages use
  12. Neither is a repeated-run reliability estimate.
- The diabetes run recorded two Groq recommendation rate limits and one triage rejected
  by deterministic compatibility checks. Fallback kept the workflow safe but is excluded
  from structured-output and first-pass success.
- Credit first-pass verification was `83.33%`; diabetes first-pass verification was
  `66.67%`; combined first-pass verification was `75%`.
- Credit fallback was `0%`; diabetes fallback was `33.33%`; combined fallback was
  `16.66%`.
- One successful run per final scenario does not measure repeated-run stability or rare
  provider failures.
- Deterministic evaluation checks contracts, citations, routing, and policy; it does not
  prove causal correctness, usefulness to reviewers, or regulatory sufficiency.
- The benchmark was not run on production traffic.

### Repeat reliability check

The two diabetes fallback cases were repeated once in isolation without changing prompts,
evidence, schemas, or policies. Neither original HTTP 429 recurred, and the unlabelled
case did not repeat its incompatible triage. Feature drift passed first-pass; unlabelled
drift required one bounded recommendation revision before passing. Both completed without
fallback.

This supplementary two-case check does not replace the first-run evaluation, erase its
failures, establish `100%` LLM availability, or provide a repeated-run reliability
estimate. The original 12-case headline metrics remain authoritative. See the
[repeat reliability report](../reports/evaluations/diabetes_risk/reliability_rerun/reliability_report.md).

## Operational limitations

- Memory is the default checkpointer. Optional SQLite survives a local process restart,
  but it is not production-grade persistence or a governed incident database.
- There is no production-grade case database, authentication, authorization, encryption design,
  audit-log service, secrets manager, or retention policy.
- There is no streaming ingestion, scheduled execution, alerting, ticketing, email, or
  production observability integration.
- Default CLI paths can replace the same scenario/mode report. A unique `--run-label`
  isolates a controlled repeat, but the CLI is still not a general immutable incident
  archive.
- Runtime artifacts are Git-ignored and must be reproduced or transferred through a
  governed artifact process.
- The project is not deployed and makes no service-level availability claim.

## Ethical and governance limitations

- Recommendations are decision support, not credit decisions or customer-level actions.
- Diabetes recommendations are monitoring decision support, not diagnosis, treatment,
  individual patient care, or autonomous clinical action.
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

The defensible combined conclusion is `cross_model_agent_requires_revision` for 12
controlled replay scenarios. It is not “production-ready,” “validated on production traffic,”
“real-time,” “autonomously remediating,” or a deployed multi-agent system.
