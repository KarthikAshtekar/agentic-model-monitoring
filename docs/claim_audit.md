# Claim Audit

This audit maps portfolio wording to implementation and generated evidence. Classifications
use five controlled labels: `directly validated`, `reasonable architectural description`,
`simulation-specific result`, `future extension`, and `unsupported`.

| Claim | Classification | Supporting artifact | Permitted wording | Wording to avoid |
|---|---|---|---|---|
| Strict two-model registry | directly validated | [`model_registry.yaml`](../configs/model_registry.yaml), [registry tests](../tests/models/test_registry.py) | “Registered two binary classifiers through strict manifests and one adapter.” | “Supports arbitrary model families automatically.” |
| Read-only diabetes onboarding | directly validated | [onboarding report](onboarding/diabetes_onboarding_report.md), [validation result](../registered_models/diabetes_risk/onboarding/validation_result.json) | “Reused an existing fitted BRFSS model and calibrator without retraining.” | “Retrained or clinically validated the diabetes model.” |
| Domain-aware policy | directly validated | [domain packs](../src/monitoring_agent/domains/), [policy tests](../tests/agent/test_domain_policy.py) | “Separated credit and screening terminology/actions through strict domain packs.” | “Provides medical diagnosis or lending decisions.” |
| LangGraph-based agentic workflow | directly validated | [`agent/graph.py`](../src/monitoring_agent/agent/graph.py), [live reports](../reports/generated/) | “Built a bounded LangGraph monitoring workflow.” | “Deployed an autonomous production agent.” |
| Evidence-backed recommendations | directly validated | [cross-model summary](../reports/evaluations/cross_model/cross_model_summary.json), [`verifier.py`](../src/monitoring_agent/agent/verifier.py) | “All 12 controlled live results grounded every final claim and action in deterministic evidence IDs.” | “The LLM cannot hallucinate under any conditions.” |
| One orchestrator agent | directly validated | [`graph.py`](../src/monitoring_agent/agent/graph.py), [ADR](adr/001-single-orchestrator-agent.md) | “Uses one LangGraph orchestrator.” | “Uses a multi-agent architecture.” |
| Conditional diagnostic routing | directly validated | [`graph.py`](../src/monitoring_agent/agent/graph.py), [live evaluation CSV](../reports/evaluations/live_groq/live_scenario_comparison.csv) | “Routes cases to data-quality, drift, performance, mixed, sufficiency, or no-additional diagnostics.” | “Agents autonomously discover arbitrary tools.” |
| Strict structured LLM output | directly validated | [`llm.py`](../src/monitoring_agent/agent/llm.py), [cross-model summary](../reports/evaluations/cross_model/cross_model_summary.json) | “Groq uses strict `json_schema`; 10/12 controlled results completed both live structured calls.” | “All future provider responses are guaranteed valid.” |
| Deterministic evidence verification | directly validated | [`verifier.py`](../src/monitoring_agent/agent/verifier.py), [evaluation report](../reports/evaluations/live_groq/live_evaluation_report.md) | “A deterministic verifier checks citations, severity, uncertainty, and policy.” | “An LLM judge proves factual correctness.” |
| Human-in-the-loop approval | directly validated | [`nodes.py`](../src/monitoring_agent/agent/nodes.py), [live reports](../reports/generated/) | “Non-normal recommendations pause for explicit approval.” | “The workflow executes approved remediation.” |
| Revision loop | directly validated | [data-quality live result](../reports/generated/data_quality_failure/live_groq/agentic_result.json) | “One bounded revision corrected the data-quality recommendation.” | “The agent self-improves until it is correct.” |
| Deterministic fallback | directly validated | [`fallback.py`](../src/monitoring_agent/agent/fallback.py), [diabetes evaluation](../reports/evaluations/diabetes_risk/live_groq_six_scenarios/live_evaluation_report.md) | “Two first-run diabetes cases safely used conservative fallback after provider/verification failures.” | “Fallback counts as successful live structured output.” |
| Repeat reliability check | directly validated | [repeat reliability report](../reports/evaluations/diabetes_risk/reliability_rerun/reliability_report.md) | “Two affected diabetes cases were repeated once; both avoided fallback, while one needed a bounded revision.” | “The rerun proves 100% LLM availability or replaces the first-run metrics.” |
| Model monitoring | reasonable architectural description | [`monitoring/`](../src/monitoring_agent/monitoring/), [methodology](monitoring_methodology.md) | “Implements replay-based model monitoring.” | “Operates a live production monitoring service.” |
| Feature drift detection | simulation-specific result | [feature-drift monitoring result](../reports/generated/feature_drift/monitoring_result.json) | “Detected controlled feature and prediction drift in the replay scenario.” | “Detected organic drift in production customers.” |
| Data-quality detection | simulation-specific result | [data-quality monitoring result](../reports/generated/data_quality_failure/monitoring_result.json) | “Blocked a replay batch containing controlled duplicate, missing, and range faults.” | “Prevents all production data incidents.” |
| Performance degradation detection | simulation-specific result | [performance monitoring result](../reports/generated/performance_degradation/monitoring_result.json) | “Detected degradation after controlled label modification.” | “Detected real concept drift in a deployed bank model.” |
| Production readiness | unsupported | [cross-model evaluation](../reports/evaluations/cross_model/cross_model_report.md), [limitations](limitations.md) | “Completed 12 controlled replays; the current cross-model verdict requires revision.” | “Production-ready.” |
| Real-time monitoring | unsupported | [architecture runtime boundary](architecture.md#runtime-boundary) | “A future real-time integration would require new ingestion and operational controls.” | “Real-time production monitoring.” |
| Automated remediation | unsupported | [`policy.py`](../src/monitoring_agent/agent/policy.py), [live reports](../reports/generated/) | “Produces recommendations for human review; executes no remediation.” | “Automatically retrained the model.” / “Autonomously remediated incidents.” |
| Multi-agent system | unsupported | [ADR](adr/001-single-orchestrator-agent.md) | “One orchestrator; specialist subgraphs are future options.” | “Multi-agent architecture.” |
| Deployment | future extension | [architecture](architecture.md), [limitations](limitations.md) | “Deployment and observability are future work.” | “Cloud-deployed monitoring platform.” |
| Live production usage | unsupported | [model metadata](../artifacts/metadata/model_metadata.json), [evaluation summary](../reports/evaluations/live_groq/live_evaluation_summary.json) | “Used a live Groq API against controlled local replay scenarios.” | “Validated on production traffic.” / “Used in live bank operations.” |

## Portfolio wording rule

Use “live” only for the Groq provider execution. Use “controlled replay,” “synthetic,” or
“simulation” for monitoring outcomes. Never let “live Groq evaluation” imply live
production data, a deployed service, automated remediation, medical validation, or
production readiness.

## Repeat reliability check

The original 12-case evaluation remains authoritative for portfolio and CV metrics. The
two supplementary diabetes repeats changed no prompts, evidence, schemas, or policies.
A successful repeat does not erase the first-run provider failure, and two observations
do not support a general reliability estimate. The deterministic fallback is part of the
intended safety architecture.
