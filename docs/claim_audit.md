# Claim Audit

This audit maps portfolio wording to implementation and generated evidence. Classifications
use five controlled labels: `directly validated`, `reasonable architectural description`,
`simulation-specific result`, `future extension`, and `unsupported`.

| Claim | Classification | Supporting artifact | Permitted wording | Wording to avoid |
|---|---|---|---|---|
| LangGraph-based agentic workflow | directly validated | [`agent/graph.py`](../src/monitoring_agent/agent/graph.py), [live reports](../reports/generated/) | “Built a bounded LangGraph monitoring workflow.” | “Deployed an autonomous production agent.” |
| Evidence-backed recommendations | directly validated | [evaluation summary](../reports/evaluations/live_groq/live_evaluation_summary.json), [`verifier.py`](../src/monitoring_agent/agent/verifier.py) | “Every live claim and action cited deterministic evidence IDs in the four-scenario evaluation.” | “The LLM cannot hallucinate under any conditions.” |
| One orchestrator agent | directly validated | [`graph.py`](../src/monitoring_agent/agent/graph.py), [ADR](adr/001-single-orchestrator-agent.md) | “Uses one LangGraph orchestrator.” | “Uses a multi-agent architecture.” |
| Conditional diagnostic routing | directly validated | [`graph.py`](../src/monitoring_agent/agent/graph.py), [live evaluation CSV](../reports/evaluations/live_groq/live_scenario_comparison.csv) | “Routes cases to data-quality, drift, performance, mixed, sufficiency, or no-additional diagnostics.” | “Agents autonomously discover arbitrary tools.” |
| Strict structured LLM output | directly validated | [`llm.py`](../src/monitoring_agent/agent/llm.py), [evaluation summary](../reports/evaluations/live_groq/live_evaluation_summary.json) | “Groq triage and recommendations use strict `json_schema`; all four final live runs parsed.” | “All future provider responses are guaranteed valid.” |
| Deterministic evidence verification | directly validated | [`verifier.py`](../src/monitoring_agent/agent/verifier.py), [evaluation report](../reports/evaluations/live_groq/live_evaluation_report.md) | “A deterministic verifier checks citations, severity, uncertainty, and policy.” | “An LLM judge proves factual correctness.” |
| Human-in-the-loop approval | directly validated | [`nodes.py`](../src/monitoring_agent/agent/nodes.py), [live reports](../reports/generated/) | “Non-normal recommendations pause for explicit approval.” | “The workflow executes approved remediation.” |
| Revision loop | directly validated | [data-quality live result](../reports/generated/data_quality_failure/live_groq/agentic_result.json) | “One bounded revision corrected the data-quality recommendation.” | “The agent self-improves until it is correct.” |
| Deterministic fallback | directly validated | [`fallback.py`](../src/monitoring_agent/agent/fallback.py), [`test_fallback.py`](../tests/agent/test_fallback.py) | “Provider, parse, or repeated verification failure produces a conservative fallback.” | “Fallback was needed in the final four live results.” |
| Model monitoring | reasonable architectural description | [`monitoring/`](../src/monitoring_agent/monitoring/), [methodology](monitoring_methodology.md) | “Implements replay-based model monitoring.” | “Operates a live production monitoring service.” |
| Feature drift detection | simulation-specific result | [feature-drift monitoring result](../reports/generated/feature_drift/monitoring_result.json) | “Detected controlled feature and prediction drift in the replay scenario.” | “Detected organic drift in production customers.” |
| Data-quality detection | simulation-specific result | [data-quality monitoring result](../reports/generated/data_quality_failure/monitoring_result.json) | “Blocked a replay batch containing controlled duplicate, missing, and range faults.” | “Prevents all production data incidents.” |
| Performance degradation detection | simulation-specific result | [performance monitoring result](../reports/generated/performance_degradation/monitoring_result.json) | “Detected degradation after controlled label modification.” | “Detected real concept drift in a deployed bank model.” |
| Production readiness | unsupported | [evaluation limitations](../reports/evaluations/live_groq/live_evaluation_report.md), [limitations](limitations.md) | “Validated with limitations on four controlled replays.” | “Production-ready.” |
| Real-time monitoring | unsupported | [architecture runtime boundary](architecture.md#runtime-boundary) | “A future real-time integration would require new ingestion and operational controls.” | “Real-time production monitoring.” |
| Automated remediation | unsupported | [`policy.py`](../src/monitoring_agent/agent/policy.py), [live reports](../reports/generated/) | “Produces recommendations for human review; executes no remediation.” | “Automatically retrained the model.” / “Autonomously remediated incidents.” |
| Multi-agent system | unsupported | [ADR](adr/001-single-orchestrator-agent.md) | “One orchestrator; specialist subgraphs are future options.” | “Multi-agent architecture.” |
| Deployment | future extension | [architecture](architecture.md), [limitations](limitations.md) | “Deployment and observability are future work.” | “Cloud-deployed monitoring platform.” |
| Live production usage | unsupported | [model metadata](../artifacts/metadata/model_metadata.json), [evaluation summary](../reports/evaluations/live_groq/live_evaluation_summary.json) | “Used a live Groq API against controlled local replay scenarios.” | “Validated on production traffic.” / “Used in live bank operations.” |

## Portfolio wording rule

Use “live” only for the Groq provider execution. Use “controlled replay,” “synthetic,” or
“simulation” for monitoring outcomes. Never let “live Groq evaluation” imply live
production data, a deployed service, automated remediation, or production readiness.

