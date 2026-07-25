# Live Groq agent evaluation

## Evaluation objective

Evaluate the existing one-orchestrator monitoring agent across four controlled replay
scenarios using deterministic structured-output, routing, grounding, policy, verification,
fallback, approval, latency, and token checks. No LLM-as-judge was used.

## Provider and model

- Provider: `groq`
- Model: `openai/gpt-oss-20b`
- Execution date: `2026-07-25T03:12:17.784751+00:00`
- Scenarios completed: `4/4`

## Scenario-level results

| Scenario | Incident | Route | Structured | Grounded | Policy | Fallback | Final status |
|---|---|---|---|---|---|---|---|
| normal_operation | `normal_operation` | `no_additional_diagnostics` | yes | yes | yes | no | `completed_no_approval_required` |
| feature_drift | `mixed_incident` | `mixed_diagnostics` | yes | yes | yes | no | `approved` |
| data_quality_failure | `data_quality_failure` | `data_quality_diagnostics` | yes | yes | yes | no | `approved` |
| performance_degradation | `performance_degradation` | `performance_diagnostics` | yes | yes | yes | no | `approved` |

## Structured-output success

`100.0%`. A fallback is explicitly excluded
from live structured-output success.

## Incident and routing compatibility

- Incident compatibility: `100.0%`
- Route compatibility: `100.0%`

## Evidence grounding

`100.0%` of scenarios had only authoritative
deterministic evidence IDs and cited every claim and action.

## Policy compliance

`100.0%` under the deterministic hard and
scenario-specific policies.

## Verification and revision behaviour

- First-pass verification: `75.0%`
- `normal_operation`: `0` revision(s), final verifier pass `true`.
- `feature_drift`: `0` revision(s), final verifier pass `true`.
- `data_quality_failure`: `1` revision(s), final verifier pass `true`.
- `performance_degradation`: `0` revision(s), final verifier pass `true`.

## Fallback usage

Fallback rate: `0.0%`.

## Approval behaviour

Approval completion rate: `100.0%`.
- `normal_operation`: required `false`, completed `true`, status `completed_no_approval_required`.
- `feature_drift`: required `true`, completed `true`, status `approved`.
- `data_quality_failure`: required `true`, completed `true`, status `approved`.
- `performance_degradation`: required `true`, completed `true`, status `approved`.

## Latency and token usage

- Mean scenario LLM latency: `3538.51` ms
- Median scenario LLM latency: `3546.1` ms
- Aggregate tokens: `18271` input /
  `7546` output / `25817` total
- `normal_operation`: `2673.26` ms; tokens `4515` input / `1296` output / `5811` total.
- `feature_drift`: `4457.58` ms; tokens `5861` input / `2098` output / `7959` total.
- `data_quality_failure`: `4418.93` ms; tokens `4563` input / `2661` output / `7224` total.
- `performance_degradation`: `2604.25` ms; tokens `3332` input / `1491` output / `4823` total.

## Concrete defects found and fixed

- Pre-live grounding audit removed synthetic citable SYSTEM IDs; run-level batch and label facts remain in structured case context, while citations now resolve only to the authoritative monitoring evidence registry.
- Agentic report paths were separated into fake/ and live_groq/ with explicit provider, model, execution-mode, fake-LLM, run, thread, and UTC provenance fields.
- Initial non-normal recommendation parsing could bypass the bounded verifier because semantic root-cause checks ran inside Pydantic parsing. Those checks now run in the deterministic verifier, where one revision remains available.
- Initial non-normal packets included ten irrelevant pass records and exceeded the provider's token-per-minute budget across sequential triage and recommendation calls. Pass padding is now limited to normal-operation stability evidence.
- A live feature-drift approval display initially failed on a non-breaking hyphen under the Windows console encoding. Approval payloads are now rendered as ASCII-safe escaped JSON without changing the structured payload.
- Initial data-quality and performance triage attempts were rate-limited and safely fell back; their targeted reruns were retained as the final live evaluation outputs.

## Limitations

- The evaluation contains four controlled replay scenarios, not production traffic.
- Monitoring thresholds and synthetic transformations are not production-validated.
- Deterministic checks assess schema, evidence, routing, and policy, not prose quality.
- A successful replay evaluation does not establish production readiness.

## Final readiness verdict

`live_agent_validated_with_limitations`

This verdict is limited to controlled replay validation and is not a production-readiness
or deployment claim. No remediation was executed.
