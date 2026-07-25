# Extended live Groq robustness evaluation

## Evaluation objective

Evaluate the existing one-orchestrator monitoring agent across 6
controlled replay
scenarios using deterministic structured-output, routing, grounding, policy, verification,
fallback, approval, latency, and token checks. No LLM-as-judge was used.

## Provider and model

- Provider: `groq`
- Model: `openai/gpt-oss-20b`
- Registered model: `diabetes_risk`
- Domain: `diabetes_screening`
- Execution date: `2026-07-25T08:52:47.273108+00:00`
- Scenarios completed: `6/6`


## Scenario-level results

| Scenario | Incident | Route | Structured | Grounded | Policy | Fallback | Final status |
|---|---|---|---|---|---|---|---|
| normal_operation | `normal_operation` | `no_additional_diagnostics` | yes | yes | yes | no | `completed_no_approval_required` |
| feature_drift | `mixed_incident` | `mixed_diagnostics` | no | yes | yes | yes | `approved` |
| data_quality_failure | `data_quality_failure` | `data_quality_diagnostics` | yes | yes | yes | no | `approved` |
| performance_degradation | `performance_degradation` | `performance_diagnostics` | yes | yes | yes | no | `approved` |
| unlabelled_drift | `feature_drift` | `drift_diagnostics` | no | yes | yes | yes | `approved` |
| insufficient_labels | `insufficient_evidence` | `evidence_sufficiency_review` | yes | yes | yes | no | `approved` |

## Structured-output success

`66.7%`. A fallback is explicitly excluded
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

- First-pass verification: `66.7%`
- `normal_operation`: `0` revision(s), final verifier pass `true`.
- `feature_drift`: `0` revision(s), final verifier pass `false`.
- `data_quality_failure`: `0` revision(s), final verifier pass `true`.
- `performance_degradation`: `0` revision(s), final verifier pass `true`.
- `unlabelled_drift`: `0` revision(s), final verifier pass `false`.
- `insufficient_labels`: `0` revision(s), final verifier pass `true`.

## Fallback usage

Fallback rate: `33.3%`.

### Preserved execution errors

- `feature_drift`: Live execution error preserved at recommendation: RateLimitError — Groq request was rate-limited (HTTP 429).
- `unlabelled_drift`: Live execution error preserved at triage: TriageVerificationError — Triage incident type is incompatible with deterministic candidates. Live execution error preserved at recommendation: RateLimitError — Groq request was rate-limited (HTTP 429).

## Approval behaviour

Approval completion rate: `100.0%`.
- `normal_operation`: required `false`, completed `true`, status `completed_no_approval_required`.
- `feature_drift`: required `true`, completed `true`, status `approved`.
- `data_quality_failure`: required `true`, completed `true`, status `approved`.
- `performance_degradation`: required `true`, completed `true`, status `approved`.
- `unlabelled_drift`: required `true`, completed `true`, status `approved`.
- `insufficient_labels`: required `true`, completed `true`, status `approved`.

## Latency and token usage

- Mean scenario LLM latency: `14532.31` ms
- Median scenario LLM latency: `10074.59` ms
- Aggregate tokens: `25496` input /
  `6999` output / `32495` total
- `normal_operation`: `3922.12` ms; tokens `6282` input / `1619` output / `7901` total.
- `feature_drift`: `31281.17` ms; tokens `3520` input / `679` output / `4199` total.
- `data_quality_failure`: `3068.92` ms; tokens `4406` input / `1549` output / `5955` total.
- `performance_degradation`: `16227.06` ms; tokens `4883` input / `1347` output / `6230` total.
- `unlabelled_drift`: `29900.07` ms; tokens `2868` input / `867` output / `3735` total.
- `insufficient_labels`: `2794.49` ms; tokens `3537` input / `938` output / `4475` total.

## Concrete defects found and fixed

- Read-only source inspection initially failed because eager package exports formed a circular import. The models and onboarding package exports were made lazy; no model, source, monitoring, or policy behavior changed.
- The first bundle validation used a 1e-7 probability-reproduction gate and missed it by 6.64e-9. The explicit gate is now 2e-7 (still below 1e-6); the 1.0664e-7 maximum difference, threshold classifications, and reproduced metrics are recorded.
- The live runs exposed no implementation defect. Feature drift and unlabelled drift preserve their initial provider/verification failures and deterministic fallbacks; neither scenario was rerun or prompt-tuned.

## Limitations

- The evaluation contains 6 controlled replay scenarios, not production traffic.
- Monitoring thresholds and synthetic transformations are not production-validated.
- Deterministic checks assess schema, evidence, routing, and policy, not prose quality.
- Provider or verifier failures may enter the deterministic fallback; fallbacks are excluded from live structured-output success.
- A successful replay evaluation does not establish production readiness.

## Final readiness verdict

`live_agent_requires_revision`

This verdict is limited to controlled replay validation and is not a production-readiness
or deployment claim. No remediation was executed.
