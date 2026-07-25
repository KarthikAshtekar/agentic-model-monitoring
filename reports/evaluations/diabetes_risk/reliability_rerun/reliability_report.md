# Repeat reliability check

- Evaluation type: `repeat_reliability_run`
- Replaces original evaluation: `false`
- Original result preserved: `true`
- Model ID: `diabetes_risk`
- Scenario: `feature_drift, unlabelled_drift`
- Provider: `groq`
- Provider model: `openai/gpt-oss-20b`
- Execution timestamp: `2026-07-25T11:34:04.900918+00:00`
- Original result path: `reports/generated/diabetes_risk/feature_drift/live_groq/agentic_result.json, reports/generated/diabetes_risk/unlabelled_drift/live_groq/agentic_result.json`
- Repeat result path: `reports/generated/diabetes_risk/reliability_rerun_01/feature_drift/agentic_result.json, reports/generated/diabetes_risk/reliability_rerun_01/unlabelled_drift/agentic_result.json`
- Original hashes verified: `true`

## Objective

Determine whether the two first-run diabetes fallbacks reproduced under the same model
bundle, scenario inputs, prompts, evidence, schemas, and deterministic policies. This is
supplementary to, and does not replace, the authoritative first-run evaluation.

## Repeat results

| Scenario | Classification | Structured | Final verifier | Fallback | Provider error | Output error | LLM latency ms |
|---|---|---:|---:|---:|---:|---:|---:|
| `feature_drift` | `provider_failure_not_reproduced` | yes | yes | no | no | no | 9383.07 |
| `unlabelled_drift` | `mixed_repeat_outcome` | yes | yes | no | no | no | 78258.36 |

### `feature_drift`

- Original issue: Groq recommendation HTTP 429.
- Classification: `provider_failure_not_reproduced`.
- Original provider failure reproduced: `false`.
- Original model-output verification failure reproduced: `false`.
- Different repeat recommendation revision required: `false`.
- Repeat incident/route: `mixed_incident` /
  `mixed_diagnostics`.
- Repeat grounding/policy/approval: `true` /
  `true` /
  `true`.
- Repeat revisions/fallback: `0` /
  `false`.
- Repeat tokens: `7681` input /
  `2577` output / `10258` total.
- Original result: `reports/generated/diabetes_risk/feature_drift/live_groq/agentic_result.json`.
- Repeat result: `reports/generated/diabetes_risk/reliability_rerun_01/feature_drift/agentic_result.json`.

### `unlabelled_drift`

- Original issue: Incompatible initial triage rejected by deterministic verification; subsequent Groq recommendation HTTP 429.
- Classification: `mixed_repeat_outcome`.
- Original provider failure reproduced: `false`.
- Original model-output verification failure reproduced: `false`.
- Different repeat recommendation revision required: `true`.
- Repeat incident/route: `feature_drift` /
  `drift_diagnostics`.
- Repeat grounding/policy/approval: `true` /
  `true` /
  `true`.
- Repeat revisions/fallback: `1` /
  `false`.
- Repeat tokens: `9860` input /
  `3902` output / `13762` total.
- Original result: `reports/generated/diabetes_risk/unlabelled_drift/live_groq/agentic_result.json`.
- Repeat result: `reports/generated/diabetes_risk/reliability_rerun_01/unlabelled_drift/agentic_result.json`.

## Aggregate repeat-only metrics

- Structured-output success: `100.0%`
- Incident compatibility: `100.0%`
- Route compatibility: `100.0%`
- Evidence grounding: `100.0%`
- Policy compliance: `100.0%`
- First-pass verification: `50.0%`
- Fallback: `0.0%`
- Approval completion: `100.0%`
- Mean repeat LLM latency: `43820.71` ms
- Repeat tokens: `17541` input /
  `6479` output /
  `24020` total

## Authoritative metrics remain unchanged

The original diabetes evaluation remains `6/6` complete with `66.67%` structured-output
success and `33.33%` fallback. The original cross-model evaluation remains `12/12`
complete with `83.33%` structured-output success and `16.66%` fallback. These first-run
figures remain the headline evaluation and CV metrics.

## Interpretation

The deterministic fallback is part of the intended safety architecture. A successful
repeat does not erase the first-run failure, and a failed repeat is not automatically an
implementation defect. No universal or production reliability claim is made.

## Limitations

- This is a two-case repeat reliability check, not a replacement evaluation.
- The original 12-case first-run evaluation remains authoritative.
- A successful repeat does not erase an original provider or output failure.
- Two repeat calls do not estimate general or production reliability.
- No prompts, evidence, routing, verification, revision, or approval policies changed.
