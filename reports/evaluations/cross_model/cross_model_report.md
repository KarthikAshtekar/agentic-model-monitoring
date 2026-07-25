# Cross-model live agent evaluation

This report combines already validated model-specific summaries. It does not rerun or
rejudge either model.

| Model | Domain | Complete | Structured | Grounded | Policy | First pass | Revision | Fallback |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `credit_default` | `credit_risk` | 6/6 | 100.0% | 100.0% | 100.0% | 83.3% | 16.7% | 0.0% |
| `diabetes_risk` | `diabetes_screening` | 6/6 | 66.7% | 100.0% | 100.0% | 66.7% | 0.0% | 33.3% |

## Combined controlled-replay metrics

- Models/domains: `2` / `2`
- Scenarios completed: `12/12`
- Structured output: `83.3%`
- Incident compatibility: `100.0%`
- Route compatibility: `100.0%`
- Evidence grounding: `100.0%`
- Policy compliance: `100.0%`
- First-pass verification: `75.0%`
- Revision: `8.3%`
- Fallback: `16.7%`
- Approval completion: `100.0%`

Model-level ROC-AUC, PR-AUC, recall, or other performance metrics are deliberately not
averaged across domains.

## Limitations

- The comparison contains controlled replay scenarios, not production traffic.
- Credit and diabetes performance metrics are not averaged across domains.
- Both model evaluations use internal historical reference splits.
- No deployment, diagnosis, credit decision, or remediation was executed.

## Verdict

`cross_model_agent_requires_revision`

This is not a production-readiness or deployment claim.
