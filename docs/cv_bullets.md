# CV and Portfolio Wording

## Project title options

1. **Agentic Model Risk & Monitoring Copilot** — recommended
2. **Evidence-Grounded LangGraph Model Monitoring Agent**
3. **Human-Governed Agentic Model Monitoring**
4. **Adapter-Based Cross-Domain Model Monitoring Copilot**

## Recommended two-pointer CV version

- Built a registry-driven LangGraph monitoring copilot for credit and diabetes-risk
  classifiers, with strict domain policies and human-approved Groq recommendations.
- Completed 12 controlled live replays with 100% incident/routing compatibility,
  evidence grounding, policy compliance, and approval completion.

## Recommended three-pointer CV version

- Built a LangGraph monitoring copilot with conditional routing, strict Groq outputs, and explicit human approval.
- Grounded every claim and action in deterministic evidence IDs, with policy checks and one bounded revision.
- Onboarded an existing fitted BRFSS XGBoost/calibration pipeline without retraining and reproduced 50,736 held-out scores within 2e-7.

## Four-pointer CV version

- Registered two self-contained binary-classifier bundles behind one strict adapter and model registry.
- Implemented deterministic data-quality, feature/prediction-drift, and labelled-performance evidence checks.
- Orchestrated strict Groq triage with LangGraph routing, evidence verification, bounded revision, and approval.
- Evaluated 12 controlled live runs with 100% grounding/policy and a measured 16.66% safe fallback rate.

## One-line project summary

Built a registry-driven LangGraph copilot that monitors credit and diabetes-risk replay
evidence, verifies structured Groq recommendations, and gates non-normal actions.

## Repeat reliability check

Keep the original 12-case first-run metrics in CV and portfolio wording. Two affected
diabetes cases were repeated separately with unchanged prompts, evidence, and policies;
both completed without fallback, while unlabelled drift used one bounded revision. This
supplementary check does not support a `100%` availability or production-reliability
claim and does not replace the original measured `83.33%` structured-output and `16.66%`
fallback rates.

Final defensible claim:

> Built an adapter-based LangGraph monitoring framework validated across credit-risk and
> diabetes-screening binary classifiers.

## Technology line

Python · LangGraph/SQLite · Groq GPT-OSS 20B · Pydantic · scikit-learn/XGBoost · pandas ·
PyArrow · pytest · Ruff

## LinkedIn or project-description version

Built a replay-based Agentic Model Risk & Monitoring Copilot around two self-contained
binary-classifier bundles. Deterministic Python components calculate data-quality,
feature/prediction-drift, and labelled-performance evidence; one LangGraph orchestrator
routes Groq GPT-OSS triage and recommendations through exact evidence-ID and policy
verification. Non-normal recommendations pause for human approval, with one bounded
revision and a deterministic fallback. Across 12 controlled live-provider replays, the
final outputs achieved 100% incident/route compatibility, evidence grounding, policy
compliance, and approval completion; live structured output was 83.33% and safe fallback
was 16.66%. The BRFSS model was reused without retraining and is framed as survey-based
screening, not diagnosis. This is a portfolio replay MVP, not a production deployment.

## Claims to avoid

- “Production-ready” or “deployed to production”
- “Real-time production monitoring”
- “Multi-agent architecture”
- “Validated on production traffic”
- “Automatically retrained the model”
- “Autonomously remediated incidents”
- “100% accurate” or “hallucination-proof”
- “Detected real concept drift” without the qualifier that labels were deliberately
  modified in a controlled replay
- “Generalizes to all model-risk incidents” based on six scenarios
- “Clinically validated,” “diagnostic AI,” or “treatment recommendation”
