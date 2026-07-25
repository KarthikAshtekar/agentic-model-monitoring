# 3–5 Minute Demo Guide

## Before the meeting

Activate the project environment from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
```

For a live-provider demo, confirm that the untracked `.env` contains `GROQ_API_KEY` and
that `LLM_MODEL=openai/gpt-oss-20b`. Do not display the file or key. Keep the
[extended live evaluation report](../reports/evaluations/live_groq_six_scenarios/live_evaluation_report.md)
open as a no-network backup.

Do not regenerate the bundle, scenarios, or deterministic reports for an ordinary demo.
The existing artifacts are the reviewed source of truth.

## Recommended live sequence

### 1. Show the architecture — 30 seconds

Open the Mermaid diagram in the [README](../README.md#architecture).

Say:

> “The left side is deterministic: bundle loading, monitoring metrics, evidence IDs, and
> incident candidates. One LangGraph orchestrator handles conditional routing and asks
> Groq for structured synthesis. The right side is controlled again: deterministic
> verification, one revision or fallback, and human approval.”

Emphasize that the diagram contains one orchestrator, not multiple agents.

### 2. Show deterministic monitoring output — 30 seconds

Open:

```text
reports/generated/performance_degradation/monitoring_report.md
```

Say:

> “This report exists before the LLM call. Python measured labelled performance against
> the reference baseline and produced stable evidence IDs such as
> `PERF-RECALL-OPERATING` and `PERF-PRAUC`. The LLM cannot change these values.”

### 3. Run the normal case — 30 seconds

```powershell
python scripts\run_agentic_monitoring.py --scenario normal_operation
```

Say:

> “Normal operation should route to `no_additional_diagnostics`, pass verification, and
> finish without an approval interrupt. That is important because human review should be
> reserved for consequential cases.”

If avoiding an extra API call, show the existing report instead:

```powershell
Get-Content reports\generated\normal_operation\live_groq\agentic_report.md
```

### 4. Run the non-normal performance case — 60–90 seconds

Performance degradation is the clearest primary demo because deterministic performance
evidence, LLM interpretation, verification, and approval have distinct roles:

```powershell
python scripts\run_agentic_monitoring.py `
  --scenario performance_degradation `
  --reviewer "demo-reviewer"
```

When prompted, enter:

```text
approve
```

Say:

> “The scenario modifies labels, not features or model scores, so the deterministic engine
> isolates labelled degradation. Groq receives only the compact evidence packet and
> proposes controlled actions. The graph pauses before finalization; approval records a
> decision but does not execute retraining or another remediation.”

### 5. Show evidence IDs and verifier — 30 seconds

Open:

```text
reports/generated/performance_degradation/live_groq/agentic_result.json
```

Point to:

- `claims[].evidence_ids`
- `recommended_actions[].evidence_ids`
- `overall_evidence_ids`
- `verification.status`
- `verification.policy_checks`

Say:

> “Every claim and action cites deterministic IDs. The verifier checks those IDs, incident
> compatibility, severity, uncertainty, action policy, and approval. Prompt instructions
> are helpful, but this deterministic check is the actual control.”

### 6. Show approval and final report — 30 seconds

In the same JSON or Markdown report, show:

- `approval_required`
- `approval_decision`
- `final_status`
- provider/model/execution provenance

Say:

> “The final status is approved, but no action was executed. Fake and live outputs are in
> separate directories, so offline tests cannot be mistaken for live provider results.”

### 7. Show the evaluation summary — 30 seconds

```powershell
python scripts\evaluate_live_agent.py --extended
```

Say:

> “This evaluator makes no network call and uses no judge LLM. Across six controlled
> replays, incident/routing compatibility, grounding, policy, and approval were 100%.
> Five recommendations passed immediately; data quality used the one bounded revision;
> final fallback usage was zero.”

### 8. Optional durable-resume proof — 30 seconds

Show the already-generated report under:

```text
reports/generated/performance_degradation/live_groq_persistent/persistent-demo-001/
```

Point to `checkpoint_backend=sqlite`, `resumed_from_checkpoint=true`,
`pause_count=1`, and `resume_count=1`. Explain that one Python process stopped at the
interrupt and another approved it without another Groq call. This is local SQLite
persistence for development and demonstration, not production-grade durability.

## Existing-artifacts demo with no live calls

Use this sequence when Wi-Fi, API access, or provider limits are uncertain:

```powershell
Get-Content reports\generated\performance_degradation\monitoring_report.md
Get-Content reports\generated\performance_degradation\live_groq\agentic_report.md
python scripts\evaluate_live_agent.py --extended
```

The evaluation command only reads completed local reports. Clearly say that the displayed
agentic report came from an earlier live Groq run.

## Offline fake-provider backup

To demonstrate graph behavior without presenting the result as live:

```powershell
python scripts\run_agentic_monitoring.py `
  --scenario performance_degradation `
  --decision approve `
  --reviewer "offline-demo" `
  --use-fake-llm
```

The result is written under `reports/generated/performance_degradation/fake/` and is
explicitly labelled `execution_mode=fake` and `is_fake_llm=true`.

## Troubleshooting

### Missing API key

Symptom: the CLI stops with a concise `GROQ_API_KEY is not configured` error.

Action: add the key to the repository-root untracked `.env`; do not paste it into source,
commands, screenshots, or reports. Use the existing-artifacts or fake-provider demo if a
key cannot be configured safely.

### Rate limit

Symptom: a safe `RateLimitError` category appears and the graph may use deterministic
fallback.

Action: do not repeatedly retry. Wait for the provider window to reset, run only the one
scenario needed for the demo, or use the reviewed existing live report. Non-normal
evidence packets intentionally omit irrelevant pass padding to reduce token pressure.

### Scenario files missing

Symptom: the CLI says scenario artifacts are missing.

Action: only then rebuild them:

```powershell
python scripts\generate_monitoring_scenarios.py --all --overwrite
python scripts\run_monitoring.py --all
```

### Existing reports

Running a scenario replaces that execution mode's report for the same scenario. It does
not overwrite deterministic monitoring files or the other mode: live outputs go to
`live_groq/`, fake outputs to `fake/`. SQLite demonstrations write under a separate
`live_groq_persistent/<thread-id>/` directory. Copy reviewed reports elsewhere before a
demo if you need an immutable historical snapshot.

### Windows PowerShell line continuation

The backtick must be the final character on the line—no trailing space:

```powershell
python scripts\run_agentic_monitoring.py `
  --scenario performance_degradation `
  --reviewer "demo-reviewer"
```

Alternatively, place the command on one line.

### Provider timeout

The live client uses a 30-second timeout and one provider retry. If it times out, the
workflow records a safe error and falls back rather than fabricating a live result. For a
time-limited interview, switch to the existing-artifacts demo instead of repeatedly
calling the provider.
