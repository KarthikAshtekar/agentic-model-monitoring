# ADR 002: Adapter-based binary-classification onboarding

- Status: accepted
- Date: 2026-07-25

## Context

The original runtime was tied to one credit-default bundle. A second fitted binary model
needed to reuse monitoring, evidence, orchestration, verification, approval, and
evaluation without weakening the original credit contract or inventing business facts.

## Decision

Use a strict model registry plus one reusable binary-classification adapter. Put
model-specific artifact, feature, threshold, reference, monitoring, governance, and
provenance values in strict manifests. Put business language and action boundaries in
domain policy packs. Namespace new scenarios and reports by `model_id`, while preserving
legacy credit paths and authoritative files byte-for-byte.

Onboarding is a separate read-only inspection and deterministic export workflow. The
runtime never guesses a target, threshold, positive-class meaning, business action,
approval rule, or protected attribute from filenames.

## Consequences

- Both registered models use the same deterministic monitoring and one-orchestrator graph.
- Unknown/disabled models and malformed manifests fail explicitly.
- Credit compatibility remains available through a wrapper and legacy-path fallback.
- New model families or tasks require a new justified adapter; they are not forced into
  this binary contract.
- Domain policy remains separate from estimator code.
- The registry is local YAML, not a deployment or model-management service.
