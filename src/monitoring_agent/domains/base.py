"""Strict loader for domain policy packs."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DomainPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain_id: str
    domain_name: str
    positive_outcome_wording: str
    prediction_unit_wording: str
    safe_terminology: list[str] = Field(min_length=1)
    allowed_actions: list[str] = Field(min_length=1)
    prohibited_claims: list[str]
    approval_policy: str
    required_uncertainty_language: list[str] = Field(min_length=1)
    domain_limitations: list[str] = Field(min_length=1)

    @property
    def safe_business_terminology(self) -> list[str]:
        """Compatibility name used in existing monitoring result schemas."""
        return self.safe_terminology

    @property
    def uncertainty_wording(self) -> list[str]:
        """Compatibility name used by existing prompt construction."""
        return self.required_uncertainty_language

    @property
    def domain_specific_limitations(self) -> list[str]:
        """Compatibility name used in existing monitoring result schemas."""
        return self.domain_limitations


def load_domain_policy(domain_id: str) -> DomainPolicy:
    path = Path(__file__).resolve().parent / domain_id / "policy.yaml"
    if not path.is_file():
        raise KeyError(f"Unknown domain policy pack: {domain_id!r}.")
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    policy = DomainPolicy.model_validate(payload)
    if policy.domain_id != domain_id:
        raise ValueError("Domain directory and policy domain_id do not match.")
    return policy
