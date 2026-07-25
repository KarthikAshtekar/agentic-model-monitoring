"""Deterministic evidence identifiers, uniqueness, and severity helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable

from monitoring_agent.monitoring.schemas import EvidenceItem, EvidenceSeverity

SEVERITY_RANK: dict[EvidenceSeverity, int] = {
    "info": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


def sanitise_identifier(value: str) -> str:
    """Convert a feature or metric name into a stable evidence-ID segment."""
    normalised = re.sub(r"[^A-Za-z0-9]+", "-", value.strip()).strip("-")
    return normalised.upper() or "UNNAMED"


def evidence_id(*parts: str) -> str:
    """Build a readable deterministic evidence ID."""
    return "-".join(sanitise_identifier(part) for part in parts)


class EvidenceRegistry:
    """Collect evidence while rejecting duplicate IDs immediately."""

    def __init__(self) -> None:
        self._items: list[EvidenceItem] = []
        self._ids: set[str] = set()

    def add(self, item: EvidenceItem) -> None:
        """Add one item after enforcing the run-level evidence contract."""
        if item.evidence_id in self._ids:
            raise ValueError(f"Duplicate evidence ID: {item.evidence_id}")
        if item.status in {"warning", "critical"} and not item.evidence_id:
            raise ValueError("Warning and critical conclusions require an evidence ID.")
        self._ids.add(item.evidence_id)
        self._items.append(item)

    def extend(self, items: Iterable[EvidenceItem]) -> None:
        """Add multiple evidence records under the same uniqueness guard."""
        for item in items:
            self.add(item)

    @property
    def items(self) -> list[EvidenceItem]:
        """Return a defensive copy in deterministic insertion order."""
        return list(self._items)


def maximum_severity(items: Iterable[EvidenceItem]) -> EvidenceSeverity:
    """Return the highest evidence severity, defaulting to informational."""
    return max(
        (item.severity for item in items),
        key=lambda severity: SEVERITY_RANK[severity],
        default="info",
    )
