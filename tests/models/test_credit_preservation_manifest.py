"""The persisted credit baseline manifest protects authoritative outputs."""

from __future__ import annotations

import hashlib
import json

from monitoring_agent.paths import PROJECT_ROOT


def test_credit_preservation_manifest_matches_every_authoritative_file() -> None:
    path = PROJECT_ROOT / "reports/refactor/credit_preservation_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["model_id"] == "credit_default"
    assert len(payload["files"]) == 22
    for entry in payload["files"]:
        artifact = PROJECT_ROOT / entry["relative_path"]
        assert entry["must_remain_byte_identical"] is True
        assert artifact.stat().st_size == entry["file_size"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == entry["sha256"]
