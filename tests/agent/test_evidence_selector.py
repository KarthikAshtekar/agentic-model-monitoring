"""Evidence-packet selection invariants."""

from monitoring_agent.agent.evidence_selector import build_evidence_packet
from tests.agent.helpers import load_result


def test_all_critical_and_warning_evidence_is_retained() -> None:
    result = load_result("feature_drift")
    packet = build_evidence_packet(result)
    selected_ids = {item["evidence_id"] for item in packet["evidence"]}
    required_ids = {
        item.evidence_id
        for item in result.evidence
        if item.status in {"critical", "warning"}
    }
    assert required_ids <= selected_ids


def test_evidence_cap_is_respected() -> None:
    packet = build_evidence_packet(load_result("normal_operation"))
    assert len(packet["evidence"]) <= 30


def test_evidence_ids_remain_exact_and_unique() -> None:
    result = load_result("data_quality_failure")
    packet = build_evidence_packet(result)
    ids = [item["evidence_id"] for item in packet["evidence"]]
    original_ids = {item.evidence_id for item in result.evidence}
    assert len(ids) == len(set(ids))
    assert set(ids) <= original_ids


def test_non_normal_packet_does_not_add_pass_padding() -> None:
    packet = build_evidence_packet(load_result("feature_drift"))
    assert all(item["status"] != "pass" for item in packet["evidence"])
