"""Fail-closed tests for CPU affinity and load evidence."""

from __future__ import annotations

import pytest

from quartz.host_resources import (
    HostResourceError,
    evaluate_host_resource_snapshot,
    parse_cpu_list,
)


CONTRACT = {
    "cpu_affinity_mode": "auto_lowest_utilization",
    "sample_seconds": 1.0,
    "max_load_per_logical_cpu": 0.5,
    "max_target_sibling_utilization_percent": 50.0,
    "max_competing_process_cpu_percent": 50.0,
    "require_guard_for_profiles": ["full"],
}


def _snapshot() -> dict:
    return {
        "schema_version": 1,
        "selected_cpu": 4,
        "thread_siblings": [4, 16],
        "affinity_before": list(range(24)),
        "affinity_after": [4],
        "logical_cpu_count": 24,
        "load_per_logical_cpu": 0.25,
        "target_sibling_utilization_percent": 12.0,
        "kernel_isolated_cpus": [],
        "competing_processes": [],
    }


def test_parse_linux_cpu_list():
    assert parse_cpu_list("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]
    assert parse_cpu_list("") == []
    with pytest.raises(ValueError, match="invalid CPU range"):
        parse_cpu_list("3-1")


def test_full_profile_rejects_overlapping_cpu_competitor():
    snapshot = _snapshot()
    snapshot["competing_processes"] = [
        {"pid": 99, "name": "other-study", "cpu_percent": 95.0, "affinity": [4]}
    ]
    with pytest.raises(HostResourceError, match="competing process") as caught:
        evaluate_host_resource_snapshot(snapshot, CONTRACT, profile_name="full")
    assert caught.value.snapshot["guard_required"] is True
    assert caught.value.snapshot["isolation_level"] == "pinned_contended"


def test_diagnostic_records_contention_without_claiming_control():
    snapshot = _snapshot()
    snapshot["target_sibling_utilization_percent"] = 80.0
    result = evaluate_host_resource_snapshot(
        snapshot, CONTRACT, profile_name="diagnostic"
    )
    assert result["guard_required"] is False
    assert result["guard_passed"] is False
    assert result["isolation_level"] == "pinned_contended"


def test_quiet_affinity_is_not_called_kernel_isolated():
    result = evaluate_host_resource_snapshot(_snapshot(), CONTRACT, profile_name="full")
    assert result["guard_passed"] is True
    assert result["isolation_level"] == "pinned_quiescent"


def test_kernel_isolation_requires_kernel_evidence_and_clean_guard():
    snapshot = _snapshot()
    snapshot["kernel_isolated_cpus"] = [4, 16]
    result = evaluate_host_resource_snapshot(snapshot, CONTRACT, profile_name="full")
    assert result["isolation_level"] == "kernel_isolated"


def test_partial_smt_isolation_is_only_pinning():
    snapshot = _snapshot()
    snapshot["kernel_isolated_cpus"] = [4]
    result = evaluate_host_resource_snapshot(snapshot, CONTRACT, profile_name="full")
    assert result["isolation_level"] == "pinned_quiescent"
