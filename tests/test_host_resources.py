"""Fail-closed tests for CPU affinity and load evidence."""

from __future__ import annotations

import pytest

from quartz import host_resources
from quartz.host_resources import (
    GPU_MONITOR_METHOD,
    SAMPLING_METHOD,
    HostResourceError,
    evaluate_host_resource_snapshot,
    parse_cpu_list,
    parse_nvidia_pmon,
)


CONTRACT = {
    "cpu_affinity_mode": "auto_lowest_utilization",
    "sample_seconds": 1.0,
    "sample_count": 3,
    "minimum_resident_samples": 2,
    "max_load_per_logical_cpu": 0.5,
    "max_target_sibling_utilization_percent": 50.0,
    "max_competing_process_cpu_percent": 50.0,
    "gpu_process_monitor": GPU_MONITOR_METHOD,
    "gpu_minimum_active_samples": 2,
    "max_external_gpu_sm_utilization_percent": 5.0,
    "require_guard_for_profiles": ["full"],
    "require_gpu_monitor_for_profiles": ["full"],
}


def _snapshot() -> dict:
    return {
        "schema_version": 2,
        "sampling_method": SAMPLING_METHOD,
        "sample_count": 3,
        "sample_interval_seconds": 1.0 / 3.0,
        "selected_cpu": 4,
        "thread_siblings": [4, 16],
        "affinity_before": list(range(24)),
        "affinity_after": [4],
        "logical_cpu_count": 24,
        "load_per_logical_cpu": 0.25,
        "target_sibling_utilization_percent": 12.0,
        "target_sibling_busy_sample_count": 0,
        "kernel_isolated_cpus": [],
        "residency_evidence": {"completed_sample_count": 3},
        "resident_competing_processes": [],
        "competing_processes": [],
        "affinity_overlap_inventory": [],
        "gpu_process_monitor": {
            "method": GPU_MONITOR_METHOD,
            "available": True,
            "observed_sample_count": 3,
            "external_compute_processes": [],
        },
    }


def test_parse_linux_cpu_list():
    assert parse_cpu_list("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]
    assert parse_cpu_list("") == []
    with pytest.raises(ValueError, match="invalid CPU range"):
        parse_cpu_list("3-1")


def test_full_profile_rejects_overlapping_cpu_competitor():
    snapshot = _snapshot()
    snapshot["resident_competing_processes"] = [
        {
            "pid": 99,
            "name": "other-study",
            "peak_resident_cpu_percent": 95.0,
            "threshold_exceeding_sample_count": 2,
            "affinity": [4],
        }
    ]
    with pytest.raises(HostResourceError, match="competing process") as caught:
        evaluate_host_resource_snapshot(snapshot, CONTRACT, profile_name="full")
    assert caught.value.snapshot["guard_required"] is True
    assert caught.value.snapshot["isolation_level"] == "pinned_contended"


def test_diagnostic_records_contention_without_claiming_control():
    snapshot = _snapshot()
    snapshot["target_sibling_utilization_percent"] = 80.0
    snapshot["target_sibling_busy_sample_count"] = 2
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


def test_broad_affinity_is_inventory_only_when_activity_was_elsewhere(monkeypatch):
    monkeypatch.setattr(
        host_resources,
        "_process_metadata",
        lambda pid: {
            "pid": pid,
            "name": "code",
            "cmdline": ["code"],
            "cwd": "/tmp",
            "affinity": list(range(24)),
        },
    )
    intervals = [
        [{"pid": 99, "processor": 7, "cpu_seconds": 0.8}],
        [{"pid": 99, "processor": 8, "cpu_seconds": 0.8}],
        [{"pid": 99, "processor": 9, "cpu_seconds": 0.8}],
    ]
    competitors, inventory = host_resources._summarize_process_activity(
        intervals,
        [1.0, 1.0, 1.0],
        sibling_set={4, 16},
        threshold_percent=50.0,
        minimum_resident_samples=2,
    )
    assert competitors == []
    assert len(inventory) == 1
    assert inventory[0]["blocking"] is False
    assert inventory[0]["observed_processors"] == [7, 8, 9]


def test_sustained_residence_blocks_but_one_sample_burst_does_not(monkeypatch):
    monkeypatch.setattr(
        host_resources,
        "_process_metadata",
        lambda pid: {
            "pid": pid,
            "name": "worker",
            "cmdline": ["worker"],
            "cwd": "/tmp",
            "affinity": [4, 16],
        },
    )
    sustained = [
        [{"pid": 99, "processor": 4, "cpu_seconds": 0.8}],
        [{"pid": 99, "processor": 16, "cpu_seconds": 0.7}],
        [],
    ]
    competitors, _ = host_resources._summarize_process_activity(
        sustained,
        [1.0, 1.0, 1.0],
        sibling_set={4, 16},
        threshold_percent=50.0,
        minimum_resident_samples=2,
    )
    assert competitors[0]["threshold_exceeding_sample_count"] == 2

    transient = [sustained[0], [], []]
    competitors, _ = host_resources._summarize_process_activity(
        transient,
        [1.0, 1.0, 1.0],
        sibling_set={4, 16},
        threshold_percent=50.0,
        minimum_resident_samples=2,
    )
    assert competitors == []


def test_full_profile_fails_closed_when_residency_evidence_is_missing():
    snapshot = _snapshot()
    del snapshot["sampling_method"]
    with pytest.raises(HostResourceError, match="residency sampling"):
        evaluate_host_resource_snapshot(snapshot, CONTRACT, profile_name="full")


def test_gpu_graphics_rows_are_inventory_but_sustained_compute_blocks():
    graphics = "\n".join(
        [
            "# gpu pid type sm mem enc dec command",
            "0 101 G 80 10 - - chrome",
            "0 101 G 75 9 - - chrome",
            "0 101 G 70 8 - - chrome",
        ]
    )
    parsed = parse_nvidia_pmon(
        graphics,
        device_index=0,
        requested_samples=3,
        threshold_percent=5.0,
        minimum_active_samples=2,
        excluded_pids=set(),
    )
    assert len(parsed["process_inventory"]) == 1
    assert parsed["external_compute_processes"] == []

    compute = graphics.replace(" G ", " C ").replace("chrome", "python")
    parsed = parse_nvidia_pmon(
        compute,
        device_index=0,
        requested_samples=3,
        threshold_percent=5.0,
        minimum_active_samples=2,
        excluded_pids=set(),
    )
    assert parsed["external_compute_processes"][0]["pid"] == 101


def test_thread_stat_parser_handles_spaces_in_command_name():
    fields = ["0"] * 45
    fields[0] = "R"
    fields[11] = "7"
    fields[12] = "5"
    fields[19] = "1234"
    fields[36] = "16"
    assert host_resources._parse_thread_stat(
        "99 (worker with spaces) " + " ".join(fields)
    ) == (12, 16, 1234)
