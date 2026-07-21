"""Linux CPU-affinity and host-load guards for controlled experiments.

The guard records what was actually observed.  Pinning one logical CPU is not
described as exclusive isolation unless the kernel has explicitly isolated it;
SMT siblings and competing processes are evaluated separately.
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Any, Mapping

import psutil


class HostResourceError(RuntimeError):
    """Raised when a required host-resource contract is not satisfied."""

    def __init__(self, message: str, snapshot: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.snapshot = dict(snapshot)


def parse_cpu_list(value: str) -> list[int]:
    """Parse Linux CPU-list syntax such as ``0-3,8,10-11``."""
    cpus: set[int] = set()
    for item in value.strip().split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"invalid CPU range: {item!r}")
            cpus.update(range(start, end + 1))
        else:
            cpu = int(item)
            if cpu < 0:
                raise ValueError(f"invalid CPU id: {item!r}")
            cpus.add(cpu)
    return sorted(cpus)


def _read_cpu_list(path: Path) -> list[int]:
    try:
        return parse_cpu_list(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []


def _thread_siblings(cpu: int) -> list[int]:
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    siblings = _read_cpu_list(path)
    return siblings or [cpu]


def _validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if contract.get("cpu_affinity_mode") != "auto_lowest_utilization":
        raise ValueError(
            "host_resource_contract.cpu_affinity_mode must be auto_lowest_utilization"
        )
    sample_seconds = float(contract.get("sample_seconds", 0.0))
    if not 0.0 < sample_seconds <= 30.0:
        raise ValueError("host_resource_contract.sample_seconds must be in (0, 30]")
    thresholds: dict[str, float] = {}
    for key in (
        "max_load_per_logical_cpu",
        "max_target_sibling_utilization_percent",
        "max_competing_process_cpu_percent",
    ):
        value = float(contract.get(key, -1.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"host_resource_contract.{key} must be finite and non-negative"
            )
        thresholds[key] = value
    required_profiles = contract.get("require_guard_for_profiles")
    if not isinstance(required_profiles, list) or not all(
        isinstance(item, str) and item for item in required_profiles
    ):
        raise ValueError(
            "host_resource_contract.require_guard_for_profiles must be a string list"
        )
    return {
        "cpu_affinity_mode": "auto_lowest_utilization",
        "sample_seconds": sample_seconds,
        **thresholds,
        "require_guard_for_profiles": list(required_profiles),
    }


def evaluate_host_resource_snapshot(
    snapshot: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    profile_name: str,
) -> dict[str, Any]:
    """Evaluate an observed snapshot without inflating pinning into isolation."""
    validated = _validate_contract(contract)
    result = dict(snapshot)
    failures: list[str] = []
    if result.get("affinity_after") != [result.get("selected_cpu")]:
        failures.append("process affinity was not pinned to exactly one logical CPU")
    load_per_cpu = float(result.get("load_per_logical_cpu", float("inf")))
    if load_per_cpu > validated["max_load_per_logical_cpu"]:
        failures.append(
            "host load per logical CPU exceeds "
            f"{validated['max_load_per_logical_cpu']:.3f}"
        )
    target_utilization = float(
        result.get("target_sibling_utilization_percent", float("inf"))
    )
    if target_utilization > validated["max_target_sibling_utilization_percent"]:
        failures.append(
            "selected CPU/SMT-sibling utilization exceeds "
            f"{validated['max_target_sibling_utilization_percent']:.1f}%"
        )
    competitors = result.get("competing_processes")
    if not isinstance(competitors, list):
        failures.append("competing process inventory is unavailable")
    elif competitors:
        failures.append(
            f"{len(competitors)} competing process(es) overlap the selected CPU siblings"
        )

    selected_cpu = int(result["selected_cpu"])
    kernel_isolated = set(result.get("kernel_isolated_cpus", []))
    sibling_set = set(result.get("thread_siblings", [selected_cpu]))
    result.update(
        {
            "guard_required": profile_name in validated["require_guard_for_profiles"],
            "guard_passed": not failures,
            "guard_failures": failures,
            "isolation_level": (
                "kernel_isolated"
                if sibling_set.issubset(kernel_isolated) and not failures
                else "pinned_quiescent"
                if not failures
                else "pinned_contended"
            ),
            "thresholds": {
                key: validated[key]
                for key in (
                    "max_load_per_logical_cpu",
                    "max_target_sibling_utilization_percent",
                    "max_competing_process_cpu_percent",
                )
            },
        }
    )
    if result["guard_required"] and failures:
        raise HostResourceError(
            "required host resource guard failed: " + "; ".join(failures), result
        )
    return result


def prepare_host_resources(
    contract: Mapping[str, Any], *, profile_name: str
) -> dict[str, Any]:
    """Sample load, choose a quiet allowed CPU, pin affinity, and enforce guard."""
    validated = _validate_contract(contract)
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise HostResourceError(
            "Linux sched affinity APIs are required",
            {"schema_version": 1, "platform_supported": False},
        )

    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        raise HostResourceError(
            "the process has no allowed CPUs",
            {"schema_version": 1, "platform_supported": True},
        )

    current = psutil.Process()
    excluded_pids = {current.pid, *(parent.pid for parent in current.parents())}
    sampled_processes: list[psutil.Process] = []
    for process in psutil.process_iter():
        if process.pid in excluded_pids:
            continue
        try:
            process.cpu_percent(None)
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        sampled_processes.append(process)

    per_cpu = psutil.cpu_percent(interval=validated["sample_seconds"], percpu=True)
    if len(per_cpu) <= max(allowed):
        raise HostResourceError(
            "per-CPU utilization sampling did not cover the affinity mask",
            {
                "schema_version": 1,
                "affinity_before": allowed,
                "sampled_cpu_count": len(per_cpu),
            },
        )

    sibling_map = {cpu: _thread_siblings(cpu) for cpu in allowed}
    selected_cpu = min(
        allowed,
        key=lambda cpu: (
            max(float(per_cpu[item]) for item in sibling_map[cpu]),
            cpu,
        ),
    )
    siblings = sibling_map[selected_cpu]
    os.sched_setaffinity(0, {selected_cpu})
    affinity_after = sorted(os.sched_getaffinity(0))

    competitor_threshold = validated["max_competing_process_cpu_percent"]
    competitors: list[dict[str, Any]] = []
    sibling_set = set(siblings)
    for process in sampled_processes:
        try:
            cpu_percent = float(process.cpu_percent(None))
            affinity = sorted(process.cpu_affinity())
            if cpu_percent < competitor_threshold or not sibling_set.intersection(
                affinity
            ):
                continue
            competitors.append(
                {
                    "pid": process.pid,
                    "name": process.name(),
                    "cpu_percent": cpu_percent,
                    "affinity": affinity,
                }
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    competitors.sort(key=lambda row: (-row["cpu_percent"], row["pid"]))

    logical_cpu_count = psutil.cpu_count(logical=True) or len(per_cpu)
    load1, load5, load15 = os.getloadavg()
    snapshot = {
        "schema_version": 1,
        "platform_supported": True,
        "cpu_affinity_mode": validated["cpu_affinity_mode"],
        "sample_seconds": validated["sample_seconds"],
        "logical_cpu_count": logical_cpu_count,
        "load_average": {"1m": load1, "5m": load5, "15m": load15},
        "load_per_logical_cpu": load1 / logical_cpu_count,
        "affinity_before": allowed,
        "affinity_after": affinity_after,
        "selected_cpu": selected_cpu,
        "thread_siblings": siblings,
        "selected_cpu_utilization_percent": float(per_cpu[selected_cpu]),
        "target_sibling_utilization_percent": max(
            float(per_cpu[item]) for item in siblings
        ),
        "target_sibling_utilization": {
            str(item): float(per_cpu[item]) for item in siblings
        },
        "kernel_isolated_cpus": _read_cpu_list(
            Path("/sys/devices/system/cpu/isolated")
        ),
        "competing_processes": competitors,
    }
    return evaluate_host_resource_snapshot(
        snapshot, validated, profile_name=profile_name
    )
