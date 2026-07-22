"""Linux CPU/GPU contention guards for controlled experiments.

The guard records measured activity rather than treating a broad scheduler
affinity mask as proof of contention.  CPU selection uses repeated per-CPU
samples.  Process attribution combines per-thread CPU-time deltas with the
Linux ``/proc`` last-processor field, while ``nvidia-smi pmon`` distinguishes
active CUDA compute contexts from display-only graphics contexts.

Pinning one logical CPU is not described as exclusive isolation unless the
kernel has explicitly isolated it.  Aggregate SMT utilization remains a
fail-closed backstop when process-level attribution is incomplete.
"""

from __future__ import annotations

import math
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import psutil


SAMPLING_METHOD = "psutil_percpu_linux_proc_thread_residency_v2"
GPU_MONITOR_METHOD = "nvidia_smi_pmon"


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


def _contract_int(
    contract: Mapping[str, Any], key: str, *, minimum: int, maximum: int
) -> int:
    value = contract.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"host_resource_contract.{key} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(
            f"host_resource_contract.{key} must be in [{minimum}, {maximum}]"
        )
    return value


def _string_list(contract: Mapping[str, Any], key: str) -> list[str]:
    value = contract.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"host_resource_contract.{key} must be a string list")
    return list(value)


def validate_host_resource_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the versioned A15 host-resource contract."""
    if contract.get("cpu_affinity_mode") != "auto_lowest_utilization":
        raise ValueError(
            "host_resource_contract.cpu_affinity_mode must be auto_lowest_utilization"
        )
    sample_seconds = float(contract.get("sample_seconds", 0.0))
    if not 0.0 < sample_seconds <= 30.0:
        raise ValueError("host_resource_contract.sample_seconds must be in (0, 30]")
    sample_count = _contract_int(contract, "sample_count", minimum=3, maximum=30)
    minimum_resident_samples = _contract_int(
        contract, "minimum_resident_samples", minimum=1, maximum=sample_count
    )
    gpu_minimum_active_samples = _contract_int(
        contract, "gpu_minimum_active_samples", minimum=1, maximum=sample_count
    )
    thresholds: dict[str, float] = {}
    for key in (
        "max_load_per_logical_cpu",
        "max_target_sibling_utilization_percent",
        "max_competing_process_cpu_percent",
        "max_external_gpu_sm_utilization_percent",
    ):
        value = float(contract.get(key, -1.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"host_resource_contract.{key} must be finite and non-negative"
            )
        thresholds[key] = value
    if contract.get("gpu_process_monitor") != GPU_MONITOR_METHOD:
        raise ValueError(
            "host_resource_contract.gpu_process_monitor must be nvidia_smi_pmon"
        )
    return {
        "cpu_affinity_mode": "auto_lowest_utilization",
        "sample_seconds": sample_seconds,
        "sample_count": sample_count,
        "minimum_resident_samples": minimum_resident_samples,
        "gpu_minimum_active_samples": gpu_minimum_active_samples,
        **thresholds,
        "gpu_process_monitor": GPU_MONITOR_METHOD,
        "require_guard_for_profiles": _string_list(
            contract, "require_guard_for_profiles"
        ),
        "require_gpu_monitor_for_profiles": _string_list(
            contract, "require_gpu_monitor_for_profiles"
        ),
    }


def _parse_thread_stat(value: str) -> tuple[int, int, int]:
    """Return ``(cpu_ticks, last_processor, start_ticks)`` from Linux stat."""
    close = value.rfind(")")
    if close < 0:
        raise ValueError("thread stat has no closing command delimiter")
    fields = value[close + 1 :].split()
    # fields[0] is kernel field 3 (state). utime/stime/starttime/processor are
    # kernel fields 14/15/22/39 respectively.
    if len(fields) <= 36:
        raise ValueError("thread stat is truncated")
    cpu_ticks = int(fields[11]) + int(fields[12])
    start_ticks = int(fields[19])
    processor = int(fields[36])
    return cpu_ticks, processor, start_ticks


def _read_thread_snapshot(
    *, excluded_pids: set[int], proc_root: Path = Path("/proc")
) -> tuple[dict[tuple[int, int, int], tuple[int, int]], dict[str, int]]:
    """Read CPU time and last-processor evidence for every visible thread."""
    rows: dict[tuple[int, int, int], tuple[int, int]] = {}
    processes_seen = 0
    threads_seen = 0
    unreadable_threads = 0
    try:
        pid_paths = list(proc_root.iterdir())
    except OSError:
        return rows, {
            "processes_seen": 0,
            "threads_seen": 0,
            "unreadable_threads": 1,
        }
    for pid_path in pid_paths:
        if not pid_path.name.isdigit():
            continue
        pid = int(pid_path.name)
        if pid in excluded_pids:
            continue
        processes_seen += 1
        try:
            task_paths = list((pid_path / "task").iterdir())
        except OSError:
            unreadable_threads += 1
            continue
        for task_path in task_paths:
            if not task_path.name.isdigit():
                continue
            tid = int(task_path.name)
            try:
                cpu_ticks, processor, start_ticks = _parse_thread_stat(
                    (task_path / "stat").read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, ValueError):
                unreadable_threads += 1
                continue
            rows[(pid, tid, start_ticks)] = (cpu_ticks, processor)
            threads_seen += 1
    return rows, {
        "processes_seen": processes_seen,
        "threads_seen": threads_seen,
        "unreadable_threads": unreadable_threads,
    }


def _thread_activity_delta(
    before: Mapping[tuple[int, int, int], tuple[int, int]],
    after: Mapping[tuple[int, int, int], tuple[int, int]],
    *,
    elapsed_seconds: float,
    clock_ticks: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if elapsed_seconds <= 0.0 or clock_ticks <= 0:
        return rows
    for identity, (after_ticks, processor) in after.items():
        prior = before.get(identity)
        if prior is None:
            continue
        delta_ticks = after_ticks - prior[0]
        if delta_ticks <= 0:
            continue
        pid, tid, _ = identity
        cpu_seconds = delta_ticks / clock_ticks
        rows.append(
            {
                "pid": pid,
                "tid": tid,
                "processor": processor,
                "cpu_seconds": cpu_seconds,
                "cpu_percent": 100.0 * cpu_seconds / elapsed_seconds,
            }
        )
    return rows


def _process_metadata(pid: int) -> dict[str, Any]:
    row: dict[str, Any] = {"pid": pid}
    try:
        process = psutil.Process(pid)
        with process.oneshot():
            row["name"] = process.name()
            row["cmdline"] = process.cmdline()
            try:
                row["cwd"] = process.cwd()
            except (psutil.AccessDenied, FileNotFoundError):
                row["cwd"] = None
            try:
                row["affinity"] = sorted(process.cpu_affinity())
            except (AttributeError, psutil.AccessDenied):
                row["affinity"] = None
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        row.update(name="<exited-or-inaccessible>", cmdline=[], cwd=None, affinity=None)
    return row


def _summarize_process_activity(
    interval_rows: Sequence[Sequence[Mapping[str, Any]]],
    interval_seconds: Sequence[float],
    *,
    sibling_set: set[int],
    threshold_percent: float,
    minimum_resident_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_pid_intervals: dict[int, list[dict[str, float | set[int]]]] = defaultdict(
        lambda: [
            {"total_seconds": 0.0, "resident_seconds": 0.0, "processors": set()}
            for _ in interval_rows
        ]
    )
    for index, rows in enumerate(interval_rows):
        for row in rows:
            pid = int(row["pid"])
            processor = int(row["processor"])
            cpu_seconds = float(row["cpu_seconds"])
            bucket = per_pid_intervals[pid][index]
            bucket["total_seconds"] = float(bucket["total_seconds"]) + cpu_seconds
            processors = bucket["processors"]
            assert isinstance(processors, set)
            processors.add(processor)
            if processor in sibling_set:
                bucket["resident_seconds"] = (
                    float(bucket["resident_seconds"]) + cpu_seconds
                )

    resident_competitors: list[dict[str, Any]] = []
    affinity_inventory: list[dict[str, Any]] = []
    total_elapsed = sum(interval_seconds)
    for pid, buckets in per_pid_intervals.items():
        metadata = _process_metadata(pid)
        interval_resident_percent = [
            100.0 * float(bucket["resident_seconds"]) / elapsed
            if elapsed > 0.0
            else 0.0
            for bucket, elapsed in zip(buckets, interval_seconds, strict=True)
        ]
        total_seconds = sum(float(bucket["total_seconds"]) for bucket in buckets)
        resident_seconds = sum(float(bucket["resident_seconds"]) for bucket in buckets)
        observed_processors = sorted(
            {int(processor) for bucket in buckets for processor in bucket["processors"]}
        )
        resident_samples = sum(value > 0.0 for value in interval_resident_percent)
        threshold_samples = sum(
            value >= threshold_percent for value in interval_resident_percent
        )
        activity = {
            **metadata,
            "observed_processors": observed_processors,
            "total_cpu_seconds": total_seconds,
            "total_cpu_percent": (
                100.0 * total_seconds / total_elapsed if total_elapsed > 0.0 else 0.0
            ),
            "resident_cpu_seconds": resident_seconds,
            "mean_resident_cpu_percent": (
                100.0 * resident_seconds / total_elapsed if total_elapsed > 0.0 else 0.0
            ),
            "peak_resident_cpu_percent": max(interval_resident_percent, default=0.0),
            "resident_sample_count": resident_samples,
            "threshold_exceeding_sample_count": threshold_samples,
            "sample_count": len(interval_rows),
        }
        if threshold_samples >= minimum_resident_samples:
            resident_competitors.append(activity)

        affinity = metadata.get("affinity")
        overlaps = isinstance(affinity, list) and bool(
            sibling_set.intersection(affinity)
        )
        if overlaps and activity["total_cpu_percent"] >= threshold_percent:
            affinity_inventory.append(
                {
                    **activity,
                    "blocking": False,
                    "classification": "broad_affinity_inventory_only",
                }
            )

    ordering = lambda row: (  # noqa: E731
        -int(row["threshold_exceeding_sample_count"]),
        -float(row["peak_resident_cpu_percent"]),
        int(row["pid"]),
    )
    resident_competitors.sort(key=ordering)
    affinity_inventory.sort(key=ordering)
    return resident_competitors, affinity_inventory


def parse_nvidia_pmon(
    output: str,
    *,
    device_index: int,
    requested_samples: int,
    threshold_percent: float,
    minimum_active_samples: int,
    excluded_pids: set[int],
) -> dict[str, Any]:
    """Parse ``nvidia-smi pmon`` without treating graphics rows as compute."""
    by_pid: dict[int, dict[str, Any]] = {}
    target_rows = 0
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 5:
            continue
        try:
            gpu_index = int(fields[0])
        except ValueError:
            continue
        if gpu_index != device_index:
            continue
        target_rows += 1
        if fields[1] == "-":
            continue
        try:
            pid = int(fields[1])
        except ValueError:
            continue
        process_type = fields[2]
        sm_text = fields[3]
        memory_text = fields[4]
        sm_percent = None if sm_text == "-" else float(sm_text)
        memory_percent = None if memory_text == "-" else float(memory_text)
        row = by_pid.setdefault(
            pid,
            {
                "pid": pid,
                "type": process_type,
                "command": fields[-1],
                "sm_samples_percent": [],
                "memory_samples_percent": [],
            },
        )
        if sm_percent is not None:
            row["sm_samples_percent"].append(sm_percent)
        if memory_percent is not None:
            row["memory_samples_percent"].append(memory_percent)

    inventory: list[dict[str, Any]] = []
    external_compute: list[dict[str, Any]] = []
    for pid, row in by_pid.items():
        sm_samples = list(row.pop("sm_samples_percent"))
        memory_samples = list(row.pop("memory_samples_percent"))
        active_samples = sum(value >= threshold_percent for value in sm_samples)
        summary = {
            **row,
            "sm_observation_count": len(sm_samples),
            "active_sample_count": active_samples,
            "max_sm_percent": max(sm_samples, default=0.0),
            "mean_sm_percent": (
                sum(sm_samples) / len(sm_samples) if sm_samples else 0.0
            ),
            "max_memory_percent": max(memory_samples, default=0.0),
            "excluded_as_experiment_family": pid in excluded_pids,
        }
        inventory.append(summary)
        is_compute = "C" in str(summary["type"])
        if (
            is_compute
            and pid not in excluded_pids
            and active_samples >= minimum_active_samples
        ):
            external_compute.append(summary)
    inventory.sort(key=lambda row: (-float(row["max_sm_percent"]), int(row["pid"])))
    external_compute.sort(
        key=lambda row: (-float(row["max_sm_percent"]), int(row["pid"]))
    )
    return {
        "method": GPU_MONITOR_METHOD,
        "device_index": device_index,
        "requested_sample_count": requested_samples,
        "observed_sample_count": requested_samples if target_rows else 0,
        "process_inventory": inventory,
        "external_compute_processes": external_compute,
    }


def _gpu_context_inventory(device_index: int) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "-i",
        str(device_index),
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc), "contexts": []}
    if process.returncode != 0:
        return {
            "available": False,
            "error": process.stderr.strip() or f"exit {process.returncode}",
            "contexts": [],
        }
    contexts: list[dict[str, Any]] = []
    for line in process.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 3 or not fields[0].isdigit():
            continue
        memory_text = fields[2]
        contexts.append(
            {
                "pid": int(fields[0]),
                "process_name": fields[1],
                "used_gpu_memory_mib": (
                    int(memory_text) if memory_text.isdigit() else None
                ),
            }
        )
    return {"available": True, "error": None, "contexts": contexts}


def _collect_gpu_monitor(
    process: subprocess.Popen[str] | None,
    *,
    startup_error: str | None,
    device_index: int | None,
    requested_samples: int,
    threshold_percent: float,
    minimum_active_samples: int,
    excluded_pids: set[int],
    timeout_seconds: float,
) -> dict[str, Any]:
    if process is None or device_index is None:
        return {
            "method": GPU_MONITOR_METHOD,
            "available": False,
            "error": startup_error or "CUDA device index was not supplied",
            "device_index": device_index,
            "requested_sample_count": requested_samples,
            "observed_sample_count": 0,
            "process_inventory": [],
            "external_compute_processes": [],
            "compute_context_inventory": {"available": False, "contexts": []},
        }
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        return {
            "method": GPU_MONITOR_METHOD,
            "available": False,
            "error": "nvidia-smi pmon timed out",
            "stderr": stderr.strip(),
            "device_index": device_index,
            "requested_sample_count": requested_samples,
            "observed_sample_count": 0,
            "process_inventory": [],
            "external_compute_processes": [],
            "compute_context_inventory": _gpu_context_inventory(device_index),
        }
    parsed = parse_nvidia_pmon(
        stdout,
        device_index=device_index,
        requested_samples=requested_samples,
        threshold_percent=threshold_percent,
        minimum_active_samples=minimum_active_samples,
        excluded_pids=excluded_pids,
    )
    available = process.returncode == 0 and (
        parsed["observed_sample_count"] >= requested_samples
    )
    return {
        **parsed,
        "available": available,
        "returncode": process.returncode,
        "stderr": stderr.strip(),
        "error": None
        if available
        else stderr.strip() or "nvidia-smi pmon returned incomplete samples",
        "compute_context_inventory": _gpu_context_inventory(device_index),
    }


def evaluate_host_resource_snapshot(
    snapshot: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    profile_name: str,
) -> dict[str, Any]:
    """Evaluate observed CPU residence and GPU compute activity fail-closed."""
    validated = validate_host_resource_contract(contract)
    result = dict(snapshot)
    failures: list[str] = []
    selected_cpu_raw = result.get("selected_cpu")
    if isinstance(selected_cpu_raw, bool) or not isinstance(selected_cpu_raw, int):
        failures.append("selected CPU evidence is unavailable")
        selected_cpu = -1
    else:
        selected_cpu = selected_cpu_raw
    if result.get("affinity_after") != [selected_cpu]:
        failures.append("process affinity was not pinned to exactly one logical CPU")
    load_per_cpu = float(result.get("load_per_logical_cpu", float("inf")))
    if load_per_cpu > validated["max_load_per_logical_cpu"]:
        failures.append(
            "host load per logical CPU exceeds "
            f"{validated['max_load_per_logical_cpu']:.3f}"
        )

    sampling_ok = (
        result.get("sampling_method") == SAMPLING_METHOD
        and result.get("sample_count") == validated["sample_count"]
        and isinstance(result.get("residency_evidence"), dict)
        and result["residency_evidence"].get("completed_sample_count")
        == validated["sample_count"]
    )
    if not sampling_ok:
        failures.append("thread-residency sampling evidence is incomplete")
    target_utilization = float(
        result.get("target_sibling_utilization_percent", float("inf"))
    )
    busy_samples = result.get("target_sibling_busy_sample_count")
    if not isinstance(busy_samples, int):
        failures.append("SMT-sibling sustained-utilization evidence is unavailable")
    elif (
        busy_samples >= validated["minimum_resident_samples"]
        or target_utilization > validated["max_target_sibling_utilization_percent"]
    ):
        failures.append(
            "selected CPU/SMT-sibling sustained utilization exceeds "
            f"{validated['max_target_sibling_utilization_percent']:.1f}%"
        )

    resident_competitors = result.get("resident_competing_processes")
    if not isinstance(resident_competitors, list):
        failures.append("resident competing-process evidence is unavailable")
    elif resident_competitors:
        failures.append(
            f"{len(resident_competitors)} competing process(es) were measured "
            "on the selected CPU siblings"
        )
    if not isinstance(result.get("affinity_overlap_inventory"), list):
        failures.append("non-blocking affinity inventory is unavailable")

    gpu_required = profile_name in validated["require_gpu_monitor_for_profiles"]
    gpu_monitor = result.get("gpu_process_monitor")
    if not isinstance(gpu_monitor, dict):
        if gpu_required:
            failures.append("GPU process monitor evidence is unavailable")
    else:
        if gpu_required and (
            gpu_monitor.get("method") != GPU_MONITOR_METHOD
            or gpu_monitor.get("available") is not True
            or gpu_monitor.get("observed_sample_count") != validated["sample_count"]
        ):
            failures.append("required GPU process monitor did not complete")
        gpu_competitors = gpu_monitor.get("external_compute_processes")
        if not isinstance(gpu_competitors, list):
            if gpu_required:
                failures.append("external GPU compute-process evidence is unavailable")
        elif gpu_competitors:
            failures.append(
                f"{len(gpu_competitors)} external process(es) showed sustained GPU SM activity"
            )

    sibling_set = set(result.get("thread_siblings", [selected_cpu]))
    kernel_isolated = set(result.get("kernel_isolated_cpus", []))
    guard_required = profile_name in {
        *validated["require_guard_for_profiles"],
        *validated["require_gpu_monitor_for_profiles"],
    }
    result.update(
        {
            "guard_required": guard_required,
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
                    "max_external_gpu_sm_utilization_percent",
                    "minimum_resident_samples",
                    "gpu_minimum_active_samples",
                )
            },
        }
    )
    if guard_required and failures:
        raise HostResourceError(
            "required host resource guard failed: " + "; ".join(failures), result
        )
    return result


def prepare_host_resources(
    contract: Mapping[str, Any],
    *,
    profile_name: str,
    cuda_device_index: int | None = None,
) -> dict[str, Any]:
    """Measure sustained activity, choose a quiet SMT pair, and pin the run."""
    validated = validate_host_resource_contract(contract)
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise HostResourceError(
            "Linux sched affinity APIs are required",
            {"schema_version": 2, "platform_supported": False},
        )

    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        raise HostResourceError(
            "the process has no allowed CPUs",
            {"schema_version": 2, "platform_supported": True},
        )

    current = psutil.Process()
    excluded_pids = {current.pid, *(parent.pid for parent in current.parents())}
    gpu_process: subprocess.Popen[str] | None = None
    gpu_startup_error: str | None = None
    if cuda_device_index is not None:
        command = [
            "nvidia-smi",
            "pmon",
            "-i",
            str(cuda_device_index),
            "-s",
            "um",
            "-c",
            str(validated["sample_count"]),
        ]
        try:
            gpu_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            excluded_pids.add(gpu_process.pid)
        except OSError as exc:
            gpu_startup_error = str(exc)

    psutil.cpu_percent(interval=None, percpu=True)
    previous_threads, initial_coverage = _read_thread_snapshot(
        excluded_pids=excluded_pids
    )
    previous_time = time.monotonic()
    per_cpu_samples: list[list[float]] = []
    activity_intervals: list[list[dict[str, Any]]] = []
    activity_seconds: list[float] = []
    coverage_rows = [initial_coverage]
    interval_seconds = validated["sample_seconds"] / validated["sample_count"]
    clock_ticks = int(os.sysconf("SC_CLK_TCK"))
    for _ in range(validated["sample_count"]):
        per_cpu = [
            float(value)
            for value in psutil.cpu_percent(interval=interval_seconds, percpu=True)
        ]
        current_time = time.monotonic()
        current_threads, coverage = _read_thread_snapshot(excluded_pids=excluded_pids)
        elapsed = current_time - previous_time
        per_cpu_samples.append(per_cpu)
        activity_seconds.append(elapsed)
        activity_intervals.append(
            _thread_activity_delta(
                previous_threads,
                current_threads,
                elapsed_seconds=elapsed,
                clock_ticks=clock_ticks,
            )
        )
        coverage_rows.append(coverage)
        previous_threads = current_threads
        previous_time = current_time

    if any(len(sample) <= max(allowed) for sample in per_cpu_samples):
        raise HostResourceError(
            "per-CPU utilization sampling did not cover the affinity mask",
            {
                "schema_version": 2,
                "affinity_before": allowed,
                "sampled_cpu_counts": [len(sample) for sample in per_cpu_samples],
            },
        )

    sibling_map = {cpu: _thread_siblings(cpu) for cpu in allowed}
    pair_samples = {
        cpu: [
            max(float(sample[item]) for item in sibling_map[cpu])
            for sample in per_cpu_samples
        ]
        for cpu in allowed
    }
    selected_cpu = min(
        allowed,
        key=lambda cpu: (
            sum(pair_samples[cpu]) / len(pair_samples[cpu]),
            max(pair_samples[cpu]),
            cpu,
        ),
    )
    siblings = sibling_map[selected_cpu]
    sibling_set = set(siblings)
    os.sched_setaffinity(0, {selected_cpu})
    affinity_after = sorted(os.sched_getaffinity(0))

    competitors, affinity_inventory = _summarize_process_activity(
        activity_intervals,
        activity_seconds,
        sibling_set=sibling_set,
        threshold_percent=validated["max_competing_process_cpu_percent"],
        minimum_resident_samples=validated["minimum_resident_samples"],
    )
    target_samples = pair_samples[selected_cpu]
    target_threshold = validated["max_target_sibling_utilization_percent"]
    gpu_monitor = _collect_gpu_monitor(
        gpu_process,
        startup_error=gpu_startup_error,
        device_index=cuda_device_index,
        requested_samples=validated["sample_count"],
        threshold_percent=validated["max_external_gpu_sm_utilization_percent"],
        minimum_active_samples=validated["gpu_minimum_active_samples"],
        excluded_pids=excluded_pids,
        timeout_seconds=max(validated["sample_seconds"] + 10.0, 15.0),
    )

    logical_cpu_count = psutil.cpu_count(logical=True) or len(per_cpu_samples[0])
    load1, load5, load15 = os.getloadavg()
    snapshot = {
        "schema_version": 2,
        "platform_supported": True,
        "cpu_affinity_mode": validated["cpu_affinity_mode"],
        "sampling_method": SAMPLING_METHOD,
        "sample_seconds": validated["sample_seconds"],
        "sample_count": validated["sample_count"],
        "sample_interval_seconds": interval_seconds,
        "logical_cpu_count": logical_cpu_count,
        "load_average": {"1m": load1, "5m": load5, "15m": load15},
        "load_per_logical_cpu": load1 / logical_cpu_count,
        "affinity_before": allowed,
        "affinity_after": affinity_after,
        "selected_cpu": selected_cpu,
        "thread_siblings": siblings,
        "selected_cpu_utilization_percent": sum(
            sample[selected_cpu] for sample in per_cpu_samples
        )
        / len(per_cpu_samples),
        "target_sibling_utilization_percent": sum(target_samples) / len(target_samples),
        "target_sibling_peak_utilization_percent": max(target_samples),
        "target_sibling_busy_sample_count": sum(
            value > target_threshold for value in target_samples
        ),
        "target_sibling_utilization_samples": target_samples,
        "target_sibling_utilization": {
            str(item): sum(sample[item] for sample in per_cpu_samples)
            / len(per_cpu_samples)
            for item in siblings
        },
        "kernel_isolated_cpus": _read_cpu_list(
            Path("/sys/devices/system/cpu/isolated")
        ),
        "residency_evidence": {
            "method": "/proc/<pid>/task/<tid>/stat cpu-time delta plus last processor",
            "completed_sample_count": len(activity_intervals),
            "clock_ticks_per_second": clock_ticks,
            "processes_seen_min": min(row["processes_seen"] for row in coverage_rows),
            "threads_seen_min": min(row["threads_seen"] for row in coverage_rows),
            "unreadable_thread_stat_total": sum(
                row["unreadable_threads"] for row in coverage_rows
            ),
            "attribution_limit": (
                "CPU-time deltas are attributed to the last processor observed at "
                "each sub-sample boundary; aggregate per-CPU utilization is the "
                "fail-closed backstop for migration or inaccessible processes."
            ),
        },
        "resident_competing_processes": competitors,
        # Compatibility alias: unlike v1 this contains only measured residents,
        # never processes selected merely because of a broad affinity mask.
        "competing_processes": competitors,
        "affinity_overlap_inventory": affinity_inventory,
        "gpu_process_monitor": gpu_monitor,
    }
    return evaluate_host_resource_snapshot(
        snapshot, validated, profile_name=profile_name
    )
