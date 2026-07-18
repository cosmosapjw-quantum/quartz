"""A15 matched CPU/CUDA evaluator service-curve diagnostics.

This module reuses :func:`quartz.experiments.service_curve.build_eval_model`
but strengthens the measurement contract for a paired backend comparison:

* one deterministically initialized model state is copied to both devices;
* each (batch, inflight, repetition) pair consumes byte-identical CPU-created
  inputs;
* CPU and CUDA output parity is checked before any timing row is accepted;
* raw repetition rows are retained and summarized without an efficacy or
  production-promotion verdict.

CPU ``inflight`` work is deliberately serial while CUDA ``inflight`` work uses
one stream per outstanding batch.  The two rows therefore characterize explicit
device-specific service contracts; they do not establish energy efficiency,
play strength, or production scheduler quality.
"""

from __future__ import annotations

import hashlib
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from quartz.experiment_manifest import canonical_sha256
from quartz.experiments import service_curve as _service_curve


A15_SCHEMA_VERSION = 1
EXPERIMENT_ID = "a15_matched_service_curve_v1"
EXECUTION_MODE = "matched_cpu_cuda_service_curve_diagnostic"
ROLE = "ablation_readiness"
# The matched diagnostic validates the execution/measurement substrate only.
# A15's pre-existing axis status may be stronger, but this run must not inherit
# that status as evidence newly earned by a four-cell readiness smoke.
EVIDENCE_STATUS = "skeleton_only"
CLAIM_SCOPE = "ablation_readiness_only"
PLOT_CATEGORY = "DIAGNOSTIC"
REQUIRED_BACKENDS = ("cpu", "cuda")


def build_eval_model(
    in_ch: int, channels: int, blocks: int, board: int, actions: int
) -> Any:
    """Use the existing service-curve workload implementation unchanged."""
    return _service_curve.build_eval_model(in_ch, channels, blocks, board, actions)


def _positive_int(value: Any, *, label: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    parsed = int(value)
    lower = 0 if allow_zero else 1
    if parsed < lower:
        raise ValueError(f"{label} must be >= {lower}")
    return parsed


def _positive_int_grid(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    parsed = [_positive_int(item, label=label) for item in value]
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"{label} contains duplicates")
    if parsed != sorted(parsed):
        raise ValueError(f"{label} must be sorted")
    return parsed


def validate_config(payload: Mapping[str, Any], profile_name: str) -> dict[str, Any]:
    """Validate and return the selected immutable measurement profile."""
    if payload.get("format_version") != A15_SCHEMA_VERSION:
        raise ValueError("unsupported A15 config format_version")
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("A15 config experiment_id mismatch")

    model = payload.get("model")
    if not isinstance(model, dict):
        raise ValueError("model must be an object")
    required_model = ("label", "in_ch", "channels", "blocks", "board", "actions")
    missing_model = [key for key in required_model if key not in model]
    if missing_model:
        raise ValueError(f"model missing keys: {missing_model}")
    for key in required_model[1:]:
        _positive_int(model[key], label=f"model.{key}")

    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or profile_name not in profiles:
        raise ValueError(f"unknown A15 profile: {profile_name}")
    profile = profiles[profile_name]
    if not isinstance(profile, dict):
        raise ValueError(f"profile {profile_name} must be an object")
    batch_sizes = _positive_int_grid(profile.get("batch_sizes"), label="batch_sizes")
    inflight_grid = _positive_int_grid(
        profile.get("inflight_grid"), label="inflight_grid"
    )
    repetitions = _positive_int(profile.get("repetitions"), label="repetitions")
    n_waves = _positive_int(profile.get("n_waves"), label="n_waves")
    warmup_waves = _positive_int(
        profile.get("warmup_waves"), label="warmup_waves", allow_zero=True
    )
    if profile_name == "full" and repetitions < 5:
        raise ValueError("full profile requires at least five repetitions")

    seed_contract = payload.get("seed_contract")
    if not isinstance(seed_contract, dict):
        raise ValueError("seed_contract must be an object")
    for key in ("model_seed", "input_seed", "parity_probe_seed"):
        _positive_int(seed_contract.get(key), label=key, allow_zero=True)
    if seed_contract.get("mode") != "fixed_common_random_numbers":
        raise ValueError("seed_contract.mode must be fixed_common_random_numbers")
    if seed_contract.get("backend_order") != "alternating_by_cell_and_repetition":
        raise ValueError(
            "seed_contract.backend_order must be alternating_by_cell_and_repetition"
        )
    for key in (
        "same_model_state_across_backends",
        "same_input_bytes_within_backend_pair",
    ):
        if seed_contract.get(key) is not True:
            raise ValueError(f"seed_contract.{key} must be true")

    runtime = payload.get("runtime_contract")
    if not isinstance(runtime, dict):
        raise ValueError("runtime_contract must be an object")
    for key in ("cpu_intraop_threads", "cpu_interop_threads"):
        _positive_int(runtime.get(key), label=f"runtime_contract.{key}")
    for key in (
        "allow_tf32",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "deterministic_algorithms",
    ):
        if not isinstance(runtime.get(key), bool):
            raise ValueError(f"runtime_contract.{key} must be boolean")

    parity = payload.get("parity")
    if not isinstance(parity, dict):
        raise ValueError("parity must be an object")
    for key in ("atol", "rtol"):
        value = float(parity.get(key, -1.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"parity.{key} must be finite and non-negative")
    _positive_int(parity.get("probe_batch"), label="parity.probe_batch")
    if parity.get("require_finite") is not True:
        raise ValueError("parity.require_finite must be true")
    if float(parity.get("require_policy_top1_agreement", -1.0)) != 1.0:
        raise ValueError("parity.require_policy_top1_agreement must equal 1.0")

    return {
        "name": profile_name,
        "batch_sizes": batch_sizes,
        "inflight_grid": inflight_grid,
        "repetitions": repetitions,
        "n_waves": n_waves,
        "warmup_waves": warmup_waves,
    }


def tensor_sha256(tensor: Any) -> str:
    """Hash a tensor's shape, dtype, and contiguous CPU bytes."""
    cpu = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(int(v) for v in cpu.shape)).encode("utf-8"))
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def model_state_sha256(model: Any) -> str:
    """Canonical hash for all parameters and buffers in a model state dict."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor_sha256(tensor).encode("ascii"))
    return digest.hexdigest()


def derived_input_seed(
    base_seed: int, batch_size: int, inflight: int, repetition: int
) -> int:
    payload = {
        "schema_version": A15_SCHEMA_VERSION,
        "base_seed": int(base_seed),
        "batch_size": int(batch_size),
        "inflight": int(inflight),
        "repetition": int(repetition),
    }
    return int(canonical_sha256(payload)[:16], 16) % (2**63 - 1)


def make_matched_inputs(
    *,
    batch_size: int,
    inflight: int,
    in_ch: int,
    board: int,
    base_seed: int,
    repetition: int,
) -> tuple[list[Any], str, int]:
    """Create the canonical CPU tensors used by both backend rows."""
    import torch

    seed = derived_input_seed(base_seed, batch_size, inflight, repetition)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tensors = [
        torch.randn(
            batch_size, in_ch, board, board, generator=generator, dtype=torch.float32
        )
        for _ in range(inflight)
    ]
    digest = hashlib.sha256()
    for index, tensor in enumerate(tensors):
        digest.update(str(index).encode("ascii"))
        digest.update(tensor_sha256(tensor).encode("ascii"))
    return tensors, digest.hexdigest(), seed


def build_workload_identity(
    *,
    model_spec: Mapping[str, Any],
    model_state_hash: str,
    profile: Mapping[str, Any],
    seed_contract: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    builder_source_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Return the full backend-independent workload contract and its hash."""
    contract = {
        "schema_version": A15_SCHEMA_VERSION,
        "workload_builder": "quartz.experiments.service_curve.build_eval_model",
        "builder_source_sha256": str(builder_source_sha256),
        "model_spec": dict(model_spec),
        "model_state_sha256": str(model_state_hash),
        "profile": dict(profile),
        "seed_contract": dict(seed_contract),
        "dtype": "torch.float32",
        "input_distribution": "torch.randn_cpu_standard_normal",
        "runtime_contract": dict(runtime_contract),
        "backend_submission_semantics": {
            "cpu": "serial_batches_within_wave",
            "cuda": "one_cuda_stream_per_inflight_batch_then_wave_synchronize",
        },
    }
    return contract, canonical_sha256(contract)


def check_backend_parity(
    cpu_model: Any,
    cuda_model: Any,
    probe: Any,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Fail-closed semantic check before accepting wall-clock measurements."""
    import torch

    try:
        cuda_device = next(cuda_model.parameters()).device
    except StopIteration as exc:
        raise ValueError("CUDA parity model has no parameters") from exc
    if cuda_device.type != "cuda":
        raise ValueError(f"CUDA parity model is on {cuda_device}, expected cuda")
    with torch.no_grad():
        cpu_policy, cpu_value = cpu_model(probe)
        cuda_policy, cuda_value = cuda_model(probe.to(cuda_device))
    cuda_policy_cpu = cuda_policy.detach().cpu()
    cuda_value_cpu = cuda_value.detach().cpu()
    finite = bool(
        torch.isfinite(cpu_policy).all()
        and torch.isfinite(cpu_value).all()
        and torch.isfinite(cuda_policy_cpu).all()
        and torch.isfinite(cuda_value_cpu).all()
    )
    shape_match = bool(
        cpu_policy.shape == cuda_policy_cpu.shape
        and cpu_value.shape == cuda_value_cpu.shape
    )
    policy_close = bool(
        shape_match
        and torch.allclose(cpu_policy, cuda_policy_cpu, atol=atol, rtol=rtol)
    )
    value_close = bool(
        shape_match and torch.allclose(cpu_value, cuda_value_cpu, atol=atol, rtol=rtol)
    )
    top1_agreement = (
        float(
            (cpu_policy.argmax(dim=1) == cuda_policy_cpu.argmax(dim=1)).float().mean()
        )
        if shape_match and cpu_policy.ndim == 2 and cpu_policy.shape[0] > 0
        else 0.0
    )
    policy_max_abs = (
        float((cpu_policy - cuda_policy_cpu).abs().max()) if shape_match else None
    )
    value_max_abs = (
        float((cpu_value - cuda_value_cpu).abs().max()) if shape_match else None
    )
    passed = bool(
        finite
        and shape_match
        and policy_close
        and value_close
        and top1_agreement == 1.0
    )
    return {
        "schema_version": A15_SCHEMA_VERSION,
        "passed": passed,
        "finite": finite,
        "shape_match": shape_match,
        "policy_allclose": policy_close,
        "value_allclose": value_close,
        "policy_top1_agreement": top1_agreement,
        "policy_max_abs_error": policy_max_abs,
        "value_max_abs_error": value_max_abs,
        "atol": float(atol),
        "rtol": float(rtol),
        "probe_input_sha256": tensor_sha256(probe),
    }


def measure_prepared_inputs(
    model: Any,
    device: Any,
    canonical_inputs: Sequence[Any],
    *,
    n_waves: int,
    warmup_waves: int,
    time_fn: Callable[[], float] = time.perf_counter,
) -> dict[str, float | int]:
    """Measure one repetition with input construction and transfer excluded."""
    import torch

    if not canonical_inputs:
        raise ValueError("canonical_inputs must not be empty")
    is_cuda = getattr(device, "type", str(device)) == "cuda"
    prepared = [tensor.to(device) for tensor in canonical_inputs]
    streams = [torch.cuda.Stream(device=device) for _ in prepared] if is_cuda else []
    if is_cuda:
        # Inputs are materialized on the default stream and then consumed on
        # per-inflight streams.  Establish visibility before the timed region.
        torch.cuda.synchronize(device)

    def wave() -> None:
        with torch.no_grad():
            if is_cuda:
                for stream, tensor in zip(streams, prepared):
                    with torch.cuda.stream(stream):
                        model(tensor)
                torch.cuda.synchronize(device)
            else:
                for tensor in prepared:
                    model(tensor)

    for _ in range(int(warmup_waves)):
        wave()
    if is_cuda:
        torch.cuda.synchronize(device)

    started = time_fn()
    for _ in range(int(n_waves)):
        wave()
    if is_cuda:
        torch.cuda.synchronize(device)
    elapsed_s = max(1e-12, float(time_fn() - started))

    batch_size = int(canonical_inputs[0].shape[0])
    inflight = len(canonical_inputs)
    total_batches = inflight * int(n_waves)
    total_items = batch_size * total_batches
    return {
        "elapsed_s": elapsed_s,
        "total_items": total_items,
        "total_batches": total_batches,
        "items_per_s": total_items / elapsed_s,
        "amortized_ms_per_batch": elapsed_s * 1000.0 / total_batches,
        "wave_latency_ms": elapsed_s * 1000.0 / int(n_waves),
    }


def run_matched_measurements(
    cpu_model: Any,
    cuda_model: Any,
    *,
    profile: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    base_input_seed: int,
    workload_identity_sha256: str,
    cuda_device_index: int = 0,
) -> list[dict[str, Any]]:
    """Run alternating CPU/CUDA repetitions and retain every raw row."""
    import torch

    cpu_device = torch.device("cpu")
    cuda_device = torch.device(f"cuda:{int(cuda_device_index)}")
    rows: list[dict[str, Any]] = []
    cell_index = 0
    for batch_size in profile["batch_sizes"]:
        for inflight in profile["inflight_grid"]:
            cell_id = f"b{int(batch_size)}_i{int(inflight)}"
            for repetition in range(int(profile["repetitions"])):
                canonical_inputs, input_hash, input_seed = make_matched_inputs(
                    batch_size=int(batch_size),
                    inflight=int(inflight),
                    in_ch=int(model_spec["in_ch"]),
                    board=int(model_spec["board"]),
                    base_seed=int(base_input_seed),
                    repetition=repetition,
                )
                order = (
                    REQUIRED_BACKENDS
                    if (cell_index + repetition) % 2 == 0
                    else tuple(reversed(REQUIRED_BACKENDS))
                )
                for order_index, backend in enumerate(order):
                    model = cpu_model if backend == "cpu" else cuda_model
                    device = cpu_device if backend == "cpu" else cuda_device
                    measured = measure_prepared_inputs(
                        model,
                        device,
                        canonical_inputs,
                        n_waves=int(profile["n_waves"]),
                        warmup_waves=int(profile["warmup_waves"]),
                    )
                    rows.append(
                        {
                            "schema_version": A15_SCHEMA_VERSION,
                            "axis_id": "A15",
                            "role": ROLE,
                            "evidence_status": EVIDENCE_STATUS,
                            "claim_scope": CLAIM_SCOPE,
                            "workload_identity_sha256": workload_identity_sha256,
                            "cell_id": cell_id,
                            "fixture_id": f"{cell_id}_r{repetition}_{backend}",
                            "backend": backend,
                            "submission_semantics": (
                                "serial_batches_within_wave"
                                if backend == "cpu"
                                else "one_cuda_stream_per_inflight_batch_then_wave_synchronize"
                            ),
                            "batch_size": int(batch_size),
                            "inflight": int(inflight),
                            "repetition": repetition,
                            "order_index": order_index,
                            "first_backend": order[0],
                            "input_seed": input_seed,
                            "input_sha256": input_hash,
                            "n_waves": int(profile["n_waves"]),
                            "warmup_waves": int(profile["warmup_waves"]),
                            "metric": "items_per_s",
                            "value": float(measured["items_per_s"]),
                            **measured,
                        }
                    )
            cell_index += 1
    return rows


def pair_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Join exact CPU/CUDA repetition pairs and reject incomplete cells."""
    grouped: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["cell_id"]), int(row["repetition"]))
        backend = str(row["backend"])
        if backend in grouped[key]:
            raise ValueError(f"duplicate backend row for {key}: {backend}")
        grouped[key][backend] = row

    paired: list[dict[str, Any]] = []
    for (cell_id, repetition), backends in sorted(grouped.items()):
        if set(backends) != set(REQUIRED_BACKENDS):
            raise ValueError(f"incomplete CPU/CUDA pair for {(cell_id, repetition)}")
        cpu = backends["cpu"]
        cuda = backends["cuda"]
        for key in (
            "workload_identity_sha256",
            "batch_size",
            "inflight",
            "input_seed",
            "input_sha256",
            "n_waves",
            "warmup_waves",
        ):
            if cpu[key] != cuda[key]:
                raise ValueError(f"pair mismatch for {(cell_id, repetition)}: {key}")
        cpu_rate = float(cpu["items_per_s"])
        cuda_rate = float(cuda["items_per_s"])
        if not (
            math.isfinite(cpu_rate)
            and math.isfinite(cuda_rate)
            and cpu_rate > 0
            and cuda_rate > 0
        ):
            raise ValueError(
                f"non-positive or non-finite throughput for {(cell_id, repetition)}"
            )
        paired.append(
            {
                "schema_version": A15_SCHEMA_VERSION,
                "axis_id": "A15",
                "role": ROLE,
                "evidence_status": EVIDENCE_STATUS,
                "claim_scope": CLAIM_SCOPE,
                "workload_identity_sha256": cpu["workload_identity_sha256"],
                "cell_id": cell_id,
                "batch_size": int(cpu["batch_size"]),
                "inflight": int(cpu["inflight"]),
                "repetition": repetition,
                "first_backend": cpu["first_backend"],
                "input_seed": int(cpu["input_seed"]),
                "input_sha256": cpu["input_sha256"],
                "cpu_items_per_s": cpu_rate,
                "cuda_items_per_s": cuda_rate,
                "cuda_to_cpu_throughput_ratio": cuda_rate / cpu_rate,
                "cpu_amortized_ms_per_batch": float(cpu["amortized_ms_per_batch"]),
                "cuda_amortized_ms_per_batch": float(cuda["amortized_ms_per_batch"]),
            }
        )
    return paired


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["backend"]), int(row["batch_size"]), int(row["inflight"]))
        ].append(row)
    aggregates: list[dict[str, Any]] = []
    for (backend, batch_size, inflight), group in sorted(grouped.items()):
        rates = [float(row["items_per_s"]) for row in group]
        batch_ms = [float(row["amortized_ms_per_batch"]) for row in group]
        mean_rate = statistics.fmean(rates)
        rate_sd = statistics.pstdev(rates) if len(rates) > 1 else 0.0
        aggregates.append(
            {
                "backend": backend,
                "batch_size": batch_size,
                "inflight": inflight,
                "repetitions": len(group),
                "items_per_s_median": statistics.median(rates),
                "items_per_s_mean": mean_rate,
                "items_per_s_cv": rate_sd / mean_rate if mean_rate > 0 else None,
                "amortized_ms_per_batch_median": statistics.median(batch_ms),
            }
        )
    return aggregates


def diagnostic_summary(
    rows: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    *,
    profile: Mapping[str, Any],
    parity: Mapping[str, Any],
    workload_identity_sha256: str,
) -> dict[str, Any]:
    expected_pairs = (
        len(profile["batch_sizes"])
        * len(profile["inflight_grid"])
        * int(profile["repetitions"])
    )
    aggregates = aggregate_rows(rows)
    complete = bool(len(rows) == expected_pairs * 2 and len(paired) == expected_pairs)
    return {
        "schema_version": A15_SCHEMA_VERSION,
        "axis_id": "A15",
        "role": ROLE,
        "evidence_status": EVIDENCE_STATUS,
        "experiment_id": EXPERIMENT_ID,
        "execution_mode": EXECUTION_MODE,
        "plot_category": PLOT_CATEGORY,
        "claim_status": "PREPARATION_DIAGNOSTIC_ONLY",
        "claim_scope": CLAIM_SCOPE,
        "promotion": {"auto": False, "eligible": False},
        "workload_identity_sha256": workload_identity_sha256,
        "profile": dict(profile),
        "semantic_parity": dict(parity),
        "raw_row_count": len(rows),
        "paired_row_count": len(paired),
        "expected_paired_row_count": expected_pairs,
        "matched_pair_contract_complete": complete,
        "pipeline_ready_for_full_profile": bool(complete and parity.get("passed")),
        "full_matrix_executed": bool(profile.get("name") == "full" and complete),
        "aggregates": aggregates,
        "prohibited_conclusions": [
            "no play-strength or decision-quality conclusion",
            "no energy-efficiency conclusion because CPU/GPU power is not controlled",
            "no production scheduler promotion",
            "no shipped-network conclusion from the representative evaluator body",
            "no CPU-superiority or CUDA-superiority generalization beyond this runtime contract",
        ],
    }


def render_diagnostic_plot(
    path: Path,
    aggregates: Sequence[Mapping[str, Any]],
    *,
    workload_identity_sha256: str,
) -> None:
    """Render the minimum useful, explicitly DIAGNOSTIC service-curve plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {"cpu": "#35618f", "cuda": "#d55e00"}
    for backend in REQUIRED_BACKENDS:
        backend_rows = [row for row in aggregates if row["backend"] == backend]
        for inflight in sorted({int(row["inflight"]) for row in backend_rows}):
            cells = sorted(
                (row for row in backend_rows if int(row["inflight"]) == inflight),
                key=lambda row: int(row["batch_size"]),
            )
            label = f"{backend.upper()}, inflight={inflight}"
            x = [int(row["batch_size"]) for row in cells]
            axes[0].plot(
                x,
                [float(row["items_per_s_median"]) for row in cells],
                marker="o",
                color=colors[backend],
                linestyle="-" if inflight == 1 else "--",
                label=label,
            )
            axes[1].plot(
                x,
                [float(row["amortized_ms_per_batch_median"]) for row in cells],
                marker="o",
                color=colors[backend],
                linestyle="-" if inflight == 1 else "--",
                label=label,
            )

    axes[0].set_title("Measured evaluator throughput")
    axes[0].set_xlabel("Batch size [items]")
    axes[0].set_ylabel("Throughput [items/s]")
    axes[0].set_yscale("log")
    axes[1].set_title("Amortized batch time (not response latency)")
    axes[1].set_xlabel("Batch size [items]")
    axes[1].set_ylabel("Amortized time [ms/submitted batch]")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(
        "A15 DIAGNOSTIC ONLY — matched representative workload; no efficiency promotion\n"
        f"workload {workload_identity_sha256[:16]}",
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    fig.savefig(path, dpi=160)
    plt.close(fig)
