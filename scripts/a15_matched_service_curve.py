#!/usr/bin/env python3
"""Run the A15 matched CPU/CUDA service-curve diagnostic.

The ``diagnostic`` profile is a readiness smoke.  The separate ``full`` profile
retains the complete 24-cell batch x inflight matrix and repeated timings.  Both
profiles are diagnostic-only and cannot promote a scheduler or efficiency
claim.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quartz.experiment_manifest import (  # noqa: E402
    atomic_json_dump,
    build_run_manifest,
    file_sha256,
    finalize_run_manifest,
    utc_now,
)
from quartz.experiments import a15_matched_service_curve as a15  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "a15_matched_service_curve.v1.json"
DEFAULT_OUTPUT = (
    REPO_ROOT / "results" / "idea_foundry" / "a15_matched_service_curve_diagnostic"
)


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("A15 config root must be an object")
    return payload


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty JSONL: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_complete_output(
    output_dir: Path, *, config_path: Path, profile_name: str
) -> dict[str, Any]:
    """Validate an already-published result for idempotent v2 retry.

    Partial or drifted directories are never reused.  A complete directory can
    be returned only when its source, input, and retained artifact hashes still
    match the current checkout and requested profile.
    """

    artifact_names = {
        "rows.jsonl",
        "summary.json",
        "diagnostic.png",
        "service_curve_rows.v1.csv",
        "paired_backend_rows.v1.csv",
        "plot_metadata.v1.json",
    }
    expected_names = artifact_names | {"run_manifest.json"}
    observed_names = {path.name for path in output_dir.iterdir()}
    if observed_names != expected_names:
        raise RuntimeError(
            "existing A15 output is incomplete or contains unexpected entries: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"unexpected={sorted(observed_names - expected_names)}"
        )
    for name in expected_names:
        path = output_dir / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"existing A15 output entry is not a regular file: {path}"
            )
    try:
        manifest = json.loads(
            (output_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"existing A15 output metadata is invalid: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(summary, dict):
        raise RuntimeError("existing A15 output metadata must be JSON objects")
    requested = manifest.get("resolved_config", {}).get("profile", {}).get("name")
    if requested != profile_name:
        raise RuntimeError(
            f"existing A15 output profile drift: {requested!r} != {profile_name!r}"
        )
    if (
        manifest.get("status") != "completed_no_promotion"
        or summary.get("execution_status") != "completed_no_promotion"
    ):
        raise RuntimeError("existing A15 output is not a completed no-promotion result")
    if manifest.get("evidence_status") != a15.EVIDENCE_STATUS:
        raise RuntimeError("existing A15 output evidence status drift")
    config_hash = file_sha256(config_path)
    input_rows = manifest.get("input_hashes")
    if not isinstance(input_rows, list) or not any(
        row.get("name") == "a15_config" and row.get("sha256") == config_hash
        for row in input_rows
        if isinstance(row, dict)
    ):
        raise RuntimeError("existing A15 output config hash drift")
    source_rows = manifest.get("source_hashes")
    if not isinstance(source_rows, list) or not source_rows:
        raise RuntimeError("existing A15 output lacks source hashes")
    for row in source_rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise RuntimeError("existing A15 output has an invalid source hash row")
        source = (REPO_ROOT / row["path"]).resolve()
        if not source.is_file() or file_sha256(source) != row.get("sha256"):
            raise RuntimeError(f"existing A15 output source hash drift: {source}")
    artifact_rows = manifest.get("artifacts")
    if not isinstance(artifact_rows, list):
        raise RuntimeError("existing A15 output lacks artifact hashes")
    artifact_records = {
        row.get("path"): row for row in artifact_rows if isinstance(row, dict)
    }
    if set(artifact_records) != artifact_names or len(artifact_rows) != len(
        artifact_names
    ):
        raise RuntimeError("existing A15 output artifact inventory drift")
    for row in artifact_records.values():
        artifact = output_dir / row["path"]
        if not artifact.is_file() or file_sha256(artifact) != row.get("sha256"):
            raise RuntimeError(f"existing A15 artifact hash drift: {artifact}")
    return manifest


def cuda_runtime_proof(torch: Any, device_index: int) -> dict[str, Any]:
    if torch.version.cuda is None:
        raise RuntimeError(
            "A15 requires a CUDA PyTorch build (torch.version.cuda is None)"
        )
    if getattr(torch.version, "hip", None) is not None:
        raise RuntimeError("A15 CUDA lane refuses a ROCm/HIP PyTorch build")
    if not torch.cuda.is_available():
        raise RuntimeError("A15 requires torch.cuda.is_available() == True")
    if int(device_index) < 0 or int(device_index) >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device index is not visible: {device_index}")
    properties = torch.cuda.get_device_properties(int(device_index))
    torch_uuid = str(getattr(properties, "uuid", "")).strip()
    if not torch_uuid:
        raise RuntimeError("PyTorch did not expose a UUID for the selected CUDA device")
    command = [
        "nvidia-smi",
        f"--id=GPU-{torch_uuid}",
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"nvidia-smi proof failed: {exc}") from exc
    if process.returncode != 0 or not process.stdout.strip():
        raise RuntimeError(
            f"nvidia-smi proof failed with exit {process.returncode}: {process.stderr.strip()}"
        )
    name, uuid, driver, memory_mb = [
        part.strip() for part in process.stdout.splitlines()[0].split(",")
    ]
    if uuid.removeprefix("GPU-") != torch_uuid.removeprefix("GPU-"):
        raise RuntimeError(
            "PyTorch and nvidia-smi selected different CUDA device UUIDs"
        )
    return {
        "torch_version": str(torch.__version__),
        "torch_cuda_build": str(torch.version.cuda),
        "torch_hip_build": None,
        "cuda_device_index": int(device_index),
        "torch_device_name": str(properties.name),
        "torch_device_uuid": torch_uuid,
        "torch_total_memory_bytes": int(properties.total_memory),
        "nvidia_smi_name": name,
        "nvidia_smi_uuid": uuid,
        "nvidia_driver_version": driver,
        "nvidia_smi_memory_total_mb": int(memory_mb),
        "nvidia_smi_command": command,
    }


def configure_runtime(torch: Any, runtime: Mapping[str, Any]) -> dict[str, Any]:
    intraop = int(runtime["cpu_intraop_threads"])
    interop = int(runtime["cpu_interop_threads"])
    if intraop < 1 or interop < 1:
        raise ValueError("CPU intra-op and inter-op thread counts must be positive")
    torch.set_num_threads(intraop)
    torch.set_num_interop_threads(interop)
    if torch.get_num_threads() != intraop or torch.get_num_interop_threads() != interop:
        raise RuntimeError("PyTorch did not honor the pinned CPU thread contract")

    allow_tf32 = bool(runtime.get("allow_tf32", False))
    cudnn_benchmark = bool(runtime.get("cudnn_benchmark", False))
    cudnn_deterministic = bool(runtime.get("cudnn_deterministic", True))
    deterministic_algorithms = bool(runtime.get("deterministic_algorithms", True))
    torch.use_deterministic_algorithms(deterministic_algorithms)
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
    return {
        "cpu_intraop_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "dtype": "torch.float32",
        "input_transfer_in_timing": False,
        "input_generation_in_timing": False,
        "power_measurement": "not_collected",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--profile", choices=("diagnostic", "full"), default="diagnostic"
    )
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import torch

    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    profile = a15.validate_config(cfg, args.profile)
    model_spec = dict(cfg["model"])
    runtime_contract = configure_runtime(torch, cfg["runtime_contract"])
    hardware = cuda_runtime_proof(torch, args.cuda_device)

    final_output_dir = args.output_dir.resolve()
    if final_output_dir.exists():
        if not final_output_dir.is_dir():
            raise SystemExit(f"output path is not a directory: {final_output_dir}")
        if any(final_output_dir.iterdir()):
            validate_complete_output(
                final_output_dir,
                config_path=config_path,
                profile_name=str(profile["name"]),
            )
            print(
                json.dumps(
                    {
                        "status": "completed_no_promotion",
                        "axis_id": "A15",
                        "profile": profile["name"],
                        "output_dir": str(final_output_dir),
                        "idempotent_reuse": True,
                    },
                    sort_keys=True,
                )
            )
            return 0
        final_output_dir.rmdir()
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{final_output_dir.name}.attempt-",
            dir=final_output_dir.parent,
        )
    )
    mpl_cache_dir = Path(
        tempfile.mkdtemp(prefix=".a15-matplotlib-", dir=final_output_dir.parent)
    )
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)

    model_seed = int(cfg["seed_contract"]["model_seed"])
    torch.manual_seed(model_seed)
    cpu_model = a15.build_eval_model(
        in_ch=int(model_spec["in_ch"]),
        channels=int(model_spec["channels"]),
        blocks=int(model_spec["blocks"]),
        board=int(model_spec["board"]),
        actions=int(model_spec["actions"]),
    ).eval()
    state_hash = a15.model_state_sha256(cpu_model)
    cuda_device = torch.device(f"cuda:{int(args.cuda_device)}")
    cuda_model = copy.deepcopy(cpu_model).to(cuda_device).eval()
    cuda_state_hash = a15.model_state_sha256(cuda_model)
    if cuda_state_hash != state_hash:
        raise RuntimeError("CPU/CUDA model state hashes differ after device copy")

    service_curve_source = REPO_ROOT / "quartz" / "experiments" / "service_curve.py"
    workload_contract, workload_hash = a15.build_workload_identity(
        model_spec=model_spec,
        model_state_hash=state_hash,
        profile=profile,
        seed_contract=cfg["seed_contract"],
        runtime_contract={
            **runtime_contract,
            "torch_version": str(torch.__version__),
            "torch_cuda_build": str(torch.version.cuda),
        },
        builder_source_sha256=file_sha256(service_curve_source),
    )

    probe_generator = torch.Generator(device="cpu")
    probe_generator.manual_seed(int(cfg["seed_contract"]["parity_probe_seed"]))
    probe = torch.randn(
        int(cfg["parity"]["probe_batch"]),
        int(model_spec["in_ch"]),
        int(model_spec["board"]),
        int(model_spec["board"]),
        generator=probe_generator,
        dtype=torch.float32,
    )
    parity = a15.check_backend_parity(
        cpu_model,
        cuda_model,
        probe,
        atol=float(cfg["parity"]["atol"]),
        rtol=float(cfg["parity"]["rtol"]),
    )

    source_paths = [
        Path(__file__),
        REPO_ROOT / "quartz" / "experiments" / "a15_matched_service_curve.py",
        service_curve_source,
        REPO_ROOT / "quartz" / "experiment_manifest.py",
        config_path,
    ]
    resolved_config = {
        "schema_version": a15.A15_SCHEMA_VERSION,
        "axis_id": "A15",
        "profile": profile,
        "model": model_spec,
        "model_seed": model_seed,
        "model_state_sha256": state_hash,
        "workload_identity_sha256": workload_hash,
        "runtime_contract": runtime_contract,
        "hardware": hardware,
        "semantic_parity": parity,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
    }
    manifest = build_run_manifest(
        experiment_id=a15.EXPERIMENT_ID,
        execution_mode=a15.EXECUTION_MODE,
        resolved_config=resolved_config,
        repo_root=REPO_ROOT,
        source_paths=source_paths,
        argv=sys.argv if argv is None else [str(Path(__file__)), *argv],
        started_at=utc_now(),
        assumptions=cfg["assumptions"],
        prohibited_inferences=cfg["prohibited_inferences"],
    )
    manifest.update(
        {
            "schema_version": a15.A15_SCHEMA_VERSION,
            "axis_id": "A15",
            "role": a15.ROLE,
            "evidence_status": a15.EVIDENCE_STATUS,
            "claim_status": "PREPARATION_DIAGNOSTIC_ONLY",
            "claim_scope": a15.CLAIM_SCOPE,
            "auto_promoted": False,
            "promotion": {"auto": False, "eligible": False},
            "workload_identity": workload_contract,
            "workload_identity_sha256": workload_hash,
            "semantic_parity": parity,
            "source_hashes": list(manifest["sources"]),
            "input_hashes": [
                {
                    "name": "a15_config",
                    "path": str(config_path.relative_to(REPO_ROOT)),
                    "sha256": file_sha256(config_path),
                }
            ],
            "seed_contract": dict(cfg["seed_contract"]),
        }
    )
    manifest_path = output_dir / "run_manifest.json"
    atomic_json_dump(manifest_path, manifest)

    artifact_paths: list[Path] = []
    try:
        if not parity["passed"]:
            raise RuntimeError(
                f"CPU/CUDA semantic parity failed: {json.dumps(parity, sort_keys=True)}"
            )
        rows = a15.run_matched_measurements(
            cpu_model,
            cuda_model,
            profile=profile,
            model_spec=model_spec,
            base_input_seed=int(cfg["seed_contract"]["input_seed"]),
            workload_identity_sha256=workload_hash,
            cuda_device_index=int(args.cuda_device),
        )
        paired = a15.pair_rows(rows)
        summary = a15.diagnostic_summary(
            rows,
            paired,
            profile=profile,
            parity=parity,
            workload_identity_sha256=workload_hash,
        )
        summary.update(
            {
                "execution_status": "completed_no_promotion",
                "model": model_spec,
                "model_state_sha256": state_hash,
                "runtime_contract": runtime_contract,
                "hardware": hardware,
                "workload_identity": workload_contract,
                "seed_contract": dict(cfg["seed_contract"]),
                "note": (
                    "Matched wall-clock service-curve diagnostic only. CPU inflight batches are "
                    "serial; CUDA inflight batches use distinct streams. The throughput ratio is "
                    "descriptive for this pinned contract, not an efficiency or production claim."
                ),
            }
        )

        rows_jsonl_path = output_dir / "rows.jsonl"
        rows_path = output_dir / "service_curve_rows.v1.csv"
        paired_path = output_dir / "paired_backend_rows.v1.csv"
        summary_path = output_dir / "summary.json"
        plot_path = output_dir / "diagnostic.png"
        plot_metadata_path = output_dir / "plot_metadata.v1.json"
        write_jsonl(rows_jsonl_path, rows)
        write_csv(rows_path, rows)
        write_csv(paired_path, paired)
        atomic_json_dump(summary_path, summary)
        a15.render_diagnostic_plot(
            plot_path,
            summary["aggregates"],
            workload_identity_sha256=workload_hash,
        )
        atomic_json_dump(
            plot_metadata_path,
            {
                "schema_version": a15.A15_SCHEMA_VERSION,
                "axis_id": "A15",
                "role": a15.ROLE,
                "evidence_status": a15.EVIDENCE_STATUS,
                "claim_scope": a15.CLAIM_SCOPE,
                "plot_category": a15.PLOT_CATEGORY,
                "plot_path": plot_path.name,
                "source_data": [
                    {"path": rows_path.name, "sha256": file_sha256(rows_path)},
                    {"path": paired_path.name, "sha256": file_sha256(paired_path)},
                ],
                "workload_identity_sha256": workload_hash,
                "quantity": "median throughput and amortized batch time by backend/batch/inflight",
                "interpretation": "backend service-curve shape under one pinned representative workload",
                "does_not_show": [
                    "play strength or decision quality",
                    "controlled energy efficiency",
                    "production scheduler benefit",
                    "shipped-network behavior",
                ],
            },
        )
        artifact_paths = [
            rows_jsonl_path,
            rows_path,
            paired_path,
            summary_path,
            plot_path,
            plot_metadata_path,
        ]
        manifest = finalize_run_manifest(
            manifest,
            output_dir=output_dir,
            artifact_paths=artifact_paths,
            status="completed_no_promotion",
        )
        atomic_json_dump(manifest_path, manifest)
        shutil.rmtree(mpl_cache_dir, ignore_errors=True)
        os.replace(output_dir, final_output_dir)
    except Exception as exc:
        shutil.rmtree(mpl_cache_dir, ignore_errors=True)
        failure_path = output_dir / "failure.v1.json"
        atomic_json_dump(
            failure_path,
            {
                "schema_version": a15.A15_SCHEMA_VERSION,
                "axis_id": "A15",
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "workload_identity_sha256": workload_hash,
            },
        )
        retained = [path for path in artifact_paths if path.exists()]
        manifest = finalize_run_manifest(
            manifest,
            output_dir=output_dir,
            artifact_paths=[*retained, failure_path],
            status="failed",
        )
        atomic_json_dump(manifest_path, manifest)
        raise

    print(
        json.dumps(
            {
                "status": "completed_no_promotion",
                "axis_id": "A15",
                "profile": profile["name"],
                "plot_category": a15.PLOT_CATEGORY,
                "workload_identity_sha256": workload_hash,
                "semantic_parity_passed": parity["passed"],
                "paired_rows": len(paired),
                "output_dir": str(final_output_dir),
                "cpu_intraop_threads": runtime_contract["cpu_intraop_threads"],
                "cpu_interop_threads": runtime_contract["cpu_interop_threads"],
                "host": platform.node(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
