#!/usr/bin/env python3
"""Prepare or finalize the A19 graph-seed ablation without claim promotion."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quartz.idea_foundry.a19_ablation import (  # noqa: E402
    A19_AXIS_ID,
    A19PreparationError,
    build_ablation_contract,
    canonical_json_bytes,
    file_sha256,
    generate_topology,
    load_json_strict,
    load_jsonl_strict,
    load_screen_plan,
    rank_graph_seeds,
    validate_inputs,
    validate_proxy_rows,
    validate_run_id,
    write_json,
    write_jsonl,
)


SCHEMA_VERSION = 1
ROLE = "ablation_readiness"
EVIDENCE_STATUS = "skeleton_only"
CLAIM_SCOPE = "ablation_readiness_only"
PROMOTION = {
    "auto": False,
    "eligible": False,
    "reason": "A19 readiness and proxy screening do not establish efficacy",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise A19PreparationError(f"{label} must not be a symlink: {path}")
    if not path.exists() or not path.is_file():
        raise A19PreparationError(f"{label} must be an existing regular file: {path}")
    return path.resolve()


def _repo_file(path: Path, label: str) -> Path:
    resolved = _regular_file(path, label)
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise A19PreparationError(
            f"{label} must remain inside the repository: {path}"
        ) from exc
    return resolved


def _check_digest(path: Path, expected: Any, label: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise A19PreparationError(f"{label} expected SHA-256 is invalid")
    actual = file_sha256(path)
    if actual != expected:
        raise A19PreparationError(
            f"{label} hash mismatch: expected {expected}, got {actual}"
        )
    return actual


def _validate_controller_contract(path: Path) -> dict[str, Any]:
    payload = load_json_strict(path)
    expected_keys = {
        "schema_version",
        "axis_id",
        "artifact_kind",
        "immutable_during_ablation",
        "is_model_checkpoint",
        "game",
        "controller_identity",
        "runtime_contract",
        "allowed_mutations",
        "prohibited_inferences",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise A19PreparationError("frozen controller has an invalid top-level schema")
    if payload["schema_version"] != 1 or payload["axis_id"] != A19_AXIS_ID:
        raise A19PreparationError("frozen controller version/axis mismatch")
    if payload["artifact_kind"] != "frozen_controller_contract":
        raise A19PreparationError(
            "controller input is not a frozen controller contract"
        )
    if payload["immutable_during_ablation"] is not True:
        raise A19PreparationError(
            "controller contract must be immutable during ablation"
        )
    if payload["is_model_checkpoint"] is not False:
        raise A19PreparationError(
            "controller contract must not claim to be a model checkpoint"
        )
    if payload["game"] != "gomoku7":
        raise A19PreparationError("controller contract game must be gomoku7")
    if (
        not isinstance(payload["controller_identity"], dict)
        or not payload["controller_identity"]
    ):
        raise A19PreparationError("controller_identity must be a non-empty object")
    if (
        not isinstance(payload["runtime_contract"], dict)
        or not payload["runtime_contract"]
    ):
        raise A19PreparationError("runtime_contract must be a non-empty object")
    if payload["allowed_mutations"] != []:
        raise A19PreparationError("frozen controller allowed_mutations must be empty")
    if (
        not isinstance(payload["prohibited_inferences"], list)
        or not payload["prohibited_inferences"]
    ):
        raise A19PreparationError("controller prohibited_inferences must be non-empty")
    return payload


def _validate_replay_npz(
    path: Path,
    *,
    replicate_seed: int,
    train_positions: int,
    validation_positions: int,
    optimizer_steps: int,
    batch_size: int,
) -> dict[str, Any]:
    expected_keys = {
        "replay_format",
        "states",
        "policy_ptr",
        "policy_idx",
        "policy_val",
        "n_actions",
        "values",
        "metadata_json",
    }
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != expected_keys:
                raise A19PreparationError(
                    f"replay keys mismatch for {path}: {sorted(payload.files)}"
                )
            replay_format = payload["replay_format"]
            states = payload["states"]
            ptr = payload["policy_ptr"]
            idx = payload["policy_idx"]
            val = payload["policy_val"]
            actions = payload["n_actions"]
            values = payload["values"]
            metadata = payload["metadata_json"]
            if replay_format.shape != (1,) or int(replay_format[0]) != 2:
                raise A19PreparationError(f"{path} is not sparse replay format v2")
            if (
                states.ndim != 4
                or states.shape[1:] != (17, 7, 7)
                or states.dtype != np.float32
            ):
                raise A19PreparationError(
                    f"{path} states must have float32 shape [N,17,7,7]"
                )
            count = int(states.shape[0])
            if (
                ptr.shape != (count + 1,)
                or values.shape != (count,)
                or actions.shape != (count,)
            ):
                raise A19PreparationError(
                    f"{path} replay arrays have inconsistent row counts"
                )
            if metadata.shape != (count,):
                raise A19PreparationError(f"{path} metadata row count is inconsistent")
            if int(ptr[0]) != 0 or int(ptr[-1]) != len(idx) or len(idx) != len(val):
                raise A19PreparationError(f"{path} sparse policy pointers are invalid")
            if np.any(np.diff(ptr) <= 0) or np.any(actions != 49):
                raise A19PreparationError(
                    f"{path} sparse policy/action contract is invalid"
                )
            if np.any(idx < 0) or np.any(idx >= 49):
                raise A19PreparationError(
                    f"{path} sparse policy contains invalid action indices"
                )
            if (
                not np.all(np.isfinite(states))
                or not np.all(np.isfinite(val))
                or not np.all(np.isfinite(values))
            ):
                raise A19PreparationError(
                    f"{path} contains non-finite numerical values"
                )
            if np.any(val < 0.0) or np.any(values < -1.0) or np.any(values > 1.0):
                raise A19PreparationError(
                    f"{path} target values are outside their allowed ranges"
                )
            policy_sums = np.add.reduceat(val, ptr[:-1])
            if not np.allclose(policy_sums, 1.0, rtol=0.0, atol=2e-6):
                raise A19PreparationError(f"{path} sparse policies are not normalized")
            import hashlib

            state_hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in states]
            unique_count = len(set(state_hashes))
            required = train_positions + validation_positions
            if unique_count < required:
                raise A19PreparationError(
                    f"{path} has {unique_count} unique state groups, fewer than required {required}"
                )
            group_digest = hashlib.sha256(
                canonical_json_bytes(sorted(set(state_hashes)))
            ).hexdigest()
            ordered_groups = sorted(
                set(state_hashes),
                key=lambda state_hash: hashlib.sha256(
                    f"A19-split-v1:{replicate_seed}:{state_hash}".encode("ascii")
                ).hexdigest(),
            )
            train_groups = ordered_groups[:train_positions]
            validation_groups = ordered_groups[
                train_positions : train_positions + validation_positions
            ]
            schedule_rng = random.Random((replicate_seed << 32) ^ 0xA19)
            batch_schedule = [
                schedule_rng.randrange(train_positions)
                for _ in range(optimizer_steps * batch_size)
            ]
            split_contract = {
                "schema_version": 1,
                "replicate_seed": replicate_seed,
                "method": "deduplicated_state_hash_v1",
                "train_state_group_hashes": train_groups,
                "validation_state_group_hashes": validation_groups,
                "train_split_sha256": hashlib.sha256(
                    canonical_json_bytes(train_groups)
                ).hexdigest(),
                "validation_split_sha256": hashlib.sha256(
                    canonical_json_bytes(validation_groups)
                ).hexdigest(),
                "batch_schedule_method": "python_random_index_schedule_v1",
                "batch_schedule_sha256": hashlib.sha256(
                    canonical_json_bytes(batch_schedule)
                ).hexdigest(),
                "optimizer_steps": optimizer_steps,
                "batch_size": batch_size,
            }
            return {
                "row_count": count,
                "unique_state_group_count": unique_count,
                "state_group_sha256": group_digest,
                "state_shape": list(states.shape[1:]),
                "action_count": 49,
                "replay_format": 2,
                "split_contract": split_contract,
            }
    except A19PreparationError:
        raise
    except Exception as exc:
        raise A19PreparationError(f"cannot validate replay {path}: {exc}") from exc


def validate_replay_manifest(
    path: Path, plan: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = _regular_file(path, "replay source manifest")
    raw = load_json_strict(manifest_path)
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "axis_id",
        "game",
        "source_status",
        "sources",
    }:
        raise A19PreparationError(
            "replay source manifest has an invalid top-level schema"
        )
    if raw["schema_version"] != 1 or raw["axis_id"] != A19_AXIS_ID:
        raise A19PreparationError("replay source manifest version/axis mismatch")
    if raw["game"] != "gomoku7":
        raise A19PreparationError("the A19 v1 proxy is preregistered for gomoku7")
    if raw["source_status"] != "trained_bootstrap_non_promoted":
        raise A19PreparationError(
            "replay sources must remain labelled non-promoted bootstrap training"
        )
    sources = raw["sources"]
    if not isinstance(sources, list) or len(sources) != len(plan.replicate_seeds):
        raise A19PreparationError(
            "replay source count does not match paired replicate seeds"
        )
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict) or set(source) != {
            "replicate_seed",
            "replay_path",
            "replay_sha256",
            "checkpoint_path",
            "checkpoint_sha256",
            "checkpoint_status_path",
            "checkpoint_status_sha256",
        }:
            raise A19PreparationError(f"replay source {index} has an invalid schema")
        seed = source["replicate_seed"]
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed not in plan.replicate_seeds
        ):
            raise A19PreparationError(
                f"replay source {index} has an unregistered replicate seed"
            )
        if seed in seen:
            raise A19PreparationError(f"duplicate replay source seed: {seed}")
        seen.add(seed)
        replay = _repo_file(REPO_ROOT / source["replay_path"], "replay source")
        checkpoint = _repo_file(
            REPO_ROOT / source["checkpoint_path"], "source checkpoint"
        )
        status_path = _repo_file(
            REPO_ROOT / source["checkpoint_status_path"], "checkpoint status"
        )
        replay_hash = _check_digest(replay, source["replay_sha256"], "replay source")
        checkpoint_hash = _check_digest(
            checkpoint, source["checkpoint_sha256"], "source checkpoint"
        )
        status_hash = _check_digest(
            status_path, source["checkpoint_status_sha256"], "checkpoint status"
        )
        status = load_json_strict(status_path)
        if not isinstance(status, dict):
            raise A19PreparationError("checkpoint status must be a JSON object")
        if status.get("preferred_posttrain_checkpoint") != "latest.pt":
            raise A19PreparationError(
                "source checkpoint status does not prefer latest.pt"
            )
        if status.get("best_checkpoint_bootstrap_seeded") is not True:
            raise A19PreparationError(
                "source status must record bootstrap-seeded best.pt"
            )
        if status.get("saw_promotion") is not False:
            raise A19PreparationError(
                "promoted source checkpoints are outside this preregistration"
            )
        replay_summary = _validate_replay_npz(
            replay,
            replicate_seed=seed,
            train_positions=int(plan.budget_contract["train_positions"]),
            validation_positions=int(plan.budget_contract["validation_positions"]),
            optimizer_steps=int(plan.budget_contract["optimizer_steps"]),
            batch_size=int(plan.budget_contract["batch_size"]),
        )
        normalized.append(
            {
                "replicate_seed": seed,
                "replay": {"path": source["replay_path"], "sha256": replay_hash},
                "source_checkpoint": {
                    "path": source["checkpoint_path"],
                    "sha256": checkpoint_hash,
                    "status": "trained_bootstrap_non_promoted",
                },
                "checkpoint_status": {
                    "path": source["checkpoint_status_path"],
                    "sha256": status_hash,
                    "preferred_posttrain_checkpoint": "latest.pt",
                    "best_checkpoint_bootstrap_seeded": True,
                    "saw_promotion": False,
                },
                "replay_summary": replay_summary,
            }
        )
    if seen != set(plan.replicate_seeds):
        raise A19PreparationError("paired replay source coverage is incomplete")
    normalized.sort(key=lambda row: row["replicate_seed"])
    return raw, normalized


def _resource_estimate(plan: Any, topology: dict[str, Any]) -> dict[str, int]:
    architecture = plan.architecture
    channels = int(architecture["channels"])
    nodes = int(architecture["nodes"])
    global_blocks = int(architecture["global_blocks"])
    board_area = 49
    input_channels = 17
    topology_edges = len(topology["edges"])
    input_parameters = input_channels * channels * 3 * 3
    node_parameters = nodes * (2 * channels + channels * channels * 3 * 3)
    route_parameters = topology_edges
    # Pre-norm self-attention + 4x MLP, twice. This is the v1 estimator
    # contract; an implementation must report measured counts against it.
    global_parameters_per_block = (
        4 * channels
        + 3 * channels * channels
        + 3 * channels
        + channels * channels
        + channels
        + channels * 4 * channels
        + 4 * channels
        + 4 * channels * channels
        + channels
    )
    policy_parameters = channels + 1
    value_parameters = 2 * channels * channels + channels + channels + 1
    parameters = (
        input_parameters
        + node_parameters
        + route_parameters
        + global_blocks * global_parameters_per_block
        + policy_parameters
        + value_parameters
    )
    input_flops = 2 * board_area * input_channels * channels * 3 * 3
    node_flops = nodes * 2 * board_area * channels * channels * 3 * 3
    routing_flops = max(0, topology_edges - nodes) * board_area * channels
    global_flops_per_block = (
        2 * board_area * channels * 3 * channels
        + 4 * board_area * board_area * channels
        + 2 * board_area * channels * channels
        + 2 * board_area * channels * 4 * channels
        + 2 * board_area * 4 * channels * channels
    )
    head_flops = (
        2 * board_area * channels + 2 * (2 * channels) * channels + 2 * channels
    )
    flops = (
        input_flops
        + node_flops
        + routing_flops
        + global_blocks * global_flops_per_block
        + head_flops
    )
    return {
        "parameters": parameters,
        "flops": flops,
        "topology_edges": topology_edges,
        "nodes": nodes,
        "channels": channels,
        "global_blocks": global_blocks,
    }


def _source_hashes() -> list[dict[str, str]]:
    paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "quartz" / "idea_foundry" / "a19_ablation.py",
    )
    return [
        {"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path)}
        for path in paths
    ]


def _input_hashes(
    *,
    screen_plan_path: Path,
    replay_manifest_path: Path,
    controller_path: Path,
    replay_sources: Sequence[dict[str, Any]],
    proxy_results_path: Path | None = None,
) -> list[dict[str, str]]:
    rows = [
        {
            "name": "screen_plan",
            "path": str(screen_plan_path),
            "sha256": file_sha256(screen_plan_path),
        },
        {
            "name": "replay_source_manifest",
            "path": str(replay_manifest_path),
            "sha256": file_sha256(replay_manifest_path),
        },
        {
            "name": "frozen_controller",
            "path": str(controller_path),
            "sha256": file_sha256(controller_path),
        },
    ]
    for source in replay_sources:
        seed = source["replicate_seed"]
        for kind in ("replay", "source_checkpoint", "checkpoint_status"):
            item = source[kind]
            rows.append(
                {
                    "name": f"seed_{seed}_{kind}",
                    "path": item["path"],
                    "sha256": item["sha256"],
                }
            )
    if proxy_results_path is not None:
        rows.append(
            {
                "name": "proxy_results",
                "path": str(proxy_results_path),
                "sha256": file_sha256(proxy_results_path),
            }
        )
    return rows


def _seed_contract(plan: Any) -> dict[str, Any]:
    return {
        "graph_seeds": list(plan.graph_seeds),
        "paired_replicate_seeds": list(plan.replicate_seeds),
        "common_random_numbers_within_replicate": True,
        "grouping_unit": "deduplicated_state_hash",
        "split_method": plan.proxy_training["split_method"],
    }


def _expect_identity(payload: Any, expected: Mapping[str, Any], label: str) -> None:
    if not isinstance(payload, dict):
        raise A19PreparationError(f"existing {label} must be a JSON object")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise A19PreparationError(
                f"existing {label} identity drift for {key}: "
                f"expected {value!r}, got {payload.get(key)!r}"
            )


def _validate_existing_output(
    output_dir: Path,
    *,
    run_id: str,
    plan: Any,
    expected_source_hashes: Sequence[dict[str, str]],
    expected_input_hashes: Sequence[dict[str, str]],
) -> None:
    """Accept only a byte-complete prior launch from the same invocation."""

    if output_dir.is_symlink() or not output_dir.is_dir():
        raise A19PreparationError(
            f"existing output target is not a regular directory: {output_dir}"
        )
    expected_names = {
        "run_manifest.json",
        "rows.jsonl",
        "summary.json",
        "diagnostic.png",
        "diagnostic_interpretation.md",
        "a19_ablation_contract.v1.json",
        "a19_split_contract.v1.json",
        "a19_graph_seed_shortlist.v1.json",
    }
    children = list(output_dir.iterdir())
    actual_names = {child.name for child in children}
    if actual_names != expected_names:
        raise A19PreparationError(
            "existing output is incomplete or contains unexpected files: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    if any(child.is_symlink() or not child.is_file() for child in children):
        raise A19PreparationError(
            "existing output must contain only regular, non-symlink files"
        )

    manifest = load_json_strict(output_dir / "run_manifest.json")
    _expect_identity(
        manifest,
        {
            "schema_version": SCHEMA_VERSION,
            "axis_id": A19_AXIS_ID,
            "role": ROLE,
            "evidence_status": EVIDENCE_STATUS,
            "claim_scope": CLAIM_SCOPE,
            "execution_mode": "launch_contract_only",
            "status": "completed_no_promotion",
            "execution_status": "completed_no_promotion",
            "run_id": run_id,
            "auto_promoted": False,
            "promotion": PROMOTION,
            "seed_contract": _seed_contract(plan),
        },
        "run_manifest.json",
    )
    if manifest.get("source_hashes") != list(expected_source_hashes):
        raise A19PreparationError("existing run_manifest.json source hash drift")
    if manifest.get("input_hashes") != list(expected_input_hashes):
        raise A19PreparationError("existing run_manifest.json input hash drift")

    expected_artifact_names = expected_names - {"run_manifest.json"}
    artifact_rows = manifest.get("artifacts")
    if not isinstance(artifact_rows, list) or len(artifact_rows) != len(
        expected_artifact_names
    ):
        raise A19PreparationError(
            "existing run_manifest.json artifact list is incomplete"
        )
    seen_artifacts: set[str] = set()
    for index, record in enumerate(artifact_rows):
        if not isinstance(record, dict) or set(record) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise A19PreparationError(
                f"existing run_manifest.json artifact row {index} has invalid schema"
            )
        name = record["path"]
        if (
            not isinstance(name, str)
            or name not in expected_artifact_names
            or name in seen_artifacts
        ):
            raise A19PreparationError(
                f"existing run_manifest.json artifact path is invalid: {name!r}"
            )
        seen_artifacts.add(name)
        artifact_path = output_dir / name
        if record["size_bytes"] != artifact_path.stat().st_size:
            raise A19PreparationError(f"existing artifact size drift: {name}")
        if record["sha256"] != file_sha256(artifact_path):
            raise A19PreparationError(f"existing artifact SHA-256 drift: {name}")
    if seen_artifacts != expected_artifact_names:
        raise A19PreparationError("existing run_manifest.json artifact coverage drift")

    summary = load_json_strict(output_dir / "summary.json")
    _expect_identity(
        summary,
        {
            "schema_version": SCHEMA_VERSION,
            "axis_id": A19_AXIS_ID,
            "role": ROLE,
            "evidence_status": EVIDENCE_STATUS,
            "claim_scope": CLAIM_SCOPE,
            "execution_mode": "launch_contract_only",
            "execution_status": "completed_no_promotion",
            "run_id": run_id,
            "outcome_detail": "READY_FOR_PROXY_LAUNCH_NO_SHORTLIST",
            "shortlist_status": "not_measured",
            "shortlisted_graph_seeds": [],
            "proxy_trainer_status": "absent_not_implemented",
            "promotion": PROMOTION,
        },
        "summary.json",
    )
    shortlist = load_json_strict(output_dir / "a19_graph_seed_shortlist.v1.json")
    _expect_identity(
        shortlist,
        {
            "schema_version": SCHEMA_VERSION,
            "axis_id": A19_AXIS_ID,
            "role": ROLE,
            "claim_scope": CLAIM_SCOPE,
            "status": "not_measured",
            "shortlisted_graph_seeds": [],
            "ranking": [],
            "promotion": PROMOTION,
        },
        "a19_graph_seed_shortlist.v1.json",
    )
    contract = load_json_strict(output_dir / "a19_ablation_contract.v1.json")
    _expect_identity(
        contract,
        {
            "schema_version": SCHEMA_VERSION,
            "axis_id": A19_AXIS_ID,
            "role": ROLE,
            "evidence_status": EVIDENCE_STATUS,
            "claim_scope": CLAIM_SCOPE,
            "execution_status": "ready_for_graph_seed_proxy_launch",
            "shortlist_status": "not_measured",
            "shortlisted_graph_seeds": [],
            "auto_promoted": False,
        },
        "a19_ablation_contract.v1.json",
    )
    trainer = contract.get("proxy_trainer")
    if (
        not isinstance(trainer, dict)
        or trainer.get("measured_finalize_enabled") is not False
    ):
        raise A19PreparationError(
            "existing A19 contract unexpectedly enables measured finalize"
        )

    split_contract = load_json_strict(output_dir / "a19_split_contract.v1.json")
    _expect_identity(
        split_contract,
        {
            "schema_version": SCHEMA_VERSION,
            "axis_id": A19_AXIS_ID,
            "role": ROLE,
            "evidence_status": EVIDENCE_STATUS,
            "claim_scope": CLAIM_SCOPE,
            "split_method": plan.proxy_training["split_method"],
        },
        "a19_split_contract.v1.json",
    )
    split_rows = split_contract.get("contracts")
    if not isinstance(split_rows, list) or [
        row.get("replicate_seed") for row in split_rows if isinstance(row, dict)
    ] != list(plan.replicate_seeds):
        raise A19PreparationError("existing split-contract replicate coverage drift")

    rows = load_jsonl_strict(output_dir / "rows.jsonl")
    if len(rows) != len(plan.graph_seeds):
        raise A19PreparationError("existing rows.jsonl graph-seed coverage drift")
    if [row.get("graph_seed") for row in rows] != list(plan.graph_seeds):
        raise A19PreparationError("existing rows.jsonl graph-seed order drift")
    for row in rows:
        _expect_identity(
            row,
            {
                "schema_version": SCHEMA_VERSION,
                "axis_id": A19_AXIS_ID,
                "role": ROLE,
                "evidence_status": EVIDENCE_STATUS,
                "claim_scope": CLAIM_SCOPE,
                "replicate_seed": None,
                "metric": "topology_contract",
                "evidence_scope": "structural_readiness_only",
            },
            "rows.jsonl row",
        )
    if (
        not (output_dir / "diagnostic.png")
        .read_bytes()
        .startswith(b"\x89PNG\r\n\x1a\n")
    ):
        raise A19PreparationError("existing diagnostic.png is not a PNG artifact")
    interpretation = (output_dir / "diagnostic_interpretation.md").read_text(
        encoding="utf-8"
    )
    if (
        "DIAGNOSTIC" not in interpretation
        or "does not show" not in interpretation.lower()
    ):
        raise A19PreparationError("existing diagnostic interpretation identity drift")


def _plot_launch(path: Path, structural_rows: Sequence[dict[str, Any]]) -> None:
    mpl_cache = Path(tempfile.mkdtemp(prefix=".mplconfig-", dir=path.parent))
    previous_mpl_config = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        seeds = [str(row["graph_seed"]) for row in structural_rows]
        edge_counts = [row["resources"]["topology_edges"] for row in structural_rows]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(seeds, edge_counts, color="#64748b")
        ax.set_xlabel("Preregistered graph seed")
        ax.set_ylabel("Directed edge count [count]")
        ax.set_title("DIAGNOSTIC — A19 topology-density parity (not evaluator quality)")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    finally:
        if previous_mpl_config is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = previous_mpl_config
        shutil.rmtree(mpl_cache, ignore_errors=True)


def _plot_measured(path: Path, summaries: Sequence[dict[str, Any]]) -> None:
    mpl_cache = Path(tempfile.mkdtemp(prefix=".mplconfig-", dir=path.parent))
    previous_mpl_config = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        seeds = [str(row["graph_seed"]) for row in summaries]
        scores = [row["weighted_rank_percentile_mean"] for row in summaries]
        errors = [row["weighted_rank_percentile_se"] for row in summaries]
        colors = ["#2563eb" if row["selected"] else "#94a3b8" for row in summaries]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(seeds, scores, yerr=errors, color=colors, capsize=3)
        ax.set_xlabel("Preregistered graph seed (ordered by screen rank)")
        ax.set_ylabel("Weighted paired-rank percentile [unitless; lower is better]")
        ax.set_title("DIAGNOSTIC — A19 fixed-replay proxy screen (not play strength)")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    finally:
        if previous_mpl_config is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = previous_mpl_config
        shutil.rmtree(mpl_cache, ignore_errors=True)


def _base_summary(*, run_id: str, mode: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "axis_id": A19_AXIS_ID,
        "role": ROLE,
        "execution_status": "completed_no_promotion",
        "evidence_status": EVIDENCE_STATUS,
        "claim_scope": CLAIM_SCOPE,
        "execution_mode": mode,
        "run_id": run_id,
        "promotion": dict(PROMOTION),
    }


def run(args: argparse.Namespace) -> int:
    run_id = validate_run_id(args.run_id)
    output_dir = args.output_dir.resolve()
    screen_plan_path = _regular_file(args.screen_plan, "screen plan")
    replay_manifest_path = _regular_file(args.replay_manifest, "replay source manifest")
    controller_path = _regular_file(args.controller_checkpoint, "frozen controller")
    plan = load_screen_plan(screen_plan_path)
    _, replay_sources = validate_replay_manifest(replay_manifest_path, plan)
    split_contracts = {
        row["replicate_seed"]: row["replay_summary"]["split_contract"]
        for row in replay_sources
    }
    controller_hash = _check_digest(
        controller_path, args.controller_sha256, "frozen controller"
    )
    _validate_controller_contract(controller_path)
    if args.proxy_results is not None:
        raise A19PreparationError(
            "PROXY_EXECUTOR_NOT_IMPLEMENTED: measured finalize is disabled until "
            "an in-repository A19 trainer/evaluator recomputes raw validation predictions, "
            "policy KL, value MSE, parameter count, and FLOPs from versioned candidate artifacts"
        )
    expected_source_hashes = _source_hashes()
    expected_input_hashes = _input_hashes(
        screen_plan_path=screen_plan_path,
        replay_manifest_path=replay_manifest_path,
        controller_path=controller_path,
        replay_sources=replay_sources,
    )
    if output_dir.exists():
        _validate_existing_output(
            output_dir,
            run_id=run_id,
            plan=plan,
            expected_source_hashes=expected_source_hashes,
            expected_input_hashes=expected_input_hashes,
        )
        return 0
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        structural_rows = []
        for graph_seed in plan.graph_seeds:
            topology = generate_topology(graph_seed, plan.architecture)
            structural_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "axis_id": A19_AXIS_ID,
                    "role": ROLE,
                    "evidence_status": EVIDENCE_STATUS,
                    "claim_scope": CLAIM_SCOPE,
                    "graph_seed": graph_seed,
                    "replicate_seed": None,
                    "metric": "topology_contract",
                    "value": topology["topology_sha256"],
                    "topology_sha256": topology["topology_sha256"],
                    "resources": _resource_estimate(plan, topology),
                    "evidence_scope": "structural_readiness_only",
                }
            )
        resource_vectors = {
            canonical_json_bytes(row["resources"]) for row in structural_rows
        }
        if len(resource_vectors) != 1:
            raise A19PreparationError(
                "graph seeds do not have identical resource estimates"
            )
        topology_hashes = {row["topology_sha256"] for row in structural_rows}
        if len(topology_hashes) != len(structural_rows):
            raise A19PreparationError(
                "preregistered graph seeds produced duplicate topologies"
            )

        proxy_path: Path | None = None
        summaries: list[dict[str, Any]] = []
        rank_rows: list[dict[str, Any]] = []
        if args.proxy_results is not None:
            proxy_path = _regular_file(args.proxy_results, "proxy results")
            inputs = validate_inputs(
                controller_checkpoint=controller_path,
                expected_controller_sha256=controller_hash,
                replay_corpus=replay_manifest_path,
                expected_replay_corpus_sha256=file_sha256(replay_manifest_path),
                proxy_results=proxy_path,
                expected_proxy_results_sha256=args.proxy_results_sha256,
            )
            proxy_rows = validate_proxy_rows(
                load_jsonl_strict(proxy_path),
                plan=plan,
                inputs=inputs,
                split_contracts=split_contracts,
            )
            summaries, rank_rows = rank_graph_seeds(proxy_rows, plan)
            contract = build_ablation_contract(
                plan=plan, inputs=inputs, summaries=summaries
            )
            shortlist_status = "measured_proxy_shortlist"
            rows = [
                {
                    "schema_version": SCHEMA_VERSION,
                    "axis_id": A19_AXIS_ID,
                    "role": ROLE,
                    "evidence_status": EVIDENCE_STATUS,
                    "claim_scope": CLAIM_SCOPE,
                    "graph_seed": row["graph_seed"],
                    "replicate_seed": row["replicate_seed"],
                    "metric": "weighted_rank_percentile",
                    "value": row["weighted_rank_percentile"],
                    "metrics": row["metrics"],
                    "metric_ranks": row["metric_ranks"],
                    "topology_sha256": row["topology_sha256"],
                    "controller_sha256": row["controller_sha256"],
                    "replay_corpus_sha256": row["replay_corpus_sha256"],
                    "replay_source_sha256": row["replay_source_sha256"],
                    "source_checkpoint_sha256": row["source_checkpoint_sha256"],
                    "train_split_sha256": row["train_split_sha256"],
                    "validation_split_sha256": row["validation_split_sha256"],
                    "batch_schedule_sha256": row["batch_schedule_sha256"],
                    "evaluator_checkpoint": row["evaluator_checkpoint"],
                    "candidate_receipt": row["candidate_receipt"],
                    "budget": row["budget"],
                    "resources": row["resources"],
                    "evidence_scope": "fixed_replay_proxy_diagnostic_only",
                }
                for row in rank_rows
            ]
            _plot_measured(temp_dir / "diagnostic.png", summaries)
            interpretation = """# A19 graph-seed screen plot interpretation

- Category: **DIAGNOSTIC**
- Quantity: preregistered weighted paired-rank percentile across policy KL and value MSE; lower is better.
- Provenance: the plotted rows are bound to the frozen controller, real replay-source manifest, candidate checkpoint hashes, fixed budgets, and paired seeds 41/42/43.
- Interpretation: blue bars identify the deterministic proxy shortlist under the preregistered ranking rule.
- This does not show: play strength, Elo, production readiness, or superiority outside the fixed replay proxy.
- Next plot: paired fixed-evaluation and fixed-wall-clock comparisons for the complete frozen-controller ablation matrix.
"""
        else:
            shortlist_status = "not_measured"
            rows = structural_rows
            contract = {
                "schema_version": SCHEMA_VERSION,
                "axis_id": A19_AXIS_ID,
                "role": ROLE,
                "evidence_status": EVIDENCE_STATUS,
                "claim_scope": CLAIM_SCOPE,
                "execution_status": "ready_for_graph_seed_proxy_launch",
                "auto_promoted": False,
                "controller_contract": {
                    "mode": "frozen_exact_sha256",
                    "path": str(controller_path),
                    "sha256": controller_hash,
                    "mutation_allowed": False,
                },
                "replay_sources": replay_sources,
                "graph_seeds": list(plan.graph_seeds),
                "replicate_seeds": list(plan.replicate_seeds),
                "shortlisted_graph_seeds": [],
                "shortlist_status": shortlist_status,
                "budget_contract": dict(plan.budget_contract),
                "proxy_training": dict(plan.proxy_training),
                "proxy_trainer": {
                    "status": "absent_not_implemented",
                    "measured_finalize_enabled": False,
                    "external_candidate_receipts_are_sufficient": False,
                    "required_unblocker": "in-repository trainer/evaluator that persists and recomputes raw validation predictions and metrics",
                },
                "resource_basis": "symbolic_estimate_v1_not_measured",
                "variants": [dict(row) for row in plan.ablation_variants],
                "prohibited_inferences": list(plan.prohibited_inferences),
            }
            _plot_launch(temp_dir / "diagnostic.png", structural_rows)
            interpretation = """# A19 launch-readiness plot interpretation

- Category: **DIAGNOSTIC**
- Quantity: directed topology edge count for each preregistered graph seed.
- Provenance: topology generator v1, the tracked A19 screen plan, real replay inputs for paired seeds 41/42/43, and the frozen controller contract.
- Interpretation: every seed has exactly 110 edges and the full symbolic resource vector is identical, so topology density cannot confound the planned seed screen.
- This does not show: evaluator quality, a measured shortlist, play strength, Elo, or production readiness.
- Next plot: weighted paired-rank percentile after all 24 candidate proxy rows and checkpoint hashes pass the finalize gate.
"""

        write_jsonl(temp_dir / "rows.jsonl", rows)
        (temp_dir / "diagnostic_interpretation.md").write_text(
            interpretation, encoding="utf-8"
        )
        write_json(temp_dir / "a19_ablation_contract.v1.json", contract)
        write_json(
            temp_dir / "a19_split_contract.v1.json",
            {
                "schema_version": SCHEMA_VERSION,
                "axis_id": A19_AXIS_ID,
                "role": ROLE,
                "evidence_status": EVIDENCE_STATUS,
                "claim_scope": CLAIM_SCOPE,
                "split_method": plan.proxy_training["split_method"],
                "contracts": [split_contracts[seed] for seed in plan.replicate_seeds],
            },
        )
        shortlist = {
            "schema_version": SCHEMA_VERSION,
            "axis_id": A19_AXIS_ID,
            "role": ROLE,
            "claim_scope": CLAIM_SCOPE,
            "status": shortlist_status,
            "shortlisted_graph_seeds": [
                row["graph_seed"] for row in summaries if row.get("selected")
            ],
            "ranking": summaries,
            "promotion": dict(PROMOTION),
        }
        write_json(temp_dir / "a19_graph_seed_shortlist.v1.json", shortlist)
        mode = "measured_fixed_replay_proxy" if proxy_path else "launch_contract_only"
        summary = _base_summary(run_id=run_id, mode=mode)
        summary.update(
            {
                "outcome_detail": (
                    "MEASURED_PROXY_SHORTLIST_NO_PROMOTION"
                    if proxy_path
                    else "READY_FOR_PROXY_LAUNCH_NO_SHORTLIST"
                ),
                "shortlist_status": shortlist_status,
                "shortlisted_graph_seeds": shortlist["shortlisted_graph_seeds"],
                "graph_seed_count": len(plan.graph_seeds),
                "paired_replicate_seeds": list(plan.replicate_seeds),
                "real_replay_sources_validated": len(replay_sources),
                "resource_contract_equal_across_graph_seeds": True,
                "topology_hashes_unique": True,
                "diagnostic_category": "DIAGNOSTIC",
                "split_contracts_persisted": True,
                "proxy_trainer_status": "absent_not_implemented",
                "blockers": [
                    "A19 proxy candidate trainer/executor is not implemented; no measured shortlist exists."
                ]
                if proxy_path is None
                else [],
                "prohibited_inferences": list(plan.prohibited_inferences),
            }
        )
        write_json(temp_dir / "summary.json", summary)
        input_hashes = list(expected_input_hashes)
        artifact_paths = [
            temp_dir / "rows.jsonl",
            temp_dir / "summary.json",
            temp_dir / "diagnostic.png",
            temp_dir / "diagnostic_interpretation.md",
            temp_dir / "a19_ablation_contract.v1.json",
            temp_dir / "a19_split_contract.v1.json",
            temp_dir / "a19_graph_seed_shortlist.v1.json",
        ]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "axis_id": A19_AXIS_ID,
            "role": ROLE,
            "evidence_status": EVIDENCE_STATUS,
            "claim_scope": CLAIM_SCOPE,
            "execution_mode": mode,
            "status": "completed_no_promotion",
            "execution_status": "completed_no_promotion",
            "run_id": run_id,
            "completed_at": utc_now(),
            "source_hashes": list(expected_source_hashes),
            "input_hashes": input_hashes,
            "seed_contract": _seed_contract(plan),
            "runtime": {
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "platform": platform.platform(),
                "argv": list(sys.argv),
            },
            "promotion": dict(PROMOTION),
            "auto_promoted": False,
            "artifacts": [
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in artifact_paths
            ],
            "prohibited_inferences": list(plan.prohibited_inferences),
        }
        write_json(temp_dir / "run_manifest.json", manifest)
        os.replace(temp_dir, output_dir)
        return 0
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-plan", type=Path, required=True)
    parser.add_argument("--replay-manifest", type=Path, required=True)
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--controller-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--proxy-results", type=Path)
    parser.add_argument("--proxy-results-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.proxy_results is None) != (args.proxy_results_sha256 is None):
        raise A19PreparationError(
            "--proxy-results and --proxy-results-sha256 must be supplied together"
        )
    try:
        return run(args)
    except A19PreparationError as exc:
        print(f"A19 preparation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
