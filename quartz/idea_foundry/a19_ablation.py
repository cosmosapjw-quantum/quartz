"""Fail-closed preparation for the A19 graph-seed ablation.

This module does not train an evaluator and does not claim evaluator quality.
It turns preregistered, paired replay-proxy measurements into a deterministic
graph-seed shortlist while binding every row to the exact replay corpus,
controller contract, topology, and evaluator checkpoint that produced it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import re
from statistics import fmean, stdev
from typing import Any, Iterable, Mapping, Sequence


A19_SCREEN_SCHEMA_VERSION = 1
A19_AXIS_ID = "A19"
A19_CLAIM_SCOPE = "graph_seed_screen_preparation_only"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ALLOWED_METRICS = frozenset({"policy_kl", "value_mse", "latency_ms", "top1_accuracy"})
_RESOURCE_FIELDS = frozenset(
    {"parameters", "flops", "topology_edges", "nodes", "channels", "global_blocks"}
)


class A19PreparationError(ValueError):
    """Raised when an A19 preparation input violates the frozen contract."""


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: str
    weight: float


@dataclass(frozen=True)
class ScreenPlan:
    graph_seeds: tuple[int, ...]
    replicate_seeds: tuple[int, ...]
    shortlist_size: int
    metrics: tuple[MetricSpec, ...]
    budget_contract: Mapping[str, int]
    resource_fields: tuple[str, ...]
    resource_relative_tolerance: float
    architecture: Mapping[str, Any]
    proxy_training: Mapping[str, Any]
    ablation_variants: tuple[Mapping[str, Any], ...]
    prohibited_inferences: tuple[str, ...]


@dataclass(frozen=True)
class ScreenInputs:
    controller_checkpoint: Path
    controller_sha256: str
    replay_corpus: Path
    replay_corpus_sha256: str
    proxy_results: Path
    proxy_results_sha256: str
    replicate_replay_sha256: Mapping[int, str]
    replicate_checkpoint_sha256: Mapping[int, str]


def _reject_constant(value: str) -> None:
    raise A19PreparationError(f"non-finite JSON number is forbidden: {value}")


def _object_no_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise A19PreparationError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def load_json_strict(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise A19PreparationError(f"cannot read JSON input {path}: {exc}") from exc
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_object_no_duplicates,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise A19PreparationError(f"invalid JSON input {path}: {exc}") from exc


def load_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise A19PreparationError(f"cannot read JSONL input {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise A19PreparationError(f"blank JSONL row at {path}:{line_number}")
        try:
            row = json.loads(
                line,
                parse_constant=_reject_constant,
                object_pairs_hook=_object_no_duplicates,
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise A19PreparationError(
                f"invalid JSONL row at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise A19PreparationError(
                f"JSONL row must be an object at {path}:{line_number}"
            )
        rows.append(row)
    if not rows:
        raise A19PreparationError(f"proxy results contain no rows: {path}")
    return rows


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise A19PreparationError(f"cannot hash input {path}: {exc}") from exc
    return digest.hexdigest()


def _require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise A19PreparationError(f"{label} must not be a symlink: {path}")
    if not path.exists():
        raise A19PreparationError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise A19PreparationError(f"{label} is not a regular file: {path}")
    return path.resolve()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise A19PreparationError(f"{label} must be a lowercase 64-character SHA-256")
    return value


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise A19PreparationError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise A19PreparationError(f"{label} must be >= {minimum}")
    return value


def _require_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise A19PreparationError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise A19PreparationError(f"{label} must be a finite number")
    if minimum is not None and number < minimum:
        raise A19PreparationError(f"{label} must be >= {minimum}")
    return number


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise A19PreparationError(f"{label} must be a non-empty string")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise A19PreparationError(
            f"{label} keys mismatch: missing={missing}, extra={extra}"
        )


def load_screen_plan(path: Path) -> ScreenPlan:
    plan_path = _require_regular_file(path, "screen plan")
    raw = load_json_strict(plan_path)
    if not isinstance(raw, dict):
        raise A19PreparationError("screen plan must be a JSON object")
    required = {
        "schema_version",
        "axis_id",
        "claim_scope",
        "graph_seeds",
        "replicate_seeds",
        "shortlist_size",
        "metrics",
        "budget_contract",
        "resource_match",
        "architecture",
        "proxy_training",
        "ablation_variants",
        "prohibited_inferences",
    }
    _require_exact_keys(raw, required, "screen plan")
    if raw["schema_version"] != A19_SCREEN_SCHEMA_VERSION:
        raise A19PreparationError("unsupported A19 screen-plan schema_version")
    if raw["axis_id"] != A19_AXIS_ID:
        raise A19PreparationError("screen plan axis_id must be A19")
    if raw["claim_scope"] != A19_CLAIM_SCOPE:
        raise A19PreparationError(f"screen plan claim_scope must be {A19_CLAIM_SCOPE}")

    graph_seeds_raw = raw["graph_seeds"]
    replicate_seeds_raw = raw["replicate_seeds"]
    if not isinstance(graph_seeds_raw, list) or len(graph_seeds_raw) < 2:
        raise A19PreparationError(
            "graph_seeds must contain at least two preregistered seeds"
        )
    if not isinstance(replicate_seeds_raw, list) or len(replicate_seeds_raw) < 3:
        raise A19PreparationError(
            "replicate_seeds must contain at least three paired seeds"
        )
    graph_seeds = tuple(
        _require_int(seed, "graph_seed", minimum=0) for seed in graph_seeds_raw
    )
    replicate_seeds = tuple(
        _require_int(seed, "replicate_seed", minimum=0) for seed in replicate_seeds_raw
    )
    if len(set(graph_seeds)) != len(graph_seeds):
        raise A19PreparationError("graph_seeds must be unique")
    if len(set(replicate_seeds)) != len(replicate_seeds):
        raise A19PreparationError("replicate_seeds must be unique")

    shortlist_size = _require_int(raw["shortlist_size"], "shortlist_size", minimum=1)
    if shortlist_size >= len(graph_seeds):
        raise A19PreparationError(
            "shortlist_size must be smaller than graph seed count"
        )

    metric_rows = raw["metrics"]
    if not isinstance(metric_rows, list) or not metric_rows:
        raise A19PreparationError("metrics must be a non-empty array")
    metrics: list[MetricSpec] = []
    for index, metric in enumerate(metric_rows):
        if not isinstance(metric, dict):
            raise A19PreparationError(f"metrics[{index}] must be an object")
        _require_exact_keys(
            metric, {"name", "direction", "weight"}, f"metrics[{index}]"
        )
        name = _require_nonempty_string(metric["name"], f"metrics[{index}].name")
        direction = metric["direction"]
        if name not in _ALLOWED_METRICS:
            raise A19PreparationError(f"unsupported screening metric: {name}")
        if direction not in {"min", "max"}:
            raise A19PreparationError(f"metrics[{index}].direction must be min or max")
        weight = _require_number(
            metric["weight"], f"metrics[{index}].weight", minimum=0.0
        )
        if weight <= 0.0:
            raise A19PreparationError(f"metrics[{index}].weight must be > 0")
        metrics.append(MetricSpec(name=name, direction=direction, weight=weight))
    if len({metric.name for metric in metrics}) != len(metrics):
        raise A19PreparationError("screening metric names must be unique")
    if not math.isclose(sum(metric.weight for metric in metrics), 1.0, abs_tol=1e-12):
        raise A19PreparationError(
            "screening metric weights must sum to exactly 1 within 1e-12"
        )

    budget_raw = raw["budget_contract"]
    if not isinstance(budget_raw, dict) or not budget_raw:
        raise A19PreparationError("budget_contract must be a non-empty object")
    budget_contract = {
        _require_nonempty_string(key, "budget_contract key"): _require_int(
            value, f"budget_contract.{key}", minimum=1
        )
        for key, value in budget_raw.items()
    }

    resource = raw["resource_match"]
    if not isinstance(resource, dict):
        raise A19PreparationError("resource_match must be an object")
    _require_exact_keys(resource, {"fields", "relative_tolerance"}, "resource_match")
    resource_fields_raw = resource["fields"]
    if not isinstance(resource_fields_raw, list) or not resource_fields_raw:
        raise A19PreparationError("resource_match.fields must be a non-empty array")
    resource_fields = tuple(
        _require_nonempty_string(field, "resource field")
        for field in resource_fields_raw
    )
    if len(set(resource_fields)) != len(resource_fields):
        raise A19PreparationError("resource_match.fields must be unique")
    if not set(resource_fields).issubset(_RESOURCE_FIELDS):
        raise A19PreparationError("resource_match contains an unsupported field")
    resource_tolerance = _require_number(
        resource["relative_tolerance"], "resource_match.relative_tolerance", minimum=0.0
    )
    if resource_tolerance > 0.1:
        raise A19PreparationError("resource_match.relative_tolerance must be <= 0.1")

    architecture = raw["architecture"]
    if not isinstance(architecture, dict):
        raise A19PreparationError("architecture must be an object")
    expected_architecture = {
        "nodes": 40,
        "channels": 144,
        "cells": 2,
        "nodes_per_cell": 20,
        "max_in_degree": 3,
        "global_blocks": 2,
        "node_op": "one_conv_pre_activation_residual",
        "routing": "soft_train_then_static_prune",
    }
    if architecture != expected_architecture:
        raise A19PreparationError(
            "architecture must exactly match the A19 RW-ResT-Lite v1 contract"
        )

    proxy_training = raw["proxy_training"]
    if not isinstance(proxy_training, dict):
        raise A19PreparationError("proxy_training must be an object")
    expected_training_keys = {
        "optimizer",
        "learning_rate",
        "weight_decay",
        "deterministic_algorithms",
        "split_method",
        "required_device_type",
    }
    _require_exact_keys(proxy_training, expected_training_keys, "proxy_training")
    if proxy_training["optimizer"] != "adamw":
        raise A19PreparationError("proxy_training.optimizer must be adamw")
    _require_number(
        proxy_training["learning_rate"], "proxy_training.learning_rate", minimum=0.0
    )
    if float(proxy_training["learning_rate"]) <= 0.0:
        raise A19PreparationError("proxy_training.learning_rate must be > 0")
    _require_number(
        proxy_training["weight_decay"], "proxy_training.weight_decay", minimum=0.0
    )
    if proxy_training["deterministic_algorithms"] is not True:
        raise A19PreparationError("proxy training must enable deterministic algorithms")
    if proxy_training["split_method"] != "deduplicated_state_hash_v1":
        raise A19PreparationError(
            "proxy_training.split_method must be deduplicated_state_hash_v1"
        )
    if proxy_training["required_device_type"] != "cuda":
        raise A19PreparationError("proxy_training.required_device_type must be cuda")

    variants_raw = raw["ablation_variants"]
    if not isinstance(variants_raw, list) or len(variants_raw) < 2:
        raise A19PreparationError(
            "ablation_variants must contain at least two variants"
        )
    variants: list[Mapping[str, Any]] = []
    variant_ids: set[str] = set()
    for index, variant in enumerate(variants_raw):
        if not isinstance(variant, dict):
            raise A19PreparationError(f"ablation_variants[{index}] must be an object")
        _require_exact_keys(
            variant,
            {"id", "graph_seed_policy", "parameter_match_class", "purpose"},
            f"ablation_variants[{index}]",
        )
        variant_id = _require_nonempty_string(
            variant["id"], f"ablation_variants[{index}].id"
        )
        if variant_id in variant_ids:
            raise A19PreparationError("ablation variant ids must be unique")
        variant_ids.add(variant_id)
        if variant["graph_seed_policy"] not in {"none", "shortlisted"}:
            raise A19PreparationError(
                "variant graph_seed_policy must be none or shortlisted"
            )
        _require_nonempty_string(
            variant["parameter_match_class"],
            f"ablation_variants[{index}].parameter_match_class",
        )
        _require_nonempty_string(
            variant["purpose"], f"ablation_variants[{index}].purpose"
        )
        variants.append(dict(variant))
    required_variants = {
        "resnet_fcn",
        "static_ws_lite",
        "learnable_static_weights",
        "soft_routing",
        "attention_only",
        "combined",
        "heavier_upper_bound",
    }
    if variant_ids != required_variants:
        raise A19PreparationError(
            f"ablation variant ids mismatch: expected={sorted(required_variants)}"
        )

    prohibited_raw = raw["prohibited_inferences"]
    if not isinstance(prohibited_raw, list) or not prohibited_raw:
        raise A19PreparationError("prohibited_inferences must be a non-empty array")
    prohibited = tuple(
        _require_nonempty_string(value, "prohibited inference")
        for value in prohibited_raw
    )

    return ScreenPlan(
        graph_seeds=graph_seeds,
        replicate_seeds=replicate_seeds,
        shortlist_size=shortlist_size,
        metrics=tuple(metrics),
        budget_contract=budget_contract,
        resource_fields=resource_fields,
        resource_relative_tolerance=resource_tolerance,
        architecture=architecture,
        proxy_training=dict(proxy_training),
        ablation_variants=tuple(variants),
        prohibited_inferences=prohibited,
    )


def validate_inputs(
    *,
    controller_checkpoint: Path,
    expected_controller_sha256: str,
    replay_corpus: Path,
    expected_replay_corpus_sha256: str,
    proxy_results: Path,
    expected_proxy_results_sha256: str,
) -> ScreenInputs:
    controller = _require_regular_file(controller_checkpoint, "controller checkpoint")
    replay = _require_regular_file(replay_corpus, "replay corpus")
    proxy = _require_regular_file(proxy_results, "proxy results")
    expected_controller = _require_sha256(
        expected_controller_sha256, "controller SHA-256"
    )
    expected_replay = _require_sha256(
        expected_replay_corpus_sha256, "replay corpus SHA-256"
    )
    expected_proxy = _require_sha256(
        expected_proxy_results_sha256, "proxy results SHA-256"
    )
    actual_controller = file_sha256(controller)
    actual_replay = file_sha256(replay)
    actual_proxy = file_sha256(proxy)
    mismatches = []
    for label, expected, actual in (
        ("controller checkpoint", expected_controller, actual_controller),
        ("replay corpus", expected_replay, actual_replay),
        ("proxy results", expected_proxy, actual_proxy),
    ):
        if expected != actual:
            mismatches.append(f"{label}: expected {expected}, got {actual}")
    if mismatches:
        raise A19PreparationError("input hash mismatch: " + "; ".join(mismatches))
    replay_manifest = load_json_strict(replay)
    if (
        not isinstance(replay_manifest, dict)
        or replay_manifest.get("axis_id") != A19_AXIS_ID
    ):
        raise A19PreparationError("replay corpus must be an A19 replay-source manifest")
    source_rows = replay_manifest.get("sources")
    if not isinstance(source_rows, list) or not source_rows:
        raise A19PreparationError("A19 replay-source manifest contains no sources")
    replicate_replay_sha256: dict[int, str] = {}
    replicate_checkpoint_sha256: dict[int, str] = {}
    for index, source in enumerate(source_rows):
        if not isinstance(source, dict):
            raise A19PreparationError(f"replay source {index} must be an object")
        seed = _require_int(
            source.get("replicate_seed"), f"replay source {index}.replicate_seed"
        )
        if seed in replicate_replay_sha256:
            raise A19PreparationError(f"duplicate replay-source replicate seed: {seed}")
        replicate_replay_sha256[seed] = _require_sha256(
            source.get("replay_sha256"), f"replay source {index}.replay_sha256"
        )
        replicate_checkpoint_sha256[seed] = _require_sha256(
            source.get("checkpoint_sha256"),
            f"replay source {index}.checkpoint_sha256",
        )
    return ScreenInputs(
        controller_checkpoint=controller,
        controller_sha256=actual_controller,
        replay_corpus=replay,
        replay_corpus_sha256=actual_replay,
        proxy_results=proxy,
        proxy_results_sha256=actual_proxy,
        replicate_replay_sha256=replicate_replay_sha256,
        replicate_checkpoint_sha256=replicate_checkpoint_sha256,
    )


def generate_topology(
    graph_seed: int, architecture: Mapping[str, Any]
) -> dict[str, Any]:
    """Generate the deterministic degree-capped DAG bound to ``graph_seed``.

    Each 20-node cell has a mandatory chain.  Every later node adds up to two
    earlier parents, sampled without replacement with a local-distance bias.
    This changes wiring while keeping the node operation and edge-count rule
    fixed across seeds.
    """

    graph_seed = _require_int(graph_seed, "graph_seed", minimum=0)
    cells = int(architecture["cells"])
    nodes_per_cell = int(architecture["nodes_per_cell"])
    max_in_degree = int(architecture["max_in_degree"])
    edges: list[dict[str, Any]] = []
    for cell in range(cells):
        rng = random.Random((graph_seed << 16) ^ (cell * 0x9E3779B1) ^ 0xA19)
        offset = cell * nodes_per_cell
        edges.append(
            {
                "cell": cell,
                "src": -1,
                "src_kind": "cell_input",
                "dst": offset,
                "mandatory_chain": True,
            }
        )
        for local_dst in range(1, nodes_per_cell):
            chain_parent = local_dst - 1
            parents = [chain_parent]
            candidates = [
                candidate
                for candidate in range(local_dst - 1)
                if candidate != chain_parent
            ]
            extra_count = min(max_in_degree - 1, len(candidates))
            for _ in range(extra_count):
                weights = [1.0 / (local_dst - candidate) for candidate in candidates]
                selected = rng.choices(candidates, weights=weights, k=1)[0]
                parents.append(selected)
                candidates.remove(selected)
            for parent in sorted(parents):
                edges.append(
                    {
                        "cell": cell,
                        "src": offset + parent,
                        "src_kind": "node",
                        "dst": offset + local_dst,
                        "mandatory_chain": parent == chain_parent,
                    }
                )
    payload = {
        "schema_version": A19_SCREEN_SCHEMA_VERSION,
        "axis_id": A19_AXIS_ID,
        "graph_seed": graph_seed,
        "architecture": dict(architecture),
        "edges": edges,
    }
    payload["topology_sha256"] = sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def _resolve_checkpoint_path(raw_path: str, proxy_results: Path) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = proxy_results.parent / candidate
    return _require_regular_file(candidate, "evaluator checkpoint")


def _resolve_proxy_sidecar(raw_path: str, proxy_results: Path, label: str) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = proxy_results.parent / candidate
    return _require_regular_file(candidate, label)


def _object_sha256(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _validate_deterministic_runtime(
    runtime: Any, *, required_device_type: str, label: str
) -> dict[str, Any]:
    expected_keys = {
        "python_version",
        "torch_version",
        "device_type",
        "device_name",
        "cuda_version",
        "cudnn_version",
        "deterministic_algorithms",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cublas_workspace_config",
    }
    if not isinstance(runtime, dict):
        raise A19PreparationError(f"{label} must be an object")
    _require_exact_keys(runtime, expected_keys, label)
    for key in (
        "python_version",
        "torch_version",
        "device_type",
        "device_name",
        "cuda_version",
        "cudnn_version",
    ):
        _require_nonempty_string(runtime[key], f"{label}.{key}")
    if runtime["device_type"] != required_device_type:
        raise A19PreparationError(f"{label}.device_type must be {required_device_type}")
    if (
        required_device_type == "cuda"
        and "NVIDIA" not in runtime["device_name"].upper()
    ):
        raise A19PreparationError(f"{label}.device_name must identify an NVIDIA device")
    if runtime["deterministic_algorithms"] is not True:
        raise A19PreparationError(f"{label}.deterministic_algorithms must be true")
    if runtime["cudnn_benchmark"] is not False:
        raise A19PreparationError(f"{label}.cudnn_benchmark must be false")
    if runtime["cudnn_deterministic"] is not True:
        raise A19PreparationError(f"{label}.cudnn_deterministic must be true")
    if runtime["cublas_workspace_config"] not in {":4096:8", ":16:8"}:
        raise A19PreparationError(
            f"{label}.cublas_workspace_config is not a deterministic CUDA setting"
        )
    return dict(runtime)


def _load_candidate_checkpoint_metadata(
    checkpoint_path: Path,
    *,
    expected_metadata: Mapping[str, Any],
    expected_parameter_count: int,
    label: str,
) -> None:
    try:
        import torch

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise A19PreparationError(
            f"{label} is not a safe versioned Torch candidate checkpoint: {exc}"
        ) from exc
    expected_keys = {
        "schema_version",
        "axis_id",
        "artifact_kind",
        "metadata",
        "model_state_dict",
        "trainable_parameter_names",
    }
    if not isinstance(payload, dict):
        raise A19PreparationError(f"{label} checkpoint payload must be an object")
    _require_exact_keys(payload, expected_keys, f"{label} checkpoint")
    if payload["schema_version"] != A19_SCREEN_SCHEMA_VERSION:
        raise A19PreparationError(f"{label} checkpoint schema_version is unsupported")
    if payload["axis_id"] != A19_AXIS_ID:
        raise A19PreparationError(f"{label} checkpoint axis_id must be A19")
    if payload["artifact_kind"] != "a19_proxy_candidate_checkpoint":
        raise A19PreparationError(f"{label} checkpoint artifact_kind is invalid")
    if payload["metadata"] != dict(expected_metadata):
        raise A19PreparationError(f"{label} embedded checkpoint metadata drift")
    state_dict = payload["model_state_dict"]
    if not isinstance(state_dict, dict) or not state_dict:
        raise A19PreparationError(f"{label} model_state_dict must be non-empty")
    if not all(
        isinstance(name, str) and torch.is_tensor(value)
        for name, value in state_dict.items()
    ):
        raise A19PreparationError(
            f"{label} model_state_dict must contain only named tensors"
        )
    trainable_names = payload["trainable_parameter_names"]
    if (
        not isinstance(trainable_names, list)
        or not trainable_names
        or len(set(trainable_names)) != len(trainable_names)
        or any(name not in state_dict for name in trainable_names)
    ):
        raise A19PreparationError(
            f"{label} trainable_parameter_names must uniquely reference state tensors"
        )
    trainable_tensors = [state_dict[name] for name in trainable_names]
    if any(tensor.device.type == "meta" for tensor in trainable_tensors):
        raise A19PreparationError(f"{label} trainable tensors must contain real data")
    actual_parameter_count = sum(int(tensor.numel()) for tensor in trainable_tensors)
    if actual_parameter_count != expected_parameter_count:
        raise A19PreparationError(
            f"{label} recomputed trainable parameter count drift: "
            f"expected {expected_parameter_count}, got {actual_parameter_count}"
        )


def _validate_operator_trace(
    trace_path: Path,
    *,
    expected_sha256: str,
    graph_seed: int,
    replicate_seed: int,
    topology_sha256: str,
    expected_flops: int,
    label: str,
) -> None:
    if file_sha256(trace_path) != expected_sha256:
        raise A19PreparationError(f"{label} operator trace hash mismatch")
    trace = load_json_strict(trace_path)
    expected_keys = {
        "schema_version",
        "axis_id",
        "artifact_kind",
        "graph_seed",
        "replicate_seed",
        "topology_sha256",
        "operations",
        "total_flops",
    }
    if not isinstance(trace, dict):
        raise A19PreparationError(f"{label} operator trace must be an object")
    _require_exact_keys(trace, expected_keys, f"{label} operator trace")
    if (
        trace["schema_version"] != A19_SCREEN_SCHEMA_VERSION
        or trace["axis_id"] != A19_AXIS_ID
        or trace["artifact_kind"] != "a19_operator_trace"
        or trace["graph_seed"] != graph_seed
        or trace["replicate_seed"] != replicate_seed
        or trace["topology_sha256"] != topology_sha256
    ):
        raise A19PreparationError(f"{label} operator trace identity drift")
    operations = trace["operations"]
    if not isinstance(operations, list) or not operations:
        raise A19PreparationError(f"{label} operator trace contains no operations")
    recomputed = 0
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            raise A19PreparationError(f"{label} operation {index} must be an object")
        _require_exact_keys(operation, {"name", "flops"}, f"{label} operation {index}")
        _require_nonempty_string(operation["name"], f"{label} operation {index}.name")
        recomputed += _require_int(
            operation["flops"], f"{label} operation {index}.flops", minimum=1
        )
    total = _require_int(trace["total_flops"], f"{label}.total_flops", minimum=1)
    if total != recomputed or total != expected_flops:
        raise A19PreparationError(
            f"{label} recomputed operator FLOPs drift: trace={total}, "
            f"sum={recomputed}, expected={expected_flops}"
        )


def _validate_candidate_receipt(
    receipt_path: Path,
    *,
    expected_receipt_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    graph_seed: int,
    replicate_seed: int,
    topology_sha256: str,
    controller_sha256: str,
    replay_manifest_sha256: str,
    replay_source_sha256: str,
    source_checkpoint_sha256: str,
    split_contract: Mapping[str, Any],
    budget: Mapping[str, int],
    optimizer_contract: Mapping[str, Any],
    resources: Mapping[str, float],
    required_device_type: str,
    label: str,
) -> dict[str, Any]:
    expected_receipt_sha256 = _require_sha256(
        expected_receipt_sha256, f"{label}.receipt_sha256"
    )
    actual_receipt_sha256 = file_sha256(receipt_path)
    if actual_receipt_sha256 != expected_receipt_sha256:
        raise A19PreparationError(f"{label} candidate receipt hash mismatch")
    receipt = load_json_strict(receipt_path)
    expected_keys = {
        "schema_version",
        "axis_id",
        "artifact_kind",
        "graph_seed",
        "replicate_seed",
        "topology_sha256",
        "controller_sha256",
        "replay_manifest_sha256",
        "replay_source_sha256",
        "source_checkpoint_sha256",
        "train_split_sha256",
        "validation_split_sha256",
        "batch_schedule_sha256",
        "checkpoint",
        "budget",
        "optimizer_contract",
        "deterministic_runtime",
        "resources",
        "resource_provenance",
    }
    if not isinstance(receipt, dict):
        raise A19PreparationError(f"{label} candidate receipt must be an object")
    _require_exact_keys(receipt, expected_keys, f"{label} candidate receipt")
    expected_values = {
        "schema_version": A19_SCREEN_SCHEMA_VERSION,
        "axis_id": A19_AXIS_ID,
        "artifact_kind": "a19_proxy_candidate_receipt",
        "graph_seed": graph_seed,
        "replicate_seed": replicate_seed,
        "topology_sha256": topology_sha256,
        "controller_sha256": controller_sha256,
        "replay_manifest_sha256": replay_manifest_sha256,
        "replay_source_sha256": replay_source_sha256,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "train_split_sha256": split_contract["train_split_sha256"],
        "validation_split_sha256": split_contract["validation_split_sha256"],
        "batch_schedule_sha256": split_contract["batch_schedule_sha256"],
    }
    for key, expected in expected_values.items():
        if receipt[key] != expected:
            raise A19PreparationError(f"{label} candidate receipt {key} drift")
    if receipt["checkpoint"] != {
        "path": str(checkpoint_path),
        "sha256": checkpoint_sha256,
    }:
        raise A19PreparationError(f"{label} candidate receipt checkpoint drift")
    if receipt["budget"] != dict(budget):
        raise A19PreparationError(f"{label} candidate receipt budget drift")
    if receipt["optimizer_contract"] != dict(optimizer_contract):
        raise A19PreparationError(f"{label} candidate receipt optimizer drift")
    if receipt["resources"] != dict(resources):
        raise A19PreparationError(f"{label} candidate receipt resources drift")
    resource_provenance = receipt["resource_provenance"]
    if not isinstance(resource_provenance, dict):
        raise A19PreparationError(f"{label} resource_provenance must be an object")
    _require_exact_keys(
        resource_provenance,
        {
            "parameter_count_method",
            "flop_count_method",
            "operator_trace_path",
            "operator_trace_sha256",
        },
        f"{label}.resource_provenance",
    )
    if resource_provenance["parameter_count_method"] != "sum_trainable_parameters_v1":
        raise A19PreparationError(f"{label} parameter-count method is unsupported")
    if resource_provenance["flop_count_method"] != "operator_trace_v1":
        raise A19PreparationError(f"{label} FLOP-count method is unsupported")
    operator_trace_hash = _require_sha256(
        resource_provenance["operator_trace_sha256"],
        f"{label}.resource_provenance.operator_trace_sha256",
    )
    operator_trace_path = _resolve_proxy_sidecar(
        _require_nonempty_string(
            resource_provenance["operator_trace_path"],
            f"{label}.resource_provenance.operator_trace_path",
        ),
        receipt_path,
        "operator trace",
    )
    parameters = _require_int(
        resources.get("parameters"), f"{label}.resources.parameters", minimum=1
    )
    flops = _require_int(resources.get("flops"), f"{label}.resources.flops", minimum=1)
    _validate_operator_trace(
        operator_trace_path,
        expected_sha256=operator_trace_hash,
        graph_seed=graph_seed,
        replicate_seed=replicate_seed,
        topology_sha256=topology_sha256,
        expected_flops=flops,
        label=label,
    )
    runtime = _validate_deterministic_runtime(
        receipt["deterministic_runtime"],
        required_device_type=required_device_type,
        label=f"{label}.deterministic_runtime",
    )
    embedded_metadata = {
        "graph_seed": graph_seed,
        "replicate_seed": replicate_seed,
        "topology_sha256": topology_sha256,
        "controller_sha256": controller_sha256,
        "replay_source_sha256": replay_source_sha256,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "train_split_sha256": split_contract["train_split_sha256"],
        "validation_split_sha256": split_contract["validation_split_sha256"],
        "batch_schedule_sha256": split_contract["batch_schedule_sha256"],
        "budget_sha256": _object_sha256(dict(budget)),
        "optimizer_contract_sha256": _object_sha256(dict(optimizer_contract)),
        "resources_sha256": _object_sha256(dict(resources)),
        "runtime_sha256": _object_sha256(runtime),
    }
    _load_candidate_checkpoint_metadata(
        checkpoint_path,
        expected_metadata=embedded_metadata,
        expected_parameter_count=parameters,
        label=label,
    )
    return {
        "path": str(receipt_path),
        "sha256": actual_receipt_sha256,
        "deterministic_runtime": runtime,
        "resource_provenance": dict(resource_provenance),
    }


def validate_proxy_rows(
    rows: Sequence[dict[str, Any]],
    *,
    plan: ScreenPlan,
    inputs: ScreenInputs,
    split_contracts: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected_row_keys = {
        "schema_version",
        "axis_id",
        "graph_seed",
        "replicate_seed",
        "topology_sha256",
        "controller_sha256",
        "replay_corpus_sha256",
        "replay_source_sha256",
        "source_checkpoint_sha256",
        "train_split_sha256",
        "validation_split_sha256",
        "batch_schedule_sha256",
        "evaluator_checkpoint",
        "budget",
        "metrics",
        "resources",
    }
    expected_pairs = {
        (graph_seed, replicate_seed)
        for graph_seed in plan.graph_seeds
        for replicate_seed in plan.replicate_seeds
    }
    seen_pairs: set[tuple[int, int]] = set()
    normalized: list[dict[str, Any]] = []
    checkpoint_hash_to_pair: dict[str, tuple[int, int]] = {}
    for index, row in enumerate(rows):
        label = f"proxy row {index + 1}"
        _require_exact_keys(row, expected_row_keys, label)
        if row["schema_version"] != A19_SCREEN_SCHEMA_VERSION:
            raise A19PreparationError(f"{label} schema_version is unsupported")
        if row["axis_id"] != A19_AXIS_ID:
            raise A19PreparationError(f"{label} axis_id must be A19")
        graph_seed = _require_int(row["graph_seed"], f"{label}.graph_seed", minimum=0)
        replicate_seed = _require_int(
            row["replicate_seed"], f"{label}.replicate_seed", minimum=0
        )
        pair = (graph_seed, replicate_seed)
        if pair not in expected_pairs:
            raise A19PreparationError(f"{label} is not preregistered: pair={pair}")
        if pair in seen_pairs:
            raise A19PreparationError(f"duplicate graph/replicate pair: {pair}")
        seen_pairs.add(pair)

        topology = generate_topology(graph_seed, plan.architecture)
        topology_hash = _require_sha256(
            row["topology_sha256"], f"{label}.topology_sha256"
        )
        if topology_hash != topology["topology_sha256"]:
            raise A19PreparationError(
                f"{label} topology hash does not match graph_seed"
            )
        controller_hash = _require_sha256(
            row["controller_sha256"], f"{label}.controller_sha256"
        )
        replay_hash = _require_sha256(
            row["replay_corpus_sha256"], f"{label}.replay_corpus_sha256"
        )
        if controller_hash != inputs.controller_sha256:
            raise A19PreparationError(
                f"{label} was measured with a different controller"
            )
        if replay_hash != inputs.replay_corpus_sha256:
            raise A19PreparationError(
                f"{label} was measured with a different replay corpus"
            )
        replay_source_hash = _require_sha256(
            row["replay_source_sha256"], f"{label}.replay_source_sha256"
        )
        source_checkpoint_hash = _require_sha256(
            row["source_checkpoint_sha256"], f"{label}.source_checkpoint_sha256"
        )
        if replay_source_hash != inputs.replicate_replay_sha256.get(replicate_seed):
            raise A19PreparationError(
                f"{label} was measured with the wrong paired replay source"
            )
        if source_checkpoint_hash != inputs.replicate_checkpoint_sha256.get(
            replicate_seed
        ):
            raise A19PreparationError(
                f"{label} was measured with the wrong paired source checkpoint"
            )
        split_contract = split_contracts.get(replicate_seed)
        if not isinstance(split_contract, Mapping):
            raise A19PreparationError(
                f"{label} has no persisted train/validation split contract"
            )
        for split_key in (
            "train_split_sha256",
            "validation_split_sha256",
            "batch_schedule_sha256",
        ):
            row_split_hash = _require_sha256(row[split_key], f"{label}.{split_key}")
            if row_split_hash != split_contract.get(split_key):
                raise A19PreparationError(f"{label} {split_key} drift")

        checkpoint = row["evaluator_checkpoint"]
        if not isinstance(checkpoint, dict):
            raise A19PreparationError(f"{label}.evaluator_checkpoint must be an object")
        _require_exact_keys(
            checkpoint,
            {"path", "sha256", "receipt_path", "receipt_sha256"},
            f"{label}.evaluator_checkpoint",
        )
        checkpoint_path = _resolve_checkpoint_path(
            _require_nonempty_string(
                checkpoint["path"], f"{label}.evaluator_checkpoint.path"
            ),
            inputs.proxy_results,
        )
        checkpoint_hash = _require_sha256(
            checkpoint["sha256"], f"{label}.evaluator_checkpoint.sha256"
        )
        actual_checkpoint_hash = file_sha256(checkpoint_path)
        if checkpoint_hash != actual_checkpoint_hash:
            raise A19PreparationError(f"{label} evaluator checkpoint hash mismatch")
        other_pair = checkpoint_hash_to_pair.get(checkpoint_hash)
        if other_pair is not None and other_pair != pair:
            raise A19PreparationError(
                "one evaluator checkpoint hash cannot represent multiple graph/replicate pairs"
            )
        checkpoint_hash_to_pair[checkpoint_hash] = pair
        receipt_path = _resolve_proxy_sidecar(
            _require_nonempty_string(
                checkpoint["receipt_path"],
                f"{label}.evaluator_checkpoint.receipt_path",
            ),
            inputs.proxy_results,
            "candidate receipt",
        )
        receipt_hash = _require_sha256(
            checkpoint["receipt_sha256"],
            f"{label}.evaluator_checkpoint.receipt_sha256",
        )

        budget = row["budget"]
        if not isinstance(budget, dict):
            raise A19PreparationError(f"{label}.budget must be an object")
        _require_exact_keys(budget, set(plan.budget_contract), f"{label}.budget")
        normalized_budget = {
            key: _require_int(value, f"{label}.budget.{key}", minimum=1)
            for key, value in budget.items()
        }
        if normalized_budget != dict(plan.budget_contract):
            raise A19PreparationError(
                f"{label} budget differs from preregistered budget"
            )

        metrics_raw = row["metrics"]
        if not isinstance(metrics_raw, dict):
            raise A19PreparationError(f"{label}.metrics must be an object")
        expected_metric_names = {metric.name for metric in plan.metrics}
        _require_exact_keys(metrics_raw, expected_metric_names, f"{label}.metrics")
        metrics = {
            name: _require_number(value, f"{label}.metrics.{name}", minimum=0.0)
            for name, value in metrics_raw.items()
        }
        if "top1_accuracy" in metrics and metrics["top1_accuracy"] > 1.0:
            raise A19PreparationError(f"{label}.metrics.top1_accuracy must be <= 1")

        resources_raw = row["resources"]
        if not isinstance(resources_raw, dict):
            raise A19PreparationError(f"{label}.resources must be an object")
        _require_exact_keys(
            resources_raw, set(plan.resource_fields), f"{label}.resources"
        )
        resources = {
            name: _require_number(value, f"{label}.resources.{name}", minimum=1.0)
            for name, value in resources_raw.items()
        }
        structural_expected = {
            "topology_edges": float(len(topology["edges"])),
            "nodes": float(plan.architecture["nodes"]),
            "channels": float(plan.architecture["channels"]),
            "global_blocks": float(plan.architecture["global_blocks"]),
        }
        for name, expected in structural_expected.items():
            if name in resources and resources[name] != expected:
                raise A19PreparationError(
                    f"{label}.resources.{name} does not match its topology contract"
                )

        receipt_identity = _validate_candidate_receipt(
            receipt_path,
            expected_receipt_sha256=receipt_hash,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_hash,
            graph_seed=graph_seed,
            replicate_seed=replicate_seed,
            topology_sha256=topology_hash,
            controller_sha256=controller_hash,
            replay_manifest_sha256=replay_hash,
            replay_source_sha256=replay_source_hash,
            source_checkpoint_sha256=source_checkpoint_hash,
            split_contract=split_contract,
            budget=normalized_budget,
            optimizer_contract=plan.proxy_training,
            resources=resources,
            required_device_type=str(plan.proxy_training["required_device_type"]),
            label=label,
        )

        normalized.append(
            {
                "schema_version": A19_SCREEN_SCHEMA_VERSION,
                "axis_id": A19_AXIS_ID,
                "graph_seed": graph_seed,
                "replicate_seed": replicate_seed,
                "topology_sha256": topology_hash,
                "controller_sha256": controller_hash,
                "replay_corpus_sha256": replay_hash,
                "replay_source_sha256": replay_source_hash,
                "source_checkpoint_sha256": source_checkpoint_hash,
                "train_split_sha256": split_contract["train_split_sha256"],
                "validation_split_sha256": split_contract["validation_split_sha256"],
                "batch_schedule_sha256": split_contract["batch_schedule_sha256"],
                "evaluator_checkpoint": {
                    "path": str(checkpoint_path),
                    "sha256": checkpoint_hash,
                    "receipt_path": str(receipt_path),
                    "receipt_sha256": receipt_hash,
                },
                "candidate_receipt": receipt_identity,
                "budget": normalized_budget,
                "metrics": metrics,
                "resources": resources,
            }
        )
    missing = sorted(expected_pairs - seen_pairs)
    if missing:
        raise A19PreparationError(
            f"paired graph/replicate coverage is incomplete: missing={missing}"
        )
    if seen_pairs != expected_pairs:
        raise A19PreparationError(
            "proxy row coverage does not exactly match the preregistration"
        )

    for resource_name in plan.resource_fields:
        values = [row["resources"][resource_name] for row in normalized]
        reference = min(values)
        relative_span = (max(values) - reference) / reference
        if relative_span > plan.resource_relative_tolerance + 1e-15:
            raise A19PreparationError(
                f"{resource_name} relative span {relative_span:.6g} exceeds "
                f"tolerance {plan.resource_relative_tolerance:.6g}"
            )
    return sorted(
        normalized, key=lambda row: (row["graph_seed"], row["replicate_seed"])
    )


def _average_tied_ranks(
    values: Mapping[int, float], *, direction: str
) -> dict[int, float]:
    ordered = sorted(
        values.items(), key=lambda item: (item[1], item[0]), reverse=direction == "max"
    )
    ranks: dict[int, float] = {}
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        for graph_seed, _ in ordered[cursor:end]:
            ranks[graph_seed] = average_rank
        cursor = end
    return ranks


def rank_graph_seeds(
    rows: Sequence[dict[str, Any]], plan: ScreenPlan
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_pair = {(row["graph_seed"], row["replicate_seed"]): row for row in rows}
    n_graphs = len(plan.graph_seeds)
    rank_rows: list[dict[str, Any]] = []
    replicate_scores: dict[int, list[float]] = defaultdict(list)
    per_seed_metric_values: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for replicate_seed in plan.replicate_seeds:
        metric_ranks: dict[str, dict[int, float]] = {}
        for metric in plan.metrics:
            values = {
                graph_seed: by_pair[(graph_seed, replicate_seed)]["metrics"][
                    metric.name
                ]
                for graph_seed in plan.graph_seeds
            }
            metric_ranks[metric.name] = _average_tied_ranks(
                values, direction=metric.direction
            )
        for graph_seed in plan.graph_seeds:
            source_row = by_pair[(graph_seed, replicate_seed)]
            weighted_percentile = sum(
                metric.weight
                * ((metric_ranks[metric.name][graph_seed] - 1.0) / max(1, n_graphs - 1))
                for metric in plan.metrics
            )
            replicate_scores[graph_seed].append(weighted_percentile)
            for metric in plan.metrics:
                per_seed_metric_values[graph_seed][metric.name].append(
                    by_pair[(graph_seed, replicate_seed)]["metrics"][metric.name]
                )
            rank_rows.append(
                {
                    "schema_version": A19_SCREEN_SCHEMA_VERSION,
                    "axis_id": A19_AXIS_ID,
                    "graph_seed": graph_seed,
                    "replicate_seed": replicate_seed,
                    "weighted_rank_percentile": weighted_percentile,
                    "metric_ranks": {
                        metric.name: metric_ranks[metric.name][graph_seed]
                        for metric in plan.metrics
                    },
                    "metrics": dict(by_pair[(graph_seed, replicate_seed)]["metrics"]),
                    "topology_sha256": source_row["topology_sha256"],
                    "controller_sha256": source_row["controller_sha256"],
                    "replay_corpus_sha256": source_row["replay_corpus_sha256"],
                    "replay_source_sha256": source_row["replay_source_sha256"],
                    "source_checkpoint_sha256": source_row["source_checkpoint_sha256"],
                    "train_split_sha256": source_row["train_split_sha256"],
                    "validation_split_sha256": source_row["validation_split_sha256"],
                    "batch_schedule_sha256": source_row["batch_schedule_sha256"],
                    "evaluator_checkpoint": dict(source_row["evaluator_checkpoint"]),
                    "candidate_receipt": dict(source_row["candidate_receipt"]),
                    "budget": dict(source_row["budget"]),
                    "resources": dict(source_row["resources"]),
                }
            )

    summaries: list[dict[str, Any]] = []
    for graph_seed in plan.graph_seeds:
        scores = replicate_scores[graph_seed]
        score_mean = fmean(scores)
        score_se = stdev(scores) / math.sqrt(len(scores)) if len(scores) > 1 else 0.0
        summaries.append(
            {
                "graph_seed": graph_seed,
                "topology_sha256": generate_topology(graph_seed, plan.architecture)[
                    "topology_sha256"
                ],
                "weighted_rank_percentile_mean": score_mean,
                "weighted_rank_percentile_se": score_se,
                "replicate_count": len(scores),
                "metric_means": {
                    metric.name: fmean(per_seed_metric_values[graph_seed][metric.name])
                    for metric in plan.metrics
                },
            }
        )
    summaries.sort(
        key=lambda row: (row["weighted_rank_percentile_mean"], row["graph_seed"])
    )
    selected = {row["graph_seed"] for row in summaries[: plan.shortlist_size]}
    for rank, row in enumerate(summaries, start=1):
        row["screen_rank"] = rank
        row["selected"] = row["graph_seed"] in selected
    return summaries, sorted(
        rank_rows, key=lambda row: (row["graph_seed"], row["replicate_seed"])
    )


def build_ablation_contract(
    *,
    plan: ScreenPlan,
    inputs: ScreenInputs,
    summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    shortlist = [row for row in summaries if row["selected"]]
    variant_rows = []
    for variant in plan.ablation_variants:
        graph_seeds: list[int | None]
        if variant["graph_seed_policy"] == "shortlisted":
            graph_seeds = [int(row["graph_seed"]) for row in shortlist]
        else:
            graph_seeds = [None]
        for graph_seed in graph_seeds:
            variant_rows.append(
                {
                    "variant_id": variant["id"],
                    "graph_seed": graph_seed,
                    "parameter_match_class": variant["parameter_match_class"],
                    "purpose": variant["purpose"],
                    "controller_sha256": inputs.controller_sha256,
                    "controller_mutation_allowed": False,
                    "paired_train_seeds": list(plan.replicate_seeds),
                    "same_eval_count_required": True,
                    "same_wall_clock_view_required": True,
                    "heldout_board_size_separate": True,
                }
            )
    return {
        "schema_version": A19_SCREEN_SCHEMA_VERSION,
        "axis_id": A19_AXIS_ID,
        "role": "ablation_readiness",
        "evidence_status": "skeleton_only",
        "claim_scope": "ablation_readiness_only",
        "execution_status": "ready_for_training_ablation",
        "auto_promoted": False,
        "controller_contract": {
            "mode": "frozen_exact_sha256",
            "path": str(inputs.controller_checkpoint),
            "sha256": inputs.controller_sha256,
            "mutation_allowed": False,
        },
        "replay_contract": {
            "path": str(inputs.replay_corpus),
            "sha256": inputs.replay_corpus_sha256,
            "reuse_exact_corpus": True,
        },
        "selection_contract": {
            "method": "preregistered_weighted_mean_paired_rank_percentile",
            "lower_is_better": True,
            "tie_break": "ascending_graph_seed",
            "shortlist_size": plan.shortlist_size,
            "metrics": [metric.__dict__ for metric in plan.metrics],
        },
        "shortlisted_graph_seeds": [int(row["graph_seed"]) for row in shortlist],
        "variants": variant_rows,
        "prohibited_inferences": list(plan.prohibited_inferences),
    }


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise A19PreparationError(
            "run_id must use 1-128 ASCII letters, digits, dot, underscore, or hyphen"
        )
    return run_id


def write_json(path: Path, payload: Any) -> None:
    path.write_bytes(canonical_json_bytes(payload))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(dict(row)))


__all__ = [
    "A19_AXIS_ID",
    "A19_CLAIM_SCOPE",
    "A19_SCREEN_SCHEMA_VERSION",
    "A19PreparationError",
    "MetricSpec",
    "ScreenInputs",
    "ScreenPlan",
    "build_ablation_contract",
    "canonical_json_bytes",
    "file_sha256",
    "generate_topology",
    "load_json_strict",
    "load_jsonl_strict",
    "load_screen_plan",
    "rank_graph_seeds",
    "validate_inputs",
    "validate_proxy_rows",
    "validate_run_id",
    "write_json",
    "write_jsonl",
]
