"""Deterministic A19 graph-seed proxy trainer and artifact writer.

The full profile implements the registered 40-node, 144-channel topology and
emits every checkpoint/receipt/operator-trace field consumed by the strict A19
finalizer.  The pilot profile intentionally uses a smaller graph subset and
width and is labelled diagnostic; it cannot be finalized as the preregistered
screen.
"""

from __future__ import annotations

import hashlib
import math
import os
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from quartz.experiment_manifest import atomic_json_dump, file_sha256
from quartz.idea_foundry.a19_ablation import (
    ScreenInputs,
    ScreenPlan,
    canonical_json_bytes,
    generate_topology,
    validate_proxy_rows,
)
from quartz.idea_foundry.axis_workflow import atomic_jsonl_dump


REPO_ROOT = Path(__file__).resolve().parents[2]


class A19ProxyError(RuntimeError):
    """Raised when the A19 proxy cannot honor its deterministic contract."""


@dataclass(frozen=True)
class ReplayBatch:
    train_states: np.ndarray
    train_policies: np.ndarray
    train_values: np.ndarray
    validation_states: np.ndarray
    validation_policies: np.ndarray
    validation_values: np.ndarray
    split_contract: Mapping[str, Any]


def _state_hash(state: np.ndarray) -> str:
    return hashlib.sha256(state.tobytes()).hexdigest()


def _split_contract(
    state_hashes: Sequence[str],
    *,
    replicate_seed: int,
    train_positions: int,
    validation_positions: int,
    optimizer_steps: int,
    batch_size: int,
) -> tuple[dict[str, Any], list[int]]:
    unique = sorted(set(state_hashes))
    ordered = sorted(
        unique,
        key=lambda value: hashlib.sha256(
            f"A19-split-v1:{replicate_seed}:{value}".encode("ascii")
        ).hexdigest(),
    )
    required = train_positions + validation_positions
    if len(ordered) < required:
        raise A19ProxyError(
            f"replicate {replicate_seed} has {len(ordered)} unique states; {required} required"
        )
    train = ordered[:train_positions]
    validation = ordered[train_positions:required]
    schedule_rng = random.Random((replicate_seed << 32) ^ 0xA19)
    schedule = [
        schedule_rng.randrange(train_positions)
        for _ in range(optimizer_steps * batch_size)
    ]
    contract = {
        "schema_version": 1,
        "replicate_seed": replicate_seed,
        "method": "deduplicated_state_hash_v1",
        "train_state_group_hashes": train,
        "validation_state_group_hashes": validation,
        "train_split_sha256": hashlib.sha256(canonical_json_bytes(train)).hexdigest(),
        "validation_split_sha256": hashlib.sha256(
            canonical_json_bytes(validation)
        ).hexdigest(),
        "batch_schedule_method": "python_random_index_schedule_v1",
        "batch_schedule_sha256": hashlib.sha256(
            canonical_json_bytes(schedule)
        ).hexdigest(),
        "optimizer_steps": optimizer_steps,
        "batch_size": batch_size,
    }
    return contract, schedule


def load_replay(
    path: Path,
    *,
    replicate_seed: int,
    train_positions: int,
    validation_positions: int,
    optimizer_steps: int,
    batch_size: int,
) -> tuple[ReplayBatch, list[int]]:
    if not path.is_file() or path.is_symlink():
        raise A19ProxyError(f"replay input is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        states = np.asarray(payload["states"], dtype=np.float32)
        pointers = np.asarray(payload["policy_ptr"], dtype=np.int64)
        indices = np.asarray(payload["policy_idx"], dtype=np.int64)
        values = np.asarray(payload["policy_val"], dtype=np.float32)
        value_targets = np.asarray(payload["values"], dtype=np.float32)
    hashes = [_state_hash(state) for state in states]
    contract, schedule = _split_contract(
        hashes,
        replicate_seed=replicate_seed,
        train_positions=train_positions,
        validation_positions=validation_positions,
        optimizer_steps=optimizer_steps,
        batch_size=batch_size,
    )
    first_by_hash: dict[str, int] = {}
    for index, state_hash in enumerate(hashes):
        first_by_hash.setdefault(state_hash, index)

    def materialize(groups: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        row_indices = [first_by_hash[group] for group in groups]
        dense = np.zeros((len(row_indices), 49), dtype=np.float32)
        for output_index, source_index in enumerate(row_indices):
            start = int(pointers[source_index])
            end = int(pointers[source_index + 1])
            dense[output_index, indices[start:end]] = values[start:end]
        return (
            np.ascontiguousarray(states[row_indices]),
            dense,
            np.ascontiguousarray(value_targets[row_indices]),
        )

    train = materialize(contract["train_state_group_hashes"])
    validation = materialize(contract["validation_state_group_hashes"])
    return (
        ReplayBatch(
            train_states=train[0],
            train_policies=train[1],
            train_values=train[2],
            validation_states=validation[0],
            validation_policies=validation[1],
            validation_values=validation[2],
            split_contract=contract,
        ),
        schedule,
    )


def build_model(topology: Mapping[str, Any], channels: int):
    import torch
    from torch import nn

    class GlobalBlock(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(width)
            self.qkv = nn.Linear(width, 3 * width)
            self.output = nn.Linear(width, width)
            self.norm2 = nn.LayerNorm(width)
            self.fc1 = nn.Linear(width, 4 * width)
            self.fc2 = nn.Linear(4 * width, width)

        def forward(self, inputs):
            normalized = self.norm1(inputs)
            query, key, value = self.qkv(normalized).chunk(3, dim=-1)
            weights = torch.softmax(
                torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1]),
                dim=-1,
            )
            inputs = inputs + self.output(torch.matmul(weights, value))
            normalized = self.norm2(inputs)
            return inputs + self.fc2(torch.nn.functional.gelu(self.fc1(normalized)))

    class GraphProxy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            architecture = topology["architecture"]
            self.nodes_per_cell = int(architecture["nodes_per_cell"])
            self.cells = int(architecture["cells"])
            self.stem = nn.Conv2d(17, channels, 3, padding=1, bias=False)
            self.norms = nn.ModuleList(
                [nn.GroupNorm(1, channels) for _ in range(int(architecture["nodes"]))]
            )
            self.convs = nn.ModuleList(
                [
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False)
                    for _ in range(int(architecture["nodes"]))
                ]
            )
            self.route_weights = nn.Parameter(torch.zeros(len(topology["edges"])))
            self.globals = nn.ModuleList(
                [
                    GlobalBlock(channels)
                    for _ in range(int(architecture["global_blocks"]))
                ]
            )
            self.policy = nn.Conv2d(channels, 1, 1)
            self.value1 = nn.Linear(channels, channels, bias=True)
            self.value2 = nn.Linear(channels, channels, bias=False)
            self.value_out = nn.Linear(channels, 1, bias=True)
            incoming: dict[int, list[tuple[int, Mapping[str, Any]]]] = {}
            for edge_index, edge in enumerate(topology["edges"]):
                incoming.setdefault(int(edge["dst"]), []).append((edge_index, edge))
            self.incoming = incoming

        def forward(self, inputs):
            cell_input = self.stem(inputs)
            node_outputs: dict[int, Any] = {}
            for cell in range(self.cells):
                offset = cell * self.nodes_per_cell
                current: list[Any] = []
                for local_index in range(self.nodes_per_cell):
                    node_index = offset + local_index
                    parents = []
                    for route_index, edge in self.incoming[node_index]:
                        source = (
                            cell_input
                            if edge["src_kind"] == "cell_input"
                            else node_outputs[int(edge["src"])]
                        )
                        parents.append(
                            torch.sigmoid(self.route_weights[route_index]) * source
                        )
                    mixed = torch.stack(parents, dim=0).mean(dim=0)
                    output = mixed + self.convs[node_index](
                        torch.nn.functional.gelu(self.norms[node_index](mixed))
                    )
                    node_outputs[node_index] = output
                    current.append(output)
                cell_input = torch.stack(current, dim=0).mean(dim=0)
            tokens = cell_input.flatten(2).transpose(1, 2)
            for block in self.globals:
                tokens = block(tokens)
            features = tokens.transpose(1, 2).reshape(-1, channels, 7, 7)
            policy_logits = self.policy(features).flatten(1)
            pooled = features.mean(dim=(2, 3))
            hidden = torch.nn.functional.gelu(self.value1(pooled))
            hidden = torch.nn.functional.gelu(self.value2(hidden))
            value = torch.tanh(self.value_out(hidden)).squeeze(1)
            return policy_logits, value

    return GraphProxy()


def operator_trace(
    topology: Mapping[str, Any], channels: int, graph_seed: int, replicate_seed: int
) -> dict[str, Any]:
    area = 49
    nodes = int(topology["architecture"]["nodes"])
    global_blocks = int(topology["architecture"]["global_blocks"])
    input_flops = 2 * area * 17 * channels * 3 * 3
    node_flops = nodes * 2 * area * channels * channels * 3 * 3
    routing_flops = max(0, len(topology["edges"]) - nodes) * area * channels
    global_flops = global_blocks * (
        2 * area * channels * 3 * channels
        + 4 * area * area * channels
        + 2 * area * channels * channels
        + 2 * area * channels * 4 * channels
        + 2 * area * 4 * channels * channels
    )
    head_flops = 2 * area * channels + 2 * (2 * channels) * channels + 2 * channels
    operations = [
        {"name": "input_projection", "flops": input_flops},
        {"name": "graph_node_convolutions", "flops": node_flops},
        {"name": "graph_routing", "flops": routing_flops},
        {"name": "global_mixing", "flops": global_flops},
        {"name": "policy_value_heads", "flops": head_flops},
    ]
    return {
        "schema_version": 1,
        "axis_id": "A19",
        "artifact_kind": "a19_operator_trace",
        "graph_seed": graph_seed,
        "replicate_seed": replicate_seed,
        "topology_sha256": topology["topology_sha256"],
        "operations": operations,
        "total_flops": sum(row["flops"] for row in operations),
    }


def deterministic_runtime(torch: Any, device: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "device_type": "cuda",
        "device_name": f"NVIDIA {properties.name}"
        if "NVIDIA" not in properties.name.upper()
        else properties.name,
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": str(torch.backends.cudnn.version()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
    }


def _train_candidate(
    *,
    topology: Mapping[str, Any],
    replay: ReplayBatch,
    schedule: Sequence[int],
    plan: ScreenPlan,
    graph_seed: int,
    replicate_seed: int,
    channels: int,
    optimizer_steps: int,
    batch_size: int,
    output_dir: Path,
    controller_sha256: str,
    replay_manifest_sha256: str,
    replay_source_sha256: str,
    source_checkpoint_sha256: str,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise A19ProxyError("A19 requires a visible NVIDIA CUDA runtime")
    device = torch.device("cuda:0")
    torch.manual_seed(replicate_seed)
    torch.cuda.manual_seed_all(replicate_seed)
    model = build_model(topology, channels).to(device)
    optimizer_contract = dict(plan.proxy_training)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_contract["learning_rate"]),
        weight_decay=float(optimizer_contract["weight_decay"]),
    )
    train_states = torch.from_numpy(replay.train_states)
    train_policies = torch.from_numpy(replay.train_policies)
    train_values = torch.from_numpy(replay.train_values)
    model.train()
    for step in range(optimizer_steps):
        start = step * batch_size
        indices = list(schedule[start : start + batch_size])
        states = train_states[indices].to(device)
        policies = train_policies[indices].to(device)
        values = train_values[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, predictions = model(states)
        policy_loss = functional.kl_div(
            functional.log_softmax(logits, dim=1), policies, reduction="batchmean"
        )
        value_loss = functional.mse_loss(predictions, values)
        (policy_loss + value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        validation_states = torch.from_numpy(replay.validation_states).to(device)
        validation_policies = torch.from_numpy(replay.validation_policies).to(device)
        validation_values = torch.from_numpy(replay.validation_values).to(device)
        logits, predictions = model(validation_states)
        policy_kl = float(
            functional.kl_div(
                functional.log_softmax(logits, dim=1),
                validation_policies,
                reduction="batchmean",
            ).item()
        )
        value_mse = float(functional.mse_loss(predictions, validation_values).item())
    if not math.isfinite(policy_kl) or not math.isfinite(value_mse):
        raise A19ProxyError("A19 proxy produced non-finite validation metrics")

    trainable_names = [name for name, _ in model.named_parameters()]
    state_dict = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    parameters = sum(int(parameter.numel()) for parameter in model.parameters())
    trace = operator_trace(topology, channels, graph_seed, replicate_seed)
    resources = {
        "parameters": parameters,
        "flops": int(trace["total_flops"]),
        "topology_edges": len(topology["edges"]),
        "nodes": int(topology["architecture"]["nodes"]),
        "channels": channels,
        "global_blocks": int(topology["architecture"]["global_blocks"]),
    }
    runtime = deterministic_runtime(torch, device)
    budget = {
        **dict(plan.budget_contract),
        "optimizer_steps": optimizer_steps,
        "batch_size": batch_size,
        "train_positions": int(replay.train_states.shape[0]),
        "validation_positions": int(replay.validation_states.shape[0]),
    }

    def object_hash(value: Any) -> str:
        return hashlib.sha256(canonical_json_bytes(value)).hexdigest()

    metadata = {
        "graph_seed": graph_seed,
        "replicate_seed": replicate_seed,
        "topology_sha256": topology["topology_sha256"],
        "controller_sha256": controller_sha256,
        "replay_source_sha256": replay_source_sha256,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "train_split_sha256": replay.split_contract["train_split_sha256"],
        "validation_split_sha256": replay.split_contract["validation_split_sha256"],
        "batch_schedule_sha256": replay.split_contract["batch_schedule_sha256"],
        "budget_sha256": object_hash(budget),
        "optimizer_contract_sha256": object_hash(optimizer_contract),
        "resources_sha256": object_hash(resources),
        "runtime_sha256": object_hash(runtime),
    }
    stem = f"graph_{graph_seed}.seed_{replicate_seed}"
    checkpoint_path = (output_dir / "checkpoints" / f"{stem}.a19ckpt").resolve()
    trace_path = (output_dir / "operator_traces" / f"{stem}.json").resolve()
    receipt_path = (output_dir / "receipts" / f"{stem}.json").resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "axis_id": "A19",
            "artifact_kind": "a19_proxy_candidate_checkpoint",
            "metadata": metadata,
            "model_state_dict": state_dict,
            "trainable_parameter_names": trainable_names,
        },
        checkpoint_path,
    )
    atomic_json_dump(trace_path, trace)
    checkpoint_sha256 = file_sha256(checkpoint_path)
    trace_sha256 = file_sha256(trace_path)
    receipt = {
        "schema_version": 1,
        "axis_id": "A19",
        "artifact_kind": "a19_proxy_candidate_receipt",
        "graph_seed": graph_seed,
        "replicate_seed": replicate_seed,
        "topology_sha256": topology["topology_sha256"],
        "controller_sha256": controller_sha256,
        "replay_manifest_sha256": replay_manifest_sha256,
        "replay_source_sha256": replay_source_sha256,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "train_split_sha256": replay.split_contract["train_split_sha256"],
        "validation_split_sha256": replay.split_contract["validation_split_sha256"],
        "batch_schedule_sha256": replay.split_contract["batch_schedule_sha256"],
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha256},
        "budget": budget,
        "optimizer_contract": optimizer_contract,
        "deterministic_runtime": runtime,
        "resources": resources,
        "resource_provenance": {
            "parameter_count_method": "sum_trainable_parameters_v1",
            "flop_count_method": "operator_trace_v1",
            "operator_trace_path": str(trace_path),
            "operator_trace_sha256": trace_sha256,
        },
    }
    atomic_json_dump(receipt_path, receipt)
    return {
        "schema_version": 1,
        "axis_id": "A19",
        "graph_seed": graph_seed,
        "replicate_seed": replicate_seed,
        "topology_sha256": topology["topology_sha256"],
        "controller_sha256": controller_sha256,
        "replay_corpus_sha256": replay_manifest_sha256,
        "replay_source_sha256": replay_source_sha256,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "train_split_sha256": replay.split_contract["train_split_sha256"],
        "validation_split_sha256": replay.split_contract["validation_split_sha256"],
        "batch_schedule_sha256": replay.split_contract["batch_schedule_sha256"],
        "evaluator_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "receipt_path": str(receipt_path),
            "receipt_sha256": file_sha256(receipt_path),
        },
        "budget": budget,
        "metrics": {"policy_kl": policy_kl, "value_mse": value_mse},
        "resources": resources,
    }


def run_proxy_screen(
    *,
    plan: ScreenPlan,
    screen_plan_path: Path,
    replay_manifest: Mapping[str, Any],
    replay_manifest_path: Path,
    controller_path: Path,
    profile: str,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    del seed  # graph and replicate seeds fully determine the registered run
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if profile == "full":
        graph_seeds = list(plan.graph_seeds)
        replicate_seeds = list(plan.replicate_seeds)
        channels = int(plan.architecture["channels"])
        budget = dict(plan.budget_contract)
        preregistered_complete = True
    elif profile == "pilot":
        graph_seeds = list(plan.graph_seeds[:2])
        replicate_seeds = list(plan.replicate_seeds[:1])
        channels = 24
        budget = {
            "optimizer_steps": 2,
            "batch_size": 8,
            "train_positions": 32,
            "validation_positions": 16,
        }
        preregistered_complete = False
    else:
        raise A19ProxyError(f"unsupported A19 profile: {profile}")
    if output_dir.exists():
        raise A19ProxyError(f"A19 output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    source_by_seed = {
        int(row["replicate_seed"]): row for row in replay_manifest["sources"]
    }
    controller_sha256 = file_sha256(controller_path)
    replay_manifest_sha256 = file_sha256(replay_manifest_path)
    replays: dict[int, tuple[ReplayBatch, list[int]]] = {}
    for replicate_seed in replicate_seeds:
        source = source_by_seed[replicate_seed]
        replay_path = REPO_ROOT / source["replay_path"]
        if file_sha256(replay_path) != source["replay_sha256"]:
            raise A19ProxyError(f"replay hash drift for seed {replicate_seed}")
        replays[replicate_seed] = load_replay(
            replay_path,
            replicate_seed=replicate_seed,
            train_positions=int(budget["train_positions"]),
            validation_positions=int(budget["validation_positions"]),
            optimizer_steps=int(budget["optimizer_steps"]),
            batch_size=int(budget["batch_size"]),
        )
    rows: list[dict[str, Any]] = []
    for graph_seed in graph_seeds:
        topology = generate_topology(graph_seed, plan.architecture)
        for replicate_seed in replicate_seeds:
            source = source_by_seed[replicate_seed]
            replay, schedule = replays[replicate_seed]
            rows.append(
                _train_candidate(
                    topology=topology,
                    replay=replay,
                    schedule=schedule,
                    plan=plan,
                    graph_seed=graph_seed,
                    replicate_seed=replicate_seed,
                    channels=channels,
                    optimizer_steps=int(budget["optimizer_steps"]),
                    batch_size=int(budget["batch_size"]),
                    output_dir=output_dir,
                    controller_sha256=controller_sha256,
                    replay_manifest_sha256=replay_manifest_sha256,
                    replay_source_sha256=source["replay_sha256"],
                    source_checkpoint_sha256=source["checkpoint_sha256"],
                )
            )
    rows_path = output_dir / "proxy_results.jsonl"
    atomic_jsonl_dump(rows_path, rows)
    strict_validation_passed = False
    if preregistered_complete:
        inputs = ScreenInputs(
            controller_checkpoint=controller_path.resolve(),
            controller_sha256=controller_sha256,
            replay_corpus=replay_manifest_path.resolve(),
            replay_corpus_sha256=replay_manifest_sha256,
            proxy_results=rows_path.resolve(),
            proxy_results_sha256=file_sha256(rows_path),
            replicate_replay_sha256={
                replicate_seed: source_by_seed[replicate_seed]["replay_sha256"]
                for replicate_seed in replicate_seeds
            },
            replicate_checkpoint_sha256={
                replicate_seed: source_by_seed[replicate_seed]["checkpoint_sha256"]
                for replicate_seed in replicate_seeds
            },
        )
        validated = validate_proxy_rows(
            rows,
            plan=plan,
            inputs=inputs,
            split_contracts={
                replicate_seed: replays[replicate_seed][0].split_contract
                for replicate_seed in replicate_seeds
            },
        )
        strict_validation_passed = len(validated) == len(rows)
    summary = {
        "schema_version": 1,
        "axis_id": "A19",
        "profile": profile,
        "execution_status": "completed_no_promotion",
        "outcome_detail": (
            "PREREGISTERED_PROXY_ROWS_COMPLETED"
            if preregistered_complete
            else "PILOT_PROXY_ROWS_COMPLETED_NOT_PREREGISTERED_SCREEN"
        ),
        "candidate_count": len(rows),
        "graph_seeds": graph_seeds,
        "replicate_seeds": replicate_seeds,
        "channels": channels,
        "budget": budget,
        "preregistered_complete": preregistered_complete,
        "strict_finalize_validation_passed": strict_validation_passed,
        "proxy_results": str(rows_path.resolve()),
        "proxy_results_sha256": file_sha256(rows_path),
        "claim_scope": "fixed_replay_proxy_diagnostic_only",
        "promotion": {"auto": False, "eligible": False},
        "prohibited_inferences": ["play_strength", "production_readiness"],
    }
    atomic_json_dump(output_dir / "summary.json", summary)
    artifact_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "run_manifest.json"
    )
    input_paths = [
        screen_plan_path.resolve(),
        replay_manifest_path.resolve(),
        controller_path.resolve(),
    ]
    for replicate_seed in replicate_seeds:
        source = source_by_seed[replicate_seed]
        input_paths.extend(
            [
                (REPO_ROOT / source["replay_path"]).resolve(),
                (REPO_ROOT / source["checkpoint_path"]).resolve(),
            ]
        )
    atomic_json_dump(
        output_dir / "run_manifest.json",
        {
            "schema_version": 1,
            "axis_id": "A19",
            "profile": profile,
            "execution_status": "completed_no_promotion",
            "claim_scope": "fixed_replay_proxy_diagnostic_only",
            "source_hashes": [
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "sha256": file_sha256(path),
                }
                for path in (
                    Path(__file__).resolve(),
                    REPO_ROOT / "quartz" / "idea_foundry" / "a19_ablation.py",
                    REPO_ROOT / "scripts" / "a19_proxy_screen.py",
                )
            ],
            "input_hashes": [
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "sha256": file_sha256(path),
                }
                for path in input_paths
            ],
            "artifacts": [
                {
                    "path": str(path.relative_to(output_dir)),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in artifact_paths
            ],
            "promotion": {"auto": False, "eligible": False},
        },
    )
    return summary
