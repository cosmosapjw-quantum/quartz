#!/usr/bin/env python3
"""Frozen-checkpoint phase 1.5 ablation runner.

This runner replaces the older prior-revision study. It follows the clean-split
matrix in ``phase15_strategy_revision_v2.md``:

- Group A: substrate/controller sanity
- Group B: refresh isolation on top of A4
- Group C: legacy anchor comparison only
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import controller_sweep as sweep

from quartz.autotune_runtime import apply_runtime_overrides
from quartz.contract_summary import summarize_plain_contracts
from quartz.encoders import get_encoder
from quartz.models_torch import AlphaZeroNet
from quartz.phase15_ablation import (
    Phase15System,
    apply_system_readout,
    candidate_undercoverage,
    first_revision_budget,
    kl_divergence,
    load_systems_config,
    normalize_policy,
    policy_argmax,
    system_semantic_signature,
    summarize_rows,
    topk_recall,
)
from quartz.phase15_suite import (
    annotate_position_suite,
    bucket_counts,
    bucket_thresholds,
    merge_suite_policy_artifacts,
    mine_balanced_suite,
    read_suite_policy_artifacts,
    split_suite_policy_artifacts,
    write_suite_policy_artifacts,
)
from quartz.phase15_trace import (
    build_trace_artifact,
    load_cached_trace,
    store_cached_trace,
    trace_cache_key,
    trace_cache_salt,
)
from quartz.runtime_support import NNSearchClient, encode_board, load_torch_state_dict


def json_dump(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def jsonl_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_phase15_contracts(
    *,
    execution_mode: str,
    game: str,
    checkpoints: list[CheckpointRef],
    systems: list[Phase15System],
    budgets: list[int],
    trace_cache_salt_value: str,
    reference_checkpoint: CheckpointRef | None = None,
    reference_system: Phase15System | None = None,
    oracle_checkpoint: CheckpointRef | None = None,
    oracle_system: Phase15System | None = None,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    runner_contract = {
        "contract_type": "runner",
        "execution_mode": str(execution_mode),
        "game": str(game),
        "budgets": [int(item) for item in budgets],
        "trace_cache_salt": str(trace_cache_salt_value),
    }
    if extra:
        runner_contract.update(copy.deepcopy(extra))
    contracts.append(runner_contract)
    for checkpoint in checkpoints:
        contracts.append(
            {
                "contract_type": "checkpoint",
                "checkpoint_id": str(checkpoint.id),
                "checkpoint_path": str(checkpoint.path),
            }
        )
    for system in systems:
        contracts.append(
            {
                "contract_type": "system",
                "system_id": str(system.id),
                "semantic_signature": list(system_semantic_signature(system)),
                "system": asdict(system),
            }
        )
    if reference_checkpoint is not None:
        contracts.append(
            {
                "contract_type": "reference_checkpoint",
                "checkpoint_id": str(reference_checkpoint.id),
                "checkpoint_path": str(reference_checkpoint.path),
            }
        )
    if reference_system is not None:
        contracts.append(
            {
                "contract_type": "reference_system",
                "system_id": str(reference_system.id),
                "semantic_signature": list(system_semantic_signature(reference_system)),
                "system": asdict(reference_system),
            }
        )
    if oracle_checkpoint is not None:
        contracts.append(
            {
                "contract_type": "oracle_checkpoint",
                "checkpoint_id": str(oracle_checkpoint.id),
                "checkpoint_path": str(oracle_checkpoint.path),
            }
        )
    if oracle_system is not None:
        contracts.append(
            {
                "contract_type": "oracle_system",
                "system_id": str(oracle_system.id),
                "semantic_signature": list(system_semantic_signature(oracle_system)),
                "system": asdict(oracle_system),
            }
        )
    return contracts


def summarize_phase15_contracts(contracts: list[dict[str, Any]]) -> dict[str, Any]:
    return summarize_plain_contracts(contracts)


def parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_csv_strings(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


@dataclass(frozen=True)
class CheckpointRef:
    id: str
    path: str


class FrozenCheckpointHarness:
    def __init__(self, checkpoint: CheckpointRef, base_cfg: dict[str, Any], device, rust_binary: str):
        self.checkpoint = checkpoint
        self.base_cfg = copy.deepcopy(base_cfg)
        self.base_cfg["_name"] = base_cfg["_name"]
        try:
            self.base_cfg["_encoder"] = get_encoder(self.base_cfg["_name"])
        except Exception:
            self.base_cfg["_encoder"] = None
        self.device = device
        self.rust_binary = rust_binary
        self.model = AlphaZeroNet(self.base_cfg).to(device)
        self.model.load_state_dict(load_torch_state_dict(checkpoint.path, __import__("torch"), map_location=device))
        self.model.eval()
        self._clients: dict[str, Any] = {}
        self._search_cache: dict[tuple[str, str, int], dict[str, Any]] = {}
        self._prior_cache: dict[str, np.ndarray] = {}
        self._feature_cache: dict[str, np.ndarray] = {}

    def close(self) -> None:
        for client in self._clients.values():
            try:
                client.stop()
            except Exception:
                pass
        self._clients.clear()

    def _position_key(self, position: dict[str, Any]) -> str:
        return json.dumps(
            {
                "id": position.get("id"),
                "board": position.get("board"),
                "player": position.get("player"),
                "fen": position.get("fen"),
                "state_meta": position.get("state_meta"),
                "features": position.get("features"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _position_features(self, position: dict[str, Any]) -> np.ndarray:
        key = self._position_key(position)
        cached = self._feature_cache.get(key)
        if cached is not None:
            return cached.copy()
        if "features" in position:
            features = np.asarray(position["features"], dtype=np.float32)
        else:
            board = np.asarray(position["board"], dtype=np.int8)
            player = int(position["player"])
            features = encode_board(self.base_cfg, board, player)
        self._feature_cache[key] = features.copy()
        return features

    def prior_policy(self, position: dict[str, Any]) -> np.ndarray:
        key = self._position_key(position)
        cached = self._prior_cache.get(key)
        if cached is not None:
            return cached.copy()
        features = self._position_features(position)
        torch = __import__("torch")
        x = torch.from_numpy(np.expand_dims(features, axis=0)).to(self.device)
        with torch.inference_mode():
            logits, _values = self.model(x)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        probs = normalize_policy(probs)
        self._prior_cache[key] = probs.copy()
        return probs

    def prime_prior_cache(self, positions: list[dict[str, Any]], batch_size: int = 64) -> None:
        pending: list[tuple[str, np.ndarray]] = []
        for position in positions:
            key = self._position_key(position)
            if key in self._prior_cache:
                continue
            pending.append((key, self._position_features(position)))
        if not pending:
            return
        torch = __import__("torch")
        for start in range(0, len(pending), max(1, int(batch_size))):
            chunk = pending[start : start + max(1, int(batch_size))]
            feat_batch = np.stack([features for _key, features in chunk], axis=0).astype(np.float32, copy=False)
            x = torch.from_numpy(feat_batch).to(self.device)
            with torch.inference_mode():
                logits, _values = self.model(x)
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            for (key, _features), prob in zip(chunk, probs, strict=False):
                self._prior_cache[key] = normalize_policy(prob).copy()

    def _client_key(self, system: Phase15System) -> str:
        cfg = apply_runtime_overrides(self.base_cfg, system.search_overrides)
        return json.dumps(
            {
                "rust_binary": self.rust_binary,
                "search_cfg": {
                    "game": cfg.get("_name"),
                    "seed": cfg.get("seed"),
                    "search_profile": cfg.get("search_profile", "quartz"),
                    "penalty_mode": cfg.get("penalty_mode", "GatedRefresh"),
                    "hbar_penalty_cap": cfg.get("hbar_penalty_cap", 0.3),
                    "sigma_0": cfg.get("sigma_0", 0.3),
                    "min_visits": cfg.get("min_visits", 50),
                    "check_interval": cfg.get("check_interval", 100),
                    "prior_refresh_rate": cfg.get("prior_refresh_rate", 0.0),
                    "prior_refresh_temp": cfg.get("prior_refresh_temp", 1.0),
                    "c_puct": cfg.get("c_puct", 0.0),
                    "n_threads": cfg.get("n_threads", 1),
                    "batch_size": cfg.get("batch_size", 8),
                    "batch_timeout_us": cfg.get("batch_timeout_us"),
                    "root_only_shaping": cfg.get("root_only_shaping"),
                    "vl_mode": cfg.get("vl_mode"),
                    "tt_enabled": cfg.get("tt_enabled"),
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _cfg_for_system(self, system: Phase15System, budget: int) -> dict[str, Any]:
        cfg = apply_runtime_overrides(self.base_cfg, system.search_overrides)
        cfg["iters"] = max(1, int(budget))
        return cfg

    def _get_client(self, system: Phase15System, budget: int):
        key = self._client_key(system)
        client = self._clients.get(key)
        if client is not None:
            client.cfg.update(self._cfg_for_system(system, budget))
            return client
        cfg = self._cfg_for_system(system, budget)
        client = NNSearchClient(self.model, cfg, self.device, self.rust_binary)
        client.start()
        self._clients[key] = client
        return client

    def search_policy(self, position: dict[str, Any], system: Phase15System, budget: int) -> dict[str, Any]:
        from quartz.replay import dense_policy_from_sparse

        pos_key = self._position_key(position)
        cache_key = (pos_key, self._client_key(system), int(budget))
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        client = self._get_client(system, budget)
        board = np.asarray(position.get("board", []), dtype=np.int8) if "board" in position else None
        player = int(position.get("player", 1))
        fen = position.get("fen")
        state_meta = dict(position.get("state_meta") or {})
        t0 = time.perf_counter()
        payload = client.search_move(
            board,
            player,
            penalty_mode=client.cfg.get("penalty_mode", "None"),
            fen=fen,
            state_meta=state_meta,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        policy = dense_policy_from_sparse(payload.get("policy", []), int(self.base_cfg["actions"]))
        row = {
            "search_policy": normalize_policy(policy).tolist(),
            "best_move": int(payload.get("best_move", -1)),
            "value": float(payload.get("value", 0.0)),
            "p_flip": float(payload.get("p_flip", 0.0)),
            "latency_ms": float(elapsed_ms),
        }
        self._search_cache[cache_key] = dict(row)
        return row


def resolve_checkpoint_refs(args: argparse.Namespace, base_dir: Path) -> list[CheckpointRef]:
    rows = sweep.resolve_checkpoint_paths(args, base_dir)
    refs = []
    for idx, row in enumerate(rows):
        refs.append(CheckpointRef(id=f"C{idx+1:02d}_{Path(row).stem}", path=row))
    return refs


def checkpoint_family_label(path_str: str) -> str:
    path = Path(path_str)
    if path.parent.name.startswith("seed_") and len(path.parents) >= 2:
        return path.parent.parent.name
    return path.parent.name


def validate_checkpoint_refs(args: argparse.Namespace, checkpoints: list[CheckpointRef]) -> None:
    if not checkpoints:
        raise ValueError("no checkpoints resolved")
    if args.checkpoints:
        return
    if not args.checkpoint_dir:
        return

    all_discovered = sweep.discover_checkpoint_paths(Path(args.checkpoint_dir), limit=None)
    if len(all_discovered) > len(checkpoints):
        raise ValueError(
            "phase15 assays require curated checkpoint selection; "
            f"--checkpoint-dir discovered {len(all_discovered)} checkpoints but only the first "
            f"{len(checkpoints)} lexical paths would be used. Pass explicit --checkpoints "
            "for weak/mid/strong coverage."
        )

    families = [checkpoint_family_label(ref.path) for ref in checkpoints]
    if len(sorted(set(families))) < min(3, len(checkpoints)):
        raise ValueError(
            "phase15 assays expect checkpoint diversity across weak/mid/strong regimes; "
            f"resolved checkpoints collapse to families {sorted(set(families))}. "
            "Pass explicit --checkpoints instead of relying on directory order."
        )


def choose_reference_checkpoint(checkpoints: list[CheckpointRef], explicit: str | None) -> CheckpointRef:
    if explicit:
        for ref in checkpoints:
            if ref.id == explicit or ref.path == explicit:
                return ref
        raise ValueError(f"reference checkpoint not found: {explicit}")
    return checkpoints[-1]


def choose_checkpoint(checkpoints: list[CheckpointRef], explicit: str | None, default: CheckpointRef) -> CheckpointRef:
    if not explicit:
        return default
    for ref in checkpoints:
        if ref.id == explicit or ref.path == explicit:
            return ref
    raise ValueError(f"checkpoint not found: {explicit}")


def require_system(systems: list[Phase15System], system_id: str) -> Phase15System:
    system = next((item for item in systems if item.id == system_id), None)
    if system is None:
        raise ValueError(f"systems config must include {system_id}")
    return system


def build_oracle_system(
    all_systems: list[Phase15System],
    *,
    oracle_system_id: str | None,
    oracle_profile: str,
    reference_system: Phase15System,
) -> Phase15System:
    if oracle_system_id:
        return require_system(all_systems, oracle_system_id)
    overrides = dict(reference_system.search_overrides)
    overrides["search_profile"] = str(oracle_profile)
    return Phase15System(
        id="ORACLE",
        label=f"oracle({oracle_profile})",
        group="A",
        substrate=reference_system.substrate,
        controller=reference_system.controller,
        refresh_operator="none",
        search_overrides=overrides,
        execution_mode="posthoc",
    )


def load_or_generate_positions(args: argparse.Namespace, base_cfg: dict[str, Any], *, count: int | None = None) -> list[dict[str, Any]]:
    if args.positions_file:
        payload = json.loads(Path(args.positions_file).read_text(encoding="utf-8"))
        positions = payload.get("positions", payload) if isinstance(payload, dict) else payload
        if not isinstance(positions, list) or not positions:
            raise ValueError(f"positions file is empty or invalid: {args.positions_file}")
        artifact_path = None
        if isinstance(payload, dict):
            raw_artifact_path = payload.get("suite_artifacts_file")
            if isinstance(raw_artifact_path, str) and raw_artifact_path:
                artifact_path = (Path(args.positions_file).parent / raw_artifact_path).resolve()
        artifacts = read_suite_policy_artifacts(artifact_path)
        out = []
        for idx, row in enumerate(merge_suite_policy_artifacts(positions, artifacts)):
            item = dict(row)
            item.setdefault("id", f"P{idx+1:04d}")
            out.append(item)
        return out
    rows = sweep.generate_random_positions(
        args.game,
        base_cfg,
        count=int(count if count is not None else args.suite_size),
        seed=int(args.seed),
        min_moves=args.position_min_moves,
        max_moves=args.position_max_moves,
    )
    for idx, row in enumerate(rows):
        row["id"] = f"P{idx+1:04d}"
    return rows


def prepare_bucketized_suite(
    reference: FrozenCheckpointHarness,
    oracle: FrozenCheckpointHarness,
    positions: list[dict[str, Any]],
    reference_system: Phase15System,
    oracle_system: Phase15System,
    low_budget: int,
    oracle_budget: int,
    bucket_thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    reference.prime_prior_cache(positions)
    return annotate_position_suite(
        positions,
        prior_policy_fn=reference.prior_policy,
        low_policy_fn=lambda row: np.asarray(
            reference.search_policy(row, reference_system, low_budget)["search_policy"], dtype=np.float32
        ),
        reference_policy_fn=lambda row: np.asarray(
            reference.search_policy(row, reference_system, oracle_budget)["search_policy"], dtype=np.float32
        ),
        oracle_policy_fn=lambda row: np.asarray(
            oracle.search_policy(row, oracle_system, oracle_budget)["search_policy"], dtype=np.float32
        ),
        thresholds=bucket_thresholds,
    )


def make_trace_budgets(target_budget: int, base_budgets: list[int], allow_extra: bool) -> list[int]:
    rows = sorted({int(b) for b in base_budgets if int(b) <= int(target_budget)})
    if target_budget not in rows:
        rows.append(int(target_budget))
    if allow_extra:
        higher = next((int(b) for b in sorted(set(base_budgets)) if int(b) > int(target_budget)), None)
        if higher is not None:
            rows.append(higher)
    return rows


def build_search_trace(
    harness: FrozenCheckpointHarness,
    checkpoint: CheckpointRef,
    position: dict[str, Any],
    system: Phase15System,
    trace_budgets: list[int],
    cache_dir: Path | None,
) -> tuple[list[np.ndarray], list[float], bool]:
    cache_key = trace_cache_key(
        checkpoint.id,
        checkpoint.path,
        harness._position_key(position),
        system.id,
        system_semantic_signature(system),
        trace_budgets,
        code_salt=trace_cache_salt(),
    )
    cached = load_cached_trace(cache_dir, cache_key)
    if cached is not None:
        policies = [
            np.asarray(policy, dtype=np.float32)
            for policy in cached.get("trace_policies", [])
        ]
        latencies = [float(x) for x in cached.get("trace_latencies_ms", [])]
        if len(policies) == len(trace_budgets) and len(latencies) == len(trace_budgets):
            return policies, latencies, True
    policies = []
    latencies: list[float] = []
    for budget in trace_budgets:
        row = harness.search_policy(position, system, budget)
        policies.append(np.asarray(row["search_policy"], dtype=np.float32))
        latencies.append(float(row.get("latency_ms", 0.0)))
    store_cached_trace(
        cache_dir,
        cache_key,
        build_trace_artifact(trace_budgets, policies, latencies, source="fresh"),
    )
    return policies, latencies, False


def build_search_trace_bundle(
    harness: FrozenCheckpointHarness,
    checkpoint: CheckpointRef,
    position: dict[str, Any],
    system: Phase15System,
    budgets: list[int],
    cache_dir: Path | None,
) -> tuple[list[int], dict[int, np.ndarray], dict[int, float], bool]:
    bundle_budgets = sorted({int(budget) for budget in budgets})
    policies, latencies, trace_reused = build_search_trace(
        harness,
        checkpoint,
        position,
        system,
        bundle_budgets,
        cache_dir,
    )
    return (
        bundle_budgets,
        {
            int(budget): np.asarray(policy, dtype=np.float32)
            for budget, policy in zip(bundle_budgets, policies, strict=False)
        },
        {
            int(budget): float(latency)
            for budget, latency in zip(bundle_budgets, latencies, strict=False)
        },
        bool(trace_reused),
    )


def slice_trace_bundle(
    trace_bundle_policies: dict[int, np.ndarray],
    trace_bundle_latencies_ms: dict[int, float],
    *,
    target_budget: int,
    base_budgets: list[int],
    allow_extra: bool,
) -> tuple[list[int], list[np.ndarray], list[float]]:
    trace_budgets = make_trace_budgets(target_budget, base_budgets, allow_extra=allow_extra)
    return (
        trace_budgets,
        [np.asarray(trace_bundle_policies[int(budget)], dtype=np.float32) for budget in trace_budgets],
        [float(trace_bundle_latencies_ms[int(budget)]) for budget in trace_budgets],
    )


def build_row(
    checkpoint: CheckpointRef,
    position: dict[str, Any],
    system: Phase15System,
    budget: int,
    prior_input: np.ndarray,
    final_policy: np.ndarray,
    reference_policy: np.ndarray,
    oracle_policy: np.ndarray,
    trace_meta: dict[str, Any],
    *,
    alias_of: str | None = None,
    trace_reused: bool,
) -> dict[str, Any]:
    reference_best = policy_argmax(reference_policy)
    oracle_best = policy_argmax(oracle_policy)
    argmax_prior = policy_argmax(prior_input)
    argmax_effective = policy_argmax(final_policy)
    bucket_tags = set(position.get("bucket_tags", []))
    trace_acquire_ms = float(trace_meta.get("trace_acquire_ms", 0.0))
    readout_ms = float(trace_meta.get("readout_ms", 0.0))
    effective_runtime_ms = float(trace_meta.get("effective_runtime_ms", trace_acquire_ms + readout_ms))
    row = {
        "group": system.group,
        "system": system.id,
        "system_label": system.label,
        "execution_mode": system.execution_mode,
        "search_continuation": str(trace_meta.get("search_continuation", "independent_restarts")),
        "checkpoint_id": checkpoint.id,
        "checkpoint_path": checkpoint.path,
        "position_id": str(position["id"]),
        "position_bucket": ",".join(sorted(bucket_tags)),
        "budget": int(budget),
        "argmax_prior": int(argmax_prior),
        "argmax_effective": int(argmax_effective),
        "reference_best": int(reference_best),
        "oracle_best": int(oracle_best),
        "accuracy_to_reference": int(argmax_effective == reference_best),
        "accuracy_to_oracle": int(argmax_effective == oracle_best),
        "topk_recall_reference": topk_recall(final_policy, reference_policy, k=3),
        "topk_recall_oracle": topk_recall(final_policy, oracle_policy, k=3),
        "wrong_prior_vs_reference": int(argmax_prior != reference_best),
        "wrong_prior_vs_oracle": int(argmax_prior != oracle_best),
        "wrong_prior_correction_reference": int(argmax_prior != reference_best and argmax_effective == reference_best),
        "wrong_prior_correction_oracle": int(argmax_prior != oracle_best and argmax_effective == oracle_best),
        "easy_case_regret_reference": int("easy_good_prior" in bucket_tags and argmax_effective != reference_best),
        "easy_case_regret_oracle": int("easy_good_prior" in bucket_tags and argmax_effective != oracle_best),
        "kl_to_reference": float(kl_divergence(final_policy, reference_policy)),
        "kl_to_oracle": float(kl_divergence(final_policy, oracle_policy)),
        "trace_acquire_ms": trace_acquire_ms,
        "readout_ms": readout_ms,
        "effective_runtime_ms": effective_runtime_ms,
        "trace_reused": int(trace_reused),
        "revision_occurred": int(argmax_effective != argmax_prior),
        "revision_step": first_revision_budget(
            trace_meta.get("argmax_path", []),
            trace_meta.get("trace_budgets", []),
            argmax_prior,
        ),
        "num_revisions": int(trace_meta.get("num_revisions", 0)),
        "argmax_persistence": float(trace_meta.get("argmax_persistence", 0.0)),
        "top2_margin_stability": float(trace_meta.get("top2_margin_stability", 0.0)),
        "challenger_overlap": float(trace_meta.get("challenger_overlap", 1.0)),
        "posterior_entropy_slope": float(trace_meta.get("posterior_entropy_slope", 0.0)),
        "revision_flip_flop_count": int(trace_meta.get("revision_flip_flop_count", 0)),
        "argmax_path": list(trace_meta.get("argmax_path", [])),
        "entropy_path": list(trace_meta.get("entropy_path", [])),
        "effective_policy": normalize_policy(final_policy).tolist(),
    }
    if alias_of is not None:
        row["alias_of"] = str(alias_of)
    if "commit_confidence" in trace_meta:
        row["commit_confidence"] = float(trace_meta["commit_confidence"])
        row["commit_applied"] = int(trace_meta.get("commit_applied", 0))
        row["commit_latency"] = trace_meta.get("commit_latency")
    if "root_candidate_set" in trace_meta:
        row["root_candidate_set"] = list(trace_meta["root_candidate_set"])
        row["root_candidate_scores"] = list(trace_meta.get("root_candidate_scores", []))
        row["candidate_undercoverage"] = candidate_undercoverage(row["root_candidate_set"], oracle_best)
        row["challenger_recall_k"] = int(trace_meta.get("challenger_recall_k", 0))
    if "budget_burst_triggered" in trace_meta:
        row["budget_burst_triggered"] = int(trace_meta["budget_burst_triggered"])
        row["extra_budget_used"] = int(trace_meta.get("extra_budget_used", 0))
        row["burst_budget"] = int(trace_meta.get("burst_budget", budget))
    if "continuation_fallback_reason" in trace_meta:
        row["continuation_fallback_reason"] = str(trace_meta["continuation_fallback_reason"])
    return row


def suite_policy_artifact(position: dict[str, Any], key: str) -> np.ndarray | None:
    raw = position.get(key)
    if not isinstance(raw, list) or not raw:
        return None
    return np.asarray(raw, dtype=np.float32)


def validate_cached_suite_payload(
    suite_payload: dict[str, Any],
    *,
    reference_checkpoint: CheckpointRef,
    reference_system: Phase15System,
    oracle_checkpoint: CheckpointRef,
    oracle_system: Phase15System,
) -> None:
    expected = {
        "reference_checkpoint": asdict(reference_checkpoint),
        "reference_system": asdict(reference_system),
        "oracle_checkpoint": asdict(oracle_checkpoint),
        "oracle_system": asdict(oracle_system),
    }
    for key, value in expected.items():
        actual = suite_payload.get(key)
        if actual != value:
            raise ValueError(
                "cached position_suite metadata does not match the current phase15 reference/oracle contract; "
                f"mismatch at {key}. Rebuild the suite instead of using --skip-suite-build."
            )


def run_group_rows(
    checkpoints: list[CheckpointRef],
    suite: list[dict[str, Any]],
    systems: list[Phase15System],
    budgets: list[int],
    oracle_budget: int,
    base_cfg: dict[str, Any],
    device,
    rust_binary: str,
    reference_system: Phase15System,
    reference_checkpoint: CheckpointRef,
    oracle_system: Phase15System,
    oracle_checkpoint: CheckpointRef,
    cache_dir: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0
    systems_by_id = {system.id: system for system in systems}
    for checkpoint in checkpoints:
        harness = FrozenCheckpointHarness(checkpoint, base_cfg, device, rust_binary)
        reference_harness = harness if checkpoint.path == reference_checkpoint.path else FrozenCheckpointHarness(
            reference_checkpoint, base_cfg, device, rust_binary
        )
        oracle_harness = reference_harness if reference_checkpoint.path == oracle_checkpoint.path else (
            harness if checkpoint.path == oracle_checkpoint.path else FrozenCheckpointHarness(
                oracle_checkpoint, base_cfg, device, rust_binary
            )
        )
        try:
            harness.prime_prior_cache(suite)
            if reference_harness is not harness:
                reference_harness.prime_prior_cache(suite)
            if oracle_harness not in {harness, reference_harness}:
                oracle_harness.prime_prior_cache(suite)
            for position in suite:
                prior_input = harness.prior_policy(position)
                reference_policy = suite_policy_artifact(position, "reference_policy")
                if reference_policy is None:
                    reference_policy = np.asarray(
                        reference_harness.search_policy(position, reference_system, oracle_budget)["search_policy"],
                        dtype=np.float32,
                    )
                oracle_policy = suite_policy_artifact(position, "oracle_policy")
                if oracle_policy is None:
                    oracle_policy = np.asarray(
                        oracle_harness.search_policy(position, oracle_system, oracle_budget)["search_policy"],
                        dtype=np.float32,
                    )
                for system in systems:
                    trace_system = systems_by_id.get(system.report_alias, system) if system.report_alias else system
                    _bundle_budgets, trace_bundle_policies, trace_bundle_latencies_ms, trace_reused = (
                        build_search_trace_bundle(
                            harness,
                            checkpoint,
                            position,
                            trace_system,
                            budgets,
                            cache_dir,
                        )
                    )
                    if trace_reused:
                        cache_hits += 1
                    else:
                        cache_misses += 1
                    for budget in budgets:
                        trace_budgets, search_trace, trace_latencies_ms = slice_trace_bundle(
                            trace_bundle_policies,
                            trace_bundle_latencies_ms,
                            target_budget=budget,
                            base_budgets=budgets,
                            allow_extra=(trace_system.refresh_operator == "budget_routing"),
                        )
                        readout_t0 = time.perf_counter()
                        final_policy, trace_meta = apply_system_readout(
                            system,
                            prior_input,
                            search_trace,
                            trace_budgets,
                            budget,
                        )
                        readout_ms = (time.perf_counter() - readout_t0) * 1000.0
                        trace_meta = {
                            **trace_meta,
                            "trace_budgets": [int(x) for x in trace_budgets],
                            "trace_latencies_ms": [float(x) for x in trace_latencies_ms],
                            "trace_acquire_ms": float(sum(trace_latencies_ms)),
                            "readout_ms": float(readout_ms),
                            "effective_runtime_ms": float(sum(trace_latencies_ms) + readout_ms),
                            "search_continuation": "independent_restarts",
                        }
                        rows.append(
                            build_row(
                                checkpoint,
                                position,
                                system,
                                budget,
                                prior_input,
                                np.asarray(final_policy, dtype=np.float32),
                                reference_policy,
                                oracle_policy,
                                trace_meta,
                                alias_of=system.report_alias,
                                trace_reused=trace_reused,
                            )
                        )
        finally:
            harness.close()
            if reference_harness is not harness:
                reference_harness.close()
            if oracle_harness not in {harness, reference_harness}:
                oracle_harness.close()
    return rows, {
        "trace_cache_unit": "trace_bundle",
        "trace_bundle_cache_hits": int(cache_hits),
        "trace_bundle_cache_misses": int(cache_misses),
        "trace_bundle_cache_hit_rate": float(cache_hits / max(1, cache_hits + cache_misses)),
        "trace_cache_hits": int(cache_hits),
        "trace_cache_misses": int(cache_misses),
        "trace_cache_hit_rate": float(cache_hits / max(1, cache_hits + cache_misses)),
    }


def build_summary_payload(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return summarize_rows(
        rows,
        [
            "accuracy_to_reference",
            "accuracy_to_oracle",
            "wrong_prior_correction_reference",
            "wrong_prior_correction_oracle",
            "easy_case_regret_reference",
            "easy_case_regret_oracle",
            "topk_recall_reference",
            "topk_recall_oracle",
            "kl_to_reference",
            "kl_to_oracle",
            "trace_acquire_ms",
            "readout_ms",
            "effective_runtime_ms",
            "argmax_persistence",
            "top2_margin_stability",
            "candidate_undercoverage",
            "commit_confidence",
            "commit_applied",
            "challenger_recall_k",
            "budget_burst_triggered",
            "extra_budget_used",
        ],
    )


def build_semantic_summary_payload(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], dict[str, Any]] = {}
    metric_keys = [
        "accuracy_to_reference",
        "accuracy_to_oracle",
        "wrong_prior_correction_reference",
        "wrong_prior_correction_oracle",
        "easy_case_regret_reference",
        "easy_case_regret_oracle",
        "topk_recall_reference",
        "topk_recall_oracle",
        "kl_to_reference",
        "kl_to_oracle",
        "trace_acquire_ms",
        "readout_ms",
        "effective_runtime_ms",
    ]
    for row in rows:
        key = (
            str(row.get("alias_of") or row["system"]),
            int(row["budget"]),
            str(row.get("execution_mode", "posthoc")),
        )
        acc = grouped.setdefault(
            key,
            {
                "source_system": key[0],
                "budget": key[1],
                "execution_mode": key[2],
                "systems_present": set(),
                "rows": 0,
            },
        )
        acc["systems_present"].add(str(row["system"]))
        acc["rows"] += 1
        for metric_key in metric_keys:
            if metric_key in row:
                acc[metric_key] = acc.get(metric_key, 0.0) + float(row[metric_key])
    out = []
    for acc in grouped.values():
        n_rows = max(1, int(acc["rows"]))
        item = {
            "source_system": acc["source_system"],
            "budget": acc["budget"],
            "execution_mode": acc["execution_mode"],
            "systems_present": sorted(acc["systems_present"]),
            "rows": n_rows,
        }
        for metric_key in metric_keys:
            if metric_key in acc:
                item[metric_key] = acc[metric_key] / n_rows
        out.append(item)
    out.sort(key=lambda row: (row["budget"], row["source_system"], row["execution_mode"]))
    return out


def classify_phase15_headwind(
    *,
    readout_ratio_mean: float,
    accuracy_to_reference_mean: float,
    kl_to_reference_mean: float,
) -> str:
    runtime_flag = float(readout_ratio_mean) >= 0.20
    semantic_flag = float(kl_to_reference_mean) >= 0.20 or float(accuracy_to_reference_mean) < 0.50
    if runtime_flag and semantic_flag:
        return "mixed_readout_cost_and_semantic_drift"
    if runtime_flag:
        return "readout_cost"
    if semantic_flag:
        return "semantic_drift"
    return "trace_acquire_cost"


def build_headwind_summary_payload(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("alias_of") or row["system"]),
            int(row["budget"]),
            str(row.get("execution_mode", "posthoc")),
        )
        acc = grouped.setdefault(
            key,
            {
                "source_system": key[0],
                "budget": key[1],
                "execution_mode": key[2],
                "systems_present": set(),
                "rows": 0,
                "trace_acquire_ms": 0.0,
                "readout_ms": 0.0,
                "effective_runtime_ms": 0.0,
                "accuracy_to_reference": 0.0,
                "accuracy_to_oracle": 0.0,
                "kl_to_reference": 0.0,
                "kl_to_oracle": 0.0,
                "revision_occurred": 0.0,
            },
        )
        acc["systems_present"].add(str(row["system"]))
        acc["rows"] += 1
        for key_name in (
            "trace_acquire_ms",
            "readout_ms",
            "effective_runtime_ms",
            "accuracy_to_reference",
            "accuracy_to_oracle",
            "kl_to_reference",
            "kl_to_oracle",
            "revision_occurred",
        ):
            acc[key_name] += float(row.get(key_name, 0.0))
    out = []
    for acc in grouped.values():
        n_rows = max(1, int(acc["rows"]))
        trace_acquire_ms = float(acc["trace_acquire_ms"] / n_rows)
        readout_ms = float(acc["readout_ms"] / n_rows)
        effective_runtime_ms = float(acc["effective_runtime_ms"] / n_rows)
        accuracy_to_reference = float(acc["accuracy_to_reference"] / n_rows)
        accuracy_to_oracle = float(acc["accuracy_to_oracle"] / n_rows)
        kl_to_reference = float(acc["kl_to_reference"] / n_rows)
        kl_to_oracle = float(acc["kl_to_oracle"] / n_rows)
        readout_ratio = float(readout_ms / max(1e-9, effective_runtime_ms))
        out.append(
            {
                "source_system": acc["source_system"],
                "budget": acc["budget"],
                "execution_mode": acc["execution_mode"],
                "systems_present": sorted(acc["systems_present"]),
                "rows": n_rows,
                "trace_acquire_ms": trace_acquire_ms,
                "readout_ms": readout_ms,
                "effective_runtime_ms": effective_runtime_ms,
                "readout_ratio_mean": readout_ratio,
                "accuracy_to_reference": accuracy_to_reference,
                "accuracy_to_oracle": accuracy_to_oracle,
                "kl_to_reference": kl_to_reference,
                "kl_to_oracle": kl_to_oracle,
                "revision_rate": float(acc["revision_occurred"] / n_rows),
                "speedup_headwind": classify_phase15_headwind(
                    readout_ratio_mean=readout_ratio,
                    accuracy_to_reference_mean=accuracy_to_reference,
                    kl_to_reference_mean=kl_to_reference,
                ),
            }
        )
    out.sort(key=lambda row: (row["budget"], row["source_system"], row["execution_mode"]))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.5 clean-split frozen-checkpoint ablation runner")
    parser.add_argument("--game", default="gomoku7", choices=[
        "gomoku7",
        "gomoku15",
        "gomoku15_free",
        "gomoku15_std",
        "gomoku15_omok",
        "gomoku15_renju",
        "gomoku15_caro",
        "tictactoe",
    ])
    parser.add_argument("--output", default="results/phase15_ablation")
    parser.add_argument("--checkpoints", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--max-checkpoints", type=int, default=3)
    parser.add_argument("--bootstrap-if-empty", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=2)
    parser.add_argument("--bootstrap-games", type=int, default=8)
    parser.add_argument("--bootstrap-eval-games", type=int, default=4)
    parser.add_argument("--bootstrap-seeds", default="41,42,43")
    parser.add_argument("--force-bootstrap", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rust-binary", default="./target/release/mcts_demo")
    parser.add_argument("--positions-file", default=None)
    parser.add_argument("--suite-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--position-min-moves", type=int, default=None)
    parser.add_argument("--position-max-moves", type=int, default=None)
    parser.add_argument("--reference-checkpoint", default=None)
    parser.add_argument("--oracle-checkpoint", default=None)
    parser.add_argument("--reference-system", default="A0")
    parser.add_argument("--oracle-system", default=None)
    parser.add_argument("--oracle-profile", default="baseline_strict")
    parser.add_argument("--budgets", default="8,16,32,64")
    parser.add_argument("--oracle-budget", type=int, default=256)
    parser.add_argument("--systems-config", default=None)
    parser.add_argument("--systems", default="A0,A1,A2,A3,A4,B0,B1,B2,B3,C0,C1,C2")
    parser.add_argument("--groups", default="A,B,C")
    parser.add_argument("--skip-suite-build", action="store_true")
    parser.add_argument("--suite-source", choices=["random", "mined"], default="random")
    parser.add_argument("--suite-candidate-multiplier", type=int, default=4)
    parser.add_argument("--bucket-min-count", type=int, default=4)
    parser.add_argument("--trace-cache-dir", default=None)
    parser.add_argument("--disable-trace-cache", action="store_true")
    parser.add_argument("--write-default-systems-config", default=None)
    parser.add_argument("--confident-threshold", type=float, default=0.55)
    parser.add_argument("--ambiguous-margin", type=float, default=0.10)
    parser.add_argument("--root-conflict-topk", type=int, default=2)
    parser.add_argument("--deep-conflict-topk", type=int, default=2)
    parser.add_argument("--search-stall-timeout-s", type=float, default=45.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["QUARTZ_SEARCH_STALL_TIMEOUT_S"] = str(float(args.search_stall_timeout_s))
    base_dir = Path(args.output) / args.game
    base_dir.mkdir(parents=True, exist_ok=True)

    base_cfg, device = sweep.build_base_cfg(args.game, args.device)
    base_cfg["seed"] = int(args.seed)
    checkpoints = resolve_checkpoint_refs(args, base_dir)
    validate_checkpoint_refs(args, checkpoints)
    all_systems = load_systems_config(args.systems_config, base_cfg)
    if args.write_default_systems_config:
        json_dump(Path(args.write_default_systems_config), {"systems": [asdict(system) for system in all_systems]})

    selected_ids = set(parse_csv_strings(args.systems))
    selected_groups = set(parse_csv_strings(args.groups))
    systems = [system for system in all_systems if system.id in selected_ids and system.group in selected_groups]
    if not systems:
        raise ValueError("no systems selected")

    reference_system = require_system(all_systems, args.reference_system)
    reference_checkpoint = choose_reference_checkpoint(checkpoints, args.reference_checkpoint)
    oracle_checkpoint = choose_checkpoint(checkpoints, args.oracle_checkpoint, reference_checkpoint)
    oracle_system = build_oracle_system(
        all_systems,
        oracle_system_id=args.oracle_system,
        oracle_profile=args.oracle_profile,
        reference_system=reference_system,
    )
    trace_cache_dir = None if args.disable_trace_cache else Path(args.trace_cache_dir or (base_dir / "trace_cache"))

    manifest = {
        "format_version": 2,
        "execution_mode": "posthoc",
        "game": args.game,
        "device": str(device),
        "rust_binary": args.rust_binary,
        "checkpoints": [asdict(ref) for ref in checkpoints],
        "budgets": parse_csv_ints(args.budgets),
        "oracle_budget": int(args.oracle_budget),
        "systems": [asdict(system) for system in systems],
        "reference_system": asdict(reference_system),
        "reference_checkpoint": asdict(reference_checkpoint),
        "oracle_system": asdict(oracle_system),
        "oracle_checkpoint": asdict(oracle_checkpoint),
        "groups": sorted(selected_groups),
        "suite_source": "file" if args.positions_file else args.suite_source,
        "trace_cache_dir": None if trace_cache_dir is None else str(trace_cache_dir),
        "trace_cache_salt": trace_cache_salt(),
        "seed": int(args.seed),
        "search_stall_timeout_s": float(args.search_stall_timeout_s),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": (
            "Phase 1.5 posthoc runner: substrate/controller sanity, posthoc refresh readout, "
            "legacy anchor comparison. Reference and oracle policies are tracked separately."
        ),
    }
    phase15_contracts = build_phase15_contracts(
        execution_mode="posthoc",
        game=args.game,
        checkpoints=checkpoints,
        systems=systems,
        budgets=parse_csv_ints(args.budgets),
        trace_cache_salt_value=trace_cache_salt(),
        reference_checkpoint=reference_checkpoint,
        reference_system=reference_system,
        oracle_checkpoint=oracle_checkpoint,
        oracle_system=oracle_system,
        extra={
            "oracle_budget": int(args.oracle_budget),
            "suite_source": "file" if args.positions_file else args.suite_source,
            "seed": int(args.seed),
            "search_stall_timeout_s": float(args.search_stall_timeout_s),
        },
    )
    manifest["contract_summary"] = summarize_phase15_contracts(phase15_contracts)
    json_dump(base_dir / "phase15_manifest.json", manifest)

    candidate_count = int(args.suite_size)
    if not args.positions_file and args.suite_source == "mined":
        candidate_count = max(int(args.suite_size), int(args.suite_size) * int(args.suite_candidate_multiplier))
    positions = load_or_generate_positions(args, base_cfg, count=candidate_count)
    suite_path = base_dir / "position_suite.json"
    suite_artifact_path = base_dir / "position_suite_artifacts.npz"
    if args.skip_suite_build and suite_path.exists():
        suite_payload = json.loads(suite_path.read_text(encoding="utf-8"))
        validate_cached_suite_payload(
            suite_payload,
            reference_checkpoint=reference_checkpoint,
            reference_system=reference_system,
            oracle_checkpoint=oracle_checkpoint,
            oracle_system=oracle_system,
        )
        suite_artifacts = read_suite_policy_artifacts(suite_artifact_path)
        suite = merge_suite_policy_artifacts(list(suite_payload.get("positions", [])), suite_artifacts)
    else:
        thresholds = bucket_thresholds(
            confident_threshold=float(args.confident_threshold),
            ambiguous_margin=float(args.ambiguous_margin),
            root_conflict_topk=int(args.root_conflict_topk),
            deep_conflict_topk=int(args.deep_conflict_topk),
        )
        ref_harness = FrozenCheckpointHarness(reference_checkpoint, base_cfg, device, args.rust_binary)
        oracle_harness = ref_harness if oracle_checkpoint.path == reference_checkpoint.path else FrozenCheckpointHarness(
            oracle_checkpoint, base_cfg, device, args.rust_binary
        )
        try:
            annotated = prepare_bucketized_suite(
                ref_harness,
                oracle_harness,
                positions,
                reference_system=reference_system,
                oracle_system=oracle_system,
                low_budget=min(parse_csv_ints(args.budgets)),
                oracle_budget=int(args.oracle_budget),
                bucket_thresholds=thresholds,
            )
            suite = (
                mine_balanced_suite(
                    annotated,
                    suite_size=int(args.suite_size),
                    bucket_min_count=int(args.bucket_min_count),
                    seed=int(args.seed),
                )
                if not args.positions_file and args.suite_source == "mined"
                else annotated[: int(args.suite_size)]
            )
        finally:
            ref_harness.close()
            if oracle_harness is not ref_harness:
                oracle_harness.close()
        compact_suite, suite_artifacts = split_suite_policy_artifacts(suite)
        write_suite_policy_artifacts(suite_artifact_path, suite_artifacts)
        json_dump(
            suite_path,
            {
                "reference_checkpoint": asdict(reference_checkpoint),
                "reference_system": asdict(reference_system),
                "oracle_checkpoint": asdict(oracle_checkpoint),
                "oracle_system": asdict(oracle_system),
                "positions": compact_suite,
                "suite_artifacts_file": suite_artifact_path.name,
                "bucket_thresholds": thresholds,
                "suite_source": "file" if args.positions_file else args.suite_source,
                "bucket_counts": bucket_counts(suite),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    rows, trace_cache_stats = run_group_rows(
        checkpoints,
        suite,
        systems,
        parse_csv_ints(args.budgets),
        oracle_budget=int(args.oracle_budget),
        base_cfg=base_cfg,
        device=device,
        rust_binary=args.rust_binary,
        reference_system=reference_system,
        reference_checkpoint=reference_checkpoint,
        oracle_system=oracle_system,
        oracle_checkpoint=oracle_checkpoint,
        cache_dir=trace_cache_dir,
    )
    jsonl_dump(base_dir / "assays" / "phase15_rows.jsonl", rows)
    json_dump(
        base_dir / "phase15_summary.json",
        {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "contract_summary": summarize_phase15_contracts(phase15_contracts),
            "raw_summary": build_summary_payload(rows),
            "semantic_summary": build_semantic_summary_payload(rows),
            "headwind_summary": build_headwind_summary_payload(rows),
            "trace_cache_stats": trace_cache_stats,
        },
    )
    print(f"\nPrepared phase15 assays in {base_dir}", flush=True)


if __name__ == "__main__":
    main()
