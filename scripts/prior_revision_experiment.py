#!/usr/bin/env python3
"""Frozen-checkpoint prior revision assays for B0/B1/N1/N2."""

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

from quartz.prior_revision import (
    PriorRevisionSystem,
    apply_prior_corruption,
    apply_revision_operator,
    candidate_undercoverage,
    classify_position_buckets,
    compute_argmax_path_metrics,
    entropy,
    first_revision_budget,
    kl_divergence,
    load_systems_config,
    normalize_policy,
    policy_argmax,
    summarize_rows,
    topk_recall,
)


def json_dump(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def jsonl_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


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
        import torch
        from quartz.alphazero_train import AlphaZeroNet, NNSearchClient, encode_board, get_encoder, load_torch_state_dict

        self.checkpoint = checkpoint
        self.base_cfg = copy.deepcopy(base_cfg)
        self.base_cfg["_name"] = base_cfg["_name"]
        try:
            self.base_cfg["_encoder"] = get_encoder(self.base_cfg["_name"])
        except Exception:
            self.base_cfg["_encoder"] = None
        self.device = device
        self.rust_binary = rust_binary
        self._torch = torch
        self._alphazero_net_cls = AlphaZeroNet
        self._load_torch_state_dict = load_torch_state_dict
        self._encode_board = encode_board
        self._client_cls = NNSearchClient

        self.model = AlphaZeroNet(self.base_cfg).to(device)
        self.model.load_state_dict(load_torch_state_dict(checkpoint.path, torch, map_location=device))
        self.model.eval()
        self._clients: dict[str, Any] = {}
        self._search_cache: dict[tuple[str, str, int], dict[str, Any]] = {}
        self._prior_cache: dict[str, np.ndarray] = {}

    def close(self) -> None:
        for client in self._clients.values():
            try:
                client.stop()
            except Exception:
                pass
        self._clients.clear()

    def _position_key(self, position: dict[str, Any]) -> str:
        if position.get("id"):
            return str(position["id"])
        return json.dumps(
            {
                "board": position.get("board"),
                "player": position.get("player"),
                "fen": position.get("fen"),
                "state_meta": position.get("state_meta"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def prior_policy(self, position: dict[str, Any]) -> np.ndarray:
        key = self._position_key(position)
        cached = self._prior_cache.get(key)
        if cached is not None:
            return cached.copy()

        if "features" in position:
            features = np.asarray(position["features"], dtype=np.float32)
        else:
            board = np.asarray(position["board"], dtype=np.int8)
            player = int(position["player"])
            features = self._encode_board(self.base_cfg, board, player)

        x = self._torch.from_numpy(np.expand_dims(features, axis=0)).to(self.device)
        with self._torch.inference_mode():
            logits, _values = self.model(x)
            probs = self._torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        probs = normalize_policy(probs)
        self._prior_cache[key] = probs.copy()
        return probs

    def _client_key(self, system: PriorRevisionSystem) -> str:
        return json.dumps(
            {
                "search_overrides": system.search_overrides,
                "rust_binary": self.rust_binary,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _cfg_for_system(self, system: PriorRevisionSystem, budget: int) -> dict[str, Any]:
        from quartz.alphazero_train import apply_runtime_overrides

        cfg = apply_runtime_overrides(self.base_cfg, system.search_overrides)
        cfg["iters"] = max(1, int(budget))
        return cfg

    def _get_client(self, system: PriorRevisionSystem, budget: int):
        key = self._client_key(system)
        client = self._clients.get(key)
        if client is not None:
            client.cfg.update(self._cfg_for_system(system, budget))
            return client
        cfg = self._cfg_for_system(system, budget)
        client = self._client_cls(self.model, cfg, self.device, self.rust_binary)
        client.start()
        self._clients[key] = client
        return client

    def search_policy(self, position: dict[str, Any], system: PriorRevisionSystem, budget: int) -> dict[str, Any]:
        from quartz.alphazero_train import dense_policy_from_sparse

        pos_key = self._position_key(position)
        cache_key = (pos_key, self._client_key(system), int(budget))
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        client = self._get_client(system, budget)
        game_name = self.base_cfg["_name"]
        board = np.asarray(position.get("board", []), dtype=np.int8) if "board" in position else None
        player = int(position.get("player", 1))
        fen = position.get("fen")
        state_meta = dict(position.get("state_meta") or {})
        t0 = time.perf_counter()
        payload = client.search_move(board, player, penalty_mode=client.cfg.get("penalty_mode", "GatedRefresh"), fen=fen, state_meta=state_meta)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        policy = dense_policy_from_sparse(payload.get("policy", []), int(self.base_cfg["actions"]))
        row = {
            "search_policy": normalize_policy(policy).tolist(),
            "best_move": int(payload.get("best_move", -1)),
            "value": float(payload.get("value", 0.0)),
            "p_flip": float(payload.get("p_flip", 0.0)),
            "latency_ms": float(elapsed_ms),
            "game": game_name,
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
            "prior revision assays require curated checkpoint selection; "
            f"--checkpoint-dir discovered {len(all_discovered)} checkpoints but only the first "
            f"{len(checkpoints)} lexical paths would be used. Pass explicit --checkpoints "
            "for weak/mid/strong coverage."
        )

    families = [checkpoint_family_label(ref.path) for ref in checkpoints]
    unique_families = sorted(set(families))
    if len(unique_families) < min(3, len(checkpoints)):
        raise ValueError(
            "prior revision assays expect checkpoint diversity across weak/mid/strong regimes; "
            f"resolved checkpoints collapse to families {unique_families}. "
            "Pass explicit --checkpoints instead of relying on directory order."
        )


def choose_reference_checkpoint(checkpoints: list[CheckpointRef], explicit: str | None) -> CheckpointRef:
    if explicit:
        for ref in checkpoints:
            if ref.id == explicit or ref.path == explicit:
                return ref
        raise ValueError(f"reference checkpoint not found: {explicit}")
    return checkpoints[-1]


def load_or_generate_positions(args: argparse.Namespace, base_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if args.positions_file:
        payload = json.loads(Path(args.positions_file).read_text(encoding="utf-8"))
        positions = payload.get("positions", payload) if isinstance(payload, dict) else payload
        if not isinstance(positions, list) or not positions:
            raise ValueError(f"positions file is empty or invalid: {args.positions_file}")
        out = []
        for idx, row in enumerate(positions):
            item = dict(row)
            item.setdefault("id", f"P{idx+1:04d}")
            out.append(item)
        return out
    rows = sweep.generate_random_positions(
        args.game,
        base_cfg,
        count=int(args.suite_size),
        seed=int(args.seed),
        min_moves=args.position_min_moves,
        max_moves=args.position_max_moves,
    )
    for idx, row in enumerate(rows):
        row["id"] = f"P{idx+1:04d}"
    return rows


def prepare_bucketized_suite(reference: FrozenCheckpointHarness, positions: list[dict[str, Any]],
                             base_system: PriorRevisionSystem, low_budget: int, oracle_budget: int,
                             bucket_thresholds: dict[str, Any]) -> list[dict[str, Any]]:
    suite = []
    for row in positions:
        prior = reference.prior_policy(row)
        low = np.asarray(reference.search_policy(row, base_system, low_budget)["search_policy"], dtype=np.float32)
        oracle = np.asarray(reference.search_policy(row, base_system, oracle_budget)["search_policy"], dtype=np.float32)
        item = dict(row)
        item["bucket_tags"] = classify_position_buckets(prior, low, oracle, thresholds=bucket_thresholds)
        item["prior_argmax"] = policy_argmax(prior)
        item["oracle_best"] = policy_argmax(oracle)
        item["low_budget_best"] = policy_argmax(low)
        suite.append(item)
    return suite


def select_positions_for_experiment(suite: list[dict[str, Any]], experiment: str) -> list[dict[str, Any]]:
    bucket_map = {
        "E1": {"wrong_top1", "wrong_confident", "wrong_top2swap"},
        "E3": {"root_conflict", "generic", "shallow_trap"},
        "E5": {"root_conflict", "deep_conflict"},
    }
    required = bucket_map[experiment]
    rows = [row for row in suite if required.intersection(set(row.get("bucket_tags", [])))]
    return rows


def compute_system_trace(prior_input: np.ndarray, search_trace: list[np.ndarray], trace_budgets: list[int],
                         system: PriorRevisionSystem) -> dict[str, Any]:
    effective_path = []
    per_step_logs = []
    final_meta: dict[str, Any] = {}
    for budget, search_policy in zip(trace_budgets, search_trace):
        effective, meta = apply_revision_operator(system, prior_input, search_policy, budget)
        effective_path.append(normalize_policy(effective))
        per_step_logs.append(
            {
                "budget": int(budget),
                "argmax_effective": policy_argmax(effective_path[-1]),
                "entropy": entropy(effective_path[-1]),
                **meta,
            }
        )
        final_meta = meta
    argmax_path = [policy_argmax(policy) for policy in effective_path]
    metrics = compute_argmax_path_metrics(argmax_path)
    return {
        "final_policy": effective_path[-1] if effective_path else normalize_policy(prior_input),
        "effective_path": [policy.tolist() for policy in effective_path],
        "argmax_path": argmax_path,
        "entropy_path": [float(entropy(policy)) for policy in effective_path],
        "per_step_logs": per_step_logs,
        "trace_budgets": [int(budget) for budget in trace_budgets],
        **metrics,
        **final_meta,
    }


def build_search_trace(harness: FrozenCheckpointHarness, position: dict[str, Any], system: PriorRevisionSystem,
                       trace_budgets: list[int]) -> tuple[list[np.ndarray], float]:
    policies = []
    latency_ms = 0.0
    for budget in trace_budgets:
        row = harness.search_policy(position, system, budget)
        policies.append(np.asarray(row["search_policy"], dtype=np.float32))
        latency_ms += float(row.get("latency_ms", 0.0))
    return policies, latency_ms


def make_trace_budgets(target_budget: int, base_budgets: list[int]) -> list[int]:
    rows = sorted({int(b) for b in base_budgets if int(b) <= int(target_budget)})
    if target_budget not in rows:
        rows.append(int(target_budget))
    return rows


def build_row_base(experiment: str, checkpoint: CheckpointRef, position: dict[str, Any], budget: int,
                   system: PriorRevisionSystem, prior_input: np.ndarray, final_policy: np.ndarray,
                   oracle_policy: np.ndarray, trace_payload: dict[str, Any], compute_time_ms: float) -> dict[str, Any]:
    oracle_best = policy_argmax(oracle_policy)
    argmax_prior = policy_argmax(prior_input)
    argmax_effective = policy_argmax(final_policy)
    row = {
        "experiment": experiment,
        "checkpoint_id": checkpoint.id,
        "checkpoint_path": checkpoint.path,
        "position_id": str(position["id"]),
        "position_bucket": ",".join(position.get("bucket_tags", [])),
        "budget": int(budget),
        "system": system.id,
        "system_label": system.label,
        "argmax_prior": int(argmax_prior),
        "argmax_effective": int(argmax_effective),
        "oracle_best": int(oracle_best),
        "revision_occurred": int(argmax_effective != argmax_prior),
        "revision_step": first_revision_budget(trace_payload["argmax_path"], trace_payload["trace_budgets"], argmax_prior),
        "num_revisions": int(trace_payload["num_revisions"]),
        "posterior_entropy_t": list(trace_payload["entropy_path"]),
        "kl_prior_to_effective_t": [
            float(kl_divergence(prior_input, np.asarray(policy, dtype=np.float32)))
            for policy in trace_payload["effective_path"]
        ],
        "candidate_set_contains_oracle_best": int(
            1 - candidate_undercoverage(trace_payload.get("root_candidate_set", []), oracle_best)
        ) if trace_payload.get("root_candidate_set") is not None else None,
        "topk_recall": topk_recall(final_policy, oracle_policy, k=3),
        "compute_time_ms": float(compute_time_ms),
        "argmax_path": list(trace_payload["argmax_path"]),
        "entropy_path": list(trace_payload["entropy_path"]),
        "effective_policy": normalize_policy(final_policy).tolist(),
    }
    if "root_candidate_set" in trace_payload:
        row["root_candidate_set"] = list(trace_payload["root_candidate_set"])
        row["root_candidate_scores"] = list(trace_payload.get("root_candidate_scores", []))
        row["undercoverage_flag"] = candidate_undercoverage(row["root_candidate_set"], oracle_best)
    if "posterior_search" in trace_payload:
        row["posterior_search"] = list(trace_payload["posterior_search"])
        row["dual_gate"] = float(trace_payload.get("dual_gate", 0.0))
        row["posterior_norm"] = float(trace_payload.get("posterior_norm", 0.0))
    return row


def run_experiment_rows(experiment: str, checkpoints: list[CheckpointRef], suite: list[dict[str, Any]],
                        systems: list[PriorRevisionSystem], budgets: list[int], oracle_budget: int,
                        corruptions: list[str], base_cfg: dict[str, Any], device, rust_binary: str,
                        reference_system: PriorRevisionSystem) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_positions = select_positions_for_experiment(suite, experiment)
    if not selected_positions:
        return rows

    for checkpoint in checkpoints:
        harness = FrozenCheckpointHarness(checkpoint, base_cfg, device, rust_binary)
        try:
            for position in selected_positions:
                prior_base = harness.prior_policy(position)
                oracle_policy = np.asarray(
                    harness.search_policy(position, reference_system, oracle_budget)["search_policy"],
                    dtype=np.float32,
                )
                corruption_list = corruptions if experiment == "E1" else [None]
                for corruption in corruption_list:
                    prior_input = (
                        apply_prior_corruption(prior_base, corruption, policy_argmax(oracle_policy))
                        if corruption is not None
                        else prior_base
                    )
                    for budget in budgets:
                        trace_budgets = make_trace_budgets(budget, budgets)
                        for system in systems:
                            search_trace, latency_ms = build_search_trace(harness, position, system, trace_budgets)
                            trace_payload = compute_system_trace(prior_input, search_trace, trace_budgets, system)
                            final_policy = np.asarray(trace_payload["final_policy"], dtype=np.float32)

                            row = build_row_base(
                                experiment,
                                checkpoint,
                                position,
                                budget,
                                system,
                                prior_input,
                                final_policy,
                                oracle_policy,
                                trace_payload,
                                compute_time_ms=latency_ms,
                            )

                            if corruption is not None:
                                row["corruption"] = corruption
                                row["correction_success"] = int(
                                    policy_argmax(prior_input) != policy_argmax(oracle_policy)
                                    and policy_argmax(final_policy) == policy_argmax(oracle_policy)
                                )
                                row["false_revision"] = int(
                                    policy_argmax(prior_input) == policy_argmax(oracle_policy)
                                    and policy_argmax(final_policy) != policy_argmax(oracle_policy)
                                )
                                row["overcorrection_rate"] = int(
                                    policy_argmax(prior_input) != policy_argmax(oracle_policy)
                                    and policy_argmax(final_policy) != policy_argmax(oracle_policy)
                                    and policy_argmax(final_policy) != policy_argmax(prior_input)
                                )
                                row["kl_to_oracle"] = float(kl_divergence(final_policy, oracle_policy))
                            if experiment == "E3":
                                row["root_accuracy"] = int(policy_argmax(final_policy) == policy_argmax(oracle_policy))
                                row["gain_per_sim"] = float(row["topk_recall"]) / max(1, int(budget))
                            if experiment == "E5":
                                row["accuracy"] = int(policy_argmax(final_policy) == policy_argmax(oracle_policy))
                                row["correction_gain"] = float(
                                    kl_divergence(prior_input, oracle_policy) - kl_divergence(final_policy, oracle_policy)
                                )
                                row["kl_to_oracle"] = float(kl_divergence(final_policy, oracle_policy))
                            rows.append(row)
        finally:
            harness.close()
    return rows


def build_summary_payload(experiment_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    metrics_by_experiment = {
        "E1": ["correction_success", "false_revision", "topk_recall", "kl_to_oracle", "compute_time_ms"],
        "E3": ["root_accuracy", "topk_recall", "gain_per_sim", "compute_time_ms"],
        "E5": ["accuracy", "correction_gain", "kl_to_oracle", "compute_time_ms"],
    }
    summary = {}
    for experiment, rows in experiment_rows.items():
        summary[experiment] = summarize_rows(rows, metrics_by_experiment[experiment])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prior revision frozen-checkpoint assay runner")
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
    parser.add_argument("--output", default="results/prior_revision")
    parser.add_argument("--checkpoints", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--max-checkpoints", type=int, default=3)
    parser.add_argument("--bootstrap-if-empty", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=2)
    parser.add_argument("--bootstrap-games", type=int, default=8)
    parser.add_argument("--bootstrap-eval-games", type=int, default=4)
    parser.add_argument("--bootstrap-seeds", default="41,42,43")
    parser.add_argument("--force-bootstrap", action="store_true")
    parser.add_argument("--backend", default="torch", choices=["auto", "torch", "jax"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rust-binary", default="./target/release/mcts_demo")
    parser.add_argument("--positions-file", default=None)
    parser.add_argument("--suite-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--position-min-moves", type=int, default=None)
    parser.add_argument("--position-max-moves", type=int, default=None)
    parser.add_argument("--reference-checkpoint", default=None)
    parser.add_argument("--budgets", default="8,16,32,64")
    parser.add_argument("--oracle-budget", type=int, default=256)
    parser.add_argument("--systems-config", default=None)
    parser.add_argument("--systems", default="B0,B1,N1,N2")
    parser.add_argument("--experiments", default="E1,E3,E5")
    parser.add_argument("--corruptions", default="swap_top12,inflate_wrong_confidence")
    parser.add_argument("--skip-suite-build", action="store_true")
    parser.add_argument("--write-default-systems-config", default=None)
    parser.add_argument("--confident-threshold", type=float, default=0.55)
    parser.add_argument("--ambiguous-margin", type=float, default=0.10)
    parser.add_argument("--root-conflict-topk", type=int, default=2)
    parser.add_argument("--deep-conflict-topk", type=int, default=2)
    parser.add_argument(
        "--search-stall-timeout-s",
        type=float,
        default=45.0,
        help="Per-search Rust callback read timeout; mirrors controller_sweep defaults",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["QUARTZ_SEARCH_STALL_TIMEOUT_S"] = str(float(args.search_stall_timeout_s))
    base_dir = Path(args.output) / args.game
    base_dir.mkdir(parents=True, exist_ok=True)

    base_cfg, device = sweep.build_base_cfg(args.game, args.device)
    checkpoints = resolve_checkpoint_refs(args, base_dir)
    validate_checkpoint_refs(args, checkpoints)
    if len(checkpoints) < 3:
        print(f"[WARN] prior revision assays are designed for >=3 checkpoints; got {len(checkpoints)}", flush=True)
    all_systems = load_systems_config(args.systems_config, base_cfg)
    reference_system = next((system for system in all_systems if system.id == "B0"), None)
    if reference_system is None:
        raise ValueError("systems config must include a B0 reference system for suite/oracle construction")
    selected_systems = parse_csv_strings(args.systems)
    systems = [system for system in all_systems if system.id in selected_systems]
    if not systems:
        raise ValueError("no systems selected")

    if args.write_default_systems_config:
        json_dump(Path(args.write_default_systems_config), {"systems": [asdict(system) for system in all_systems]})

    manifest = {
        "format_version": 1,
        "game": args.game,
        "device": str(device),
        "rust_binary": args.rust_binary,
        "checkpoints": [asdict(ref) for ref in checkpoints],
        "budgets": parse_csv_ints(args.budgets),
        "oracle_budget": int(args.oracle_budget),
        "systems": [asdict(system) for system in systems],
        "experiments": parse_csv_strings(args.experiments),
        "corruptions": parse_csv_strings(args.corruptions),
        "search_stall_timeout_s": float(args.search_stall_timeout_s),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Phase-1 prior revision runner: root-policy assay over frozen-checkpoint search evidence.",
    }
    json_dump(base_dir / "prior_revision_manifest.json", manifest)

    reference_checkpoint = choose_reference_checkpoint(checkpoints, args.reference_checkpoint)
    if args.reference_checkpoint is None:
        print(
            f"[WARN] no --reference-checkpoint provided; using {reference_checkpoint.id} for bucketization/oracle suite",
            flush=True,
        )
    positions = load_or_generate_positions(args, base_cfg)
    suite_path = base_dir / "position_suite.json"
    if args.skip_suite_build and suite_path.exists():
        suite_payload = json.loads(suite_path.read_text(encoding="utf-8"))
        suite = list(suite_payload.get("positions", []))
    else:
        bucket_thresholds = {
            "confident_threshold": float(args.confident_threshold),
            "ambiguous_margin": float(args.ambiguous_margin),
            "root_conflict_topk": int(args.root_conflict_topk),
            "deep_conflict_topk": int(args.deep_conflict_topk),
        }
        ref_harness = FrozenCheckpointHarness(reference_checkpoint, base_cfg, device, args.rust_binary)
        try:
            suite = prepare_bucketized_suite(
                ref_harness,
                positions,
                base_system=reference_system,
                low_budget=min(parse_csv_ints(args.budgets)),
                oracle_budget=int(args.oracle_budget),
                bucket_thresholds=bucket_thresholds,
            )
        finally:
            ref_harness.close()
        json_dump(
            suite_path,
            {
                "reference_checkpoint": asdict(reference_checkpoint),
                "positions": suite,
                "bucket_thresholds": bucket_thresholds,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    experiment_rows: dict[str, list[dict[str, Any]]] = {}
    budgets = parse_csv_ints(args.budgets)
    corruptions = parse_csv_strings(args.corruptions)
    for experiment in parse_csv_strings(args.experiments):
        if experiment not in {"E1", "E3", "E5"}:
            raise ValueError(f"unsupported experiment for phase-1 runner: {experiment}")
        print(f"\n=== {experiment} ===", flush=True)
        rows = run_experiment_rows(
            experiment,
            checkpoints,
            suite,
            systems,
            budgets,
            oracle_budget=int(args.oracle_budget),
            corruptions=corruptions,
            base_cfg=base_cfg,
            device=device,
            rust_binary=args.rust_binary,
            reference_system=reference_system,
        )
        experiment_rows[experiment] = rows
        jsonl_dump(base_dir / "assays" / f"{experiment}.jsonl", rows)
        print(f"{experiment}: wrote {len(rows)} rows", flush=True)

    summary_payload = build_summary_payload(experiment_rows)
    json_dump(
        base_dir / "prior_revision_summary.json",
        {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary_payload,
        },
    )
    print(f"\nPrepared prior revision assays in {base_dir}", flush=True)


if __name__ == "__main__":
    main()
