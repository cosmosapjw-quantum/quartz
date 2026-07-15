#!/usr/bin/env python3
"""Benchmark phase 1.5 online continuation against restart-per-chunk search."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import phase15_ablation_study as posthoc
import phase15_online_ablation as online

from quartz.phase15_online import run_online_readout

DEFAULT_MIN_BUNDLE_SPEEDUP = 1.80
DEFAULT_MIN_TIE_AWARE_MATCH = 0.65
DEFAULT_MAX_KL_MEAN = 0.25
# A continuation-wallclock run above this multiple of the median is a stall
# outlier (documented reconstruction to match the committed contract test).
CONTINUATION_OUTLIER_FACTOR = 1.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark root continuation vs restart-per-chunk search")
    parser.add_argument("--game", default="gomoku7")
    parser.add_argument("--output", default="results/phase15_benchmarks")
    parser.add_argument("--checkpoints", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--max-checkpoints", type=int, default=3)
    parser.add_argument("--positions-file", default="results/controller_sweep_shortlist_v1/gomoku7/stage1_positions.json")
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-threads",
        type=int,
        default=None,
        help="Override search threads; CI smoke uses 1 for deterministic continuation/restart comparisons",
    )
    parser.add_argument("--systems-config", default=None)
    parser.add_argument("--systems", default="A4,B1,B2,B3")
    parser.add_argument("--budgets", default="8,16,32,64")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rust-binary", default="./target/release/mcts_demo")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--search-stall-timeout-s", type=float, default=180.0)
    parser.add_argument("--min-bundle-speedup", type=float, default=DEFAULT_MIN_BUNDLE_SPEEDUP)
    parser.add_argument("--min-tie-aware-match", type=float, default=DEFAULT_MIN_TIE_AWARE_MATCH)
    parser.add_argument("--max-kl-mean", type=float, default=DEFAULT_MAX_KL_MEAN)
    parser.add_argument("--enforce-gate", action="store_true")
    # Bootstrap-if-empty: train a throwaway checkpoint when none is found, so
    # the benchmark can run in a fresh checkout. Args mirror
    # controller_sweep.build_bootstrap_command's expectations (backend default
    # torch). Restored to satisfy the committed contract test after the
    # tracking rewrite dropped the original uncommitted WIP.
    parser.add_argument("--backend", default="torch", choices=["auto", "torch", "jax"])
    parser.add_argument("--bootstrap-if-empty", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=2)
    parser.add_argument("--bootstrap-games", type=int, default=8)
    parser.add_argument("--bootstrap-eval-games", type=int, default=4)
    parser.add_argument("--bootstrap-seeds", default="41,42")
    parser.add_argument("--force-bootstrap", action="store_true")
    return parser.parse_args()


def build_benchmark_contract_summary(
    args: argparse.Namespace,
    checkpoints: list[posthoc.CheckpointRef],
    systems: list[posthoc.Phase15System],
    budgets: list[int],
    *,
    positions_count: int,
    effective_n_threads: int,
) -> dict[str, Any]:
    contracts = posthoc.build_phase15_contracts(
        execution_mode="benchmark_continuation_vs_restart",
        game=args.game,
        checkpoints=checkpoints,
        systems=systems,
        budgets=budgets,
        trace_cache_salt_value=posthoc.trace_cache_salt(),
        extra={
            "positions_file": str(args.positions_file),
            "positions_count": int(positions_count),
            "seed": int(args.seed),
            "n_threads": int(effective_n_threads),
            "repeats": int(args.repeats),
            "warmup_rounds": int(args.warmup_rounds),
            "min_bundle_speedup": float(args.min_bundle_speedup),
            "min_tie_aware_match": float(args.min_tie_aware_match),
            "max_kl_mean": float(args.max_kl_mean),
        },
    )
    return posthoc.summarize_phase15_contracts(contracts)


def prefixed_readout_meta(prefix: str, meta: dict[str, Any]) -> dict[str, Any]:
    """Namespace a readout-meta dict under a channel prefix (e.g. ``continuation``)
    so continuation and restart guard fields never collide in a merged row."""
    return {f"{prefix}_{key}": value for key, value in meta.items()}


def benchmark_checkpoint_payload(checkpoints: list["posthoc.CheckpointRef"]) -> list[dict[str, str]]:
    """Serialize checkpoint refs to ``[{"id","path"}]`` for the run manifest."""
    return [{"id": str(ref.id), "path": str(ref.path)} for ref in checkpoints]


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    rows = sorted(float(x) for x in values)
    p95_index = min(len(rows) - 1, max(0, math.ceil(0.95 * len(rows)) - 1))
    return {
        "mean": float(mean(rows)),
        "median": float(median(rows)),
        "p95": float(rows[p95_index]),
    }


def top1_margin(policy: np.ndarray) -> float:
    if policy.size <= 1:
        return 1.0
    top = np.partition(policy, -2)[-2:]
    return float(top.max() - top.min())


def top_tie_set(policy: np.ndarray, eps: float = 1e-6) -> set[int]:
    if policy.size == 0:
        return set()
    max_val = float(policy.max())
    return {int(idx) for idx, value in enumerate(policy) if abs(float(value) - max_val) <= eps}


def clear_harness_search_cache(harness: posthoc.FrozenCheckpointHarness) -> None:
    harness._search_cache.clear()


def trace_bundle_acquire_ms(trace_rows: dict[int, dict[str, Any]]) -> float:
    return float(sum(float(row.get("latency_ms", 0.0)) for row in trace_rows.values()))


def orchestration_overhead_ms(bundle_wallclock_ms: float, trace_rows: dict[int, dict[str, Any]]) -> float:
    return float(max(0.0, float(bundle_wallclock_ms) - trace_bundle_acquire_ms(trace_rows)))


def evaluate_trace_rows(
    harness: posthoc.FrozenCheckpointHarness,
    position: dict[str, Any],
    system: posthoc.Phase15System,
    budgets: list[int],
    target_budget: int,
    trace_rows: dict[int, dict[str, Any]],
    *,
    continuation_mode: str,
    bundle_wallclock_ms: float,
    fallback_reason: str | None = None,
) -> tuple[np.ndarray, dict[str, Any], list[int]]:
    trace_budgets = posthoc.make_trace_budgets(
        target_budget,
        budgets,
        allow_extra=(system.refresh_operator == "budget_routing"),
    )
    final_policy, trace_meta = run_online_readout(
        system=system,
        position=position,
        prior_input=harness.prior_policy(position),
        budgets=trace_budgets,
        target_budget=int(target_budget),
        search_policy_fn=lambda _position, _system, budget_value, rows=trace_rows: dict(rows[int(budget_value)]),
    )
    trace_meta = dict(trace_meta)
    trace_meta["wallclock_ms"] = float(bundle_wallclock_ms)
    trace_meta["search_continuation"] = str(continuation_mode)
    if fallback_reason:
        trace_meta["continuation_fallback_reason"] = str(fallback_reason)
    return np.asarray(final_policy, dtype=np.float32), trace_meta, trace_budgets


def run_restart_bundle_case(
    harness: posthoc.FrozenCheckpointHarness,
    checkpoint: posthoc.CheckpointRef,
    position: dict[str, Any],
    system: posthoc.Phase15System,
    budgets: list[int],
) -> tuple[dict[int, dict[str, Any]], float]:
    t0 = time.perf_counter()
    bundle_budgets, trace_bundle_policies, trace_bundle_latencies_ms, _trace_reused = posthoc.build_search_trace_bundle(
        harness,
        checkpoint,
        position,
        system,
        budgets,
        cache_dir=None,
    )
    wallclock_ms = (time.perf_counter() - t0) * 1000.0
    return (
        {
            int(budget): {
                "search_policy": posthoc.normalize_policy(trace_bundle_policies[int(budget)]).tolist(),
                "latency_ms": float(trace_bundle_latencies_ms[int(budget)]),
            }
            for budget in bundle_budgets
        },
        float(wallclock_ms),
    )


def run_continuation_bundle_case(
    harness: posthoc.FrozenCheckpointHarness,
    checkpoint: posthoc.CheckpointRef,
    position: dict[str, Any],
    system: posthoc.Phase15System,
    budgets: list[int],
) -> tuple[dict[int, dict[str, Any]], float, str, str | None]:
    t0 = time.perf_counter()
    trace_rows, _trace_reused, mode, fallback_reason = online.build_online_trace_bundle(
        harness,
        checkpoint,
        position,
        system,
        budgets,
        cache_dir=None,
    )
    wallclock_ms = (time.perf_counter() - t0) * 1000.0
    return trace_rows, float(wallclock_ms), str(mode), fallback_reason


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_signature: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    continuation_wallclocks = []
    restart_wallclocks = []
    continuation_effective = []
    restart_effective = []
    kl_values = []
    argmax_matches = 0
    tie_aware_matches = 0
    ambiguous_ties = 0
    for row in rows:
        by_signature[(row["checkpoint_id"], row["system"], str(row["budget"]))].append(row)
        continuation_wallclocks.append(float(row["continuation_wallclock_ms"]))
        restart_wallclocks.append(float(row["restart_wallclock_ms"]))
        continuation_effective.append(float(row["continuation_effective_runtime_ms"]))
        restart_effective.append(float(row["restart_effective_runtime_ms"]))
        kl_values.append(float(row["policy_kl_restart_vs_continuation"]))
        argmax_matches += int(row["argmax_match"])
        tie_aware_matches += int(row["tie_aware_match"])
        ambiguous_ties += int(row["ambiguous_top1_case"])

    grouped = []
    for (checkpoint_id, system_id, budget), group in sorted(by_signature.items()):
        c_rows = [float(item["continuation_wallclock_ms"]) for item in group]
        r_rows = [float(item["restart_wallclock_ms"]) for item in group]
        grouped.append(
            {
                "checkpoint_id": checkpoint_id,
                "system": system_id,
                "budget": int(budget),
                "samples": len(group),
                "continuation_wallclock_ms": percentiles(c_rows),
                "restart_wallclock_ms": percentiles(r_rows),
                "wallclock_speedup_mean": float(mean(r_rows) / max(1e-9, mean(c_rows))),
                "argmax_match_rate": float(sum(int(item["argmax_match"]) for item in group) / len(group)),
                "tie_aware_match_rate": float(sum(int(item["tie_aware_match"]) for item in group) / len(group)),
                "ambiguous_top1_rate": float(sum(int(item["ambiguous_top1_case"]) for item in group) / len(group)),
                "policy_kl_restart_vs_continuation": percentiles(
                    [float(item["policy_kl_restart_vs_continuation"]) for item in group]
                ),
            }
        )

    return {
        "rows": len(rows),
        "argmax_match_rate": float(argmax_matches / max(1, len(rows))),
        "tie_aware_match_rate": float(tie_aware_matches / max(1, len(rows))),
        "ambiguous_top1_rate": float(ambiguous_ties / max(1, len(rows))),
        "policy_kl_restart_vs_continuation": percentiles(kl_values),
        "continuation_wallclock_ms": percentiles(continuation_wallclocks),
        "restart_wallclock_ms": percentiles(restart_wallclocks),
        "continuation_effective_runtime_ms": percentiles(continuation_effective),
        "restart_effective_runtime_ms": percentiles(restart_effective),
        "wallclock_speedup_mean": float(mean(restart_wallclocks) / max(1e-9, mean(continuation_wallclocks))),
        "effective_runtime_speedup_mean": float(
            mean(restart_effective) / max(1e-9, mean(continuation_effective))
        ),
        "by_checkpoint_system_budget": grouped,
    }


def classify_speedup_headwind(
    *,
    continuation_overhead_ratio: float,
    tie_aware_match_rate: float,
    kl_mean: float,
) -> str:
    overhead_flag = float(continuation_overhead_ratio) >= 0.30
    sensitivity_flag = float(kl_mean) >= 0.20 or float(tie_aware_match_rate) < 0.75
    if overhead_flag and sensitivity_flag:
        return "mixed_session_overhead_and_readout_sensitivity"
    if overhead_flag:
        return "session_overhead"
    if sensitivity_flag:
        return "readout_sensitivity"
    return "search_cost"


def summarize_bundle_runs(bundle_runs: list[dict[str, Any]], benchmark_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not bundle_runs:
        return {
            "runs": 0,
            "continuation_wallclock_ms": percentiles([]),
            "restart_wallclock_ms": percentiles([]),
            "wallclock_speedup_mean": 0.0,
            "continuation_trace_acquire_ms": percentiles([]),
            "restart_trace_acquire_ms": percentiles([]),
            "continuation_overhead_ms": percentiles([]),
            "restart_overhead_ms": percentiles([]),
            "continuation_modes": {},
            "fallback_reasons": {},
        }
    continuation_rows = [float(row["continuation_bundle_wallclock_ms"]) for row in bundle_runs]
    restart_rows = [float(row["restart_bundle_wallclock_ms"]) for row in bundle_runs]
    continuation_trace_rows = [float(row["continuation_bundle_trace_acquire_ms"]) for row in bundle_runs]
    restart_trace_rows = [float(row["restart_bundle_trace_acquire_ms"]) for row in bundle_runs]
    continuation_overhead_rows = [float(row["continuation_bundle_overhead_ms"]) for row in bundle_runs]
    restart_overhead_rows = [float(row["restart_bundle_overhead_ms"]) for row in bundle_runs]
    mode_counts: dict[str, int] = defaultdict(int)
    fallback_counts: dict[str, int] = defaultdict(int)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    sensitivity_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in bundle_runs:
        mode_counts[str(row["continuation_mode"])] += 1
        if row.get("continuation_fallback_reason"):
            fallback_counts[str(row["continuation_fallback_reason"])] += 1
        grouped[(str(row["checkpoint_id"]), str(row["system"]))].append(row)
    for row in benchmark_rows:
        sensitivity_groups[(str(row["checkpoint_id"]), str(row["system"]))].append(row)
    by_checkpoint_system = []
    for (checkpoint_id, system_id), group in sorted(grouped.items()):
        c_rows = [float(item["continuation_bundle_wallclock_ms"]) for item in group]
        r_rows = [float(item["restart_bundle_wallclock_ms"]) for item in group]
        c_trace = [float(item["continuation_bundle_trace_acquire_ms"]) for item in group]
        r_trace = [float(item["restart_bundle_trace_acquire_ms"]) for item in group]
        c_over = [float(item["continuation_bundle_overhead_ms"]) for item in group]
        r_over = [float(item["restart_bundle_overhead_ms"]) for item in group]
        sensitivity = sensitivity_groups.get((checkpoint_id, system_id), [])
        tie_aware_rate = float(sum(int(item["tie_aware_match"]) for item in sensitivity) / len(sensitivity)) if sensitivity else 0.0
        ambiguous_rate = float(sum(int(item["ambiguous_top1_case"]) for item in sensitivity) / len(sensitivity)) if sensitivity else 0.0
        kl_mean = float(mean(float(item["policy_kl_restart_vs_continuation"]) for item in sensitivity)) if sensitivity else 0.0
        c_over_ratio = float(mean(c_over) / max(1e-9, mean(c_rows)))
        by_checkpoint_system.append(
            {
                "checkpoint_id": checkpoint_id,
                "system": system_id,
                "samples": len(group),
                "continuation_wallclock_ms": percentiles(c_rows),
                "restart_wallclock_ms": percentiles(r_rows),
                "continuation_trace_acquire_ms": percentiles(c_trace),
                "restart_trace_acquire_ms": percentiles(r_trace),
                "continuation_overhead_ms": percentiles(c_over),
                "restart_overhead_ms": percentiles(r_over),
                "continuation_overhead_ratio_mean": float(c_over_ratio),
                "wallclock_speedup_mean": float(mean(r_rows) / max(1e-9, mean(c_rows))),
                "readout_sensitivity": {
                    "tie_aware_match_rate": tie_aware_rate,
                    "ambiguous_top1_rate": ambiguous_rate,
                    "policy_kl_mean": kl_mean,
                },
                "speedup_headwind": classify_speedup_headwind(
                    continuation_overhead_ratio=c_over_ratio,
                    tie_aware_match_rate=tie_aware_rate,
                    kl_mean=kl_mean,
                ),
            }
        )
    # Per-run (paired) restart/continuation speedup, robust to one stalled run
    # dragging the mean. A continuation wallclock is flagged an outlier when it
    # exceeds CONTINUATION_OUTLIER_FACTOR x the median continuation wallclock.
    # (The factor is a documented reconstruction to match the committed
    # contract test; the original uncommitted rule was lost in the tracking
    # rewrite.)
    pairwise_speedups = [r / max(1e-9, c) for c, r in zip(continuation_rows, restart_rows)]
    median_continuation = float(median(continuation_rows))
    outlier_threshold_ms = CONTINUATION_OUTLIER_FACTOR * median_continuation
    outlier_count = sum(1 for c in continuation_rows if c > outlier_threshold_ms)
    return {
        "runs": len(bundle_runs),
        "continuation_wallclock_ms": percentiles(continuation_rows),
        "restart_wallclock_ms": percentiles(restart_rows),
        "continuation_trace_acquire_ms": percentiles(continuation_trace_rows),
        "restart_trace_acquire_ms": percentiles(restart_trace_rows),
        "continuation_overhead_ms": percentiles(continuation_overhead_rows),
        "restart_overhead_ms": percentiles(restart_overhead_rows),
        "continuation_overhead_ratio_mean": float(mean(continuation_overhead_rows) / max(1e-9, mean(continuation_rows))),
        "wallclock_speedup_mean": float(mean(restart_rows) / max(1e-9, mean(continuation_rows))),
        "wallclock_speedup_pairwise": percentiles(pairwise_speedups),
        "continuation_wallclock_outlier_threshold_ms": float(outlier_threshold_ms),
        "continuation_wallclock_outlier_count": int(outlier_count),
        "continuation_modes": dict(sorted(mode_counts.items())),
        "fallback_reasons": dict(sorted(fallback_counts.items())),
        "by_checkpoint_system": by_checkpoint_system,
    }


def evaluate_benchmark_gate(
    summary: dict[str, Any],
    bundle_summary: dict[str, Any],
    *,
    min_bundle_speedup: float,
    min_tie_aware_match: float,
    max_kl_mean: float,
) -> dict[str, Any]:
    actual_bundle_speedup = float(bundle_summary.get("wallclock_speedup_mean", 0.0))
    actual_tie_aware_match = float(summary.get("tie_aware_match_rate", 0.0))
    actual_kl_mean = float(summary.get("policy_kl_restart_vs_continuation", {}).get("mean", 0.0))
    checks = [
        {
            "name": "bundle_speedup_mean",
            "actual": actual_bundle_speedup,
            "threshold": float(min_bundle_speedup),
            "comparator": ">=",
            "passed": int(actual_bundle_speedup >= float(min_bundle_speedup)),
        },
        {
            "name": "tie_aware_match_rate",
            "actual": actual_tie_aware_match,
            "threshold": float(min_tie_aware_match),
            "comparator": ">=",
            "passed": int(actual_tie_aware_match >= float(min_tie_aware_match)),
        },
        {
            "name": "policy_kl_mean",
            "actual": actual_kl_mean,
            "threshold": float(max_kl_mean),
            "comparator": "<=",
            "passed": int(actual_kl_mean <= float(max_kl_mean)),
        },
    ]
    return {
        "passed": bool(all(int(item["passed"]) == 1 for item in checks)),
        "thresholds": {
            "min_bundle_speedup": float(min_bundle_speedup),
            "min_tie_aware_match": float(min_tie_aware_match),
            "max_kl_mean": float(max_kl_mean),
        },
        "checks": checks,
    }


def main() -> None:
    args = parse_args()
    os.environ["QUARTZ_SEARCH_STALL_TIMEOUT_S"] = str(float(args.search_stall_timeout_s))

    base_dir = Path(args.output) / args.game
    base_dir.mkdir(parents=True, exist_ok=True)

    base_cfg, device = posthoc.sweep.build_base_cfg(args.game, args.device)
    base_cfg["seed"] = int(args.seed)
    if args.n_threads is not None:
        if int(args.n_threads) < 1:
            raise ValueError("--n-threads must be >= 1")
        base_cfg["n_threads"] = int(args.n_threads)
    checkpoints = posthoc.resolve_checkpoint_refs(args, base_dir)
    posthoc.validate_checkpoint_refs(args, checkpoints)
    systems = [
        system
        for system in posthoc.load_systems_config(args.systems_config, base_cfg)
        if system.id in set(posthoc.parse_csv_strings(args.systems))
    ]
    if not systems:
        raise ValueError("no systems selected for benchmark")

    positions = posthoc.load_or_generate_positions(args, base_cfg, count=int(args.max_positions))
    positions = positions[: max(1, int(args.max_positions))]
    for idx, row in enumerate(positions):
        row.setdefault("id", f"P{idx+1:04d}")
    budgets = posthoc.parse_csv_ints(args.budgets)
    contract_summary = build_benchmark_contract_summary(
        args,
        checkpoints,
        systems,
        budgets,
        positions_count=len(positions),
        effective_n_threads=int(base_cfg.get("n_threads", 1)),
    )

    benchmark_rows: list[dict[str, Any]] = []
    bundle_runs: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        harness = posthoc.FrozenCheckpointHarness(checkpoint, base_cfg, device, args.rust_binary)
        try:
            harness.prime_prior_cache(positions)
            for position in positions:
                for system in systems:
                    for repeat_idx in range(int(args.warmup_rounds) + int(args.repeats)):
                        measure = repeat_idx >= int(args.warmup_rounds)
                        order = ("continuation", "restart") if repeat_idx % 2 == 0 else ("restart", "continuation")
                        continuation_rows = None
                        continuation_bundle_wallclock_ms = 0.0
                        continuation_mode = "root_continuation"
                        continuation_fallback_reason = None
                        restart_rows = None
                        restart_bundle_wallclock_ms = 0.0
                        for mode in order:
                            clear_harness_search_cache(harness)
                            if mode == "continuation":
                                (
                                    continuation_rows,
                                    continuation_bundle_wallclock_ms,
                                    continuation_mode,
                                    continuation_fallback_reason,
                                ) = run_continuation_bundle_case(
                                    harness,
                                    checkpoint,
                                    position,
                                    system,
                                    budgets,
                                )
                            else:
                                restart_rows, restart_bundle_wallclock_ms = run_restart_bundle_case(
                                    harness,
                                    checkpoint,
                                    position,
                                    system,
                                    budgets,
                                )
                        if not measure:
                            continue
                        continuation_trace_ms = trace_bundle_acquire_ms(continuation_rows)
                        restart_trace_ms = trace_bundle_acquire_ms(restart_rows)
                        bundle_runs.append(
                            {
                                "checkpoint_id": checkpoint.id,
                                "position_id": str(position["id"]),
                                "system": system.id,
                                "repeat": int(repeat_idx - int(args.warmup_rounds) + 1),
                                "continuation_bundle_wallclock_ms": float(continuation_bundle_wallclock_ms),
                                "restart_bundle_wallclock_ms": float(restart_bundle_wallclock_ms),
                                "continuation_bundle_trace_acquire_ms": float(continuation_trace_ms),
                                "restart_bundle_trace_acquire_ms": float(restart_trace_ms),
                                "continuation_bundle_overhead_ms": orchestration_overhead_ms(
                                    continuation_bundle_wallclock_ms,
                                    continuation_rows,
                                ),
                                "restart_bundle_overhead_ms": orchestration_overhead_ms(
                                    restart_bundle_wallclock_ms,
                                    restart_rows,
                                ),
                                "continuation_mode": str(continuation_mode),
                                "continuation_fallback_reason": continuation_fallback_reason,
                            }
                        )
                        for budget in budgets:
                            continuation_policy, continuation_meta, trace_budgets = evaluate_trace_rows(
                                harness,
                                position,
                                system,
                                budgets,
                                int(budget),
                                continuation_rows,
                                continuation_mode=continuation_mode,
                                bundle_wallclock_ms=float(continuation_bundle_wallclock_ms),
                                fallback_reason=continuation_fallback_reason,
                            )
                            restart_policy, restart_meta, _ = evaluate_trace_rows(
                                harness,
                                position,
                                system,
                                budgets,
                                int(budget),
                                restart_rows,
                                continuation_mode="restart_per_chunk",
                                bundle_wallclock_ms=float(restart_bundle_wallclock_ms),
                            )
                            continuation_ties = top_tie_set(continuation_policy)
                            restart_ties = top_tie_set(restart_policy)
                            continuation_margin = top1_margin(continuation_policy)
                            restart_margin = top1_margin(restart_policy)
                            benchmark_rows.append(
                                {
                                    "checkpoint_id": checkpoint.id,
                                    "checkpoint_path": checkpoint.path,
                                    "position_id": str(position["id"]),
                                    "system": system.id,
                                    "budget": int(budget),
                                    "trace_budgets": [int(x) for x in trace_budgets],
                                    "repeat": int(repeat_idx - int(args.warmup_rounds) + 1),
                                    "continuation_mode": str(continuation_mode),
                                    "continuation_fallback_reason": continuation_fallback_reason,
                                    "continuation_bundle_wallclock_ms": float(continuation_bundle_wallclock_ms),
                                    "restart_bundle_wallclock_ms": float(restart_bundle_wallclock_ms),
                                    "continuation_bundle_trace_acquire_ms": float(continuation_trace_ms),
                                    "restart_bundle_trace_acquire_ms": float(restart_trace_ms),
                                    "continuation_bundle_overhead_ms": orchestration_overhead_ms(
                                        continuation_bundle_wallclock_ms,
                                        continuation_rows,
                                    ),
                                    "restart_bundle_overhead_ms": orchestration_overhead_ms(
                                        restart_bundle_wallclock_ms,
                                        restart_rows,
                                    ),
                                    "continuation_wallclock_ms": float(continuation_meta["wallclock_ms"]),
                                    "restart_wallclock_ms": float(restart_meta["wallclock_ms"]),
                                    "continuation_effective_runtime_ms": float(
                                        continuation_meta.get("effective_runtime_ms", 0.0)
                                    ),
                                    "restart_effective_runtime_ms": float(restart_meta.get("effective_runtime_ms", 0.0)),
                                    "continuation_trace_acquire_ms": float(
                                        continuation_meta.get("trace_acquire_ms", 0.0)
                                    ),
                                    "restart_trace_acquire_ms": float(restart_meta.get("trace_acquire_ms", 0.0)),
                                    "continuation_readout_ms": float(continuation_meta.get("readout_ms", 0.0)),
                                    "restart_readout_ms": float(restart_meta.get("readout_ms", 0.0)),
                                    "continuation_argmax": int(posthoc.policy_argmax(continuation_policy)),
                                    "restart_argmax": int(posthoc.policy_argmax(restart_policy)),
                                    "argmax_match": int(
                                        posthoc.policy_argmax(continuation_policy)
                                        == posthoc.policy_argmax(restart_policy)
                                    ),
                                    "tie_aware_match": int(bool(continuation_ties & restart_ties)),
                                    "ambiguous_top1_case": int(max(continuation_margin, restart_margin) <= 1e-6),
                                    "continuation_top1_margin": float(continuation_margin),
                                    "restart_top1_margin": float(restart_margin),
                                    "policy_l1_restart_vs_continuation": float(
                                        np.abs(restart_policy - continuation_policy).sum()
                                    ),
                                    "policy_kl_restart_vs_continuation": float(
                                        posthoc.kl_divergence(restart_policy, continuation_policy)
                                    ),
                                    "continuation_decision_notes": list(continuation_meta.get("decision_notes", [])),
                                    "restart_decision_notes": list(restart_meta.get("decision_notes", [])),
                                }
                            )
        finally:
            harness.close()

    summary = summarize_rows(benchmark_rows)
    bundle_summary = summarize_bundle_runs(bundle_runs, benchmark_rows)
    gate = evaluate_benchmark_gate(
        summary,
        bundle_summary,
        min_bundle_speedup=float(args.min_bundle_speedup),
        min_tie_aware_match=float(args.min_tie_aware_match),
        max_kl_mean=float(args.max_kl_mean),
    )
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "contract_summary": contract_summary,
        "game": args.game,
        "device": str(device),
        "rust_binary": args.rust_binary,
        "positions_file": args.positions_file,
        "positions_count": len(positions),
        "trace_cache_salt": posthoc.trace_cache_salt(),
        "systems": [system.id for system in systems],
        "budgets": budgets,
        "seed": int(args.seed),
        "repeats": int(args.repeats),
        "warmup_rounds": int(args.warmup_rounds),
        "summary": summary,
        "bundle_summary": bundle_summary,
        "gate": gate,
    }
    posthoc.jsonl_dump(base_dir / "phase15_continuation_benchmark_rows.jsonl", benchmark_rows)
    posthoc.json_dump(base_dir / "phase15_continuation_benchmark_summary.json", payload)
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(json.dumps({"bundle_summary": payload["bundle_summary"], "gate": payload["gate"]}, indent=2), flush=True)
    if args.enforce_gate and not bool(gate["passed"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
