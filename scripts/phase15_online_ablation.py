#!/usr/bin/env python3
"""Online chunked phase 1.5 ablation runner."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import phase15_ablation_study as posthoc

from quartz.phase15_online import run_online_readout
from quartz.phase15_suite import bucket_counts, bucket_thresholds, mine_balanced_suite


def build_online_trace_lookup(
    harness,
    checkpoint,
    position,
    system,
    trace_budgets,
    cache_dir,
):
    trace_policies, trace_latencies_ms, trace_reused = posthoc.build_search_trace(
        harness,
        checkpoint,
        position,
        system,
        trace_budgets,
        cache_dir,
    )
    rows = {}
    for budget, policy, latency_ms in zip(trace_budgets, trace_policies, trace_latencies_ms, strict=False):
        rows[int(budget)] = {
            "search_policy": posthoc.normalize_policy(policy).tolist(),
            "latency_ms": float(latency_ms),
        }
    return rows, bool(trace_reused)


def build_online_trace_bundle(
    harness,
    checkpoint,
    position,
    system,
    budgets,
    cache_dir,
):
    ordered_budgets = [int(budget) for budget in sorted({int(item) for item in budgets})]
    try:
        client = harness._get_client(system, int(ordered_budgets[0]))
        return run_online_readout_continuation(
            client,
            position,
            system,
            ordered_budgets,
            int(ordered_budgets[-1]),
        ), False, "root_continuation", None
    except Exception as exc:
        rows, reused = build_online_trace_lookup(
            harness,
            checkpoint,
            position,
            system,
            ordered_budgets,
            cache_dir,
        )
        return rows, bool(reused), "restart_per_chunk", f"{type(exc).__name__}: {exc}"


def build_position_job(position: dict[str, object]) -> dict[str, object]:
    job: dict[str, object] = {}
    if "fen" in position:
        job["fen"] = position["fen"]
    elif "board" in position:
        job["board"] = position["board"]
        job["player"] = int(position.get("player", 1))
    if "state_meta" in position and isinstance(position["state_meta"], dict):
        job.update(position["state_meta"])
    return job


def run_online_readout_continuation(client, position, system, trace_budgets, target_budget):
    from quartz.replay import dense_policy_from_sparse

    session_id = None
    prev_budget = 0
    trace_rows = {}
    try:
        open_t0 = time.perf_counter()
        open_payload = client.open_search_engine_session(
            [build_position_job(position)],
            penalty_mode=client.cfg.get("penalty_mode", "None"),
            iters=int(trace_budgets[0]),
        )
        open_elapsed_ms = (time.perf_counter() - open_t0) * 1000.0
        session_id = open_payload.get("session_id") if isinstance(open_payload, dict) else None
        results = open_payload.get("results", []) if isinstance(open_payload, dict) else []
        if session_id is None or not isinstance(results, list) or not results:
            raise RuntimeError("engine session open failed")
        first_row = dict(results[0])
        first_row["search_policy"] = posthoc.normalize_policy(
            dense_policy_from_sparse(first_row.get("policy", []), int(client.cfg["actions"]))
        ).tolist()
        first_row["latency_ms"] = float(first_row.get("latency_ms", open_elapsed_ms))
        trace_rows[int(trace_budgets[0])] = first_row
        prev_budget = int(trace_budgets[0])
        for budget in trace_budgets[1:]:
            delta = int(budget) - int(prev_budget)
            step_t0 = time.perf_counter()
            payload = client.step_search_engine_session(session_id, updates=[{}], iters=delta)
            step_elapsed_ms = (time.perf_counter() - step_t0) * 1000.0
            results = payload.get("results", []) if isinstance(payload, dict) else []
            if not isinstance(results, list) or not results:
                raise RuntimeError("engine session step failed")
            row = dict(results[0])
            row["search_policy"] = posthoc.normalize_policy(
                dense_policy_from_sparse(row.get("policy", []), int(client.cfg["actions"]))
            ).tolist()
            row["latency_ms"] = float(row.get("latency_ms", step_elapsed_ms))
            trace_rows[int(budget)] = row
            prev_budget = int(budget)
        return trace_rows
    finally:
        if session_id is not None:
            try:
                client.close_search_session(session_id)
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.5 online chunked ablation runner")
    parser.add_argument("--game", default="gomoku7")
    parser.add_argument("--output", default="results/phase15_online_ablation")
    parser.add_argument("--checkpoints", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--max-checkpoints", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rust-binary", default="./target/release/mcts_demo")
    parser.add_argument("--positions-file", default=None)
    parser.add_argument("--suite-size", type=int, default=96)
    parser.add_argument("--suite-source", choices=["random", "mined"], default="random")
    parser.add_argument("--suite-candidate-multiplier", type=int, default=4)
    parser.add_argument("--bucket-min-count", type=int, default=4)
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
    parser.add_argument("--systems", default="B1,B2,B3,C0,C1,C2")
    parser.add_argument("--groups", default="B,C")
    parser.add_argument("--confident-threshold", type=float, default=0.55)
    parser.add_argument("--ambiguous-margin", type=float, default=0.10)
    parser.add_argument("--root-conflict-topk", type=int, default=2)
    parser.add_argument("--deep-conflict-topk", type=int, default=2)
    parser.add_argument("--trace-cache-dir", default=None)
    parser.add_argument("--disable-trace-cache", action="store_true")
    parser.add_argument("--search-stall-timeout-s", type=float, default=45.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["QUARTZ_SEARCH_STALL_TIMEOUT_S"] = str(float(args.search_stall_timeout_s))
    base_dir = Path(args.output) / args.game
    base_dir.mkdir(parents=True, exist_ok=True)

    base_cfg, device = posthoc.sweep.build_base_cfg(args.game, args.device)
    base_cfg["seed"] = int(args.seed)
    checkpoints = posthoc.resolve_checkpoint_refs(args, base_dir)
    posthoc.validate_checkpoint_refs(args, checkpoints)
    all_systems = posthoc.load_systems_config(args.systems_config, base_cfg)
    selected_ids = set(posthoc.parse_csv_strings(args.systems))
    selected_groups = set(posthoc.parse_csv_strings(args.groups))
    systems = [system for system in all_systems if system.id in selected_ids and system.group in selected_groups]
    if not systems:
        raise ValueError("no online systems selected")

    reference_system = posthoc.require_system(all_systems, args.reference_system)
    reference_checkpoint = posthoc.choose_reference_checkpoint(checkpoints, args.reference_checkpoint)
    oracle_checkpoint = posthoc.choose_checkpoint(checkpoints, args.oracle_checkpoint, reference_checkpoint)
    oracle_system = posthoc.build_oracle_system(
        all_systems,
        oracle_system_id=args.oracle_system,
        oracle_profile=args.oracle_profile,
        reference_system=reference_system,
    )
    trace_cache_dir = None if args.disable_trace_cache else Path(args.trace_cache_dir or (base_dir / "trace_cache"))

    candidate_count = int(args.suite_size)
    if not args.positions_file and args.suite_source == "mined":
        candidate_count = max(int(args.suite_size), int(args.suite_size) * int(args.suite_candidate_multiplier))
    positions = posthoc.load_or_generate_positions(args, base_cfg, count=candidate_count)
    thresholds = bucket_thresholds(
        confident_threshold=float(args.confident_threshold),
        ambiguous_margin=float(args.ambiguous_margin),
        root_conflict_topk=int(args.root_conflict_topk),
        deep_conflict_topk=int(args.deep_conflict_topk),
    )

    ref_harness = posthoc.FrozenCheckpointHarness(reference_checkpoint, base_cfg, device, args.rust_binary)
    oracle_harness = ref_harness if oracle_checkpoint.path == reference_checkpoint.path else posthoc.FrozenCheckpointHarness(
        oracle_checkpoint, base_cfg, device, args.rust_binary
    )
    try:
        annotated = posthoc.prepare_bucketized_suite(
            ref_harness,
            oracle_harness,
            positions,
            reference_system=reference_system,
            oracle_system=oracle_system,
            low_budget=min(posthoc.parse_csv_ints(args.budgets)),
            oracle_budget=int(args.oracle_budget),
            bucket_thresholds=thresholds,
        )
    finally:
        ref_harness.close()
        if oracle_harness is not ref_harness:
            oracle_harness.close()

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

    manifest = {
        "format_version": 1,
        "execution_mode": "online_chunked_root_continuation",
        "search_continuation": "root_continuation_preferred",
        "limitations": [
            "The runner prefers resident root continuation and falls back to restart_per_chunk only on protocol failure.",
            "Rows record the actual search_continuation mode used for each assay.",
        ],
        "game": args.game,
        "device": str(device),
        "checkpoints": [posthoc.asdict(ref) for ref in checkpoints],
        "reference_checkpoint": posthoc.asdict(reference_checkpoint),
        "oracle_checkpoint": posthoc.asdict(oracle_checkpoint),
        "reference_system": posthoc.asdict(reference_system),
        "oracle_system": posthoc.asdict(oracle_system),
        "systems": [posthoc.asdict(system) for system in systems],
        "budgets": posthoc.parse_csv_ints(args.budgets),
        "oracle_budget": int(args.oracle_budget),
        "suite_source": "file" if args.positions_file else args.suite_source,
        "bucket_counts": bucket_counts(suite),
        "trace_cache_dir": None if trace_cache_dir is None else str(trace_cache_dir),
        "trace_cache_salt": posthoc.trace_cache_salt(),
        "seed": int(args.seed),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    phase15_contracts = posthoc.build_phase15_contracts(
        execution_mode="online_chunked_root_continuation",
        game=args.game,
        checkpoints=checkpoints,
        systems=systems,
        budgets=posthoc.parse_csv_ints(args.budgets),
        trace_cache_salt_value=posthoc.trace_cache_salt(),
        reference_checkpoint=reference_checkpoint,
        reference_system=reference_system,
        oracle_checkpoint=oracle_checkpoint,
        oracle_system=oracle_system,
        extra={
            "oracle_budget": int(args.oracle_budget),
            "suite_source": "file" if args.positions_file else args.suite_source,
            "seed": int(args.seed),
            "search_continuation": "root_continuation_preferred",
        },
    )
    manifest["contract_summary"] = posthoc.summarize_phase15_contracts(phase15_contracts)
    posthoc.json_dump(base_dir / "phase15_online_manifest.json", manifest)

    rows = []
    trace_cache_hits = 0
    trace_cache_misses = 0
    continuation_traces = 0
    continuation_fallback_traces = 0
    continuation_fallback_reasons: dict[str, int] = {}
    budgets = posthoc.parse_csv_ints(args.budgets)
    for checkpoint in checkpoints:
        harness = posthoc.FrozenCheckpointHarness(checkpoint, base_cfg, device, args.rust_binary)
        reference_eval = harness if checkpoint.path == reference_checkpoint.path else posthoc.FrozenCheckpointHarness(
            reference_checkpoint, base_cfg, device, args.rust_binary
        )
        oracle_eval = reference_eval if reference_checkpoint.path == oracle_checkpoint.path else (
            harness if checkpoint.path == oracle_checkpoint.path else posthoc.FrozenCheckpointHarness(
                oracle_checkpoint, base_cfg, device, args.rust_binary
            )
        )
        try:
            harness.prime_prior_cache(suite)
            if reference_eval is not harness:
                reference_eval.prime_prior_cache(suite)
            if oracle_eval not in {harness, reference_eval}:
                oracle_eval.prime_prior_cache(suite)
            for position in suite:
                prior_input = harness.prior_policy(position)
                reference_policy = posthoc.suite_policy_artifact(position, "reference_policy")
                if reference_policy is None:
                    reference_policy = posthoc.np.asarray(
                        reference_eval.search_policy(position, reference_system, int(args.oracle_budget))["search_policy"],
                        dtype=posthoc.np.float32,
                    )
                oracle_policy = posthoc.suite_policy_artifact(position, "oracle_policy")
                if oracle_policy is None:
                    oracle_policy = posthoc.np.asarray(
                        oracle_eval.search_policy(position, oracle_system, int(args.oracle_budget))["search_policy"],
                        dtype=posthoc.np.float32,
                    )
                for system in systems:
                    trace_rows, trace_reused, continuation_mode, fallback_reason = build_online_trace_bundle(
                        harness,
                        checkpoint,
                        position,
                        system,
                        budgets,
                        trace_cache_dir,
                    )
                    if continuation_mode == "root_continuation":
                        continuation_traces += 1
                    else:
                        continuation_fallback_traces += 1
                        if fallback_reason:
                            continuation_fallback_reasons[fallback_reason] = (
                                continuation_fallback_reasons.get(fallback_reason, 0) + 1
                            )
                        if trace_reused:
                            trace_cache_hits += 1
                        else:
                            trace_cache_misses += 1
                    for budget in budgets:
                        trace_budgets = posthoc.make_trace_budgets(
                            budget,
                            budgets,
                            allow_extra=(system.refresh_operator == "budget_routing"),
                        )
                        final_policy, trace_meta = run_online_readout(
                            system=system,
                            position=position,
                            prior_input=prior_input,
                            budgets=trace_budgets,
                            target_budget=int(budget),
                            search_policy_fn=lambda _position, _system, budget_value, rows=trace_rows: dict(
                                rows[int(budget_value)]
                            ),
                        )
                        trace_meta["search_continuation"] = continuation_mode
                        if fallback_reason:
                            trace_meta["continuation_fallback_reason"] = fallback_reason
                        rows.append(
                            posthoc.build_row(
                                checkpoint,
                                position,
                                system,
                                budget,
                                prior_input,
                                posthoc.np.asarray(final_policy, dtype=posthoc.np.float32),
                                reference_policy,
                                oracle_policy,
                                trace_meta,
                                trace_reused=False,
                            )
                        )
        finally:
            harness.close()
            if reference_eval is not harness:
                reference_eval.close()
            if oracle_eval not in {harness, reference_eval}:
                oracle_eval.close()

    posthoc.jsonl_dump(base_dir / "assays" / "phase15_online_rows.jsonl", rows)
    posthoc.json_dump(
        base_dir / "phase15_online_summary.json",
        {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "contract_summary": posthoc.summarize_phase15_contracts(phase15_contracts),
            "raw_summary": posthoc.build_summary_payload(rows),
            "semantic_summary": posthoc.build_semantic_summary_payload(rows),
            "headwind_summary": posthoc.build_headwind_summary_payload(rows),
            "trace_cache_stats": {
                "trace_cache_unit": "trace_bundle",
                "trace_bundle_cache_hits": int(trace_cache_hits),
                "trace_bundle_cache_misses": int(trace_cache_misses),
                "trace_bundle_cache_hit_rate": float(trace_cache_hits / max(1, trace_cache_hits + trace_cache_misses)),
                "trace_cache_hits": int(trace_cache_hits),
                "trace_cache_misses": int(trace_cache_misses),
                "trace_cache_hit_rate": float(trace_cache_hits / max(1, trace_cache_hits + trace_cache_misses)),
            },
            "continuation_stats": {
                "root_continuation_traces": int(continuation_traces),
                "restart_fallback_traces": int(continuation_fallback_traces),
                "budget_rows_emitted": int(len(rows)),
                "fallback_reasons": continuation_fallback_reasons,
            },
        },
    )
    print(f"\nPrepared online phase15 assays in {base_dir}", flush=True)


if __name__ == "__main__":
    main()
