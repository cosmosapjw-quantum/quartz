#!/usr/bin/env python3
"""Mine a bucket-balanced suite for phase 1.5 assays."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import phase15_ablation_study as runner

from quartz.phase15_suite import (
    bucket_counts,
    bucket_thresholds,
    mine_balanced_suite,
    split_suite_policy_artifacts,
    write_suite_policy_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine a bucket-balanced phase 1.5 suite"
    )
    parser.add_argument("--game", default="gomoku7")
    parser.add_argument("--output", default="results/phase15_suite_mining")
    parser.add_argument("--checkpoints", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--max-checkpoints", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rust-binary", default="./target/release/mcts_demo")
    parser.add_argument("--reference-checkpoint", default=None)
    parser.add_argument("--oracle-checkpoint", default=None)
    parser.add_argument("--reference-system", default="A0")
    parser.add_argument("--oracle-system", default=None)
    parser.add_argument("--oracle-profile", default="baseline_strict")
    parser.add_argument("--budgets", default="8,16,32,64")
    parser.add_argument("--oracle-budget", type=int, default=256)
    parser.add_argument("--systems-config", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suite-size", type=int, default=96)
    parser.add_argument("--candidate-count", type=int, default=384)
    parser.add_argument("--bucket-min-count", type=int, default=4)
    parser.add_argument("--position-min-moves", type=int, default=None)
    parser.add_argument("--position-max-moves", type=int, default=None)
    parser.add_argument("--confident-threshold", type=float, default=0.55)
    parser.add_argument("--ambiguous-margin", type=float, default=0.10)
    parser.add_argument("--root-conflict-topk", type=int, default=2)
    parser.add_argument("--deep-conflict-topk", type=int, default=2)
    parser.add_argument("--positions-out", default=None)
    parser.add_argument("--search-stall-timeout-s", type=float, default=45.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.output) / args.game
    base_dir.mkdir(parents=True, exist_ok=True)
    base_cfg, device = runner.sweep.build_base_cfg(args.game, args.device)
    checkpoints = runner.resolve_checkpoint_refs(args, base_dir)
    runner.validate_checkpoint_refs(args, checkpoints)
    systems = runner.load_systems_config(args.systems_config, base_cfg)
    reference_system = runner.require_system(systems, args.reference_system)
    reference_checkpoint = runner.choose_reference_checkpoint(
        checkpoints, args.reference_checkpoint
    )
    oracle_checkpoint = runner.choose_checkpoint(
        checkpoints, args.oracle_checkpoint, reference_checkpoint
    )
    oracle_system = runner.build_oracle_system(
        systems,
        oracle_system_id=args.oracle_system,
        oracle_profile=args.oracle_profile,
        reference_system=reference_system,
    )
    thresholds = bucket_thresholds(
        confident_threshold=float(args.confident_threshold),
        ambiguous_margin=float(args.ambiguous_margin),
        root_conflict_topk=int(args.root_conflict_topk),
        deep_conflict_topk=int(args.deep_conflict_topk),
    )
    positions = runner.load_or_generate_positions(
        args, base_cfg, count=int(args.candidate_count)
    )

    ref_harness = runner.FrozenCheckpointHarness(
        reference_checkpoint, base_cfg, device, args.rust_binary
    )
    oracle_harness = (
        ref_harness
        if oracle_checkpoint.path == reference_checkpoint.path
        else runner.FrozenCheckpointHarness(
            oracle_checkpoint, base_cfg, device, args.rust_binary
        )
    )
    try:
        annotated = runner.prepare_bucketized_suite(
            ref_harness,
            oracle_harness,
            positions,
            reference_system=reference_system,
            oracle_system=oracle_system,
            low_budget=min(runner.parse_csv_ints(args.budgets)),
            oracle_budget=int(args.oracle_budget),
            bucket_thresholds=thresholds,
        )
    finally:
        ref_harness.close()
        if oracle_harness is not ref_harness:
            oracle_harness.close()

    suite = mine_balanced_suite(
        annotated,
        suite_size=int(args.suite_size),
        bucket_min_count=int(args.bucket_min_count),
        seed=int(args.seed),
    )
    out_path = Path(args.positions_out or (base_dir / "mined_suite.json"))
    artifact_path = out_path.with_name(f"{out_path.stem}_artifacts.npz")
    compact_suite, suite_artifacts = split_suite_policy_artifacts(suite)
    write_suite_policy_artifacts(artifact_path, suite_artifacts)
    runner.json_dump(
        out_path,
        {
            "game": args.game,
            "reference_checkpoint": runner.asdict(reference_checkpoint),
            "oracle_checkpoint": runner.asdict(oracle_checkpoint),
            "reference_system": runner.asdict(reference_system),
            "oracle_system": runner.asdict(oracle_system),
            "bucket_thresholds": thresholds,
            "bucket_counts": bucket_counts(suite),
            "suite_artifacts_file": artifact_path.name,
            "positions": compact_suite,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    print(f"\nWrote mined phase15 suite to {out_path}", flush=True)


if __name__ == "__main__":
    main()
