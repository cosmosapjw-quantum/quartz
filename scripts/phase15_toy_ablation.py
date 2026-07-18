#!/usr/bin/env python3
# ruff: noqa: E402
"""Self-contained toy/small Phase 1.5 candidate ablation runner.

Restored after the tracking rewrite dropped the original uncommitted WIP.
Behavior is pinned by ``tests/test_phase15_ablation.py``
(``test_phase15_toy_ablation_builds_posthoc_and_benchmark_candidate_commands``):
both the posthoc and benchmark sub-commands must receive the ``--systems``
preset EXPANDED to the candidate CSV, and ``--enforce-gate`` must be omitted
unless explicitly requested.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import phase15_benchmark_ci_smoke as smoke

from quartz.phase15_ablation import phase15_systems_csv, resolve_phase15_systems_arg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic toy Phase15 ablation over current candidates"
    )
    parser.add_argument("--game", default="gomoku7")
    parser.add_argument("--output", default="results/phase15_toy_ablation")
    parser.add_argument("--rust-binary", default="./target/release/mcts_demo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--systems", default=phase15_systems_csv("small"))
    parser.add_argument("--groups", default="A,B")
    parser.add_argument("--budgets", default="8,16,32,64")
    parser.add_argument("--oracle-budget", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-positions", type=int, default=4)
    parser.add_argument(
        "--run", choices=["posthoc", "benchmark", "both"], default="both"
    )
    parser.add_argument("--benchmark-repeats", type=int, default=1)
    parser.add_argument("--benchmark-warmup-rounds", type=int, default=0)
    parser.add_argument("--enforce-benchmark-gate", action="store_true")
    parser.add_argument("--search-stall-timeout-s", type=float, default=180.0)
    return parser.parse_args()


def build_posthoc_command(
    args: argparse.Namespace, checkpoint_path: Path, positions_path: Path
) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "phase15_ablation_study.py"),
        "--game",
        str(args.game),
        "--output",
        str(Path(args.output) / "posthoc"),
        "--checkpoints",
        str(checkpoint_path),
        "--reference-checkpoint",
        str(checkpoint_path),
        "--oracle-checkpoint",
        str(checkpoint_path),
        "--positions-file",
        str(positions_path),
        "--suite-size",
        str(int(args.max_positions)),
        "--systems",
        phase15_systems_csv(args.systems),
        "--groups",
        str(args.groups),
        "--budgets",
        str(args.budgets),
        "--oracle-budget",
        str(int(args.oracle_budget)),
        "--seed",
        str(int(args.seed)),
        "--device",
        str(args.device),
        "--rust-binary",
        str(args.rust_binary),
        "--search-stall-timeout-s",
        str(float(args.search_stall_timeout_s)),
    ]


def build_benchmark_command(
    args: argparse.Namespace, checkpoint_path: Path, positions_path: Path
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "phase15_benchmark.py"),
        "--game",
        str(args.game),
        "--output",
        str(Path(args.output) / "benchmark"),
        "--checkpoints",
        str(checkpoint_path),
        "--positions-file",
        str(positions_path),
        "--max-positions",
        str(int(args.max_positions)),
        "--systems",
        phase15_systems_csv(args.systems),
        "--budgets",
        str(args.budgets),
        "--device",
        str(args.device),
        "--repeats",
        str(int(args.benchmark_repeats)),
        "--warmup-rounds",
        str(int(args.benchmark_warmup_rounds)),
        "--search-stall-timeout-s",
        str(float(args.search_stall_timeout_s)),
        "--rust-binary",
        str(args.rust_binary),
    ]
    if bool(args.enforce_benchmark_gate):
        command.append("--enforce-gate")
    return command


def build_toy_manifest(
    args: argparse.Namespace,
    *,
    checkpoint_path: Path,
    positions_path: Path,
    commands: dict[str, list[str]],
) -> dict[str, Any]:
    root = Path(args.output)
    return {
        "format_version": 1,
        "execution_mode": "toy_small_ablation",
        "claim_status": "SMOKE-ONLY; performance and quality claims remain ABLATION-PENDING",
        "limitations": [
            "This runner uses one deterministic randomly initialized checkpoint.",
            "The fixed position suite is intentionally tiny and is not a validation benchmark.",
            "Use it to check candidate plumbing, telemetry, manifest stability, and gross runtime failures.",
        ],
        "game": str(args.game),
        "systems_arg": str(args.systems),
        "systems": list(resolve_phase15_systems_arg(str(args.systems))),
        "groups": str(args.groups),
        "budgets": str(args.budgets),
        "oracle_budget": int(args.oracle_budget),
        "seed": int(args.seed),
        "max_positions": int(args.max_positions),
        "run": str(args.run),
        "benchmark_gate_enforced": bool(args.enforce_benchmark_gate),
        "fixtures": {
            "checkpoint_path": str(checkpoint_path),
            "positions_path": str(positions_path),
        },
        "artifacts": {
            "posthoc_summary_path": str(
                root / "posthoc" / str(args.game) / "phase15_summary.json"
            ),
            "benchmark_summary_path": str(
                root
                / "benchmark"
                / str(args.game)
                / "phase15_continuation_benchmark_summary.json"
            ),
        },
        "commands": commands,
    }


def main() -> None:
    args = parse_args()
    base_dir = Path(args.output) / str(args.game)
    base_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path, positions_path = smoke.write_ci_smoke_inputs(
        base_dir, game=str(args.game), seed=int(args.seed)
    )
    commands: dict[str, list[str]] = {}
    if args.run in {"posthoc", "both"}:
        commands["posthoc"] = build_posthoc_command(
            args, checkpoint_path, positions_path
        )
    if args.run in {"benchmark", "both"}:
        commands["benchmark"] = build_benchmark_command(
            args, checkpoint_path, positions_path
        )

    manifest = build_toy_manifest(
        args,
        checkpoint_path=checkpoint_path,
        positions_path=positions_path,
        commands=commands,
    )
    manifest_path = base_dir / "phase15_toy_ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for command in commands.values():
        subprocess.run(command, check=True)

    print(
        json.dumps(
            {"manifest_path": str(manifest_path), "commands": commands}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
