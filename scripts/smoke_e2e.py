#!/usr/bin/env python3
"""One-command end-to-end smoke for external audit and local verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd))
    merged_env = None
    if env:
        merged_env = os.environ.copy()
        merged_env.update(env)
    subprocess.run(cmd, cwd=cwd, check=True, env=merged_env)


def sha256_file_prefix(path: Path, prefix_len: int = 16) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:prefix_len]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal QUARTZ end-to-end smoke.")
    parser.add_argument("--rust-binary", default="./target/release/mcts_demo")
    parser.add_argument("--output", default="results/audit_e2e_smoke")
    parser.add_argument("--game", default="gomoku7")
    parser.add_argument("--study", default="search_vl")
    parser.add_argument(
        "--conditions",
        default="T1_noS_noVL,T2_S_noVL",
        help="Comma-separated train conditions for the smoke subset.",
    )
    parser.add_argument(
        "--eval-conditions",
        default="E1_noS_noVL,E2_S_noVL",
        help="Comma-separated eval conditions for the smoke subset.",
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--games-per-iter", type=int, default=4)
    parser.add_argument("--eval-games", type=int, default=2)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=999999,
        help="Training-time checkpoint eval interval; large default skips internal promotion eval in smoke runs.",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--timeout-hours", type=int, default=1)
    parser.add_argument("--keep-output", action="store_true")
    parser.add_argument(
        "--no-autotune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable autotune warmup inside the smoke ablation.",
    )
    parser.add_argument(
        "--include-strict-reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include the baseline_strict eval reference in the smoke ablation.",
    )
    parser.add_argument(
        "--disable-torch-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force eager inference in smoke subprocesses to avoid long inductor warmup.",
    )
    parser.add_argument(
        "--eval-stall-timeout-s",
        type=float,
        default=45.0,
        help="Fail fast if Rust shared-eval stops making progress during the smoke.",
    )
    parser.add_argument(
        "--build-rust-if-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attempt `cargo build --release --bin mcts_demo` when the Rust binary is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    rust_binary = (repo_root / args.rust_binary).resolve() if not Path(args.rust_binary).is_absolute() else Path(args.rust_binary)
    if not rust_binary.exists():
        if args.build_rust_if_missing:
            run(["cargo", "build", "--release", "--bin", "mcts_demo"], repo_root)
        if not rust_binary.exists():
            raise SystemExit(
                "Rust binary not found at "
                f"{rust_binary}. This smoke is source-level reproducible, not prebuilt-binary self-contained. "
                "Build with `cargo build --release --bin mcts_demo` or rerun with a valid `--rust-binary`."
            )

    output_root = repo_root / args.output
    if output_root.exists() and not args.keep_output:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    smoke_contract = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(repo_root),
        "rust_binary": str(rust_binary),
        "rust_binary_exists": bool(rust_binary.exists()),
        "rust_binary_sha256": sha256_file_prefix(rust_binary),
        "study": args.study,
        "conditions": args.conditions,
        "eval_conditions": args.eval_conditions,
        "game": args.game,
        "iterations": int(args.iterations),
        "games_per_iter": int(args.games_per_iter),
        "eval_games": int(args.eval_games),
        "eval_interval": int(args.eval_interval),
        "seed": int(args.seed),
        "timeout_hours": int(args.timeout_hours),
        "no_autotune": bool(args.no_autotune),
        "include_strict_reference": bool(args.include_strict_reference),
        "disable_torch_compile": bool(args.disable_torch_compile),
        "eval_stall_timeout_s": float(args.eval_stall_timeout_s),
        "artifact_contract": {
            "required_outputs": [
                f"{args.game}/study_manifest.json",
                f"{args.game}/evaluation_matrix.json",
                f"{args.game}/ablation_report.json",
            ],
            "mode": "source_level_smoke",
        },
    }
    (output_root / "smoke_contract.json").write_text(json.dumps(smoke_contract, indent=2), encoding="utf-8")

    run([sys.executable, "-m", "quartz.train", "--help"], repo_root)
    run([sys.executable, "-c", "from quartz.evaluation import _run_all; _run_all()"], repo_root)
    ablation_env = {}
    if args.disable_torch_compile:
        ablation_env["QUARTZ_NO_COMPILE"] = "1"
        ablation_env["QUARTZ_DISABLE_COMPILE"] = "1"
    if args.eval_stall_timeout_s > 0:
        ablation_env["QUARTZ_EVAL_STALL_TIMEOUT_S"] = str(float(args.eval_stall_timeout_s))
    ablation_cmd = [
        sys.executable,
        "scripts/ablation_study.py",
        "--study",
        args.study,
        "--conditions",
        args.conditions,
        "--eval-conditions",
        args.eval_conditions,
        "--game",
        args.game,
        "--iterations",
        str(args.iterations),
        "--games-per-iter",
        str(args.games_per_iter),
        "--eval-games",
        str(args.eval_games),
        "--eval-interval",
        str(args.eval_interval),
        "--seeds",
        str(args.seed),
        "--paired-seed-eval",
        "--resident-session",
        "--timeout-hours",
        str(args.timeout_hours),
        "--rust-binary",
        str(rust_binary),
        "--output",
        str(output_root),
    ]
    if args.no_autotune:
        ablation_cmd.append("--no-autotune")
    if args.include_strict_reference:
        ablation_cmd.append("--include-strict-reference")
    run(ablation_cmd, repo_root, env=ablation_env)
    report_dir = output_root / args.game
    expected = [
        report_dir / "study_manifest.json",
        report_dir / "evaluation_matrix.json",
        report_dir / "ablation_report.json",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise SystemExit(f"Smoke completed but expected artifacts are missing: {', '.join(missing)}")
    print(f"Smoke complete. Artifacts: {report_dir}")
    print(f"Smoke contract: {output_root / 'smoke_contract.json'}")


if __name__ == "__main__":
    main()
