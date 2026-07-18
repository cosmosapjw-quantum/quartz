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
import threading
import time
from pathlib import Path


def jsonl_append(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))


def smoke_artifact_paths(output_root: Path) -> dict[str, Path]:
    return {
        "stdout_log": output_root / "stdout.log",
        "stderr_log": output_root / "stderr.log",
        "events_jsonl": output_root / "events.jsonl",
        "python_trace_jsonl": output_root / "python_trace.jsonl",
        "rust_server_trace_jsonl": output_root / "rust_server_trace.jsonl",
        "summary_json": output_root / "summary.json",
    }


def _tee_stream(stream, sink, mirror) -> None:
    try:
        for line in iter(stream.readline, ""):
            sink.write(line)
            sink.flush()
            mirror.write(line)
            mirror.flush()
    finally:
        try:
            stream.close()
        except Exception:
            pass


def run(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
    *,
    label: str,
    artifact_paths: dict[str, Path] | None = None,
) -> None:
    print("+", " ".join(cmd))
    merged_env = None
    if env:
        merged_env = os.environ.copy()
        merged_env.update(env)
    stdout_log = artifact_paths["stdout_log"] if artifact_paths else None
    stderr_log = artifact_paths["stderr_log"] if artifact_paths else None
    events_jsonl = artifact_paths["events_jsonl"] if artifact_paths else None
    event = {
        "ts": time.time(),
        "label": str(label),
        "cmd": list(cmd),
        "cwd": str(cwd),
    }
    if events_jsonl is not None:
        jsonl_append(events_jsonl, {"event": "command_begin", **event})
    if stdout_log is None or stderr_log is None:
        subprocess.run(cmd, cwd=cwd, check=True, env=merged_env)
        if events_jsonl is not None:
            jsonl_append(
                events_jsonl, {"event": "command_end", **event, "returncode": 0}
            )
        return

    with (
        stdout_log.open("a", encoding="utf-8") as stdout_handle,
        stderr_log.open("a", encoding="utf-8") as stderr_handle,
    ):
        stdout_handle.write(f"\n$ {' '.join(cmd)}\n")
        stderr_handle.write(f"\n$ {' '.join(cmd)}\n")
        stdout_handle.flush()
        stderr_handle.flush()
        t0 = time.time()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stdout_thread = threading.Thread(
            target=_tee_stream,
            args=(proc.stdout, stdout_handle, sys.stdout),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_tee_stream,
            args=(proc.stderr, stderr_handle, sys.stderr),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        returncode = proc.wait()
        stdout_thread.join()
        stderr_thread.join()
        if events_jsonl is not None:
            jsonl_append(
                events_jsonl,
                {
                    "event": "command_end",
                    **event,
                    "elapsed_s": round(max(0.0, time.time() - t0), 3),
                    "returncode": int(returncode),
                },
            )
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, cmd)


def sha256_file_prefix(path: Path, prefix_len: int = 16) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:prefix_len]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal QUARTZ end-to-end smoke."
    )
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
    # P3 (audit_codex_20260425.md W6): bumped from 4 → 16 → 64. On gomoku7
    # random-init games yield ~5 samples/game, so 64 games × 5 ≈ 320 samples
    # — comfortably above the default train batch=256. Combined with the
    # `--no-pipeline` default below (inline self-play that blocks until all
    # `--games` are generated before checking the batch threshold), iter 1
    # reliably crosses batch and `verify_training_fired` sees ≥1 SGD row.
    # Concurrent mode would not work here: it only waits for `min_new=1` per
    # iteration and would skip SGD with iterations=1.
    parser.add_argument("--games-per-iter", type=int, default=64)
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
    parser.add_argument(
        "--safe-runtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use conservative smoke defaults: disable resident session unless explicitly re-enabled and force eager eval.",
    )
    parser.add_argument(
        "--resident-session",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override resident-session usage inside the smoke ablation.",
    )
    parser.add_argument(
        "--no-pipeline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Force inline self-play in quartz.train so a single short iteration fills "
            "replay past the SGD batch threshold. The concurrent path waits only for "
            "min_new=1 per iter and won't accumulate to batch=256 with iterations=1."
        ),
    )
    return parser.parse_args()


def build_ablation_command(
    args: argparse.Namespace, rust_binary: Path, output_root: Path
) -> list[str]:
    cmd = [
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
        "--timeout-hours",
        str(args.timeout_hours),
        "--rust-binary",
        str(rust_binary),
        "--output",
        str(output_root),
    ]
    resident_session = (
        bool(args.resident_session)
        if args.resident_session is not None
        else not bool(args.safe_runtime)
    )
    if resident_session:
        cmd.append("--resident-session")
    if args.no_autotune:
        cmd.append("--no-autotune")
    if args.include_strict_reference:
        cmd.append("--include-strict-reference")
    if getattr(args, "no_pipeline", True):
        cmd.append("--no-pipeline")
    return cmd


def count_sgd_rows(train_log: Path) -> int:
    """Count rows in `train_log.jsonl` where `loss` is a non-null number.

    P3 (audit_codex_20260425.md W6): a smoke that produces zero SGD
    rows certifies imports/transport, not training. This helper is
    consumed by `verify_training_fired` to fail-fast when the smoke
    parameters silently keep replay below the SGD batch threshold.
    """
    if not train_log.exists():
        return 0
    rows = 0
    with train_log.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if payload.get("loss") is not None:
                rows += 1
    return rows


def verify_training_fired(report_dir: Path) -> tuple[int, list[Path]]:
    """Scan all `train_log.jsonl` files under `report_dir/models/**` and
    return `(total_sgd_rows, scanned_logs)`. Raises `SystemExit` if zero
    SGD rows fired across all conditions.
    """
    models_dir = report_dir / "models"
    scanned: list[Path] = []
    total = 0
    if models_dir.is_dir():
        for log_path in sorted(models_dir.rglob("train_log.jsonl")):
            scanned.append(log_path)
            total += count_sgd_rows(log_path)
    if not scanned:
        raise SystemExit(
            f"P3 smoke contract: no train_log.jsonl files under {models_dir}; "
            "ablation_smoke produced no per-condition training logs."
        )
    if total == 0:
        details = ", ".join(str(p.relative_to(report_dir)) for p in scanned)
        raise SystemExit(
            f"P3 smoke contract: 0 SGD rows across {len(scanned)} train_log.jsonl files "
            f"({details}). Replay never crossed the batch threshold — increase "
            "--games-per-iter or override --batch downward."
        )
    return total, scanned


def build_smoke_summary(
    *,
    args: argparse.Namespace,
    output_root: Path,
    artifact_paths: dict[str, Path],
    rust_binary: Path,
    success: bool,
    missing_outputs: list[str],
    error: str | None = None,
) -> dict[str, object]:
    report_dir = output_root / args.game
    return {
        "success": bool(success),
        "game": str(args.game),
        "study": str(args.study),
        "rust_binary": str(rust_binary),
        "report_dir": str(report_dir),
        "missing_outputs": list(missing_outputs),
        "error": error,
        "artifacts": {name: str(path) for name, path in artifact_paths.items()},
        "trace_counts": {
            "events": count_jsonl_rows(artifact_paths["events_jsonl"]),
            "python_trace": count_jsonl_rows(artifact_paths["python_trace_jsonl"]),
            "rust_server_trace": count_jsonl_rows(
                artifact_paths["rust_server_trace_jsonl"]
            ),
        },
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    rust_binary = (
        (repo_root / args.rust_binary).resolve()
        if not Path(args.rust_binary).is_absolute()
        else Path(args.rust_binary)
    )
    output_root = repo_root / args.output
    if output_root.exists() and not args.keep_output:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_paths = smoke_artifact_paths(output_root)
    for path in artifact_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    if not rust_binary.exists():
        if args.build_rust_if_missing:
            run(
                ["cargo", "build", "--release", "--bin", "mcts_demo"],
                repo_root,
                label="cargo_build",
                artifact_paths=artifact_paths,
            )
        if not rust_binary.exists():
            raise SystemExit(
                "Rust binary not found at "
                f"{rust_binary}. This smoke is source-level reproducible, not prebuilt-binary self-contained. "
                "Build with `cargo build --release --bin mcts_demo` or rerun with a valid `--rust-binary`."
            )

    mpl_config_dir = output_root / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    base_env = {
        "MPLCONFIGDIR": str(mpl_config_dir),
        "QUARTZ_STALL_TRACE_PATH": str(artifact_paths["python_trace_jsonl"]),
        "QUARTZ_RUST_SERVER_TRACE": str(artifact_paths["rust_server_trace_jsonl"]),
    }
    if args.safe_runtime:
        base_env["QUARTZ_SAFE_RUNTIME"] = "1"
        base_env["QUARTZ_DISABLE_ASYNC_PIPELINE"] = "1"
        base_env["QUARTZ_SAFE_BOOTSTRAP_TARGET_CAP"] = "16"
        base_env["QUARTZ_SAFE_SELFPLAY_PARALLEL_CAP"] = "4"
        base_env["QUARTZ_SAFE_SELFPLAY_BATCH_GAMES_CAP"] = "4"

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
        "safe_runtime": bool(args.safe_runtime),
        "resident_session": bool(args.resident_session)
        if args.resident_session is not None
        else None,
        "no_pipeline": bool(args.no_pipeline),
        "safe_runtime_overrides": (
            {
                "bootstrap_target_cap": int(
                    base_env["QUARTZ_SAFE_BOOTSTRAP_TARGET_CAP"]
                ),
                "selfplay_parallel_cap": int(
                    base_env["QUARTZ_SAFE_SELFPLAY_PARALLEL_CAP"]
                ),
                "selfplay_batch_games_cap": int(
                    base_env["QUARTZ_SAFE_SELFPLAY_BATCH_GAMES_CAP"]
                ),
            }
            if args.safe_runtime
            else None
        ),
        "artifact_contract": {
            "required_outputs": [
                f"{args.game}/study_manifest.json",
                f"{args.game}/evaluation_matrix.json",
                f"{args.game}/ablation_report.json",
            ],
            "required_logs": [
                "stdout.log",
                "stderr.log",
                "events.jsonl",
                "python_trace.jsonl",
                "rust_server_trace.jsonl",
                "summary.json",
            ],
            "mode": "source_level_smoke",
        },
    }
    (output_root / "smoke_contract.json").write_text(
        json.dumps(smoke_contract, indent=2), encoding="utf-8"
    )

    summary_error = None
    missing = []
    success = False
    ablation_env = dict(base_env)
    if args.disable_torch_compile:
        ablation_env["QUARTZ_NO_COMPILE"] = "1"
        ablation_env["QUARTZ_DISABLE_COMPILE"] = "1"
    if args.eval_stall_timeout_s > 0:
        ablation_env["QUARTZ_EVAL_STALL_TIMEOUT_S"] = str(
            float(args.eval_stall_timeout_s)
        )
    ablation_cmd = build_ablation_command(args, rust_binary, output_root)
    try:
        run(
            [sys.executable, "-m", "quartz.train", "--help"],
            repo_root,
            env=base_env,
            label="train_help",
            artifact_paths=artifact_paths,
        )
        run(
            [
                sys.executable,
                "-c",
                "from quartz.evaluation import _run_all; _run_all()",
            ],
            repo_root,
            env=base_env,
            label="evaluation_probe",
            artifact_paths=artifact_paths,
        )
        run(
            ablation_cmd,
            repo_root,
            env=ablation_env,
            label="ablation_smoke",
            artifact_paths=artifact_paths,
        )
        report_dir = output_root / args.game
        expected = [
            report_dir / "study_manifest.json",
            report_dir / "evaluation_matrix.json",
            report_dir / "ablation_report.json",
        ]
        missing = [str(path) for path in expected if not path.exists()]
        if missing:
            raise SystemExit(
                f"Smoke completed but expected artifacts are missing: {', '.join(missing)}"
            )
        # P3 (audit_codex_20260425.md W6): explicit post-run assertion that
        # at least one SGD row fired across all conditions. Without this
        # the smoke certifies imports/transport but not training.
        sgd_rows, scanned_logs = verify_training_fired(report_dir)
        print(
            f"P3 smoke contract: {sgd_rows} SGD rows across "
            f"{len(scanned_logs)} train_log.jsonl files."
        )
        success = True
        print(f"Smoke complete. Artifacts: {report_dir}")
        print(f"Smoke contract: {output_root / 'smoke_contract.json'}")
    except Exception as exc:
        summary_error = str(exc)
        raise
    finally:
        summary = build_smoke_summary(
            args=args,
            output_root=output_root,
            artifact_paths=artifact_paths,
            rust_binary=rust_binary,
            success=success,
            missing_outputs=missing,
            error=summary_error,
        )
        artifact_paths["summary_json"].write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
