#!/usr/bin/env python3
"""Plan, preflight, and analyze the metacontroller training/runtime 2x2 study."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quartz.experiment_manifest import atomic_json_dump, file_sha256  # noqa: E402
from quartz.idea_foundry.metacontroller_factorial import (  # noqa: E402
    FactorialHarnessError,
    build_plan,
    raw_row_template,
    run_preflight,
    training_manifest_template,
    write_analysis,
)
from quartz.idea_foundry.metacontroller_factorial_execution import (  # noqa: E402
    RUNTIME_BINARY_RELATIVE,
    build_feature_binary,
    campaign_status,
    materialize_execution_config,
    prepare_preflight_artifact,
    probe_foundry_binary,
    run_arena,
    run_training_treatments,
    resume_factorial_campaign,
    smoke_runtime_adapter,
    start_factorial_campaign,
)


DEFAULT_CONFIG = REPO_ROOT / "configs" / "metacontroller_factorial.v1.json"


def _repo_path(raw_path: str | Path, *, label: str, must_exist: bool = False) -> Path:
    path = Path(raw_path)
    resolved = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise FactorialHarnessError(
            f"{label} must remain inside the repository"
        ) from exc
    if must_exist and (not resolved.is_file() or resolved.is_symlink()):
        raise FactorialHarnessError(f"{label} is not a regular file: {resolved}")
    return resolved


def _results_path(raw_path: str | Path, *, label: str) -> Path:
    path = _repo_path(raw_path, label=label)
    try:
        path.relative_to((REPO_ROOT / "results").resolve())
    except ValueError as exc:
        raise FactorialHarnessError(f"{label} must remain under results/") from exc
    return path


def _emit(payload: dict, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"status={payload.get('status')}")
    if payload.get("profile") is not None:
        print(f"profile={payload['profile']}")
    counts = payload.get("counts") or {}
    if counts:
        print("counts=" + json.dumps(counts, sort_keys=True))
    blockers = payload.get("blockers") or []
    for blocker in blockers:
        print(f"BLOCKED {blocker['code']}: {blocker['message']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QUARTZ metacontroller training/runtime 2x2 factorial harness"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--config", default=str(DEFAULT_CONFIG))
        subparser.add_argument("--profile", default="smoke")
        subparser.add_argument("--json", action="store_true")

    plan_parser = subparsers.add_parser("plan", help="emit the deterministic design")
    add_common(plan_parser)
    plan_parser.add_argument("--output", default=None)

    preflight_parser = subparsers.add_parser(
        "preflight", help="validate live adapters, inputs, and treatment receipts"
    )
    add_common(preflight_parser)
    preflight_parser.add_argument("--output", default=None)

    schema_parser = subparsers.add_parser(
        "raw-schema", help="print the exact raw game-row contract"
    )
    schema_parser.add_argument("--json", action="store_true")

    training_schema_parser = subparsers.add_parser(
        "training-schema", help="print the exact training-treatment receipt contract"
    )
    training_schema_parser.add_argument("--json", action="store_true")

    analyze_parser = subparsers.add_parser(
        "analyze", help="validate paired raw rows and compute seed-level contrasts"
    )
    add_common(analyze_parser)
    analyze_parser.add_argument("--preflight", required=True)
    analyze_parser.add_argument("--rows", required=True)
    analyze_parser.add_argument("--output-dir", required=True)

    build_parser = subparsers.add_parser(
        "build-runtime", help="build and probe the isolated idea-foundry Rust binary"
    )
    build_parser.add_argument("--json", action="store_true")

    probe_parser = subparsers.add_parser(
        "probe-runtime", help="verify a Rust binary's live Foundry capabilities"
    )
    probe_parser.add_argument("--rust-binary", default=RUNTIME_BINARY_RELATIVE)
    probe_parser.add_argument("--json", action="store_true")

    smoke_parser = subparsers.add_parser(
        "runtime-smoke", help="exercise shadow and active A01 on one identical root"
    )
    smoke_parser.add_argument("--checkpoint", required=True)
    smoke_parser.add_argument("--rust-binary", default=RUNTIME_BINARY_RELATIVE)
    smoke_parser.add_argument("--device", default="cpu")
    smoke_parser.add_argument("--output", default=None)
    smoke_parser.add_argument("--json", action="store_true")

    prepare_parser = subparsers.add_parser(
        "prepare-run", help="freeze a run-local config and anchor checkpoint hash"
    )
    add_common(prepare_parser)
    prepare_parser.add_argument("--run-id", required=True)
    prepare_parser.add_argument("--anchor-checkpoint", required=True)

    train_parser = subparsers.add_parser(
        "train", help="execute paired OFF/ON training treatments sequentially"
    )
    add_common(train_parser)
    train_parser.add_argument("--initial-checkpoint-template", required=True)
    train_parser.add_argument("--rust-binary", default=RUNTIME_BINARY_RELATIVE)
    train_parser.add_argument("--iterations", type=int, required=True)
    train_parser.add_argument("--games-per-iteration", type=int, required=True)
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--timeout-hours", type=float, default=24.0)
    train_parser.add_argument("--state", default=None)

    arena_parser = subparsers.add_parser(
        "arena", help="execute the planned fixed-opening game matrix sequentially"
    )
    add_common(arena_parser)
    arena_parser.add_argument("--preflight", required=True)
    arena_parser.add_argument("--output", required=True)
    arena_parser.add_argument("--rust-binary", default=RUNTIME_BINARY_RELATIVE)
    arena_parser.add_argument("--device", default="cuda")

    live_preflight_parser = subparsers.add_parser(
        "live-preflight", help="validate receipts plus the feature-binary capability"
    )
    add_common(live_preflight_parser)
    live_preflight_parser.add_argument("--output", required=True)
    live_preflight_parser.add_argument("--rust-binary", default=RUNTIME_BINARY_RELATIVE)

    run_parser = subparsers.add_parser(
        "run",
        help="execute the hash-bound training, preflight, arena, and analysis campaign",
    )
    add_common(run_parser)
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--anchor-checkpoint", required=True)
    run_parser.add_argument("--initial-checkpoint-template", required=True)
    run_parser.add_argument("--rust-binary", default=RUNTIME_BINARY_RELATIVE)
    run_parser.add_argument("--iterations", type=int, required=True)
    run_parser.add_argument("--games-per-iteration", type=int, required=True)
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument("--timeout-hours", type=float, default=24.0)

    resume_parser = subparsers.add_parser(
        "resume", help="resume a campaign from its frozen invocation contract"
    )
    resume_parser.add_argument("--run-id", required=True)
    resume_parser.add_argument("--json", action="store_true")

    status_parser = subparsers.add_parser(
        "status", help="inspect campaign state and current artifact hashes"
    )
    status_parser.add_argument("--run-id", required=True)
    status_parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "build-runtime":
            payload = build_feature_binary(repo_root=REPO_ROOT)
            payload["status"] = "READY"
            _emit(payload, as_json=args.json)
            return 0
        if args.command == "probe-runtime":
            payload = probe_foundry_binary(args.rust_binary, repo_root=REPO_ROOT)
            payload["status"] = "READY"
            _emit(payload, as_json=args.json)
            return 0
        if args.command == "runtime-smoke":
            payload = smoke_runtime_adapter(
                checkpoint=args.checkpoint,
                rust_binary=args.rust_binary,
                device=args.device,
                repo_root=REPO_ROOT,
            )
            if args.output:
                output = _results_path(args.output, label="runtime smoke output")
                atomic_json_dump(output, payload)
            _emit(payload, as_json=args.json)
            return 0
        if args.command == "raw-schema":
            payload = raw_row_template()
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == "training-schema":
            payload = training_manifest_template()
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == "status":
            payload = campaign_status(args.run_id, repo_root=REPO_ROOT)
            _emit(payload, as_json=args.json)
            return 0
        if args.command == "resume":
            payload = resume_factorial_campaign(args.run_id, repo_root=REPO_ROOT)
            _emit(payload, as_json=args.json)
            return 130 if payload.get("status") == "interrupted" else 0

        config_path = _repo_path(args.config, label="config", must_exist=True)
        if args.command == "run":
            payload = start_factorial_campaign(
                base_config_path=config_path,
                profile_name=args.profile,
                run_id=args.run_id,
                anchor_checkpoint=args.anchor_checkpoint,
                initial_checkpoint_template=args.initial_checkpoint_template,
                rust_binary=args.rust_binary,
                iterations=args.iterations,
                games_per_iteration=args.games_per_iteration,
                device=args.device,
                timeout_s=args.timeout_hours * 3600.0,
                repo_root=REPO_ROOT,
            )
            _emit(payload, as_json=args.json)
            return 130 if payload.get("status") == "interrupted" else 0
        if args.command == "prepare-run":
            resolved_path, resolved = materialize_execution_config(
                base_config_path=config_path,
                profile_name=args.profile,
                run_id=args.run_id,
                anchor_checkpoint=args.anchor_checkpoint,
                repo_root=REPO_ROOT,
            )
            payload = {
                "status": "PREPARED_NOT_EXECUTED",
                "profile": args.profile,
                "config": str(resolved_path.relative_to(REPO_ROOT)),
                "config_sha256": file_sha256(resolved_path),
                "experiment_id": resolved["experiment_id"],
            }
            _emit(payload, as_json=args.json)
            return 0
        if args.command == "plan":
            payload = build_plan(config_path, args.profile, repo_root=REPO_ROOT)
            if args.output:
                output = _results_path(args.output, label="plan output")
                atomic_json_dump(output, payload)
            _emit(payload, as_json=args.json)
            return 0

        if args.command == "preflight":
            payload = run_preflight(config_path, args.profile, repo_root=REPO_ROOT)
            if args.output:
                output = _results_path(args.output, label="preflight output")
                atomic_json_dump(output, payload)
            _emit(payload, as_json=args.json)
            return 0 if payload["ready"] else 2

        if args.command == "live-preflight":
            output = _results_path(args.output, label="preflight output")
            payload = prepare_preflight_artifact(
                config_path=config_path,
                profile_name=args.profile,
                output_path=output,
                rust_binary=args.rust_binary,
                repo_root=REPO_ROOT,
            )
            _emit(payload, as_json=args.json)
            return 0 if payload["ready"] else 2

        if args.command == "train":
            state_path = (
                None
                if args.state is None
                else _results_path(args.state, label="campaign state")
            )
            payload = run_training_treatments(
                config_path=config_path,
                profile_name=args.profile,
                initial_checkpoint_template=args.initial_checkpoint_template,
                rust_binary=args.rust_binary,
                iterations=args.iterations,
                games_per_iteration=args.games_per_iteration,
                device=args.device,
                timeout_s=args.timeout_hours * 3600.0,
                repo_root=REPO_ROOT,
                state_path=state_path,
            )
            _emit(payload, as_json=args.json)
            return 0

        if args.command == "arena":
            preflight_path = _repo_path(
                args.preflight, label="preflight", must_exist=True
            )
            output = _results_path(args.output, label="raw row output")
            payload = run_arena(
                config_path=config_path,
                profile_name=args.profile,
                preflight_path=preflight_path,
                output_path=output,
                rust_binary=args.rust_binary,
                device=args.device,
                repo_root=REPO_ROOT,
            )
            _emit(payload, as_json=args.json)
            return 0

        if args.command == "analyze":
            preflight_path = _repo_path(
                args.preflight, label="preflight", must_exist=True
            )
            rows_path = _repo_path(args.rows, label="raw rows", must_exist=True)
            output_dir = _results_path(args.output_dir, label="analysis output")
            payload = write_analysis(
                config_path=config_path,
                profile_name=args.profile,
                preflight_path=preflight_path,
                raw_rows_path=rows_path,
                output_dir=output_dir,
                repo_root=REPO_ROOT,
            )
            _emit(payload, as_json=args.json)
            return 0
        raise FactorialHarnessError(f"unknown command: {args.command}")
    except FactorialHarnessError as exc:
        print(f"METACONTROLLER FACTORIAL ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":

    def _interrupt_on_termination(_signum, _frame) -> None:
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, _interrupt_on_termination)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("METACONTROLLER FACTORIAL INTERRUPTED", file=sys.stderr)
        raise SystemExit(130) from None
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
