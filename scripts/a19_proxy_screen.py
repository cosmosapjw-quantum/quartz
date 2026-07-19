#!/usr/bin/env python3
"""Train and evaluate the A19 deterministic graph-seed proxy screen."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quartz.experiment_manifest import file_sha256  # noqa: E402
from quartz.idea_foundry.a19_ablation import (  # noqa: E402
    A19PreparationError,
    load_json_strict,
    load_screen_plan,
)
from quartz.idea_foundry.a19_proxy import A19ProxyError, run_proxy_screen  # noqa: E402


DEFAULT_PLAN = REPO_ROOT / "configs" / "idea_foundry.a19.screen.v1.json"
DEFAULT_REPLAYS = REPO_ROOT / "configs" / "idea_foundry.a19.replays.v1.json"
DEFAULT_CONTROLLER = REPO_ROOT / "configs" / "idea_foundry.a19.controller.v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--screen-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--replay-manifest", type=Path, default=DEFAULT_REPLAYS)
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = load_screen_plan(args.screen_plan.resolve())
        replay_manifest = load_json_strict(args.replay_manifest.resolve())
        if not isinstance(replay_manifest, dict):
            raise A19ProxyError("A19 replay manifest must be an object")
        controller = load_json_strict(args.controller.resolve())
        if not isinstance(controller, dict) or controller.get("axis_id") != "A19":
            raise A19ProxyError("A19 frozen controller identity mismatch")
        if controller.get("immutable_during_ablation") is not True:
            raise A19ProxyError("A19 controller must be immutable")
        for source in replay_manifest.get("sources", []):
            replay_path = REPO_ROOT / source["replay_path"]
            checkpoint_path = REPO_ROOT / source["checkpoint_path"]
            if file_sha256(replay_path) != source["replay_sha256"]:
                raise A19ProxyError(f"replay hash drift: {replay_path}")
            if file_sha256(checkpoint_path) != source["checkpoint_sha256"]:
                raise A19ProxyError(f"checkpoint hash drift: {checkpoint_path}")
        summary = run_proxy_screen(
            plan=plan,
            screen_plan_path=args.screen_plan.resolve(),
            replay_manifest=replay_manifest,
            replay_manifest_path=args.replay_manifest.resolve(),
            controller_path=args.controller.resolve(),
            profile=args.profile,
            seed=args.seed,
            output_dir=args.output_dir.resolve(),
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (A19PreparationError, A19ProxyError, OSError, ValueError) as exc:
        print(f"A19 PROXY BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
