#!/usr/bin/env python3
"""Run or validate the A18 matched deterministic-evaluator ablation.

The default ``run`` command trains paired smoke/study candidates from declared
real bootstrap checkpoints and then writes diagnostic-only artifacts.  The
``analyze`` command never manufactures missing models: every candidate must be
a completed, metadata-bound A18 checkpoint or the command fails closed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quartz.idea_foundry.a18_ablation import (  # noqa: E402
    A18ContractError,
    analyze_candidates,
    inspect_inputs,
    load_spec,
    run_smoke_or_study,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A18 parameter/FLOP-matched evaluator ablation substrate"
    )
    parser.add_argument("--spec", required=True, help="Version-1 A18 study JSON")
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output",
        required=True,
        help="Campaign output directory",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device; use cuda for the registered GPU lane",
    )
    parser.add_argument(
        "command",
        choices=("inspect", "run", "analyze"),
        help="inspect inputs, train+analyze, or analyze existing candidates",
    )
    return parser.parse_args(argv)


def _device_preflight(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        if not torch.cuda.is_available() or torch.version.cuda is None:
            raise A18ContractError(
                "CUDA device requested but torch CUDA runtime is unavailable"
            )
        try:
            torch.empty(1, device=device)
        except Exception as exc:
            raise A18ContractError(f"CUDA device allocation failed: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        spec = load_spec(args.spec)
        if args.command == "inspect":
            rows = inspect_inputs(spec)
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "axis_id": "A18",
                        "scientific_status": "INPUTS_INSPECTED_NOT_EFFICACY",
                        "ready_for_paired_training": True,
                        "inputs": rows,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        _device_preflight(args.device)
        if args.command == "run":
            payload = run_smoke_or_study(spec, args.output, device=args.device)
        else:
            # analyze_candidates performs strict existence/metadata/hash checks;
            # no random or untrained fallback is permitted here.
            payload = analyze_candidates(spec, args.output, device=args.device)
        print(
            json.dumps(
                {
                    "schema_version": payload["schema_version"],
                    "axis_id": payload["axis_id"],
                    "scientific_status": payload["scientific_status"],
                    "evidence_tier": payload["evidence_tier"],
                    "manifest": str((Path(args.output) / "manifest.v1.json").resolve()),
                    "automatic_promotion": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except A18ContractError as exc:
        print(f"A18 BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
