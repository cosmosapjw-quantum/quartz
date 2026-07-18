#!/usr/bin/env python3
"""Derive exact-state-disjoint A18 evaluation replay shards.

Each evaluation shard keeps the source replay order and removes every position
whose exact float32 state identity occurs in the paired training replay.  The
source replay remains separately hashed, so derivation never hides the observed
cross-seed overlap.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quartz.idea_foundry.a18_ablation import (  # noqa: E402
    A18ContractError,
    derive_state_disjoint_evaluation_replay,
    sha256_file,
    verify_state_disjoint_evaluation_replay_receipt,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _parse_mapping(raw: str) -> list[tuple[int, int]]:
    pairs = []
    for token in raw.split(","):
        try:
            training_seed, source_seed = token.split(":", 1)
            pairs.append((int(training_seed), int(source_seed)))
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                "mapping must use training_seed:source_seed pairs"
            ) from exc
    training_seeds = [pair[0] for pair in pairs]
    if not pairs or len(training_seeds) != len(set(training_seeds)):
        raise argparse.ArgumentTypeError("mapping training seeds must be unique")
    if any(training_seed == source_seed for training_seed, source_seed in pairs):
        raise argparse.ArgumentTypeError("training and source seeds must differ")
    return pairs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-root",
        default="results/phase15_ablation/gomoku7/bootstrap",
    )
    parser.add_argument(
        "--output-dir",
        default="results/idea_foundry_inputs/a18_study_v1",
    )
    parser.add_argument("--mapping", default="41:42,42:43,43:41")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        mapping = _parse_mapping(args.mapping)
        bootstrap_root = Path(args.bootstrap_root)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        receipts = []
        generated = 0
        reused = 0
        for training_seed, source_seed in mapping:
            training_replay = bootstrap_root / f"seed_{training_seed}" / "replay.npz"
            source_replay = bootstrap_root / f"seed_{source_seed}" / "replay.npz"
            output_replay = (
                output_dir
                / f"train_seed_{training_seed}.eval_from_seed_{source_seed}.exact_state_disjoint.npz"
            )
            receipt_path = output_replay.with_suffix(".receipt.v1.json")
            if not args.force and output_replay.exists() != receipt_path.exists():
                raise A18ContractError(
                    "derived shard and receipt must either both exist or both be absent: "
                    f"{output_replay}"
                )
            if not args.force and output_replay.exists():
                try:
                    existing_receipt = json.loads(
                        receipt_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    raise A18ContractError(
                        f"invalid existing holdout derivation receipt: {receipt_path}"
                    ) from exc
                receipt = verify_state_disjoint_evaluation_replay_receipt(
                    training_replay,
                    source_replay,
                    output_replay,
                    existing_receipt,
                    training_seed=training_seed,
                    evaluation_source_seed=source_seed,
                )
                reused += 1
            else:
                receipt = derive_state_disjoint_evaluation_replay(
                    training_replay,
                    source_replay,
                    output_replay,
                    training_seed=training_seed,
                    evaluation_source_seed=source_seed,
                )
                _atomic_json(receipt_path, receipt)
                generated += 1
            receipts.append(
                {
                    **receipt,
                    "receipt": str(receipt_path.resolve()),
                    "receipt_sha256": sha256_file(receipt_path),
                }
            )
        bundle = {
            "schema_version": 1,
            "axis_id": "A18",
            "artifact_kind": "evaluation_replay_derivation_bundle",
            "scientific_status": "DERIVED_INPUT_NOT_EFFICACY",
            "mapping": [
                {"training_seed": training, "evaluation_source_seed": source}
                for training, source in mapping
            ],
            "receipts": receipts,
            "source_hashes": [
                {
                    "path": "scripts/a18_prepare_holdouts.py",
                    "sha256": sha256_file(Path(__file__)),
                },
                {
                    "path": "quartz/idea_foundry/a18_ablation.py",
                    "sha256": sha256_file(
                        REPO_ROOT / "quartz" / "idea_foundry" / "a18_ablation.py"
                    ),
                },
            ],
        }
        bundle_path = output_dir / "holdout_derivation_bundle.v1.json"
        _atomic_json(bundle_path, bundle)
        print(
            json.dumps(
                {
                    "bundle": str(bundle_path.resolve()),
                    "receipts": len(receipts),
                    "generated": generated,
                    "reused": reused,
                }
            )
        )
        return 0
    except (A18ContractError, argparse.ArgumentTypeError) as exc:
        print(f"A18 HOLDOUT BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
