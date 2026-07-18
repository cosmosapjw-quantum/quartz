#!/usr/bin/env python3
"""symmetry_orbit_lab runner — audit the signature operators for the
game-agnostic FORBIDDEN constraint (see
``quartz/experiments/symmetry_orbit.py`` and
``docs/METACOGNITIVE_EXPERIMENTS.md``).

This is a diagnostic, not a screen with a kill-criterion. It reports, per
operator, whether its behavior obeys its transform law (invariant scalar
readouts, equivariant committed move) under action / dihedral-D4 / trace-bundle
/ move-order permutations and zero-mass clones, and confirms that deliberately
index-dependent negative controls are flagged. The overall verdict
``game_agnostic_constraint_upheld`` is True iff every real operator obeys its
law AND the harness catches the negative controls.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quartz.experiment_manifest import (  # noqa: E402
    atomic_json_dump,
    build_run_manifest,
    file_sha256,
    finalize_run_manifest,
    utc_now,
)
from quartz.experiments import symmetry_orbit as lab  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "symmetry_orbit_audit.v1.json"


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError("unsupported config format_version")
    if payload.get("experiment_id") != lab.EXPERIMENT_ID:
        raise ValueError("config experiment_id mismatch")
    return payload


def write_operator_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--n-actions", type=int, default=None)
    parser.add_argument("--board-side", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/metacognitive_root/symmetry_orbit_v1"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)

    seed = int(args.seed if args.seed is not None else config["default_seed"])
    n_trials = int(
        args.n_trials if args.n_trials is not None else config["default_n_trials"]
    )
    n_actions = int(
        args.n_actions if args.n_actions is not None else config["default_n_actions"]
    )
    board_side = int(
        args.board_side if args.board_side is not None else config["default_board_side"]
    )
    eps = float(config.get("default_eps", 1e-9))

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"output directory is not empty; pass --overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = {
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "seed": seed,
        "n_trials": n_trials,
        "n_actions": n_actions,
        "board_side": board_side,
        "eps": eps,
    }
    started_at = utc_now()
    manifest = build_run_manifest(
        experiment_id=lab.EXPERIMENT_ID,
        execution_mode=lab.EXECUTION_MODE,
        resolved_config=resolved_config,
        repo_root=REPO_ROOT,
        source_paths=[
            Path(__file__),
            REPO_ROOT / "quartz" / "experiments" / "symmetry_orbit.py",
            REPO_ROOT / "quartz" / "phase15_signatures.py",
            REPO_ROOT / "quartz" / "experiments" / "forked_voc.py",
            config_path,
        ],
        argv=sys.argv if argv is None else [str(Path(__file__)), *argv],
        started_at=started_at,
        assumptions=config["assumptions"],
        prohibited_inferences=config["prohibited_inferences"],
    )
    manifest_path = output_dir / "run_manifest.json"
    atomic_json_dump(manifest_path, manifest)

    audit = lab.audit(
        seed=seed,
        n_trials=n_trials,
        n_actions=n_actions,
        board_side=board_side,
        eps=eps,
    )

    operator_rows = (
        list(audit["operators"])
        + list(audit["clone_robustness"])
        + list(audit["negative_controls"])
    )
    operators_csv = output_dir / "operators.csv"
    summary_json = output_dir / "summary.json"
    write_operator_csv(operators_csv, operator_rows)
    atomic_json_dump(
        summary_json,
        {
            "format_version": 1,
            "experiment_id": lab.EXPERIMENT_ID,
            "execution_mode": lab.EXECUTION_MODE,
            "claim_status": "synthetic_screening_only",
            "config_sha256": resolved_config["config_sha256"],
            "audit": audit,
        },
    )
    manifest = finalize_run_manifest(
        manifest, output_dir=output_dir, artifact_paths=[operators_csv, summary_json]
    )
    atomic_json_dump(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "game_agnostic_constraint_upheld": audit[
                    "game_agnostic_constraint_upheld"
                ],
                "n_operator_violations": audit["n_operator_violations"],
                "negative_controls_all_caught": audit["negative_controls_all_caught"],
                "run_contract_hash": manifest["run_contract_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
