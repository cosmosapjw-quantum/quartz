#!/usr/bin/env python3
"""Run one dependency-light QUARTZ idea-foundry contract gate.

The command produces the same minimal artifact trio for every A01--A26 lane.
It validates deterministic contracts and safety invariants only; it never
promotes efficacy or play-strength status.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quartz.experiment_manifest import (  # noqa: E402
    atomic_json_dump,
    file_sha256,
    git_provenance,
)
from quartz.idea_foundry.gates import AXIS_TYPE_BY_ID, run_axis_contract_gate  # noqa: E402


SCHEMA_VERSION = 1
CLAIM_SCOPE = "synthetic_contract_gate_only"
PROHIBITED_INFERENCES = (
    "No play-strength, Elo, CPU/GPU efficiency, or universal superiority claim.",
    "No synthetic result may activate production search or promote an axis automatically.",
    "Analysis-only axes may not emit an online search-control action.",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _source_hashes(axis_id: str) -> list[dict[str, str]]:
    axis = AXIS_TYPE_BY_ID[axis_id]()
    module_path = Path(sys.modules[axis.__class__.__module__].__file__).resolve()
    paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "quartz" / "experiment_manifest.py",
        REPO_ROOT / "quartz" / "idea_foundry" / "__init__.py",
        REPO_ROOT / "quartz" / "idea_foundry" / "contracts.py",
        REPO_ROOT / "quartz" / "idea_foundry" / "serialization.py",
        REPO_ROOT / "quartz" / "idea_foundry" / "gates.py",
        REPO_ROOT / "configs" / "idea_foundry.axes.v1.json",
        module_path,
    )
    unique = sorted(set(paths))
    return [
        {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": file_sha256(path),
        }
        for path in unique
    ]


def _manifest_base(axis_id: str, role: str, seed: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "quartz_idea_foundry_first_contract_gate",
        "axis_id": axis_id,
        "role": role,
        "claim_scope": CLAIM_SCOPE,
        "execution_mode": "synthetic_contract_gate",
        "gate_evidence_status": "contract_only",
        "evidence_status_origin": "axis_registry_preexisting",
        "status": "running",
        "started_at": utc_now(),
        "completed_at": None,
        "source_hashes": _source_hashes(axis_id),
        "input_hashes": {},
        "seed_contract": {
            "seed": seed,
            "fixture_generator": "quartz.idea_foundry.gates.contract_observation",
            "deterministic_replay_required": True,
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "argv": list(sys.argv),
        },
        "git": git_provenance(REPO_ROOT),
        "auto_promoted": False,
        "prohibited_inferences": list(PROHIBITED_INFERENCES),
        "artifacts": [],
    }


def run(axis_id: str, role: str, output_dir: Path, seed: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    rows_path = output_dir / "rows.jsonl"
    summary_path = output_dir / "summary.json"
    manifest = _manifest_base(axis_id, role, seed)
    atomic_json_dump(manifest_path, manifest)
    try:
        result = run_axis_contract_gate(axis_id, role=role, seed=seed)
        rows = [
            {
                "schema_version": SCHEMA_VERSION,
                "axis_id": axis_id,
                "role": role,
                "fixture_id": row.get("fixture_id", result["fixture_id"]),
                "metric": row["metric"],
                "value": row["value"],
            }
            for row in result["rows"]
        ]
        rows.extend(
            [
                {
                    "schema_version": SCHEMA_VERSION,
                    "axis_id": axis_id,
                    "role": role,
                    "fixture_id": result["fixture_id"],
                    "metric": "proposal_count",
                    "value": result["proposal_count"],
                },
                {
                    "schema_version": SCHEMA_VERSION,
                    "axis_id": axis_id,
                    "role": role,
                    "fixture_id": result["fixture_id"],
                    "metric": "proposal_hash",
                    "value": result["proposal_hash"],
                },
            ]
        )
        _atomic_jsonl(rows_path, rows)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "axis_id": axis_id,
            "axis_symbol": result["axis_symbol"],
            "role": role,
            "execution_status": result["execution_status"],
            "evidence_status": result["evidence_status"],
            "axis_registry_status": result["axis_registry_status"],
            "claim_scope": CLAIM_SCOPE,
            "execution_mode": "synthetic_contract_gate",
            "gate_evidence_status": "contract_only",
            "evidence_status_origin": "axis_registry_preexisting",
            "fixture_id": result["fixture_id"],
            "fixture_ids": result["fixture_ids"],
            "fixture_hash": result["fixture_hash"],
            "fixture_bank_hash": result["fixture_bank_hash"],
            "proposal_hash": result["proposal_hash"],
            "proposal_count": result["proposal_count"],
            "promotion": result["promotion"],
            "outcome_detail": result["outcome_detail"],
            "prohibited_inferences": list(PROHIBITED_INFERENCES),
        }
        atomic_json_dump(summary_path, summary)
        manifest["input_hashes"] = {
            "contract_fixture": result["fixture_hash"],
            "contract_fixture_bank": result["fixture_bank_hash"],
        }
        manifest["status"] = result["execution_status"]
        manifest["completed_at"] = utc_now()
        manifest["artifacts"] = [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in (rows_path, summary_path)
        ]
        atomic_json_dump(manifest_path, manifest)
        return 0
    except Exception as exc:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "axis_id": axis_id,
            "role": role,
            "execution_status": "failed",
            "evidence_status": "unchanged",
            "claim_scope": CLAIM_SCOPE,
            "promotion": {"auto": False, "eligible": False, "reason": "contract_failure"},
            "error": f"{type(exc).__name__}: {exc}",
        }
        atomic_json_dump(summary_path, summary)
        manifest["status"] = "failed"
        manifest["completed_at"] = utc_now()
        manifest["error"] = summary["error"]
        manifest["artifacts"] = [
            {
                "path": summary_path.name,
                "size_bytes": summary_path.stat().st_size,
                "sha256": file_sha256(summary_path),
            }
        ]
        atomic_json_dump(manifest_path, manifest)
        print(summary["error"], file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", required=True, choices=sorted(AXIS_TYPE_BY_ID))
    parser.add_argument("--role", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args(argv)
    return run(args.axis, args.role, args.output_dir, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
