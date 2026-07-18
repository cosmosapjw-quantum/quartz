"""Research-grade gate for phase15 ablations (Stage 7 / C10).

Ports the CLAIM_LEDGER "Ablation Start Conditions" into a checkable gate for the
phase15 stack (the legacy `scripts/ablation_study.py::enforce_research_grade`
gate is not wired into the phase15 runner). A phase15 result may only be
promoted above `ABLATION-PENDING` when this gate passes.

Checks (each returns `(ok, detail)`):

1. ``seed_families``    — >= `min_seed_families` distinct training-seed families
   among the checkpoints (parsed from the `seed_<n>` path/id segment).
2. ``paired_coverage``  — every compared system covers the identical
   `(checkpoint, position, budget)` tuple set (paired protocol).
3. ``single_salt``      — exactly one `trace_code_salt` across all rows (neither
   missing salts nor pre/post trace-schema-bump mixing are accepted).
4. ``artifact_hashes``  — every checkpoint plus the positions/config artifacts
   exists and its recorded sha256 matches the file bytes.
5. ``rows_preserved``   — the row count equals
   `len(checkpoints) * len(positions) * len(budgets) * len(systems)` (failure /
   non-improvement rows are preserved, not dropped).

`check_research_grade` aggregates them; `enforce_research_grade` raises
`SystemExit` with the unmet list when strict enforcement is requested.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from quartz.experiment_manifest import file_sha256

_SEED_RE = re.compile(r"seed[_\-]?(\d+)")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def checkpoint_seed_family(checkpoint: str) -> str | None:
    """Extract the `seed_<n>` family label from a checkpoint id or path."""
    m = _SEED_RE.search(str(checkpoint))
    return f"seed_{m.group(1)}" if m else None


def count_seed_families(checkpoints: Iterable[str]) -> int:
    fams = {checkpoint_seed_family(c) for c in checkpoints}
    fams.discard(None)
    return len(fams)


def check_seed_families(checkpoints: Sequence[str], min_seed_families: int) -> tuple[bool, dict[str, Any]]:
    n = count_seed_families(checkpoints)
    return n >= int(min_seed_families), {
        "n_seed_families": n,
        "min_required": int(min_seed_families),
        "families": sorted(f for f in {checkpoint_seed_family(c) for c in checkpoints} if f),
    }


def _coverage_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row["checkpoint_id"]), str(row["position_id"]), int(row["budget"]))


def check_paired_coverage(rows: Sequence[dict[str, Any]], systems: Sequence[str]) -> tuple[bool, dict[str, Any]]:
    by_system: dict[str, set[tuple[str, str, int]]] = {s: set() for s in systems}
    for row in rows:
        s = str(row.get("system"))
        if s in by_system:
            by_system[s].add(_coverage_key(row))
    if not systems:
        return False, {"reason": "no systems"}
    reference = by_system[systems[0]]
    mismatched = [s for s in systems if by_system[s] != reference]
    return (len(mismatched) == 0 and len(reference) > 0), {
        "reference_system": systems[0],
        "reference_n_tuples": len(reference),
        "mismatched_systems": mismatched,
        "per_system_counts": {s: len(v) for s, v in by_system.items()},
    }


def check_single_salt(rows: Sequence[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    salts = {
        str(row.get("trace_code_salt"))
        for row in rows
        if isinstance(row.get("trace_code_salt"), str)
        and str(row.get("trace_code_salt")).strip()
    }
    missing = sum(
        1
        for row in rows
        if not isinstance(row.get("trace_code_salt"), str)
        or not str(row.get("trace_code_salt")).strip()
    )
    ok = bool(rows) and len(salts) == 1 and missing == 0
    return ok, {
        "n_distinct_salts": len(salts),
        "salts": sorted(salts),
        "rows_missing_salt": missing,
    }


def _verify_hash_entry(
    entry: Any,
    *,
    label: str,
    artifact_root: Path,
    fallback_path: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Verify one artifact hash record against bytes on disk.

    The preferred schema is ``{"path": "relative/or/absolute", "sha256":
    "..."}``.  For checkpoint records only, a plain digest remains readable
    when the checkpoint map key itself resolves to the artifact path.  A bare
    digest with no resolvable path is deliberately not treated as proof.
    """

    if isinstance(entry, Mapping):
        raw_path = entry.get("path")
        expected = entry.get("sha256")
    else:
        raw_path = fallback_path
        expected = entry

    detail: dict[str, Any] = {"label": label}
    if not isinstance(expected, str) or _SHA256_RE.fullmatch(expected) is None:
        return False, {**detail, "reason": "invalid_sha256"}
    if not isinstance(raw_path, str) or not raw_path:
        return False, {**detail, "reason": "missing_path", "expected_sha256": expected.lower()}

    path = Path(raw_path)
    if not path.is_absolute():
        path = artifact_root / path
    path = path.resolve()
    detail.update({"path": str(path), "expected_sha256": expected.lower()})
    if not path.is_file():
        return False, {**detail, "reason": "missing_file"}

    actual = file_sha256(path)
    detail["actual_sha256"] = actual
    if actual != expected.lower():
        return False, {**detail, "reason": "sha256_mismatch"}
    detail["verified"] = True
    return True, detail


def check_artifact_hashes(
    manifest: dict[str, Any],
    checkpoints: Sequence[str],
    *,
    artifact_root: str | Path | None = None,
) -> tuple[bool, dict[str, Any]]:
    hashes = manifest.get("stage7_artifact_hashes") or manifest.get("artifact_hashes") or {}
    ckpt_hashes = hashes.get("checkpoints", {}) if isinstance(hashes, dict) else {}
    root = Path(artifact_root or ".").resolve()
    missing = [str(c) for c in checkpoints if str(c) not in ckpt_hashes]
    verified: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []

    for checkpoint in checkpoints:
        key = str(checkpoint)
        if key not in ckpt_hashes:
            continue
        ok_entry, detail = _verify_hash_entry(
            ckpt_hashes[key], label=f"checkpoint:{key}", artifact_root=root, fallback_path=key
        )
        (verified if ok_entry else invalid).append(detail)

    positions_key = next(
        (key for key in ("positions", "suite", "positions_file") if isinstance(hashes, dict) and key in hashes),
        None,
    )
    config_key = next(
        (key for key in ("systems_config", "config") if isinstance(hashes, dict) and key in hashes),
        None,
    )
    for kind, key in (("positions", positions_key), ("config", config_key)):
        if key is None:
            invalid.append({"label": kind, "reason": "missing_hash_entry"})
            continue
        ok_entry, detail = _verify_hash_entry(hashes[key], label=kind, artifact_root=root)
        (verified if ok_entry else invalid).append(detail)

    ok = not missing and not invalid and len(verified) == len(checkpoints) + 2
    return ok, {
        "missing_checkpoint_hashes": missing,
        "artifact_root": str(root),
        "verified_artifacts": verified,
        "invalid_artifacts": invalid,
    }


def check_rows_preserved(
    rows: Sequence[dict[str, Any]],
    *,
    n_checkpoints: int,
    n_positions: int,
    n_budgets: int,
    systems: Sequence[str],
) -> tuple[bool, dict[str, Any]]:
    expected = int(n_checkpoints) * int(n_positions) * int(n_budgets) * len(systems)
    actual = sum(1 for row in rows if str(row.get("system")) in set(systems))
    return actual == expected, {"expected_rows": expected, "actual_rows": actual}


def check_research_grade(
    *,
    checkpoints: Sequence[str],
    rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
    systems: Sequence[str],
    n_positions: int,
    n_budgets: int,
    analyzer_report: dict[str, Any] | None = None,
    min_seed_families: int = 3,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    ok_seed, d_seed = check_seed_families(checkpoints, min_seed_families)
    checks["seed_families"] = {"ok": ok_seed, **d_seed}

    ok_cov, d_cov = check_paired_coverage(rows, systems)
    checks["paired_coverage"] = {"ok": ok_cov, **d_cov}

    ok_salt, d_salt = check_single_salt(rows)
    checks["single_salt"] = {"ok": ok_salt, **d_salt}

    ok_hash, d_hash = check_artifact_hashes(manifest, checkpoints, artifact_root=artifact_root)
    checks["artifact_hashes"] = {"ok": ok_hash, **d_hash}

    ok_rows, d_rows = check_rows_preserved(
        rows, n_checkpoints=len(checkpoints), n_positions=n_positions, n_budgets=n_budgets, systems=systems
    )
    checks["rows_preserved"] = {"ok": ok_rows, **d_rows}

    if analyzer_report is not None:
        has_a2b = bool(analyzer_report.get("interpretation_flags") is not None)
        checks["a2b_report"] = {"ok": has_a2b, "has_interpretation_flags": has_a2b}

    unmet = [name for name, c in checks.items() if not c["ok"]]
    return {
        "research_grade_ready": len(unmet) == 0,
        "unmet": unmet,
        "checks": checks,
    }


def enforce_research_grade(report: dict[str, Any]) -> None:
    """Raise SystemExit with the unmet-criteria list when the gate fails."""
    if not report.get("research_grade_ready", False):
        raise SystemExit(
            "research-grade gate FAILED; unmet criteria: " + ", ".join(report.get("unmet", []))
        )
