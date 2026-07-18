"""Tests for quartz.phase15_research_grade (Stage 7 / C10)."""

from pathlib import Path

import pytest

from quartz.experiment_manifest import file_sha256
from quartz.phase15_research_grade import (
    check_paired_coverage,
    check_research_grade,
    check_seed_families,
    check_single_salt,
    checkpoint_seed_family,
    count_seed_families,
    enforce_research_grade,
)


def test_checkpoint_seed_family_parsing():
    assert checkpoint_seed_family("results/.../seed_101/gen_8.pt") == "seed_101"
    assert checkpoint_seed_family("C01_seed_42_gen_2") == "seed_42"
    assert checkpoint_seed_family("no-seed-here") is None
    assert count_seed_families(["seed_101/a", "seed_102/b", "seed_101/c"]) == 2


def test_check_seed_families_pass_and_fail():
    ok, d = check_seed_families(["seed_101/g8", "seed_102/g8", "seed_103/g8"], 3)
    assert ok is True and d["n_seed_families"] == 3
    ok2, _ = check_seed_families(["seed_101/g8", "seed_101/g5"], 3)
    assert ok2 is False


def _rows(systems, keys, salt="S", extra_pos=None):
    rows = []
    for s in systems:
        for (c, p, b) in keys:
            rows.append({"system": s, "checkpoint_id": c, "position_id": p, "budget": b, "trace_code_salt": salt})
    return rows


def _artifact_manifest(root: Path, checkpoints: list[str]) -> dict:
    checkpoint_entries = {}
    for checkpoint in checkpoints:
        path = root / checkpoint
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"checkpoint:{checkpoint}".encode())
        checkpoint_entries[checkpoint] = {"path": checkpoint, "sha256": file_sha256(path)}
    positions = root / "positions.json"
    positions.write_text("[]\n", encoding="utf-8")
    config = root / "systems.json"
    config.write_text("{}\n", encoding="utf-8")
    return {
        "stage7_artifact_hashes": {
            "checkpoints": checkpoint_entries,
            "positions": {"path": positions.name, "sha256": file_sha256(positions)},
            "systems_config": {"path": config.name, "sha256": file_sha256(config)},
        }
    }


def test_paired_coverage_equal_and_mismatch():
    keys = [("C1", "P1", 8), ("C1", "P1", 16)]
    ok, d = check_paired_coverage(_rows(["A4", "B13"], keys), ["A4", "B13"])
    assert ok is True and d["mismatched_systems"] == []
    # drop one tuple from B13 => mismatch
    rows = _rows(["A4", "B13"], keys)
    rows = [r for r in rows if not (r["system"] == "B13" and r["budget"] == 16)]
    ok2, d2 = check_paired_coverage(rows, ["A4", "B13"])
    assert ok2 is False and "B13" in d2["mismatched_systems"]


def test_single_salt_requires_exactly_one_present_value():
    assert check_single_salt([])[0] is False
    assert check_single_salt([{"trace_code_salt": None}])[0] is False
    assert check_single_salt([{"trace_code_salt": ""}])[0] is False
    assert check_single_salt([{"trace_code_salt": "S"}, {"trace_code_salt": "S"}])[0] is True
    assert check_single_salt([{"trace_code_salt": "S"}, {"trace_code_salt": "T"}])[0] is False


def test_research_grade_compliant_and_failures(tmp_path):
    keys = [("seed_101/g8", "P1", 8), ("seed_102/g8", "P1", 8), ("seed_103/g8", "P1", 8)]
    checkpoints = ["seed_101/g8", "seed_102/g8", "seed_103/g8"]
    systems = ["A4", "B13"]
    rows = _rows(systems, keys)
    manifest = _artifact_manifest(tmp_path, checkpoints)
    report = check_research_grade(
        checkpoints=checkpoints, rows=rows, manifest=manifest, systems=systems,
        n_positions=1, n_budgets=1, analyzer_report={"interpretation_flags": {}}, min_seed_families=3,
        artifact_root=tmp_path,
    )
    assert report["research_grade_ready"] is True
    assert report["unmet"] == []
    enforce_research_grade(report)  # must not raise

    # remove a checkpoint hash => artifact_hashes fails
    bad_manifest = {"stage7_artifact_hashes": {"checkpoints": {}, "positions": "x", "systems_config": "y"}}
    bad = check_research_grade(
        checkpoints=checkpoints, rows=rows, manifest=bad_manifest, systems=systems,
        n_positions=1, n_budgets=1, min_seed_families=3, artifact_root=tmp_path,
    )
    assert bad["research_grade_ready"] is False
    assert "artifact_hashes" in bad["unmet"]
    with pytest.raises(SystemExit):
        enforce_research_grade(bad)

    # only 2 seed families => seed_families fails
    few = check_research_grade(
        checkpoints=["seed_101/g8", "seed_101/g5"], rows=rows, manifest=manifest, systems=systems,
        n_positions=1, n_budgets=1, min_seed_families=3, artifact_root=tmp_path,
    )
    assert "seed_families" in few["unmet"]


def test_rows_preserved_detects_dropped_rows(tmp_path):
    keys = [("seed_101/g8", "P1", 8), ("seed_102/g8", "P1", 8)]
    systems = ["A4", "B13"]
    rows = _rows(systems, keys)
    # 2 systems * 1 checkpoint-count?? here checkpoints distinct per key; use n_checkpoints=2
    checkpoints = ["seed_101/g8", "seed_102/g8"]
    manifest = _artifact_manifest(tmp_path, checkpoints)
    report = check_research_grade(
        checkpoints=checkpoints, rows=rows, manifest=manifest,
        systems=systems, n_positions=1, n_budgets=1, min_seed_families=2, artifact_root=tmp_path,
    )
    # expected = 2 ckpt * 1 pos * 1 budget * 2 systems = 4; actual = 4
    assert report["checks"]["rows_preserved"]["ok"] is True
    # drop one row => fails
    report2 = check_research_grade(
        checkpoints=checkpoints, rows=rows[:-1], manifest=manifest,
        systems=systems, n_positions=1, n_budgets=1, min_seed_families=2, artifact_root=tmp_path,
    )
    assert report2["checks"]["rows_preserved"]["ok"] is False


def test_research_grade_rejects_hash_drift(tmp_path):
    checkpoints = ["seed_101/g8"]
    manifest = _artifact_manifest(tmp_path, checkpoints)
    (tmp_path / checkpoints[0]).write_text("mutated\n", encoding="utf-8")
    rows = _rows(["A4"], [(checkpoints[0], "P1", 8)])
    report = check_research_grade(
        checkpoints=checkpoints,
        rows=rows,
        manifest=manifest,
        systems=["A4"],
        n_positions=1,
        n_budgets=1,
        min_seed_families=1,
        artifact_root=tmp_path,
    )
    assert report["checks"]["artifact_hashes"]["ok"] is False
    assert report["checks"]["artifact_hashes"]["invalid_artifacts"][0]["reason"] == "sha256_mismatch"
