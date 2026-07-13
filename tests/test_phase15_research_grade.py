"""Tests for quartz.phase15_research_grade (Stage 7 / C10)."""

import pytest

from quartz.phase15_research_grade import (
    check_paired_coverage,
    check_research_grade,
    check_seed_families,
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


def test_paired_coverage_equal_and_mismatch():
    keys = [("C1", "P1", 8), ("C1", "P1", 16)]
    ok, d = check_paired_coverage(_rows(["A4", "B13"], keys), ["A4", "B13"])
    assert ok is True and d["mismatched_systems"] == []
    # drop one tuple from B13 => mismatch
    rows = _rows(["A4", "B13"], keys)
    rows = [r for r in rows if not (r["system"] == "B13" and r["budget"] == 16)]
    ok2, d2 = check_paired_coverage(rows, ["A4", "B13"])
    assert ok2 is False and "B13" in d2["mismatched_systems"]


def test_research_grade_compliant_and_failures():
    keys = [("seed_101/g8", "P1", 8), ("seed_102/g8", "P1", 8), ("seed_103/g8", "P1", 8)]
    checkpoints = ["seed_101/g8", "seed_102/g8", "seed_103/g8"]
    systems = ["A4", "B13"]
    rows = _rows(systems, keys)
    manifest = {
        "stage7_artifact_hashes": {
            "checkpoints": {c: "abc" for c in checkpoints},
            "positions": "deadbeef",
            "systems_config": "cafef00d",
        }
    }
    report = check_research_grade(
        checkpoints=checkpoints, rows=rows, manifest=manifest, systems=systems,
        n_positions=1, n_budgets=1, analyzer_report={"interpretation_flags": {}}, min_seed_families=3,
    )
    assert report["research_grade_ready"] is True
    assert report["unmet"] == []
    enforce_research_grade(report)  # must not raise

    # remove a checkpoint hash => artifact_hashes fails
    bad_manifest = {"stage7_artifact_hashes": {"checkpoints": {}, "positions": "x", "systems_config": "y"}}
    bad = check_research_grade(
        checkpoints=checkpoints, rows=rows, manifest=bad_manifest, systems=systems,
        n_positions=1, n_budgets=1, min_seed_families=3,
    )
    assert bad["research_grade_ready"] is False
    assert "artifact_hashes" in bad["unmet"]
    with pytest.raises(SystemExit):
        enforce_research_grade(bad)

    # only 2 seed families => seed_families fails
    few = check_research_grade(
        checkpoints=["seed_101/g8", "seed_101/g5"], rows=rows, manifest=manifest, systems=systems,
        n_positions=1, n_budgets=1, min_seed_families=3,
    )
    assert "seed_families" in few["unmet"]


def test_rows_preserved_detects_dropped_rows():
    keys = [("seed_101/g8", "P1", 8), ("seed_102/g8", "P1", 8)]
    systems = ["A4", "B13"]
    rows = _rows(systems, keys)
    # 2 systems * 1 checkpoint-count?? here checkpoints distinct per key; use n_checkpoints=2
    report = check_research_grade(
        checkpoints=["seed_101/g8", "seed_102/g8"], rows=rows, manifest={"stage7_artifact_hashes": {"checkpoints": {"seed_101/g8": "h", "seed_102/g8": "h"}, "positions": "p", "systems_config": "c"}},
        systems=systems, n_positions=1, n_budgets=1, min_seed_families=2,
    )
    # expected = 2 ckpt * 1 pos * 1 budget * 2 systems = 4; actual = 4
    assert report["checks"]["rows_preserved"]["ok"] is True
    # drop one row => fails
    report2 = check_research_grade(
        checkpoints=["seed_101/g8", "seed_102/g8"], rows=rows[:-1], manifest={"stage7_artifact_hashes": {"checkpoints": {"seed_101/g8": "h", "seed_102/g8": "h"}, "positions": "p", "systems_config": "c"}},
        systems=systems, n_positions=1, n_budgets=1, min_seed_families=2,
    )
    assert report2["checks"]["rows_preserved"]["ok"] is False
