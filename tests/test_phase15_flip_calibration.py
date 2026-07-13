"""Tests for scripts/phase15_flip_calibration.py (Stage 7 / C8)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = REPO_ROOT / "scripts" / "phase15_flip_calibration.py"
    spec = importlib.util.spec_from_file_location("phase15_flip_calibration", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_reliability_bins_and_ece_hand_computed():
    m = _load()
    rel = m.reliability_diagram([0.1, 0.1, 0.9, 0.9], [0, 0, 1, 1], n_bins=2)
    assert rel["n"] == 4
    # bin0 [0,0.5): mean_pred 0.1, mean_obs 0; bin1 [0.5,1]: mean_pred 0.9, obs 1
    assert rel["bins"][0]["mean_pred"] == pytest.approx(0.1)
    assert rel["bins"][0]["mean_obs"] == pytest.approx(0.0)
    assert rel["bins"][1]["mean_obs"] == pytest.approx(1.0)
    # ECE = 0.5*|0-0.1| + 0.5*|1-0.9| = 0.1
    assert rel["ece"] == pytest.approx(0.1)


def test_brier_matches_closed_form():
    m = _load()
    rel = m.reliability_diagram([0.1, 0.1, 0.9, 0.9], [0, 0, 1, 1], n_bins=2)
    # mean((0.1)^2,(0.1)^2,(0.1)^2,(0.1)^2) = 0.01
    assert rel["brier"] == pytest.approx(0.01)


def test_virtual_stop_realized_budget_accounting():
    m = _load()
    recs = [
        {"budget": 8, "s_h1": 0.5, "s_pflip": 0.4, "y": 0},
        {"budget": 16, "s_h1": 0.95, "s_pflip": 0.6, "y": 1},
        {"budget": 32, "s_h1": 0.99, "s_pflip": 0.99, "y": 1},
    ]
    # H1 fires at budget 16 (first >= 0.9)
    r = m.virtual_stop_budget(recs, "s_h1", 0.9)
    assert r["stop_budget"] == 16 and r["agreement"] == 1 and r["fired"] is True
    # P_flip never reaches 0.9 until 32 => stops at 32
    r2 = m.virtual_stop_budget(recs, "s_pflip", 0.9)
    assert r2["stop_budget"] == 32 and r2["fired"] is True
    # threshold above all => never fires => stop at last decision budget
    r3 = m.virtual_stop_budget(recs, "s_h1", 1.5)
    assert r3["stop_budget"] == 32 and r3["fired"] is False


def test_bundle_decision_records_excludes_holdout():
    m = _load()
    bundle = {
        "trace_budgets": [8, 16, 32],
        "trace_policies": [[0.6, 0.4], [0.55, 0.45], [0.9, 0.1]],
        "trace_p_flips": [0.4, 0.3, 0.05],
    }
    recs = m.bundle_decision_records(bundle, n_boot=500)
    # holdout budget (32) excluded => 2 decision records at 8, 16
    assert [r["budget"] for r in recs] == [8, 16]
    # y = argmax(pi_b) == argmax(holdout=[0.9,0.1] -> 0); all argmax 0 => y=1
    assert all(r["y"] == 1 for r in recs)
    assert all(r["s_pflip"] is not None for r in recs)


def _pair_bundle(y8, y16, sh1_8, sh1_16, spf_8, spf_16):
    return [
        {"budget": 8, "s_h1": sh1_8, "s_pflip": spf_8, "y": y8},
        {"budget": 16, "s_h1": sh1_16, "s_pflip": spf_16, "y": y16},
    ]


def test_matched_budget_h1_dies_when_pflip_agrees_more():
    m = _load()
    # A: H1 stops@8 (wrong, y=0); P_flip stops@16 (right, y=1)
    # B: H1 stops@16 (wrong, y=0); P_flip stops@8 (right, y=1)
    # mean budget both = 12 (matched); every paired delta = -1 => H1 dies
    a = _pair_bundle(0, 1, 0.95, 0.99, 0.5, 0.95)
    b = _pair_bundle(1, 0, 0.5, 0.99, 0.95, 0.5)
    per_bundle = [a, b] * 6
    res = m.matched_budget_calibration(per_bundle, h1_threshold=0.9, seed=0)
    assert res["insufficient"] is False
    assert res["budget_rel_gap"] == pytest.approx(0.0, abs=1e-9)
    assert res["mean_agreement_delta_h1_minus_pflip"] == pytest.approx(-1.0)
    assert res["h1_dies"] is True
    assert res["h1_survives"] is False


def test_matched_budget_h1_survives_on_tie():
    m = _load()
    # Both stop at 16 with the same agreement => delta 0 => survives.
    a = _pair_bundle(0, 1, 0.5, 0.95, 0.5, 0.95)
    per_bundle = [a] * 8
    res = m.matched_budget_calibration(per_bundle, h1_threshold=0.9, seed=0)
    assert res["insufficient"] is False
    assert res["mean_agreement_delta_h1_minus_pflip"] == pytest.approx(0.0)
    assert res["h1_dies"] is False
    assert res["h1_survives"] is True


def test_analyze_runs_end_to_end_on_synthetic_bundles():
    m = _load()
    bundles = [
        {"trace_budgets": [8, 16, 32], "trace_policies": [[0.5, 0.5], [0.7, 0.3], [0.95, 0.05]], "trace_p_flips": [0.5, 0.2, 0.02]},
        {"trace_budgets": [8, 16, 32], "trace_policies": [[0.4, 0.6], [0.5, 0.5], [0.9, 0.1]], "trace_p_flips": [0.5, 0.4, 0.05]},
    ]
    res = m.analyze(bundles, n_boot=400, seed=0)
    assert res["n_bundles"] == 2
    assert res["reliability_h1"]["n"] >= 1
    assert "matched_budget_calibration" in res
