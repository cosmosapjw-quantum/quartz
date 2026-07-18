"""Tests for scripts/phase15_o6_burst_precision.py (Stage 7 / C9)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = REPO_ROOT / "scripts" / "phase15_o6_burst_precision.py"
    spec = importlib.util.spec_from_file_location("phase15_o6_burst_precision", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _records(n, burst_hard_rate, nonburst_hard_rate, burst_frac, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        burst = 1 if rng.random() < burst_frac else 0
        rate = burst_hard_rate if burst else nonburst_hard_rate
        hard = 1 if rng.random() < rate else 0
        recs.append(
            {
                "checkpoint_id": "C1",
                "position_id": f"P{i}",
                "burst": burst,
                "hard": hard,
            }
        )
    return recs


def test_o6_lift_alive_when_burst_tracks_difficulty():
    m = _load()
    # bursts land on hard positions (0.8) far above base (0.2) => lift > 1
    recs = _records(
        400, burst_hard_rate=0.8, nonburst_hard_rate=0.15, burst_frac=0.3, seed=1
    )
    res = m.compute_o6_lift(recs, n_boot=500, seed=0)
    assert res["degenerate"] is False
    assert res["lift"] > 1.2
    assert res["lift_ci_low"] > 1.0
    assert res["o6_lane_alive"] is True
    assert res["o6_kill_lift_ci_includes_one"] is False


def test_o6_kill_when_burst_fires_at_base_rate():
    m = _load()
    # burst hard-rate == base hard-rate => lift ~ 1 => CI includes 1 => kill
    recs = _records(
        400, burst_hard_rate=0.3, nonburst_hard_rate=0.3, burst_frac=0.3, seed=2
    )
    res = m.compute_o6_lift(recs, n_boot=500, seed=0)
    assert res["lift"] == pytest.approx(1.0, abs=0.25)
    assert res["o6_kill_lift_ci_includes_one"] is True
    assert res["o6_lane_alive"] is False


def test_o6_degenerate_when_too_few_burst_events():
    m = _load()
    recs = _records(
        400, burst_hard_rate=0.9, nonburst_hard_rate=0.2, burst_frac=0.03, seed=3
    )
    res = m.compute_o6_lift(recs, n_boot=300, seed=0)
    # < 30 burst events among 400 * 0.03 ~ 12 => degenerate
    assert res["degenerate"] is True
    assert res["o6_lane_alive"] is False


def test_o6_degenerate_when_burst_rate_saturated():
    m = _load()
    recs = _records(
        200, burst_hard_rate=0.5, nonburst_hard_rate=0.5, burst_frac=0.98, seed=4
    )
    res = m.compute_o6_lift(recs, n_boot=200, seed=0)
    assert res["degenerate"] is True
    assert res["degenerate_reason"] == "burst_rate_out_of_range"


def test_build_records_join_and_missing_bundle_excluded():
    m = _load()
    from quartz.experiments.forked_voc import label_trace_bundle  # noqa: F401

    online_rows = [
        {"checkpoint_id": "C1", "position_id": "P1", "budget_burst_triggered": 1},
        {
            "checkpoint_id": "C1",
            "position_id": "P1",
            "budget_burst_triggered": 0,
        },  # OR => burst 1
        {"checkpoint_id": "C1", "position_id": "P2", "budget_burst_triggered": 0},
        {
            "checkpoint_id": "C1",
            "position_id": "P3",
            "budget_burst_triggered": 1,
        },  # bundle missing
    ]
    # P1: late-flip bundle (overturns => hard 1); P2: stable (hard 0). P3 has no bundle.
    bundles = {
        ("C1", "P1"): {
            "trace_budgets": [8, 16, 32],
            "trace_policies": [[0.45, 0.44, 0.11], [0.4, 0.5, 0.1], [0.3, 0.62, 0.08]],
        },
        ("C1", "P2"): {
            "trace_budgets": [8, 16, 32],
            "trace_policies": [[0.7, 0.2, 0.1], [0.7, 0.2, 0.1], [0.7, 0.2, 0.1]],
        },
    }
    records = m.build_records(online_rows, bundles)
    by_pos = {r["position_id"]: r for r in records}
    assert set(by_pos) == {"P1", "P2"}  # P3 excluded (missing bundle)
    assert by_pos["P1"]["burst"] == 1 and by_pos["P1"]["hard"] == 1
    assert by_pos["P2"]["burst"] == 0 and by_pos["P2"]["hard"] == 0
