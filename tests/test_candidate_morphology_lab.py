"""Tests for candidate_morphology_lab and the H1 synthetic discrimination gate."""

import importlib.util
import json
from pathlib import Path

import pytest

from quartz.experiments import candidate_morphology as cm
from quartz.experiments import h1_synthetic_gate as h1

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# candidate morphology world + regret decomposition
# --------------------------------------------------------------------------- #

def test_build_world_splits_by_prior_and_is_deterministic():
    means = [0.66, 0.5, 0.5, 0.5, 0.5, 0.5]
    a = cm.build_world(means, n_visible=2, prior_noise=0.1, seed=7, trial=3)
    b = cm.build_world(means, n_visible=2, prior_noise=0.1, seed=7, trial=3)
    assert a.visible == b.visible and a.hidden_queue == b.hidden_queue
    assert len(a.visible) == 2 and len(a.hidden_queue) == 4
    assert set(a.visible) | set(a.hidden_queue) == set(range(6))
    assert a.global_best_mean == pytest.approx(0.66)
    # zero prior noise => visible pool is exactly the true top arms
    clean = cm.build_world(means, n_visible=1, prior_noise=0.0, seed=1, trial=0)
    assert clean.visible == [0]


def test_regret_decomposition_identity():
    means = [0.62, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    records, _ = cm.run_experiment(
        means, n_visible=3, prior_noise=0.15, budgets=[16, 32], widen_costs=[1, 4], trials=25, seed=99
    )
    assert records
    for r in records:
        assert r.total_regret == pytest.approx(r.omission_regret + r.ranking_regret)
        assert r.omission_regret >= -1e-12
        assert r.ranking_regret >= -1e-12
        assert r.correct_selection in (0, 1)
        assert r.best_revealed in (0, 1)
        # omission is zero exactly when the global best was revealed
        assert (r.omission_regret <= 1e-12) == bool(r.best_revealed)


def test_no_widen_never_reveals_and_spends_full_budget():
    means = [0.62, 0.5, 0.5, 0.5, 0.5, 0.5]
    records, _ = cm.run_experiment(
        means, n_visible=2, prior_noise=0.2, budgets=[32], widen_costs=[2], trials=10, seed=5,
        allocators=["no_widen"],
    )
    for r in records:
        assert r.n_widens == 0
        assert r.n_visible_at_commit == 2
        assert r.budget_used == 32
        assert r.stopped_early == 0


def test_eager_widen_reveals_all_affordable():
    means = [0.62, 0.5, 0.5, 0.5, 0.5, 0.5]  # 6 arms, 2 visible => 4 hidden
    # cheap widen + ample budget => all four hidden revealed (6 visible at commit)
    records, _ = cm.run_experiment(
        means, n_visible=2, prior_noise=0.2, budgets=[64], widen_costs=[1], trials=8, seed=3,
        allocators=["eager_widen"],
    )
    for r in records:
        assert r.n_widens == 4
        assert r.n_visible_at_commit == 6
        assert r.best_revealed == 1  # everything visible => best always revealed


def test_priced_widen_does_not_widen_when_unaffordable():
    means = [0.62, 0.5, 0.5, 0.5, 0.5, 0.5]
    # widen_cost larger than the budget can afford with reserve => no widening
    records, _ = cm.run_experiment(
        means, n_visible=2, prior_noise=0.2, budgets=[16], widen_costs=[64], trials=8, seed=11,
        allocators=["priced_widen"],
    )
    for r in records:
        assert r.n_widens == 0


def test_summarize_paired_delta_linearity():
    means = [0.6, 0.56, 0.53, 0.5, 0.5, 0.5, 0.5, 0.5]
    _, summaries = cm.run_experiment(
        means, n_visible=4, prior_noise=0.12, budgets=[32], widen_costs=[1, 2], trials=40, seed=42
    )
    non_baseline = [s for s in summaries if s["allocator"] != cm.BASELINE_ALLOCATOR]
    assert non_baseline
    for s in non_baseline:
        assert s["paired_trials_vs_baseline"] == 40
        total = s["paired_total_delta_vs_baseline"]
        om = s["paired_omission_delta_vs_baseline"]
        rank = s["paired_ranking_delta_vs_baseline"]
        assert total == pytest.approx(om + rank, abs=1e-9)
    baseline = [s for s in summaries if s["allocator"] == cm.BASELINE_ALLOCATOR]
    for s in baseline:
        assert s["paired_omission_delta_vs_baseline"] is None


def test_widening_kill_verdict_demote_and_keep():
    demote = [
        {"scenario_id": "x", "allocator": "priced_widen", "budget": 16, "widen_cost": 1,
         "paired_omission_delta_vs_baseline": 0.01, "paired_omission_delta_mc95_high": 0.03,
         "paired_ranking_delta_vs_baseline": 0.0,
         "paired_total_delta_vs_baseline": 0.01, "paired_total_delta_mc95_high": 0.03},
    ]
    v = cm.widening_kill_verdict(demote)
    assert v["widening_lane_demoted"] is True
    assert v["n_ci_separated_omission_improvements"] == 0

    keep = [
        {"scenario_id": "y", "allocator": "eager_widen", "budget": 32, "widen_cost": 1,
         "paired_omission_delta_vs_baseline": -0.04, "paired_omission_delta_mc95_high": -0.02,
         "paired_ranking_delta_vs_baseline": 0.08,
         "paired_total_delta_vs_baseline": 0.04, "paired_total_delta_mc95_high": 0.06},
    ]
    v2 = cm.widening_kill_verdict(keep)
    assert v2["widening_lane_demoted"] is False
    assert v2["n_ci_separated_omission_improvements"] == 1
    # omission improved but total did not => no net improvement
    assert v2["net_total_improvement_found"] is False
    assert v2["ci_separated_omission_improvements"][0]["scenario_id"] == "y"


# --------------------------------------------------------------------------- #
# H1 synthetic discrimination gate
# --------------------------------------------------------------------------- #

def test_synthetic_counts_shape_and_total():
    counts = h1.synthetic_counts(0.5, 32, 6)
    assert len(counts) == 6
    assert sum(counts) == 32
    assert counts[0] == max(counts)  # peak on arm 0
    uniform = h1.synthetic_counts(0.0, 30, 6)
    assert max(uniform) - min(uniform) <= 1  # near-uniform


def test_h1_gate_passes_on_synthetic_ground_truth():
    g = h1.run_gate(n_boot=800)
    assert g["gate_pass"] is True
    assert g["discriminates_at_each_budget"] is True
    assert g["not_saturated"] is True
    assert g["monotone_in_budget"] is True
    assert g["stability_global_min"] < 0.99


def test_h1_gate_detects_saturation():
    # feed only sharply-peaked positions => stability pinned near 1 => not discriminating
    g = h1.run_gate(peaks=(0.7, 0.75, 0.8), budgets=(32, 64), n_arms=6, n_boot=800)
    assert g["discriminates_at_each_budget"] is False
    assert g["gate_pass"] is False


# --------------------------------------------------------------------------- #
# scenario bank + runner smoke
# --------------------------------------------------------------------------- #

def _load_runner():
    path = REPO_ROOT / "scripts" / "candidate_morphology_lab.py"
    spec = importlib.util.spec_from_file_location("candidate_morphology_lab", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checked_in_scenario_bank_validates():
    runner = _load_runner()
    bank = runner.load_scenario_bank(runner.DEFAULT_SCENARIO_BANK)
    assert bank["experiment_id"] == cm.EXPERIMENT_ID
    assert len(bank["scenarios"]) >= 3


def test_runner_smoke_writes_artifacts(tmp_path):
    runner = _load_runner()
    out = tmp_path / "run"
    rc = runner.main([
        "--scenarios", "moderate_noise_k8",
        "--trials", "12",
        "--budgets", "16,32",
        "--widen-costs", "1,4",
        "--skip-h1-gate",
        "--output-dir", str(out),
    ])
    assert rc == 0
    for name in ("run_manifest.json", "summary.csv", "summary.json", "trials.jsonl.gz"):
        assert (out / name).exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["claim_status"] == "synthetic_screening_only"
    assert "widening_kill_verdict" in summary
    assert "widening_lane_demoted" in summary["widening_kill_verdict"]
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["artifacts"]
    assert manifest["prohibited_inferences"]
