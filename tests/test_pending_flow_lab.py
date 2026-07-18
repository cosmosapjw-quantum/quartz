"""Tests for pending_flow_lab — count-only WU-UCT synthetic screen + Rust bridge parse."""

import importlib.util
import json
from pathlib import Path

import pytest

from quartz.experiments import pending_flow as pf

REPO_ROOT = Path(__file__).resolve().parents[1]

ARMS = [0.62, 0.58, 0.55, 0.53, 0.50, 0.48, 0.45, 0.42]

# A minimal Rust VL-ablation Ablation-1 table snippet (the real format).
RUST_LOG_SNIPPET = """
          Mode   Agree  Entrop  Q_Sprd      NPS   AvgVV   DupRt  MaxP
  ────────────────────────────────────────────────────────────────────
    Fixed(1,1)   55.0%   2.875   1.024    25000   1.000   0.129     3
      Adaptive   55.0%   3.060   1.095        0   0.142   0.218     3
    VvisitOnly   55.0%   3.000   1.025        0   0.000   0.328     3
    VvalueOnly   60.0%   3.018   0.990        0   0.102   0.000     0
      Disabled   40.0%   3.226   1.035        0   0.000   0.000     0
"""


def test_simulate_bounds_and_crn():
    a = pf.simulate(ARMS, 8, "fixed", waves=120, seed=7)
    b = pf.simulate(ARMS, 8, "fixed", waves=120, seed=7)
    assert a == b  # common random numbers => deterministic
    assert 0.0 <= a["mean_dup_rate"] <= 1.0
    assert 0.0 <= a["mean_throughput"] <= 1.0
    assert a["best_arm_visit_share"] > 1.0 / len(ARMS)  # concentrates on the best arm


def test_disabled_collides_more_than_fixed_at_high_w():
    dis = pf.simulate(ARMS, 16, "disabled", waves=200, seed=1)
    fix = pf.simulate(ARMS, 16, "fixed", waves=200, seed=1)
    # virtual loss must de-collide: disabled piles up, fixed spreads
    assert dis["mean_dup_rate"] > fix["mean_dup_rate"]
    assert dis["mean_throughput"] < fix["mean_throughput"]


def test_single_worker_never_collides():
    for policy in pf.VL_POLICIES:
        r = pf.simulate(ARMS, 1, policy, waves=80, seed=3)
        assert r["mean_dup_rate"] == pytest.approx(0.0)
        assert r["mean_throughput"] == pytest.approx(1.0)


def test_screen_and_kill_verdict_shape():
    grid = [1, 2, 4, 8, 16]
    rows = pf.screen(ARMS, grid, waves=150, seed=20260713)
    assert len(rows) == len(grid) * len(pf.VL_POLICIES)
    v = pf.kill_verdict(rows, grid)
    assert "h5_adaptive_dup_lane_alive" in v
    assert "h4_adaptive_throughput_lane_alive" in v
    assert v["high_worker"] == 16 and v["low_worker"] == 2
    # in this abstract model adaptive does not beat fixed on collisions
    assert v["h5_adaptive_dup_lane_alive"] is False


def _load_runner():
    path = REPO_ROOT / "scripts" / "pending_flow_lab.py"
    spec = importlib.util.spec_from_file_location("pending_flow_lab", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_rust_ablation_finds_the_real_reframing():
    runner = _load_runner()
    parsed = runner.parse_rust_ablation(RUST_LOG_SNIPPET)
    assert set(parsed["modes"]) >= {"Fixed(1,1)", "Adaptive", "Disabled"}
    # the key real finding: adaptive does NOT reduce dup_rate ...
    assert parsed["real_adaptive_reduces_dup_rate"] is False
    assert parsed["real_dup_rate_adaptive"] > parsed["real_dup_rate_fixed"]
    # ... but it DOES lower virtual-loss pessimism at preserved agreement
    assert parsed["real_adaptive_lowers_pessimism"] is True
    assert parsed["real_avg_vvalue_adaptive"] < parsed["real_avg_vvalue_fixed"]
    assert parsed["real_agreement_preserved"] is True


def test_config_loads():
    runner = _load_runner()
    cfg = runner.load_config(runner.DEFAULT_CONFIG)
    assert cfg["experiment_id"] == pf.EXPERIMENT_ID
    assert cfg["scenarios"]


def test_runner_smoke_with_rust_bridge(tmp_path):
    runner = _load_runner()
    log_path = tmp_path / "rust.log"
    log_path.write_text(RUST_LOG_SNIPPET, encoding="utf-8")
    out = tmp_path / "run"
    rc = runner.main(
        [
            "--scenarios",
            "one_best_cluster_k8",
            "--waves",
            "120",
            "--worker-grid",
            "1,2,4,8",
            "--rust-log",
            str(log_path),
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0
    for name in ("run_manifest.json", "summary.json", "rust_vl_ablation.log"):
        assert (out / name).exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["combined_verdict"]["h5_dup_reduction_lane_demoted"] is True
    assert summary["rust_bridge"]["real_adaptive_lowers_pessimism"] is True
