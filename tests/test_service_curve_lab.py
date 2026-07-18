"""Tests for service_curve_lab — GPU evaluator service curve (CPU-safe)."""

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from quartz.experiments import service_curve as sc

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_eval_model_output_shapes():
    m = sc.build_eval_model(in_ch=4, channels=8, blocks=2, board=5, actions=25).eval()
    x = torch.randn(3, 4, 5, 5)
    with torch.no_grad():
        p, v = m(x)
    assert p.shape == (3, 25)
    assert v.shape == (3, 1)


def test_measure_point_fields_on_cpu():
    m = sc.build_eval_model(in_ch=4, channels=8, blocks=1, board=5, actions=25).eval()
    dev = torch.device("cpu")
    r = sc.measure_point(
        m, dev, batch_size=4, inflight=2, in_ch=4, board=5, n_waves=3, warmup=1
    )
    assert r["batch_size"] == 4 and r["inflight"] == 2
    assert r["items_per_s"] > 0.0
    assert r["ms_per_batch"] > 0.0
    assert r["power_watts"] is None  # cpu => no nvidia-smi sampling


def test_measure_point_deterministic_with_injected_clock():
    m = sc.build_eval_model(in_ch=4, channels=8, blocks=1, board=5, actions=25).eval()
    dev = torch.device("cpu")
    ticks = iter([0.0, 2.0])  # start=0, end=2 seconds elapsed

    def fake_clock():
        return next(ticks)

    r = sc.measure_point(
        m,
        dev,
        batch_size=10,
        inflight=2,
        in_ch=4,
        board=5,
        n_waves=5,
        warmup=1,
        time_fn=fake_clock,
    )
    # items = batch*inflight*n_waves = 10*2*5 = 100 over 2.0s => 50 items/s
    assert r["items_per_s"] == pytest.approx(50.0)
    # ms/batch = 2000ms / (inflight*n_waves=10) = 200
    assert r["ms_per_batch"] == pytest.approx(200.0)


def test_scheduler_verdict_alive_and_dead():
    # inflight 2 beats inflight 1 by 20% at batch 64 => lane alive
    alive_rows = [
        {"batch_size": 32, "inflight": 1, "items_per_s": 800.0},
        {"batch_size": 64, "inflight": 1, "items_per_s": 1000.0},
        {"batch_size": 64, "inflight": 2, "items_per_s": 1200.0},
    ]
    v = sc.scheduler_verdict(alive_rows, min_gain=0.05)
    assert v["h4_inflight_scheduler_lane_alive"] is True
    assert v["best_overall_inflight"] == 2
    assert v["inflight_throughput_gain"] == pytest.approx(0.2)

    # inflight never beats fixed by more than min_gain => lane demoted
    dead_rows = [
        {"batch_size": 64, "inflight": 1, "items_per_s": 1000.0},
        {"batch_size": 64, "inflight": 2, "items_per_s": 1010.0},
        {"batch_size": 128, "inflight": 1, "items_per_s": 1020.0},
    ]
    d = sc.scheduler_verdict(dead_rows, min_gain=0.05)
    assert d["h4_inflight_scheduler_lane_alive"] is False


def test_scheduler_verdict_knee():
    rows = [
        {"batch_size": 16, "inflight": 1, "items_per_s": 500.0},
        {"batch_size": 64, "inflight": 1, "items_per_s": 950.0},
        {"batch_size": 256, "inflight": 1, "items_per_s": 1000.0},
    ]
    v = sc.scheduler_verdict(rows)
    # smallest batch reaching >=90% of peak (1000) is batch 64 (950)
    assert v["knee"]["batch_size"] == 64


def _load_runner():
    path = REPO_ROOT / "scripts" / "service_curve_lab.py"
    spec = importlib.util.spec_from_file_location("service_curve_lab", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_loads():
    runner = _load_runner()
    cfg = runner.load_config(runner.DEFAULT_CONFIG)
    assert cfg["experiment_id"] == sc.EXPERIMENT_ID
    assert cfg["prohibited_inferences"]


def test_runner_smoke_on_cpu(tmp_path):
    runner = _load_runner()
    out = tmp_path / "run"
    rc = runner.main(
        [
            "--device",
            "cpu",
            "--batch-sizes",
            "4,8",
            "--inflight-grid",
            "1,2",
            "--n-waves",
            "2",
            "--warmup",
            "1",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0
    for name in ("run_manifest.json", "service_curve.csv", "summary.json"):
        assert (out / name).exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["device"] == "cpu"
    assert "scheduler_verdict" in summary
    assert len(summary["rows"]) == 4
