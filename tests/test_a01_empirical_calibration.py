"""Test suite for A01 empirical calibration and wrong-stop gate evaluation."""

import pytest
from scripts.a01_empirical_calibration import evaluate_flip_calibration, clopper_pearson_upper_95


def test_clopper_pearson_upper_95() -> None:
    # 0 wrong stops out of 100
    upper_0 = clopper_pearson_upper_95(0, 100)
    assert 0.0 < upper_0 <= 0.05
    # 5 wrong stops out of 100
    upper_5 = clopper_pearson_upper_95(5, 100)
    assert upper_5 > 0.05


def test_evaluate_flip_calibration_ideal() -> None:
    # Perfect calibration: 100 stops with 0 flips (pred 0.01), 100 non-stops with flips (pred 0.99)
    rows = [{"p_flip": 0.01, "flipped_at_continuation": 0} for _ in range(100)] + [
        {"p_flip": 0.99, "flipped_at_continuation": 1} for _ in range(100)
    ]
    res = evaluate_flip_calibration(rows, risk_limit=0.05)
    assert res["status"] == "EVALUATED"
    assert res["n_samples"] == 200
    assert res["n_stops"] == 100
    assert res["n_wrong_stops"] == 0
    assert res["wrong_stop_rate"] == 0.0
    assert res["brier_score"] < 0.01
    assert res["ece"] < 0.02
    assert res["quality_gate_passed"] is True
