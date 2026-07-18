"""Tests for quartz.phase15_signatures — the Part B / B1 signature battery.

These pin the game-agnostic search-behavior metrics against hand-derived
values and pin the THESIS.md P3 non-circularity contract for VOC-tightness
(it must accept an offline oracle-derived proxy as an explicit argument).
"""

import math

import numpy as np
import pytest

from quartz.phase15_signatures import (
    SIGNATURE_SCHEMA_VERSION,
    argmax_path,
    budget_entropy,
    budget_gini,
    concentration_vs_budget,
    final_sparsity,
    first_revision_step,
    flip_flop_rate,
    k_eff,
    k_eff_trajectory,
    policy_entropy,
    trace_signature_summary,
    voc_tightness,
)


def test_policy_entropy_uniform_and_delta():
    # uniform over K → H = log K; K_eff = K.
    for K in (2, 4, 8):
        p = [1.0 / K] * K
        assert policy_entropy(p) == pytest.approx(math.log(K))
        assert k_eff(p) == pytest.approx(K)
    # delta → H = 0, K_eff = 1.
    assert policy_entropy([0.0, 1.0, 0.0]) == pytest.approx(0.0)
    assert k_eff([0.0, 1.0, 0.0]) == pytest.approx(1.0)


def test_entropy_accepts_visit_counts_not_only_probabilities():
    # Visit-count vector renormalizes; [2,2] behaves like uniform-2.
    assert k_eff([2, 2]) == pytest.approx(2.0)
    assert k_eff([9, 1]) == pytest.approx(k_eff([0.9, 0.1]))


def test_empty_and_zero_vectors_are_safe():
    assert policy_entropy([]) == 0.0
    assert k_eff([]) == 0.0
    assert k_eff([0.0, 0.0]) == 0.0
    assert argmax_path([[0.0, 0.0], [0.1, 0.9]]) == [-1, 1]


def test_argmax_path_and_first_revision():
    # argmax goes 0,0,1  → first revision at step 2.
    tp = [[0.6, 0.4], [0.55, 0.45], [0.4, 0.6]]
    assert argmax_path(tp) == [0, 0, 1]
    assert first_revision_step(tp) == 2
    # never revises → None.
    assert first_revision_step([[0.9, 0.1], [0.8, 0.2]]) is None


def test_flip_flop_rate():
    # argmax 0,1,0,1 over 4 steps → 3 changes / 3 adjacencies = 1.0.
    tp = [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]]
    assert flip_flop_rate(tp) == pytest.approx(1.0)
    # stable → 0.0.
    assert flip_flop_rate([[0.9, 0.1], [0.8, 0.2]]) == pytest.approx(0.0)
    # single step → 0.0 (no adjacency).
    assert flip_flop_rate([[0.5, 0.5]]) == pytest.approx(0.0)


def test_empty_steps_are_skipped_not_counted_as_revision():
    # An all-zero (empty) middle step must not register as a flip.
    tp = [[0.9, 0.1], [0.0, 0.0], [0.9, 0.1]]
    assert first_revision_step(tp) is None
    assert flip_flop_rate(tp) == pytest.approx(0.0)


def test_final_sparsity():
    # final policy uniform over 2 of 4 legal → K_eff=2, /4 = 0.5.
    tp = [[0.25, 0.25, 0.25, 0.25], [0.5, 0.5, 0.0, 0.0]]
    assert final_sparsity(tp, n_legal=4) == pytest.approx(0.5)
    # no n_legal → denominator = support size (2) → 1.0.
    assert final_sparsity(tp) == pytest.approx(1.0)
    assert final_sparsity([]) == 0.0


def test_concentration_vs_budget_slope_negative_when_narrowing():
    # K_eff falls as budget grows → negative slope per log-budget.
    budgets = [8, 16, 32, 64]
    tp = [
        [0.25, 0.25, 0.25, 0.25],  # K_eff 4
        [0.4, 0.3, 0.2, 0.1],
        [0.7, 0.2, 0.1, 0.0],
        [0.9, 0.1, 0.0, 0.0],  # K_eff ~1.4
    ]
    out = concentration_vs_budget(budgets, tp)
    assert out["k_eff_first"] == pytest.approx(4.0)
    assert out["k_eff_last"] < out["k_eff_first"]
    assert out["k_eff_slope_per_log_budget"] is not None
    assert out["k_eff_slope_per_log_budget"] < 0.0


def test_budget_gini_flat_vs_concentrated():
    # Every move equal → Gini 0.
    assert budget_gini([10, 10, 10, 10]) == pytest.approx(0.0, abs=1e-9)
    # One move hoards all budget → Gini → (n-1)/n = 0.75 for n=4.
    assert budget_gini([0, 0, 0, 40]) == pytest.approx(0.75, abs=1e-9)
    # fewer than 2 moves → 0.
    assert budget_gini([5]) == 0.0
    assert budget_gini([0, 0]) == 0.0


def test_budget_entropy_matches_policy_entropy():
    assert budget_entropy([1, 1, 1, 1]) == pytest.approx(math.log(4))


def test_voc_tightness_requires_explicit_oracle_proxy_and_ranks_monotone():
    # Perfect monotone relation → spearman 1.0 (non-circularity guard: the
    # proxy is supplied by the caller, never computed from the engine).
    budgets = [8, 16, 24, 40, 64]
    voc_proxy = [0.1, 0.2, 0.35, 0.5, 0.9]  # analyst/oracle-derived
    assert voc_tightness(budgets, voc_proxy) == pytest.approx(1.0)
    # Perfect anti-monotone → -1.0.
    assert voc_tightness(budgets, list(reversed(voc_proxy))) == pytest.approx(-1.0)


def test_voc_tightness_undefined_cases_return_none():
    assert voc_tightness([1.0], [2.0]) is None  # <2 points
    assert voc_tightness([5, 5, 5], [1, 2, 3]) is None  # zero variance in x
    assert voc_tightness([1, 2, 3], [7, 7, 7]) is None  # zero variance in y


def test_voc_tightness_pearson_vs_spearman_and_bad_method():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [1.0, 4.0, 9.0, 16.0]  # monotone but nonlinear
    # spearman sees perfect monotonicity; pearson < 1 due to curvature.
    assert voc_tightness(x, y, method="spearman") == pytest.approx(1.0)
    assert voc_tightness(x, y, method="pearson") < 1.0
    with pytest.raises(ValueError):
        voc_tightness(x, y, method="kendall")


def test_trace_signature_summary_bundles_o1_o5():
    trace = {
        "trace_budgets": [8, 32, 64],
        "trace_policies": [
            [0.34, 0.33, 0.33],
            [0.5, 0.3, 0.2],
            [0.8, 0.15, 0.05],
        ],
        "trace_latencies_ms": [1.0, 2.0, 3.0],
    }
    s = trace_signature_summary(trace, n_legal=3)
    assert s["signature_schema_version"] == SIGNATURE_SCHEMA_VERSION
    assert len(s["k_eff_trajectory"]) == 3
    assert s["k_eff_trajectory"] == pytest.approx(
        k_eff_trajectory(trace["trace_policies"])
    )
    assert s["n_budget_steps"] == 3
    assert s["first_revision_step"] is None  # argmax stays 0
    assert s["flip_flop_rate"] == pytest.approx(0.0)
    assert 0.0 < s["final_sparsity"] <= 1.0
    # narrowing trace → negative K_eff slope
    assert s["concentration_vs_budget"]["k_eff_slope_per_log_budget"] < 0.0


def test_trace_signature_summary_handles_empty_trace():
    s = trace_signature_summary({"trace_budgets": [], "trace_policies": []})
    assert s["k_eff_trajectory"] == []
    assert s["first_revision_step"] is None
    assert s["flip_flop_rate"] == 0.0
    assert s["final_sparsity"] == 0.0
    assert s["n_budget_steps"] == 0
