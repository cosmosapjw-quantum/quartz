import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quartz.phase15_ablation import (
    PHASE15_CANDIDATE_SYSTEMS,
    PHASE15_CI_SMOKE_SYSTEMS,
    PHASE15_FULL_SYSTEMS,
    PHASE15_SMALL_ABLATION_SYSTEMS,
    Phase15System,
    apply_system_readout,
    build_root_challenger_set,
    classify_position_buckets,
    load_systems_config,
    make_default_systems,
    normalize_policy,
    phase15_systems_csv,
    policy_argmax,
    resolve_phase15_systems_arg,
    validate_phase15_systems,
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_phase15_runner():
    root = Path(__file__).resolve().parents[1]
    return load_module("phase15_runner_script", root / "scripts" / "phase15_ablation_study.py")


def load_phase15_online_runner():
    root = Path(__file__).resolve().parents[1]
    return load_module("phase15_online_runner_script", root / "scripts" / "phase15_online_ablation.py")


def load_phase15_benchmark_runner():
    root = Path(__file__).resolve().parents[1]
    return load_module("phase15_benchmark_script", root / "scripts" / "phase15_benchmark.py")


def load_phase15_benchmark_ci_smoke_runner():
    root = Path(__file__).resolve().parents[1]
    return load_module("phase15_benchmark_ci_smoke_script", root / "scripts" / "phase15_benchmark_ci_smoke.py")


def load_phase15_toy_ablation_runner():
    root = Path(__file__).resolve().parents[1]
    return load_module("phase15_toy_ablation_script", root / "scripts" / "phase15_toy_ablation.py")


def load_phase15_analysis_runner():
    root = Path(__file__).resolve().parents[1]
    return load_module("phase15_analysis_script", root / "scripts" / "phase15_analyze_results.py")


def test_make_default_systems_exposes_clean_split_groups():
    systems = make_default_systems({})
    assert [system.id for system in systems] == list(PHASE15_FULL_SYSTEMS)
    assert [system.id for system in systems] == [
        "A0",
        "A1",
        "A2",
        "A3",
        "A4",
        "B0",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B9",
        "B10",
        "B11",
        "B12",
        "C0",
        "C1",
        "C2",
    ]
    assert systems[5].report_alias == "A4"
    assert systems[5].execution_mode == "posthoc"
    assert systems[8].params["comparison_role"] == "budget_scheduler"
    assert systems[10].report_alias == "A4"
    assert systems[10].params["comparison_role"] == "a4_equivalence_anchor"
    assert systems[14].params["comparison_role"] == "argmax_tie_guarded_readout"
    assert systems[15].params["comparison_role"] == "snapshot_safe_stabilized_readout"
    assert systems[16].params["comparison_role"] == "adaptive_snapshot_safe_stabilized_readout"
    assert systems[17].params["comparison_role"] == "entropy_expansion_stabilized_readout"


def test_phase15_system_presets_cover_current_candidates():
    assert resolve_phase15_systems_arg("ci_smoke") == PHASE15_CI_SMOKE_SYSTEMS
    assert resolve_phase15_systems_arg("small") == PHASE15_SMALL_ABLATION_SYSTEMS
    assert resolve_phase15_systems_arg("toy") == PHASE15_SMALL_ABLATION_SYSTEMS
    assert resolve_phase15_systems_arg("candidates") == PHASE15_CANDIDATE_SYSTEMS
    assert resolve_phase15_systems_arg("full") == PHASE15_FULL_SYSTEMS
    assert resolve_phase15_systems_arg("B1,B12") == ("B1", "B12")
    assert phase15_systems_csv("small") == "A4,B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12"


def test_group_a_and_b_defaults_do_not_use_refresh_legacy_substrate():
    systems = {system.id: system for system in make_default_systems({})}
    for system_id in (
        "A0",
        "A1",
        "A2",
        "A3",
        "A4",
        "B0",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B9",
        "B10",
        "B11",
        "B12",
    ):
        system = systems[system_id]
        assert system.search_overrides.get("root_only_shaping") is True
        assert system.search_overrides.get("penalty_mode") not in {"GatedRefreshLegacy", "PFlipMixture", "SelfAdaptive"}


def test_legacy_anchor_defaults_are_preserved_only_in_group_c():
    systems = {system.id: system for system in make_default_systems({})}
    assert systems["C0"].search_overrides["penalty_mode"] == "GatedRefreshLegacy"
    assert systems["C1"].search_overrides["penalty_mode"] == "PFlipMixture"
    assert systems["C2"].search_overrides["penalty_mode"] == "SelfAdaptive"


def test_dual_channel_commit_prefers_stable_search_posterior():
    system = Phase15System(
        id="B1",
        label="dual",
        group="B",
        substrate="S1",
        controller="QuartzVL",
        refresh_operator="dual_channel_commit",
        params={"commit_threshold": 0.4, "gate_scale": 1.0, "divergence_scale": 0.2, "entropy_scale": 0.2},
    )
    prior = normalize_policy(np.array([0.75, 0.20, 0.05], dtype=np.float32))
    trace = [
        normalize_policy(np.array([0.20, 0.70, 0.10], dtype=np.float32)),
        normalize_policy(np.array([0.18, 0.72, 0.10], dtype=np.float32)),
        normalize_policy(np.array([0.16, 0.74, 0.10], dtype=np.float32)),
    ]
    effective, meta = apply_system_readout(system, prior, trace, [8, 16, 32], 32)
    assert policy_argmax(effective) == 1
    assert meta["commit_confidence"] >= meta["commit_threshold"]
    assert meta["commit_applied"] == 1


def test_root_challenger_refresh_returns_restricted_candidate_set():
    system = Phase15System(
        id="B2",
        label="challenger",
        group="B",
        substrate="S1",
        controller="QuartzVL",
        refresh_operator="root_challenger",
        params={"challenger_k": 2, "candidate_score_mix": 0.5, "snapshot_alpha": 0.5},
    )
    prior = normalize_policy(np.array([0.60, 0.25, 0.10, 0.05], dtype=np.float32))
    trace = [
        normalize_policy(np.array([0.45, 0.25, 0.25, 0.05], dtype=np.float32)),
        normalize_policy(np.array([0.15, 0.20, 0.60, 0.05], dtype=np.float32)),
    ]
    effective, meta = apply_system_readout(system, prior, trace, [8, 16], 16)
    assert 2 <= len(meta["root_candidate_set"]) <= 4
    assert policy_argmax(effective) in meta["root_candidate_set"]


def test_root_challenger_set_expands_near_ties_past_k():
    prior = normalize_policy(np.array([0.40, 0.30, 0.20, 0.10], dtype=np.float32))
    posterior = normalize_policy(np.array([0.10, 0.2999, 0.2998, 0.3001], dtype=np.float32))
    candidates, _scores = build_root_challenger_set(
        prior,
        posterior,
        {"challenger_k": 2, "candidate_score_mix": 0.0, "candidate_tie_eps": 5e-4, "challenger_max_extra": 2},
    )
    assert candidates[:2] == [3, 1]
    assert 2 in candidates


def test_budget_routing_can_burst_to_next_budget_on_instability():
    system = Phase15System(
        id="B3",
        label="routing",
        group="B",
        substrate="S1",
        controller="QuartzVL",
        refresh_operator="budget_routing",
        params={"persistence_floor": 0.95, "margin_stability_floor": 0.95},
    )
    prior = normalize_policy(np.array([0.55, 0.25, 0.20], dtype=np.float32))
    trace = [
        normalize_policy(np.array([0.40, 0.35, 0.25], dtype=np.float32)),
        normalize_policy(np.array([0.30, 0.45, 0.25], dtype=np.float32)),
        normalize_policy(np.array([0.10, 0.75, 0.15], dtype=np.float32)),
    ]
    effective, meta = apply_system_readout(system, prior, trace, [8, 16, 32], 16)
    assert meta["budget_burst_triggered"] == 1
    assert meta["extra_budget_used"] == 16
    assert policy_argmax(effective) == 1


def test_root_dual_posterior_applies_sparse_root_only_revision():
    system = Phase15System(
        id="B4",
        label="root dual",
        group="B",
        substrate="S1",
        controller="QuartzVL",
        refresh_operator="root_dual_posterior",
        params={
            "revision_threshold": 0.25,
            "max_posterior_weight": 0.9,
            "challenger_k": 2,
            "candidate_score_mix": 0.2,
            "prior_anchor_k": 1,
            "posterior_anchor_k": 2,
        },
    )
    prior = normalize_policy(np.array([0.70, 0.20, 0.10, 0.0], dtype=np.float32))
    trace = [
        normalize_policy(np.array([0.15, 0.20, 0.60, 0.05], dtype=np.float32)),
        normalize_policy(np.array([0.10, 0.20, 0.65, 0.05], dtype=np.float32)),
    ]

    effective, meta = apply_system_readout(system, prior, trace, [8, 16], 16)

    assert policy_argmax(effective) == 2
    assert meta["belief_revision_operator"] == "root_dual_posterior"
    assert meta["belief_revision_scope"] == "root_only"
    assert meta["prior_base_mutated"] == 0
    assert meta["revision_occurred"] == 1
    assert set(meta["root_candidate_set"]).issubset({0, 1, 2, 3})
    np.testing.assert_allclose(prior, [0.70, 0.20, 0.10, 0.0], rtol=1e-6, atol=1e-7)


def test_root_dual_posterior_gate_preserves_prior_on_weak_evidence():
    system = Phase15System(
        id="B4",
        label="root dual",
        group="B",
        substrate="S1",
        controller="QuartzVL",
        refresh_operator="root_dual_posterior",
        params={"revision_threshold": 0.99, "challenger_k": 2},
    )
    prior = normalize_policy(np.array([0.80, 0.10, 0.10], dtype=np.float32))
    trace = [normalize_policy(np.array([0.60, 0.20, 0.20], dtype=np.float32))]

    effective, meta = apply_system_readout(system, prior, trace, [8], 8)

    np.testing.assert_allclose(effective, prior, rtol=1e-6, atol=1e-7)
    assert meta["revision_occurred"] == 0
    assert meta["posterior_weight"] == 0.0


def test_root_posterior_snapshot_reports_pure_root_readout_contract():
    system = Phase15System("B5", "snapshot", "B", "S1", "QuartzVL", "root_posterior_snapshot")
    prior = normalize_policy(np.array([0.80, 0.10, 0.10], dtype=np.float32))
    posterior = normalize_policy(np.array([0.10, 0.75, 0.15], dtype=np.float32))

    effective, meta = apply_system_readout(system, prior, [posterior], [8], 8)

    np.testing.assert_allclose(effective, posterior, rtol=1e-6, atol=1e-7)
    assert meta["belief_revision_operator"] == "root_posterior_snapshot"
    assert meta["belief_revision_scope"] == "root_only"
    assert meta["prior_base_mutated"] == 0
    assert meta["posterior_weight"] == 1.0


def test_confidence_bound_posterior_records_volatility_bound():
    system = Phase15System(
        "B6",
        "bound",
        "B",
        "S1",
        "QuartzVL",
        "confidence_bound_posterior",
        params={"confidence_threshold": 0.0, "posterior_weight": 0.8, "challenger_k": 2},
    )
    prior = normalize_policy(np.array([0.60, 0.30, 0.10], dtype=np.float32))
    trace = [
        normalize_policy(np.array([0.30, 0.60, 0.10], dtype=np.float32)),
        normalize_policy(np.array([0.20, 0.70, 0.10], dtype=np.float32)),
        normalize_policy(np.array([0.25, 0.65, 0.10], dtype=np.float32)),
    ]

    effective, meta = apply_system_readout(system, prior, trace, [4, 8, 16], 16)

    assert policy_argmax(effective) == 1
    assert meta["belief_revision_operator"] == "confidence_bound_posterior"
    assert meta["mean_trace_volatility"] >= 0.0
    assert "confidence_bound_scores" in meta


def test_robust_valley_posterior_prefers_temporally_stable_candidate():
    system = Phase15System(
        "B7",
        "robust",
        "B",
        "S1",
        "QuartzVL",
        "robust_valley_posterior",
        params={"prior_weight": 0.0, "challenger_k": 2, "candidate_score_mix": 0.0},
    )
    prior = normalize_policy(np.array([0.45, 0.45, 0.10], dtype=np.float32))
    trace = [
        normalize_policy(np.array([0.20, 0.65, 0.15], dtype=np.float32)),
        normalize_policy(np.array([0.55, 0.35, 0.10], dtype=np.float32)),
        normalize_policy(np.array([0.25, 0.60, 0.15], dtype=np.float32)),
    ]

    effective, meta = apply_system_readout(system, prior, trace, [4, 8, 16], 16)

    assert policy_argmax(effective) == 1
    assert meta["belief_revision_operator"] == "robust_valley_posterior"
    assert "robust_valley_scores" in meta


def test_entropy_annealed_posterior_smooths_unstable_trace():
    system = Phase15System(
        "B8",
        "annealed",
        "B",
        "S1",
        "QuartzVL",
        "entropy_annealed_posterior",
        params={"posterior_weight": 1.0, "temperature_min": 0.5, "temperature_max": 2.0},
    )
    prior = normalize_policy(np.array([0.60, 0.30, 0.10], dtype=np.float32))
    trace = [
        normalize_policy(np.array([0.85, 0.10, 0.05], dtype=np.float32)),
        normalize_policy(np.array([0.10, 0.85, 0.05], dtype=np.float32)),
    ]

    effective, meta = apply_system_readout(system, prior, trace, [4, 8], 8)

    assert meta["belief_revision_operator"] == "entropy_annealed_posterior"
    assert meta["annealing_temperature"] > 1.0
    assert meta["effective_entropy"] > meta["posterior_entropy"]
    assert effective.shape == prior.shape


def test_guarded_root_dual_posterior_falls_back_to_snapshot_on_thin_margin():
    system = Phase15System(
        "B9",
        "guarded",
        "B",
        "S1",
        "QuartzVL",
        "guarded_root_dual_posterior",
        params={
            "revision_threshold": 0.0,
            "max_posterior_weight": 0.9,
            "challenger_k": 2,
            "candidate_score_mix": 0.0,
            "guard_persistence_floor": 0.0,
            "guard_margin_floor": 0.05,
            "guard_margin_stability_floor": 0.0,
        },
    )
    prior = normalize_policy(np.array([0.60, 0.25, 0.15], dtype=np.float32))
    trace = [
        normalize_policy(np.array([0.55, 0.30, 0.15], dtype=np.float32)),
        normalize_policy(np.array([0.39, 0.40, 0.21], dtype=np.float32)),
    ]

    effective, meta = apply_system_readout(system, prior, trace, [8, 16], 16)

    np.testing.assert_allclose(effective, trace[-1], rtol=1e-6, atol=1e-7)
    assert meta["belief_revision_operator"] == "guarded_root_dual_posterior"
    assert meta["guard_vetoed"] == 1
    assert meta["fallback_operator"] == "root_posterior_snapshot"
    assert "thin_margin" in meta["guard_reason"]


def test_guarded_root_dual_posterior_applies_when_argmax_is_stable_and_clear():
    system = Phase15System(
        "B9",
        "guarded",
        "B",
        "S1",
        "QuartzVL",
        "guarded_root_dual_posterior",
        params={
            "revision_threshold": 0.0,
            "max_posterior_weight": 0.9,
            "challenger_k": 2,
            "candidate_score_mix": 0.2,
            "prior_anchor_k": 1,
            "posterior_anchor_k": 2,
            "guard_persistence_floor": 0.5,
            "guard_margin_floor": 0.10,
            "guard_margin_stability_floor": 0.0,
        },
    )
    prior = normalize_policy(np.array([0.70, 0.20, 0.10, 0.0], dtype=np.float32))
    trace = [
        normalize_policy(np.array([0.15, 0.20, 0.60, 0.05], dtype=np.float32)),
        normalize_policy(np.array([0.10, 0.20, 0.65, 0.05], dtype=np.float32)),
    ]

    effective, meta = apply_system_readout(system, prior, trace, [8, 16], 16)

    assert policy_argmax(effective) == 2
    assert meta["belief_revision_operator"] == "guarded_root_dual_posterior"
    assert meta["guard_vetoed"] == 0
    assert meta["guard_reason"] == "passed"
    assert meta["posterior_weight"] > 0.0
    assert meta["kl_posterior_to_effective"] > 0.0


def test_snapshot_trace_stabilized_posterior_preserves_snapshot_argmax_and_topk():
    system = Phase15System(
        "B10",
        "stabilized",
        "B",
        "S1",
        "QuartzVL",
        "snapshot_trace_stabilized_posterior",
        params={
            "stabilization_weight": 0.25,
            "snapshot_anchor_k": 3,
            "stability_floor": 0.0,
            "margin_stability_floor": 0.0,
        },
    )
    prior = normalize_policy(np.array([0.60, 0.20, 0.15, 0.05], dtype=np.float32))
    trace = [
        normalize_policy(np.array([0.40, 0.35, 0.20, 0.05], dtype=np.float32)),
        normalize_policy(np.array([0.42, 0.33, 0.20, 0.05], dtype=np.float32)),
        normalize_policy(np.array([0.44, 0.32, 0.19, 0.05], dtype=np.float32)),
    ]

    effective, meta = apply_system_readout(system, prior, trace, [4, 8, 16], 16)

    assert policy_argmax(effective) == policy_argmax(trace[-1])
    assert set(np.argsort(-effective, kind="stable")[:3]) == set(np.argsort(-trace[-1], kind="stable")[:3])
    assert meta["belief_revision_operator"] == "snapshot_trace_stabilized_posterior"
    assert meta["stabilizer_applied"] == 1
    assert meta["stabilization_reason"] == "passed"
    assert meta["snapshot_anchor_preserved"] == 1
    assert meta["stabilization_weight"] == pytest.approx(0.25)


def test_snapshot_trace_stabilized_posterior_falls_back_when_topk_anchor_breaks():
    system = Phase15System(
        "B10",
        "stabilized",
        "B",
        "S1",
        "QuartzVL",
        "snapshot_trace_stabilized_posterior",
        params={
            "stabilization_weight": 0.80,
            "snapshot_anchor_k": 3,
            "stability_floor": 0.0,
            "margin_stability_floor": 0.0,
        },
    )
    prior = normalize_policy(np.array([0.60, 0.20, 0.15, 0.05], dtype=np.float32))
    trace = [
        normalize_policy(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)),
        normalize_policy(np.array([0.42, 0.33, 0.20, 0.05], dtype=np.float32)),
    ]

    effective, meta = apply_system_readout(system, prior, trace, [4, 8], 8)

    np.testing.assert_allclose(effective, trace[-1], rtol=1e-6, atol=1e-7)
    assert meta["stabilizer_applied"] == 0
    assert "topk_anchor_break" in meta["stabilization_reason"]
    assert meta["fallback_operator"] == "root_posterior_snapshot"


def test_adaptive_snapshot_trace_stabilized_posterior_selects_largest_safe_weight():
    system = Phase15System(
        "B11",
        "adaptive stabilized",
        "B",
        "S1",
        "QuartzVL",
        "adaptive_snapshot_trace_stabilized_posterior",
        params={
            "max_stabilization_weight": 0.80,
            "candidate_weight_step": 0.20,
            "snapshot_anchor_k": 3,
            "stability_floor": 0.0,
            "margin_stability_floor": 0.0,
        },
    )
    prior = normalize_policy(np.array([0.60, 0.20, 0.15, 0.05], dtype=np.float32))
    trace = [
        normalize_policy(np.array([0.05, 0.58, 0.27, 0.10], dtype=np.float32)),
        normalize_policy(np.array([0.44, 0.32, 0.19, 0.05], dtype=np.float32)),
    ]

    effective, meta = apply_system_readout(system, prior, trace, [8, 16], 16)

    assert policy_argmax(effective) == policy_argmax(trace[-1])
    assert set(np.argsort(-effective, kind="stable")[:3]) == set(np.argsort(-trace[-1], kind="stable")[:3])
    assert meta["belief_revision_operator"] == "adaptive_snapshot_trace_stabilized_posterior"
    assert meta["adaptive_stabilizer"] == 1
    assert meta["stabilizer_applied"] == 1
    assert meta["stabilization_reason"] == "passed"
    assert meta["selected_stabilization_weight"] == pytest.approx(0.40)
    assert meta["stabilization_weight"] == pytest.approx(0.40)
    assert meta["candidate_weight_count"] == 4


def test_adaptive_snapshot_trace_stabilized_posterior_falls_back_when_no_weight_is_safe():
    system = Phase15System(
        "B11",
        "adaptive stabilized",
        "B",
        "S1",
        "QuartzVL",
        "adaptive_snapshot_trace_stabilized_posterior",
        params={
            "max_stabilization_weight": 0.80,
            "candidate_weight_step": 0.40,
            "snapshot_anchor_k": 3,
            "stability_floor": 0.0,
            "margin_stability_floor": 0.0,
        },
    )
    prior = normalize_policy(np.array([0.60, 0.20, 0.15, 0.05], dtype=np.float32))
    trace = [
        normalize_policy(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)),
        normalize_policy(np.array([0.42, 0.33, 0.20, 0.05], dtype=np.float32)),
    ]

    effective, meta = apply_system_readout(system, prior, trace, [8, 16], 16)

    np.testing.assert_allclose(effective, trace[-1], rtol=1e-6, atol=1e-7)
    assert meta["adaptive_stabilizer"] == 1
    assert meta["stabilizer_applied"] == 0
    assert meta["selected_stabilization_weight"] == pytest.approx(0.0)
    assert meta["fallback_operator"] == "root_posterior_snapshot"
    assert "no_safe_weight" in meta["stabilization_reason"]


def test_entropy_expansion_stabilized_posterior_applies_only_on_positive_entropy_slope():
    system = Phase15System(
        "B12",
        "entropy expansion stabilized",
        "B",
        "S1",
        "QuartzVL",
        "entropy_expansion_stabilized_posterior",
        params={
            "max_stabilization_weight": 0.45,
            "candidate_weight_step": 0.05,
            "min_stabilization_weight": 0.05,
            "entropy_slope_floor": 0.25,
            "snapshot_anchor_k": 3,
            "stability_floor": 0.0,
            "margin_stability_floor": 0.0,
        },
    )
    prior = normalize_policy(np.array([0.60, 0.20, 0.15, 0.05], dtype=np.float32))
    trace = [
        normalize_policy(np.array([0.90, 0.05, 0.03, 0.02], dtype=np.float32)),
        normalize_policy(np.array([0.44, 0.32, 0.19, 0.05], dtype=np.float32)),
    ]

    effective, meta = apply_system_readout(system, prior, trace, [8, 16], 16)

    assert policy_argmax(effective) == policy_argmax(trace[-1])
    assert meta["belief_revision_operator"] == "entropy_expansion_stabilized_posterior"
    assert meta["adaptive_stabilizer"] == 1
    assert meta["stabilizer_applied"] == 1
    assert meta["stabilization_reason"] == "passed"
    assert meta["entropy_expansion_gate_passed"] == 1
    assert meta["posterior_entropy_slope"] >= meta["entropy_slope_floor"]
    assert meta["selected_stabilization_weight"] == pytest.approx(0.45)


def test_entropy_expansion_stabilized_posterior_falls_back_without_entropy_expansion():
    system = Phase15System(
        "B12",
        "entropy expansion stabilized",
        "B",
        "S1",
        "QuartzVL",
        "entropy_expansion_stabilized_posterior",
        params={
            "max_stabilization_weight": 0.45,
            "candidate_weight_step": 0.05,
            "min_stabilization_weight": 0.05,
            "entropy_slope_floor": 0.25,
            "snapshot_anchor_k": 3,
            "stability_floor": 0.0,
            "margin_stability_floor": 0.0,
        },
    )
    prior = normalize_policy(np.array([0.60, 0.20, 0.15, 0.05], dtype=np.float32))
    trace = [
        normalize_policy(np.array([0.42, 0.33, 0.20, 0.05], dtype=np.float32)),
        normalize_policy(np.array([0.90, 0.05, 0.03, 0.02], dtype=np.float32)),
    ]

    effective, meta = apply_system_readout(system, prior, trace, [8, 16], 16)

    np.testing.assert_allclose(effective, trace[-1], rtol=1e-6, atol=1e-7)
    assert meta["adaptive_stabilizer"] == 1
    assert meta["stabilizer_applied"] == 0
    assert meta["entropy_expansion_gate_passed"] == 0
    assert meta["selected_stabilization_weight"] == pytest.approx(0.0)
    assert meta["fallback_operator"] == "root_posterior_snapshot"
    assert "entropy_slope_below_floor" in meta["stabilization_reason"]


def test_build_row_exports_root_dual_revision_metadata():
    runner = load_phase15_runner()
    checkpoint = runner.CheckpointRef(id="C01", path="/tmp/model.pt")
    position = {"id": "P0001", "bucket_tags": ["wrong_top1"]}
    system = Phase15System("B4", "root dual", "B", "S1", "QuartzVL", "root_dual_posterior")
    prior = normalize_policy(np.array([0.7, 0.2, 0.1], dtype=np.float32))
    posterior = normalize_policy(np.array([0.1, 0.8, 0.1], dtype=np.float32))
    trace_meta = {
        "argmax_path": [0, 1],
        "trace_budgets": [8, 16],
        "belief_revision_operator": "root_dual_posterior",
        "belief_revision_scope": "root_only",
        "prior_base_mutated": 0,
        "revision_occurred": 1,
        "revision_confidence": 0.8,
        "revision_threshold": 0.5,
        "posterior_weight": 0.8,
        "kl_prior_to_effective": 0.4,
    }

    row = runner.build_row(
        checkpoint,
        position,
        system,
        16,
        prior,
        posterior,
        posterior,
        posterior,
        trace_meta,
        trace_reused=True,
    )

    assert row["belief_revision_operator"] == "root_dual_posterior"
    assert row["belief_revision_scope"] == "root_only"
    assert row["prior_base_mutated"] == 0
    assert row["revision_occurred"] == 1
    assert row["revision_confidence"] == pytest.approx(0.8)
    assert row["kl_prior_to_effective"] == pytest.approx(0.4)


def test_build_row_exports_candidate_comparison_role_metadata():
    runner = load_phase15_runner()
    checkpoint = runner.CheckpointRef(id="C01", path="/tmp/model.pt")
    position = {"id": "P0001", "bucket_tags": ["wrong_top1"]}
    system = Phase15System(
        "B3",
        "budget router",
        "B",
        "S1",
        "QuartzVL",
        "budget_routing",
        params={"comparison_role": "budget_scheduler", "budget_confounded": True},
    )
    prior = normalize_policy(np.array([0.7, 0.2, 0.1], dtype=np.float32))
    final = normalize_policy(np.array([0.1, 0.8, 0.1], dtype=np.float32))
    row = runner.build_row(
        checkpoint,
        position,
        system,
        16,
        prior,
        final,
        final,
        final,
        {"trace_budgets": [8, 16], "argmax_path": [0, 1]},
        trace_reused=False,
    )
    assert row["comparison_role"] == "budget_scheduler"
    assert row["budget_confounded"] == 1


def test_build_row_exports_guarded_readout_metadata():
    runner = load_phase15_runner()
    checkpoint = runner.CheckpointRef(id="C01", path="/tmp/model.pt")
    position = {"id": "P0001", "bucket_tags": ["ambiguous"]}
    system = Phase15System(
        "B9",
        "guarded",
        "B",
        "S1",
        "QuartzVL",
        "guarded_root_dual_posterior",
        params={"comparison_role": "argmax_tie_guarded_readout"},
    )
    prior = normalize_policy(np.array([0.7, 0.2, 0.1], dtype=np.float32))
    final = normalize_policy(np.array([0.45, 0.46, 0.09], dtype=np.float32))
    row = runner.build_row(
        checkpoint,
        position,
        system,
        16,
        prior,
        final,
        final,
        final,
        {
            "trace_budgets": [8, 16],
            "argmax_path": [0, 1],
            "belief_revision_operator": "guarded_root_dual_posterior",
            "guard_operator": "argmax_tie_guard",
            "guard_vetoed": 1,
            "guard_reason": "thin_margin",
            "fallback_operator": "root_posterior_snapshot",
            "guard_margin": 0.01,
            "guard_margin_floor": 0.05,
        },
        trace_reused=False,
    )
    assert row["comparison_role"] == "argmax_tie_guarded_readout"
    assert row["guard_operator"] == "argmax_tie_guard"
    assert row["guard_vetoed"] == 1
    assert row["guard_reason"] == "thin_margin"
    assert row["fallback_operator"] == "root_posterior_snapshot"
    assert row["guard_margin"] == pytest.approx(0.01)
    assert row["guard_margin_floor"] == pytest.approx(0.05)


def test_build_row_exports_snapshot_stabilizer_metadata():
    runner = load_phase15_runner()
    checkpoint = runner.CheckpointRef(id="C01", path="/tmp/model.pt")
    position = {"id": "P0001", "bucket_tags": ["generic"]}
    system = Phase15System(
        "B10",
        "stabilized",
        "B",
        "S1",
        "QuartzVL",
        "snapshot_trace_stabilized_posterior",
        params={"comparison_role": "snapshot_safe_stabilized_readout"},
    )
    prior = normalize_policy(np.array([0.7, 0.2, 0.1], dtype=np.float32))
    final = normalize_policy(np.array([0.6, 0.3, 0.1], dtype=np.float32))

    row = runner.build_row(
        checkpoint,
        position,
        system,
        16,
        prior,
        final,
        final,
        final,
        {
            "trace_budgets": [8, 16],
            "argmax_path": [0, 0],
            "belief_revision_operator": "snapshot_trace_stabilized_posterior",
            "adaptive_stabilizer": 1,
            "entropy_expansion_gate_passed": 1,
            "stabilizer_applied": 1,
            "stabilization_reason": "passed",
            "stabilization_weight": 0.2,
            "selected_stabilization_weight": 0.2,
            "max_stabilization_weight": 0.45,
            "candidate_weight_step": 0.05,
            "entropy_slope_floor": 0.25,
            "candidate_weight_count": 9,
            "snapshot_anchor_k": 3,
            "snapshot_anchor_preserved": 1,
            "snapshot_argmax": 0,
            "stabilized_argmax": 0,
        },
        trace_reused=False,
    )

    assert row["comparison_role"] == "snapshot_safe_stabilized_readout"
    assert row["adaptive_stabilizer"] == 1
    assert row["entropy_expansion_gate_passed"] == 1
    assert row["stabilizer_applied"] == 1
    assert row["stabilization_reason"] == "passed"
    assert row["stabilization_weight"] == pytest.approx(0.2)
    assert row["selected_stabilization_weight"] == pytest.approx(0.2)
    assert row["max_stabilization_weight"] == pytest.approx(0.45)
    assert row["candidate_weight_step"] == pytest.approx(0.05)
    assert row["entropy_slope_floor"] == pytest.approx(0.25)
    assert row["candidate_weight_count"] == 9
    assert row["snapshot_anchor_k"] == 3
    assert row["snapshot_anchor_preserved"] == 1
    assert row["snapshot_argmax"] == 0
    assert row["stabilized_argmax"] == 0


def test_build_summary_payload_aggregates_guarded_readout_rates():
    runner = load_phase15_runner()
    rows = [
        {"group": "B", "system": "B9", "budget": 16, "position_bucket": "ambiguous", "guard_vetoed": 1},
        {"group": "B", "system": "B9", "budget": 16, "position_bucket": "wrong_top1", "guard_vetoed": 0},
    ]

    summary = runner.build_summary_payload(rows)

    assert summary == [
        {
            "group": "B",
            "system": "B9",
            "budget": 16,
            "rows": 2,
            "guard_vetoed": 0.5,
            "bucket_counts": {"ambiguous": 1, "wrong_top1": 1},
        }
    ]


def test_build_summary_payload_aggregates_snapshot_stabilizer_rates():
    runner = load_phase15_runner()
    rows = [
        {"group": "B", "system": "B10", "budget": 16, "position_bucket": "generic", "stabilizer_applied": 1},
        {"group": "B", "system": "B10", "budget": 16, "position_bucket": "ambiguous", "stabilizer_applied": 0},
    ]

    summary = runner.build_summary_payload(rows)

    assert summary == [
        {
            "group": "B",
            "system": "B10",
            "budget": 16,
            "rows": 2,
            "stabilizer_applied": 0.5,
            "bucket_counts": {"ambiguous": 1, "generic": 1},
        }
    ]


def test_classify_position_buckets_marks_wrong_and_late_evidence_cases():
    prior = normalize_policy(np.array([0.65, 0.20, 0.15], dtype=np.float32))
    low = normalize_policy(np.array([0.60, 0.25, 0.15], dtype=np.float32))
    oracle = normalize_policy(np.array([0.10, 0.75, 0.15], dtype=np.float32))
    buckets = classify_position_buckets(prior, low, oracle)
    assert "wrong_top1" in buckets
    assert "wrong_confident" in buckets
    assert "late_evidence" in buckets


def test_load_systems_config_accepts_explicit_json(tmp_path: Path):
    path = tmp_path / "systems.json"
    path.write_text(
        json.dumps(
            {
                "systems": [
                    {
                        "id": "A0",
                        "label": "baseline",
                        "group": "A",
                        "substrate": "S0",
                        "controller": "none",
                        "refresh_operator": "none",
                        "search_overrides": {"search_profile": "baseline", "root_only_shaping": True},
                        "params": {},
                        "execution_mode": "posthoc",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    systems = load_systems_config(str(path), {})
    assert len(systems) == 1
    assert systems[0].id == "A0"


def test_validate_phase15_systems_rejects_legacy_mode_in_group_b():
    system = Phase15System(
        id="B9",
        label="bad",
        group="B",
        substrate="S1",
        controller="QuartzVL",
        refresh_operator="none",
        search_overrides={"root_only_shaping": True, "penalty_mode": "GatedRefreshLegacy"},
    )
    try:
        validate_phase15_systems([system])
    except ValueError as exc:
        assert "legacy penalty mode" in str(exc)
    else:
        raise AssertionError("expected validator to reject legacy penalty in Group B")


def test_build_row_tracks_reference_and_oracle_separately():
    runner = load_phase15_runner()
    checkpoint = runner.CheckpointRef(id="C01", path="/tmp/model.pt")
    position = {"id": "P0001", "bucket_tags": ["easy_good_prior"]}
    system = Phase15System("A0", "baseline", "A", "S0", "none", "none", execution_mode="posthoc")
    prior = normalize_policy(np.array([0.8, 0.2], dtype=np.float32))
    final = normalize_policy(np.array([0.1, 0.9], dtype=np.float32))
    reference = normalize_policy(np.array([0.2, 0.8], dtype=np.float32))
    oracle = normalize_policy(np.array([0.9, 0.1], dtype=np.float32))
    row = runner.build_row(
        checkpoint,
        position,
        system,
        16,
        prior,
        final,
        reference,
        oracle,
        {
            "trace_budgets": [8, 16],
            "argmax_path": [0, 1],
            "trace_acquire_ms": 11.0,
            "readout_ms": 0.3,
            "effective_runtime_ms": 11.3,
        },
        trace_reused=True,
    )
    assert row["accuracy_to_reference"] == 1
    assert row["accuracy_to_oracle"] == 0
    assert row["wrong_prior_correction_reference"] == 1
    assert row["wrong_prior_correction_oracle"] == 0
    assert row["trace_acquire_ms"] == 11.0
    assert row["readout_ms"] == 0.3
    assert row["trace_reused"] == 1


def test_build_row_keeps_continuation_fallback_reason():
    runner = load_phase15_runner()
    checkpoint = runner.CheckpointRef(id="C01", path="/tmp/model.pt")
    position = {"id": "P0001", "bucket_tags": []}
    system = Phase15System("B1", "dual", "B", "S1", "QuartzVL", "dual_channel_commit", execution_mode="online")
    prior = normalize_policy(np.array([0.6, 0.4], dtype=np.float32))
    final = normalize_policy(np.array([0.5, 0.5], dtype=np.float32))
    reference = normalize_policy(np.array([0.5, 0.5], dtype=np.float32))
    oracle = normalize_policy(np.array([0.5, 0.5], dtype=np.float32))
    row = runner.build_row(
        checkpoint,
        position,
        system,
        16,
        prior,
        final,
        reference,
        oracle,
        {
            "trace_budgets": [8, 16],
            "argmax_path": [0, 0],
            "trace_acquire_ms": 10.0,
            "readout_ms": 0.2,
            "continuation_fallback_reason": "RuntimeError: resident session unavailable",
        },
        trace_reused=False,
    )
    assert row["continuation_fallback_reason"] == "RuntimeError: resident session unavailable"


def test_build_semantic_summary_collapses_alias_rows():
    runner = load_phase15_runner()
    rows = [
        {
            "group": "A",
            "system": "A4",
            "budget": 16,
            "execution_mode": "posthoc",
            "accuracy_to_reference": 1.0,
            "accuracy_to_oracle": 1.0,
            "wrong_prior_correction_reference": 0.0,
            "wrong_prior_correction_oracle": 0.0,
            "easy_case_regret_reference": 0.0,
            "easy_case_regret_oracle": 0.0,
            "topk_recall_reference": 1.0,
            "topk_recall_oracle": 1.0,
            "kl_to_reference": 0.0,
            "kl_to_oracle": 0.0,
            "trace_acquire_ms": 12.0,
            "readout_ms": 0.2,
            "effective_runtime_ms": 12.2,
        },
        {
            "group": "B",
            "system": "B0",
            "alias_of": "A4",
            "budget": 16,
            "execution_mode": "posthoc",
            "accuracy_to_reference": 1.0,
            "accuracy_to_oracle": 1.0,
            "wrong_prior_correction_reference": 0.0,
            "wrong_prior_correction_oracle": 0.0,
            "easy_case_regret_reference": 0.0,
            "easy_case_regret_oracle": 0.0,
            "topk_recall_reference": 1.0,
            "topk_recall_oracle": 1.0,
            "kl_to_reference": 0.0,
            "kl_to_oracle": 0.0,
            "trace_acquire_ms": 12.0,
            "readout_ms": 0.1,
            "effective_runtime_ms": 12.1,
        },
    ]
    summary = runner.build_semantic_summary_payload(rows)
    assert len(summary) == 1
    assert summary[0]["source_system"] == "A4"
    assert summary[0]["systems_present"] == ["A4", "B0"]


def test_build_headwind_summary_collapses_alias_rows_and_classifies():
    runner = load_phase15_runner()
    rows = [
        {
            "group": "A",
            "system": "A4",
            "budget": 16,
            "execution_mode": "posthoc",
            "accuracy_to_reference": 0.25,
            "accuracy_to_oracle": 0.25,
            "kl_to_reference": 0.30,
            "kl_to_oracle": 0.35,
            "trace_acquire_ms": 12.0,
            "readout_ms": 4.0,
            "effective_runtime_ms": 16.0,
            "revision_occurred": 1.0,
        },
        {
            "group": "B",
            "system": "B0",
            "alias_of": "A4",
            "budget": 16,
            "execution_mode": "posthoc",
            "accuracy_to_reference": 0.25,
            "accuracy_to_oracle": 0.25,
            "kl_to_reference": 0.30,
            "kl_to_oracle": 0.35,
            "trace_acquire_ms": 10.0,
            "readout_ms": 6.0,
            "effective_runtime_ms": 16.0,
            "revision_occurred": 0.0,
        },
    ]
    summary = runner.build_headwind_summary_payload(rows)
    assert len(summary) == 1
    assert summary[0]["source_system"] == "A4"
    assert summary[0]["systems_present"] == ["A4", "B0"]
    assert summary[0]["readout_ratio_mean"] == 0.3125
    assert summary[0]["speedup_headwind"] == "mixed_readout_cost_and_semantic_drift"


def test_online_trace_lookup_preserves_budget_rows(monkeypatch):
    online_runner = load_phase15_online_runner()

    def fake_build_search_trace(harness, checkpoint, position, system, trace_budgets, cache_dir):
        return (
            [
                np.asarray([0.7, 0.3], dtype=np.float32),
                np.asarray([0.2, 0.8], dtype=np.float32),
            ],
            [5.0, 9.0],
            True,
        )

    monkeypatch.setattr(online_runner.posthoc, "build_search_trace", fake_build_search_trace)
    rows, reused = online_runner.build_online_trace_lookup(
        harness=object(),
        checkpoint=object(),
        position={"id": "P1"},
        system=Phase15System("B1", "dual", "B", "S1", "QuartzVL", "dual_channel_commit"),
        trace_budgets=[8, 16],
        cache_dir=None,
    )
    assert reused is True
    assert rows[8]["latency_ms"] == 5.0
    assert rows[16]["latency_ms"] == 9.0
    assert rows[16]["search_policy"][1] > rows[16]["search_policy"][0]


def test_online_trace_bundle_prefers_single_continuation_trace(monkeypatch):
    online_runner = load_phase15_online_runner()
    calls = []

    def fake_continuation(client, position, system, trace_budgets, target_budget):
        calls.append((tuple(trace_budgets), int(target_budget)))
        return {
            8: {"search_policy": [0.7, 0.3], "latency_ms": 5.0},
            16: {"search_policy": [0.2, 0.8], "latency_ms": 9.0},
            32: {"search_policy": [0.1, 0.9], "latency_ms": 13.0},
        }

    monkeypatch.setattr(online_runner, "run_online_readout_continuation", fake_continuation)
    monkeypatch.setattr(
        online_runner,
        "build_online_trace_lookup",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fallback should not run")),
    )

    class DummyHarness:
        def _get_client(self, system, budget):
            assert budget == 8
            return object()

    rows, reused, mode, reason = online_runner.build_online_trace_bundle(
        DummyHarness(),
        checkpoint=object(),
        position={"id": "P1"},
        system=Phase15System("B1", "dual", "B", "S1", "QuartzVL", "dual_channel_commit"),
        budgets=[16, 8, 32, 16],
        cache_dir=None,
    )
    assert calls == [((8, 16, 32), 32)]
    assert reused is False
    assert mode == "root_continuation"
    assert reason is None
    assert rows[32]["search_policy"][1] == 0.9


def test_online_trace_bundle_falls_back_once_per_system_position(monkeypatch):
    online_runner = load_phase15_online_runner()
    fallback_calls = []

    def fake_continuation(*args, **kwargs):
        raise RuntimeError("resident session unavailable")

    def fake_lookup(harness, checkpoint, position, system, trace_budgets, cache_dir):
        fallback_calls.append(tuple(trace_budgets))
        return (
            {
                8: {"search_policy": [0.6, 0.4], "latency_ms": 5.0},
                16: {"search_policy": [0.4, 0.6], "latency_ms": 8.0},
            },
            True,
        )

    monkeypatch.setattr(online_runner, "run_online_readout_continuation", fake_continuation)
    monkeypatch.setattr(online_runner, "build_online_trace_lookup", fake_lookup)

    class DummyHarness:
        def _get_client(self, system, budget):
            return object()

    rows, reused, mode, reason = online_runner.build_online_trace_bundle(
        DummyHarness(),
        checkpoint=object(),
        position={"id": "P1"},
        system=Phase15System("B3", "routing", "B", "S1", "QuartzVL", "budget_routing"),
        budgets=[8, 16],
        cache_dir=None,
    )
    assert fallback_calls == [(8, 16)]
    assert reused is True
    assert mode == "restart_per_chunk"
    assert "RuntimeError: resident session unavailable" == reason
    assert rows[16]["search_policy"][1] == 0.6


def test_online_continuation_helper_emits_dense_search_policy():
    online_runner = load_phase15_online_runner()

    class FakeClient:
        cfg = {"actions": 4, "penalty_mode": "None"}

        def open_search_engine_session(self, jobs, penalty_mode="None", iters=None):
            assert jobs
            return {
                "session_id": 7,
                "results": [
                    {
                        "policy": [[1, 0.75], [3, 0.25]],
                        "iterations": 8,
                        "best_move": 1,
                    }
                ],
            }

        def step_search_engine_session(self, session_id, updates=None, iters=None):
            assert session_id == 7
            return {
                "results": [
                    {
                        "policy": [[2, 0.60], [3, 0.40]],
                        "iterations": 16,
                        "best_move": 2,
                    }
                ],
            }

        def close_search_session(self, session_id):
            assert session_id == 7

    rows = online_runner.run_online_readout_continuation(
        FakeClient(),
        {"id": "P1", "board": [0] * 49, "player": 1},
        Phase15System("A4", "baseline", "A", "S1", "QuartzVL", "none"),
        [8, 16],
        16,
    )
    assert np.allclose(rows[8]["search_policy"], [0.0, 0.75, 0.0, 0.25])
    assert np.allclose(rows[16]["search_policy"], [0.0, 0.0, 0.6, 0.4])
    assert rows[8]["latency_ms"] >= 0.0
    assert rows[16]["latency_ms"] >= 0.0


def test_prepare_bucketized_suite_embeds_shared_policy_artifacts():
    runner = load_phase15_runner()

    class DummyHarness:
        def prime_prior_cache(self, positions):
            return None

        def prior_policy(self, position):
            return np.asarray([0.6, 0.4], dtype=np.float32)

        def search_policy(self, position, system, budget):
            if budget == 8:
                return {"search_policy": [0.55, 0.45]}
            if budget == 64:
                return {"search_policy": [0.20, 0.80]}
            return {"search_policy": [0.10, 0.90]}

    suite = runner.prepare_bucketized_suite(
        DummyHarness(),
        DummyHarness(),
        [{"id": "P1"}],
        reference_system=Phase15System("A0", "baseline", "A", "S0", "none", "none"),
        oracle_system=Phase15System("ORACLE", "oracle", "A", "S0", "none", "none"),
        low_budget=8,
        oracle_budget=64,
        bucket_thresholds={
            "confident_threshold": 0.55,
            "ambiguous_margin": 0.10,
            "root_conflict_topk": 2,
            "deep_conflict_topk": 2,
        },
    )
    assert np.allclose(suite[0]["prior_policy"], [0.6, 0.4])
    assert np.allclose(suite[0]["low_budget_policy"], [0.55, 0.45])
    assert np.allclose(suite[0]["reference_policy"], [0.2, 0.8])
    assert np.allclose(suite[0]["oracle_policy"], [0.2, 0.8])


def test_suite_policy_artifact_reads_embedded_policy():
    runner = load_phase15_runner()
    policy = runner.suite_policy_artifact({"reference_policy": [0.25, 0.75]}, "reference_policy")
    assert policy is not None
    assert policy.tolist() == [0.25, 0.75]


def test_build_search_trace_bundle_slices_budget_prefixes(monkeypatch):
    runner = load_phase15_runner()
    calls = []

    def fake_build_search_trace(harness, checkpoint, position, system, trace_budgets, cache_dir):
        calls.append(tuple(trace_budgets))
        return (
            [
                np.asarray([0.7, 0.3], dtype=np.float32),
                np.asarray([0.4, 0.6], dtype=np.float32),
                np.asarray([0.1, 0.9], dtype=np.float32),
            ],
            [5.0, 9.0, 17.0],
            True,
        )

    monkeypatch.setattr(runner, "build_search_trace", fake_build_search_trace)
    bundle_budgets, policy_map, latency_map, reused = runner.build_search_trace_bundle(
        harness=object(),
        checkpoint=object(),
        position={"id": "P1"},
        system=Phase15System("B3", "routing", "B", "S1", "QuartzVL", "budget_routing"),
        budgets=[32, 8, 16, 16],
        cache_dir=None,
    )
    assert calls == [(8, 16, 32)]
    assert bundle_budgets == [8, 16, 32]
    assert reused is True
    assert np.allclose(policy_map[16], [0.4, 0.6])
    trace_budgets, trace_policies, trace_latencies = runner.slice_trace_bundle(
        policy_map,
        latency_map,
        target_budget=16,
        base_budgets=[8, 16, 32],
        allow_extra=True,
    )
    assert trace_budgets == [8, 16, 32]
    assert np.allclose(trace_policies[-1], [0.1, 0.9])
    assert trace_latencies == [5.0, 9.0, 17.0]


def test_suite_sidecar_split_and_merge_roundtrip(tmp_path: Path):
    from quartz.phase15_suite import (
        merge_suite_policy_artifacts,
        read_suite_policy_artifacts,
        split_suite_policy_artifacts,
        write_suite_policy_artifacts,
    )

    suite = [
        {
            "id": "P1",
            "bucket_tags": ["generic"],
            "reference_policy": [0.2, 0.8],
            "oracle_policy": [0.1, 0.9],
            "prior_policy": [0.7, 0.3],
            "low_budget_policy": [0.6, 0.4],
        }
    ]
    compact, artifacts = split_suite_policy_artifacts(suite)
    assert "reference_policy" not in compact[0]
    assert compact[0]["policy_artifact_ref"] == "P1"
    path = tmp_path / "suite_artifacts.json"
    write_suite_policy_artifacts(path, artifacts)
    loaded = read_suite_policy_artifacts(path)
    merged = merge_suite_policy_artifacts(compact, loaded)
    assert np.allclose(merged[0]["reference_policy"], [0.2, 0.8])
    assert np.allclose(merged[0]["oracle_policy"], [0.1, 0.9])


def test_suite_sidecar_preserves_distinct_row_ids_without_collision(tmp_path: Path):
    from quartz.phase15_suite import (
        merge_suite_policy_artifacts,
        read_suite_policy_artifacts,
        split_suite_policy_artifacts,
        write_suite_policy_artifacts,
    )

    suite = [
        {"id": "a/b", "reference_policy": [0.9, 0.1]},
        {"id": "a_b", "reference_policy": [0.2, 0.8]},
    ]
    compact, artifacts = split_suite_policy_artifacts(suite)
    path = tmp_path / "suite_artifacts.npz"
    write_suite_policy_artifacts(path, artifacts)
    merged = merge_suite_policy_artifacts(compact, read_suite_policy_artifacts(path))
    assert np.allclose(merged[0]["reference_policy"], [0.9, 0.1])
    assert np.allclose(merged[1]["reference_policy"], [0.2, 0.8])


def test_frozen_checkpoint_harness_position_key_uses_semantic_state_not_only_id():
    runner = load_phase15_runner()
    harness = object.__new__(runner.FrozenCheckpointHarness)
    left = runner.FrozenCheckpointHarness._position_key(
        harness,
        {"id": "P1", "board": [0, 0, 0, 0], "player": 1},
    )
    right = runner.FrozenCheckpointHarness._position_key(
        harness,
        {"id": "P1", "board": [1, 0, 0, 0], "player": 1},
    )
    assert left != right


def test_build_search_trace_cache_key_distinguishes_positions_with_same_id(tmp_path: Path):
    runner = load_phase15_runner()
    harness_base = object.__new__(runner.FrozenCheckpointHarness)

    class DummyHarness:
        def _position_key(self, position):
            return runner.FrozenCheckpointHarness._position_key(harness_base, position)

        def search_policy(self, position, system, budget):
            board = list(position.get("board", []))
            if board and board[0] == 1:
                return {"search_policy": [0.1, 0.9], "latency_ms": 2.0}
            return {"search_policy": [0.8, 0.2], "latency_ms": 1.0}

    system = Phase15System("A4", "baseline", "A", "S1", "QuartzVL", "none")
    checkpoint = runner.CheckpointRef(id="C1", path="/tmp/model.pt")
    pos_a = {"id": "P1", "board": [0, 0, 0, 0], "player": 1}
    pos_b = {"id": "P1", "board": [1, 0, 0, 0], "player": 1}
    policies_a, _, _ = runner.build_search_trace(DummyHarness(), checkpoint, pos_a, system, [8], tmp_path)
    policies_b, _, _ = runner.build_search_trace(DummyHarness(), checkpoint, pos_b, system, [8], tmp_path)
    assert np.allclose(policies_a[0], [0.8, 0.2])
    assert np.allclose(policies_b[0], [0.1, 0.9])


def test_trace_cache_key_changes_with_code_salt():
    from quartz.phase15_trace import trace_cache_key

    key_a = trace_cache_key("C1", "/tmp/model.pt", "P1", ("sig",), [8, 16], code_salt="salt-a")
    key_b = trace_cache_key("C1", "/tmp/model.pt", "P1", ("sig",), [8, 16], code_salt="salt-b")
    assert key_a != key_b


def test_trace_cache_key_ignores_system_id_uses_search_signature_only():
    """A0-b regression: the cache key must depend on search-relevant
    identity only, not on an opaque system id — two different ids with
    the same search signature must collide onto the same key so they
    provably share one search trace."""
    from quartz.phase15_trace import trace_cache_key

    key_b1 = trace_cache_key("C1", "/tmp/model.pt", "P1", ("same-search-cfg",), [8, 16])
    key_b2 = trace_cache_key("C1", "/tmp/model.pt", "P1", ("same-search-cfg",), [8, 16])
    assert key_b1 == key_b2, "identical search signatures must produce identical cache keys"

    key_different_cfg = trace_cache_key("C1", "/tmp/model.pt", "P1", ("different-search-cfg",), [8, 16])
    assert key_b1 != key_different_cfg, "different search signatures must produce different cache keys"


def test_search_relevant_signature_ignores_readout_but_not_search_overrides():
    """A0-b regression: B1-style and B2-style systems that share
    identical `search_overrides` but differ in `refresh_operator`/
    `params`/`id` must reduce to the SAME search-relevant signature,
    since refresh_operator/params only drive post-hoc readout
    (`apply_system_readout`) and never reach the Rust engine."""
    from quartz.phase15_ablation import search_relevant_signature

    shared_overrides = {"penalty_mode": "GatedRefresh", "root_only_shaping": True}
    system_b1 = Phase15System(
        "B1", "dual_channel_commit", "B", "S1", "QuartzVL", "dual_channel_commit",
        search_overrides=shared_overrides, params={"challenger_k": 4},
    )
    system_b2 = Phase15System(
        "B2", "root_challenger", "B", "S1", "QuartzVL", "root_challenger",
        search_overrides=shared_overrides, params={"snapshot_alpha": 0.5},
    )
    assert search_relevant_signature(system_b1) == search_relevant_signature(system_b2)

    system_different_search = Phase15System(
        "B3", "budget_routing", "B", "S1", "QuartzVL", "budget_routing",
        search_overrides={"penalty_mode": "None", "root_only_shaping": True},
        params={},
    )
    assert search_relevant_signature(system_b1) != search_relevant_signature(system_different_search)


def test_build_search_trace_shares_cache_across_systems_with_identical_search_overrides(tmp_path: Path):
    """A0-b end-to-end regression: build_search_trace must actually
    reuse a cached trace (not just produce an equal hash in isolation)
    across two systems whose search_overrides are identical but whose
    id/refresh_operator/params differ (the real B1 vs B2 shape) —
    proving they share one search rather than paying for and comparing
    two independently-searched traces."""
    runner = load_phase15_runner()
    harness_base = object.__new__(runner.FrozenCheckpointHarness)
    calls: list[str] = []

    class DummyHarness:
        def _position_key(self, position):
            return runner.FrozenCheckpointHarness._position_key(harness_base, position)

        def search_policy(self, position, system, budget):
            calls.append(system.id)
            return {"search_policy": [0.7, 0.3], "latency_ms": 1.0}

    shared_overrides = {"penalty_mode": "GatedRefresh", "root_only_shaping": True}
    system_b1 = Phase15System(
        "B1", "dual_channel_commit", "B", "S1", "QuartzVL", "dual_channel_commit",
        search_overrides=shared_overrides, params={"challenger_k": 4},
    )
    system_b2 = Phase15System(
        "B2", "root_challenger", "B", "S1", "QuartzVL", "root_challenger",
        search_overrides=shared_overrides, params={"snapshot_alpha": 0.5},
    )
    checkpoint = runner.CheckpointRef(id="C1", path="/tmp/model.pt")
    pos = {"id": "P1", "board": [0, 0, 0, 0], "player": 1}

    policies_b1, _, reused_b1 = runner.build_search_trace(DummyHarness(), checkpoint, pos, system_b1, [8], tmp_path)
    policies_b2, _, reused_b2 = runner.build_search_trace(DummyHarness(), checkpoint, pos, system_b2, [8], tmp_path)

    assert calls == ["B1"], (
        f"B1 and B2 share identical search_overrides and must share one cached "
        f"trace; expected only B1 to actually search, got calls={calls}"
    )
    assert reused_b1 is False, "first call must be a fresh search"
    assert reused_b2 is True, "second call must hit the cache built by the first"
    assert np.allclose(policies_b1[0], policies_b2[0])


def test_trace_cache_salt_covers_runner_and_config_schema_files():
    from quartz.phase15_trace import TRACE_CACHE_RELEVANT_PATHS

    covered = set(TRACE_CACHE_RELEVANT_PATHS)
    assert "configs/phase15_systems.default.json" in covered
    assert "quartz/phase15_suite.py" in covered
    assert "scripts/phase15_ablation_study.py" in covered
    assert "scripts/phase15_online_ablation.py" in covered
    assert "scripts/phase15_benchmark.py" in covered


def test_validate_cached_suite_payload_rejects_stale_reference_oracle_contract():
    runner = load_phase15_runner()
    payload = {
        "reference_checkpoint": {"id": "C01", "path": "/tmp/old.pt"},
        "reference_system": asdict(Phase15System("A0", "baseline", "A", "S0", "none", "none")),
        "oracle_checkpoint": {"id": "C01", "path": "/tmp/old.pt"},
        "oracle_system": asdict(Phase15System("ORACLE", "oracle", "A", "S0", "none", "none")),
    }
    try:
        runner.validate_cached_suite_payload(
            payload,
            reference_checkpoint=runner.CheckpointRef(id="C02", path="/tmp/new.pt"),
            reference_system=Phase15System("A0", "baseline", "A", "S0", "none", "none"),
            oracle_checkpoint=runner.CheckpointRef(id="C02", path="/tmp/new.pt"),
            oracle_system=Phase15System("ORACLE", "oracle", "A", "S0", "none", "none"),
        )
    except ValueError as exc:
        assert "cached position_suite metadata" in str(exc)
    else:
        raise AssertionError("expected stale cached suite metadata to be rejected")


def test_validate_checkpoint_refs_rejects_truncated_lexical_directory_selection(tmp_path: Path):
    runner = load_phase15_runner()
    root = tmp_path / "models"
    for family in ("F1_legacy_base", "F2_legacy_krefresh", "F3_theory_base", "F4_theory_krefresh"):
        for seed in ("seed_41", "seed_42", "seed_43"):
            path = root / family / seed
            path.mkdir(parents=True, exist_ok=True)
            (path / "best.pt").write_bytes(b"x")

    refs = [
        runner.CheckpointRef(id="C01_best", path=str(root / "F1_legacy_base" / "seed_41" / "best.pt")),
        runner.CheckpointRef(id="C02_best", path=str(root / "F1_legacy_base" / "seed_42" / "best.pt")),
        runner.CheckpointRef(id="C03_best", path=str(root / "F1_legacy_base" / "seed_43" / "best.pt")),
    ]
    args = SimpleNamespace(checkpoints=None, checkpoint_dir=str(root))

    try:
        runner.validate_checkpoint_refs(args, refs)
    except ValueError as exc:
        assert "curated checkpoint selection" in str(exc)
    else:
        raise AssertionError("expected lexical truncation to be rejected")


def test_validate_checkpoint_refs_accepts_explicit_curated_checkpoint_files(tmp_path: Path):
    runner = load_phase15_runner()
    root = tmp_path / "models"
    refs = []
    for idx, family in enumerate(("F1_legacy_base", "F2_legacy_krefresh", "F3_theory_base"), start=1):
        path = root / family / "seed_41"
        path.mkdir(parents=True, exist_ok=True)
        ckpt = path / "best.pt"
        ckpt.write_bytes(b"x")
        refs.append(runner.CheckpointRef(id=f"C{idx:02d}_best", path=str(ckpt)))

    args = SimpleNamespace(checkpoints=",".join(ref.path for ref in refs), checkpoint_dir=None)
    runner.validate_checkpoint_refs(args, refs)


def test_phase15_bootstrap_args_satisfy_controller_sweep_command_contract(monkeypatch, tmp_path: Path):
    runner = load_phase15_runner()
    monkeypatch.setattr(sys, "argv", ["phase15_ablation_study.py", "--bootstrap-if-empty"])
    args = runner.parse_args()
    command = runner.sweep.build_bootstrap_command(args, seed=41, output_dir=tmp_path / "seed_41")
    assert command[command.index("--backend") + 1] == "torch"


def test_phase15_benchmark_args_fail_closed_without_checkpoints(monkeypatch, tmp_path: Path):
    benchmark = load_phase15_benchmark_runner()
    monkeypatch.setattr(sys, "argv", ["phase15_benchmark.py"])
    args = benchmark.parse_args()
    with pytest.raises(RuntimeError, match="no checkpoints found"):
        benchmark.posthoc.resolve_checkpoint_refs(args, tmp_path)


def test_phase15_benchmark_bootstrap_args_satisfy_controller_sweep_command_contract(monkeypatch, tmp_path: Path):
    benchmark = load_phase15_benchmark_runner()
    monkeypatch.setattr(sys, "argv", ["phase15_benchmark.py", "--bootstrap-if-empty"])
    args = benchmark.parse_args()
    command = benchmark.posthoc.sweep.build_bootstrap_command(args, seed=41, output_dir=tmp_path / "seed_41")
    assert command[command.index("--backend") + 1] == "torch"


def test_phase15_online_args_fail_closed_without_checkpoints(monkeypatch, tmp_path: Path):
    online = load_phase15_online_runner()
    monkeypatch.setattr(sys, "argv", ["phase15_online_ablation.py"])
    args = online.parse_args()
    with pytest.raises(RuntimeError, match="no checkpoints found"):
        online.posthoc.resolve_checkpoint_refs(args, tmp_path)


def test_phase15_online_bootstrap_args_satisfy_controller_sweep_command_contract(monkeypatch, tmp_path: Path):
    online = load_phase15_online_runner()
    monkeypatch.setattr(sys, "argv", ["phase15_online_ablation.py", "--bootstrap-if-empty"])
    args = online.parse_args()
    command = online.posthoc.sweep.build_bootstrap_command(args, seed=41, output_dir=tmp_path / "seed_41")
    assert command[command.index("--backend") + 1] == "torch"


def test_phase15_contract_summary_uses_shared_schema():
    runner = load_phase15_runner()
    checkpoints = [runner.CheckpointRef(id="C01", path="/tmp/c01.pt")]
    systems = [
        Phase15System(
            id="A4",
            label="substrate",
            group="A",
            substrate="S1",
            controller="QuartzVL",
            refresh_operator="none",
            params={"penalty_scale": 0.25},
        )
    ]
    contracts = runner.build_phase15_contracts(
        execution_mode="posthoc",
        game="gomoku7",
        checkpoints=checkpoints,
        systems=systems,
        budgets=[8, 16, 32],
        trace_cache_salt_value="salt-v1",
        extra={"seed": 7},
    )
    summary = runner.summarize_phase15_contracts(contracts)
    assert summary["count"] == len(contracts)
    assert summary["discarded_count"] == 0
    assert summary["legacy_partial_count"] == 0
    assert summary["hash_key"] == "stable_json_hash"
    assert isinstance(summary["collection_hash"], str) and summary["collection_hash"]


def test_benchmark_summary_reports_tie_aware_match_rate():
    benchmark = load_phase15_benchmark_runner()
    rows = [
        {
            "checkpoint_id": "C01",
            "system": "A4",
            "budget": 16,
            "continuation_wallclock_ms": 10.0,
            "restart_wallclock_ms": 20.0,
            "continuation_effective_runtime_ms": 5.0,
            "restart_effective_runtime_ms": 15.0,
            "policy_kl_restart_vs_continuation": 0.1,
            "argmax_match": 0,
            "tie_aware_match": 1,
            "ambiguous_top1_case": 1,
        },
        {
            "checkpoint_id": "C01",
            "system": "A4",
            "budget": 16,
            "continuation_wallclock_ms": 12.0,
            "restart_wallclock_ms": 24.0,
            "continuation_effective_runtime_ms": 6.0,
            "restart_effective_runtime_ms": 18.0,
            "policy_kl_restart_vs_continuation": 0.2,
            "argmax_match": 1,
            "tie_aware_match": 1,
            "ambiguous_top1_case": 0,
        },
    ]
    summary = benchmark.summarize_rows(rows)
    assert summary["argmax_match_rate"] == 0.5
    assert summary["tie_aware_match_rate"] == 1.0
    assert summary["ambiguous_top1_rate"] == 0.5
    assert summary["by_checkpoint_system_budget"][0]["tie_aware_match_rate"] == 1.0


def test_benchmark_contract_summary_uses_shared_schema():
    benchmark = load_phase15_benchmark_runner()
    checkpoints = [benchmark.posthoc.CheckpointRef(id="C01", path="/tmp/c01.pt")]
    systems = [
        Phase15System(
            id="B2",
            label="challenger",
            group="B",
            substrate="S1",
            controller="QuartzVL",
            refresh_operator="root_challenger",
            params={"challenger_k": 2},
        )
    ]
    args = SimpleNamespace(
        game="gomoku7",
        positions_file="/tmp/positions.json",
        seed=7,
        repeats=2,
        warmup_rounds=1,
        min_bundle_speedup=1.8,
        min_tie_aware_match=0.65,
        max_kl_mean=0.25,
    )
    summary = benchmark.build_benchmark_contract_summary(
        args,
        checkpoints,
        systems,
        [8, 16, 32],
        positions_count=4,
    )
    assert summary["count"] >= 3
    assert summary["discarded_count"] == 0
    assert summary["legacy_partial_count"] == 0
    assert summary["hash_key"] == "stable_json_hash"
    assert isinstance(summary["collection_hash"], str) and summary["collection_hash"]


def test_benchmark_bundle_summary_reports_modes_and_fallbacks():
    benchmark = load_phase15_benchmark_runner()
    summary = benchmark.summarize_bundle_runs(
        [
            {
                "checkpoint_id": "C01",
                "system": "A4",
                "continuation_bundle_wallclock_ms": 10.0,
                "restart_bundle_wallclock_ms": 20.0,
                "continuation_bundle_trace_acquire_ms": 7.0,
                "restart_bundle_trace_acquire_ms": 18.0,
                "continuation_bundle_overhead_ms": 3.0,
                "restart_bundle_overhead_ms": 2.0,
                "continuation_mode": "root_continuation",
                "continuation_fallback_reason": None,
            },
            {
                "checkpoint_id": "C01",
                "system": "A4",
                "continuation_bundle_wallclock_ms": 1200.0,
                "restart_bundle_wallclock_ms": 24.0,
                "continuation_bundle_trace_acquire_ms": 8.0,
                "restart_bundle_trace_acquire_ms": 21.0,
                "continuation_bundle_overhead_ms": 4.0,
                "restart_bundle_overhead_ms": 3.0,
                "continuation_mode": "restart_per_chunk",
                "continuation_fallback_reason": "RuntimeError: resident session unavailable",
            },
        ],
        [
            {
                "checkpoint_id": "C01",
                "system": "A4",
                "tie_aware_match": 1,
                "ambiguous_top1_case": 0,
                "policy_kl_restart_vs_continuation": 0.12,
            },
            {
                "checkpoint_id": "C01",
                "system": "A4",
                "tie_aware_match": 0,
                "ambiguous_top1_case": 1,
                "policy_kl_restart_vs_continuation": 0.28,
            },
        ],
    )
    assert summary["runs"] == 2
    assert summary["continuation_modes"] == {"restart_per_chunk": 1, "root_continuation": 1}
    assert summary["fallback_reasons"] == {"RuntimeError: resident session unavailable": 1}
    assert summary["continuation_overhead_ms"]["mean"] == 3.5
    assert summary["wallclock_speedup_pairwise"]["median"] == pytest.approx(1.01)
    assert summary["continuation_wallclock_outlier_count"] == 1
    assert summary["by_checkpoint_system"][0]["speedup_headwind"] == "readout_sensitivity"


def test_benchmark_prefixed_readout_meta_exports_guard_fields():
    benchmark = load_phase15_benchmark_runner()
    fields = benchmark.prefixed_readout_meta(
        "continuation",
        {
            "guard_operator": "argmax_tie_guard",
            "guard_reason": "thin_margin",
            "fallback_operator": "root_posterior_snapshot",
            "guard_vetoed": 1,
            "guard_margin": 0.01,
            "guard_margin_floor": 0.03,
            "proposal_argmax": 4,
            "posterior_argmax": 7,
        },
    )

    assert fields == {
        "continuation_guard_operator": "argmax_tie_guard",
        "continuation_guard_reason": "thin_margin",
        "continuation_fallback_operator": "root_posterior_snapshot",
        "continuation_guard_vetoed": 1,
        "continuation_guard_margin": 0.01,
        "continuation_guard_margin_floor": 0.03,
        "continuation_proposal_argmax": 4,
        "continuation_posterior_argmax": 7,
    }


def test_phase15_analysis_reports_paired_posthoc_deltas():
    analysis = load_phase15_analysis_runner()
    rows = [
        {
            "checkpoint_id": "C01",
            "position_id": "P1",
            "budget": 8,
            "system": "A4",
            "accuracy_to_oracle": 1,
            "topk_recall_oracle": 1,
            "kl_to_oracle": 0.30,
        },
        {
            "checkpoint_id": "C01",
            "position_id": "P1",
            "budget": 8,
            "system": "B9",
            "accuracy_to_oracle": 1,
            "topk_recall_oracle": 0,
            "kl_to_oracle": 0.40,
        },
        {
            "checkpoint_id": "C01",
            "position_id": "P2",
            "budget": 8,
            "system": "A4",
            "accuracy_to_oracle": 0,
            "topk_recall_oracle": 0,
            "kl_to_oracle": 0.50,
        },
        {
            "checkpoint_id": "C01",
            "position_id": "P2",
            "budget": 8,
            "system": "B9",
            "accuracy_to_oracle": 1,
            "topk_recall_oracle": 1,
            "kl_to_oracle": 0.40,
        },
    ]

    deltas = analysis.paired_posthoc_deltas(rows, targets=("B9",), baselines=("A4",))

    assert deltas == [
        {
            "target": "B9",
            "baseline": "A4",
            "pairs": 2,
            "delta_accuracy_to_oracle": 0.5,
            "delta_topk_recall_oracle": 0.0,
            "delta_kl_to_oracle": 0.0,
            "accuracy_win_count": 1,
            "accuracy_loss_count": 0,
            "accuracy_tie_count": 1,
        }
    ]


def test_phase15_analysis_reports_guard_reasons_by_system_and_budget():
    analysis = load_phase15_analysis_runner()
    rows = [
        {"system": "B9", "budget": 8, "guard_vetoed": 1, "guard_reason": "thin_margin"},
        {"system": "B9", "budget": 8, "guard_vetoed": 0, "guard_reason": "passed"},
        {"system": "B9", "budget": 16, "guard_vetoed": 1, "guard_reason": "unstable_suffix;thin_margin"},
    ]

    guard = analysis.guard_summary(rows, system="B9")

    assert guard["system"] == "B9"
    assert guard["rows"] == 3
    assert guard["veto_rate"] == pytest.approx(2 / 3)
    assert guard["reason_counts"] == {
        "passed": 1,
        "thin_margin": 1,
        "unstable_suffix;thin_margin": 1,
    }
    assert guard["by_budget"] == [
        {"budget": 8, "rows": 2, "veto_rate": 0.5},
        {"budget": 16, "rows": 1, "veto_rate": 1.0},
    ]


def test_phase15_analysis_reports_stabilizer_reasons_by_system_and_budget():
    analysis = load_phase15_analysis_runner()
    rows = [
        {"system": "B10", "budget": 8, "stabilizer_applied": 1, "stabilization_reason": "passed"},
        {
            "system": "B10",
            "budget": 8,
            "stabilizer_applied": 0,
            "stabilization_reason": "topk_anchor_break",
        },
        {
            "system": "B10",
            "budget": 16,
            "stabilizer_applied": 0,
            "stabilization_reason": "unstable_suffix",
        },
    ]

    stabilizer = analysis.stabilizer_summary(rows, system="B10")

    assert stabilizer["system"] == "B10"
    assert stabilizer["rows"] == 3
    assert stabilizer["applied_rate"] == pytest.approx(1 / 3)
    assert stabilizer["reason_counts"] == {
        "passed": 1,
        "topk_anchor_break": 1,
        "unstable_suffix": 1,
    }
    assert stabilizer["by_budget"] == [
        {"budget": 8, "rows": 2, "applied_rate": 0.5},
        {"budget": 16, "rows": 1, "applied_rate": 0.0},
    ]


def test_phase15_analysis_reports_budget_scheduler_confounds_by_system_and_budget():
    analysis = load_phase15_analysis_runner()
    rows = [
        {
            "system": "B3",
            "budget": 8,
            "budget_confounded": 1,
            "budget_burst_triggered": 0,
            "extra_budget_used": 0,
            "burst_budget": 8,
            "burst_reason": "none",
        },
        {
            "system": "B3",
            "budget": 8,
            "budget_confounded": 1,
            "budget_burst_triggered": 1,
            "extra_budget_used": 8,
            "burst_budget": 16,
            "burst_reason": "low_persistence",
        },
        {
            "system": "B3",
            "budget": 16,
            "budget_confounded": 1,
            "budget_burst_triggered": 1,
            "extra_budget_used": 16,
            "burst_budget": 32,
            "burst_reason": "unstable_margin",
        },
    ]

    summary = analysis.budget_scheduler_summary(rows, system="B3")

    assert summary["system"] == "B3"
    assert summary["rows"] == 3
    assert summary["budget_confounded_rate"] == 1.0
    assert summary["budget_burst_rate"] == pytest.approx(2 / 3)
    assert summary["extra_budget_used_mean"] == pytest.approx(8.0)
    assert summary["positive_extra_budget_used_mean"] == pytest.approx(12.0)
    assert summary["burst_reason_counts"] == {
        "low_persistence": 1,
        "none": 1,
        "unstable_margin": 1,
    }
    assert summary["by_budget"] == [
        {
            "budget": 8,
            "rows": 2,
            "budget_burst_rate": 0.5,
            "extra_budget_used_mean": 4.0,
            "burst_budget_mean": 12.0,
        },
        {
            "budget": 16,
            "rows": 1,
            "budget_burst_rate": 1.0,
            "extra_budget_used_mean": 16.0,
            "burst_budget_mean": 32.0,
        },
    ]


def test_phase15_analysis_builds_claim_firewalled_interpretation_flags():
    analysis = load_phase15_analysis_runner()
    report = analysis.build_analysis_report(
        posthoc_rows=[
            {
                "checkpoint_id": "C01",
                "position_id": "P1",
                "budget": 8,
                "system": "A4",
                "accuracy_to_oracle": 1,
                "topk_recall_oracle": 1,
                "kl_to_oracle": 0.10,
            },
            {
                "checkpoint_id": "C01",
                "position_id": "P1",
                "budget": 8,
                "system": "B9",
                "accuracy_to_oracle": 1,
                "topk_recall_oracle": 0,
                "kl_to_oracle": 0.30,
                "guard_vetoed": 1,
                "guard_reason": "thin_margin",
            },
        ],
        benchmark_rows=[],
        targets=("B9",),
        baselines=("A4",),
    )

    assert report["claim_status"] == "ANALYSIS-ONLY; candidate quality remains ABLATION-PENDING"
    assert "B9_no_accuracy_gain_vs_A4_with_higher_kl" in report["interpretation_flags"]
    assert report["posthoc_guard_summary"][0]["system"] == "B9"


def test_phase15_analysis_builds_snapshot_stabilizer_sections():
    analysis = load_phase15_analysis_runner()
    report = analysis.build_analysis_report(
        posthoc_rows=[
            {
                "checkpoint_id": "C01",
                "position_id": "P1",
                "budget": 8,
                "system": "A4",
                "accuracy_to_oracle": 1,
                "topk_recall_oracle": 1,
                "kl_to_oracle": 0.10,
            },
            {
                "checkpoint_id": "C01",
                "position_id": "P1",
                "budget": 8,
                "system": "B10",
                "accuracy_to_oracle": 1,
                "topk_recall_oracle": 1,
                "kl_to_oracle": 0.09,
                "stabilizer_applied": 1,
                "stabilization_reason": "passed",
            },
        ],
        benchmark_rows=[
            {
                "system": "B10",
                "budget": 8,
                "tie_aware_match": 1,
                "argmax_match": 0,
                "policy_kl_restart_vs_continuation": 0.40,
                "continuation_stabilizer_applied": 1,
                "continuation_stabilization_reason": "passed",
                "restart_stabilizer_applied": 0,
                "restart_stabilization_reason": "topk_anchor_break",
            }
        ],
        targets=("B10",),
        baselines=("A4",),
    )

    assert report["posthoc_stabilizer_summary"][0]["system"] == "B10"
    assert report["benchmark_stabilizer_summary"][0]["system"] == "B10"
    assert report["benchmark_stabilizer_summary"][0]["prefix"] == "continuation"
    assert "B10_rehearsal_lower_kl_without_accuracy_or_topk_loss_vs_A4" in report["interpretation_flags"]


def test_phase15_analysis_auto_targets_tracks_telemetry_candidates():
    analysis = load_phase15_analysis_runner()
    posthoc_rows = [
        {
            "checkpoint_id": "C01",
            "position_id": "P1",
            "budget": 8,
            "system": "A4",
            "accuracy_to_oracle": 1,
            "topk_recall_oracle": 1,
            "kl_to_oracle": 0.20,
        },
        {
            "checkpoint_id": "C01",
            "position_id": "P1",
            "budget": 8,
            "system": "B3",
            "accuracy_to_oracle": 1,
            "topk_recall_oracle": 1,
            "kl_to_oracle": 0.16,
            "budget_confounded": 1,
            "budget_burst_triggered": 1,
            "extra_budget_used": 8,
        },
        {
            "checkpoint_id": "C01",
            "position_id": "P1",
            "budget": 8,
            "system": "B10",
            "accuracy_to_oracle": 1,
            "topk_recall_oracle": 1,
            "kl_to_oracle": 0.18,
            "stabilizer_applied": 1,
            "stabilization_reason": "passed",
        },
        {
            "checkpoint_id": "C01",
            "position_id": "P1",
            "budget": 8,
            "system": "B12",
            "accuracy_to_oracle": 1,
            "topk_recall_oracle": 1,
            "kl_to_oracle": 0.17,
            "stabilizer_applied": 1,
            "stabilization_reason": "passed",
            "entropy_expansion_gate_passed": 1,
        },
    ]

    report = analysis.build_analysis_report(
        posthoc_rows=posthoc_rows,
        benchmark_rows=[],
        targets=("auto",),
        baselines=("A4",),
    )

    assert report["targets"] == ["B3", "B10", "B12"]
    assert [row["target"] for row in report["paired_posthoc_deltas"]] == ["B3", "B10", "B12"]
    assert report["analysis_coverage"]["telemetry_candidate_systems"] == ["B3", "B10", "B12"]
    assert report["analysis_coverage"]["untargeted_telemetry_systems"] == []
    assert "analysis_targets_omit_available_telemetry_candidates" not in report["interpretation_flags"]


def test_phase15_analysis_flags_explicit_targets_that_omit_telemetry_candidates():
    analysis = load_phase15_analysis_runner()
    report = analysis.build_analysis_report(
        posthoc_rows=[
            {
                "checkpoint_id": "C01",
                "position_id": "P1",
                "budget": 8,
                "system": "A4",
                "accuracy_to_oracle": 1,
                "topk_recall_oracle": 1,
                "kl_to_oracle": 0.20,
            },
            {
                "checkpoint_id": "C01",
                "position_id": "P1",
                "budget": 8,
                "system": "B10",
                "accuracy_to_oracle": 1,
                "topk_recall_oracle": 1,
                "kl_to_oracle": 0.18,
                "stabilizer_applied": 1,
                "stabilization_reason": "passed",
            },
            {
                "checkpoint_id": "C01",
                "position_id": "P1",
                "budget": 8,
                "system": "B12",
                "accuracy_to_oracle": 1,
                "topk_recall_oracle": 1,
                "kl_to_oracle": 0.17,
                "stabilizer_applied": 1,
                "stabilization_reason": "passed",
                "entropy_expansion_gate_passed": 1,
            },
        ],
        benchmark_rows=[],
        targets=("B10",),
        baselines=("A4",),
    )

    assert report["analysis_coverage"]["telemetry_candidate_systems"] == ["B10", "B12"]
    assert report["analysis_coverage"]["untargeted_telemetry_systems"] == ["B12"]
    assert report["analysis_coverage"]["missing_target_systems"] == []
    assert "analysis_targets_omit_available_telemetry_candidates" in report["interpretation_flags"]


def test_phase15_analysis_orders_system_ids_naturally_in_coverage_and_auto_targets():
    analysis = load_phase15_analysis_runner()
    report = analysis.build_analysis_report(
        posthoc_rows=[
            {"checkpoint_id": "C01", "position_id": "P1", "budget": 8, "system": "B10", "stabilizer_applied": 1},
            {"checkpoint_id": "C01", "position_id": "P1", "budget": 8, "system": "A4"},
            {"checkpoint_id": "C01", "position_id": "P1", "budget": 8, "system": "B9", "guard_vetoed": 1},
        ],
        benchmark_rows=[],
        targets=("auto",),
        baselines=("A4",),
    )

    assert report["targets"] == ["B9", "B10"]
    assert report["analysis_coverage"]["available_systems"] == ["A4", "B9", "B10"]
    assert report["analysis_coverage"]["telemetry_candidate_systems"] == ["B9", "B10"]


def test_phase15_analysis_report_includes_budget_scheduler_confound_section():
    analysis = load_phase15_analysis_runner()
    report = analysis.build_analysis_report(
        posthoc_rows=[
            {
                "checkpoint_id": "C01",
                "position_id": "P1",
                "budget": 8,
                "system": "A4",
                "accuracy_to_oracle": 0,
                "topk_recall_oracle": 0,
                "kl_to_oracle": 0.4,
            },
            {
                "checkpoint_id": "C01",
                "position_id": "P1",
                "budget": 8,
                "system": "B3",
                "accuracy_to_oracle": 1,
                "topk_recall_oracle": 1,
                "kl_to_oracle": 0.2,
                "budget_confounded": 1,
                "budget_burst_triggered": 1,
                "extra_budget_used": 8,
                "burst_budget": 16,
                "burst_reason": "low_persistence",
            },
        ],
        benchmark_rows=[],
        targets=("B3",),
        baselines=("A4",),
    )

    assert report["posthoc_budget_scheduler_summary"][0]["system"] == "B3"
    assert report["posthoc_budget_scheduler_summary"][0]["budget_burst_rate"] == 1.0
    assert "B3_budget_confounded_quality_signal" in report["interpretation_flags"]


def test_benchmark_checkpoint_payload_records_checkpoint_paths():
    benchmark = load_phase15_benchmark_runner()
    payload = benchmark.benchmark_checkpoint_payload(
        [
            benchmark.posthoc.CheckpointRef(id="C01_best", path="/tmp/c01/best.pt"),
            benchmark.posthoc.CheckpointRef(id="C02_best", path="/tmp/c02/best.pt"),
        ]
    )
    assert payload == [
        {"id": "C01_best", "path": "/tmp/c01/best.pt"},
        {"id": "C02_best", "path": "/tmp/c02/best.pt"},
    ]


def test_benchmark_gate_uses_bundle_speedup_tie_aware_and_kl():
    benchmark = load_phase15_benchmark_runner()
    gate = benchmark.evaluate_benchmark_gate(
        {
            "tie_aware_match_rate": 0.82,
            "policy_kl_restart_vs_continuation": {"mean": 0.18},
        },
        {
            "wallclock_speedup_mean": 2.05,
        },
        min_bundle_speedup=1.8,
        min_tie_aware_match=0.75,
        max_kl_mean=0.25,
    )
    assert gate["passed"] is True
    assert [item["name"] for item in gate["checks"]] == [
        "bundle_speedup_mean",
        "tie_aware_match_rate",
        "policy_kl_mean",
    ]


def test_phase15_benchmark_ci_smoke_builds_self_contained_command():
    smoke = load_phase15_benchmark_ci_smoke_runner()
    args = SimpleNamespace(
        game="gomoku7",
        output="results/phase15_ci_gate",
        rust_binary="./target/release/mcts_demo",
        systems="A4,B1,B2,B3",
        budgets="8,16,32,64",
        seed=7,
        search_stall_time_s=180.0,
        search_stall_timeout_s=180.0,
    )
    command = smoke.build_benchmark_command(args, Path("/tmp/model.pt"), Path("/tmp/positions.json"))
    assert command[0] == sys.executable
    assert "--checkpoints" in command
    assert "/tmp/model.pt" in command
    assert "--positions-file" in command
    assert "/tmp/positions.json" in command
    assert "--seed" in command
    assert "7" in command
    assert "--enforce-gate" in command


def test_phase15_benchmark_ci_smoke_can_expand_small_candidate_preset_without_gate():
    smoke = load_phase15_benchmark_ci_smoke_runner()
    args = SimpleNamespace(
        game="gomoku7",
        output="results/phase15_small_gate",
        rust_binary="./target/release/mcts_demo",
        systems="small",
        budgets="8,16,32,64",
        seed=7,
        search_stall_timeout_s=180.0,
        enforce_gate=False,
    )
    command = smoke.build_benchmark_command(args, Path("/tmp/model.pt"), Path("/tmp/positions.json"))
    assert command[command.index("--systems") + 1] == "A4,B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12"
    assert "--enforce-gate" not in command


def test_phase15_benchmark_ci_smoke_report_includes_benchmark_contract_summary():
    smoke = load_phase15_benchmark_ci_smoke_runner()
    args = SimpleNamespace(
        game="gomoku7",
        output="results/phase15_ci_gate",
        systems="A4,B1,B2",
        budgets="8,16,32,64",
        seed=7,
        search_stall_timeout_s=180.0,
    )
    report = smoke.build_ci_smoke_contract_summary(
        args,
        checkpoint_path=Path("/tmp/model.pt"),
        positions_path=Path("/tmp/positions.json"),
        benchmark_payload={
            "contract_summary": {
                "count": 6,
                "collection_hash": "abc123",
                "discarded_count": 0,
                "legacy_partial_count": 0,
                "hash_key": "stable_json_hash",
            }
        },
    )
    assert report["runner"]["game"] == "gomoku7"
    assert report["runner"]["systems"] == ["A4", "B1", "B2"]
    assert report["artifacts"]["checkpoint_path"] == "/tmp/model.pt"
    assert report["benchmark_contract_summary"]["collection_hash"] == "abc123"
    assert report["benchmark_contract_summary"]["hash_key"] == "stable_json_hash"


def test_phase15_toy_ablation_builds_posthoc_and_benchmark_candidate_commands():
    toy = load_phase15_toy_ablation_runner()
    args = SimpleNamespace(
        game="gomoku7",
        output="results/phase15_toy_ablation",
        rust_binary="./target/release/mcts_demo",
        device="auto",
        systems="small",
        groups="A,B",
        budgets="8,16,32,64",
        oracle_budget=64,
        seed=7,
        max_positions=4,
        run="both",
        benchmark_repeats=1,
        benchmark_warmup_rounds=0,
        enforce_benchmark_gate=False,
        search_stall_timeout_s=180.0,
    )
    posthoc_command = toy.build_posthoc_command(args, Path("/tmp/model.pt"), Path("/tmp/positions.json"))
    benchmark_command = toy.build_benchmark_command(args, Path("/tmp/model.pt"), Path("/tmp/positions.json"))
    assert posthoc_command[posthoc_command.index("--systems") + 1] == "A4,B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12"
    assert posthoc_command[posthoc_command.index("--groups") + 1] == "A,B"
    assert posthoc_command[posthoc_command.index("--reference-checkpoint") + 1] == "/tmp/model.pt"
    assert benchmark_command[benchmark_command.index("--systems") + 1] == "A4,B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12"
    assert "--enforce-gate" not in benchmark_command


def test_phase15_benchmark_ci_smoke_positions_are_deterministic():
    smoke = load_phase15_benchmark_ci_smoke_runner()
    positions = smoke.deterministic_positions(7)
    assert len(positions) == 4
    assert positions[0]["id"] == "P0001"
    assert positions[1]["id"] == "P0002"
    assert positions[2]["id"] == "P0003"
    assert positions[3]["id"] == "P0004"
    assert len(positions[0]["board"]) == 49
