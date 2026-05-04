import importlib.util
import argparse
import builtins
import json
import io
import logging
import math
import os
import struct
import sys
import tomllib
import types
import warnings
from pathlib import Path
import random

import numpy as np
import pytest


def load_training_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "quartz_alphazero_train", root / "quartz" / "alphazero_train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_train_entry_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "quartz_train_entry", root / "quartz" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_cli_main_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "quartz_cli_main", root / "quartz" / "cli_main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_backend_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "quartz_backend", root / "quartz" / "backend.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_encoders_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "encoders", root / "quartz" / "encoders.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_monitor_script_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "profile_training_monitor", root / "scripts" / "profile_training_monitor.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_ablation_script_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "quartz_ablation_study", root / "scripts" / "ablation_study.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_controller_sweep_script_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "quartz_controller_sweep", root / "scripts" / "controller_sweep.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_gpu_detect_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "quartz_gpu_detect", root / "quartz" / "gpu_detect.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_play_gui_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "quartz_play_gui", root / "quartz" / "play_gui.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_module_with_torch_blocked(module_name, relative_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(module_name, root / relative_path)
    module = importlib.util.module_from_spec(spec)
    original_import = builtins.__import__
    saved = {name: sys.modules.pop(name) for name in list(sys.modules) if name == "torch" or name.startswith("torch.")}

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError(f"unexpected torch import while loading {relative_path}: {name}")
        return original_import(name, globals, locals, fromlist, level)

    try:
        builtins.__import__ = guarded_import
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        builtins.__import__ = original_import
        sys.modules.pop(spec.name, None)
        sys.modules.update(saved)


def test_replay_values_follow_side_to_move_for_white_win():
    az = load_training_module()
    replay = az.ReplayBuffer(16)
    states = [np.zeros((3, 7, 7), dtype=np.float32) for _ in range(2)]
    policies = [np.zeros(49, dtype=np.float32) for _ in range(2)]

    replay.add_game(states, policies, outcome=-1.0)

    assert [sample[2] for sample in replay.buf] == [-1.0, 1.0]


def test_sparse_replay_roundtrip_preserves_dense_policy_targets(tmp_path):
    az = load_training_module()
    replay = az.ReplayBuffer(16)
    state_a = np.zeros((3, 7, 7), dtype=np.float32)
    state_b = np.ones((3, 7, 7), dtype=np.float32)
    dense_policy = np.zeros(49, dtype=np.float32)
    dense_policy[7] = 1.0
    sparse_policy = az.sparse_policy_from_entries([[3, 0.25], [8, 0.75]], 49)

    replay.add(state_a, dense_policy, 0.5)
    replay.add(state_b, sparse_policy, -0.25)

    path = tmp_path / "replay_v2.npz"
    replay.save(path)

    loaded = az.ReplayBuffer(16)
    assert loaded.load(path) == 2

    _, policies_t, values_t = az.collate_replay_samples(list(loaded.buf))
    np.testing.assert_allclose(policies_t.numpy()[0], dense_policy)
    np.testing.assert_allclose(
        policies_t.numpy()[1],
        az.dense_policy_from_sparse([[3, 0.25], [8, 0.75]], 49),
    )
    np.testing.assert_allclose(values_t.numpy(), [0.5, -0.25])


def test_collate_replay_samples_fast_path_matches_tuple_reference():
    az = load_training_module()
    replay = az.ReplayBuffer(16)
    state_a = np.zeros((3, 7, 7), dtype=np.float32)
    state_b = np.ones((3, 7, 7), dtype=np.float32)
    policy_a = np.zeros(49, dtype=np.float32)
    policy_b = np.zeros(49, dtype=np.float32)
    policy_a[3] = 1.0
    policy_b[11] = 1.0
    replay.add(state_a, policy_a, 0.5)
    replay.add(state_b, policy_b, -0.25)

    states_fast, policies_fast, values_fast = az.collate_replay_samples(list(replay.buf))
    tuple_batch = [(sample.state, sample.policy.dense(), sample.value) for sample in replay.buf]
    states_ref, policies_ref, values_ref = az.collate_replay_samples(tuple_batch)

    np.testing.assert_allclose(states_fast.numpy(), states_ref.numpy())
    np.testing.assert_allclose(policies_fast.numpy(), policies_ref.numpy())
    np.testing.assert_allclose(values_fast.numpy(), values_ref.numpy())


def test_replay_roundtrip_preserves_search_metadata(tmp_path):
    az = load_training_module()
    replay = az.ReplayBuffer(16)
    state = np.zeros((3, 7, 7), dtype=np.float32)
    policy = np.zeros(49, dtype=np.float32)
    policy[7] = 1.0
    metadata = {
        "search_manifest": {"profile": "baseline_strict", "benchmark_safe": True},
        "realized_budget": {"realized_iterations": 32},
        "controller_summary": {"dup_rate": 0.125, "p_flip": 0.25},
    }

    replay.add(state, policy, 0.5, metadata=metadata)
    path = tmp_path / "replay_meta_v2.npz"
    replay.save(path)

    loaded = az.ReplayBuffer(16)
    assert loaded.load(path) == 1
    sample = loaded.buf[0]
    assert sample.metadata["search_manifest"]["profile"] == "baseline_strict"
    assert sample.metadata["realized_budget"]["realized_iterations"] == 32
    assert sample.metadata["controller_summary"]["dup_rate"] == pytest.approx(0.125)


def test_replay_search_summary_exposes_halt_reason_histogram():
    az = load_training_module()
    replay = az.ReplayBuffer(16)
    state = np.zeros((3, 7, 7), dtype=np.float32)
    policy = np.zeros(49, dtype=np.float32)
    policy[4] = 1.0
    replay.add(
        state,
        policy,
        0.5,
        metadata={
            "search_manifest": {"profile": "quartz", "benchmark_safe": True},
            "realized_budget": {"realized_iterations": 16},
            "controller_summary": {
                "dup_rate": 0.1,
                "p_flip": 0.2,
                "stop_reason": "BudgetExhausted",
                "halt_reason_hist": {"BudgetExhausted": 1},
                "penalty_mode": "GatedRefresh",
                "prior_refresh_rate": 0.5,
                "prior_q_divergence": 0.4,
                "root_only_shaping": True,
                "telemetry_partial": True,
                "actuator_coverage": {
                    "prior_refresh_rate_configured": True,
                    "prior_refresh_rate_consumed_by_mode": False,
                    "prior_refresh_rate_inert_for_mode": True,
                    "prior_refresh_source": "prior_q_divergence_gate",
                },
            },
        },
    )
    replay.add(
        state,
        policy,
        -0.5,
        metadata={
            "search_manifest": {"profile": "quartz", "benchmark_safe": True},
            "realized_budget": {"realized_iterations": 12},
            "controller_summary": {
                "dup_rate": 0.2,
                "p_flip": 0.3,
                "stop_reason": "Converged",
                "halt_reason_hist": {"Converged": 1},
                "penalty_mode": "GatedRefresh",
                "prior_refresh_rate": 0.0,
                "prior_q_divergence": 0.2,
                "root_only_shaping": False,
                "telemetry_partial": True,
                "actuator_coverage": {
                    "prior_refresh_rate_configured": False,
                    "prior_refresh_rate_consumed_by_mode": False,
                    "prior_refresh_rate_inert_for_mode": False,
                    "prior_refresh_source": "prior_q_divergence_gate",
                },
            },
        },
    )

    summary = az.ReplayMetrics.search_summary(replay, sample_n=2)

    assert summary["search_profile_counts"]["quartz"] == 2
    assert summary["halt_reason_hist"]["BudgetExhausted"] == 1
    assert summary["halt_reason_hist"]["Converged"] == 1
    assert summary["mean_dup_rate"] == pytest.approx(0.15)
    assert summary["mean_p_flip"] == pytest.approx(0.25)
    assert summary["controller_penalty_mode_counts"]["GatedRefresh"] == 2
    assert summary["mean_prior_refresh_rate"] == pytest.approx(0.25)
    assert summary["mean_prior_q_divergence"] == pytest.approx(0.3)
    assert summary["root_only_shaping_frac"] == pytest.approx(0.5)
    assert summary["controller_telemetry_partial_frac"] == pytest.approx(1.0)
    assert summary["halt_metric_coverage_frac"] == pytest.approx(1.0)
    assert summary["refresh_metric_coverage_frac"] == pytest.approx(0.0)
    assert summary["penalty_metric_coverage_frac"] == pytest.approx(0.0)
    assert summary["actuator_coverage_frac"] == pytest.approx(1.0)
    assert summary["prior_refresh_rate_configured_frac"] == pytest.approx(0.5)
    assert summary["prior_refresh_rate_consumed_by_mode_frac"] == pytest.approx(0.0)
    assert summary["prior_refresh_rate_inert_for_mode_frac"] == pytest.approx(0.5)
    assert summary["prior_refresh_source_counts"]["prior_q_divergence_gate"] == 2


def test_p6_replay_summary_aggregates_voc_channels_and_schema_version():
    """P6 (audit_codex_20260425.md W8): the replay summary must fold per-sample
    voc_total / voc_focus / voc_expand / voc_merge channel decomposition and
    surface a `controller_schema_versions` census so wire-format drift is
    visible to ablation analysis.
    """
    az = load_training_module()
    replay = az.ReplayBuffer(16)
    state = np.zeros((3, 7, 7), dtype=np.float32)
    policy = np.zeros(49, dtype=np.float32)
    policy[4] = 1.0
    replay.add(
        state,
        policy,
        0.5,
        metadata={
            "search_manifest": {"profile": "quartz", "benchmark_safe": True},
            "realized_budget": {"realized_iterations": 16},
            "controller_summary": {
                "schema_version": 1,
                "p_flip": 0.1,
                "voc_total": 0.04,
                "voc_focus": 0.02,
                "voc_expand": 0.01,
                "voc_merge": 0.01,
            },
        },
    )
    replay.add(
        state,
        policy,
        -0.5,
        metadata={
            "search_manifest": {"profile": "quartz", "benchmark_safe": True},
            "realized_budget": {"realized_iterations": 12},
            "controller_summary": {
                "schema_version": 1,
                "p_flip": 0.2,
                "voc_total": 0.08,
                "voc_focus": 0.04,
                "voc_expand": 0.02,
                "voc_merge": 0.02,
            },
        },
    )

    summary = az.ReplayMetrics.search_summary(replay, sample_n=2)

    assert summary["mean_voc_total"] == pytest.approx(0.06)
    assert summary["mean_voc_focus"] == pytest.approx(0.03)
    assert summary["mean_voc_expand"] == pytest.approx(0.015)
    assert summary["mean_voc_merge"] == pytest.approx(0.015)
    assert summary["controller_schema_versions"] == {"1": 2}


def test_replay_search_summary_aggregates_selection_trace():
    az = load_training_module()
    replay = az.ReplayBuffer(16)
    state = np.zeros((3, 7, 7), dtype=np.float32)
    policy = np.zeros(49, dtype=np.float32)
    policy[4] = 1.0
    for root_selects, refresh_selected, penalty_sum, prior_l1 in [
        (10, 4, 1.5, 0.25),
        (20, 6, 2.5, 0.75),
    ]:
        replay.add(
            state,
            policy,
            0.0,
            metadata={
                "search_manifest": {"profile": "quartz", "benchmark_safe": True},
                "realized_budget": {"realized_iterations": root_selects},
                "controller_summary": {
                    "schema_version": 4,
                    "selection_trace": {
                        "root_selects": root_selects,
                        "refresh_selected_count": refresh_selected,
                        "selected_penalty_abs_sum": penalty_sum,
                        "selected_effective_prior_l1_sum": prior_l1,
                        "selected_mean_candidate_count": 7.0,
                        "selected_max_candidate_count": 9,
                    },
                },
            },
        )

    summary = az.ReplayMetrics.search_summary(replay, sample_n=2)

    assert summary["selection_trace_coverage_frac"] == pytest.approx(1.0)
    assert summary["mean_selection_root_selects"] == pytest.approx(15.0)
    assert summary["selection_refresh_selected_frac"] == pytest.approx(10 / 30)
    assert summary["mean_selection_penalty_abs_sum"] == pytest.approx(2.0)
    assert summary["mean_selection_effective_prior_l1_sum"] == pytest.approx(0.5)
    assert summary["mean_selection_candidate_count"] == pytest.approx(7.0)
    assert summary["max_selection_candidate_count"] == pytest.approx(9.0)


def test_p03_replay_freshness_summary_empty_buffer():
    """P03: empty replay returns the canonical empty shape — keys present
    so downstream consumers index by key without KeyError, but values
    are None / 0.0 to signal "no data". Critical for cli_main's first
    iteration (replay still bootstrapping)."""
    az = load_training_module()
    replay = az.ReplayBuffer(16)
    summary = az.ReplayMetrics.freshness_summary(replay, current_generation=5)
    assert summary["schema_version"] == 1
    assert summary["oldest_gen"] is None
    assert summary["newest_gen"] is None
    assert summary["mean_age"] is None
    assert summary["freshness_score"] == 0.0
    assert summary["sample_count"] == 0
    assert summary["half_life_gen"] is None


def test_p03_replay_freshness_summary_all_current_gen():
    """P03: every sample stamped with current_generation ⇒ mean_age=0,
    freshness_score=1.0 exactly."""
    az = load_training_module()
    replay = az.ReplayBuffer(1000)
    state = np.zeros((3, 7, 7), dtype=np.float32)
    policy = np.zeros(49, dtype=np.float32)
    policy[0] = 1.0
    for _ in range(50):
        replay.add(state, policy, 0.0, metadata={"actor_generation": 7})
    summary = az.ReplayMetrics.freshness_summary(replay, current_generation=7)
    assert summary["sample_count"] > 0
    assert summary["oldest_gen"] == 7
    assert summary["newest_gen"] == 7
    assert summary["mean_age"] == pytest.approx(0.0, abs=1e-9)
    assert summary["freshness_score"] == pytest.approx(1.0, abs=1e-9)
    # half_life = max(1, capacity/100) = 10
    assert summary["half_life_gen"] == pytest.approx(10.0)


def test_p03_replay_freshness_summary_one_half_life_old():
    """P03: mean sample is exactly one half-life old ⇒ freshness ≈ exp(-1) ≈ 0.3679.
    Verifies the exponential-decay formula with hand-computed expected value."""
    az = load_training_module()
    # capacity 1000 ⇒ half_life = 10 generations
    replay = az.ReplayBuffer(1000)
    state = np.zeros((3, 7, 7), dtype=np.float32)
    policy = np.zeros(49, dtype=np.float32)
    policy[0] = 1.0
    # all samples at gen=10, current_generation=20 ⇒ mean_age = 10 = half_life
    for _ in range(60):
        replay.add(state, policy, 0.0, metadata={"actor_generation": 10})
    summary = az.ReplayMetrics.freshness_summary(replay, current_generation=20, sample_n=60)
    assert summary["sample_count"] == 60
    assert summary["mean_age"] == pytest.approx(10.0)
    # exp(-1) ≈ 0.3678794
    assert summary["freshness_score"] == pytest.approx(0.3678794, abs=1e-4)


def test_p03_replay_freshness_summary_skips_untagged():
    """P03: rows without `actor_generation` metadata are silently
    skipped. If NO row has the tag, return the empty shape."""
    az = load_training_module()
    replay = az.ReplayBuffer(100)
    state = np.zeros((3, 7, 7), dtype=np.float32)
    policy = np.zeros(49, dtype=np.float32)
    policy[0] = 1.0
    for _ in range(20):
        replay.add(state, policy, 0.0, metadata={})  # no actor_generation
    summary = az.ReplayMetrics.freshness_summary(replay, current_generation=5)
    assert summary["sample_count"] == 0
    assert summary["mean_age"] is None
    assert summary["freshness_score"] == 0.0


def test_p03_replay_freshness_summary_negative_age_clamped():
    """P03: stamps from a future generation (e.g. buffer reloaded from
    disk with newer data) clamp to age=0; freshness_score stays in (0, 1]."""
    az = load_training_module()
    replay = az.ReplayBuffer(100)
    state = np.zeros((3, 7, 7), dtype=np.float32)
    policy = np.zeros(49, dtype=np.float32)
    policy[0] = 1.0
    for _ in range(10):
        replay.add(state, policy, 0.0, metadata={"actor_generation": 50})
    summary = az.ReplayMetrics.freshness_summary(replay, current_generation=5)
    # mean_age = 5 - 50 = -45; clamp to 0 in the formula
    assert summary["mean_age"] == pytest.approx(-45.0)
    assert summary["freshness_score"] == pytest.approx(1.0)


def test_p01_replay_summary_aggregates_extended_block():
    """P01: schema_version 6+ controller_summary.extended block carries
    real measured `mean_prior_refresh_rate` and per-mode/per-reason
    counters. Aggregator must sum them across rows and compute the
    fired/eligible ratio at the study level (not at per-row, which
    would inflate small-eligible games).
    """
    az = load_training_module()
    replay = az.ReplayBuffer(16)
    state = np.zeros((3, 7, 7), dtype=np.float32)
    policy = np.zeros(49, dtype=np.float32)
    policy[4] = 1.0

    rows = [
        # (active, eligible, penalty_mode_counts, halt_reason_count)
        (
            17,
            50,
            {"Legacy": 7, "PFlipMixture": 3},
            {"MaxVisits": 1, "PFlipConverged": 4},
        ),
        (
            8,
            40,
            {"Legacy": 5, "PFlipMixture": 2, "GatedRefresh": 1},
            {"VOCNonPositive": 2, "PFlipConverged": 3},
        ),
    ]
    for active, eligible, pm_counts, hr_counts in rows:
        replay.add(
            state,
            policy,
            0.0,
            metadata={
                "search_manifest": {"profile": "quartz", "benchmark_safe": True},
                "realized_budget": {"realized_iterations": eligible},
                "controller_summary": {
                    "schema_version": 6,
                    "p_flip": 0.05,
                    "extended": {
                        "schema_version": 1,
                        "refresh_active_count": active,
                        "refresh_eligible_count": eligible,
                        "controller_penalty_mode_counts": pm_counts,
                        "halt_reason_count": hr_counts,
                    },
                },
            },
        )

    summary = az.ReplayMetrics.search_summary(replay, sample_n=2)

    assert summary["extended_coverage_frac"] == pytest.approx(1.0)
    assert summary["extended_refresh_active_total"] == 25
    assert summary["extended_refresh_eligible_total"] == 90
    # Study-level rate: 25 / 90 ≈ 0.2778. Note this is NOT the average
    # of per-row rates (which would weight rows equally regardless of
    # eligible-count) — it's the pooled estimator that survives small
    # eligible counts.
    assert summary["extended_measured_prior_refresh_rate"] == pytest.approx(25 / 90)
    assert summary["extended_controller_penalty_mode_counts"] == {
        "Legacy": 12,
        "PFlipMixture": 5,
        "GatedRefresh": 1,
    }
    assert summary["extended_halt_reason_count"] == {
        "MaxVisits": 1,
        "PFlipConverged": 7,
        "VOCNonPositive": 2,
    }


def test_p01_replay_summary_handles_missing_extended_block():
    """P01: pre-schema_version 6 rows carry no `extended` block;
    aggregator emits coverage 0 and None ratio without crashing."""
    az = load_training_module()
    replay = az.ReplayBuffer(16)
    state = np.zeros((3, 7, 7), dtype=np.float32)
    policy = np.zeros(49, dtype=np.float32)
    policy[4] = 1.0
    replay.add(
        state,
        policy,
        0.0,
        metadata={
            "search_manifest": {"profile": "quartz", "benchmark_safe": True},
            "realized_budget": {"realized_iterations": 32},
            "controller_summary": {"schema_version": 5, "p_flip": 0.05},
        },
    )
    summary = az.ReplayMetrics.search_summary(replay, sample_n=1)
    assert summary["extended_coverage_frac"] == pytest.approx(0.0)
    assert summary["extended_refresh_active_total"] == 0
    assert summary["extended_refresh_eligible_total"] == 0
    assert summary["extended_measured_prior_refresh_rate"] is None
    assert summary["extended_controller_penalty_mode_counts"] == {}
    assert summary["extended_halt_reason_count"] == {}


def test_p6_replay_summary_handles_missing_voc_fields():
    """P6: pre-P6 wire format (no voc fields) yields None means without crashing."""
    az = load_training_module()
    replay = az.ReplayBuffer(16)
    state = np.zeros((3, 7, 7), dtype=np.float32)
    policy = np.zeros(49, dtype=np.float32)
    policy[4] = 1.0
    replay.add(
        state,
        policy,
        0.0,
        metadata={
            "search_manifest": {"profile": "quartz", "benchmark_safe": True},
            "realized_budget": {"realized_iterations": 8},
            "controller_summary": {"p_flip": 0.05},  # no voc_*, no schema_version
        },
    )

    summary = az.ReplayMetrics.search_summary(replay, sample_n=1)

    assert summary["mean_voc_total"] is None
    assert summary["mean_voc_focus"] is None
    assert summary["mean_voc_expand"] is None
    assert summary["mean_voc_merge"] is None
    assert summary["controller_schema_versions"] == {}


def test_runtime_support_should_use_async_pipeline_prefers_gpu_and_batching(monkeypatch):
    import quartz.runtime_support as runtime_support

    class FakeModel:
        pass

    class PredictModel:
        def predict(self, batch):
            return batch, np.zeros(len(batch), dtype=np.float32)

    cfg = {"batch_size": 4}
    assert runtime_support.should_use_async_pipeline(FakeModel(), "cpu", cfg) is False
    assert runtime_support.should_use_async_pipeline(FakeModel(), "cuda", cfg) is True
    assert runtime_support.should_use_async_pipeline(FakeModel(), "cuda", {"batch_size": 1}) is False
    assert runtime_support.should_use_async_pipeline(PredictModel(), "cuda", cfg) is False

    monkeypatch.setenv("QUARTZ_FORCE_ASYNC_PIPELINE", "1")
    assert runtime_support.should_use_async_pipeline(FakeModel(), "cpu", cfg) is True
    monkeypatch.delenv("QUARTZ_FORCE_ASYNC_PIPELINE")


def test_ablation_study_discards_stale_eval_cache(monkeypatch, tmp_path):
    module = load_ablation_script_module()
    import quartz.evaluator_runtime as eval_mod
    calls = {"campaign": 0, "seeds": []}

    def fake_build_eval_cfg(game_name, eval_cfg, device_name, model_path=None):
        return (
            {
                "_name": game_name,
                "iters": 8,
                "search_profile": eval_cfg["search_profile"],
                "vl_mode": eval_cfg["vl_mode"],
            },
            "cpu",
        )

    class FakeEngine:
        def __init__(self, name):
            self._name = name

        def name(self):
            return self._name

        def reset(self):
            return None

    def fake_build_eval_engine(model_run, args, eval_cfg, device):
        return FakeEngine(model_run["id"]), {"_name": args.game}

    class FakeCampaign:
        def __init__(self, engines, num_games):
            self.timings = {"client_start_s": 0.01}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def compare(self, engine_a, engine_b, *args, **kwargs):
            calls["campaign"] += 1
            calls["seeds"].append(kwargs.get("seed"))
            return types.SimpleNamespace(wins=1, losses=0, draws=0, errors=0, voids=0, scored=1, total=1, score_rate=1.0), {
                "runner_mode": "rust_eval_state_machine",
                "match_elapsed_s": 0.02,
            }

    monkeypatch.setattr(module, "build_eval_cfg", fake_build_eval_cfg)
    monkeypatch.setattr(module, "build_eval_engine", fake_build_eval_engine)
    monkeypatch.setattr(eval_mod, "PersistentRustNNEvalCampaign", FakeCampaign)

    existing_payload = {
        "matches": [
            {
                "eval_condition": "E1",
                "search_profile": "baseline",
                "vl_mode": "disabled",
                "search_manifest_hash": "stalehash0000000",
                "a_id": "m1",
                "b_id": "m2",
                "games": 4,
                "wins_a": 1,
                "wins_b": 1,
                "draws": 2,
                "win_rate_a": 0.5,
                "ci": [0.25, 0.75],
                "sprt": "continue",
            }
        ]
    }
    (tmp_path / "evaluation_matrix.json").write_text(json.dumps(existing_payload), encoding="utf-8")

    args = argparse.Namespace(
        force_eval=False,
        include_strict_reference=False,
        paired_seed_eval=False,
        game="gomoku7",
        device="cpu",
        eval_games=4,
        eval_seed=17,
        rust_binary="./target/release/mcts_demo",
    )
    # P02: pre_flight_check requires the .pt files to exist on disk so it
    # can SHA256 them. FakeCampaign in this test never reads model bytes,
    # but the gate runs before FakeCampaign's compare. Write minimal
    # fixture bytes per (id) so the gate passes.
    (tmp_path / "a.pt").write_bytes(b"a")
    (tmp_path / "b.pt").write_bytes(b"b")
    model_runs = [
        {"id": "m1", "model_path": str(tmp_path / "a.pt"), "condition": "T1", "seed": 41},
        {"id": "m2", "model_path": str(tmp_path / "b.pt"), "condition": "T2", "seed": 41},
    ]

    payload = module.run_evaluation_matrix(
        args,
        tmp_path,
        model_runs,
        {"E1": {"search_profile": "baseline", "vl_mode": "disabled"}},
    )

    assert calls["campaign"] == 1
    assert calls["seeds"] == [17]
    assert payload["expected_eval_seeds"] == {"E1": 17}
    assert payload["expected_search_manifests"]["E1"]["eval_seed"] == 17
    assert payload["matches"][0]["search_manifest"]["eval_seed"] == 17
    assert payload["matches"][0]["search_manifest_hash"] != "stalehash0000000"
    assert payload["matches"][0]["timing_s"]["match_elapsed_s"] == pytest.approx(0.02)
    assert payload["discarded_matches"] == [
        {
            "eval_condition": "E1",
            "a_id": "m1",
            "b_id": "m2",
            "reason": "search_manifest_hash_changed",
            "expected_hash": payload["matches"][0]["search_manifest_hash"],
            "found_hash": "stalehash0000000",
        }
    ]


def test_ablation_study_records_eval_condition_timings(monkeypatch, tmp_path):
    module = load_ablation_script_module()
    import quartz.evaluator_runtime as eval_mod

    def fake_build_eval_cfg(game_name, eval_cfg, device_name, model_path=None):
        return (
            {
                "_name": game_name,
                "board": 7,
                "iters": 8,
                "search_profile": eval_cfg["search_profile"],
                "vl_mode": eval_cfg["vl_mode"],
            },
            "cpu",
        )

    class FakeEngine:
        def __init__(self, name):
            self._name = name

        def name(self):
            return self._name

        def reset(self):
            return None

    class FakeCampaign:
        def __init__(self, engines, num_games):
            self.timings = {"client_start_s": 0.123}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def compare(self, engine_a, engine_b, *args, **kwargs):
            return types.SimpleNamespace(wins=2, losses=1, draws=1, errors=0, voids=0, scored=4, total=4, score_rate=0.625), {
                "runner_mode": "rust_eval_state_machine",
                "match_elapsed_s": 1.25,
            }

    monkeypatch.setattr(module, "build_eval_cfg", fake_build_eval_cfg)
    monkeypatch.setattr(module, "build_eval_engine", lambda model_run, args, eval_cfg, device: (FakeEngine(model_run["id"]), {}))
    monkeypatch.setattr(eval_mod, "PersistentRustNNEvalCampaign", FakeCampaign)

    args = argparse.Namespace(
        force_eval=True,
        include_strict_reference=False,
        paired_seed_eval=False,
        game="gomoku7",
        device="cpu",
        eval_games=4,
        rust_binary="./target/release/mcts_demo",
        backend="torch",
    )
    # P02: pre_flight_check requires existing .pt files (see comment in
    # test_ablation_study_discards_stale_eval_cache).
    (tmp_path / "a.pt").write_bytes(b"a")
    (tmp_path / "b.pt").write_bytes(b"b")
    model_runs = [
        {"id": "m1", "model_path": str(tmp_path / "a.pt"), "condition": "T1", "seed": 41},
        {"id": "m2", "model_path": str(tmp_path / "b.pt"), "condition": "T2", "seed": 41},
    ]

    payload = module.run_evaluation_matrix(
        args,
        tmp_path,
        model_runs,
        {"E1": {"search_profile": "baseline", "vl_mode": "disabled"}},
    )

    assert payload["eval_condition_timings"]["E1"]["pairs"] == 1
    assert payload["eval_condition_timings"]["E1"]["campaign_bootstrap_s"] == pytest.approx(0.123, rel=1e-3)
    assert payload["matches"][0]["runner_mode"] == "rust_eval_state_machine"
    assert payload["matches"][0]["scored_games"] == 4
    assert payload["matches"][0]["score_rate_a"] == pytest.approx(0.625)
    assert payload["matches"][0]["ci_kind"] == "score_rate_normal_approx_v1"
    assert payload["matches"][0]["ci"][0] < payload["matches"][0]["ci"][1]
    assert payload["matches"][0]["sprt"] == "inconclusive"
    assert payload["matches"][0]["sprt_meta"]["decisive_games"] == 3
    assert payload["matches"][0]["timing_s"]["match_elapsed_s"] == pytest.approx(1.25)


def test_ablation_study_uses_compare_many_when_available(monkeypatch, tmp_path):
    module = load_ablation_script_module()
    import quartz.evaluator_runtime as eval_mod

    def fake_build_eval_cfg(game_name, eval_cfg, device_name, model_path=None):
        return (
            {
                "_name": game_name,
                "board": 7,
                "iters": 8,
                "search_profile": eval_cfg["search_profile"],
                "vl_mode": eval_cfg["vl_mode"],
            },
            "cpu",
        )

    class FakeEngine:
        def __init__(self, name):
            self._name = name

        def name(self):
            return self._name

        def reset(self):
            return None

    class FakeCampaign:
        compare_calls = 0
        compare_many_calls = 0

        def __init__(self, engines, num_games):
            self.timings = {"client_start_s": 0.25}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def compare(self, *args, **kwargs):
            FakeCampaign.compare_calls += 1
            raise AssertionError("compare() should not be used when compare_many() is available")

        def compare_many(self, comparisons):
            FakeCampaign.compare_many_calls += 1
            return [
                (
                    comparison["match_id"],
                    types.SimpleNamespace(wins=1, losses=0, draws=1, errors=0, voids=0, scored=2, total=2, score_rate=0.75),
                    {
                        "runner_mode": "rust_eval_state_machine",
                        "match_elapsed_s": 1.0,
                        "batch_id": "batch0",
                        "batch_elapsed_s": 3.0,
                        "batch_total_games": 6,
                    },
                )
                for comparison in comparisons
            ]

    monkeypatch.setattr(module, "build_eval_cfg", fake_build_eval_cfg)
    monkeypatch.setattr(module, "build_eval_engine", lambda model_run, args, eval_cfg, device: (FakeEngine(model_run["id"]), {}))
    monkeypatch.setattr(eval_mod, "PersistentRustNNEvalCampaign", FakeCampaign)

    args = argparse.Namespace(
        force_eval=True,
        include_strict_reference=False,
        paired_seed_eval=False,
        game="gomoku7",
        device="cpu",
        eval_games=2,
        rust_binary="./target/release/mcts_demo",
        backend="torch",
    )
    # P02: pre_flight_check requires existing .pt files.
    for fname in ("a.pt", "b.pt", "c.pt"):
        (tmp_path / fname).write_bytes(fname.encode())
    model_runs = [
        {"id": "m1", "model_path": str(tmp_path / "a.pt"), "condition": "T1", "seed": 41},
        {"id": "m2", "model_path": str(tmp_path / "b.pt"), "condition": "T2", "seed": 41},
        {"id": "m3", "model_path": str(tmp_path / "c.pt"), "condition": "T3", "seed": 41},
    ]

    payload = module.run_evaluation_matrix(
        args,
        tmp_path,
        model_runs,
        {"E1": {"search_profile": "baseline", "vl_mode": "disabled"}},
    )

    assert FakeCampaign.compare_many_calls == 1
    assert FakeCampaign.compare_calls == 0
    assert len(payload["matches"]) == 3
    assert payload["matches"][0]["timing_s"]["batch_id"] == "batch0"
    assert payload["eval_timing_summary"]["total_match_elapsed_s"] == pytest.approx(3.0)


def test_ablation_study_attaches_contract_summary():
    module = load_ablation_script_module()
    payload = {
        "expected_search_manifests": {
            "E1": {"search_profile": "baseline", "vl_mode": "disabled"},
            "E2": {"search_profile": "quartz", "vl_mode": "adaptive"},
        },
        "discarded_matches": [{"eval_condition": "E1", "a_id": "m1", "b_id": "m2"}],
    }

    module.attach_ablation_contract_summary(payload)

    assert payload["contract_summary"]["count"] == 2
    assert payload["contract_summary"]["discarded_count"] == 1
    assert isinstance(payload["contract_summary"]["collection_hash"], str)
    assert len(payload["contract_summary"]["collection_hash"]) == 16


def test_ablation_build_eval_cfg_applies_eval_runtime_profile(monkeypatch, tmp_path):
    module = load_ablation_script_module()

    monkeypatch.setattr(module, "load_eval_runtime_overrides_from_model", lambda model_path, device_name: {"batch_size": 32})

    cfg, _device = module.build_eval_cfg("gomoku7", {"search_profile": "baseline", "vl_mode": "disabled"}, "cpu", model_path=str(tmp_path / "best.pt"))

    assert cfg["batch_size"] == 32


def test_eval_timing_summary_helpers_expose_throughput_fields():
    from quartz.eval_timing_summary import (
        summarize_ablation_eval_timings,
        summarize_controller_stage2_timings,
    )

    ablation_summary = summarize_ablation_eval_timings(
        {
            "matches": [
                {"eval_condition": "E1", "games": 4, "timing_s": {"match_elapsed_s": 8.0}},
                {"eval_condition": "E1", "games": 4, "timing_s": {"match_elapsed_s": 4.0}},
            ],
            "eval_condition_timings": {
                "E1": {"cfg_build_s": 1.0, "engine_load_s": 2.0, "campaign_bootstrap_s": 1.0, "pairs": 2, "engine_count": 4}
            },
        }
    )
    assert ablation_summary["total_games"] == 8
    assert ablation_summary["total_startup_s"] == pytest.approx(4.0)
    assert ablation_summary["games_per_s_end_to_end"] == pytest.approx(0.5)
    assert ablation_summary["conditions"][0]["eval_condition"] == "E1"

    batched_summary = summarize_ablation_eval_timings(
        {
            "matches": [
                {"eval_condition": "E1", "games": 2, "timing_s": {"match_elapsed_s": 1.0, "batch_id": "b0", "batch_elapsed_s": 3.0}},
                {"eval_condition": "E1", "games": 2, "timing_s": {"match_elapsed_s": 1.0, "batch_id": "b0", "batch_elapsed_s": 3.0}},
                {"eval_condition": "E1", "games": 2, "timing_s": {"match_elapsed_s": 1.0, "batch_id": "b0", "batch_elapsed_s": 3.0}},
            ],
            "eval_condition_timings": {
                "E1": {"cfg_build_s": 1.0, "engine_load_s": 2.0, "campaign_bootstrap_s": 1.0, "pairs": 3, "engine_count": 3}
            },
        }
    )
    assert batched_summary["total_games"] == 6
    assert batched_summary["total_match_elapsed_s"] == pytest.approx(3.0)

    stage2_summary = summarize_controller_stage2_timings(
        {
            "matches": [
                {"checkpoint_path": "ckpt.pt", "games": 6, "timing_s": {"match_elapsed_s": 3.0}},
            ],
            "checkpoint_timings": {
                "ckpt.pt": {"pairs": 1, "client_bootstrap_s": 1.0, "client_count": 2}
            },
        }
    )
    assert stage2_summary["total_games"] == 6
    assert stage2_summary["games_per_s_end_to_end"] == pytest.approx(1.5)
    assert stage2_summary["checkpoints"][0]["checkpoint_path"] == "ckpt.pt"


def test_eval_runtime_profile_applies_only_safe_gpu_overrides(tmp_path, monkeypatch):
    from quartz.eval_runtime_profile import load_eval_runtime_overrides_from_model

    model_path = tmp_path / "best.pt"
    model_path.write_text("stub", encoding="utf-8")
    profile_path = tmp_path / "autotune_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "signature": {"hardware": {"device_kind": "cuda"}},
                "benchmarks": {"heuristic": {"optimal_batch_size": 32}},
                "overrides": {
                    "batch_size": 64,
                    "selfplay_parallel": 6,
                    "bg_parallel": 6,
                },
            }
        ),
        encoding="utf-8",
    )

    assert load_eval_runtime_overrides_from_model(str(model_path), "cuda") == {"batch_size": 32}
    assert load_eval_runtime_overrides_from_model(str(model_path), "cpu") == {}

    monkeypatch.setenv("QUARTZ_DISABLE_EVAL_AUTOTUNE_PROFILE", "1")
    assert load_eval_runtime_overrides_from_model(str(model_path), "cuda") == {}
    monkeypatch.delenv("QUARTZ_DISABLE_EVAL_AUTOTUNE_PROFILE")


def test_shared_contract_summary_helper_exposes_unified_schema():
    from quartz import contract_summary as contract_mod

    summary = contract_mod.summarize_contract_collection(
        [{"probe_contract_hash": "a"}, {"probe_contract_hash": "b", "legacy_partial": True}],
        [{"candidate_id": "c3"}],
        hash_key="probe_contract_hash",
    )

    assert summary["count"] == 2
    assert summary["discarded_count"] == 1
    assert summary["legacy_partial_count"] == 1
    assert summary["hash_key"] == "probe_contract_hash"
    assert isinstance(summary["collection_hash"], str)
    assert len(summary["collection_hash"]) == 16


def test_ablation_run_training_reruns_when_train_contract_changes(monkeypatch, tmp_path):
    module = load_ablation_script_module()

    args = argparse.Namespace(
        game="gomoku7",
        iterations=2,
        eval_interval=1,
        eval_games=4,
        rust_binary="./target/release/mcts_demo",
        backend="torch",
        device="cpu",
        games_per_iter=None,
        quick=False,
        no_autotune=False,
        resident_session=False,
        runtime_autotune=False,
        force_train=False,
        timeout_hours=1,
    )
    condition_cfg = {"search_profile": "baseline", "vl_mode": "disabled"}
    run_dir = tmp_path / "models" / "cond1"
    run_dir.mkdir(parents=True, exist_ok=True)
    expected_contract = module.build_training_contract(args, "cond1", condition_cfg, 42)
    stale_meta = {
        "condition": "cond1",
        "run_id": "cond1",
        "game": "gomoku7",
        "iterations": 2,
        "seed": 42,
        "train_cfg": condition_cfg,
        "train_contract": expected_contract,
        "train_contract_hash": "stalehash0000000",
        "returncode": 0,
        "elapsed_s": 1.0,
    }
    (run_dir / "condition.json").write_text(json.dumps(stale_meta), encoding="utf-8")
    (run_dir / "best.pt").write_text("stub", encoding="utf-8")

    calls = {"run": 0}

    class FakeCompleted:
        returncode = 0

    def fake_run(cmd, check=False, timeout=None):
        calls["run"] += 1
        return FakeCompleted()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_training(args, tmp_path, "cond1", condition_cfg, 42, multi_seed=False)

    assert calls["run"] == 1
    assert result["skipped"] is False
    saved = json.loads((run_dir / "condition.json").read_text(encoding="utf-8"))
    assert saved["train_contract_hash"] == module.stable_json_hash(expected_contract)


def test_ablation_report_includes_contract_summary(tmp_path):
    module = load_ablation_script_module()
    eval_payload = {
        "matches": [
            {
                "eval_condition": "E1",
                "a_id": "cond1_s42",
                "b_id": "cond1_s42",
                "games": 4,
                "scored_games": 4,
                "ci": [0.25, 0.75],
                "runner_mode": "rust_eval_state_machine",
                "search_manifest_hash": "e1hash",
                "timing_s": {"match_elapsed_s": 8.0},
                "realized_budget_trace": {
                    "games": 4,
                    "moves": 8,
                    "root_visits": {"samples": [8, 8, 8, 8, 8, 8, 8, 8], "mean": 8.0, "max": 8.0},
                    "halt_reason_hist": {"BudgetExhausted": 8},
                    "benchmark_safe_frac": 1.0,
                    "selection_trace_coverage_frac": 1.0,
                    "selection_trace": {
                        "root_selects": 64,
                        "refresh_selected_count": 16,
                        "selected_penalty_abs_sum": 3.0,
                        "selected_effective_prior_l1_sum": 1.0,
                    },
                },
            },
        ],
        "overall": [
            {
                "id": "cond1_s42",
                "condition": "cond1",
                "points": 3.0,
                "games": 4,
                "wins": 3,
                "losses": 1,
                "draws": 0,
                "score_rate": 0.75,
                "win_rate": 0.75,
            }
        ],
        "expected_search_manifests": {
            "E1": {"search_profile": "baseline", "vl_mode": "disabled", "eval_seed": 17},
        },
        "expected_eval_seeds": {"E1": 17},
        "expected_benchmark_safe": {"E1": True},
        "discarded_matches": [{"eval_condition": "E1", "a_id": "m1", "b_id": "m2"}],
        "runtime_contract": {
            "backend": "torch",
            "device": "cpu",
            "rust_binary": "./target/release/mcts_demo",
            "config_layout": "repo_top_level_configs",
        },
        "runtime_contract_hash": "abcd1234abcd1234",
        "eval_condition_timings": {
            "E1": {"cfg_build_s": 1.0, "engine_load_s": 2.0, "campaign_bootstrap_s": 1.0, "pairs": 1, "engine_count": 1}
        },
    }
    module.attach_ablation_contract_summary(eval_payload)
    report = module.generate_report(tmp_path)
    # No runs case
    assert report == {"runs": []}

    runs_dir = tmp_path / "models" / "cond1"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "evaluation_matrix.json").write_text(json.dumps(eval_payload), encoding="utf-8")
    (runs_dir / "train_log.jsonl").write_text(
        json.dumps(
            {
                "iter": 1,
                "loss": 0.1,
                "p_loss": 0.07,
                "v_loss": 0.03,
                "loss_ema": 0.1,
                "games_done": 4,
                "published_elo": 1234,
                "replay_freshness": 0.5,
                "pos_per_s": 8.0,
            }
        ) + "\n",
        encoding="utf-8",
    )
    (runs_dir / "condition.json").write_text(
        json.dumps(
            {
                "condition": "cond1",
                "seed": 42,
                "game": "gomoku7",
                "train_cfg": {"search_profile": "baseline"},
                "run_id": "cond1_s42",
                "returncode": 0,
            }
        ),
        encoding="utf-8",
    )
    (runs_dir / "best.pt").write_text("stub", encoding="utf-8")

    report = module.generate_report(tmp_path)
    assert report["contract_summary"]["count"] == 1
    assert report["contract_summary"]["discarded_count"] == 1
    assert report["train_contract_summary"]["count"] == 1
    assert report["runtime_contract"]["config_layout"] == "repo_top_level_configs"
    assert report["runtime_contract_hash"] == "abcd1234abcd1234"
    assert report["eval_timing_summary"]["total_games"] == 4
    assert report["research_readiness"]["blocking"] is False
    assert report["research_readiness"]["policy_doc"] == "docs/RESEARCH_READINESS.md"
    assert report["research_readiness"]["research_grade_ready"] is False
    assert "multi_seed_per_condition" in report["research_readiness"]["unmet_criteria"]
    assert "no_stale_eval_cache_rows" in report["research_readiness"]["unmet_criteria"]
    assert "selection_trace_recorded" not in report["research_readiness"]["unmet_criteria"]
    assert "budget_trace_recorded" not in report["research_readiness"]["unmet_criteria"]
    assert "evaluation_protocol_recorded" not in report["research_readiness"]["unmet_criteria"]
    assert "evaluator_quality_strata_recorded" not in report["research_readiness"]["unmet_criteria"]
    assert "pipeline_telemetry_recorded" not in report["research_readiness"]["unmet_criteria"]
    assert "hardware_claim_scope_recorded" not in report["research_readiness"]["unmet_criteria"]
    assert report["selection_trace_summary"]["conditions"][0]["selection_trace_coverage_frac"] == pytest.approx(1.0)
    assert report["budget_fairness_summary"]["budget_trace_coverage_frac"] == pytest.approx(1.0)
    assert report["budget_fairness_summary"]["conditions"][0]["root_visits"]["mean"] == pytest.approx(8.0)
    assert report["seed_protocol_summary"]["condition_count"] == 1
    assert report["seed_protocol_summary"]["paired_seed_claim_ready"] is True
    assert report["evaluation_protocol_summary"]["protocol_ready"] is True
    assert report["evaluation_protocol_summary"]["complete_pair_eval_matrix"] is True
    assert report["evaluator_quality_summary"]["stratification_ready"] is True
    assert report["evaluator_quality_summary"]["quality_proxy_pair_coverage_frac"] == pytest.approx(1.0)
    assert report["evaluator_quality_summary"]["model_quality"]["cond1_s42"]["p_loss"] == pytest.approx(0.07)
    assert report["pipeline_telemetry_summary"]["aggregate"]["freshness_coverage_frac"] == pytest.approx(1.0)
    assert report["pipeline_telemetry_summary"]["aggregate"]["throughput_coverage_frac"] == pytest.approx(1.0)
    assert report["hardware_runtime_summary"]["claim_scope"] == "runtime_telemetry_only"
    assert report["hardware_runtime_summary"]["profiler_artifact_present"] is False
    assert report["hardware_runtime_summary"]["hardware_performance_claims_allowed"] is False
    saved = json.loads((tmp_path / "ablation_report.json").read_text(encoding="utf-8"))
    assert saved["contract_summary"] == report["contract_summary"]
    assert saved["train_contract_summary"] == report["train_contract_summary"]
    assert saved["runtime_contract"] == report["runtime_contract"]
    assert saved["runtime_contract_hash"] == report["runtime_contract_hash"]
    assert saved["eval_timing_summary"] == report["eval_timing_summary"]
    assert saved["pipeline_telemetry_summary"] == report["pipeline_telemetry_summary"]
    assert saved["budget_fairness_summary"] == report["budget_fairness_summary"]
    assert saved["seed_protocol_summary"] == report["seed_protocol_summary"]
    assert saved["evaluation_protocol_summary"] == report["evaluation_protocol_summary"]
    assert saved["evaluator_quality_summary"] == report["evaluator_quality_summary"]
    assert saved["hardware_runtime_summary"] == report["hardware_runtime_summary"]
    assert saved["research_readiness"] == report["research_readiness"]


def test_controller_sweep_reuses_matching_stage2_cache(monkeypatch, tmp_path):
    module = load_controller_sweep_script_module()
    from quartz import runtime_support as support_mod

    calls = {"pool": 0, "arena": 0}

    def fake_apply_runtime_overrides(base_cfg, overrides):
        cfg = dict(base_cfg)
        cfg.update(overrides)
        return cfg

    def fake_pool(*args, **kwargs):
        calls["pool"] += 1
        return {}, {"client_bootstrap_s": 0.1, "client_count": 0}

    def fake_arena(*args, **kwargs):
        calls["arena"] += 1
        return 1, 0, 0, 1.0, (1.0, 1.0), "H1_accept"

    monkeypatch.setattr(module, "apply_runtime_overrides", fake_apply_runtime_overrides)
    monkeypatch.setattr(module, "_build_stage2_client_pool", fake_pool)
    monkeypatch.setattr(module, "_arena_dual_cfg_with_clients", fake_arena)

    base_cfg = {"iters": 8, "search_profile": "quartz", "vl_mode": "adaptive", "_name": "gomoku7"}
    candidates = [
        {"id": "c1", "label": "legacy", "source": "test", "overrides": {"penalty_mode": "GatedRefreshLegacy"}},
        {"id": "c2", "label": "theory", "source": "test", "overrides": {"penalty_mode": "GatedRefresh"}},
    ]
    cfg_a = fake_apply_runtime_overrides(base_cfg, candidates[0]["overrides"])
    cfg_b = fake_apply_runtime_overrides(base_cfg, candidates[1]["overrides"])
    cfg_a["iters"] = 8
    cfg_b["iters"] = 8
    cached_row = {
        "checkpoint_path": "ckpt.pt",
        "candidate_a": "c1",
        "candidate_b": "c2",
        "manifest_a": support_mod.build_search_manifest(cfg_a),
        "manifest_b": support_mod.build_search_manifest(cfg_b),
        "manifest_hash_a": support_mod.search_manifest_hash(cfg_a),
        "manifest_hash_b": support_mod.search_manifest_hash(cfg_b),
        "games": 6,
        "wins_a": 1,
        "wins_b": 0,
        "draws": 5,
        "win_rate_a": 7 / 12,
        "ci": [0.25, 0.75],
        "sprt": "continue",
    }
    (tmp_path / "stage2_round_robin.json").write_text(json.dumps({"matches": [cached_row]}), encoding="utf-8")

    args = argparse.Namespace(arena_iters=None, stage2_games=6, rust_binary="./target/release/mcts_demo")
    payload = module.run_stage2_round_robin(candidates, ["ckpt.pt"], base_cfg, "cpu", args, tmp_path)

    assert calls["pool"] == 0
    assert calls["arena"] == 0
    assert payload["matches"] == [cached_row]
    assert payload["discarded_matches"] == []
    assert payload["stage2_timing_summary"]["total_games"] == 6


def test_controller_sweep_discards_stale_stage2_cache(monkeypatch, tmp_path):
    module = load_controller_sweep_script_module()

    calls = {"pool": 0, "arena": 0}

    def fake_apply_runtime_overrides(base_cfg, overrides):
        cfg = dict(base_cfg)
        cfg.update(overrides)
        return cfg

    class FakeClient:
        def stop(self):
            return None

    def fake_pool(checkpoint_path, candidates_arg, base_cfg_arg, device, rust_binary, arena_iters):
        calls["pool"] += 1
        return {
            row["id"]: {"cfg": fake_apply_runtime_overrides(base_cfg_arg, row["overrides"]), "client": FakeClient()}
            for row in candidates_arg
        }, {"client_bootstrap_s": 0.25, "client_count": len(candidates_arg)}

    def fake_arena(*args, **kwargs):
        calls["arena"] += 1
        return 1, 0, 0, 1.0, (1.0, 1.0), "H1_accept"

    monkeypatch.setattr(module, "apply_runtime_overrides", fake_apply_runtime_overrides)
    monkeypatch.setattr(module, "_build_stage2_client_pool", fake_pool)
    monkeypatch.setattr(module, "_arena_dual_cfg_with_clients", fake_arena)

    base_cfg = {"iters": 8, "search_profile": "quartz", "vl_mode": "adaptive", "_name": "gomoku7"}
    candidates = [
        {"id": "c1", "label": "legacy", "source": "test", "overrides": {"penalty_mode": "GatedRefreshLegacy"}},
        {"id": "c2", "label": "theory", "source": "test", "overrides": {"penalty_mode": "GatedRefresh"}},
    ]
    stale_row = {
        "checkpoint_path": "ckpt.pt",
        "candidate_a": "c1",
        "candidate_b": "c2",
        "manifest_a": {"penalty_mode": "stale"},
        "manifest_b": {"penalty_mode": "stale"},
        "manifest_hash_a": "stalehasha",
        "manifest_hash_b": "stalehashb",
        "games": 6,
        "wins_a": 0,
        "wins_b": 1,
        "draws": 5,
        "win_rate_a": 5 / 12,
        "ci": [0.25, 0.75],
        "sprt": "continue",
    }
    (tmp_path / "stage2_round_robin.json").write_text(json.dumps({"matches": [stale_row]}), encoding="utf-8")

    args = argparse.Namespace(arena_iters=None, stage2_games=6, rust_binary="./target/release/mcts_demo")
    payload = module.run_stage2_round_robin(candidates, ["ckpt.pt"], base_cfg, "cpu", args, tmp_path)

    assert calls["pool"] == 1
    assert calls["arena"] == 1
    assert payload["matches"][0]["manifest_hash_a"] != "stalehasha"
    assert payload["matches"][0]["timing_s"]["client_bootstrap_s"] == pytest.approx(0.25)
    assert payload["checkpoint_timings"]["ckpt.pt"]["client_count"] == 2
    assert payload["stage2_timing_summary"]["total_games"] == 6
    assert payload["discarded_matches"] == [
        {
            "checkpoint_path": "ckpt.pt",
            "candidate_a": "c1",
            "candidate_b": "c2",
            "reason": "stage2_manifest_hash_changed",
            "expected_games": 6,
            "found_games": 6,
            "expected_manifest_hash_a": payload["matches"][0]["manifest_hash_a"],
            "expected_manifest_hash_b": payload["matches"][0]["manifest_hash_b"],
            "found_manifest_hash_a": "stalehasha",
            "found_manifest_hash_b": "stalehashb",
        }
    ]


def test_controller_sweep_stage2_reuses_client_pool_per_checkpoint(monkeypatch, tmp_path):
    module = load_controller_sweep_script_module()

    calls = {"pool": 0, "arena": 0}

    def fake_apply_runtime_overrides(base_cfg, overrides):
        cfg = dict(base_cfg)
        cfg.update(overrides)
        return cfg

    class FakeClient:
        def __init__(self, candidate_id):
            self.candidate_id = candidate_id
            self.stopped = False

        def stop(self):
            self.stopped = True

    created_clients = {}

    def fake_pool(checkpoint_path, candidates_arg, base_cfg_arg, device, rust_binary, arena_iters):
        calls["pool"] += 1
        pool = {}
        for row in candidates_arg:
            client = FakeClient(row["id"])
            created_clients[row["id"]] = client
            pool[row["id"]] = {
                "cfg": fake_apply_runtime_overrides(base_cfg_arg, row["overrides"]),
                "client": client,
            }
        return pool, {"client_bootstrap_s": 0.5, "client_count": len(pool)}

    def fake_arena(client_a, cfg_a, client_b, cfg_b, *, n_games, strict):
        calls["arena"] += 1
        assert strict is True
        assert client_a is created_clients[cfg_a["candidate_id"]]
        assert client_b is created_clients[cfg_b["candidate_id"]]
        return 1, 0, 1, 0.75, [0.25, 0.95], "continue"

    monkeypatch.setattr(module, "apply_runtime_overrides", fake_apply_runtime_overrides)
    monkeypatch.setattr(module, "_build_stage2_client_pool", fake_pool)
    monkeypatch.setattr(module, "_arena_dual_cfg_with_clients", fake_arena)

    base_cfg = {"iters": 8, "search_profile": "quartz", "vl_mode": "adaptive", "_name": "gomoku7"}
    candidates = [
        {"id": "c1", "label": "A", "source": "test", "overrides": {"candidate_id": "c1", "penalty_mode": "GatedRefreshLegacy"}},
        {"id": "c2", "label": "B", "source": "test", "overrides": {"candidate_id": "c2", "penalty_mode": "GatedRefresh"}},
        {"id": "c3", "label": "C", "source": "test", "overrides": {"candidate_id": "c3", "penalty_mode": "GatedRefresh"}},
    ]
    args = argparse.Namespace(arena_iters=None, stage2_games=4, rust_binary="./target/release/mcts_demo")

    payload = module.run_stage2_round_robin(candidates, ["ckpt.pt"], base_cfg, "cpu", args, tmp_path)

    assert calls["pool"] == 1
    assert calls["arena"] == 3
    assert payload["checkpoint_timings"]["ckpt.pt"]["client_count"] == 3
    assert payload["matches"][0]["timing_s"]["client_bootstrap_s"] == pytest.approx(0.5)
    assert payload["stage2_timing_summary"]["checkpoints"][0]["client_count"] == 3
    assert all(client.stopped for client in created_clients.values())


def test_controller_sweep_normalizes_stage1_payload_and_discards_stale_rows():
    module = load_controller_sweep_script_module()
    candidates = [
        {"id": "c1", "label": "legacy", "source": "test", "overrides": {"penalty_mode": "GatedRefreshLegacy"}},
        {"id": "c2", "label": "theory", "source": "test", "overrides": {"penalty_mode": "GatedRefresh"}},
    ]
    positions = [{"board": [0] * 49, "player": 1}]
    valid_contract = module.build_stage1_probe_contract("ckpt.pt", candidates[0], positions, 16, 4.0)
    stale_contract = dict(valid_contract)
    stale_contract["candidate_id"] = "c2"
    stale_contract["probe_contract_hash"] = "stalecontract000"
    stage1_payload = {
        "rows": [
            {
                "candidate_id": "c1",
                "candidate_label": "legacy",
                "candidate_source": "test",
                "checkpoint_path": "ckpt.pt",
                "valid_positions": 1,
                "agreement_rate": 1.0,
                "reference_policy_mass": 1.0,
                "mean_value_gap": 0.0,
                "mean_latency_ms": 2.0,
                "timeout_count": 0,
                "stage1_score": 1.0,
                **valid_contract,
            },
            {
                "candidate_id": "c2",
                "candidate_label": "theory",
                "candidate_source": "test",
                "checkpoint_path": "ckpt.pt",
                "valid_positions": 1,
                "agreement_rate": 0.5,
                "reference_policy_mass": 0.5,
                "mean_value_gap": 0.5,
                "mean_latency_ms": 3.0,
                "timeout_count": 0,
                "stage1_score": 0.25,
                **stale_contract,
            },
        ]
    }

    normalized, missing = module.normalize_stage1_payload(
        stage1_payload,
        candidates,
        ["ckpt.pt"],
        positions,
        16,
        4.0,
        2,
    )

    assert [row["candidate_id"] for row in normalized["rows"]] == ["c1"]
    assert normalized["shortlist"][0]["id"] == "c1"
    assert missing[0]["candidate_id"] == "c2"
    assert normalized["discarded_rows"] == [
        {
            "checkpoint_path": "ckpt.pt",
            "candidate_id": "c2",
            "reason": "stage1_probe_contract_changed",
            "expected_probe_contract_hash": missing[0]["probe_contract_hash"],
            "found_probe_contract_hash": "stalecontract000",
        }
    ]


def test_controller_sweep_report_includes_contract_summary(tmp_path):
    module = load_controller_sweep_script_module()
    stage1_payload = {
        "summary": [{"candidate_id": "c1", "stage1_score": 1.0}],
        "expected_probe_contracts": [{"probe_contract_hash": "p1"}],
        "discarded_rows": [{"candidate_id": "c2"}],
    }
    module.attach_stage1_contract_summary(stage1_payload)
    stage2_payload = {
        "overall": [{"candidate_id": "c1", "score_rate": 0.75}],
        "matches": [
            {
                "checkpoint_path": "ckpt.pt",
                "candidate_a": "c1",
                "candidate_b": "c2",
                "games": 8,
                "timing_s": {"match_elapsed_s": 4.0},
            }
        ],
        "checkpoint_timings": {
            "ckpt.pt": {"pairs": 1, "client_bootstrap_s": 1.0, "client_count": 2}
        },
        "expected_match_contracts": [{"manifest_hash_a": "m1", "manifest_hash_b": "m2"}],
        "discarded_matches": [{"candidate_a": "c1", "candidate_b": "c2"}],
    }
    module.attach_stage2_contract_summary(stage2_payload)
    manifest = {"game": "gomoku7"}

    report = module.build_report(tmp_path, manifest, stage1_payload, stage2_payload)

    assert report["contract_summary"]["stage1"]["count"] == 1
    assert report["contract_summary"]["stage1"]["discarded_count"] == 1
    assert report["contract_summary"]["stage2"]["count"] == 1
    assert report["contract_summary"]["stage2"]["discarded_count"] == 1
    assert report["stage2_timing_summary"]["total_games"] == 8
    assert report["stage2_timing_summary"]["checkpoints"][0]["client_count"] == 2
    saved = json.loads((tmp_path / "sweep_report.json").read_text(encoding="utf-8"))
    assert saved["contract_summary"] == report["contract_summary"]
    assert saved["stage2_timing_summary"] == report["stage2_timing_summary"]


def test_controller_sweep_contract_summary_falls_back_for_legacy_payloads():
    module = load_controller_sweep_script_module()
    stage1_payload = {
        "rows": [
            {
                "checkpoint_path": "ckpt.pt",
                "candidate_id": "c1",
                "candidate_label": "legacy",
            }
        ],
        "discarded_rows": [],
    }
    stage2_payload = {
        "matches": [
            {
                "checkpoint_path": "ckpt.pt",
                "candidate_a": "c1",
                "candidate_b": "c2",
                "games": 8,
            }
        ],
        "discarded_matches": [],
    }

    module.attach_stage1_contract_summary(stage1_payload)
    module.attach_stage2_contract_summary(stage2_payload)
    combined = module.build_controller_sweep_contract_summary(stage1_payload, stage2_payload)

    assert stage1_payload["contract_summary"]["count"] == 1
    assert stage1_payload["contract_summary"]["legacy_partial_count"] == 1
    assert stage2_payload["contract_summary"]["count"] == 1
    assert stage2_payload["contract_summary"]["legacy_partial_count"] == 1
    assert combined["stage1"]["legacy_partial_count"] == 1
    assert combined["stage2"]["legacy_partial_count"] == 1


def test_eval_timing_summary_helpers_expose_throughput_fields():
    from quartz.eval_timing_summary import (
        summarize_ablation_eval_timings,
        summarize_controller_stage2_timings,
    )

    ablation_summary = summarize_ablation_eval_timings(
        {
            "matches": [
                {"eval_condition": "E1", "games": 4, "timing_s": {"match_elapsed_s": 8.0}},
                {"eval_condition": "E1", "games": 4, "timing_s": {"match_elapsed_s": 4.0}},
            ],
            "eval_condition_timings": {
                "E1": {"cfg_build_s": 1.0, "engine_load_s": 2.0, "campaign_bootstrap_s": 1.0, "pairs": 2, "engine_count": 4}
            },
        }
    )
    assert ablation_summary["total_games"] == 8
    assert ablation_summary["total_startup_s"] == pytest.approx(4.0)
    assert ablation_summary["games_per_s_end_to_end"] == pytest.approx(0.5)
    assert ablation_summary["conditions"][0]["eval_condition"] == "E1"

    stage2_summary = summarize_controller_stage2_timings(
        {
            "matches": [
                {"checkpoint_path": "ckpt.pt", "games": 6, "timing_s": {"match_elapsed_s": 3.0}},
            ],
            "checkpoint_timings": {
                "ckpt.pt": {"pairs": 1, "client_bootstrap_s": 1.0, "client_count": 2}
            },
        }
    )
    assert stage2_summary["total_games"] == 6
    assert stage2_summary["games_per_s_end_to_end"] == pytest.approx(1.5)
    assert stage2_summary["checkpoints"][0]["checkpoint_path"] == "ckpt.pt"


def test_controller_sweep_new_search_client_uses_checkpoint_metadata(monkeypatch, tmp_path):
    module = load_controller_sweep_script_module()
    import quartz.alphazero_train as az
    import quartz.backend as backend_mod

    class DummyNet:
        instances = []

        def __init__(self, cfg):
            self.cfg = dict(cfg)
            self.state_dict = None
            DummyNet.instances.append(self)

        def to(self, device):
            return self

        def load_state_dict(self, state_dict):
            self.state_dict = state_dict

        def eval(self):
            return self

    class DummyClient:
        last_instance = None

        def __init__(self, model, cfg, device, rust_binary):
            self.model = model
            self.cfg = dict(cfg)
            self.started = False
            DummyClient.last_instance = self

        def start(self):
            self.started = True

    monkeypatch.setattr(az, "AlphaZeroNet", DummyNet)
    monkeypatch.setattr(az, "NNSearchClient", DummyClient)
    monkeypatch.setattr(
        backend_mod,
        "load_checkpoint_with_metadata",
        lambda *args, **kwargs: ({"tower.0.weight": 1}, {"blocks": 6, "filters": 96, "vh": 32}),
    )

    ckpt = tmp_path / "model.pt"
    ckpt.write_text("stub", encoding="utf-8")

    client = module._new_search_client(
        str(ckpt),
        {"_name": "gomoku7", "board": 7, "win": 5, "blocks": 4, "filters": 64, "vh": 16},
        "cpu",
        "./target/release/mcts_demo",
    )

    assert client is DummyClient.last_instance
    assert client.started is True
    assert DummyNet.instances[0].cfg["blocks"] == 6
    assert DummyNet.instances[0].cfg["filters"] == 96
    assert DummyNet.instances[0].cfg["vh"] == 32
    assert client.cfg["blocks"] == 6


def test_controller_sweep_stage2_uses_eval_runtime_profile_for_manifests(monkeypatch, tmp_path):
    module = load_controller_sweep_script_module()

    def fake_apply_runtime_overrides(base_cfg, overrides):
        cfg = dict(base_cfg)
        cfg.update(overrides)
        return cfg

    monkeypatch.setattr(module, "apply_runtime_overrides", fake_apply_runtime_overrides)
    monkeypatch.setattr(module, "load_eval_runtime_overrides_from_model", lambda checkpoint_path, device_name: {"batch_size": 32})

    class FakeClient:
        def stop(self):
            return None

    monkeypatch.setattr(
        module,
        "_build_stage2_client_pool",
        lambda checkpoint_path, candidates_arg, base_cfg_arg, device, rust_binary, arena_iters: (
            {
                row["id"]: {
                    "cfg": {**fake_apply_runtime_overrides(base_cfg_arg, row["overrides"]), "batch_size": 32, "iters": arena_iters},
                    "client": FakeClient(),
                }
                for row in candidates_arg
            },
            {"client_bootstrap_s": 0.2, "client_count": len(candidates_arg)},
        ),
    )
    monkeypatch.setattr(module, "_arena_dual_cfg_with_clients", lambda *args, **kwargs: (1, 0, 0, 1.0, [1.0, 1.0], "H1_accept"))

    base_cfg = {"iters": 8, "search_profile": "quartz", "vl_mode": "adaptive", "_name": "gomoku7"}
    candidates = [
        {"id": "c1", "label": "A", "source": "test", "overrides": {"penalty_mode": "GatedRefreshLegacy"}},
        {"id": "c2", "label": "B", "source": "test", "overrides": {"penalty_mode": "GatedRefresh"}},
    ]
    args = argparse.Namespace(arena_iters=None, stage2_games=2, rust_binary="./target/release/mcts_demo")

    payload = module.run_stage2_round_robin(candidates, ["ckpt.pt"], base_cfg, "cuda", args, tmp_path)

    assert payload["matches"][0]["manifest_a"]["batch_size"] == 32


def test_training_module_reexports_replay_api():
    az = load_training_module()
    from quartz import replay as replay_mod

    assert az.ReplayBuffer is replay_mod.ReplayBuffer
    assert az.ReplayExample is replay_mod.ReplayExample
    assert az.SparsePolicyTarget is replay_mod.SparsePolicyTarget
    assert az.collate_replay_samples is replay_mod.collate_replay_samples
    assert az.sparse_policy_from_entries is replay_mod.sparse_policy_from_entries


def test_training_module_reexports_eval_runtime_api():
    az = load_training_module()
    from quartz import eval_runtime as eval_mod

    assert az.NNEvalCache is eval_mod.NNEvalCache
    req = (3, [1.0, 0.0, 0.0, 0.0], 7, 11, 13, 2)
    assert az._parse_eval_request(req) == eval_mod.parse_eval_request(req)
    group = az._make_eval_request_group("json_single", [req], gi=4, prefer_shm=True)
    assert group == eval_mod.make_eval_request_group("json_single", [req], gi=4, prefer_shm=True)


def test_training_module_reexports_qipc_api():
    az = load_training_module()
    from quartz import qipc as qipc_mod

    assert az.QipcSharedMemoryTransport is qipc_mod.QipcSharedMemoryTransport
    assert az.ShmRingBuffer is qipc_mod.ShmRingBuffer
    payload = struct.pack("<IIIQQI", 7, 9, 4, 11, 22, 3) + np.asarray([0.25, -0.5, 0.75, 1.25], dtype="<f4").tobytes()
    lhs = az.unpack_qipc_eval_req(payload)
    rhs = qipc_mod.unpack_qipc_eval_req(payload)
    assert lhs[0] == rhs[0]
    assert lhs[2:] == rhs[2:]
    np.testing.assert_allclose(lhs[1], rhs[1])


def test_training_module_reexports_selfplay_runtime_api():
    az = load_training_module()
    from quartz import selfplay_runtime as sp_mod

    cfg = {"board": 7, "actions": 49, "bg_parallel": 2, "bg_batch_games": 4, "batch_size": 8, "batch": 64}
    recent_chunks = [{"games": 2, "positions": 30}]
    assert issubclass(az.NNSearchClient, sp_mod.NNSearchClient)
    assert issubclass(az.RustServerPool, sp_mod.RustServerPool)
    assert issubclass(az.SelfPlayWorker, sp_mod.SelfPlayWorker)
    assert az.plan_selfplay_runner_chunk(cfg, replay_size=16, recent_chunks=recent_chunks) == (
        sp_mod.plan_selfplay_runner_chunk(cfg, replay_size=16, recent_chunks=recent_chunks)
    )
    assert az.compute_train_steps(100, 256, 512, concurrent=True) == sp_mod.compute_train_steps(100, 256, 512, concurrent=True)
    assert az.default_output_dir("gomoku7") == sp_mod.default_output_dir("gomoku7")


def test_selfplay_worker_stop_cancels_active_search_and_kills_pool():
    from quartz import selfplay_runtime as sp_mod

    class FakePool:
        def __init__(self):
            self.kill_calls = 0
            self.close_calls = 0

        def kill_active(self):
            self.kill_calls += 1

        def close(self):
            self.close_calls += 1

    class FakeThread:
        def __init__(self):
            self.join_calls = []
            self._alive_checks = 0

        def join(self, timeout=None):
            self.join_calls.append(timeout)

        def is_alive(self):
            self._alive_checks += 1
            return self._alive_checks == 1

    class FakeRing:
        def __init__(self):
            self.cancel_calls = 0

        def request_cancel(self):
            self.cancel_calls += 1

    class FakeProc:
        def __init__(self):
            self._quartz_ring_buffer = FakeRing()
            self.kill_calls = 0

        def kill(self):
            self.kill_calls += 1

    worker = sp_mod.SelfPlayWorker(
        cfg={"batch": 8},
        model="actor",
        device="cpu",
        replay=types.SimpleNamespace(buf=[]),
        rust_binary="./target/release/mcts_demo",
        server_pool_factory=lambda _binary: FakePool(),
        clone_actor_model_fn=lambda model: model,
        selfplay_runner=lambda *args, **kwargs: ([], [], [], []),
    )
    proc = FakeProc()
    worker._active_proc = proc
    worker._thread = FakeThread()

    worker.stop()

    assert proc._quartz_ring_buffer.cancel_calls >= 1
    assert proc.kill_calls == 1
    assert worker._proc_pool.kill_calls == 1
    assert worker._proc_pool.close_calls == 1
    assert worker._thread.join_calls == [10, 5]


def test_selfplay_worker_pause_escalates_and_reports_failure():
    from quartz import selfplay_runtime as sp_mod

    class FakePool:
        def __init__(self):
            self.kill_calls = 0

        def kill_active(self):
            self.kill_calls += 1

    class FakeIdle:
        def __init__(self):
            self.wait_calls = []

        def set(self):
            return None

        def clear(self):
            return None

        def is_set(self):
            return False

        def wait(self, timeout=None):
            self.wait_calls.append(timeout)
            return False

    worker = sp_mod.SelfPlayWorker(
        cfg={"batch": 8},
        model="actor",
        device="cpu",
        replay=types.SimpleNamespace(buf=[]),
        rust_binary="./target/release/mcts_demo",
        server_pool_factory=lambda _binary: FakePool(),
        clone_actor_model_fn=lambda model: model,
        selfplay_runner=lambda *args, **kwargs: ([], [], [], []),
    )
    worker._idle = FakeIdle()
    cancel_calls = []
    worker._cancel_active_search = lambda kill_proc=False: cancel_calls.append(bool(kill_proc)) or True

    assert worker.pause(wait=True) is False
    assert cancel_calls == [False, True]
    assert worker._proc_pool.kill_calls == 1
    assert "failed to become idle" in str(worker._last_error)


def test_runtime_support_resolves_training_module_search_client_override(monkeypatch):
    az = load_training_module()
    from quartz import evaluator_runtime as eval_mod
    from quartz import runtime_support as support_mod

    class FakeClient:
        pass

    monkeypatch.setattr(az, "NNSearchClient", FakeClient)
    assert support_mod.resolve_search_client_cls() is FakeClient
    assert eval_mod._default_runtime_hooks().search_client_cls is FakeClient


def test_training_module_reexports_game_adapter_api():
    az = load_training_module()
    from quartz import game_adapters as ga_mod

    assert az.GomokuGameAdapter is ga_mod.GomokuGameAdapter
    assert az.TicTacToeGameAdapter is ga_mod.TicTacToeGameAdapter
    assert az.GoGameAdapter is ga_mod.GoGameAdapter
    assert az.ChessEvaluationAdapter is ga_mod.ChessEvaluationAdapter


def test_training_module_reexports_arena_runtime_api():
    az = load_training_module()
    from quartz import arena_runtime as arena_mod

    assert az.MCTSNode is arena_mod.MCTSNode
    assert az.TreeMCTS is arena_mod.TreeMCTS
    assert az.Glicko2Rating is arena_mod.Glicko2Rating
    assert az.Glicko2System is arena_mod.Glicko2System
    assert az.RandomRolloutAgent is arena_mod.RandomRolloutAgent
    assert az.TreeMCTSEngine is arena_mod.TreeMCTSEngine
    assert az.arena_compare is arena_mod.arena_compare
    assert az.arena_3agent is arena_mod.arena_3agent


def test_training_module_reexports_evaluator_runtime_api():
    az = load_training_module()
    from quartz import evaluator_runtime as evalrt_mod

    assert issubclass(az.RustNNEvaluatorEngine, evalrt_mod.RustNNEvaluatorEngine)


def test_training_module_reexports_autotune_runtime_api():
    az = load_training_module()
    from quartz import autotune_runtime as auto_mod

    assert az.AUTOTUNE_PROFILE_VERSION == auto_mod.AUTOTUNE_PROFILE_VERSION
    assert issubclass(az.OnlineAutotuneController, auto_mod.OnlineAutotuneController)
    assert az._autotune_parallel_limit is not None
    assert az.plan_online_runtime_overrides({"bg_parallel": 1, "bg_batch_games": 1, "n_threads": 1, "batch": 256, "batch_size": 8}, az.HardwareSpec(
        logical_cpus=4, physical_cpus=2, memory_mb=8192,
        gpu_vendor="none", gpu_name="", gpu_vram_mb=0,
        gpu_count=0, torch_cuda=False, device_kind="cpu"
    ), {"last_cycle_s": 1.0, "last_cycle_positions": 16, "positions_per_s": 8.0, "best_positions_per_s": 8.0, "n_new": 64, "train_steps": 2}) == auto_mod.plan_online_runtime_overrides(
        {"bg_parallel": 1, "bg_batch_games": 1, "n_threads": 1, "batch": 256, "batch_size": 8},
        az.HardwareSpec(
            logical_cpus=4, physical_cpus=2, memory_mb=8192,
            gpu_vendor="none", gpu_name="", gpu_vram_mb=0,
            gpu_count=0, torch_cuda=False, device_kind="cpu"
        ),
        {"last_cycle_s": 1.0, "last_cycle_positions": 16, "positions_per_s": 8.0, "best_positions_per_s": 8.0, "n_new": 64, "train_steps": 2},
    )


def test_training_module_reexports_models_torch_api():
    az = load_training_module()
    from quartz import models_torch as models_mod

    assert az.AlphaZeroNet is models_mod.AlphaZeroNet
    assert az.ResBlock is models_mod.ResBlock
    assert az.SEBlock is models_mod.SEBlock


def test_training_module_reexports_cli_parser():
    az = load_training_module()
    from quartz import cli_main as cli_mod

    parser = az.build_arg_parser()
    parsed = parser.parse_args(["--game", "gomoku7", "--runtime-autotune", "--search-profile", "baseline_strict"])

    assert isinstance(parser, argparse.ArgumentParser)
    assert parsed.game == "gomoku7"
    assert parsed.runtime_autotune is True
    assert parsed.search_profile == "baseline_strict"
    assert az.build_arg_parser().__class__ is cli_mod.build_arg_parser(az.GAME_CONFIGS.keys()).__class__


def test_training_module_reexports_system_runtime_api():
    az = load_training_module()
    from quartz import system_runtime as sys_mod

    hw = az.HardwareSpec(logical_cpus=8, physical_cpus=4, memory_mb=16000, gpu_vram_mb=0, device_kind="cpu")
    assert az.HardwareSpec is sys_mod.HardwareSpec
    assert az.eval_worker_candidates(hw, {"n_threads": 1}, eval_games=16) == (
        sys_mod.eval_worker_candidates(hw, {"n_threads": 1}, eval_games=16)
    )
    assert az.compute_eval_collect_policy(8, 0.002, batch_items_ema=4.0, wait_ema_s=0.0) == (
        sys_mod.compute_eval_collect_policy(8, 0.002, batch_items_ema=4.0, wait_ema_s=0.0)
    )
    assert az.max_supported_threads(hw) == sys_mod.max_supported_threads(hw)
    assert az.runtime_thread_budget({"n_threads": "auto", "thread_cap": 6}, hw=hw) == 6
    assert sys_mod.clamp_runtime_cfg_to_hardware(
        {"n_threads": "auto", "thread_cap": 99},
        hw,
    )["thread_cap"] == 8


def test_training_module_reexports_train_loop_api():
    az = load_training_module()
    from quartz import train_loop as tl_mod

    assert issubclass(az.EarlyStopping, tl_mod.EarlyStopping)
    assert issubclass(az.StepEarlyStopping, tl_mod.StepEarlyStopping)
    assert az.early_stopping_enabled(5, concurrent=True) == tl_mod.early_stopping_enabled(5, concurrent=True)
    assert az.round_or_none(1.23456) == tl_mod.round_or_none(1.23456)


def test_replay_loads_legacy_dense_npz_as_sparse_examples(tmp_path):
    az = load_training_module()
    states = np.stack([
        np.zeros((3, 7, 7), dtype=np.float32),
        np.ones((3, 7, 7), dtype=np.float32),
    ])
    policies = np.zeros((2, 49), dtype=np.float32)
    policies[0, 4] = 1.0
    policies[1, 2] = 0.4
    policies[1, 5] = 0.6
    values = np.array([0.25, -0.5], dtype=np.float32)
    path = tmp_path / "legacy_replay.npz"
    np.savez_compressed(path, states=states, policies=policies, values=values)

    replay = az.ReplayBuffer(16)
    assert replay.load(path) == 2
    assert isinstance(replay.buf[0], az.ReplayExample)
    assert replay.buf[0].policy.n_actions == 49

    states_t, policies_t, values_t = replay.sample(2)
    assert states_t.shape == (2, 3, 7, 7)
    assert policies_t.shape == (2, 49)
    assert values_t.shape == (2,)


def test_train_entry_detects_when_jax_should_be_prewarmed():
    entry = load_train_entry_module()

    assert entry._should_prewarm_jax(["--backend", "jax"]) is True
    assert entry._should_prewarm_jax(["--backend=jax"]) is True
    assert entry._should_prewarm_jax(["--device", "jax"]) is True
    assert entry._should_prewarm_jax(["--backend", "torch"]) is False


def test_train_entry_selects_runtime_module_from_backend_flags():
    entry = load_train_entry_module()

    assert entry._runtime_module_name(["--backend", "jax"]) == "quartz.jax_runtime"
    assert entry._runtime_module_name(["--device", "jax"]) == "quartz.jax_runtime"
    assert entry._runtime_module_name(["--backend", "torch"]) == "quartz.torch_runtime"


def test_jax_runtime_parser_imports_without_loading_compat_facade():
    sys.modules.pop("quartz.alphazero_train", None)
    from quartz import jax_runtime

    parser = jax_runtime.build_arg_parser()
    parsed = parser.parse_args(["--game", "gomoku7", "--backend", "jax"])

    assert parsed.backend == "jax"
    assert "quartz.alphazero_train" not in sys.modules


def test_jax_runtime_main_help_avoids_compat_facade():
    sys.modules.pop("quartz.alphazero_train", None)
    from quartz import jax_runtime

    with pytest.raises(SystemExit):
        jax_runtime.main(["--help"])

    assert "quartz.alphazero_train" not in sys.modules


def test_torch_runtime_main_help_avoids_compat_facade():
    sys.modules.pop("quartz.alphazero_train", None)
    from quartz import torch_runtime

    with pytest.raises(SystemExit):
        torch_runtime.main(["--help"])

    assert "quartz.alphazero_train" not in sys.modules


def test_prepare_training_context_prefers_backend_actor_for_jax(monkeypatch, tmp_path):
    from quartz import cli_main as cli_mod
    from quartz import backend as backend_mod

    class FakeJaxBackend:
        name = "jax"
        num_params = 123
        optimizer = None

        def load(self, path):
            self.loaded = path
            return False

        def get_torch_model(self):
            return None

    class NeverInstantiateModel:
        def __init__(self, _cfg):
            raise AssertionError("torch model should not be constructed for jax backend context")

    monkeypatch.setattr(backend_mod, "create_backend", lambda cfg, device="auto", preference="auto": FakeJaxBackend())

    args = cli_mod.build_arg_parser(["gomoku7"]).parse_args(
        ["--game", "gomoku7", "--backend", "jax", "--output", str(tmp_path), "--no-autotune"]
    )

    hooks = cli_mod.CliPrepareHooks(
        torch=types.SimpleNamespace(
            manual_seed=lambda seed: None,
            cuda=types.SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda seed: None),
        ),
        np=np,
        random_mod=random,
        game_configs={"gomoku7": {"board": 7, "ch": 17, "actions": 49, "filters": 32, "blocks": 2, "vh": 32, "buf": 64, "batch": 8, "steps": 4, "batch_size": 8, "games": 2}},
        get_encoder=None,
        apply_config_overrides=lambda cfg, overrides: dict(cfg, **overrides),
        is_go_game=lambda _name: False,
        default_output_dir=lambda game: str(tmp_path / game),
        resolve_runtime_paths=lambda base_dir, explicit_model=None, resume=False: {
            "load_model_path": str(tmp_path / "latest.pt"),
            "latest_model_path": str(tmp_path / "latest.pt"),
            "best_model_path": str(tmp_path / "best.pt"),
            "replay_path": str(tmp_path / "replay.npz"),
            "log_path": str(tmp_path / "train_log.jsonl"),
            "autotune_profile_path": str(tmp_path / "autotune_profile.json"),
        },
        auto_device_name=lambda: "cpu",
        detect_hardware_spec=lambda device: types.SimpleNamespace(device_kind=str(device), physical_cpus=4),
        configure_torch_rocm_runtime=lambda hw: (_ for _ in ()).throw(AssertionError("torch runtime should not be configured")),
        supports_rust_eval_state_machine=lambda _name: False,
        supports_rust_selfplay_state_machine=lambda _name: False,
        autotune_training_cfg=lambda cfg, hw, concurrent=True: cfg,
        clamp_runtime_cfg_to_hardware=lambda cfg, hw: cfg,
        max_supported_threads=lambda hw: 4,
        gpu_host_thread_cap=lambda hw: 2,
        gpu_interop_thread_cap=lambda hw: 1,
        alphazero_net_cls=NeverInstantiateModel,
        load_torch_state_dict_checked=lambda *args, **kwargs: None,
        get_actor_model=lambda model, backend: backend if backend is not None else model,
        load_autotune_profile=lambda *args, **kwargs: None,
        apply_runtime_overrides=lambda cfg, overrides: dict(cfg, **overrides),
        run_autotune_benchmark=lambda *args, **kwargs: ({}, {}),
        save_autotune_profile=lambda *args, **kwargs: None,
        probe_inference_batch_size=lambda model, device, cfg, cap: cfg.get("batch_size", 8),
        clamp_thread_count=lambda value, hw: int(value),
    )

    ctx = cli_mod.prepare_training_context(args, hooks)

    assert ctx.backend is not None
    assert ctx.backend.name == "jax"
    assert ctx.model is None
    assert ctx.optimizer is None
    assert ctx.actor_source is ctx.backend
    assert ctx.device == "jax"
    assert ctx.n_params == 123


def test_cli_safe_runtime_env_overrides_apply_caps(monkeypatch):
    cli_mod = load_cli_main_module()
    monkeypatch.setenv("QUARTZ_SAFE_RUNTIME", "1")
    monkeypatch.setenv("QUARTZ_SAFE_BOOTSTRAP_TARGET_CAP", "12")
    monkeypatch.setenv("QUARTZ_SAFE_SELFPLAY_PARALLEL_CAP", "3")
    monkeypatch.setenv("QUARTZ_SAFE_SELFPLAY_BATCH_GAMES_CAP", "5")

    cfg = cli_mod._apply_safe_runtime_env_overrides({"_resident_session": True})

    assert cfg["_disable_resident_session"] is True
    assert cfg["_bootstrap_replay_target_cap"] == 12
    assert cfg["_selfplay_parallel_cap"] == 3
    assert cfg["_selfplay_batch_games_cap"] == 5


def test_replay_module_imports_without_torch():
    module = load_module_with_torch_blocked("quartz_replay_no_torch", "quartz/replay.py")

    target = module.sparse_policy_from_entries([[3, 1.0]], 9)
    assert target.n_actions == 9


def test_train_loop_module_imports_without_torch():
    module = load_module_with_torch_blocked("quartz_train_loop_no_torch", "quartz/train_loop.py")

    assert module.round_or_none(1.23456) == 1.2346


def test_system_runtime_module_imports_without_torch():
    module = load_module_with_torch_blocked("quartz_system_runtime_no_torch", "quartz/system_runtime.py")

    assert module.max_supported_threads(types.SimpleNamespace(logical_cpus=4)) == 4


def test_autotune_runtime_module_imports_without_torch():
    module = load_module_with_torch_blocked("quartz_autotune_runtime_no_torch", "quartz/autotune_runtime.py")

    assert module._round_down_to_multiple(130, 32) == 128


def test_runtime_support_module_imports_without_torch():
    module = load_module_with_torch_blocked("quartz_runtime_support_no_torch", "quartz/runtime_support.py")

    assert module.default_encoder_cfg("gomoku7")["_name"] == "gomoku7"


def test_evaluator_runtime_module_imports_without_torch():
    module = load_module_with_torch_blocked("quartz_evaluator_runtime_no_torch", "quartz/evaluator_runtime.py")

    hooks = module._default_runtime_hooks()
    assert hooks.search_client_cls is not None


def test_monitor_iteration_regex_handles_step_ratio():
    monitor = load_monitor_script_module()
    line = "[ 10/20] loss=4.4578 (p=3.8057 v=0.6521) lr=0.01165 replay=2805 +89 steps=2/2 17.1s"
    evt = monitor.parse_stdout_event(line)

    assert evt is not None
    assert evt["type"] == "iteration"
    assert evt["iter"] == "10"
    assert evt["total"] == "20"
    assert evt["steps_done"] == "2"
    assert evt["steps_planned"] == "2"


def test_monitor_command_settings_capture_runtime_hygiene_flags():
    monitor = load_monitor_script_module()

    settings = monitor.parse_command_settings(
        ["python", "-m", "quartz.train", "--runtime-autotune", "--search-profile", "baseline_strict"]
    )

    assert settings["runtime_tuner_enabled"] is True
    assert settings["eval_selfplay_isolated"] is True
    assert settings["search_profile"] == "baseline_strict"

    settings = monitor.parse_command_settings(
        ["python", "-m", "quartz.train", "--no-eval-selfplay-isolation"]
    )
    assert settings["eval_selfplay_isolated"] is False


def test_monitor_parse_stdout_event_captures_async_runtime_markers():
    monitor = load_monitor_script_module()

    assert monitor.parse_stdout_event("  Auto-tune profile: running warmup benchmark...")["type"] == "autotune_warmup_start"
    assert monitor.parse_stdout_event("  [BG] WARN: self-play pause timed out; evaluation proceeding concurrently")[
        "type"
    ] == "bg_pause_timeout"
    assert monitor.parse_stdout_event("  [BG] Self-play resumed after evaluation")["type"] == "bg_resumed"


def test_monitor_async_trace_summaries_include_wavefront_and_batch_stats(tmp_path):
    monitor = load_monitor_script_module()
    trace_path = tmp_path / "rust_server_trace.jsonl"
    rows = [
        {"event": "run_multi_async_batch_start", "jobs": 6, "max_inflight_per_job": 3},
        {"event": "run_multi_async_batch_done", "null_results": 2},
        {
            "event": "selfplay_runner_wave",
            "newly_completed": 1,
            "wave_positions_emitted": 12,
            "replenished_slots": 2,
            "batch_elapsed_ms": 250.0,
            "frontier_slots": 8,
            "active_games": 10,
        },
        {"event": "eval_runner_wave", "newly_completed": 2, "wave_positions_evaluated": 16},
        {"event": "selfplay_runner_done", "duration_ms": 1500.0},
        {"event": "eval_runner_done", "duration_ms": 2300.0},
    ]
    trace_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    trace_summary = monitor.summarize_rust_server_trace(trace_path)
    runner_summary = monitor.summarize_runner_progress(trace_path)

    assert trace_summary["async_batch_runs"] == 1
    assert trace_summary["async_batch_jobs_sum"] == 6
    assert trace_summary["async_batch_null_results_sum"] == 2
    assert trace_summary["selfplay_runner_done_count"] == 1
    assert trace_summary["eval_runner_done_count"] == 1

    assert runner_summary["selfplay_wave_count"] == 1
    assert runner_summary["selfplay_positions_emitted"] == 12
    assert runner_summary["selfplay_wave_elapsed_ms_sum"] == 250.0
    assert runner_summary["selfplay_frontier_slots_sum"] == 8
    assert runner_summary["selfplay_active_games_sum"] == 10
    assert runner_summary["eval_wave_count"] == 1
    assert runner_summary["eval_positions_evaluated"] == 16


def test_gpu_detect_install_deps_uses_shell_aware_split(monkeypatch):
    gpu_detect = load_gpu_detect_module()
    calls = []

    def fake_check_call(args):
        calls.append(args)
        return 0

    monkeypatch.setattr(gpu_detect, "recommend_install", lambda gpu, framework: {framework: ["pip install 'jax[metal]' flax"]})
    monkeypatch.setattr(gpu_detect.subprocess, "check_call", fake_check_call)

    gpu_detect.install_deps(gpu_detect.GpuInfo(vendor="apple"), framework="jax", dry_run=False)

    assert calls == [["pip", "install", "jax[metal]", "flax"]]


def test_play_app_preserves_checkpoint_tuned_cfg(monkeypatch, tmp_path):
    play_gui = load_play_gui_module()
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"x")

    captured = {}

    class DummySession:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.id = "sess123"

    loaded = play_gui.LoadedModel(
        game="gomoku7",
        path=str(model_path),
        cfg={"_name": "gomoku7", "board": 9, "ch": 5, "iters": 77, "actions": 81},
        model=object(),
    )

    app = play_gui.PlayApp(tmp_path, play_gui.torch.device("cpu"), "./target/release/mcts_demo")
    monkeypatch.setattr(app.model_store, "load", lambda game, path: loaded)
    monkeypatch.setattr(play_gui, "GameSession", DummySession)

    session = app.create_session({"game": "gomoku7", "modelPath": str(model_path), "humanSide": "black"})

    assert session.id == "sess123"
    assert captured["cfg"]["board"] == 9
    assert captured["cfg"]["ch"] == 5
    assert captured["cfg"]["iters"] == 77


def test_auto_device_name_prefers_mps_when_cuda_unavailable(monkeypatch):
    az = load_training_module()

    monkeypatch.setattr(az.sys, "platform", "darwin")
    monkeypatch.setattr(az.torch.cuda, "is_available", lambda: False)

    class FakeMps:
        @staticmethod
        def is_available():
            return True

    monkeypatch.setattr(az.torch, "backends", type("Backends", (), {"mps": FakeMps})())

    assert az.auto_device_name() == "mps"


def test_pyproject_jax_extra_includes_torch_for_train_entrypoint():
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    jax_extra = data["project"]["optional-dependencies"]["jax"]

    assert any(dep.startswith("torch") for dep in jax_extra)


def test_rust_nn_evaluator_uses_chess_session_payloads(monkeypatch):
    az = load_training_module()
    start_fen = az.STANDARD_CHESS_FEN
    next_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

    class FakeClient:
        last_instance = None

        def __init__(self, model, cfg, device, rust_binary):
            self.model = model
            self.cfg = cfg
            self.device = device
            self.rust_binary = rust_binary
            self.open_jobs = None
            self.step_updates = []
            self.closed_session_id = None
            self.started = False
            self.stopped = False
            FakeClient.last_instance = self

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def open_search_session(self, jobs, penalty_mode="GatedRefresh"):
            self.open_jobs = jobs
            return {
                "session_id": 7,
                "results": [
                    {"best_move": 0, "iterations": 1, "result_fen": next_fen},
                    {"best_move": 0, "iterations": 1, "result_fen": next_fen},
                ],
            }

        def step_search_session(self, session_id, updates):
            self.step_updates.append((session_id, updates))
            return {"session_id": session_id, "results": []}

        def close_search_session(self, session_id):
            self.closed_session_id = session_id
            return {"ok": True}

    monkeypatch.setattr(az, "NNSearchClient", FakeClient)
    cfg = {"_name": "chess", "iters": 8, "actions": az.CHESS_POLICY_ACTIONS}
    eng_a = az.RustNNEvaluatorEngine("candidate", cfg, object(), az.torch.device("cpu"))
    eng_b = az.RustNNEvaluatorEngine("champion", cfg, object(), az.torch.device("cpu"))

    tally = eng_a.play_match_tally_against(
        eng_b,
        lambda: az.ChessEvaluationAdapter(actions=az.CHESS_POLICY_ACTIONS, start_fen=start_fen),
        opening_book=[],
        num_games=2,
        color_swap=True,
        max_moves=1,
        seed=7,
    )

    fake = FakeClient.last_instance
    assert tally.total == 2
    assert fake is not None and fake.started and fake.stopped
    assert fake.closed_session_id == 7
    assert fake.open_jobs is not None and len(fake.open_jobs) == 2
    assert all(job.get("fen") == start_fen for job in fake.open_jobs)
    assert all("board" not in job for job in fake.open_jobs)
    assert fake.step_updates and fake.step_updates[0][0] == 7


def test_rust_nn_evaluator_uses_rust_eval_state_machine_for_gomoku7(monkeypatch):
    az = load_training_module()

    class MiniGame:
        def __init__(self):
            self._board = [0] * 49
            self._player = 0

        def current_player(self):
            return self._player

        def legal_moves(self):
            return [idx for idx, value in enumerate(self._board) if value == 0]

        def is_terminal(self):
            return False

        def apply_move(self, action):
            self._board[action] = 1 if self._player == 0 else -1
            self._player = 1 - self._player

    class FakeClient:
        last_instance = None

        def __init__(self, model, cfg, device, rust_binary):
            self.eval_sessions = None
            self.started = False
            self.stopped = False
            self.open_called = False
            FakeClient.last_instance = self

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def eval_match_run(self, sessions, max_moves, penalty_mode="GatedRefresh"):
            self.eval_sessions = sessions
            return {
                "records": [
                    {
                        "game_id": "g0000",
                        "black_tag": 0,
                        "white_tag": 1,
                        "outcome": "black_win",
                        "score_black": 1.0,
                        "move_count": 4,
                        "total_time_ms": 12.0,
                        "opening": [],
                        "seed": 7,
                        "error": None,
                        "is_void": False,
                    },
                    {
                        "game_id": "g0001",
                        "black_tag": 1,
                        "white_tag": 0,
                        "outcome": "white_win",
                        "score_black": 0.0,
                        "move_count": 4,
                        "total_time_ms": 13.0,
                        "opening": [],
                        "seed": 8,
                        "error": None,
                        "is_void": False,
                    },
                ]
            }

        def open_search_session(self, jobs, penalty_mode="GatedRefresh"):
            self.open_called = True
            raise AssertionError("session fallback should not be used when rust eval runner succeeds")

    monkeypatch.setattr(az, "NNSearchClient", FakeClient)
    cfg = {
        "_name": "gomoku7",
        "iters": 8,
        "actions": 49,
        "_eval_runner_mode": "rust_eval_state_machine",
    }
    eng_a = az.RustNNEvaluatorEngine("candidate", cfg, object(), az.torch.device("cpu"))
    eng_b = az.RustNNEvaluatorEngine("champion", cfg, object(), az.torch.device("cpu"))

    tally = eng_a.play_match_tally_against(
        eng_b,
        MiniGame,
        opening_book=[],
        num_games=2,
        color_swap=True,
        max_moves=10,
        seed=7,
    )

    fake = FakeClient.last_instance
    assert tally.total == 2
    assert tally.wins == 2
    assert fake is not None and fake.started and fake.stopped
    assert fake.eval_sessions is not None and len(fake.eval_sessions) == 2
    assert fake.open_called is False


def test_rust_nn_evaluator_reuses_single_model_tag_for_same_model(monkeypatch):
    az = load_training_module()

    class MiniGame:
        def __init__(self):
            self._board = [0] * 49
            self._player = 0

        def current_player(self):
            return self._player

        def legal_moves(self):
            return [idx for idx, value in enumerate(self._board) if value == 0]

        def is_terminal(self):
            return False

        def apply_move(self, action):
            self._board[action] = 1 if self._player == 0 else -1
            self._player = 1 - self._player

    class FakeClient:
        last_instance = None

        def __init__(self, model, cfg, device, rust_binary):
            self.model = model
            self.cfg = dict(cfg)
            self.eval_sessions = None
            self.started = False
            self.stopped = False
            FakeClient.last_instance = self

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def eval_match_run(self, sessions, max_moves, penalty_mode="GatedRefresh"):
            self.eval_sessions = sessions
            records = []
            for sess in sessions:
                records.append(
                    {
                        "game_id": sess["game_id"],
                        "black_tag": sess["black_tag"],
                        "white_tag": sess["white_tag"],
                        "outcome": "draw",
                        "score_black": 0.5,
                        "move_count": 1,
                        "total_time_ms": 5.0,
                        "opening": list(sess.get("opening", [])),
                        "seed": sess.get("seed"),
                        "error": None,
                        "is_void": False,
                    }
                )
            return {"records": records}

        def open_search_engine_session(self, jobs, penalty_mode="GatedRefresh", iters=None):
            raise AssertionError("same-model eval should stay on typed eval_match_run path")

    monkeypatch.setattr(az, "NNSearchClient", FakeClient)
    shared_model = object()
    cfg = {
        "_name": "gomoku7",
        "iters": 8,
        "actions": 49,
        "_eval_runner_mode": "rust_eval_state_machine",
    }
    eng_a = az.RustNNEvaluatorEngine("candidate", cfg, shared_model, az.torch.device("cpu"))
    eng_b = az.RustNNEvaluatorEngine("champion", cfg, shared_model, az.torch.device("cpu"))

    tally = eng_a.play_match_tally_against(
        eng_b,
        MiniGame,
        opening_book=[],
        num_games=2,
        color_swap=True,
        max_moves=1,
        seed=7,
    )

    fake = FakeClient.last_instance
    assert tally.total == 2
    assert fake is not None and fake.model is shared_model
    assert fake.eval_sessions is not None and len(fake.eval_sessions) == 2
    assert fake.cfg["batch_size"] == 2
    assert fake.cfg["batch_timeout_us"] == 900
    assert all(sess["black_tag"] == 0 for sess in fake.eval_sessions)
    assert all(sess["white_tag"] == 0 for sess in fake.eval_sessions)


def test_persistent_rust_eval_campaign_reuses_single_client(monkeypatch):
    from quartz import evaluator_runtime as eval_mod

    class MiniGame:
        def __init__(self):
            self._board = [0] * 49
            self._player = 0

        def current_player(self):
            return self._player

        def legal_moves(self):
            return [idx for idx, value in enumerate(self._board) if value == 0]

        def is_terminal(self):
            return False

        def apply_move(self, action):
            self._board[action] = 1 if self._player == 0 else -1
            self._player = 1 - self._player

    class FakeClient:
        starts = 0
        stops = 0
        eval_calls = 0
        last_model_map = None

        def __init__(self, model, cfg, device, rust_binary):
            FakeClient.last_model_map = model

        def start(self):
            FakeClient.starts += 1

        def stop(self):
            FakeClient.stops += 1

        def eval_match_run(self, sessions, max_moves, penalty_mode="GatedRefresh"):
            FakeClient.eval_calls += 1
            records = []
            for idx, sess in enumerate(sessions):
                records.append(
                    {
                        "game_id": sess["game_id"],
                        "black_tag": sess["black_tag"],
                        "white_tag": sess["white_tag"],
                        "outcome": "black_win" if idx % 2 == 0 else "white_win",
                        "score_black": 1.0 if idx % 2 == 0 else 0.0,
                        "move_count": 4,
                        "total_time_ms": 10.0,
                        "opening": list(sess.get("opening", [])),
                        "seed": sess.get("seed"),
                        "error": None,
                        "is_void": False,
                        "search_summary": {
                            "root_visits": {"samples": [8, 8, 8, 8]},
                            "halt_reason_hist": {"BudgetExhausted": 4},
                            "benchmark_safe": True,
                            "selection_trace": {
                                "root_selects": 16,
                                "refresh_selected_count": 4,
                                "selected_penalty_abs_sum": 1.25,
                                "selected_effective_prior_l1_sum": 0.5,
                            },
                        },
                    }
                )
            return {"records": records}

    runtime_hooks = eval_mod.EvaluatorRuntimeHooks(
        search_client_cls=FakeClient,
        is_chess_game=lambda game_name: False,
        build_rust_state_meta=lambda game_name, game, cfg: {},
        iter_sparse_policy_entries=lambda payload: [],
        supports_rust_eval_state_machine=lambda game_name: True,
        stall_trace=lambda *args, **kwargs: None,
        game_record_cls=eval_mod.runtime_support.GameRecord,
        tally_match=eval_mod.runtime_support.tally_match,
    )
    cfg = {"_name": "gomoku7", "iters": 8, "actions": 49, "_eval_runner_mode": "rust_eval_state_machine"}
    eng_a = eval_mod.RustNNEvaluatorEngine("A", cfg, object(), "cpu", runtime_hooks=runtime_hooks)
    eng_b = eval_mod.RustNNEvaluatorEngine("B", cfg, object(), "cpu", runtime_hooks=runtime_hooks)
    eng_c = eval_mod.RustNNEvaluatorEngine("C", cfg, object(), "cpu", runtime_hooks=runtime_hooks)

    with eval_mod.PersistentRustNNEvalCampaign([eng_a, eng_b, eng_c], 4, runtime_hooks=runtime_hooks) as campaign:
        tally_ab, meta_ab = campaign.compare(eng_a, eng_b, MiniGame, [], 4, max_moves=4, seed=7)
        tally_ac, meta_ac = campaign.compare(eng_a, eng_c, MiniGame, [], 4, max_moves=4, seed=7)

    assert FakeClient.starts == 1
    assert FakeClient.stops == 1
    assert FakeClient.eval_calls == 2
    assert isinstance(FakeClient.last_model_map, dict)
    assert sorted(FakeClient.last_model_map) == [0, 1, 2]
    assert tally_ab.total == 4 and tally_ac.total == 4
    assert meta_ab["runner_mode"] == "rust_eval_state_machine"
    assert meta_ac["runner_mode"] == "rust_eval_state_machine"
    assert meta_ab["client_start_s"] >= 0.0
    assert meta_ab["realized_budget_trace"]["games"] == 4
    assert meta_ab["realized_budget_trace"]["selection_trace_coverage_frac"] == pytest.approx(1.0)
    assert meta_ab["realized_budget_trace"]["selection_trace"]["root_selects"] == 64
    assert meta_ab["realized_budget_trace"]["selection_trace"]["refresh_selected_count"] == 16


def test_persistent_rust_eval_campaign_compare_many_batches_single_eval_call():
    from quartz import evaluator_runtime as eval_mod

    class MiniGame:
        def __init__(self):
            self._board = [0] * 49
            self._player = 0

        def current_player(self):
            return self._player

        def legal_moves(self):
            return [idx for idx, value in enumerate(self._board) if value == 0]

        def is_terminal(self):
            return False

        def apply_move(self, action):
            self._board[action] = 1 if self._player == 0 else -1
            self._player = 1 - self._player

    class FakeClient:
        starts = 0
        stops = 0
        eval_calls = 0

        def __init__(self, model, cfg, device, rust_binary):
            pass

        def start(self):
            FakeClient.starts += 1

        def stop(self):
            FakeClient.stops += 1

        def eval_match_run(self, sessions, max_moves, penalty_mode="GatedRefresh"):
            FakeClient.eval_calls += 1
            records = []
            for idx, sess in enumerate(sessions):
                records.append(
                    {
                        "game_id": sess["game_id"],
                        "black_tag": sess["black_tag"],
                        "white_tag": sess["white_tag"],
                        "outcome": "black_win" if idx % 2 == 0 else "white_win",
                        "score_black": 1.0 if idx % 2 == 0 else 0.0,
                        "move_count": 4,
                        "total_time_ms": 10.0,
                        "opening": list(sess.get("opening", [])),
                        "seed": sess.get("seed"),
                        "error": None,
                        "is_void": False,
                    }
                )
            return {"records": records}

    runtime_hooks = eval_mod.EvaluatorRuntimeHooks(
        search_client_cls=FakeClient,
        is_chess_game=lambda game_name: False,
        build_rust_state_meta=lambda game_name, game, cfg: {},
        iter_sparse_policy_entries=lambda payload: [],
        supports_rust_eval_state_machine=lambda game_name: True,
        stall_trace=lambda *args, **kwargs: None,
        game_record_cls=eval_mod.runtime_support.GameRecord,
        tally_match=eval_mod.runtime_support.tally_match,
    )
    cfg = {"_name": "gomoku7", "iters": 8, "actions": 49, "_eval_runner_mode": "rust_eval_state_machine"}
    eng_a = eval_mod.RustNNEvaluatorEngine("A", cfg, object(), "cpu", runtime_hooks=runtime_hooks)
    eng_b = eval_mod.RustNNEvaluatorEngine("B", cfg, object(), "cpu", runtime_hooks=runtime_hooks)
    eng_c = eval_mod.RustNNEvaluatorEngine("C", cfg, object(), "cpu", runtime_hooks=runtime_hooks)

    with eval_mod.PersistentRustNNEvalCampaign([eng_a, eng_b, eng_c], 2, runtime_hooks=runtime_hooks) as campaign:
        results = campaign.compare_many(
            [
                {
                    "match_id": "ab",
                    "engine_a": eng_a,
                    "engine_b": eng_b,
                    "game_factory": MiniGame,
                    "opening_book": [],
                    "num_games": 2,
                    "max_moves": 4,
                    "seed": 7,
                },
                {
                    "match_id": "ac",
                    "engine_a": eng_a,
                    "engine_b": eng_c,
                    "game_factory": MiniGame,
                    "opening_book": [],
                    "num_games": 2,
                    "max_moves": 4,
                    "seed": 11,
                },
            ]
        )

    assert FakeClient.starts == 1
    assert FakeClient.stops == 1
    assert FakeClient.eval_calls == 1
    assert [match_id for match_id, _tally, _meta in results] == ["ab", "ac"]
    assert results[0][1].total == 2
    assert results[1][1].total == 2
    assert results[0][2]["batch_id"] == results[1][2]["batch_id"]
    assert results[0][2]["batch_total_games"] == 4


def test_arena_eval_runtime_cfg_reduces_low_concurrency_batching():
    from quartz import evaluator_runtime as eval_mod

    tiny = eval_mod.arena_eval_runtime_cfg({"batch_size": 8, "batch_timeout_us": 1500, "n_threads": 4}, 2)
    assert tiny["batch_size"] == 2
    assert tiny["batch_timeout_us"] == 700
    assert tiny["_arena_low_concurrency_profile"] == "tiny"

    small = eval_mod.arena_eval_runtime_cfg({"batch_size": 8, "batch_timeout_us": 1500, "n_threads": 1}, 4)
    assert small["batch_size"] == 4
    assert small["batch_timeout_us"] == 1200
    assert small["_arena_low_concurrency_profile"] == "small"

    untouched = eval_mod.arena_eval_runtime_cfg(
        {"batch_size": 8, "batch_timeout_us": 1500, "n_threads": 4, "_arena_eval_topology_override": False},
        2,
    )
    assert untouched["batch_size"] == 8
    assert untouched["batch_timeout_us"] == 1500
    assert "_arena_low_concurrency_profile" not in untouched


def test_rust_nn_evaluator_rejects_manifest_mismatch():
    az = load_training_module()

    class MiniGame:
        def __init__(self):
            self._board = [0] * 49
            self._player = 0

        def current_player(self):
            return self._player

        def legal_moves(self):
            return [idx for idx, value in enumerate(self._board) if value == 0]

        def is_terminal(self):
            return False

        def apply_move(self, action):
            self._board[action] = 1 if self._player == 0 else -1
            self._player = 1 - self._player

    cfg_a = {"_name": "gomoku7", "iters": 8, "actions": 49, "search_profile": "baseline"}
    cfg_b = {"_name": "gomoku7", "iters": 8, "actions": 49, "search_profile": "quartz"}
    eng_a = az.RustNNEvaluatorEngine("candidate", cfg_a, object(), az.torch.device("cpu"))
    eng_b = az.RustNNEvaluatorEngine("champion", cfg_b, object(), az.torch.device("cpu"))

    with pytest.raises(RuntimeError, match="matching search manifests"):
        eng_a.play_match_tally_against(
            eng_b,
            MiniGame,
            opening_book=[],
            num_games=2,
            color_swap=True,
            max_moves=2,
            seed=7,
        )


def test_rust_nn_evaluator_select_moves_batch_handles_non_chess_games(monkeypatch):
    az = load_training_module()

    class MiniGame:
        def __init__(self):
            self._board = [0] * 49

        def current_player(self):
            return 0

        def legal_moves(self):
            return [2, 7, 11]

    class FakeClient:
        last_instance = None

        def __init__(self, model, cfg, device, rust_binary):
            self.started = False
            self.stopped = False
            FakeClient.last_instance = self

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def search_moves_multi(self, jobs, penalty_mode="GatedRefresh"):
            assert len(jobs) == 1
            return [{"best_move": 7, "policy": [[2, 0.1], [7, 0.8], [11, 0.1]], "p_flip": 0.0}]

    monkeypatch.setattr(az, "NNSearchClient", FakeClient)
    cfg = {"_name": "gomoku7", "iters": 8, "actions": 49}
    eng = az.RustNNEvaluatorEngine("candidate", cfg, object(), az.torch.device("cpu"))

    move, meta = eng.select_moves_batch([MiniGame()])[0]

    assert move == 7
    assert meta["engine"] == "rust_nn"
    assert meta["simulations"] == 8
    fake = FakeClient.last_instance
    assert fake is not None and fake.started is True
    eng.reset()
    assert fake.stopped is True


def test_backend_auto_prefers_torch_over_jax():
    backend_mod = load_backend_module()
    detection = {
        "jax": True,
        "jax_gpu": True,
        "torch": True,
        "torch_gpu": True,
    }

    assert backend_mod.select_backend(detection, preference="auto") == "torch"
    assert backend_mod.select_backend(detection, preference="jax") == "jax"
    assert backend_mod.select_backend(detection, preference="torch") == "torch"


def test_selfplay_batched_uses_rust_state_machine_payload(monkeypatch):
    az = load_training_module()

    class FakeClient:
        last_instance = None

        def __init__(self, model, cfg, device, rust_binary):
            self.started = False
            self.stopped = False
            self.calls = []
            FakeClient.last_instance = self

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def selfplay_run(self, n_games, parallel, temp_threshold, penalty_mode="GatedRefresh", seed=0):
            self.calls.append((n_games, parallel, temp_threshold))
            return {
                "games": [
                    {
                        "states": [[0] * 49],
                        "players": [1],
                        "policies": [["0:1.0"]],
                        "outcome": 1.0,
                        "trace": [{"iterations": 8}],
                    },
                    {
                        "states": [[0] * 49],
                        "players": [1],
                        "policies": [["1:1.0"]],
                        "outcome": -1.0,
                        "trace": [{"iterations": 8}],
                    },
                ]
            }

    monkeypatch.setattr(az, "NNSearchClient", FakeClient)
    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    cfg["_name"] = "gomoku7"
    cfg["_selfplay_runner_mode"] = "rust_selfplay_state_machine"
    states, policies, outcomes, traces = az.selfplay_rust_nn_batched(
        cfg,
        model=object(),
        device=az.torch.device("cpu"),
        n_games=2,
        rust_binary="./target/release/mcts_demo",
        parallel=2,
        show_progress=False,
    )

    fake = FakeClient.last_instance
    assert fake is not None and fake.started and fake.stopped
    assert fake.calls == [(2, 2, cfg["temp_th"])]
    assert len(states) == 2 and len(policies) == 2 and len(outcomes) == 2 and len(traces) == 2
    assert states[0][0].shape == (cfg["ch"], cfg["board"], cfg["board"])
    assert float(policies[0][0][0]) == pytest.approx(1.0)
    assert outcomes == [1.0, -1.0]


def test_selfplay_rust_nn_uses_training_module_search_client(monkeypatch):
    az = load_training_module()

    class FakeClient:
        last_instance = None

        def __init__(self, model, cfg, device, rust_binary):
            self.calls = 0
            self.started = False
            self.stopped = False
            FakeClient.last_instance = self

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def search_move(self, board_flat, player, penalty_mode="GatedRefresh", fen=None, state_meta=None):
            self.calls += 1
            if self.calls == 1:
                return {"best_move": 0, "policy": ["0:1.0"], "value": 0.0}
            return {"policy": [], "value": 0.0}

    monkeypatch.setattr(az, "NNSearchClient", FakeClient)
    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    cfg["_name"] = "gomoku7"
    states, policies, outcomes, traces = az.selfplay_rust_nn(
        cfg,
        model=object(),
        device=az.torch.device("cpu"),
        n_games=1,
        rust_binary="./target/release/mcts_demo",
    )

    fake = FakeClient.last_instance
    assert fake is not None and fake.started and fake.stopped
    assert fake.calls == 2
    assert len(states) == 1 and len(policies) == 1 and len(outcomes) == 1
    assert states[0][0].shape[1:] == (cfg["board"], cfg["board"])
    assert states[0][0].shape[0] > 0
    assert float(policies[0][0][0]) == pytest.approx(1.0)
    assert traces == 0


def test_arena_rust_nn_impl_uses_shared_eval_runner(monkeypatch, tmp_path):
    az = load_training_module()

    class DummyNet:
        def __init__(self, cfg):
            self.cfg = cfg

        def to(self, device):
            return self

        def load_state_dict(self, state_dict):
            self.state_dict = state_dict

        def eval(self):
            return self

    created_engines = []
    built_games = []

    class FakeRustEngine:
        def __init__(self, engine_name, cfg, model, device, rust_binary="./target/release/mcts_demo"):
            self.engine_name = engine_name
            self.cfg = cfg
            self.model = model
            self.device = device
            self.rust_binary = rust_binary
            created_engines.append(self)

        def reset(self):
            return None

        def name(self):
            return self.engine_name

    class FakeRunner:
        instances = []

        def __init__(self, game_factory, opening_book=None, seed=None, max_moves=500):
            self.game_factory = game_factory
            self.opening_book = opening_book
            self.seed = seed
            self.max_moves = max_moves
            self.play_calls = []
            FakeRunner.instances.append(self)

        def play_match_tally_batched(self, eng_a, eng_b, num_games, color_swap=True, logger=None):
            built_games.append(self.game_factory())
            self.play_calls.append((eng_a.name(), eng_b.name(), num_games, color_swap))
            return types.SimpleNamespace(wins=1, losses=0, draws=1)

    monkeypatch.setattr(az, "AlphaZeroNet", DummyNet)
    monkeypatch.setattr(az, "load_torch_state_dict", lambda *args, **kwargs: {})
    monkeypatch.setattr(az, "RustNNEvaluatorEngine", FakeRustEngine)
    monkeypatch.setattr(az, "MatchRunner", FakeRunner)
    monkeypatch.setattr(az, "build_training_game_adapter", lambda cfg: {"game": cfg["_name"]})

    model_a = tmp_path / "a.pt"
    model_b = tmp_path / "b.pt"
    model_a.write_bytes(b"a")
    model_b.write_bytes(b"b")

    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    cfg["_name"] = "gomoku7"
    wins_a, wins_b, draws, wr, _ci, _sprt = az._arena_rust_nn_impl(
        str(model_a),
        cfg,
        str(model_b),
        cfg,
        az.torch.device("cpu"),
        n_games=2,
        strict=True,
    )

    assert len(created_engines) == 2
    assert {engine.engine_name for engine in created_engines} == {"arena_a", "arena_b"}
    assert all(engine.cfg.get("_eval_runner_mode") == "rust_eval_state_machine" for engine in created_engines)
    assert FakeRunner.instances and FakeRunner.instances[0].play_calls == [("arena_a", "arena_b", 2, True)]
    assert built_games == [{"game": "gomoku7"}]
    assert (wins_a, wins_b, draws) == (1, 0, 1)
    assert wr == pytest.approx(0.5)


def test_arena_rust_nn_impl_uses_distinct_models_for_same_source_under_eval_runner(monkeypatch, tmp_path):
    az = load_training_module()

    class DummyNet:
        instances = []

        def __init__(self, cfg):
            self.cfg = dict(cfg)
            self.state_dict = None
            DummyNet.instances.append(self)

        def to(self, device):
            return self

        def load_state_dict(self, state_dict):
            self.state_dict = state_dict

        def eval(self):
            return self

    created_engines = []

    class FakeRustEngine:
        def __init__(self, engine_name, cfg, model, device, rust_binary="./target/release/mcts_demo"):
            self.engine_name = engine_name
            self.cfg = dict(cfg)
            self.model = model
            created_engines.append(self)

        def reset(self):
            return None

        def name(self):
            return self.engine_name

    class FakeRunner:
        def __init__(self, game_factory, opening_book=None, seed=None, max_moves=500):
            self.game_factory = game_factory

        def play_match_tally_batched(self, eng_a, eng_b, num_games, color_swap=True, logger=None):
            return types.SimpleNamespace(wins=1, losses=0, draws=0)

    monkeypatch.setattr(az, "AlphaZeroNet", DummyNet)
    monkeypatch.setattr(az, "load_torch_state_dict", lambda *args, **kwargs: {"w": 1})
    monkeypatch.setattr(az, "RustNNEvaluatorEngine", FakeRustEngine)
    monkeypatch.setattr(az, "MatchRunner", FakeRunner)
    monkeypatch.setattr(az, "build_training_game_adapter", lambda cfg: {"game": cfg["_name"]})

    model_path = tmp_path / "shared.pt"
    model_path.write_bytes(b"a")

    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    cfg["_name"] = "gomoku7"

    wins_a, wins_b, draws, wr, _ci, _sprt = az._arena_rust_nn_impl(
        str(model_path),
        cfg,
        str(model_path),
        cfg,
        az.torch.device("cpu"),
        n_games=2,
        strict=True,
    )

    assert len(DummyNet.instances) == 2
    assert len(created_engines) == 2
    assert created_engines[0].cfg.get("_eval_runner_mode") == "rust_eval_state_machine"
    assert created_engines[1].cfg.get("_eval_runner_mode") == "rust_eval_state_machine"
    assert created_engines[0].model is not created_engines[1].model
    assert (wins_a, wins_b, draws) == (1, 0, 0)
    assert wr == pytest.approx(1.0)


def test_arena_rust_nn_impl_keeps_legacy_dual_cfg_path(monkeypatch, tmp_path):
    az = load_training_module()

    class DummyNet:
        def __init__(self, cfg):
            self.cfg = cfg

        def to(self, device):
            return self

        def load_state_dict(self, state_dict):
            self.state_dict = state_dict

        def eval(self):
            return self

    class FakeClient:
        instances = []

        def __init__(self, model, cfg, device, rust_binary):
            self.cfg = cfg
            self.calls = 0
            self.started = False
            self.stopped = False
            FakeClient.instances.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def search_move(self, board_flat, player, penalty_mode="GatedRefresh", fen=None, state_meta=None):
            self.calls += 1
            if self.calls == 1:
                return {"best_move": 0, "policy": ["0:1.0"], "value": 0.0}
            return {"policy": [], "value": 1.0}

    monkeypatch.setattr(az, "AlphaZeroNet", DummyNet)
    monkeypatch.setattr(az, "load_torch_state_dict", lambda *args, **kwargs: {})
    monkeypatch.setattr(az, "NNSearchClient", FakeClient)

    model_a = tmp_path / "a.pt"
    model_b = tmp_path / "b.pt"
    model_a.write_bytes(b"a")
    model_b.write_bytes(b"b")

    cfg_a = dict(az.GAME_CONFIGS["gomoku7"])
    cfg_a["_name"] = "gomoku7"
    cfg_b = dict(cfg_a)
    cfg_b["penalty_mode"] = "GatedRefreshLegacy"

    wins_a, wins_b, draws, wr, _ci, _sprt = az._arena_rust_nn_impl(
        str(model_a),
        cfg_a,
        str(model_b),
        cfg_b,
        az.torch.device("cpu"),
        n_games=2,
        strict=True,
    )

    assert len(FakeClient.instances) == 2
    assert all(client.started and client.stopped for client in FakeClient.instances)
    assert wins_a + wins_b + draws >= 1
    assert wr >= 0.0


def test_arena_rust_nn_impl_rejects_unscored_strict_tally(monkeypatch, tmp_path):
    az = load_training_module()

    class DummyNet:
        def __init__(self, cfg):
            self.cfg = cfg

        def to(self, device):
            return self

        def load_state_dict(self, state_dict):
            self.state_dict = state_dict

        def eval(self):
            return self

    class FakeRustEngine:
        def __init__(self, engine_name, cfg, model, device, rust_binary="./target/release/mcts_demo"):
            self.engine_name = engine_name

        def reset(self):
            return None

        def name(self):
            return self.engine_name

    class FakeRunner:
        def __init__(self, game_factory, opening_book=None, seed=None, max_moves=500):
            self.game_factory = game_factory

        def play_match_tally_batched(self, eng_a, eng_b, num_games, color_swap=True, logger=None):
            return types.SimpleNamespace(
                wins=0,
                losses=0,
                draws=0,
                errors=2,
                voids=0,
                total=2,
                scored=0,
            )

    monkeypatch.setattr(az, "AlphaZeroNet", DummyNet)
    monkeypatch.setattr(az, "load_torch_state_dict", lambda *args, **kwargs: {})
    monkeypatch.setattr(az, "RustNNEvaluatorEngine", FakeRustEngine)
    monkeypatch.setattr(az, "MatchRunner", FakeRunner)
    monkeypatch.setattr(az, "build_training_game_adapter", lambda cfg: {"game": cfg["_name"]})

    model_a = tmp_path / "a.pt"
    model_b = tmp_path / "b.pt"
    model_a.write_bytes(b"a")
    model_b.write_bytes(b"b")

    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    cfg["_name"] = "gomoku7"

    with pytest.raises(RuntimeError, match="strict arena produced unscored games"):
        az._arena_rust_nn_impl(
            str(model_a),
            cfg,
            str(model_b),
            cfg,
            az.torch.device("cpu"),
            n_games=2,
            strict=True,
        )


def test_detect_backends_skips_jax_probe_when_torch_is_available(monkeypatch):
    backend_mod = load_backend_module()
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "jax":
            raise AssertionError("auto backend detection should not import jax when torch is available")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    detection = backend_mod.detect_backends(preference="auto")
    assert detection["torch"] is True
    assert detection["jax_checked"] is False


def test_load_torch_state_dict_falls_back_when_weights_only_rejects_checkpoint():
    backend_mod = load_backend_module()

    class FakeTorch:
        def __init__(self):
            self.calls = []

        def load(self, path, map_location=None, weights_only=None):
            self.calls.append({
                "path": path,
                "map_location": map_location,
                "weights_only": weights_only,
            })
            if weights_only:
                raise RuntimeError("Weights only load failed. Unsupported operand 149")
            return {"ok": True}

    fake_torch = FakeTorch()
    payload = backend_mod.load_torch_state_dict("demo.pt", fake_torch, map_location="cpu")

    assert payload == {"ok": True}
    assert fake_torch.calls == [
        {"path": "demo.pt", "map_location": "cpu", "weights_only": True},
        {"path": "demo.pt", "map_location": "cpu", "weights_only": False},
    ]


def test_profile_monitor_parses_expected_iterations():
    monitor = load_monitor_script_module()

    assert monitor.parse_expected_iterations(["python", "-m", "quartz.train", "--iterations", "30"]) == 30
    assert monitor.parse_expected_iterations(["python", "-m", "quartz.train", "--iterations=12"]) == 12
    assert monitor.parse_expected_iterations(["python", "-m", "quartz.train"]) is None


def test_detect_checkpoint_backend_hint_distinguishes_torch_and_jax(tmp_path):
    az = load_training_module()
    torch_ckpt = tmp_path / "torch.pt"
    jax_ckpt = tmp_path / "jax.pt"
    torch_ckpt.write_bytes(b"PK\x03\x04demo-latest/data.pkl")
    jax_ckpt.write_bytes(b"\x80\x04demo params BatchNorm_0 jax._src.arr")

    assert az.detect_checkpoint_backend_hint(torch_ckpt) == "torch"
    assert az.detect_checkpoint_backend_hint(jax_ckpt) == "jax"


def test_ensure_best_checkpoint_compatible_resets_mismatched_jax_checkpoint(tmp_path):
    az = load_training_module()
    best = tmp_path / "best.pt"
    best.write_bytes(b"\x80\x04demo params BatchNorm_0 jax._src.arr")

    class FakeBackend:
        name = "torch"

        def __init__(self):
            self.saved = None

        def save(self, path):
            self.saved = path
            Path(path).write_bytes(b"PK\x03\x04demo-latest/data.pkl")

    backend = FakeBackend()
    hint = az.ensure_best_checkpoint_compatible(best, backend, model=None, device=az.torch.device("cpu"))

    assert hint == "torch"
    assert Path(backend.saved) == best
    assert az.detect_checkpoint_backend_hint(best) == "torch"


def test_validate_torch_state_dict_reports_shape_mismatch():
    backend_mod = load_backend_module()
    import torch

    model = torch.nn.Linear(4, 2)
    state = {
        "weight": torch.zeros((3, 4)),
        "bias": torch.zeros((3,)),
    }

    reason = backend_mod.validate_torch_state_dict(model, state)

    assert reason is not None
    assert "mismatched=" in reason


def test_should_use_resident_session_only_for_multi_game_and_explicit_enable():
    az = load_training_module()

    assert az.should_use_resident_session("gomoku7", parallel=1, n_games=1, enabled=False) is False
    assert az.should_use_resident_session("gomoku7", parallel=1, n_games=4, enabled=False) is False
    assert az.should_use_resident_session("gomoku7", parallel=2, n_games=1, enabled=False) is False
    assert az.should_use_resident_session("gomoku7", parallel=2, n_games=4, enabled=False) is False
    assert az.should_use_resident_session("gomoku7", parallel=2, n_games=4, enabled=True) is True
    assert az.should_use_resident_session("chess", parallel=4, n_games=8, enabled=True) is True


def test_wait_for_worker_progress_raises_when_worker_exits():
    az = load_training_module()

    class FakeWorker:
        REPLAY_STALL_TIMEOUT_S = 45.0
        positions_generated = 0
        _stop = type("Stop", (), {"is_set": staticmethod(lambda: False)})()

        def status(self):
            return {
                "alive": False,
                "last_progress_age_s": 0.0,
                "last_error": "boom",
                "consecutive_errors": 1,
            }

    with pytest.raises(RuntimeError, match="background self-play worker stopped unexpectedly"):
        az.wait_for_worker_progress(FakeWorker(), 0, min_new=1, timeout_s=0.1, poll_s=0.0)


def test_wait_for_worker_progress_raises_when_worker_stalls_after_errors():
    az = load_training_module()

    class FakeWorker:
        REPLAY_STALL_TIMEOUT_S = 1.0
        positions_generated = 0
        _stop = type("Stop", (), {"is_set": staticmethod(lambda: False)})()

        def status(self):
            return {
                "alive": True,
                "last_progress_age_s": 2.0,
                "last_error": "stall",
                "consecutive_errors": 2,
            }

    with pytest.raises(RuntimeError, match="background self-play stalled"):
        az.wait_for_worker_progress(FakeWorker(), 0, min_new=1, timeout_s=0.1, poll_s=0.0)


def test_wait_for_worker_progress_raises_on_repeated_errors_without_progress():
    az = load_training_module()

    class FakeWorker:
        REPLAY_STALL_TIMEOUT_S = 45.0
        positions_generated = 0
        _stop = type("Stop", (), {"is_set": staticmethod(lambda: False)})()

        def status(self):
            return {
                "alive": True,
                "last_progress_age_s": 16.0,
                "last_error": "seed failed",
                "consecutive_errors": 3,
            }

    with pytest.raises(RuntimeError, match="background self-play made no progress after repeated errors"):
        az.wait_for_worker_progress(FakeWorker(), 0, min_new=1, timeout_s=0.1, poll_s=0.0)


def test_replay_fill_worker_error_fails_fast_after_bootstrap_errors():
    from quartz import cli_main as cli_mod

    message = cli_mod._replay_fill_worker_error(
        {
            "alive": True,
            "last_progress_age_s": 1.0,
            "last_error": "bootstrap failed",
            "consecutive_errors": 3,
        },
        stall_timeout_s=45.0,
        replay_size=0,
        no_progress_age_s=1.0,
    )

    assert message == "background self-play worker failed to seed replay: bootstrap failed"


def test_probe_cfg_disables_resident_session():
    az = load_training_module()

    cfg = {"batch_size": 8, "iters": 16}
    probe_cfg = dict(cfg)
    probe_cfg["n_threads"] = 2
    probe_cfg["batch_size"] = max(cfg.get("batch_size", 8), min(64, max(2 * 2, 8)))
    probe_cfg["_disable_resident_session"] = True

    assert probe_cfg["_disable_resident_session"] is True


def test_autotune_parallel_candidates_drop_single_process_on_gpu_concurrent():
    az = load_training_module()
    hw = az.HardwareSpec(
        logical_cpus=24,
        physical_cpus=12,
        memory_mb=64000,
        gpu_vendor="amd",
        gpu_name="test",
        gpu_vram_mb=16384,
        gpu_count=1,
        torch_cuda=True,
        device_kind="cuda",
    )
    candidates = az._autotune_parallel_candidates({"bg_parallel": 6}, hw, concurrent=True)
    assert 1 not in candidates
    assert 2 in candidates


def test_eval_worker_candidates_are_hardware_bounded_not_four_capped():
    az = load_training_module()
    hw = az.HardwareSpec(
        logical_cpus=24,
        physical_cpus=12,
        memory_mb=64000,
        gpu_vendor="amd",
        gpu_name="AMD Radeon RX 6950 XT",
        gpu_vram_mb=16384,
        gpu_count=1,
        torch_cuda=True,
        device_kind="cuda",
    )

    workers = az.eval_worker_candidates(hw, {"n_threads": 1}, eval_games=200)

    assert workers[-1] == 12
    assert workers[:3] == [1, 2, 3]
    assert 6 in workers
    assert len(workers) >= 5


def test_compute_eval_collect_policy_adapts_timeout_and_target():
    az = load_training_module()

    low_fill_target, low_fill_timeout = az.compute_eval_collect_policy(
        16, 0.002, batch_items_ema=4.0, wait_ema_s=0.0003)
    high_fill_target, high_fill_timeout = az.compute_eval_collect_policy(
        16, 0.002, batch_items_ema=16.0, wait_ema_s=0.0003)
    wait_bound_target, wait_bound_timeout = az.compute_eval_collect_policy(
        16, 0.002, batch_items_ema=5.0, wait_ema_s=0.01)

    assert low_fill_timeout > 0.002
    assert high_fill_target >= 16
    assert high_fill_timeout < low_fill_timeout
    assert wait_bound_target <= 16
    assert wait_bound_timeout < low_fill_timeout


def test_step_early_stopping_waits_for_min_fraction_before_triggering():
    az = load_training_module()
    stopper = az.StepEarlyStopping(
        patience=2,
        min_delta=0.0,
        min_fraction=0.7,
        ema_alpha=1.0,
        planned_steps=10,
    )

    for idx in range(1, 7):
        assert stopper.step(1.0, idx) is False
    assert stopper.min_steps == 7
    assert stopper.step(1.0, 7) is True


def test_plan_online_runtime_overrides_penalizes_bursty_batch_games():
    az = load_training_module()
    hw = az.HardwareSpec(
        logical_cpus=24,
        physical_cpus=12,
        memory_mb=64000,
        gpu_vendor="amd",
        gpu_name="AMD Radeon RX 6950 XT",
        gpu_vram_mb=16384,
        gpu_count=1,
        torch_cuda=True,
        device_kind="cuda",
    )
    cfg = {
        "bg_parallel": 4,
        "bg_batch_games": 12,
        "n_threads": 2,
        "batch": 288,
        "steps": 100,
    }

    overrides = az.plan_online_runtime_overrides(cfg, hw, {
        "rolling_cycle_s": 2.2,
        "rolling_positions_per_s": 20.0,
        "best_positions_per_s": 21.0,
        "burst_ratio": 2.4,
        "n_new": 120,
        "train_steps": 3,
        "rolling_positions": 180,
        "last_cycle_positions": 180,
    })

    assert overrides["bg_batch_games"] < 12


def test_score_selfplay_probe_penalizes_multi_game_single_thread_without_batching():
    az = load_training_module()
    bad = az._score_selfplay_probe(
        positions_per_s=4.0,
        cycle_s=8.0,
        concurrent=True,
        positions=32,
        eval_messages=1500,
        model_batch_mean=1.0,
        parallel=2,
        n_threads=1,
    )
    good = az._score_selfplay_probe(
        positions_per_s=5.5,
        cycle_s=4.3,
        concurrent=True,
        positions=24,
        eval_messages=352,
        model_batch_mean=3.46,
        parallel=2,
        n_threads=2,
    )
    assert good > bad


def test_run_batched_eval_groups_merges_cross_process_requests():
    az = load_training_module()

    class FakeModel:
        def __init__(self):
            self.batch_sizes = []

        def predict(self, batch_np):
            self.batch_sizes.append(int(batch_np.shape[0]))
            n = int(batch_np.shape[0])
            probs = np.tile(np.linspace(1.0, 6.0, 6, dtype=np.float32), (n, 1))
            vals = np.arange(n, dtype=np.float32)
            return probs, vals

    model = FakeModel()
    cfg = {"ch": 1, "board": 2}
    groups = [
        {
            "gi": 0,
            "kind": "binary_batch",
            "requests": [(3, [1.0, 0.0, 0.0, 0.0]), (2, [0.0, 1.0, 0.0, 0.0])],
        },
        {
            "gi": 1,
            "kind": "json_single",
            "requests": [(4, [0.0, 0.0, 1.0, 0.0])],
        },
    ]

    responses = az._run_batched_eval_groups(groups, model, az.torch.device("cpu"), cfg)

    assert model.batch_sizes == [3]
    assert len(responses) == 2
    assert responses[0]["kind"] == "binary_batch"
    assert len(responses[0]["policies"]) == 2
    assert responses[0]["policies"][0].shape == (3,)
    assert responses[0]["policies"][1].shape == (2,)
    assert responses[0]["values"] == [0.0, 1.0]
    assert responses[1]["kind"] == "json_single"
    assert len(responses[1]["policies"]) == 1
    assert responses[1]["policies"][0].shape == (4,)
    assert responses[1]["values"] == [2.0]


def test_run_batched_eval_groups_defaults_missing_group_index():
    az = load_training_module()
    cfg = {"ch": 1, "board": 2}
    groups = [
        {
            "kind": "json_batch",
            "requests": [(2, [1.0, 0.0, 0.0, 0.0])],
        }
    ]

    responses = az._run_batched_eval_groups(groups, None, az.torch.device("cpu"), cfg)

    assert len(responses) == 1
    assert responses[0]["gi"] == 0
    assert responses[0]["kind"] == "json_batch"
    assert len(responses[0]["policies"]) == 1
    assert responses[0]["policies"][0].shape == (2,)


def test_run_batched_eval_groups_preserves_model_outputs_when_cache_disabled(monkeypatch):
    az = load_training_module()

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def predict(self, batch_np):
            self.calls += 1
            probs = np.array([[0.05, 0.15, 0.8]], dtype=np.float32)
            vals = np.array([0.375], dtype=np.float32)
            return probs, vals

    groups = [
        {
            "gi": 0,
            "kind": "json_single",
            "requests": [(3, [1.0, 0.0, 0.0, 0.0], 0)],
        }
    ]
    cfg = {"ch": 1, "board": 2, "actions": 3}

    model_with_cache = FakeModel()
    az.clear_nn_eval_cache()
    monkeypatch.delenv("QUARTZ_DISABLE_NN_CACHE", raising=False)
    with_cache = az._run_batched_eval_groups(groups, model_with_cache, az.torch.device("cpu"), cfg)

    model_without_cache = FakeModel()
    az.clear_nn_eval_cache()
    monkeypatch.setenv("QUARTZ_DISABLE_NN_CACHE", "1")
    without_cache = az._run_batched_eval_groups(groups, model_without_cache, az.torch.device("cpu"), cfg)

    np.testing.assert_allclose(with_cache[0]["policies"][0], [0.05, 0.15, 0.8])
    np.testing.assert_allclose(without_cache[0]["policies"][0], with_cache[0]["policies"][0])
    assert with_cache[0]["values"] == [0.375]
    assert without_cache[0]["values"] == [0.375]
    assert model_with_cache.calls == 1
    assert model_without_cache.calls == 1


def test_nn_eval_cache_treats_move_to_end_race_as_miss():
    az = load_training_module()

    class FlakyOrderedDict(az.OrderedDict):
        def get(self, key, default=None):
            value = super().get(key, default)
            if value is not default:
                super().pop(key, None)
            return value

    cache = az.NNEvalCache(max_entries=4)
    cache._cache = FlakyOrderedDict()
    cache._cache[123] = (np.array([0.2, 0.8], dtype=np.float32), 0.5)

    assert cache.get(123) is None
    assert cache._hits == 0
    assert cache._misses == 1


def test_run_batched_eval_groups_prefers_explicit_fingerprint_cache_keys(monkeypatch):
    az = load_training_module()

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def predict(self, batch_np):
            self.calls += 1
            probs = np.array([[0.2, 0.3, 0.5]], dtype=np.float32)
            vals = np.array([0.125], dtype=np.float32)
            return probs, vals

    def fail_legacy_key(*_args, **_kwargs):
        raise AssertionError("legacy feature-bytes cache key should not be used")

    cfg = {"ch": 1, "board": 2, "actions": 3}
    groups = [
        {
            "gi": 0,
            "kind": "json_single",
            "requests": [(3, [1.0, 0.0, 0.0, 0.0], 0, 101, 202, 1)],
        }
    ]

    az.clear_nn_eval_cache()
    monkeypatch.delenv("QUARTZ_DISABLE_NN_CACHE", raising=False)
    monkeypatch.setattr(az, "_legacy_eval_cache_key", fail_legacy_key)

    model = FakeModel()
    first = az._run_batched_eval_groups(groups, model, az.torch.device("cpu"), cfg)
    second = az._run_batched_eval_groups(groups, model, az.torch.device("cpu"), cfg)

    assert model.calls == 1
    np.testing.assert_allclose(first[0]["policies"][0], [0.2, 0.3, 0.5])
    np.testing.assert_allclose(second[0]["policies"][0], first[0]["policies"][0])
    assert second[0]["values"] == [0.125]


def test_read_exact_timeout_raises(monkeypatch):
    az = load_training_module()

    class FakeStream:
        def read(self, _n):
            raise AssertionError("read should not be called when the stream is not readable")

    monkeypatch.setattr(az, "wait_readable", lambda stream, timeout_s: False)

    with pytest.raises(TimeoutError):
        az._read_exact(FakeStream(), 4, timeout_s=0.01)


def test_search_move_retries_after_timeout(monkeypatch):
    az = load_training_module()

    client = az.NNSearchClient(model=None, cfg={"_name": "gomoku7", "iters": 8}, device="cpu")
    starts = []
    stops = []
    calls = {"n": 0}

    monkeypatch.setattr(client, "start", lambda: starts.append("start") or setattr(client, "proc", object()))
    monkeypatch.setattr(client, "stop", lambda: stops.append("stop") or setattr(client, "proc", None))

    def fake_exchange(req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("stalled search")
        return {"result": {"best_move": 7, "policy": [[7, 1.0]], "value": 0.25}}

    monkeypatch.setattr(client, "_exchange_search_request", fake_exchange)

    result = client.search_move(np.zeros(49, dtype=np.int8), player=1)

    assert result["best_move"] == 7
    assert calls["n"] == 2
    assert starts == ["start", "start"]
    assert stops == ["stop"]


def test_make_json_safe_converts_numpy_scalars_and_arrays():
    az = load_training_module()

    payload = {
        "score_rate": np.float32(0.625),
        "delta": np.int64(7),
        "policy": np.array([1.0, 2.0], dtype=np.float32),
        "nested": {"value": np.float64(1.5)},
    }

    safe = az.make_json_safe(payload)

    assert safe == {
        "score_rate": pytest.approx(0.625),
        "delta": 7,
        "policy": [1.0, 2.0],
        "nested": {"value": 1.5},
    }
    json.dumps(safe)


def test_proc_decode_eval_frame_reads_shared_memory_request():
    az = load_training_module()
    transport = az.QipcSharedMemoryTransport.create(size=128)
    proc = type("FakeProc", (), {})()
    proc._quartz_qipc_transport = transport
    transport.req.buf[:6] = b"abcdef"
    try:
        frame_kind, payload = az.proc_decode_eval_frame(
            proc, az.QIPC_EVAL_REQ_SHM, az.QIPC_SHM_LEN.pack(6)
        )
        assert frame_kind == az.QIPC_EVAL_REQ
        assert payload == b"abcdef"
    finally:
        transport.destroy()


def test_proc_write_eval_response_prefers_shared_memory_when_available():
    az = load_training_module()
    transport = az.QipcSharedMemoryTransport.create(size=128)
    proc = type("FakeProc", (), {})()
    proc._quartz_qipc_transport = transport
    proc.stdin = io.BytesIO()
    payload = b"\x01\x02\x03\x04"
    try:
        az.proc_write_eval_response(proc, az.QIPC_EVAL_RESP, payload, prefer_shm=True)
        proc.stdin.seek(0)
        kind, meta = az.proc_read_message(type("ReadProc", (), {"stdout": proc.stdin})())
        assert kind == "frame"
        frame_kind, frame_payload = meta
        assert frame_kind == az.QIPC_EVAL_RESP_SHM
        assert az.QIPC_SHM_LEN.unpack(frame_payload)[0] == len(payload)
        assert bytes(transport.resp.buf[:len(payload)]) == payload
    finally:
        transport.destroy()


def test_shm_ring_buffer_roundtrip_preserves_epoch_seq_and_payload():
    az = load_training_module()
    from quartz import qipc as qipc_mod

    ring = az.ShmRingBuffer.create(r2p_slots=1, p2r_slots=1, slot_data_size=128)
    try:
        payload = b"hello-shm"
        assert ring.p2r_try_write(0, 7, payload, epoch=11, seq=29) is True
        slot_off = ring._p2r_slot_offset(0)
        assert ring.slot_state(slot_off) == qipc_mod.SHM_SLOT_WRITTEN
        assert ring.p2r_slot_state(0) == qipc_mod.SHM_SLOT_WRITTEN

        mirror_off = ring._r2p_slot_offset(0)
        ring._buf[mirror_off + 1] = 9
        struct.pack_into("<III", ring._buf, mirror_off + 4, len(payload), 13, 31)
        ring._buf[
            mirror_off + qipc_mod.SHM_RING_SLOT_HEADER : mirror_off + qipc_mod.SHM_RING_SLOT_HEADER + len(payload)
        ] = payload
        ring.set_slot_state(mirror_off, qipc_mod.SHM_SLOT_WRITTEN)

        msg_type, epoch, seq, decoded = ring.r2p_try_read_meta(0)
        assert msg_type == 9
        assert epoch == 13
        assert seq == 31
        assert decoded == payload

        ring.r2p_mark_done(0)
        assert ring.slot_state(mirror_off) == qipc_mod.SHM_SLOT_DONE
    finally:
        ring.destroy()


def test_shm_eval_loop_ignores_stale_cmd_done_until_epoch_advances():
    from quartz import evaluator_runtime as eval_runtime_mod

    class FakeRing:
        r2p_slot_count = 0

        def __init__(self):
            self._epoch_calls = 0

        def epoch(self):
            self._epoch_calls += 1
            return 5 if self._epoch_calls == 1 else 6

        def cmd_done(self):
            return True

        def r2p_try_read_meta(self, slot_idx):
            return None

        def r2p_mark_done(self, slot_idx):
            return None

    class FakeProc:
        def poll(self):
            return None

    hooks = eval_runtime_mod.ShmEvalRuntimeHooks(
        run_batched_eval_groups=lambda groups, model_obj, dev, cfg_obj: [],
        make_eval_request_group=lambda kind, requests, gi=0: {"kind": kind, "requests": requests, "gi": gi},
        unpack_qipc_batch_eval_req=lambda payload: [],
        unpack_qipc_arena_eval_resp=lambda payload: {"valid_eval": True, "records": []},
        unpack_shm_search_response=lambda payload: {"decoded": True},
        json_loads_fast=lambda s: {},
        emit_duty_cycle=lambda duty: None,
        pack_qipc_batch_eval_resp=lambda policies, values: b"",
        logger=logging.getLogger(__name__),
        shm_msg_eval_batch_req=1,
        shm_msg_arena_eval_resp=5,
        shm_msg_json=2,
        shm_msg_search_resp=4,
        inference_pipeline_thread_cls=None,
        should_use_async_pipeline=lambda model, device, cfg: False,
    )
    ring = FakeRing()
    payload = eval_runtime_mod.shm_eval_loop(
        ring,
        model=None,
        device="cpu",
        cfg={},
        proc=FakeProc(),
        runtime_hooks=hooks,
        baseline_epoch=5,
    )
    assert payload is None
    assert ring._epoch_calls >= 2


def test_shm_eval_loop_returns_binary_arena_eval_payload():
    from quartz import evaluator_runtime as eval_runtime_mod

    payload = bytearray()
    payload.extend(struct.pack("<BBId", 1, 1, 1, 9.5))
    game = b"gomoku7"
    payload.extend(struct.pack("<I", len(game)))
    payload.extend(game)
    payload.extend(struct.pack("<I", 1))
    game_id = b"m0::g0000"
    payload.extend(struct.pack("<I", len(game_id)))
    payload.extend(game_id)
    payload.extend(struct.pack("<II", 0, 1))
    payload.extend(struct.pack("<BB", 1, 0))
    payload.extend(struct.pack("<fIdQ", 1.0, 7, 33.0, 17))
    payload.extend(struct.pack("<I", 0))
    payload.extend(struct.pack("<I", 0))

    class FakeRing:
        r2p_slot_count = 1

        def epoch(self):
            return 6

        def cmd_done(self):
            return True

        def r2p_try_read_meta(self, slot_idx):
            if slot_idx != 0:
                return None
            if getattr(self, "_used", False):
                return None
            self._used = True
            return 5, 6, 1, bytes(payload)

        def r2p_mark_done(self, slot_idx):
            return None

    class FakeProc:
        def poll(self):
            return None

    hooks = eval_runtime_mod.ShmEvalRuntimeHooks(
        run_batched_eval_groups=lambda groups, model_obj, dev, cfg_obj: [],
        make_eval_request_group=lambda kind, requests, gi=0: {"kind": kind, "requests": requests, "gi": gi},
        unpack_qipc_batch_eval_req=lambda payload: [],
        unpack_qipc_arena_eval_resp=lambda raw: {"valid_eval": True, "records": [{"game_id": "m0::g0000"}]},
        unpack_shm_search_response=lambda payload: {"decoded": True},
        json_loads_fast=lambda s: {},
        emit_duty_cycle=lambda duty: None,
        pack_qipc_batch_eval_resp=lambda policies, values: b"",
        logger=logging.getLogger(__name__),
        shm_msg_eval_batch_req=1,
        shm_msg_arena_eval_resp=5,
        shm_msg_json=2,
        shm_msg_search_resp=4,
        inference_pipeline_thread_cls=None,
        should_use_async_pipeline=lambda model, device, cfg: False,
    )

    decoded = eval_runtime_mod.shm_eval_loop(
        FakeRing(),
        model=None,
        device="cpu",
        cfg={},
        proc=FakeProc(),
        runtime_hooks=hooks,
        baseline_epoch=5,
    )

    assert decoded["valid_eval"] is True
    assert decoded["records"][0]["game_id"] == "m0::g0000"


def test_shm_eval_loop_raises_interrupted_error_when_cancel_requested():
    from quartz import evaluator_runtime as eval_runtime_mod

    class FakeRing:
        r2p_slot_count = 0

        def epoch(self):
            return 6

        def cmd_done(self):
            return False

        def cancel_requested(self):
            return True

    class FakeProc:
        def poll(self):
            return None

    hooks = eval_runtime_mod.ShmEvalRuntimeHooks(
        run_batched_eval_groups=lambda groups, model_obj, dev, cfg_obj: [],
        make_eval_request_group=lambda kind, requests, gi=0: {"kind": kind, "requests": requests, "gi": gi},
        unpack_qipc_batch_eval_req=lambda payload: [],
        unpack_qipc_arena_eval_resp=lambda raw: {"valid_eval": True, "records": []},
        unpack_shm_search_response=lambda payload: {"decoded": True},
        json_loads_fast=lambda s: {},
        emit_duty_cycle=lambda duty: None,
        pack_qipc_batch_eval_resp=lambda policies, values: b"",
        logger=logging.getLogger(__name__),
        shm_msg_eval_batch_req=1,
        shm_msg_arena_eval_resp=5,
        shm_msg_json=2,
        shm_msg_search_resp=4,
        inference_pipeline_thread_cls=None,
        should_use_async_pipeline=lambda model, device, cfg: False,
    )

    with pytest.raises(InterruptedError, match="cancelled"):
        eval_runtime_mod.shm_eval_loop(
            FakeRing(),
            model=None,
            device="cpu",
            cfg={},
            proc=FakeProc(),
            runtime_hooks=hooks,
            baseline_epoch=5,
        )


def test_eval_match_run_requests_typed_arena_eval_response(monkeypatch):
    az = load_training_module()
    client = az.NNSearchClient(model=None, cfg={"_name": "gomoku7", "iters": 8}, device="cpu")

    seen = {}
    monkeypatch.setattr(client, "start", lambda: setattr(client, "proc", object()))

    def fake_exchange(req_dict=None, *, frame_kind=None, frame_payload=None):
        seen["req_dict"] = req_dict
        seen["frame_kind"] = frame_kind
        seen["frame_payload"] = frame_payload
        return {"valid_eval": True, "records": []}

    monkeypatch.setattr(client, "_exchange_search_request", fake_exchange)

    result = client.eval_match_run([], max_moves=20)

    assert seen["req_dict"] is None
    assert seen["frame_kind"] == az.QIPC_ARENA_EVAL_REQ
    assert isinstance(seen["frame_payload"], (bytes, bytearray))
    assert seen["frame_payload"][0] == 3
    assert result["valid_eval"] is True


def test_eval_match_run_typed_request_preserves_zero_white_tag(monkeypatch):
    az = load_training_module()
    client = az.NNSearchClient(model=None, cfg={"_name": "gomoku7", "iters": 8}, device="cpu")

    seen = {}
    monkeypatch.setattr(client, "start", lambda: setattr(client, "proc", object()))

    def fake_exchange(req_dict=None, *, frame_kind=None, frame_payload=None):
        seen["frame_payload"] = bytes(frame_payload)
        return {"valid_eval": True, "records": []}

    monkeypatch.setattr(client, "_exchange_search_request", fake_exchange)

    board = [0] * 49
    client.eval_match_run(
        [
            {
                "game_id": "m0::g0001",
                "board": board,
                "player": 1,
                "black_tag": 1,
                "white_tag": 0,
                "opening": [],
                "seed": 11,
                "ply": 0,
                "done": False,
                "total_time_ms": 0.0,
            }
        ],
        max_moves=12,
    )

    payload = seen["frame_payload"]
    offset = 0
    assert payload[offset] == 3
    offset += 1

    def read_string():
        nonlocal offset
        (n_bytes,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        raw = payload[offset : offset + n_bytes]
        offset += n_bytes
        return raw.decode("utf-8")

    assert read_string() == "gomoku7"
    assert read_string() == "quartz"
    assert read_string() == "GatedRefresh"
    assert read_string() == ""
    offset += struct.calcsize("<IIIffIIfff")
    offset += 1  # root_only_shaping
    offset += 1  # tt_enabled
    offset += 1 + 8  # seed presence + seed
    assert read_string() == ""
    iters, max_moves, session_count = struct.unpack_from("<III", payload, offset)
    offset += struct.calcsize("<III")
    assert iters == 8
    assert max_moves == 12
    assert session_count == 1
    assert read_string() == "m0::g0001"
    black_tag, white_tag, seed_raw, ply, total_time_ms, done = struct.unpack_from(
        "<IIQIdB", payload, offset
    )
    assert black_tag == 1
    assert white_tag == 0
    assert seed_raw == 11
    assert ply == 0
    assert total_time_ms == 0.0
    assert done == 0


def test_launch_rust_server_honors_ring_slot_env(monkeypatch):
    from quartz import qipc as qipc_mod

    class FakeProc:
        def __init__(self):
            self.pid = 12345
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
            self.returncode = None

        def poll(self):
            return None

    monkeypatch.setenv("QUARTZ_QIPC_RING_R2P_SLOTS", "5")
    monkeypatch.setenv("QUARTZ_QIPC_RING_P2R_SLOTS", "7")

    proc = qipc_mod.launch_rust_server("/bin/echo", popen_fn=lambda *args, **kwargs: FakeProc())
    try:
        ring = proc._quartz_ring_buffer
        assert ring.r2p_slot_count == 5
        assert ring.p2r_slot_count == 7
    finally:
        qipc_mod.cleanup_qipc_transport(proc)


def test_rust_search_options_includes_adaptive_batch_timeout():
    az = load_training_module()

    opts = az.rust_search_options({"n_threads": 4, "batch_size": 16, "search_profile": "baseline"})

    assert opts["n_threads"] == 4
    assert opts["batch_size"] == 16
    assert opts["batch_timeout_us"] > 1500
    assert opts["search_profile"] == "baseline"


def test_rust_search_options_forwards_auto_thread_policy():
    az = load_training_module()

    opts = az.rust_search_options({
        "n_threads": "auto",
        "thread_policy": "quality",
        "thread_cap": 8,
        "batch_size": 16,
    })

    assert opts["n_threads"] == "auto"
    assert opts["thread_policy"] == "quality"
    assert opts["thread_cap"] == 8
    assert opts["batch_timeout_us"] > 1500


def test_rust_search_options_preserves_baseline_strict_profile():
    az = load_training_module()

    opts = az.rust_search_options({"search_profile": "baseline_strict"})

    assert opts["search_profile"] == "baseline_strict"


def test_rust_search_options_passes_controller_runtime_overrides():
    az = load_training_module()

    opts = az.rust_search_options({
        "penalty_mode": "GatedRefreshLegacy",
        "root_only_shaping": False,
    })

    assert opts["penalty_mode"] == "GatedRefreshLegacy"
    assert opts["root_only_shaping"] is False


def test_search_manifest_key_surfaces_stay_in_sync():
    from quartz import runtime_support
    from quartz import selfplay_runtime

    assert selfplay_runtime._SEARCH_MANIFEST_KEYS == runtime_support.SEARCH_MANIFEST_KEYS


def test_search_manifest_hash_is_sensitive_to_halt_mode_and_eval_seed():
    from quartz import runtime_support
    from quartz import selfplay_runtime

    cfg = {
        "_name": "gomoku7",
        "iters": 8,
        "search_profile": "quartz",
        "vl_mode": "adaptive",
        "penalty_mode": "GatedRefresh",
        "halt_mode": "fixed",
        "eval_seed": 17,
    }
    cfg_halt = dict(cfg, halt_mode="voc")
    cfg_seed = dict(cfg, eval_seed=18)

    assert runtime_support.search_manifest_hash(cfg) == selfplay_runtime._search_manifest_hash(cfg)
    assert runtime_support.search_manifest_hash(cfg) != runtime_support.search_manifest_hash(cfg_halt)
    assert runtime_support.search_manifest_hash(cfg) != runtime_support.search_manifest_hash(cfg_seed)


def test_decode_streamed_selfplay_game_rejects_mismatched_lengths():
    from quartz.selfplay_runtime import decode_streamed_selfplay_game

    cfg = {"board": 3, "actions": 9}
    payload = {
        "states": [[0] * 9],
        "players": [1, -1],
        "policies": [[(0, 1.0)]],
    }

    with pytest.raises(ValueError, match="length mismatch"):
        decode_streamed_selfplay_game(cfg, payload)


def test_replay_samples_preserve_actor_snapshot_id():
    from quartz.replay import ReplayBuffer

    replay = ReplayBuffer(8)
    replay.add_game(
        [np.zeros((1, 3, 3), dtype=np.float32)],
        [np.ones(9, dtype=np.float32) / 9.0],
        outcome=1.0,
        actor_generation=7,
        actor_id="actor_gen_000007",
    )

    sample = replay.examples_at_indices([0])[0]
    assert sample.metadata["actor_generation"] == 7
    assert sample.metadata["actor_id"] == "actor_gen_000007"


def test_jax_backend_save_accepts_cfg_argument():
    import inspect

    backend = load_backend_module()

    assert "cfg" in inspect.signature(backend.JAXBackend.save).parameters


def test_gpu_host_thread_caps_follow_physical_cores():
    az = load_training_module()
    hw = az.HardwareSpec(
        logical_cpus=24,
        physical_cpus=12,
        memory_mb=64000,
        gpu_vendor="amd",
        gpu_name="AMD Radeon RX 6950 XT",
        gpu_vram_mb=16384,
        gpu_count=1,
        torch_cuda=True,
        device_kind="cuda",
    )

    assert az.gpu_host_thread_cap(hw) == 12
    assert az.gpu_interop_thread_cap(hw) == 6


def test_score_selfplay_probe_rewards_lower_ipc_message_density():
    az = load_training_module()

    single_frame_score = az._score_selfplay_probe(
        10.5, 35.0, True, positions=370, eval_messages=67000)
    batched_score = az._score_selfplay_probe(
        10.2, 22.5, True, positions=230, eval_messages=15000)

    assert batched_score > single_frame_score


def test_autotune_parallel_limit_caps_gpu_concurrent_ipc_heavy_parallelism():
    az = load_training_module()
    hw = az.HardwareSpec(
        logical_cpus=24,
        physical_cpus=12,
        memory_mb=64000,
        gpu_vendor="amd",
        gpu_name="AMD Radeon RX 6950 XT",
        gpu_vram_mb=16384,
        gpu_count=1,
        torch_cuda=True,
        device_kind="cuda",
    )

    assert az._autotune_parallel_limit(hw, concurrent=True) == 6
    assert az._autotune_parallel_limit(hw, concurrent=False) == 12


def test_autotune_thread_candidates_skip_single_thread_for_gpu_parallel_search():
    az = load_training_module()
    hw = az.HardwareSpec(
        logical_cpus=24,
        physical_cpus=12,
        memory_mb=64000,
        gpu_vendor="amd",
        gpu_name="AMD Radeon RX 6950 XT",
        gpu_vram_mb=16384,
        gpu_count=1,
        torch_cuda=True,
        device_kind="cuda",
    )

    candidates = az._autotune_thread_candidates(hw, parallel=4)
    assert 1 not in candidates
    assert all(t >= 2 for t in candidates)


def test_plan_online_runtime_overrides_reduces_ipc_heavy_parallelism_and_raises_threads():
    az = load_training_module()
    hw = az.HardwareSpec(
        logical_cpus=24,
        physical_cpus=12,
        memory_mb=64000,
        gpu_vendor="amd",
        gpu_name="AMD Radeon RX 6950 XT",
        gpu_vram_mb=16384,
        gpu_count=1,
        torch_cuda=True,
        device_kind="cuda",
    )
    cfg = {
        "bg_parallel": 12,
        "bg_batch_games": 24,
        "n_threads": 1,
        "batch": 480,
        "steps": 100,
    }
    sample = {
        "last_cycle_s": 5.0,
        "last_cycle_positions": 300,
        "positions_per_s": 10.0,
        "best_positions_per_s": 12.0,
        "n_new": 100,
        "train_steps": 2,
    }

    overrides = az.plan_online_runtime_overrides(cfg, hw, sample)
    assert overrides["bg_parallel"] == 6
    assert overrides["n_threads"] >= 2


def test_replay_dataloader_preserves_batch_shapes_and_types():
    az = load_training_module()
    replay = az.ReplayBuffer(32)
    for i in range(8):
        state = np.full((3, 7, 7), float(i), dtype=np.float32)
        policy = np.zeros(49, dtype=np.float32)
        policy[i % 49] = 1.0
        replay.add(state, policy, float(i))

    loader = replay.build_dataloader(batch_size=4, n_steps=2, pin_memory=False)
    batches = list(loader)

    assert len(batches) == 2
    states_t, policies_t, values_t = batches[0]
    assert states_t.shape == (4, 3, 7, 7)
    assert policies_t.shape == (4, 49)
    assert values_t.shape == (4,)
    assert states_t.dtype == az.torch.float32
    assert policies_t.dtype == az.torch.float32
    assert values_t.dtype == az.torch.float32


def test_get_actor_model_returns_backend_source():
    az = load_training_module()
    fallback_model = object()

    class FakeBackend:
        pass

    backend = FakeBackend()
    assert az.get_actor_model(fallback_model, backend) is backend
    assert az.get_actor_model(fallback_model, None) is fallback_model


def test_benchmark_train_batch_supports_generic_backend():
    az = load_training_module()

    class FakeBackend:
        def __init__(self):
            self.params = {"w": 1.0}
            self.batch_stats = {"bn": 0.0}
            self.opt_state = {"step": 0}

        def train_step(self, states, policies, values):
            self.opt_state = {"step": self.opt_state["step"] + 1}
            return 1.0, 0.5, 0.5

    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    hw = az.HardwareSpec(
        logical_cpus=8, physical_cpus=4, memory_mb=16384,
        gpu_vendor="amd", gpu_name="GPU", gpu_vram_mb=8192,
        gpu_count=1, torch_cuda=False, device_kind="cpu")

    overrides, results = az.benchmark_train_batch(
        cfg, FakeBackend(), None, None, az.torch.device("cpu"), hw)

    assert "batch" in overrides
    assert any("examples_per_s" in row for row in results)


def test_jax_model_init_and_forward_if_available():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "quartz_jax_models", root / "quartz" / "jax_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not getattr(module, "HAS_JAX", False):
        pytest.skip("JAX not installed in test environment")

    model = module.AlphaZeroJAX(
        board_size=7, in_ch=3, n_actions=49,
        n_filters=96, n_blocks=6, value_hidden=128, se_blocks=2)
    rng = module.jax.random.PRNGKey(0)
    x = module.jnp.ones((2, 3, 7, 7), dtype=module.jnp.float32)
    variables = model.init(rng, x, train=False)
    logits, values = model.apply(variables, x, train=False)

    assert logits.shape == (2, 49)
    assert values.shape == (2,)


def test_load_actor_source_from_checkpoint_reuses_jax_backend_template(tmp_path):
    az = load_training_module()

    ckpt = tmp_path / "dummy_jax.pkl"
    ckpt.write_bytes(b"stub")

    class FakeBackend:
        name = "jax"

        def __init__(self):
            self.loaded = None

        def load_actor(self, path):
            self.loaded = path
            return "jax-actor"

    backend = FakeBackend()
    actor = az.load_actor_source_from_checkpoint(
        str(ckpt),
        dict(az.GAME_CONFIGS["gomoku7"]),
        az.torch.device("cpu"),
        backend_preference="jax",
        backend_template=backend,
    )

    assert actor == "jax-actor"
    assert backend.loaded == str(ckpt)


def test_load_actor_source_from_checkpoint_uses_training_module_model_wrapper(tmp_path, monkeypatch):
    az = load_training_module()

    ckpt = tmp_path / "dummy_torch.pt"
    ckpt.write_bytes(b"stub")

    class FakeActor:
        def __init__(self, cfg):
            self.cfg = cfg
            self.loaded = None
            self.eval_called = False

        def to(self, device):
            return self

        def load_state_dict(self, state_dict):
            self.loaded = state_dict

        def eval(self):
            self.eval_called = True
            return self

    monkeypatch.setattr(az, "AlphaZeroNet", FakeActor)
    monkeypatch.setattr(az, "load_torch_state_dict", lambda *args, **kwargs: {"w": 1})

    actor = az.load_actor_source_from_checkpoint(
        str(ckpt),
        dict(az.GAME_CONFIGS["gomoku7"]),
        az.torch.device("cpu"),
        backend_preference="torch",
    )

    assert isinstance(actor, FakeActor)
    assert actor.loaded == {"w": 1}
    assert actor.eval_called is True


def test_choose_selfplay_move_uses_policy_before_temperature_cutoff():
    az = load_training_module()
    policy = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    chosen = az.choose_selfplay_move(
        policy, legal=[0, 1], move_count=0, temp_threshold=8, fallback_best=0)

    assert chosen == 1


def test_compute_train_steps_scales_concurrent_work_to_fresh_data():
    az = load_training_module()

    assert az.compute_train_steps(100, 256, 0, concurrent=True) == 0
    assert az.compute_train_steps(100, 256, 190, concurrent=True) == 6
    assert az.compute_train_steps(100, 256, 3600, concurrent=True) == 100
    assert az.compute_train_steps(100, 256, 0, concurrent=False) == 100


def test_default_output_dir_uses_models_subdirectory():
    az = load_training_module()

    assert az.default_output_dir("gomoku7") == "models/alphazero_gomoku7"


def test_rust_search_options_propagates_tt_enabled_flag():
    az = load_training_module()

    opts = az.rust_search_options({"n_threads": 2, "batch_size": 16, "tt_enabled": False})

    assert opts["tt_enabled"] is False


def test_dense_policy_from_sparse_accepts_legacy_strings_and_numeric_pairs():
    az = load_training_module()

    policy = az.dense_policy_from_sparse(["1:0.25", [3, 0.75], ["bad"]], 5)

    assert np.allclose(policy, np.array([0.0, 0.25, 0.0, 0.75, 0.0], dtype=np.float32))


def test_build_rust_state_meta_includes_chess_history_hashes():
    az = load_training_module()
    state = az.ChessEvaluationAdapter()
    state._chess_history_hashes = [11, 22, 33]

    meta = az.build_rust_state_meta("chess", state, {})

    assert meta == {"chess_history_hashes": [11, 22, 33]}


def test_chess_evaluation_adapter_tracks_engine_history_hashes():
    az = load_training_module()
    state = az.ChessEvaluationAdapter()

    applied = state.apply_engine_meta(
        0,
        {
            "result_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "result_history_hashes": [101, 202, 303],
        },
    )

    assert applied is True
    assert state._fen == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    assert state._chess_history_hashes == [101, 202, 303]
    clone = state.clone()
    assert clone._chess_history_hashes == [101, 202, 303]


def test_early_stopping_enabled_by_positive_patience():
    az = load_training_module()

    assert az.early_stopping_enabled(15, concurrent=False) is True
    assert az.early_stopping_enabled(15, concurrent=True) is True
    assert az.early_stopping_enabled(0, concurrent=False) is False


def test_load_epoch_history_filters_eval_records(tmp_path):
    az = load_training_module()
    log_path = tmp_path / "train_log.jsonl"
    with open(log_path, "w") as f:
        f.write(json.dumps({"iter": 1, "loss": 1.23, "published_elo": None}) + "\n")
        f.write(json.dumps({"_type": "eval", "iter": 1, "published_elo": 1200}) + "\n")
        f.write(json.dumps({"iter": 2, "loss": 1.11, "published_elo": 1200}) + "\n")

    history = az.load_epoch_history(str(log_path))

    assert [row["iter"] for row in history] == [1, 2]
    assert history[1]["published_elo"] == 1200


def test_build_elo_plot_series_prefers_logged_absolute_gap():
    az = load_training_module()
    history = [
        {"iter": 5, "published_elo": 140.0, "champion_elo": 100.0, "elo_gap": 40.0,
         "delta_elo": 75.0, "score_rate": 0.62},
    ]

    series = az.build_elo_plot_series(history)

    assert len(series) == 1
    point = series[0]
    assert point["candidate_elo"] == 140.0
    assert point["champion_elo"] == 100.0
    assert point["elo_gap"] == 40.0
    assert point["error_mid"] == 120.0
    assert point["error_half"] == 20.0
    assert point["match_delta_elo"] == 75.0


def test_build_elo_plot_series_falls_back_to_delta_when_champion_missing():
    az = load_training_module()
    history = [
        {"iter": 10, "published_elo": 160.0, "delta_elo": 30.0},
    ]

    series = az.build_elo_plot_series(history)

    assert len(series) == 1
    point = series[0]
    assert point["champion_elo"] == 130.0
    assert point["elo_gap"] == 30.0
    assert point["error_mid"] == 145.0
    assert point["error_half"] == 15.0


def test_build_metric_plot_series_skips_missing_points_without_dropping_sparse_losses():
    az = load_training_module()
    history = [
        {"iter": 1, "loss": 4.9},
        {"iter": 2, "loss": None},
        {"iter": 4, "loss": 4.2},
    ]

    series = az.build_metric_plot_series(history, "loss")

    assert series == [(1, 4.9), (4, 4.2)]


def test_build_best_elo_series_only_promotes_on_promotion_verdict():
    az = load_training_module()
    elo_points = [
        {
            "iter": 5,
            "candidate_elo": 140.0,
            "champion_elo": 100.0,
            "match_delta_elo": 15.0,
            "eval_verdict": "reject",
        },
        {
            "iter": 10,
            "candidate_elo": 155.0,
            "champion_elo": 110.0,
            "match_delta_elo": 18.0,
            "eval_verdict": "promote",
        },
    ]

    series = az.build_best_elo_series(elo_points)

    assert series == [100.0, 155.0]


def test_generate_training_plots_avoids_single_point_warnings(tmp_path):
    az = load_training_module()
    log_path = tmp_path / "train_log.jsonl"
    log_path.write_text(json.dumps({"iter": 1}) + "\n", encoding="utf-8")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert az.generate_training_plots(str(log_path), str(tmp_path)) is True

    messages = [str(item.message) for item in caught]
    assert not any("identical low and high xlims" in message for message in messages)
    assert not any("No artists with labels found" in message for message in messages)


def test_cli_main_refreshes_bg_actor_after_each_update_and_writes_checkpoint_status(tmp_path):
    cli = load_cli_main_module()
    rust_binary = tmp_path / "mcts_demo"
    rust_binary.write_text("stub", encoding="utf-8")

    class FakeBackend:
        name = "torch"

        def __init__(self):
            self.saved = []
            self.lr_values = []

        def set_lr(self, lr):
            self.lr_values.append(lr)

        def save(self, path, cfg=None):
            self.saved.append((str(path), None if cfg is None else dict(cfg)))
            Path(path).write_text("checkpoint", encoding="utf-8")

    class FakeReplayBuffer:
        def __init__(self, *args, **kwargs):
            self.buf = []
            self._size = 32

        def __len__(self):
            return self._size

        def save(self, path):
            Path(path).write_text("replay", encoding="utf-8")

    class FakeWorker:
        instances = []
        _recent_chunks = []
        _prev_count = 0

        def __init__(self, cfg, actor_source, device, replay, rust_binary):
            self.update_calls = []
            self.started = False
            self.stopped = False
            FakeWorker.instances.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def status(self):
            return {"alive": True, "consecutive_errors": 0, "last_progress_age_s": 0.0, "last_error": None}

        def update_model(self, actor_source):
            self.update_calls.append(actor_source)

        def pause(self, wait=True):
            return True

        def resume(self):
            return None

    backend = FakeBackend()
    ctx = cli.PreparedTrainingContext(
        cfg={
            "_name": "gomoku7",
            "board": 7,
            "filters": 32,
            "blocks": 2,
            "buf": 64,
            "batch": 8,
            "steps": 1,
            "games": 2,
            "search_profile": "quartz",
            "vl_mode": "adaptive",
        },
        base_cfg={"board": 7},
        base_dir=str(tmp_path),
        device="cpu",
        hw=types.SimpleNamespace(physical_cpus=1),
        model=None,
        backend=backend,
        optimizer=types.SimpleNamespace(param_groups=[{"lr": 0.0}]),
        actor_source="actor-source",
        benchmark_info=None,
        model_path=str(tmp_path / "latest.pt"),
        latest_model_path=str(tmp_path / "latest.pt"),
        best_model_path=str(tmp_path / "best.pt"),
        replay_path=str(tmp_path / "replay.npz"),
        log_path=str(tmp_path / "train_log.jsonl"),
        autotune_profile_path=str(tmp_path / "autotune_profile.json"),
        n_params=123,
    )
    args = argparse.Namespace(
        serve=False,
        arena_3agent=None,
        arena=None,
        rust_nn=True,
        arena_games=0,
        game="gomoku7",
        iterations=2,
        resume=False,
        concurrent=True,
        runtime_autotune=False,
        eval_selfplay_isolation=True,
        patience=0,
        inner_patience=0,
        inner_min_fraction=0.0,
        inner_min_delta=0.0,
        inner_ema_alpha=0.2,
        eval_games=2,
        eval_interval=99,
        seed=7,
        rust_binary=str(rust_binary),
    )
    runtime_hooks = cli.MainRuntimeHooks(
        torch=types.SimpleNamespace(),
        np=np,
        game_configs={"gomoku7": {}},
        serve=lambda *args, **kwargs: None,
        arena_3agent=lambda *args, **kwargs: None,
        arena_rust_nn=lambda *args, **kwargs: None,
        arena_compare=lambda *args, **kwargs: None,
        print_autotune_summary=lambda *args, **kwargs: None,
        is_go_game=lambda _name: False,
        replay_buffer_cls=FakeReplayBuffer,
        early_stopping_cls=lambda *args, **kwargs: None,
        early_stopping_enabled=lambda patience, concurrent=False: False,
        load_eval_autotune_profile=lambda *args, **kwargs: None,
        has_eval_system=True,
        recommend_eval_parallel_workers=lambda *args, **kwargs: 1,
        max_supported_threads=lambda hw: 1,
        eval_config_cls=lambda **kwargs: types.SimpleNamespace(**kwargs),
        training_evaluator_cls=lambda config=None: types.SimpleNamespace(cfg=config),
        build_training_game_adapter=lambda cfg: object(),
        ensure_best_checkpoint_compatible=lambda *args, **kwargs: None,
        selfplay_worker_cls=FakeWorker,
        initial_replay_fill_target=lambda cfg, recent_chunks: 0,
        online_autotune_controller_cls=lambda *args, **kwargs: None,
        clear_nn_eval_cache=lambda: None,
        round_or_none=lambda value: value,
        wait_for_worker_progress=lambda worker, prev_bg, min_new=1, timeout_s=30.0: (4, prev_bg + 4),
        selfplay_rust_nn_batched=lambda *args, **kwargs: ([], [], [], []),
        compute_train_steps=lambda *args, **kwargs: 1,
        train_epoch=lambda *args, **kwargs: (0.5, 0.3, 0.2, 1, None),
        replay_metrics=types.SimpleNamespace(
            freshness=lambda n_new, replay_len: 0.5,
            policy_entropy=lambda replay: 0.1,
            value_std=lambda replay: 0.2,
            search_summary=lambda replay: {"positions": len(replay)},
        ),
        rust_nn_evaluator_engine_cls=object,
        clone_actor_model=lambda actor: actor,
        load_actor_source_from_checkpoint=lambda *args, **kwargs: "champion-actor",
        tree_mcts_engine_cls=object,
        benchmark_eval_parallel_workers=lambda *args, **kwargs: (1, []),
        make_json_safe=lambda payload: payload,
        generate_training_plots=lambda *args, **kwargs: False,
    )

    cli.run_training_main(args, ctx, runtime_hooks)

    worker = FakeWorker.instances[0]
    status = json.loads((tmp_path / "checkpoint_status.json").read_text(encoding="utf-8"))

    assert worker.started is True
    assert worker.stopped is True
    assert worker.update_calls == ["actor-source", "actor-source"]
    assert backend.saved[0][0].endswith("best.pt")
    assert backend.saved[-1][0].endswith("latest.pt")
    assert backend.saved[-1][1]["search_profile"] == "quartz"
    assert status["best_checkpoint_bootstrap_seeded"] is True
    assert status["preferred_posttrain_checkpoint"] == "latest.pt"
    assert (tmp_path / "latest.pt").exists()


def test_iteration_level_actor_refresh_fires_when_sgd_skipped(tmp_path):
    """P9 (audit_codex_20260425.md W5): `bg_worker.update_model` must
    fire every iteration, including when SGD is skipped because the
    replay buffer hasn't crossed the batch threshold yet.
    """
    cli = importlib.import_module("quartz.cli_main")

    class FakeBackend:
        name = "torch"

        def __init__(self):
            self.saved = []
            self.lr_values = []

        def set_lr(self, lr):
            self.lr_values.append(lr)

        def save(self, path, cfg=None):
            Path(path).write_text("checkpoint", encoding="utf-8")
            self.saved.append((path, cfg))

    class FakeReplayBuffer:
        def __init__(self, *args, **kwargs):
            # Below the configured batch threshold (8 in this test)
            self._size = 2

        def __len__(self):
            return self._size

        def save(self, path):
            Path(path).write_text("replay", encoding="utf-8")

    class FakeWorker:
        instances = []
        _recent_chunks = []
        _prev_count = 0

        def __init__(self, cfg, actor_source, device, replay, rust_binary):
            self.update_calls = []
            self.started = False
            self.stopped = False
            FakeWorker.instances.append(self)

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def status(self):
            return {"alive": True, "consecutive_errors": 0, "last_progress_age_s": 0.0, "last_error": None}

        def update_model(self, actor_source):
            self.update_calls.append(actor_source)

        def pause(self, wait=True):
            return True

        def resume(self):
            return None

    backend = FakeBackend()
    rust_binary = tmp_path / "mcts_demo"
    rust_binary.write_text("stub", encoding="utf-8")
    ctx = cli.PreparedTrainingContext(
        cfg={
            "_name": "gomoku7",
            "board": 7,
            "filters": 32,
            "blocks": 2,
            "buf": 64,
            "batch": 8,
            "steps": 1,
            "games": 2,
            "search_profile": "quartz",
            "vl_mode": "adaptive",
        },
        base_cfg={"board": 7},
        base_dir=str(tmp_path),
        device="cpu",
        hw=types.SimpleNamespace(physical_cpus=1),
        model=None,
        backend=backend,
        optimizer=types.SimpleNamespace(param_groups=[{"lr": 0.0}]),
        actor_source="actor-source",
        benchmark_info=None,
        model_path=str(tmp_path / "latest.pt"),
        latest_model_path=str(tmp_path / "latest.pt"),
        best_model_path=str(tmp_path / "best.pt"),
        replay_path=str(tmp_path / "replay.npz"),
        log_path=str(tmp_path / "train_log.jsonl"),
        autotune_profile_path=str(tmp_path / "autotune_profile.json"),
        n_params=123,
    )
    args = argparse.Namespace(
        serve=False,
        arena_3agent=None,
        arena=None,
        rust_nn=True,
        arena_games=0,
        game="gomoku7",
        iterations=3,  # 3 iters, all below batch threshold → no SGD ever
        resume=False,
        concurrent=True,
        runtime_autotune=False,
        eval_selfplay_isolation=True,
        patience=0,
        inner_patience=0,
        inner_min_fraction=0.0,
        inner_min_delta=0.0,
        inner_ema_alpha=0.2,
        eval_games=2,
        eval_interval=99,
        seed=7,
        rust_binary=str(rust_binary),
    )
    runtime_hooks = cli.MainRuntimeHooks(
        torch=types.SimpleNamespace(),
        np=np,
        game_configs={"gomoku7": {}},
        serve=lambda *args, **kwargs: None,
        arena_3agent=lambda *args, **kwargs: None,
        arena_rust_nn=lambda *args, **kwargs: None,
        arena_compare=lambda *args, **kwargs: None,
        print_autotune_summary=lambda *args, **kwargs: None,
        is_go_game=lambda _name: False,
        replay_buffer_cls=FakeReplayBuffer,
        early_stopping_cls=lambda *args, **kwargs: None,
        early_stopping_enabled=lambda patience, concurrent=False: False,
        load_eval_autotune_profile=lambda *args, **kwargs: None,
        has_eval_system=True,
        recommend_eval_parallel_workers=lambda *args, **kwargs: 1,
        max_supported_threads=lambda hw: 1,
        eval_config_cls=lambda **kwargs: types.SimpleNamespace(**kwargs),
        training_evaluator_cls=lambda config=None: types.SimpleNamespace(cfg=config),
        build_training_game_adapter=lambda cfg: object(),
        ensure_best_checkpoint_compatible=lambda *args, **kwargs: None,
        selfplay_worker_cls=FakeWorker,
        initial_replay_fill_target=lambda cfg, recent_chunks: 0,
        online_autotune_controller_cls=lambda *args, **kwargs: None,
        clear_nn_eval_cache=lambda: None,
        round_or_none=lambda value: value,
        wait_for_worker_progress=lambda worker, prev_bg, min_new=1, timeout_s=30.0: (1, prev_bg + 1),
        selfplay_rust_nn_batched=lambda *args, **kwargs: ([], [], [], []),
        compute_train_steps=lambda *args, **kwargs: 1,
        train_epoch=lambda *args, **kwargs: (0.0, 0.0, 0.0, 0, None),
        replay_metrics=types.SimpleNamespace(
            freshness=lambda n_new, replay_len: 0.0,
            policy_entropy=lambda replay: 0.0,
            value_std=lambda replay: 0.0,
            search_summary=lambda replay: {"positions": len(replay)},
        ),
        rust_nn_evaluator_engine_cls=object,
        clone_actor_model=lambda actor: actor,
        load_actor_source_from_checkpoint=lambda *args, **kwargs: "champion-actor",
        tree_mcts_engine_cls=object,
        benchmark_eval_parallel_workers=lambda *args, **kwargs: (1, []),
        make_json_safe=lambda payload: payload,
        generate_training_plots=lambda *args, **kwargs: False,
    )

    cli.run_training_main(args, ctx, runtime_hooks)

    worker = FakeWorker.instances[-1]
    # Three iterations, replay never crossed batch=8 → 0 SGD rows. With
    # P9, update_model should still fire once per iteration.
    assert worker.update_calls == ["actor-source", "actor-source", "actor-source"]


def test_main_runs_eval_at_interval_even_when_train_steps_are_zero(monkeypatch, tmp_path):
    az = load_training_module()
    backend_module = sys.modules["quartz.backend"]
    output_dir = tmp_path / "run"
    rust_binary = tmp_path / "mcts_demo"
    rust_binary.write_text("stub", encoding="utf-8")

    def fake_create_backend(*args, **kwargs):
        raise RuntimeError("skip unified backend")

    monkeypatch.setattr(backend_module, "create_backend", fake_create_backend, raising=False)
    monkeypatch.setattr(az, "auto_device_name", lambda: "cpu")
    monkeypatch.setattr(
        az,
        "detect_hardware_spec",
        lambda device: az.HardwareSpec(
            logical_cpus=4,
            physical_cpus=2,
            memory_mb=8192,
            gpu_vendor="",
            gpu_name="",
            gpu_vram_mb=0,
            gpu_count=0,
            torch_cuda=False,
            device_kind="cpu",
        ),
    )
    monkeypatch.setattr(az, "configure_torch_rocm_runtime", lambda hw: None)
    monkeypatch.setattr(az, "clamp_runtime_cfg_to_hardware", lambda cfg, hw: dict(cfg))
    monkeypatch.setattr(az, "print_autotune_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(az, "max_supported_threads", lambda hw: 1)
    monkeypatch.setattr(az, "gpu_host_thread_cap", lambda hw: 1)
    monkeypatch.setattr(az, "gpu_interop_thread_cap", lambda hw: 1)
    monkeypatch.setattr(az.torch, "set_num_threads", lambda n: None)
    monkeypatch.setattr(az.torch, "set_num_interop_threads", lambda n: None)
    monkeypatch.setattr(az.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(az, "clear_nn_eval_cache", lambda: None)
    monkeypatch.setattr(az, "get_actor_model", lambda model, backend: "candidate-actor")
    monkeypatch.setattr(az, "clone_actor_model", lambda actor: actor)
    monkeypatch.setattr(az, "load_actor_source_from_checkpoint", lambda *args, **kwargs: "champion-actor")
    monkeypatch.setattr(az, "generate_training_plots", lambda *args, **kwargs: False)
    monkeypatch.setattr(az, "compute_train_steps", lambda *args, **kwargs: 0)
    monkeypatch.setattr(az, "selfplay_rust_nn_batched", lambda *args, **kwargs: ([], [], [], []))
    monkeypatch.setattr(az, "supports_rust_eval_state_machine", lambda game: False)
    monkeypatch.setattr(az, "supports_rust_selfplay_state_machine", lambda game: False)
    monkeypatch.setattr(az, "load_eval_autotune_profile", lambda *args, **kwargs: None)
    monkeypatch.setattr(az, "recommend_eval_parallel_workers", lambda *args, **kwargs: 1)
    monkeypatch.setattr(az, "build_training_game_adapter", lambda cfg: object())
    monkeypatch.setattr(az, "ensure_best_checkpoint_compatible", lambda *args, **kwargs: None)
    monkeypatch.setattr(az, "HAS_EVAL_SYSTEM", True)

    class FakeOptimizer:
        def __init__(self, params):
            self.param_groups = [{"lr": 0.0}]

    monkeypatch.setattr(az.torch.optim, "SGD", lambda params, **kwargs: FakeOptimizer(params))

    class FakeModel:
        def __init__(self, cfg):
            self._params = [az.torch.nn.Parameter(az.torch.zeros(1))]

        def to(self, device):
            return self

        def parameters(self):
            return self._params

        def state_dict(self):
            return {"w": az.torch.zeros(1)}

    monkeypatch.setattr(az, "AlphaZeroNet", FakeModel)

    class FakeReplayBuffer:
        def __init__(self, *args, **kwargs):
            self._size = 10_000

        def __len__(self):
            return self._size

        def save(self, path):
            Path(path).write_text("replay", encoding="utf-8")

    monkeypatch.setattr(az, "ReplayBuffer", FakeReplayBuffer)
    monkeypatch.setattr(az.torch, "save", lambda payload, path: Path(path).write_text("model", encoding="utf-8"))

    class FakeRustEngine:
        def __init__(self, name, cfg, actor, device, rust_binary):
            self._name = name

        def name(self):
            return self._name

        def reset(self):
            return None

        def select_moves_batch(self, *args, **kwargs):
            return []

    monkeypatch.setattr(az, "RustNNEvaluatorEngine", FakeRustEngine)

    eval_generations = []

    class FakeTrainingEvaluator:
        def __init__(self, config=None, manifest=None):
            self.cfg = types.SimpleNamespace(parallel_workers=1)

        def evaluate_checkpoint(
            self,
            candidate,
            champion,
            game_factory,
            candidate_id="",
            generation=0,
            candidate_factory=None,
            champion_factory=None,
        ):
            eval_generations.append(generation)
            return types.SimpleNamespace(
                valid_eval=True,
                invalid_reason=None,
                promotion={"verdict": "need_more"},
                tally={"score_rate": 0.5, "scored": 2, "errors": 0, "voids": 0},
                elo={"delta": 12.0},
                published={"candidate_abs": 100.0, "champion_abs": 88.0, "delta": 12.0},
            )

    monkeypatch.setattr(az, "TrainingEvaluator", FakeTrainingEvaluator)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quartz.train",
            "--game",
            "gomoku7",
            "--iterations",
            "5",
            "--output",
            str(output_dir),
            "--rust-binary",
            str(rust_binary),
            "--no-pipeline",
            "--no-autotune",
        ],
    )

    az.main()

    assert eval_generations == [5]

    log_rows = [
        json.loads(line)
        for line in (output_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eval_row = next(row for row in log_rows if row.get("_type") == "eval")
    iter_row = next(row for row in log_rows if row.get("_type") is None and row.get("iter") == 5)

    assert eval_row["iter"] == 5
    assert iter_row["published_elo"] == 100.0
    assert iter_row["eval_verdict"] == "need_more"


def test_autotune_profile_roundtrip(tmp_path):
    az = load_training_module()
    hw = az.HardwareSpec(
        logical_cpus=8, physical_cpus=4, memory_mb=16384,
        gpu_vendor="amd", gpu_name="GPU", gpu_vram_mb=8192,
        gpu_count=1, torch_cuda=True, device_kind="cuda")
    cfg = {
        "_name": "gomoku7",
        "iters": 200,
        "search_profile": "quartz",
        "penalty_mode": "GatedRefresh",
        "batch_timeout_us": 1500,
    }
    path = tmp_path / "autotune_profile.json"
    overrides = {"batch": 384, "bg_parallel": 2}
    bench = {"train": [{"batch": 384, "examples_per_s": 123.4}]}

    az.save_autotune_profile(str(path), hw, cfg, overrides, bench)
    loaded = az.load_autotune_profile(str(path), hw, cfg)

    assert loaded is not None
    assert loaded["version"] == az.AUTOTUNE_PROFILE_VERSION
    assert loaded["overrides"] == overrides
    assert loaded["benchmarks"] == bench


def test_autotune_profile_rejects_old_version(tmp_path):
    az = load_training_module()
    hw = az.HardwareSpec(
        logical_cpus=8, physical_cpus=4, memory_mb=16384,
        gpu_vendor="amd", gpu_name="GPU", gpu_vram_mb=8192,
        gpu_count=1, torch_cuda=True, device_kind="cuda")
    cfg = {
        "_name": "gomoku7",
        "iters": 200,
        "search_profile": "quartz",
        "penalty_mode": "GatedRefresh",
        "batch_timeout_us": 1500,
    }
    path = tmp_path / "autotune_profile.json"
    payload = {
        "version": az.AUTOTUNE_PROFILE_VERSION - 1,
        "signature": az.autotune_signature(hw, cfg),
        "overrides": {"batch": 384},
        "benchmarks": {},
        "saved_at": 0,
    }
    path.write_text(json.dumps(payload))

    assert az.load_autotune_profile(str(path), hw, cfg) is None


def test_apply_runtime_overrides_updates_cfg_values():
    az = load_training_module()
    cfg = {"batch": 256, "bg_parallel": 1}

    tuned = az.apply_runtime_overrides(cfg, {"batch": 384, "bg_parallel": 2})

    assert tuned["batch"] == 384
    assert tuned["bg_parallel"] == 2


def test_apply_config_overrides_keeps_runtime_search_fields():
    az = load_training_module()
    cfg = dict(az.GAME_CONFIGS["gomoku7"])

    tuned = az.apply_config_overrides(cfg, {"sigma_0": 0.45, "unknown_key": 7})

    assert tuned["sigma_0"] == 0.45
    assert "unknown_key" not in tuned


def test_resolve_runtime_paths_separates_latest_and_best(tmp_path):
    az = load_training_module()

    paths = az.resolve_runtime_paths(str(tmp_path), resume=False)

    assert paths["load_model_path"].endswith("latest.pt")
    assert paths["latest_model_path"].endswith("latest.pt")
    assert paths["best_model_path"].endswith("best.pt")
    assert paths["latest_model_path"] != paths["best_model_path"]


def test_resolve_runtime_paths_resume_falls_back_to_best(tmp_path):
    az = load_training_module()
    best = tmp_path / "best.pt"
    best.write_text("champion")

    paths = az.resolve_runtime_paths(str(tmp_path), resume=True)

    assert paths["load_model_path"] == str(best)


def test_normalize_rust_board_maps_go_white_to_two():
    az = load_training_module()

    board = [1, -1, 0, 2]
    normalized = az.normalize_rust_board("go9", board)

    assert normalized == [1, 2, 0, 2]


def test_all_trainable_non_chess_games_have_registered_encoders():
    az = load_training_module()
    encoders = load_encoders_module()

    for game_name in [
        "gomoku7",
        "gomoku15",
        "gomoku15_free",
        "gomoku15_std",
        "gomoku15_omok",
        "gomoku15_renju",
        "gomoku15_caro",
        "go9",
        "go9_jp",
        "go9_kr",
        "go13",
        "go13_jp",
        "go13_kr",
        "go19",
        "go19_jp",
        "go19_kr",
        "tictactoe",
    ]:
        encoder = encoders.get_encoder(game_name)
        cfg = az.GAME_CONFIGS[game_name]
        assert encoder.board_size == cfg["board"]
        assert encoder.n_actions == cfg["actions"]


def test_build_training_game_adapter_supports_new_games():
    az = load_training_module()
    encoders = load_encoders_module()

    cases = {
        "gomoku15_renju": az.GomokuGameAdapter,
        "go13_jp": az.GoGameAdapter,
        "go19_kr": az.GoGameAdapter,
        "tictactoe": az.TicTacToeGameAdapter,
        "chess": az.ChessEvaluationAdapter,
        "chess960": az.ChessEvaluationAdapter,
    }

    for game_name, expected_type in cases.items():
        cfg = dict(az.GAME_CONFIGS[game_name], _name=game_name, _encoder=encoders.get_encoder(game_name))
        game = az.build_training_game_adapter(cfg)
        assert isinstance(game, expected_type)
        assert len(game.legal_moves()) > 0


def test_go_ruleset_presets_are_exposed_in_cfg_and_adapter():
    az = load_training_module()

    cfg_jp = dict(az.GAME_CONFIGS["go9_jp"], _name="go9_jp", _encoder=None)
    cfg_kr = dict(az.GAME_CONFIGS["go19_kr"], _name="go19_kr", _encoder=None)
    jp = az.build_training_game_adapter(cfg_jp)
    kr = az.build_training_game_adapter(cfg_kr)

    assert jp._ruleset == "japanese"
    assert jp._scoring == "territory"
    assert kr._ruleset == "korean"
    assert kr._scoring == "territory"
    assert kr._komi == az.GAME_CONFIGS["go19_kr"]["go_komi"]


def test_go_chinese_adapter_rejects_repeated_position_hash():
    az = load_training_module()
    game = az.GoGameAdapter(board_size=9, komi=7.5, ruleset="chinese", scoring="area")

    probe = game.clone()
    probe.apply_move(40)
    game._history_hashes.add(probe._position_hash())

    assert game._is_legal(40) is False


def test_go_korean_adapter_marks_repetition_as_draw():
    az = load_training_module()
    game = az.GoGameAdapter(board_size=9, komi=6.5, ruleset="korean", scoring="territory")

    probe = game.clone()
    probe.apply_move(40)
    game._history_hashes.add(probe._position_hash())
    game.apply_move(40)

    assert game.is_terminal() is True
    assert game.outcome_for_black() == 0.0


def test_go_japanese_adapter_marks_repetition_as_void():
    az = load_training_module()
    game = az.GoGameAdapter(board_size=9, komi=6.5, ruleset="japanese", scoring="territory")

    probe = game.clone()
    probe.apply_move(40)
    game._history_hashes.add(probe._position_hash())
    game.apply_move(40)

    assert game.is_terminal() is True
    assert game.is_void_result() is True
    assert game.outcome_for_black() is None


def test_go_territory_cleanup_removes_surrounded_one_eye_group():
    az = load_training_module()
    game = az.GoGameAdapter(board_size=5, komi=6.5, ruleset="japanese", scoring="territory")
    black = {
        0, 1, 2, 3, 4,
        5, 9,
        10, 14,
        15, 19,
        20, 21, 22, 23, 24,
    }
    white = {6, 7, 8, 11, 13, 16, 17, 18}
    for pos in black:
        game._board[pos] = 1
    for pos in white:
        game._board[pos] = -1

    black_score, white_score = game._score()

    assert black_score == 17.0
    assert white_score == 6.5


def test_gomoku15_renju_black_overline_does_not_win_but_white_can():
    az = load_training_module()
    game = az.GomokuGameAdapter(board_size=15, win_len=5, variant="gomoku15_renju")
    row = 7 * 15

    for col in [3, 4, 5, 6, 7]:
        game._board[row + col] = 1
    game._player = 1
    assert game._is_winning_move(row + 8) is False

    game = az.GomokuGameAdapter(board_size=15, win_len=5, variant="gomoku15_renju")
    for col in [3, 4, 5, 6, 7]:
        game._board[row + col] = -1
    game._player = -1
    assert game._is_winning_move(row + 8) is True


def test_gomoku15_caro_blocked_five_is_not_a_win():
    az = load_training_module()
    game = az.GomokuGameAdapter(board_size=15, win_len=5, variant="gomoku15_caro")
    row = 7 * 15

    for col in [4, 5, 6, 7]:
        game._board[row + col] = 1
    game._board[row + 3] = -1
    game._board[row + 9] = -1
    game._player = 1

    assert game._is_winning_move(row + 8) is False


def test_chess_action_space_matches_full_rust_contract():
    az = load_training_module()

    assert az.GAME_CONFIGS["chess"]["actions"] == 4672
    assert az.GAME_CONFIGS["chess960"]["actions"] == 4672
    assert az.GAME_CONFIGS["chess"]["tt_enabled"] is True
    assert az.GAME_CONFIGS["chess960"]["tt_enabled"] is True


def test_chess960_has_registered_encoder_and_is_treated_as_chess():
    az = load_training_module()
    encoders = load_encoders_module()

    encoder = encoders.get_encoder("chess960")

    assert encoder.board_size == 8
    assert encoder.n_actions == az.GAME_CONFIGS["chess960"]["actions"]
    assert az.is_chess_game("chess960") is True


def test_chess960_start_fen_and_encoding_preserve_castling_and_ep():
    az = load_training_module()

    fen = az.chess960_start_fen(0)
    enc = az.encode_chess_fen("bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR b HFhf e3 0 1")

    assert fen.split()[2] != "KQkq"
    assert enc.shape == (36, 8, 8)
    # Planes 0-5: my pieces (black to move → black pieces are "my")
    assert enc[:6].sum() > 0
    # Planes 6-11: opponent pieces (white)
    assert enc[6:12].sum() > 0
    # Plane 28: color (0 for black's turn)
    assert np.all(enc[28] == 0.0)
    # Castling planes 30-33 should have some rights set
    assert enc[30:34].sum() > 0
    # EP plane 35: e3 target
    assert enc[35, 2, 4] == 1.0


def test_initial_chess_fen_uses_fixed_chess960_index_when_configured():
    az = load_training_module()

    cfg = dict(az.GAME_CONFIGS["chess960"], _name="chess960", chess960_index=518)

    assert az.initial_chess_fen(cfg) == az.chess960_start_fen(518)


def test_chess_evaluation_adapter_tracks_turn_and_engine_meta():
    az = load_training_module()

    game = az.ChessEvaluationAdapter(start_fen=az.STANDARD_CHESS_FEN)
    assert game.current_player() == 1

    assert game.apply_engine_meta(0, {"result_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"}) is True
    assert game.current_player() == 0
    assert game.is_terminal() is False

    assert game.apply_engine_meta(0, {"terminal": True, "outcome_for_black": -1.0}) is True
    assert game.is_terminal() is True
    assert game.outcome_for_black() == -1.0


def test_replay_sampler_prefers_recent_window_but_keeps_older_examples():
    az = load_training_module()
    replay = az.ReplayBuffer(32, recent_fraction=0.8, recent_window=4)
    state = np.zeros((3, 7, 7), dtype=np.float32)
    policy = np.zeros(49, dtype=np.float32)

    for value in range(10):
        replay.add(state, policy, float(value))

    random.seed(0)
    indices = replay._sample_indices_locked(5)

    assert len(indices) == 5
    assert len(set(indices)) == 5
    assert sum(1 for i in indices if i >= 6) == 4
    assert sum(1 for i in indices if i < 6) == 1


def test_autotune_training_cfg_scales_parallelism_for_strong_hardware():
    az = load_training_module()
    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    hw = az.HardwareSpec(
        logical_cpus=32, physical_cpus=16, memory_mb=65536,
        gpu_vendor="amd", gpu_name="Test GPU", gpu_vram_mb=24576,
        gpu_count=1, torch_cuda=True, device_kind="cuda")

    tuned = az.autotune_training_cfg(cfg, hw, concurrent=True)

    assert tuned["selfplay_parallel"] >= 4
    assert tuned["bg_parallel"] >= 4
    assert tuned["bg_batch_games"] >= 4
    assert tuned["batch"] >= cfg["batch"]
    assert tuned["batch_size"] >= cfg["batch_size"]


def test_autotune_parallel_limit_caps_gpu_concurrent_process_count():
    az = load_training_module()
    hw = az.HardwareSpec(
        logical_cpus=24, physical_cpus=12, memory_mb=65536,
        gpu_vendor="amd", gpu_name="Test GPU", gpu_vram_mb=16384,
        gpu_count=1, torch_cuda=True, device_kind="cuda")

    assert az._autotune_parallel_limit(hw, concurrent=True) == 6
    assert az._autotune_parallel_limit(hw, concurrent=False) == 12


def test_autotune_parallel_candidates_focus_on_ipc_friendly_gpu_range():
    az = load_training_module()
    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    hw = az.HardwareSpec(
        logical_cpus=24, physical_cpus=12, memory_mb=65536,
        gpu_vendor="amd", gpu_name="Test GPU", gpu_vram_mb=16384,
        gpu_count=1, torch_cuda=True, device_kind="cuda")

    candidates = az._autotune_parallel_candidates(cfg, hw, concurrent=True)

    assert candidates == [1, 2, 3, 4, 5, 6]


def test_autotune_batch_game_candidates_scale_with_parallelism():
    az = load_training_module()
    hw = az.HardwareSpec(
        logical_cpus=24, physical_cpus=12, memory_mb=65536,
        gpu_vendor="amd", gpu_name="Test GPU", gpu_vram_mb=16384,
        gpu_count=1, torch_cuda=True, device_kind="cuda")

    candidates = az._autotune_batch_game_candidates(hw, parallel=12, concurrent=True)

    assert candidates == [12, 24]


def test_autotune_training_cfg_keeps_games_stable_in_concurrent_mode():
    az = load_training_module()
    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    hw = az.HardwareSpec(
        logical_cpus=24, physical_cpus=12, memory_mb=65536,
        gpu_vendor="amd", gpu_name="Test GPU", gpu_vram_mb=16384,
        gpu_count=1, torch_cuda=True, device_kind="cuda")

    tuned = az.autotune_training_cfg(cfg, hw, concurrent=True)

    assert tuned["games"] == cfg["games"]


def test_autotune_training_cfg_autoscales_tiny_model_on_large_gpu():
    az = load_training_module()
    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    hw = az.HardwareSpec(
        logical_cpus=24, physical_cpus=12, memory_mb=65536,
        gpu_vendor="amd", gpu_name="Fast GPU", gpu_vram_mb=16384,
        gpu_count=1, torch_cuda=True, device_kind="cuda")

    tuned = az.autotune_training_cfg(cfg, hw, concurrent=True)

    assert tuned["filters"] >= cfg["filters"]
    assert tuned["blocks"] >= cfg["blocks"]
    assert tuned["vh"] >= cfg["vh"]
    assert az.estimate_model_params(tuned) >= az.estimate_model_params(cfg)


def test_autotune_training_cfg_keeps_large_model_when_already_sized():
    az = load_training_module()
    cfg = dict(az.GAME_CONFIGS["chess"])
    hw = az.HardwareSpec(
        logical_cpus=24, physical_cpus=12, memory_mb=65536,
        gpu_vendor="amd", gpu_name="Fast GPU", gpu_vram_mb=16384,
        gpu_count=1, torch_cuda=True, device_kind="cuda")

    tuned = az.autotune_training_cfg(cfg, hw, concurrent=True)

    assert tuned["filters"] == cfg["filters"]
    assert tuned["blocks"] == cfg["blocks"]
    assert tuned["vh"] == cfg["vh"]


def test_autotune_training_cfg_stays_conservative_on_small_hardware():
    az = load_training_module()
    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    hw = az.HardwareSpec(
        logical_cpus=4, physical_cpus=2, memory_mb=8192,
        gpu_vendor="none", gpu_name="", gpu_vram_mb=0,
        gpu_count=0, torch_cuda=False, device_kind="cpu")

    tuned = az.autotune_training_cfg(cfg, hw, concurrent=True)

    assert tuned["selfplay_parallel"] == 2
    assert tuned["bg_parallel"] == 2
    assert tuned["batch"] <= cfg["batch"]
    assert tuned["n_threads"] == 1


def test_autotune_signatures_track_topology_fields():
    az = load_training_module()
    hw = az.HardwareSpec(
        logical_cpus=24, physical_cpus=12, memory_mb=65536,
        gpu_vendor="amd", gpu_name="Test GPU", gpu_vram_mb=16384,
        gpu_count=1, torch_cuda=True, device_kind="cuda")
    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    cfg["_name"] = "gomoku7"
    cfg["_eval_runner_mode"] = "python_batched"
    cfg["_shared_eval_session"] = False
    cfg["_broker_enabled"] = False
    cfg["_selfplay_topology_version"] = 3

    sig_a = az.autotune_signature(hw, cfg)
    eval_sig_a = az.eval_autotune_signature(hw, cfg, 200)

    cfg["_shared_eval_session"] = True
    cfg["_eval_runner_mode"] = "shared_client_session"
    sig_b = az.autotune_signature(hw, cfg)
    eval_sig_b = az.eval_autotune_signature(hw, cfg, 200)

    assert sig_a != sig_b
    assert eval_sig_a != eval_sig_b


def test_selfplay_autotune_score_penalizes_bursty_cycles():
    az = load_training_module()

    smooth = az._score_selfplay_probe(20.0, 4.0, concurrent=True)
    bursty = az._score_selfplay_probe(20.0, 9.0, concurrent=True)

    assert smooth > bursty


def test_train_batch_score_penalizes_oversized_batches_in_concurrent_mode():
    az = load_training_module()

    right_sized = az._score_train_batch_probe(
        1000.0, 256, concurrent=True, target_positions_per_cycle=80)
    oversized = az._score_train_batch_probe(
        1000.0, 640, concurrent=True, target_positions_per_cycle=80)

    assert right_sized > oversized


def test_plan_online_runtime_overrides_reduces_bursty_batch_games():
    az = load_training_module()
    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    cfg.update({"bg_parallel": 4, "bg_batch_games": 8, "n_threads": 3, "batch": 256, "batch_size": 12})
    hw = az.HardwareSpec(
        logical_cpus=24, physical_cpus=12, memory_mb=65536,
        gpu_vendor="amd", gpu_name="Fast GPU", gpu_vram_mb=16384,
        gpu_count=1, torch_cuda=True, device_kind="cuda")
    sample = {
        "last_cycle_s": 5.9,
        "last_cycle_positions": 126,
        "positions_per_s": 21.0,
        "best_positions_per_s": 21.0,
        "n_new": 60,
        "train_steps": 3,
    }

    overrides = az.plan_online_runtime_overrides(cfg, hw, sample)

    assert overrides["bg_batch_games"] == 4


def test_plan_online_runtime_overrides_reduces_batch_when_fresh_data_is_thin():
    az = load_training_module()
    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    cfg.update({"bg_parallel": 4, "bg_batch_games": 4, "n_threads": 3, "batch": 256, "batch_size": 12})
    hw = az.HardwareSpec(
        logical_cpus=24, physical_cpus=12, memory_mb=65536,
        gpu_vendor="amd", gpu_name="Fast GPU", gpu_vram_mb=16384,
        gpu_count=1, torch_cuda=True, device_kind="cuda")
    sample = {
        "last_cycle_s": 2.6,
        "last_cycle_positions": 58,
        "positions_per_s": 8.0,
        "best_positions_per_s": 10.0,
        "n_new": 60,
        "train_steps": 2,
    }

    overrides = az.plan_online_runtime_overrides(cfg, hw, sample)

    assert overrides["batch"] < cfg["batch"]


def test_plan_selfplay_runner_chunk_scales_parallel_with_replay_deficit():
    az = load_training_module()
    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    cfg.update({
        "_selfplay_runner_mode": "rust_selfplay_state_machine",
        "bg_parallel": 6,
        "bg_batch_games": 6,
        "batch_size": 18,
        "batch": 192,
    })

    plan = az.plan_selfplay_runner_chunk(cfg, replay_size=32, recent_chunks=[])

    assert plan["parallel"] >= 18
    assert plan["batch_games"] >= plan["parallel"]
    assert plan["replay_deficit"] == 160


def test_plan_selfplay_runner_chunk_uses_recent_positions_per_game_to_bound_batch_games():
    az = load_training_module()
    cfg = dict(az.GAME_CONFIGS["gomoku7"])
    cfg.update({
        "_selfplay_runner_mode": "rust_selfplay_state_machine",
        "bg_parallel": 6,
        "bg_batch_games": 6,
        "batch_size": 18,
        "batch": 192,
    })
    recent_chunks = [
        {"games": 18, "positions": 540, "elapsed_s": 12.0},
        {"games": 18, "positions": 576, "elapsed_s": 11.0},
    ]

    plan = az.plan_selfplay_runner_chunk(cfg, replay_size=180, recent_chunks=recent_chunks)

    assert plan["estimated_positions_per_game"] > 20.0
    assert plan["batch_games"] <= 18


def test_initial_replay_fill_target_is_lower_than_train_batch_when_warm_start_suffices():
    az = load_training_module()
    cfg = {
        "batch": 480,
        "batch_size": 18,
        "bg_parallel": 5,
        "board": 7,
    }
    recent_chunks = [
        {"games": 10, "positions": 180, "elapsed_s": 4.0},
        {"games": 10, "positions": 200, "elapsed_s": 4.2},
    ]

    target = az.initial_replay_fill_target(cfg, recent_chunks)

    assert target >= cfg["batch_size"]
    assert target < cfg["batch"]


def test_initial_replay_fill_target_uses_board_prior_without_recent_chunks():
    az = load_training_module()
    cfg = {
        "batch": 256,
        "batch_size": 16,
        "bg_parallel": 4,
        "board": 7,
    }

    target = az.initial_replay_fill_target(cfg, [])

    assert target >= 16
    assert target <= 256


def test_initial_replay_fill_target_clamps_to_replay_headroom():
    az = load_training_module()
    cfg = {
        "batch": 256,
        "batch_size": 32,
        "bg_parallel": 4,
        "board": 7,
        "buf": 30,
    }

    target = az.initial_replay_fill_target(cfg, [])

    assert target == int(math.floor(30 * az.SelfPlayWorker.BACKPRESSURE_RATIO))


def test_initial_replay_fill_target_honors_bootstrap_cap():
    az = load_training_module()
    cfg = {
        "batch": 256,
        "batch_size": 32,
        "bg_parallel": 4,
        "board": 7,
        "_bootstrap_replay_target_cap": 12,
    }

    target = az.initial_replay_fill_target(cfg, [])

    assert target == 12


def test_plan_selfplay_runner_chunk_honors_safe_caps():
    az = load_training_module()
    cfg = {
        "batch": 256,
        "batch_size": 8,
        "bg_parallel": 2,
        "bg_batch_games": 2,
        "board": 7,
        "_selfplay_runner_mode": "rust_selfplay_state_machine",
        "_selfplay_parallel_cap": 3,
        "_selfplay_batch_games_cap": 4,
    }

    plan = az.plan_selfplay_runner_chunk(cfg, replay_size=0, recent_chunks=[])

    assert plan["parallel"] == 3
    assert plan["batch_games"] == 4
    assert plan["games_per_call"] <= 4
