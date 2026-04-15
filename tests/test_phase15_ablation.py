import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from quartz.phase15_ablation import (
    Phase15System,
    apply_system_readout,
    build_root_challenger_set,
    classify_position_buckets,
    load_systems_config,
    make_default_systems,
    normalize_policy,
    policy_argmax,
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


def test_make_default_systems_exposes_clean_split_groups():
    systems = make_default_systems({})
    assert [system.id for system in systems] == ["A0", "A1", "A2", "A3", "A4", "B0", "B1", "B2", "B3", "C0", "C1", "C2"]
    assert systems[5].report_alias == "A4"
    assert systems[5].execution_mode == "posthoc"


def test_group_a_and_b_defaults_do_not_use_refresh_legacy_substrate():
    systems = {system.id: system for system in make_default_systems({})}
    for system_id in ("A0", "A1", "A2", "A3", "A4", "B0", "B1", "B2", "B3"):
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

    key_a = trace_cache_key("C1", "/tmp/model.pt", "P1", "A4", ("sig",), [8, 16], code_salt="salt-a")
    key_b = trace_cache_key("C1", "/tmp/model.pt", "P1", "A4", ("sig",), [8, 16], code_salt="salt-b")
    assert key_a != key_b


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
                "continuation_bundle_wallclock_ms": 12.0,
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
    assert summary["by_checkpoint_system"][0]["speedup_headwind"] == "mixed_session_overhead_and_readout_sensitivity"


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


def test_phase15_benchmark_ci_smoke_positions_are_deterministic():
    smoke = load_phase15_benchmark_ci_smoke_runner()
    positions = smoke.deterministic_positions(7)
    assert len(positions) == 4
    assert positions[0]["id"] == "P0001"
    assert positions[1]["id"] == "P0002"
    assert positions[2]["id"] == "P0003"
    assert positions[3]["id"] == "P0004"
    assert len(positions[0]["board"]) == 49
