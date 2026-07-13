"""Tests for quartz.phase15_online.run_online_readout.

A2-a audit finding: this function had ZERO test coverage before this
file. The specific bug it fixes: apply_budget_routing was called
mid-loop with only the sub-target trace accumulated so far, so its
burst_idx lookup (which requires a supra-target chunk already present
in trace_budgets) was always None — online B3 burst was unreachable
dead code, silently degrading to a plain snapshot readout.
"""

from __future__ import annotations

import numpy as np

from quartz.phase15_ablation import Phase15System, normalize_policy, policy_argmax
from quartz.phase15_online import run_online_readout


def _make_search_policy_fn(policies_by_budget: dict[int, list[float]], calls: list[int]):
    def _fn(position, system, budget):
        calls.append(int(budget))
        return {
            "search_policy": list(policies_by_budget[int(budget)]),
            "latency_ms": 1.0,
        }

    return _fn


def test_run_online_readout_budget_routing_bursts_on_instability():
    """A2-a regression: an unstable sub-target trace (argmax flips
    0->1 between budgets 8 and 16) must fetch the supra-target chunk
    (32) and use IT as the final policy — the exact online-burst path
    that was previously dead code."""
    system = Phase15System(
        id="B3",
        label="routing",
        group="B",
        substrate="S1",
        controller="QuartzVL",
        refresh_operator="budget_routing",
        params={"persistence_floor": 0.95, "margin_stability_floor": 0.95},
    )
    prior = normalize_policy(np.array([0.55, 0.25, 0.20], dtype=np.float32)).tolist()
    policies_by_budget = {
        8: [0.40, 0.35, 0.25],
        16: [0.30, 0.45, 0.25],
        32: [0.10, 0.75, 0.15],
    }
    calls: list[int] = []

    effective, meta = run_online_readout(
        system=system,
        position={"id": "P1"},
        prior_input=np.asarray(prior, dtype=np.float32),
        budgets=[8, 16, 32],
        target_budget=16,
        search_policy_fn=_make_search_policy_fn(policies_by_budget, calls),
    )

    assert calls == [8, 16, 32], f"expected the extra tier (32) to be fetched, got calls={calls}"
    assert meta["budget_burst_triggered"] == 1
    assert meta["extra_budget_used"] == 16
    assert meta["online_stop_budget"] == 32
    assert any("burst@16->32" in note for note in meta["decision_notes"])
    assert policy_argmax(effective) == 1


def test_run_online_readout_budget_routing_stable_never_fetches_extra_chunk():
    """A2-a: the flip side of the fix. A STABLE sub-target trace
    (argmax stays 0 throughout, well within the default floors) must
    NOT fetch the supra-target chunk at all — proving this is a
    genuinely adaptive (pay only when indicated) burst, not an
    always-pre-pay one."""
    system = Phase15System(
        id="B3",
        label="routing",
        group="B",
        substrate="S1",
        controller="QuartzVL",
        refresh_operator="budget_routing",
        params={},  # default floors: persistence_floor=0.60, margin_stability_floor=0.72
    )
    prior = normalize_policy(np.array([0.70, 0.20, 0.10], dtype=np.float32)).tolist()
    policies_by_budget = {
        8: [0.68, 0.20, 0.12],
        16: [0.72, 0.18, 0.10],
        32: [0.90, 0.05, 0.05],  # should never be read
    }
    calls: list[int] = []

    effective, meta = run_online_readout(
        system=system,
        position={"id": "P1"},
        prior_input=np.asarray(prior, dtype=np.float32),
        budgets=[8, 16, 32],
        target_budget=16,
        search_policy_fn=_make_search_policy_fn(policies_by_budget, calls),
    )

    assert calls == [8, 16], f"extra tier (32) must not be fetched when stable, got calls={calls}"
    assert meta["budget_burst_triggered"] == 0
    assert meta["extra_budget_used"] == 0
    assert meta["online_stop_budget"] == 16
    assert policy_argmax(effective) == 0


def test_run_online_readout_budget_routing_no_higher_tier_available():
    """Edge case: target_budget is the last configured tier, so there
    is nothing to burst to even if unstable. Must return the plain
    readout without attempting a fetch beyond the provided budgets."""
    system = Phase15System(
        id="B3",
        label="routing",
        group="B",
        substrate="S1",
        controller="QuartzVL",
        refresh_operator="budget_routing",
        params={"persistence_floor": 0.95, "margin_stability_floor": 0.95},
    )
    prior = normalize_policy(np.array([0.55, 0.25, 0.20], dtype=np.float32)).tolist()
    policies_by_budget = {
        8: [0.40, 0.35, 0.25],
        16: [0.30, 0.45, 0.25],
    }
    calls: list[int] = []

    effective, meta = run_online_readout(
        system=system,
        position={"id": "P1"},
        prior_input=np.asarray(prior, dtype=np.float32),
        budgets=[8, 16],
        target_budget=16,
        search_policy_fn=_make_search_policy_fn(policies_by_budget, calls),
    )

    assert calls == [8, 16]
    assert meta["budget_burst_triggered"] == 0
    assert meta["online_stop_budget"] == 16
    assert normalize_policy(effective) is not None


def test_run_online_readout_dual_channel_commit_returns_early():
    """Sanity coverage for the OTHER early-return path in
    run_online_readout (previously also untested): dual_channel_commit
    can stop before reaching target_budget when commit fires."""
    system = Phase15System(
        id="B1",
        label="dual channel",
        group="B",
        substrate="S1",
        controller="QuartzVL",
        refresh_operator="dual_channel_commit",
        params={"commit_threshold": 0.0},  # trivially satisfied -> commit ASAP
    )
    prior = normalize_policy(np.array([0.5, 0.3, 0.2], dtype=np.float32)).tolist()
    policies_by_budget = {
        8: [0.5, 0.3, 0.2],
        16: [0.5, 0.3, 0.2],
        32: [0.9, 0.05, 0.05],  # should never be read if commit fires at budget 16
    }
    calls: list[int] = []

    effective, meta = run_online_readout(
        system=system,
        position={"id": "P1"},
        prior_input=np.asarray(prior, dtype=np.float32),
        budgets=[8, 16, 32],
        target_budget=32,
        search_policy_fn=_make_search_policy_fn(policies_by_budget, calls),
    )

    assert meta["online_stop_budget"] < 32, "commit should stop before the full target budget"
    assert 32 not in calls, f"budget 32 must not be searched once commit fires early, got calls={calls}"
    assert normalize_policy(effective) is not None


def test_run_online_readout_h1_stops_early_on_stable_trace():
    """Stage 7 / C5: H1 argmax-stability stop halts at the first sub-target
    chunk whose Dirichlet argmax-stability clears the threshold. A concentrated
    policy at budget 8 (total visits 8 >= min_visits) is stable => stop@8."""
    system = Phase15System(
        id="B14",
        label="h1",
        group="B",
        substrate="S1",
        controller="QuartzVL",
        refresh_operator="argmax_stability_stop",
        params={"stability_threshold": 0.9, "stability_min_visits": 8, "stability_n_boot": 2000},
        execution_mode="online",
    )
    prior = normalize_policy(np.array([0.34, 0.33, 0.33], dtype=np.float32)).tolist()
    policies_by_budget = {
        8: [0.92, 0.05, 0.03],
        16: [0.9, 0.06, 0.04],
        32: [0.9, 0.06, 0.04],
    }
    calls: list[int] = []
    effective, meta = run_online_readout(
        system=system,
        position={"id": "P1"},
        prior_input=np.asarray(prior, dtype=np.float32),
        budgets=[8, 16, 32],
        target_budget=32,
        search_policy_fn=_make_search_policy_fn(policies_by_budget, calls),
    )
    assert meta["online_stop_budget"] == 8
    assert any(note.startswith("h1_stop@8") for note in meta["decision_notes"])
    assert 32 not in calls, f"H1 stopped at 8 but later budgets were searched: {calls}"
    assert meta["argmax_stability"] >= 0.9


def test_run_online_readout_h1_continues_on_unstable_trace_to_target():
    """A near-uniform (unstable) low-budget trace must NOT stop early; H1 runs
    to the target budget."""
    system = Phase15System(
        id="B14",
        label="h1",
        group="B",
        substrate="S1",
        controller="QuartzVL",
        refresh_operator="argmax_stability_stop",
        params={"stability_threshold": 0.9, "stability_min_visits": 8, "stability_n_boot": 2000},
        execution_mode="online",
    )
    prior = normalize_policy(np.array([0.34, 0.33, 0.33], dtype=np.float32)).tolist()
    policies_by_budget = {
        8: [0.4, 0.35, 0.25],
        16: [0.38, 0.34, 0.28],
        32: [0.36, 0.34, 0.30],
    }
    calls: list[int] = []
    effective, meta = run_online_readout(
        system=system,
        position={"id": "P1"},
        prior_input=np.asarray(prior, dtype=np.float32),
        budgets=[8, 16, 32],
        target_budget=32,
        search_policy_fn=_make_search_policy_fn(policies_by_budget, calls),
    )
    assert meta["online_stop_budget"] == 32
    assert 32 in calls
    assert not any(note.startswith("h1_stop") for note in meta.get("decision_notes", []))


def test_continuation_early_stop_fn_prevents_later_steps():
    """Stage 7 / C5: run_online_readout_continuation stops stepping the resident
    session once early_stop_fn fires — real compute saved."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "phase15_online_ablation_c5", root / "scripts" / "phase15_online_ablation.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    step_calls = {"n": 0}

    class DummyClient:
        cfg = {"actions": 3, "penalty_mode": "None"}

        def open_search_engine_session(self, jobs, penalty_mode="None", iters=0):
            return {"session_id": "S", "results": [{"policy": [[0, 0.9], [1, 0.05], [2, 0.05]], "latency_ms": 1.0}]}

        def step_search_engine_session(self, session_id, updates=None, iters=0):
            step_calls["n"] += 1
            return {"results": [{"policy": [[0, 0.9], [1, 0.05], [2, 0.05]], "latency_ms": 1.0}]}

        def close_search_session(self, session_id):
            pass

    # Predicate fires immediately at the first chunk.
    rows = runner.run_online_readout_continuation(
        DummyClient(), {"id": "P1"}, object(), [8, 16, 32], 32,
        early_stop_fn=lambda budget, row: True,
    )
    assert set(rows) == {8}, "should have realized only the opening chunk"
    assert step_calls["n"] == 0, "no session steps should have run after the early stop"
