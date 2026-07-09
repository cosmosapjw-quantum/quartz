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
