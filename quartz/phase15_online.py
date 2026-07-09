"""Online phase 1.5 control scaffold.

The current implementation is intentionally honest: search is re-run from the
same root state for each chunk budget, and controller decisions are applied
between chunks. This is online control, but not root-continuation search yet.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from .phase15_ablation import (
    Phase15System,
    apply_system_readout,
    budget_routing_signal,
    normalize_policy,
)


def run_online_readout(
    *,
    system: Phase15System,
    position: dict[str, Any],
    prior_input: np.ndarray,
    budgets: list[int],
    target_budget: int,
    search_policy_fn: Callable[[dict[str, Any], Phase15System, int], dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    trace_policies: list[np.ndarray] = []
    trace_budgets: list[int] = []
    trace_latencies_ms: list[float] = []
    decision_notes: list[str] = []
    stop_budget = int(target_budget)

    for budget in budgets:
        if int(budget) > int(target_budget) and system.refresh_operator != "budget_routing":
            break
        row = search_policy_fn(position, system, int(budget))
        trace_policies.append(np.asarray(row["search_policy"], dtype=np.float32))
        trace_budgets.append(int(budget))
        trace_latencies_ms.append(float(row.get("latency_ms", 0.0)))

        if int(budget) < int(target_budget):
            if system.refresh_operator == "dual_channel_commit" and len(trace_policies) >= 2:
                effective, meta = apply_system_readout(system, prior_input, trace_policies, trace_budgets, int(budget))
                if int(meta.get("commit_applied", 0)) == 1:
                    decision_notes.append(f"commit@stop={budget}")
                    stop_budget = int(budget)
                    return normalize_policy(effective), {
                        **meta,
                        "trace_budgets": trace_budgets,
                        "trace_latencies_ms": trace_latencies_ms,
                        "trace_acquire_ms": float(sum(trace_latencies_ms)),
                        "effective_runtime_ms": float(sum(trace_latencies_ms)),
                        "readout_ms": 0.0,
                        "online_stop_budget": stop_budget,
                        "search_continuation": "restart_per_chunk",
                        "decision_notes": decision_notes,
                    }
            continue

        if system.refresh_operator == "budget_routing":
            # A2-a fix: decide burst from the SUB-TARGET trace alone,
            # via budget_routing_signal — no supra-target chunk needed
            # for the decision. Before this fix, apply_budget_routing
            # was called directly here with only the sub-target trace
            # accumulated so far; its burst_idx lookup (which requires
            # a supra-target entry already present in trace_budgets)
            # was therefore always None, and the burst branch below
            # was unreachable dead code — online B3 silently degraded
            # to a plain snapshot readout. Only fetch (and pay the
            # latency for) the extra tier when instability actually
            # indicates it — this is the genuinely adaptive version of
            # the posthoc path's always-pre-paid burst check.
            signal = budget_routing_signal(trace_policies, trace_budgets, int(budget), system.params)
            next_budget = next((b for b in budgets if int(b) > int(budget)), None)
            if signal.get("unstable") and next_budget is not None:
                extra_row = search_policy_fn(position, system, int(next_budget))
                trace_policies.append(np.asarray(extra_row["search_policy"], dtype=np.float32))
                trace_budgets.append(int(next_budget))
                trace_latencies_ms.append(float(extra_row.get("latency_ms", 0.0)))

        t0 = time.perf_counter()
        effective, meta = apply_system_readout(system, prior_input, trace_policies, trace_budgets, int(budget))
        readout_ms = (time.perf_counter() - t0) * 1000.0
        stop_budget = int(meta.get("burst_budget", budget))
        if system.refresh_operator == "budget_routing" and int(meta.get("budget_burst_triggered", 0)) == 1:
            decision_notes.append(f"burst@{budget}->{stop_budget}")
        return normalize_policy(effective), {
            **meta,
            "trace_budgets": trace_budgets,
            "trace_latencies_ms": trace_latencies_ms,
            "trace_acquire_ms": float(sum(trace_latencies_ms)),
            "readout_ms": float(readout_ms),
            "effective_runtime_ms": float(sum(trace_latencies_ms) + readout_ms),
            "online_stop_budget": stop_budget,
            "search_continuation": "restart_per_chunk",
            "decision_notes": decision_notes,
        }

    t0 = time.perf_counter()
    effective, meta = apply_system_readout(system, prior_input, trace_policies, trace_budgets, int(target_budget))
    readout_ms = (time.perf_counter() - t0) * 1000.0
    return normalize_policy(effective), {
        **meta,
        "trace_budgets": trace_budgets,
        "trace_latencies_ms": trace_latencies_ms,
        "trace_acquire_ms": float(sum(trace_latencies_ms)),
        "readout_ms": float(readout_ms),
        "effective_runtime_ms": float(sum(trace_latencies_ms) + readout_ms),
        "online_stop_budget": stop_budget,
        "search_continuation": "restart_per_chunk",
        "decision_notes": decision_notes,
    }


__all__ = ["run_online_readout"]
