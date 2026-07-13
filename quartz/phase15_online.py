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
    argmax_stability_stop_params,
    budget_routing_signal,
    h3_burst_signal,
    normalize_policy,
)
from .phase15_argmax_stability import counts_from_policy, should_stop_by_argmax_stability

_ROUTING_OPERATORS = ("budget_routing", "entropy_burst_routing")


def _routing_burst_signal(system, trace_policies, trace_budgets, budget):
    """Dispatch the online sub-target instability signal for the routing
    operators (B3 budget_routing vs B15 H3 entropy-burst)."""
    if system.refresh_operator == "entropy_burst_routing":
        return h3_burst_signal(trace_policies, trace_budgets, int(budget), system.params)
    return budget_routing_signal(trace_policies, trace_budgets, int(budget), system.params)


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
    trace_p_flips: list[float | None] = []
    decision_notes: list[str] = []
    stop_budget = int(target_budget)

    for budget in budgets:
        if int(budget) > int(target_budget) and system.refresh_operator not in _ROUTING_OPERATORS:
            break
        row = search_policy_fn(position, system, int(budget))
        trace_policies.append(np.asarray(row["search_policy"], dtype=np.float32))
        trace_budgets.append(int(budget))
        trace_latencies_ms.append(float(row.get("latency_ms", 0.0)))
        trace_p_flips.append(None if row.get("p_flip") is None else float(row.get("p_flip")))

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
                        "trace_p_flips": trace_p_flips,
                        "trace_acquire_ms": float(sum(trace_latencies_ms)),
                        "effective_runtime_ms": float(sum(trace_latencies_ms)),
                        "readout_ms": 0.0,
                        "online_stop_budget": stop_budget,
                        "search_continuation": "restart_per_chunk",
                        "decision_notes": decision_notes,
                    }
            if system.refresh_operator == "argmax_stability_stop":
                # H1 online stop (Part B / B2): halt at the first sub-target
                # chunk whose Dirichlet argmax-stability clears the threshold.
                counts = counts_from_policy(
                    np.asarray(trace_policies[-1], dtype=np.float64), int(budget)
                )
                stop, h1meta = should_stop_by_argmax_stability(
                    counts, **argmax_stability_stop_params(system.params)
                )
                if stop:
                    decision_notes.append(f"h1_stop@{budget}")
                    stop_budget = int(budget)
                    effective, meta = apply_system_readout(
                        system, prior_input, trace_policies, trace_budgets, int(budget)
                    )
                    return normalize_policy(effective), {
                        **meta,
                        "trace_budgets": trace_budgets,
                        "trace_latencies_ms": trace_latencies_ms,
                        "trace_p_flips": trace_p_flips,
                        "trace_acquire_ms": float(sum(trace_latencies_ms)),
                        "effective_runtime_ms": float(sum(trace_latencies_ms)),
                        "readout_ms": 0.0,
                        "online_stop_budget": stop_budget,
                        "argmax_stability": float(h1meta["argmax_stability"]),
                        "search_continuation": "restart_per_chunk",
                        "decision_notes": decision_notes,
                    }
            continue

        if system.refresh_operator in _ROUTING_OPERATORS:
            # A2-a fix: decide the burst from the SUB-TARGET trace alone (via the
            # routing signal — budget_routing for B3, H3 entropy-burst for B15) —
            # no supra-target chunk needed for the decision. Only fetch (and pay
            # the latency for) the extra tier when instability actually indicates
            # it — the genuinely adaptive version of the posthoc always-pre-paid
            # burst check. Before the A2-a fix the supra-target burst branch was
            # unreachable dead code.
            signal = _routing_burst_signal(system, trace_policies, trace_budgets, int(budget))
            next_budget = next((b for b in budgets if int(b) > int(budget)), None)
            if signal.get("unstable") and next_budget is not None:
                extra_row = search_policy_fn(position, system, int(next_budget))
                trace_policies.append(np.asarray(extra_row["search_policy"], dtype=np.float32))
                trace_budgets.append(int(next_budget))
                trace_latencies_ms.append(float(extra_row.get("latency_ms", 0.0)))
                trace_p_flips.append(None if extra_row.get("p_flip") is None else float(extra_row.get("p_flip")))

        t0 = time.perf_counter()
        effective, meta = apply_system_readout(system, prior_input, trace_policies, trace_budgets, int(budget))
        readout_ms = (time.perf_counter() - t0) * 1000.0
        stop_budget = int(meta.get("burst_budget", budget))
        if system.refresh_operator in _ROUTING_OPERATORS and int(meta.get("budget_burst_triggered", 0)) == 1:
            decision_notes.append(f"burst@{budget}->{stop_budget}")
        return normalize_policy(effective), {
            **meta,
            "trace_budgets": trace_budgets,
            "trace_latencies_ms": trace_latencies_ms,
            "trace_p_flips": trace_p_flips,
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
        "trace_p_flips": trace_p_flips,
        "trace_acquire_ms": float(sum(trace_latencies_ms)),
        "readout_ms": float(readout_ms),
        "effective_runtime_ms": float(sum(trace_latencies_ms) + readout_ms),
        "online_stop_budget": stop_budget,
        "search_continuation": "restart_per_chunk",
        "decision_notes": decision_notes,
    }


__all__ = ["run_online_readout"]
