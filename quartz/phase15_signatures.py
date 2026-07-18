"""Game-agnostic search-behavior signatures (Part B / B1).

This module computes the O1-O6 signature battery from `THESIS.md` P3.
Every function here consumes **only** search/root statistics already
present in a phase15 trace artifact (``trace_budgets``,
``trace_policies``, ``trace_latencies_ms`` from
``phase15_trace.build_trace_artifact``) plus, for the single
skill-discriminating metric, an analyst-supplied reference oracle. No
game rules, board topology, or move semantics are used anywhere.

Signature roles (THESIS.md P3, and the plan §B1):

- **Macro-structure, predicted skill-INVARIANT** (de Groot): O1 candidate
  concentration ``K_eff = exp(H(policy))`` and its trajectory vs budget.
  Matching these is a necessary sanity check, never a claim.
- **Revision dynamics** (O5): first_revision_step, flip_flop_rate,
  final sparsity. Descriptive; feeds the P_flip / H1 stop analysis.
- **Allocation dispersion** (O2): Gini / entropy of per-move realized
  budget. Descriptive.
- **Skill-DISCRIMINATING** (Russek 2025): ``voc_tightness`` =
  correlation(per_move_budget, voc_proxy).

**Non-circularity guard (THESIS.md P3, mandatory).** ``voc_tightness``
takes ``voc_proxy_values`` as an explicit argument. That proxy is computed
*offline by the analyst* from a high-budget reference oracle; the search
engine never computes or consumes it at runtime. The signature of this
function encodes that separation on purpose: nothing in this module can
turn VOC into a control input.

All functions are pure and deterministic. ``numpy`` is used for the array
math; policy vectors are accepted as anything ``np.asarray`` can consume.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

SIGNATURE_SCHEMA_VERSION = 1

__all__ = [
    "SIGNATURE_SCHEMA_VERSION",
    "policy_entropy",
    "k_eff",
    "k_eff_trajectory",
    "argmax_path",
    "first_revision_step",
    "flip_flop_rate",
    "final_sparsity",
    "concentration_vs_budget",
    "budget_gini",
    "budget_entropy",
    "voc_tightness",
    "trace_signature_summary",
]


def _as_prob(p: Any) -> np.ndarray:
    """Coerce to a 1-D non-negative float array that sums to 1.

    Renormalizes if the input does not already sum to 1 (visit-count
    vectors are accepted as well as probability vectors). An all-zero or
    empty vector maps to an empty array, which callers treat as "no
    distribution" (entropy 0, K_eff 0).
    """
    arr = np.asarray(p, dtype=np.float64).ravel()
    if arr.size == 0:
        return arr
    arr = np.clip(arr, 0.0, None)
    total = arr.sum()
    if total <= 0.0:
        return np.zeros_like(arr)
    return arr / total


def policy_entropy(p: Any) -> float:
    """Shannon entropy H(p) in nats. Zeros contribute 0 (0·log0 := 0)."""
    prob = _as_prob(p)
    if prob.size == 0:
        return 0.0
    nz = prob[prob > 0.0]
    if nz.size == 0:
        return 0.0
    return float(-np.sum(nz * np.log(nz)))


def k_eff(p: Any) -> float:
    """O1 effective candidate count = exp(H(p)).

    Ranges from 1 (fully concentrated) to ``support size`` (uniform). An
    all-zero / empty vector returns 0.0 to signal "no live candidates".
    """
    prob = _as_prob(p)
    if prob.size == 0 or prob.sum() <= 0.0:
        return 0.0
    return float(np.exp(policy_entropy(prob)))


def k_eff_trajectory(trace_policies: Sequence[Any]) -> list[float]:
    """O1 over the budget ladder: K_eff at each recorded budget step."""
    return [k_eff(p) for p in trace_policies]


def argmax_path(trace_policies: Sequence[Any]) -> list[int]:
    """The argmax action index at each budget step (-1 for empty steps)."""
    path: list[int] = []
    for p in trace_policies:
        prob = _as_prob(p)
        if prob.size == 0 or prob.sum() <= 0.0:
            path.append(-1)
        else:
            path.append(int(np.argmax(prob)))
    return path


def first_revision_step(trace_policies: Sequence[Any]) -> int | None:
    """O5 first_revision_step: first budget-step index t>=1 whose argmax
    differs from step t-1. ``None`` if the argmax never changes (or the
    trace is too short). Empty steps (argmax -1) are skipped rather than
    counted as revisions."""
    path = [a for a in argmax_path(trace_policies) if a >= 0]
    for t in range(1, len(path)):
        if path[t] != path[t - 1]:
            return t
    return None


def flip_flop_rate(trace_policies: Sequence[Any]) -> float:
    """O5 flip_flop_rate: fraction of adjacent budget steps whose argmax
    changed. 0.0 for traces with fewer than 2 non-empty steps."""
    path = [a for a in argmax_path(trace_policies) if a >= 0]
    if len(path) < 2:
        return 0.0
    changes = sum(1 for t in range(1, len(path)) if path[t] != path[t - 1])
    return changes / (len(path) - 1)


def final_sparsity(trace_policies: Sequence[Any], n_legal: int | None = None) -> float:
    """O5 sparsity of the final-budget policy: K_eff(last) / n_legal.

    If ``n_legal`` is omitted, the support size (number of non-zero
    entries) of the final policy is used as the denominator, so the value
    still reflects "how much of the available branching the final policy
    actually concentrates on". Returns 0.0 for an empty trace."""
    if len(trace_policies) == 0:
        return 0.0
    last = _as_prob(trace_policies[-1])
    if last.size == 0 or last.sum() <= 0.0:
        return 0.0
    denom = (
        n_legal
        if (n_legal is not None and n_legal > 0)
        else int(np.count_nonzero(last))
    )
    if denom <= 0:
        return 0.0
    return float(k_eff(last) / denom)


def concentration_vs_budget(
    trace_budgets: Sequence[int], trace_policies: Sequence[Any]
) -> dict[str, Any]:
    """O1 as a function of budget consumed: the (budget, K_eff) curve plus
    a summary slope (least-squares d K_eff / d log-budget). A principled
    controller is expected to *concentrate* (K_eff fall) as budget grows,
    so the slope is predicted negative; its magnitude is a macro metric
    (predicted skill-invariant under P3)."""
    budgets = [int(b) for b in trace_budgets]
    keff = k_eff_trajectory(trace_policies)
    n = min(len(budgets), len(keff))
    budgets, keff = budgets[:n], keff[:n]
    slope: float | None = None
    if n >= 2:
        xs = np.log(np.asarray([max(b, 1) for b in budgets], dtype=np.float64))
        ys = np.asarray(keff, dtype=np.float64)
        if float(np.var(xs)) > 0.0:
            slope = float(np.polyfit(xs, ys, 1)[0])
    return {
        "budgets": budgets,
        "k_eff": keff,
        "k_eff_first": keff[0] if keff else None,
        "k_eff_last": keff[-1] if keff else None,
        "k_eff_slope_per_log_budget": slope,
    }


def budget_gini(per_move_budgets: Sequence[float]) -> float:
    """O2 Gini coefficient of realized per-move budget. 0 = every move got
    the same budget (flat allocation); higher = compute concentrated on
    few moves. Returns 0.0 for <2 moves or all-zero budgets."""
    vals = np.asarray([max(float(b), 0.0) for b in per_move_budgets], dtype=np.float64)
    n = vals.size
    if n < 2:
        return 0.0
    total = vals.sum()
    if total <= 0.0:
        return 0.0
    vals = np.sort(vals)
    # Gini = (2*sum(i*x_i) / (n*sum(x))) - (n+1)/n, with i in 1..n.
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * vals)) / (n * total) - (n + 1.0) / n)


def budget_entropy(per_move_budgets: Sequence[float]) -> float:
    """O2 entropy (nats) of the normalized per-move budget distribution.
    Complementary to Gini; high entropy = flat allocation."""
    return policy_entropy(per_move_budgets)


def voc_tightness(
    per_move_budgets: Sequence[float],
    voc_proxy_values: Sequence[float],
    *,
    method: str = "spearman",
) -> float | None:
    """Skill-DISCRIMINATING signature (THESIS.md P3, Russek 2025).

    Correlation between the compute spent on each move and the *value of
    computation* on that move. Predicted to INCREASE with checkpoint
    strength; a flat tightness across skill falsifies the motto.

    ``voc_proxy_values`` MUST be computed offline by the analyst from a
    high-budget reference oracle (e.g. shallow-vs-deep argmax disagreement
    times Q-gap). Passing it as an explicit argument is the P3
    non-circularity guard: the engine never sees VOC at runtime.

    Returns ``None`` when the correlation is undefined (fewer than 2
    points, or zero variance in either series). ``method`` is
    ``"spearman"`` (rank, robust to the proxy's arbitrary scale;
    default) or ``"pearson"``.
    """
    x = np.asarray(per_move_budgets, dtype=np.float64).ravel()
    y = np.asarray(voc_proxy_values, dtype=np.float64).ravel()
    n = min(x.size, y.size)
    if n < 2:
        return None
    x, y = x[:n], y[:n]
    if method == "spearman":
        x = _rankdata(x)
        y = _rankdata(y)
    elif method != "pearson":
        raise ValueError(f"unknown method {method!r}; use 'spearman' or 'pearson'")
    if float(np.var(x)) <= 0.0 or float(np.var(y)) <= 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank transform (ties get the mean of their rank span)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
    # average ties
    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    return ranks


def trace_signature_summary(
    trace: dict[str, Any], n_legal: int | None = None
) -> dict[str, Any]:
    """Bundle the single-trace (per-position) signatures O1/O5 from a
    phase15 trace artifact. Cross-move signatures (O2 budget dispersion,
    VOC-tightness) are computed at the study level from many traces, not
    here."""
    policies = trace.get("trace_policies", []) or []
    budgets = trace.get("trace_budgets", []) or []
    return {
        "signature_schema_version": SIGNATURE_SCHEMA_VERSION,
        "k_eff_trajectory": k_eff_trajectory(policies),
        "concentration_vs_budget": concentration_vs_budget(budgets, policies),
        "first_revision_step": first_revision_step(policies),
        "flip_flop_rate": flip_flop_rate(policies),
        "final_sparsity": final_sparsity(policies, n_legal=n_legal),
        "n_budget_steps": len(policies),
    }
