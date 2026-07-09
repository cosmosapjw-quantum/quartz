"""H2 — One-Loop finite-N curvature readout ("B13"), Part B / B2.

This is the cheapest-to-verify algorithm lane. It adds a phase15 posthoc
readout that applies a *finite-N discretization curvature correction* to
the realized visit policy.

## The reframe (mandatory — adversarial audit §0.5, §0.7 PDR item 2)

The naive form ``log π_eff(a) = log π̄(a) + 3·ℏ_eff / (4·max(N_a, N_floor))``
was attacked for double-counting: if π̄ already reflects Q (it is a
PUCT+Q-shaped visit distribution), adding another Q-like term double-counts
and diverges at ``N_a → 0``.

The surviving definition treats the extra term **not** as "more Q" but as
a **finite-N discretization curvature correction**: the realized finite-N
visit distribution π̄ is a biased, over-concentrated estimate of the
continuous policy-improvement target (Grill et al. 2020 worst-case gap
``(|A|-1)/(|A|+N)`` is largest at small N). The correction gently
re-inflates under-visited arms and, crucially, **must vanish as N grows**.

That vanishing is the entire falsifiable content (CCoT discriminating
experiment, §0.6): stratify the readout's effect by root visit count. If
the effect ``→ 0`` at large N it is a genuine curvature correction; if it
persists at large N it is double-counting and the lane is **killed**.
``one_loop_effect_kl`` in the returned metadata is the per-position
quantity to stratify.

All math here is game-agnostic: it consumes only the visit policy and the
total visit budget. No board, move, or rule semantics.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "ONE_LOOP_SCHEMA_VERSION",
    "one_loop_correction",
    "apply_one_loop_readout",
]

ONE_LOOP_SCHEMA_VERSION = 1

# Defaults are deliberately conservative and are NOT first-principles
# values. The plan defers the ℏ_eff / 3-4 constant calibration to a run
# with real search data (§0.7, A3-c note); `curvature` is the single knob
# standing in for it until then.
_DEFAULT_CURVATURE = 1.0
_DEFAULT_N_FLOOR = 1.0


def one_loop_correction(
    base_policy: np.ndarray,
    n_total: float,
    *,
    curvature: float = _DEFAULT_CURVATURE,
    n_floor: float = _DEFAULT_N_FLOOR,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the finite-N curvature correction to a visit policy.

    Parameters
    ----------
    base_policy : the realized visit distribution π̄ (need not be
        normalized; it is renormalized over its positive support).
    n_total : total root visits N. Per-arm counts are reconstructed as
        ``N_a = π̄(a) · N_total``.
    curvature : the ℏ_eff-analog scale (single tunable; see module doc).
    n_floor : floor on ``N_a`` inside the correction denominator, so a
        0/1-visit arm cannot blow the term up.

    Returns
    -------
    (effective_policy, meta). ``effective_policy`` sums to 1 over the same
    support as ``base_policy`` (unvisited arms stay exactly 0 — no
    completed-Q data is available in the trace, so they are not invented).
    The correction ``δ_a = curvature / max(N_a, n_floor)`` is added in
    log-space and then softmax-renormalized. ``δ_a → 0`` as ``N_a → ∞``,
    which is the kill-test contract.
    """
    base = np.asarray(base_policy, dtype=np.float64).ravel()
    base = np.clip(base, 0.0, None)
    total = base.sum()
    support = base > 0.0
    k = int(support.sum())
    if total <= 0.0 or k == 0:
        # nothing to correct
        out = base.copy()
        return out, {
            "one_loop_schema_version": ONE_LOOP_SCHEMA_VERSION,
            "one_loop_active": False,
            "one_loop_effect_kl": 0.0,
            "one_loop_max_delta": 0.0,
            "one_loop_support": k,
        }
    pi_bar = base / total
    if k == 1 or n_total <= 0.0 or curvature == 0.0:
        # a single live arm (or no budget) has no curvature to correct
        out = np.zeros_like(base)
        out[support] = pi_bar[support]
        return out, {
            "one_loop_schema_version": ONE_LOOP_SCHEMA_VERSION,
            "one_loop_active": False,
            "one_loop_effect_kl": 0.0,
            "one_loop_max_delta": 0.0,
            "one_loop_support": k,
        }

    n_a = pi_bar * float(n_total)
    delta = np.zeros_like(base)
    denom = np.maximum(n_a[support], float(n_floor))
    delta[support] = float(curvature) / denom

    logits = np.full_like(base, -np.inf)
    logits[support] = np.log(pi_bar[support]) + delta[support]
    # softmax over support (stable)
    m = np.max(logits[support])
    exp = np.zeros_like(base)
    exp[support] = np.exp(logits[support] - m)
    eff = exp / exp.sum()

    # KL(pi_bar || eff) over the shared support — the quantity to stratify
    # by N for the double-counting kill test.
    p = pi_bar[support]
    q = eff[support]
    effect_kl = float(np.sum(p * (np.log(p) - np.log(q))))

    return eff, {
        "one_loop_schema_version": ONE_LOOP_SCHEMA_VERSION,
        "one_loop_active": True,
        "one_loop_effect_kl": effect_kl,
        "one_loop_max_delta": float(np.max(delta[support])),
        "one_loop_support": k,
        "one_loop_n_total": float(n_total),
    }


def apply_one_loop_readout(
    prior_base: np.ndarray,
    trace_policies: list[np.ndarray],
    trace_budgets: list[int],
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Phase15 readout adapter (operator ``one_loop_finite_n`` / candidate
    "B13"). Matches the standard
    ``(prior_base, trace_policies, trace_budgets, params) -> (effective,
    meta)`` readout signature so it drops into ``apply_system_readout``.

    Uses the final-budget visit policy as π̄ and the final budget as N. The
    prior is not consumed (the correction is a property of the realized
    finite-N policy, not the prior) but is echoed into metadata for parity
    with the other readouts.
    """
    if not trace_policies:
        base = np.asarray(prior_base, dtype=np.float64).ravel()
        total = base.sum()
        eff = base / total if total > 0 else base
        return eff, {
            "belief_revision_operator": "one_loop_finite_n",
            "one_loop_schema_version": ONE_LOOP_SCHEMA_VERSION,
            "one_loop_active": False,
            "one_loop_effect_kl": 0.0,
            "effective_policy": eff.tolist(),
        }

    pi_bar = np.asarray(trace_policies[-1], dtype=np.float64).ravel()
    n_total = float(trace_budgets[-1]) if trace_budgets else 0.0
    curvature = float(params.get("one_loop_curvature", _DEFAULT_CURVATURE))
    n_floor = float(params.get("one_loop_n_floor", _DEFAULT_N_FLOOR))

    eff, meta = one_loop_correction(pi_bar, n_total, curvature=curvature, n_floor=n_floor)
    meta = dict(meta)
    meta["belief_revision_operator"] = "one_loop_finite_n"
    meta["effective_policy"] = eff.tolist()
    return eff, meta
