"""Prior-surprise diagnostic statistic (NOT a formal hypothesis test).

The audit (§1.6) flagged that Pearson's χ² as a formal hypothesis test
requires iid multinomial counts under the null — which MCTS visits do
NOT satisfy (they are PUCT-driven adaptive samples). Calling χ² with
α = 0.05 a "5% false-rejection rate" was overclaim.

This module keeps the χ² **statistic** as a diagnostic scalar (it
cleanly measures divergence of empirical visits from the prior) but
deliberately does NOT emit a p-value or false-rejection threshold.
The threshold for triggering the EXPAND channel becomes a calibrated
empirical decision (e.g. learn from self-play).

API contract:
    - :func:`prior_surprise_statistic` returns a non-negative scalar.
    - There is no ``p_value`` function. Calling code that asks for
      ``scipy.stats.chi2`` is doing the wrong thing.
    - Threshold for "EXPAND active" is a *configurable scalar*, not
      derived from a χ² CDF. The comment in :func:`expand_active_at`
      explains the non-coupling.
"""

from __future__ import annotations


def prior_surprise_statistic(
    visit_counts: list[int],
    prior: list[float],
    eps: float = 1e-6,
) -> float:
    """Pearson-χ² statistic (NOT a p-value).

    Formula:
        chi2 = sum_a (N_a - N * pi_a)^2 / (N * pi_a)

    Returns 0 when visits exactly match the prior; larger values
    indicate stronger divergence. The prior is clamped at ``eps``
    from below to avoid division by zero on actions with zero prior.

    Returns a non-negative float.
    """
    if len(visit_counts) != len(prior):
        raise ValueError(
            f"visit_counts and prior must have equal length; "
            f"got {len(visit_counts)} vs {len(prior)}"
        )
    if not visit_counts:
        return 0.0
    N = sum(visit_counts)
    if N == 0:
        return 0.0
    chi2 = 0.0
    for n_a, pi_a in zip(visit_counts, prior):
        pi_clamped = max(pi_a, eps)
        expected = N * pi_clamped
        if expected <= 0.0:
            continue
        chi2 += (n_a - expected) ** 2 / expected
    return chi2


def expand_active_at(statistic: float, threshold: float) -> bool:
    """Return True iff the surprise statistic exceeds the calibrated threshold.

    The threshold is **not** derived from a χ² inverse-CDF and the
    decision is **not** a formal hypothesis test. It is a calibrated
    empirical signal: the threshold is set by the user (or by an
    offline self-play calibration) such that the EXPAND channel
    fires roughly ``X%`` of the time on a held-out position set.

    Caller is responsible for picking a defensible threshold. The
    canonical recommendation per the audit: use a fixed threshold
    learned offline (e.g. 95th percentile of the statistic on a
    calibration set), document it in the experiment metadata, and
    never call ``scipy.stats.chi2.ppf`` here.
    """
    return statistic > threshold


# Forbidden: never expose a p-value API. Doing so would silently
# re-introduce the formal-test framing the audit flagged. If a
# downstream consumer wants a probability, they need to come back
# with a calibrated empirical threshold; not a textbook χ² CDF.
