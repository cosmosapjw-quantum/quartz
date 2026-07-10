"""H1 — Bootstrap Argmax-Stability Stop, Part B / B2.

A nonparametric, low-budget stopping signal: keep searching until the root
argmax is *stable under resampling of the visit allocation*. This is the
redesigned H1 — the original "split the visit stream into k independent
fragments" idea was killed in the adversarial audit (§0.5) because a shared
MCTS tree + virtual loss makes the fragments non-independent, so their
agreement is trivially ~1 and meaningless.

## What it actually computes (§0.7 PDR item 1)

Given observed root visit counts ``n = (n_1, ..., n_K)`` with total ``N``,
treat the visit shares as a multinomial parameter and place a Bayesian
bootstrap / Dirichlet posterior on it:

    θ ~ Dirichlet(n + α)

``argmax_stability`` = ``P(argmax(θ) == argmax(n))`` under that posterior.
This is a nonparametric flip-risk on the realized allocation — no iid
assumption across time, no stationarity assumption (it is a posterior over
the multinomial parameter, which is exchangeable). It is the honest
statistical translation of the legacy "Darwinism redundancy" intuition:
many resampled votes agreeing = redundant confidence in the current best.

It replaces the KL-LUCB certificate as the **low-budget** primary stop
(A1-a made that certificate correct-but-near-never-firing at 8-64 visits;
CLAIM_LEDGER row for Module 2). KL-LUCB stays as a high-budget /
terminal-Bernoulli backup.

## Pre-experiment discrimination gate (CCoT §0.6 — run BEFORE the lane)

The kill test for H1 is NOT "does it stop" — it is "does the stability
signal *discriminate*". If ``argmax_stability`` is stuck ~1.0 regardless of
gap and N, it is no better than the trivial point-argmax and is useless as
a stop. ``stability_discrimination_gate`` measures the spread of stability
across positions and its monotonic increase with N; a degenerate (near
zero-variance, always-saturated) signal fails the gate and the lane is
killed before the expensive paired experiment.

Game-agnostic: consumes only visit counts. No board/move/rule semantics.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = [
    "ARGMAX_STABILITY_SCHEMA_VERSION",
    "counts_from_policy",
    "dirichlet_argmax_posterior",
    "argmax_stability",
    "should_stop_by_argmax_stability",
    "stability_discrimination_gate",
]

ARGMAX_STABILITY_SCHEMA_VERSION = 1

_DEFAULT_ALPHA = 0.5  # Jeffreys prior; keeps zero-count arms in play
_DEFAULT_N_BOOT = 4000
_DEFAULT_THRESHOLD = 0.9
_DEFAULT_MIN_VISITS = 8


def counts_from_policy(policy: Sequence[float], n_total: int) -> np.ndarray:
    """Reconstruct integer visit counts from a normalized visit policy and
    total budget (offline helper for trace analysis). Rounds and clips to
    non-negative; the largest-share arm absorbs any rounding remainder so
    the counts sum to ``n_total``."""
    p = np.asarray(policy, dtype=np.float64).ravel()
    p = np.clip(p, 0.0, None)
    total = p.sum()
    if total <= 0.0 or n_total <= 0:
        return np.zeros(p.size, dtype=np.int64)
    p = p / total
    counts = np.floor(p * n_total).astype(np.int64)
    remainder = int(n_total) - int(counts.sum())
    if remainder > 0:
        # hand the remainder to the highest-share arms (largest first)
        order = np.argsort(-p)
        for i in range(remainder):
            counts[order[i % order.size]] += 1
    return counts


def dirichlet_argmax_posterior(
    visit_counts: Sequence[float],
    *,
    alpha: float = _DEFAULT_ALPHA,
    n_boot: int = _DEFAULT_N_BOOT,
    seed: int = 0,
) -> np.ndarray:
    """Return ``P(arm a is the argmax)`` under ``θ ~ Dir(counts + alpha)``.

    Estimated by ``n_boot`` posterior draws with a fixed seed (deterministic
    for tests). Returns a probability vector over arms summing to 1; an
    all-zero count vector returns a uniform vector."""
    n = np.asarray(visit_counts, dtype=np.float64).ravel()
    k = n.size
    if k == 0:
        return np.asarray([], dtype=np.float64)
    if k == 1:
        return np.asarray([1.0], dtype=np.float64)
    conc = np.clip(n, 0.0, None) + float(alpha)
    if not np.all(conc > 0.0):
        return np.full(k, 1.0 / k, dtype=np.float64)
    rng = np.random.default_rng(seed)
    # draws: (n_boot, k) Dirichlet samples via independent Gammas.
    g = rng.gamma(shape=conc, size=(int(n_boot), k))
    theta = g / g.sum(axis=1, keepdims=True)
    winners = np.argmax(theta, axis=1)
    counts = np.bincount(winners, minlength=k).astype(np.float64)
    return counts / counts.sum()


def argmax_stability(
    visit_counts: Sequence[float],
    *,
    alpha: float = _DEFAULT_ALPHA,
    n_boot: int = _DEFAULT_N_BOOT,
    seed: int = 0,
) -> float:
    """``P(argmax(θ) == argmax(counts))`` under the Dirichlet posterior —
    the stability of the current best arm. 1.0 for a single arm; 0.0 for an
    empty vector."""
    n = np.asarray(visit_counts, dtype=np.float64).ravel()
    if n.size == 0:
        return 0.0
    if n.size == 1:
        return 1.0
    observed = int(np.argmax(n))
    post = dirichlet_argmax_posterior(n, alpha=alpha, n_boot=n_boot, seed=seed)
    return float(post[observed])


def should_stop_by_argmax_stability(
    visit_counts: Sequence[float],
    *,
    threshold: float = _DEFAULT_THRESHOLD,
    min_visits: int = _DEFAULT_MIN_VISITS,
    alpha: float = _DEFAULT_ALPHA,
    n_boot: int = _DEFAULT_N_BOOT,
    seed: int = 0,
) -> tuple[bool, dict[str, Any]]:
    """Decide whether to halt: stop iff total visits ≥ ``min_visits`` AND
    ``argmax_stability ≥ threshold``. Returns (stop, meta). The
    ``min_visits`` gate mirrors the KL-LUCB ``min_total`` guard — never stop
    on a near-empty root regardless of an over-confident small-sample
    posterior."""
    n = np.asarray(visit_counts, dtype=np.float64).ravel()
    total = float(n.sum())
    stability = argmax_stability(n, alpha=alpha, n_boot=n_boot, seed=seed)
    stop = bool(total >= float(min_visits) and stability >= float(threshold))
    return stop, {
        "argmax_stability_schema_version": ARGMAX_STABILITY_SCHEMA_VERSION,
        "argmax_stability": stability,
        "argmax_index": int(np.argmax(n)) if n.size else -1,
        "total_visits": total,
        "threshold": float(threshold),
        "min_visits": int(min_visits),
        "stop": stop,
    }


def stability_discrimination_gate(
    visit_count_vectors: Sequence[Sequence[float]],
    *,
    alpha: float = _DEFAULT_ALPHA,
    n_boot: int = _DEFAULT_N_BOOT,
    seed: int = 0,
    trivial_std_eps: float = 1e-3,
) -> dict[str, Any]:
    """Pre-experiment CCoT gate. Given a set of root visit-count vectors
    (e.g. one per position at a fixed budget), measure whether the
    stability signal *discriminates* rather than saturating at 1.0.

    Returns the spread of stability and a ``discriminates`` flag: the
    signal passes only if its standard deviation across positions exceeds
    ``trivial_std_eps`` (i.e. it is not stuck at one value). A degenerate,
    always-saturated signal fails the gate → H1 is killed before the
    expensive paired experiment."""
    stats = [
        argmax_stability(v, alpha=alpha, n_boot=n_boot, seed=seed)
        for v in visit_count_vectors
        if np.asarray(v, dtype=np.float64).size > 0
    ]
    if not stats:
        return {
            "argmax_stability_schema_version": ARGMAX_STABILITY_SCHEMA_VERSION,
            "n_positions": 0,
            "stability_mean": None,
            "stability_std": None,
            "stability_min": None,
            "stability_max": None,
            "discriminates": False,
        }
    arr = np.asarray(stats, dtype=np.float64)
    std = float(np.std(arr))
    return {
        "argmax_stability_schema_version": ARGMAX_STABILITY_SCHEMA_VERSION,
        "n_positions": int(arr.size),
        "stability_mean": float(np.mean(arr)),
        "stability_std": std,
        "stability_min": float(np.min(arr)),
        "stability_max": float(np.max(arr)),
        "discriminates": bool(std > float(trivial_std_eps)),
    }
