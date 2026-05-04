"""Nested-reservoir live-set maintenance.

Per the audit (§13.6 / §6.4) this is "nested-reservoir search" rather
than nested sampling for evidence estimation. We maintain a live set
of root candidates ranked by

    Lambda_a = U_a + rho * KG_a + tau * log pi_tilde_0(a)

and remove arms whose Lambda falls below a quantile threshold;
replenish from unexplored / low-prior / high-uncertainty via Gumbel
or Thompson sampling. The "nested" name comes from the analogy with
Skilling 2006 nested sampling — but we are NOT estimating the
evidence integral; we are only borrowing the live-set + threshold
maintenance idea.

Hysteresis: an arm just-removed cannot re-enter for ``cooldown``
iterations. Prevents oscillation when a borderline arm flips
repeatedly across the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def lambda_score(
    upper_ci_a: float,
    kg_a: float,
    log_prior_smoothed_a: float,
    rho: float = 1.0,
    tau: float = 0.1,
) -> float:
    """Lambda_a = U_a + rho * KG_a + tau * log pi_tilde_0(a)."""
    return upper_ci_a + rho * kg_a + tau * log_prior_smoothed_a


def quantile(values: list[float], q: float) -> float:
    """Return the q-th quantile (linear interpolation between data points).

    Works for q in [0, 1]. Used as the bottom-25% threshold for live-
    set pruning.
    """
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n == 1:
        return s[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


@dataclass
class Reservoir:
    """Live candidate set with quantile pruning + hysteresis."""

    live: list[int] = field(default_factory=list)
    cooldown_until: dict[int, int] = field(default_factory=dict)
    max_size: int = 16
    cooldown_iters: int = 200  # 2 * default check_interval

    def is_eligible(self, idx: int, current_iter: int) -> bool:
        """An arm is eligible to (re-)enter if its cooldown has expired."""
        return self.cooldown_until.get(idx, 0) <= current_iter

    def add(self, idx: int, current_iter: int) -> bool:
        """Add ``idx`` to the live set. Returns True if added."""
        if idx in self.live:
            return False
        if len(self.live) >= self.max_size:
            return False
        if not self.is_eligible(idx, current_iter):
            return False
        self.live.append(idx)
        return True

    def remove(self, idx: int, current_iter: int) -> bool:
        """Remove ``idx`` and start its cooldown."""
        if idx not in self.live:
            return False
        self.live.remove(idx)
        self.cooldown_until[idx] = current_iter + self.cooldown_iters
        return True

    def prune_below_quantile(
        self,
        scores: dict[int, float],
        q: float,
        current_iter: int,
    ) -> list[int]:
        """Remove all live arms with score below the q-th quantile.

        ``scores`` maps live-set index → Lambda_a. Returns the list of
        removed indices.

        Subtle: if all live arms have the same score, the quantile is
        the common value; arms with score == quantile are kept (we use
        strict ``<``).
        """
        if not self.live:
            return []
        live_scores = [scores.get(i, 0.0) for i in self.live]
        threshold = quantile(live_scores, q)
        removed: list[int] = []
        for i in list(self.live):
            if scores.get(i, 0.0) < threshold:
                self.remove(i, current_iter)
                removed.append(i)
        return removed
