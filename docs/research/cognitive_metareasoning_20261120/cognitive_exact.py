"""COG-01 exact Bayesian recognition / verification reference.

Four-atom finite world, exact Fraction arithmetic. No game or NN claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F

U = F(3, 5)
COST = F(1, 10)


@dataclass(frozen=True)
class Scenario:
    p: F
    se: F
    fpr: F

    def __post_init__(self) -> None:
        for field in ("p", "se", "fpr"):
            v = getattr(self, field)
            if not isinstance(v, F) or not F(0) <= v <= F(1):
                raise ValueError(f"{field} must be an exact Fraction in [0,1]")


SOURCE = Scenario(F(1, 5), F(17, 20), F(1, 5))

SCENARIOS: dict[str, Scenario] = {
    "in_distribution": SOURCE,
    "uninformative_cue": Scenario(F(1, 5), F(1, 2), F(1, 2)),
    "inverted_cue": Scenario(F(1, 5), F(1, 5), F(17, 20)),
    "prevalence_shift": Scenario(F(2, 5), F(17, 20), F(1, 5)),
    "perfect_cue": Scenario(F(1, 5), F(1), F(0)),
}


def atoms(world: Scenario) -> tuple[tuple[F, int, str], ...]:
    """Exact four-atom world probabilities, including zero-mass atoms."""
    return (
        (world.p * world.se, 1, "+"),
        (world.p * (1 - world.se), 1, "-"),
        ((1 - world.p) * world.fpr, 0, "+"),
        ((1 - world.p) * (1 - world.fpr), 0, "-"),
    )


def cue_mass(world: Scenario, sign: str) -> F:
    if sign not in ("+", "-"):
        raise ValueError("cue must be '+' or '-'")
    return sum((w for w, _, s in atoms(world) if s == sign), F())


def posterior(world: Scenario, sign: str) -> F | None:
    """P(X=1 | sign), or None for a null-probability event."""
    denominator = cue_mass(world, sign)
    if denominator == 0:
        return None
    numerator = world.p * (world.se if sign == "+" else 1 - world.se)
    return numerator / denominator


def gross_gain(q: F) -> F:
    """Gain from perfect verification under *assumed* posterior q."""
    if not F(0) <= q <= F(1):
        raise ValueError("posterior q outside [0,1]")
    return q * (1 - U) if q <= U else U * (1 - q)


def stop_reward(x: int, q: F) -> F:
    if x not in (0, 1):
        raise ValueError("state X must be binary")
    return U if q <= U else F(x)


def realized_verification_reward(x: int) -> F:
    """Pay computation cost once and choose perfect-information action."""
    if x not in (0, 1):
        raise ValueError("state X must be binary")
    return max(U, F(x)) - COST


def true_gain_after_cue(actual_q: F, predicted_q: F) -> F:
    """Expected *gross* benefit relative to action chosen by predicted_q.

    It is not g(actual_q) when miscalibration changes the stop decision.
    """
    return actual_q * (1 - U) if predicted_q <= U else (1 - actual_q) * U


def evaluate(source: Scenario, world: Scenario) -> dict[str, F]:
    """Exact expectations, no label leakage into the learned policy.

    random_matched_rate uses the true deployment marginal verification rate
    for descriptive *non-deployable* budget matching only.
    """
    pred = {s: posterior(source, s) for s in ("+", "-")}
    true = {s: posterior(world, s) for s in ("+", "-")}
    weights = {s: cue_mass(world, s) for s in ("+", "-")}
    for s in ("+", "-"):
        if weights[s] and pred[s] is None:
            raise ValueError(f"unseen source cue {s!r} has deployment mass")

    # Null-mass cue contributions are *not* evaluated or divided by zero.
    inspect: dict[str, bool] = {}
    calibrated: dict[str, bool] = {}
    quality_ece = F(0)
    gain_ece = F(0)
    for s in ("+", "-"):
        if not weights[s]:
            inspect[s] = False
            calibrated[s] = False
            continue
        q = pred[s]
        t = true[s]
        assert q is not None and t is not None
        inspect[s] = gross_gain(q) > COST
        calibrated[s] = gross_gain(t) > COST
        quality_ece += weights[s] * abs(q - t)
        gain_ece += weights[s] * abs(gross_gain(q) - true_gain_after_cue(t, q))

    selected_rate = sum((weights[s] for s in ("+", "-") if inspect[s]), F(0))
    oracle_rate = sum((weights[s] for s in ("+", "-") if calibrated[s]), F(0))
    totals = {
        "stop_only": F(0),
        "learned_selective": F(0),
        "random_matched_rate": F(0),
        "oracle_recalibrated": F(0),
        "always_verify": F(0),
        "full_information_no_cost": F(0),
        "quality_brier": F(0),
    }
    mass = F(0)
    for prob, x, s in atoms(world):
        mass += prob
        if prob == 0:
            continue
        q = pred[s]
        t = true[s]
        assert q is not None and t is not None
        stop = stop_reward(x, q)
        verification = realized_verification_reward(x)
        actual = stop_reward(x, t)
        totals["stop_only"] += prob * stop
        totals["learned_selective"] += prob * (verification if inspect[s] else stop)
        totals["random_matched_rate"] += prob * (
            selected_rate * verification + (1 - selected_rate) * stop
        )
        totals["oracle_recalibrated"] += prob * (
            verification if calibrated[s] else actual
        )
        totals["always_verify"] += prob * verification
        totals["full_information_no_cost"] += prob * max(U, F(x))
        totals["quality_brier"] += prob * (q - x) ** 2

    if mass != 1:
        raise ArithmeticError("joint probability mass is not one")
    return {
        **totals,
        "selective_rate": selected_rate,
        "oracle_rate": oracle_rate,
        "random_matched_rate_rate": selected_rate,
        "quality_ece": quality_ece,
        "gain_ece": gain_ece,
        "net_improvement_over_stop": totals["learned_selective"] - totals["stop_only"],
        "selective_minus_random_matched": (
            totals["learned_selective"] - totals["random_matched_rate"]
        ),
    }
