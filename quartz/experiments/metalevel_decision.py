"""A game-neutral, finite Bayesian *reference* for metareasoning.

This is not a trained MCTS posterior or a wired SearchPolicy.  It specifies
terminal decision simple regret, gross expected computation value, separately
priced net gain, and finite-horizon optimal STOP/COMPUTE decisions.  Computation
outcomes are assumed conditionally independent given a fixed latent state.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any, Mapping, Sequence

NORM_TOL = 1e-10
TIE_TOL = 1e-12
MAX_HORIZON = 4


def _rational_float(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a numerical probability/cost")
    try:
        z = float(Fraction(str(value)))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"invalid rational scalar: {value}") from exc
    if not math.isfinite(z):
        raise ValueError("nonfinite scalar")
    return z


def _probability_row(values: Sequence[object]) -> tuple[float, ...]:
    if not values:
        raise ValueError("empty probability row")
    x = tuple(_rational_float(v) for v in values)
    if any(not 0 <= q <= 1 for q in x):
        raise ValueError("probabilities outside [0,1]")
    if abs(math.fsum(x) - 1.) > NORM_TOL:
        raise ValueError("probability row not normalized")
    return x


@dataclass(frozen=True)
class Experiment:
    name: str
    cost: float
    likelihood: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class FiniteProblem:
    prior: tuple[float, ...]
    utilities: tuple[tuple[float, ...], ...]
    experiments: tuple[Experiment, ...]
    lambda_cost: float
    horizon: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FiniteProblem":
        try:
            prior = _probability_row(data["posterior"])
            utilities = tuple(tuple(_rational_float(z) for z in row)
                              for row in data["utilities"])
            if len(utilities) != len(prior) or not utilities or len(utilities[0]) < 2:
                raise ValueError("utility matrix must have state rows and >=2 actions")
            if any(len(row) != len(utilities[0]) for row in utilities):
                raise ValueError("utility matrix has inconsistent action counts")
            raw = data["experiments"]
            if not raw:
                raise ValueError("no computational actions")
            experiments = []
            for exp in raw:
                name = exp["name"]
                cost = _rational_float(exp["cost"])
                rows = tuple(_probability_row(row) for row in exp["likelihood"])
                if not isinstance(name, str) or not name:
                    raise ValueError("empty experiment name")
                if cost <= 0:
                    raise ValueError("strictly positive computation cost required")
                if len(rows) != len(prior) or any(len(row) != len(rows[0]) for row in rows):
                    raise ValueError("observation channel must have one equal-width row per state")
                experiments.append(Experiment(name, cost, rows))
            if len({e.name for e in experiments}) != len(experiments):
                raise ValueError("duplicate experiment names")
            lam = _rational_float(data["lambda"])
            h = data["horizon"]
            if lam <= 0 or type(h) is not int or not 0 <= h <= MAX_HORIZON:
                raise ValueError("positive utility/resource price and bounded integer horizon")
            return cls(prior, utilities, tuple(experiments), lam, h)
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError("incomplete finite decision task") from exc

    def _belief(self, belief: Sequence[float] | None) -> tuple[float, ...]:
        if belief is None:
            return self.prior
        b = _probability_row(belief)
        if len(b) != len(self.prior):
            raise ValueError("wrong posterior dimension")
        return b

    def experiment(self, name: str) -> Experiment:
        for e in self.experiments:
            if e.name == name:
                return e
        raise ValueError(f"unknown computation: {name}")

    def terminal_action_values(self, belief: Sequence[float] | None = None) -> tuple[float, ...]:
        b = self._belief(belief)
        return tuple(math.fsum(p * row[a] for p, row in zip(b, self.utilities))
                     for a in range(len(self.utilities[0])))

    def best_action(self, belief: Sequence[float] | None = None) -> int:
        values = self.terminal_action_values(belief)
        return max(range(len(values)), key=lambda a: values[a])

    def terminal_value(self, belief: Sequence[float] | None = None) -> float:
        return max(self.terminal_action_values(belief))

    def oracle_value(self, belief: Sequence[float] | None = None) -> float:
        b = self._belief(belief)
        return math.fsum(p * max(row) for p, row in zip(b, self.utilities))

    def risk(self, belief: Sequence[float] | None = None) -> float:
        r = self.oracle_value(belief) - self.terminal_value(belief)
        if r < -NORM_TOL:
            raise ArithmeticError("Bayes decision risk less than rounding tolerance")
        return max(0.0, r)

    def possible_outcomes(self, name: str, belief: Sequence[float] | None = None
                          ) -> tuple[tuple[int, float, tuple[float, ...]], ...]:
        b = self._belief(belief)
        channel = self.experiment(name).likelihood
        result = []
        for y in range(len(channel[0])):
            mass = math.fsum(p * row[y] for p, row in zip(b, channel))
            if mass <= 0.0:
                continue
            post = tuple(p * row[y] / mass for p, row in zip(b, channel))
            result.append((y, mass, post))
        return tuple(result)

    def posterior_outcomes(self, name: str, belief: Sequence[float] | None = None
                           ) -> tuple[tuple[float, tuple[float, ...]], ...]:
        return tuple((mass, posterior) for _, mass, posterior in
                     self.possible_outcomes(name, belief))

    def posterior_after(self, name: str, observation: int,
                        belief: Sequence[float] | None = None) -> tuple[float, ...]:
        for y, _, posterior in self.possible_outcomes(name, belief):
            if y == observation:
                return posterior
        raise ValueError("zero-probability or unknown observation")

    def evsi(self, name: str, belief: Sequence[float] | None = None) -> float:
        b = self._belief(belief)
        expected = math.fsum(m * self.terminal_value(after)
                             for _, m, after in self.possible_outcomes(name, b))
        gain = expected - self.terminal_value(b)
        if gain < -NORM_TOL:
            raise ArithmeticError("negative calibrated EVSI beyond rounding tolerance")
        return max(0.0, gain)

    def one_step(self, belief: Sequence[float] | None = None) -> dict[str, object]:
        b = self._belief(belief)
        gross = {e.name: self.evsi(e.name, b) for e in self.experiments}
        rate = {e.name: gross[e.name] / e.cost for e in self.experiments}
        net = {e.name: gross[e.name] - self.lambda_cost * e.cost for e in self.experiments}
        index = max(self.experiments, key=lambda e: rate[e.name]).name
        best = max(self.experiments, key=lambda e: net[e.name]).name
        stop = net[best] <= TIE_TOL
        return {
            "gross_gain": gross, "gain_per_cost": rate, "net_gains": net,
            "ratio_choice": index if rate[index] > self.lambda_cost + TIE_TOL else "STOP",
            "net_choice": "STOP" if stop else best,
            "should_stop": stop,
        }

    def bellman(self, horizon: int, belief: Sequence[float] | None = None
                ) -> tuple[float, str]:
        if type(horizon) is not int or horizon < 0 or horizon > MAX_HORIZON:
            raise ValueError("invalid bounded computation horizon")
        b = self._belief(belief)
        def value(post: tuple[float, ...], h: int) -> tuple[float, str]:
            best_value, best_choice = self.risk(post), "STOP"
            if h == 0:
                return best_value, best_choice
            for exp in self.experiments:
                alternative = self.lambda_cost * exp.cost + math.fsum(
                    prob * value(next_post, h - 1)[0]
                    for _, prob, next_post in self.possible_outcomes(exp.name, post))
                if alternative < best_value - TIE_TOL:
                    best_value, best_choice = alternative, exp.name
            return best_value, best_choice
        return value(b, horizon)
