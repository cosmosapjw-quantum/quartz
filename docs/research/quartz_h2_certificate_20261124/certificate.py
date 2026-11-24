"""Exact finite-world H0–H2 metareasoning with optional upper bounds."""
import argparse
import json
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Query:
    candidate: int
    cost: int
    channel: dict[int, tuple[int, ...]]


@dataclass(frozen=True)
class Problem:
    utilities: tuple[tuple[int, ...], ...]
    weights: tuple[int, ...]
    outside: int
    denominator: int
    queries: tuple[Query, ...]

    def validate(self):
        if not self.utilities or len(self.utilities) != len(self.weights):
            raise ValueError("one weight per nonempty world table")
        if type(self.denominator) is not int or self.denominator <= 0:
            raise ValueError("invalid channel denominator")
        if any(type(w) is not int or w < 0 for w in self.weights) or not sum(self.weights):
            raise ValueError("invalid world weights")
        width = len(self.utilities[0])
        if type(self.outside) is not int or any(
            len(row) != width or any(type(u) is not int for u in row)
            for row in self.utilities
        ):
            raise ValueError("invalid integer utility table")
        for query in self.queries:
            if type(query.candidate) is not int or not 0 <= query.candidate < width:
                raise ValueError("invalid query candidate")
            if type(query.cost) is not int or query.cost < 0:
                raise ValueError("invalid query cost")
            if set(query.channel) != {row[query.candidate] for row in self.utilities}:
                raise ValueError("channel utility support mismatch")
            sizes = {len(row) for row in query.channel.values()}
            if len(sizes) != 1 or not min(sizes) or any(
                any(type(k) is not int or k < 0 for k in row) or sum(row) != self.denominator
                for row in query.channel.values()
            ):
                raise ValueError("channel rows must have common size and sum to denominator")


@dataclass(frozen=True)
class Plan:
    choice: int | None
    deploy: int | None = None
    children: tuple[tuple[int, "Plan"], ...] = ()


@dataclass
class Counts:
    queries: int = 0
    branches: int = 0
    world_prunes: int = 0
    reveal_prunes: int = 0
    reveal_bounds: int = 0


@dataclass(frozen=True)
class Answer:
    value: Fraction
    plan: Plan
    counts: Counts


def _stop(problem, weights):
    best = problem.outside * sum(weights)
    action = None
    for i, col in enumerate(zip(*problem.utilities)):
        score = sum(w * u for w, u in zip(weights, col))
        if score > best:
            best, action = score, i
    return best, action


def _perfect_worlds(problem, weights):
    return sum(w * max((problem.outside, *row)) for w, row in zip(weights, problem.utilities))


def _perfect_query(problem, weights, candidate):
    buckets = {}
    for w, row in zip(weights, problem.utilities):
        if not w:
            continue
        scores = buckets.setdefault(row[candidate], [0] * (len(row) + 1))
        scores[0] += w * problem.outside
        for i, utility in enumerate(row, start=1):
            scores[i] += w * utility
    return sum(max(scores) for scores in buckets.values())


def _solve(problem, weights, horizon, bounds, counts):
    scale = problem.denominator**horizon
    score, deploy = _stop(problem, weights)
    best, plan = scale * score, Plan(None, deploy)
    if not horizon or not problem.queries:
        return best, plan
    mass = sum(weights)
    world_upper = scale * _perfect_worlds(problem, weights) if bounds else None
    for i, query in enumerate(problem.queries):
        cost = scale * query.cost * mass
        if world_upper is not None and world_upper - cost <= best:
            counts.world_prunes += 1
            continue
        if bounds == "both" and horizon == 1:
            counts.reveal_bounds += 1
            possible = _perfect_query(problem, weights, query.candidate)
            if scale * possible - cost <= best:
                counts.reveal_prunes += 1
                continue
        counts.queries += 1
        total = -cost
        children = []
        rows = [query.channel[row[query.candidate]] for row in problem.utilities]
        for outcome, likelihoods in enumerate(zip(*rows)):
            child_weights = tuple(w * k for w, k in zip(weights, likelihoods))
            if not any(child_weights):
                continue
            counts.branches += 1
            value, child = _solve(problem, child_weights, horizon - 1, bounds, counts)
            total += value
            children.append((outcome, child))
        if total > best:
            best, plan = total, Plan(i, children=tuple(children))
    return best, plan


def solve(problem, horizon=2, bounds="both"):
    if type(horizon) is not int or not 0 <= horizon <= 2:
        raise ValueError("horizon must be 0, 1 or 2")
    if bounds not in ("both", "world", None):
        raise ValueError("unknown bound mode")
    problem.validate()
    counts = Counts()
    score, plan = _solve(problem, problem.weights, horizon, bounds, counts)
    scale = problem.denominator**horizon * sum(problem.weights)
    return Answer(Fraction(score, scale), plan, counts)


def parse_problem(data):
    queries = []
    for query in data["queries"]:
        channel = {}
        for key, row in query["channel"].items():
            utility = int(key)
            if str(utility) != key:
                raise ValueError("channel utility keys must be canonical integers")
            channel[utility] = tuple(row)
        queries.append(Query(query["candidate"], query["cost"], channel))
    return Problem(
        tuple(tuple(row) for row in data["utilities"]),
        tuple(data["weights"]),
        data["outside"],
        data["denominator"],
        tuple(queries),
    )


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def main():
    from dataclasses import asdict
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", type=Path)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--bounds", choices=["both", "world", "none"], default="both")
    args = parser.parse_args()
    try:
        problem = parse_problem(json.loads(args.problem.read_text(), object_pairs_hook=_unique_object))
        answer = solve(problem, args.horizon, None if args.bounds == "none" else args.bounds)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(f"invalid problem: {exc}")
    payload = asdict(answer)
    payload["value"] = str(answer.value)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
