"""Measure exact reference policy work and elapsed time on contrasting synthetic models."""
import gc
import json
import platform
import random
import statistics
import time
from dataclasses import asdict

from certificate import Problem, Query, solve


def regimes():
    rng = random.Random(1524)
    levels = (0, 10, 20)
    channel = {0: (8, 1, 1), 10: (1, 8, 1), 20: (1, 1, 8)}
    rows = tuple(tuple(rng.choice(levels) for _ in range(8)) for _ in range(40))
    weights = tuple(rng.randint(1, 4) for _ in rows)
    for name, cost, outside in [("saturated", 1, 20), ("expensive", 16, 9),
                                ("competitive", 1, 9), ("free", 0, 9)]:
        yield name, Problem(rows, weights, outside, 10,
                            tuple(Query(i, cost, channel) for i in range(8)))
    irrelevant_rows = tuple((i, j, k) for i in (0, 10) for j in (0, 10) for k in (0, 10))
    weak = {0: (6, 4), 10: (4, 6)}
    yield "uninformative_for_choice", Problem(
        irrelevant_rows, (1,) * 8, 9, 10, tuple(Query(i, 0, weak) for i in range(3))
    )


def run():
    results = {}
    modes = (None, "world", "both")
    for name, problem in regimes():
        answers = [solve(problem, bounds=mode) for mode in modes]
        assert all((a.value, a.plan) == (answers[0].value, answers[0].plan) for a in answers)
        timings = [[] for _ in modes]
        for repeat in range(60):
            gc.collect()
            for index in ((0, 1, 2) if repeat % 2 == 0 else (2, 1, 0)):
                start = time.perf_counter_ns()
                solve(problem, bounds=modes[index])
                timings[index].append(time.perf_counter_ns() - start)
        results[name] = {
            label: {"median_ns": statistics.median(elapsed), **asdict(answer.counts)}
            for label, elapsed, answer in zip(("full", "world", "both"), timings, answers)
        }
        results[name]["value"] = str(answers[0].value)
    return {"python": platform.python_version(), "machine": platform.machine(),
            "repeats": 60, "regimes": results}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
