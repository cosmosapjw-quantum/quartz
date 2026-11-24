"""Compare the native factor planner to normalized exact-world Bellman."""
import argparse
import json
import random
import subprocess
from fractions import Fraction as F
from itertools import product
from math import prod

from theory_exact import worlds

SUPPORT = (F(0), F(1, 2), F(1))
EPS = F(1, 10**12)


def model(req):
    mass = [(1 - req["m"]) / 2, (1 + req["m"]) / 2]
    factors = []
    for p, rho in zip(req["probabilities"], req["loadings"]):
        mu = sum(a * b for a, b in zip(p, SUPPORT))
        spread = max(abs(u - mu) for u, prob in zip(SUPPORT, p) if prob)
        factors.append(tuple(
            tuple(prob * (1 + z * rho * (u - mu) / spread) if spread else prob
                  for u, prob in zip(SUPPORT, p))
            for z in (-1, 1)
        ))
    return mass, factors


def oracle(req):
    q, factors = model(req)
    weighted = [
        (tuple(SUPPORT[k] for k in path), sum(
            q[z] * prod(row[z][k] for row, k in zip(factors, path)) for z in range(2)
        ), path) for path in product(range(3), repeat=len(factors))
    ]

    def value(states, h):
        b = max((req["b"], *(
            sum(w * us[i] for us, w, _ in states) for i in range(len(factors))
        )))
        options = []
        if h:
            for i, (ch, cost) in enumerate(zip(req["channels"], req["costs"])):
                v = -cost
                for y in range(len(ch[0])):
                    after = [(us, w * ch[path[i]][y], path) for us, w, path in states]
                    py = sum(w for _, w, _ in after)
                    if py:
                        v += py * value([(us, w / py, path) for us, w, path in after], h - 1)[0]
                options.append(v)
        best = max((b, *options))
        choice = next((i for i, x in enumerate(options) if x >= best - EPS), None) if best > b + EPS else None
        return best, choice

    return *value(weighted, req["horizon"]), worlds(q, factors, req["b"])


def sample():
    rng = random.Random(61124)
    for i in range(220):
        n = i % 5
        def law():
            counts = [rng.randrange(5) for _ in range(3)]
            counts[0] += not any(counts)
            return [F(x, sum(counts)) for x in counts]
        yield dict(mode="synthetic", id=str(i), probabilities=[law() for _ in range(n)],
                   loadings=[F(rng.randint(-3, 3), 4) for _ in range(n)],
                   channels=[[law() for _ in range(3)] for _ in range(n)],
                   costs=[F(rng.randrange(21), 32) for _ in range(n)],
                   b=F(i % 9, 8), m=F(i % 5 - 2, 2), horizon=i % 3)
    for delta in (F(-1, 10**9), F(0), F(1, 10**9), F(1, 10**6)):
        yield dict(mode="synthetic", id=str(delta), probabilities=[[F(1, 2), 0, F(1, 2)]],
                   loadings=[F(0)], channels=[[[1,0,0],[0,1,0],[0,0,1]]],
                   costs=[F(1, 4)+delta], b=F(1, 2), m=F(0), horizon=1)


def native(binary, requests):
    run = subprocess.run([binary], input="".join(json.dumps(x, default=float)+"\n" for x in requests),
                         capture_output=True, text=True, check=True, timeout=180)
    rows = [json.loads(line) for line in run.stdout.splitlines()]
    assert len(rows) == len(requests) and not run.stderr
    for req, res in zip(requests, rows):
        assert res.get("id") == req["id"] and "error" not in res, res
    return rows


def verify(binary):
    requests = list(sample())
    responses = native(binary, [dict(r, guard=g) for r in requests for g in (False, True)])
    stops = 0
    residual = 0
    for req, full, guarded in zip(requests, responses[::2], responses[1::2]):
        value, choice, perfect = oracle(req)
        for row in (full, guarded):
            error = max(abs(row["value"] - float(value)),
                        abs(row["perfect_information"] - float(perfect)))
            residual = max(residual, error)
            assert error < 5e-12 and row["choice"] == choice, (req, row, value, choice)
        assert (guarded["value"], guarded["choice"]) == (full["value"], full["choice"])
        if req["id"] in {"-1/1000000000", "0", "1/1000000000"}:
            assert guarded["guard_stops"] == 0
        stops += guarded["guard_stops"]
    assert stops > 0
    return {"models": len(requests), "native_responses": len(responses),
            "max_abs_error": residual, "guard_stops": stops}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary")
    print(json.dumps(verify(parser.parse_args().binary), indent=2))
