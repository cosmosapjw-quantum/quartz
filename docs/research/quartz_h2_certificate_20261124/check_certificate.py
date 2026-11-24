"""Independent normalized Bellman oracle and end-to-end checks."""
import json
import random
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path

from certificate import Problem, Query, parse_problem, solve


def plan_tuple(plan):
    return plan.choice, plan.deploy, tuple((y, plan_tuple(p)) for y, p in plan.children)


def oracle(problem, belief, horizon):
    utilities = [(problem.outside, *row) for row in problem.utilities]
    means = [sum(p * u for p, u in zip(belief, col)) for col in zip(*utilities)]
    optimum = max(means)
    decision = means.index(optimum)
    plan = (None, decision - 1 if decision else None, ())
    if horizon == 0:
        return optimum, plan
    for i, query in enumerate(problem.queries):
        distribution = [query.channel[row[query.candidate]] for row in problem.utilities]
        value = -query.cost
        children = []
        for y in range(len(distribution[0])):
            joint = tuple(p * F(row[y], problem.denominator) for p, row in zip(belief, distribution))
            p_y = sum(joint)
            if p_y:
                child_value, child_plan = oracle(problem, tuple(p / p_y for p in joint), horizon - 1)
                value += p_y * child_value
                children.append((y, child_plan))
        if value > optimum:
            optimum, plan = value, (i, None, tuple(children))
    return optimum, plan


def cases(seed, count):
    rng = random.Random(seed)
    for _ in range(count):
        width = rng.randint(1, 5)
        rows = tuple(tuple(rng.randint(-2, 3) for _ in range(width)) for _ in range(rng.randint(1, 9)))
        weights = (rng.randint(1, 4), *(rng.randrange(5) for _ in rows[1:]))
        queries = []
        for _ in range(rng.randint(0, width + 2)):
            i = rng.randrange(width)
            size = rng.randint(1, 3)
            channel = {}
            for value in sorted({row[i] for row in rows}):
                outcomes = [0] * size
                for _ in range(4):
                    outcomes[rng.randrange(size)] += 1
                channel[value] = tuple(outcomes)
            queries.append(Query(i, rng.randrange(5), channel))
        yield Problem(rows, weights, rng.randint(-2, 3), 4, tuple(queries))


class CertificateTests(unittest.TestCase):
    def fixture(self):
        return parse_problem(json.loads(Path(__file__).with_name("fixture.json").read_text()))

    def test_fixed_latent_and_cli(self):
        run = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("certificate.py")),
             str(Path(__file__).with_name("fixture.json"))],
            check=True, capture_output=True, text=True,
        )
        data = json.loads(run.stdout)
        self.assertEqual(F(data["value"]) / 100, F(543, 800))
        self.assertIsNone(data["plan"]["children"][0][1]["choice"])
        self.assertEqual(data["plan"]["children"][1][1]["choice"], 0)

    def test_invalid(self):
        p = self.fixture()
        bad_channel = Query(0, 0, {0: (4, 0), 100: (1, 1)})
        missing_support = Query(0, 0, {0: (4,)})
        for bad in [replace(p, weights=(0, 0)), replace(p, weights=(-1, 2)),
                    replace(p, utilities=((0,), (0, 1))), replace(p, denominator=0),
                    replace(p, queries=(bad_channel,)), replace(p, queries=(missing_support,)),
                    replace(p, queries=(replace(p.queries[0], cost=-1),))]:
            with self.assertRaises(ValueError):
                solve(bad)
        for h in (-1, 3, True):
            with self.assertRaises(ValueError):
                solve(p, h)

    def test_aliased_and_duplicate_cli_keys(self):
        path = Path(__file__).with_name("fixture.json")
        payload = path.read_text()
        replacements = {
            '"0": [3, 1], "00": [1, 3]': "channel utility keys must be canonical integers",
            '"0": [3, 1], "0": [1, 3]': "duplicate JSON key",
        }
        with tempfile.TemporaryDirectory() as temp:
            malformed = Path(temp) / "bad.json"
            for fields, error in replacements.items():
                with self.subTest(error=error):
                    malformed.write_text(payload.replace('"0": [3, 1]', fields))
                    run = subprocess.run(
                        [sys.executable, str(path.with_name("certificate.py")), str(malformed)],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(run.returncode, 2)
                    self.assertIn(error, run.stderr)

    def test_terminal_only_and_zero_cost_tie(self):
        self.assertEqual(solve(Problem(((),), (1,), -3, 2, ())).value, -3)
        p = Problem(((1, 1), (1, 1)), (1, 2), 1, 2, (Query(0, 0, {1: (2, 0)}),))
        for mode in (None, "world", "both"):
            answer = solve(p, bounds=mode)
            self.assertEqual(plan_tuple(answer.plan), (None, None, ()))
            self.assertEqual(answer.value, 1)

    def test_garbling_prunes_irrelevant_query(self):
        p = Problem(((0, 0), (0, 2), (1, 0), (1, 2)), (1, 1, 1, 1), 1, 2,
                    (Query(0, 0, {0: (2, 0), 1: (0, 2)}),))
        result = solve(p, 1)
        self.assertEqual(result.value, 1)
        self.assertEqual(result.counts.reveal_prunes, 1)
        self.assertEqual(result.counts.world_prunes, 0)

    def test_randomized_exact_policy_and_value(self):
        reasons = {"world_prunes": 0, "reveal_prunes": 0}
        for p in cases(20261124, 240):
            belief = tuple(F(w, sum(p.weights)) for w in p.weights)
            for h in range(3):
                expected = oracle(p, belief, h)
                for mode in (None, "world", "both"):
                    actual = solve(p, h, mode)
                    self.assertEqual((actual.value, plan_tuple(actual.plan)), expected)
                for key in reasons:
                    reasons[key] += getattr(actual.counts, key)
        self.assertTrue(all(reasons.values()), reasons)


if __name__ == "__main__":
    unittest.main(verbosity=2)
