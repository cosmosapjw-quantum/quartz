"""Independent finite-world tests for QUARTZ COG-01 (stdlib only)."""
from fractions import Fraction as F
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cognitive_exact import (
    Scenario, SOURCE, SCENARIOS, U, COST, posterior, gross_gain,
    evaluate, atoms,
)


class Cog01ExactTests(unittest.TestCase):
    def test_source_posterior_identity(self):
        self.assertEqual(posterior(SOURCE, "+"), F(17, 33))
        self.assertEqual(posterior(SOURCE, "-"), F(3, 67))
        self.assertEqual(sum(pr for pr, _, _ in atoms(SOURCE)), 1)

    def test_gross_gain_piecewise_and_peak(self):
        self.assertEqual(gross_gain(F(0)), 0)
        self.assertEqual(gross_gain(F(1)), 0)
        self.assertEqual(gross_gain(U), U * (1 - U))
        self.assertEqual(gross_gain(F(17, 33)), F(34, 165))

    def test_exact_in_distribution_pins(self):
        r = evaluate(SOURCE, SOURCE)
        self.assertEqual(r["stop_only"], F(3, 5))
        self.assertEqual(r["learned_selective"], F(127, 200))
        self.assertEqual(r["random_matched_rate"], F(2967, 5000))
        self.assertEqual(r["selective_rate"], F(33, 100))
        self.assertEqual(r["learned_selective"] - r["random_matched_rate"], F(26, 625))
        self.assertEqual(r["quality_ece"], 0)
        self.assertEqual(r["gain_ece"], 0)

    def test_negative_inversion_preserved(self):
        r = evaluate(SOURCE, SCENARIOS["inverted_cue"])
        self.assertEqual(r["learned_selective"], F(68, 125))
        self.assertLess(r["learned_selective"], r["stop_only"])
        self.assertGreater(r["gain_ece"], 0)
        self.assertGreater(r["quality_ece"], 0)

    def test_monotone_oracle_within_each_world(self):
        for label, world in SCENARIOS.items():
            with self.subTest(label=label):
                r = evaluate(SOURCE, world)
                self.assertGreaterEqual(r["oracle_recalibrated"], r["learned_selective"])
                self.assertGreaterEqual(r["full_information_no_cost"], r["oracle_recalibrated"])
                self.assertEqual(r["selective_rate"], r["random_matched_rate_rate"])

    def test_independent_four_atom_enumeration(self):
        # Separate reference: expand probabilities for X=0/1, S=+/- directly.
        qplus = F(17, 33)
        qminus = F(3, 67)
        for label, world in SCENARIOS.items():
            with self.subTest(label=label):
                pq = {"+": qplus, "-": qminus}
                joint = {
                    (1, "+"): world.p * world.se,
                    (1, "-"): world.p * (1 - world.se),
                    (0, "+"): (1 - world.p) * world.fpr,
                    (0, "-"): (1 - world.p) * (1 - world.fpr),
                }
                self.assertEqual(sum(joint.values()), 1)
                qpred = {s: min(q, 1) for s, q in pq.items()}
                inspect = {s: min(q * (1 - U), U * (1 - q)) > COST for s, q in qpred.items()}
                ref_rate = sum(prob for (x, s), prob in joint.items() if inspect[s])
                ref_base = sum(prob * (F(x) if qpred[s] > U else U) for (x, s), prob in joint.items())
                ref_select = sum(
                    prob * ((max(U, F(x)) - COST) if inspect[s] else (F(x) if qpred[s] > U else U))
                    for (x, s), prob in joint.items()
                )
                ref_random = sum(
                    prob * (ref_rate * (max(U, F(x)) - COST) + (1 - ref_rate) * (F(x) if qpred[s] > U else U))
                    for (x, s), prob in joint.items()
                )
                ref_brier = sum(prob * (qpred[s] - x) ** 2 for (x, s), prob in joint.items())
                r = evaluate(SOURCE, world)
                for key, value in [
                    ("selective_rate", ref_rate), ("stop_only", ref_base),
                    ("learned_selective", ref_select),
                    ("random_matched_rate", ref_random), ("quality_brier", ref_brier)
                ]:
                    self.assertEqual(r[key], value, (label, key))

    def test_zero_probability_cue(self):
        only_negative = Scenario(F(0), F(0), F(0))
        self.assertIsNone(posterior(only_negative, "+"))
        r = evaluate(SOURCE, only_negative)
        self.assertEqual(r["quality_brier"], F(9, 4489))  # q(-)^2
        self.assertEqual(r["selective_rate"], 0)

    def test_unseen_source_cue_rejected(self):
        source_only_positive = Scenario(F(1), F(1), F(0))
        with self.assertRaises(ValueError):
            evaluate(source_only_positive, SOURCE)

    def test_invalid_probabilities_rejected(self):
        for params in [
            (F(-1, 10), F(1, 2), F(1, 2)),
            (F(1, 2), F(11, 10), F(1, 2)),
            (F(1, 2), F(1, 2), F(2)),
        ]:
            with self.subTest(params=params), self.assertRaises(ValueError):
                Scenario(*params)


if __name__ == "__main__":
    unittest.main()
