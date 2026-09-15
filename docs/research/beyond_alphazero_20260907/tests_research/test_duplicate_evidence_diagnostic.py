"""Analytic boundaries and independent rational oracles for the W1 diagnostic."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from duplicate_evidence_diagnostic import evaluate_case


class GaussianEvidenceTests(unittest.TestCase):
    def test_one_observation_all_cases_agree(self):
        for kind in ("exact_replay", "shared_error", "independent"):
            row = evaluate_case(1, kind)
            self.assertEqual(row["correct_posterior_variance"], 0.5)
            self.assertEqual(row["iid_posterior_variance"], 0.5)
            self.assertAlmostEqual(row["iid_prior_predictive_coverage"], .95, places=14)

    def test_replay_never_adds_information(self):
        row = evaluate_case(4096, "exact_replay")
        self.assertEqual(row["correct_posterior_variance"], .5)
        self.assertEqual(row["distinct_observations"], 1)
        self.assertAlmostEqual(row["iid_actual_mse"], (1+4096**2)/4097**2)

    def test_shared_error_matches_independent_rational_oracle(self):
        # Derived by conditioning the full observation vector in the independent review.
        for n in (1, 2, 3, 8, 64, 512, 4096):
            row = evaluate_case(n, "shared_error")
            self.assertAlmostEqual(row["correct_posterior_variance"], (n+3)/(5*n+3))
            self.assertAlmostEqual(row["iid_actual_mse"], (n*n+3*n+4)/(4*(n+1)**2))

    def test_independent_case_is_calibrated(self):
        for n in (8, 64, 4096):
            row = evaluate_case(n, "independent")
            self.assertEqual(row["correct_posterior_variance"], row["iid_posterior_variance"])
            self.assertAlmostEqual(row["iid_actual_mse"], row["iid_posterior_variance"])
            self.assertAlmostEqual(row["iid_prior_predictive_coverage"], .95, places=14)

    def test_two_different_shared_error_floors(self):
        row = evaluate_case(10**9, "shared_error")
        self.assertAlmostEqual(row["true_mean_noise_variance"], .25, places=8)
        self.assertAlmostEqual(row["correct_posterior_variance"], .2, places=8)
        self.assertLess(row["iid_prior_predictive_coverage"], .001)
        self.assertTrue(math.isfinite(row["iid_actual_mse"]))

    def test_invalid_design_rejected(self):
        for n in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                evaluate_case(n, "shared_error")
        with self.assertRaises(ValueError):
            evaluate_case(1, "unknown")


if __name__ == "__main__":
    unittest.main()
