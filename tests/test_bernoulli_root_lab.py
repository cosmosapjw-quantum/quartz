"""Regression tests for the synthetic root-ranking-risk experiment."""

from __future__ import annotations

import gzip
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from quartz import experiment_manifest
from quartz.experiments import bernoulli_root as lab


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_BANK = REPO_ROOT / "configs" / "metacognitive_root_scenarios.v1.json"
SCRIPT = REPO_ROOT / "scripts" / "bernoulli_root_lab.py"


class BetaMathTests(unittest.TestCase):
    def test_integer_beta_cdf_known_cases_and_complement(self):
        self.assertAlmostEqual(lab.beta_cdf_integer(0.25, 1, 1), 0.25)
        self.assertAlmostEqual(lab.beta_cdf_integer(0.25, 2, 1), 0.25**2)
        for alpha, beta, x in ((1, 2, 0.2), (2, 3, 0.4), (5, 4, 0.7)):
            self.assertAlmostEqual(
                lab.beta_cdf_integer(x, alpha, beta),
                1.0 - lab.beta_cdf_integer(1.0 - x, beta, alpha),
                places=12,
            )

    def test_exact_one_step_kg(self):
        posteriors = [(1, 1), (1, 1)]
        # Current value 1/2. A success raises one arm to 2/3; a failure
        # leaves the other arm at 1/2, so KG = (1/2)(2/3)+(1/2)(1/2)-1/2.
        self.assertEqual(lab.one_step_kg_exact(posteriors, 0), Fraction(1, 12))
        self.assertEqual(lab.one_step_kg_exact(posteriors, 0), lab.one_step_kg_exact(posteriors, 1))


class AllocationTests(unittest.TestCase):
    def setUp(self):
        self.means = [0.60, 0.54, 0.48, 0.42]
        self.budget = 16
        self.tape = lab.RewardTape(self.means, self.budget, seed=11, trial=3)

    def test_runners_receive_no_true_means_and_use_exact_budget(self):
        for name, runner in lab.RUNNERS.items():
            with self.subTest(name=name):
                self.assertEqual(next(iter(inspect.signature(runner).parameters)), "num_arms")
                result = runner(
                    len(self.means),
                    self.budget,
                    self.tape,
                    lab.random.Random(lab.stable_seed("test", name)),
                )
                self.assertEqual(sum(result.pulls), self.budget)
                self.assertIn(result.selected_arm, range(len(self.means)))

    def test_uniform_balance_and_sequential_halving_guard(self):
        result = lab.run_uniform(len(self.means), self.budget, self.tape, lab.random.Random(7))
        self.assertEqual(result.pulls, [4, 4, 4, 4])
        with self.assertRaises(ValueError):
            lab.run_raw_sequential_halving(
                len(self.means),
                3,
                lab.RewardTape(self.means, 3, seed=1, trial=0),
                lab.random.Random(1),
            )

    def test_rank_risk_posterior_accounting(self):
        result = lab.run_kg_rank_risk(
            len(self.means), self.budget, self.tape, lab.random.Random(99)
        )
        self.assertEqual(result.kg_steps + result.fallback_steps, self.budget)
        self.assertTrue(all(success <= pulls for success, pulls in zip(result.successes, result.pulls)))

    def test_canonical_reward_tape_is_permutation_coupled(self):
        canonical_means = [0.61, 0.53, 0.47, 0.39]
        permutation = [2, 0, 3, 1]
        direct = lab.RewardTape(canonical_means, 12, seed=5, trial=7, arm_keys=[0, 1, 2, 3])
        permuted = lab.RewardTape(
            [canonical_means[index] for index in permutation],
            12,
            seed=5,
            trial=7,
            arm_keys=permutation,
        )
        for presented, canonical in enumerate(permutation):
            for pull_index in range(12):
                self.assertEqual(
                    permuted.pull(presented, pull_index),
                    direct.pull(canonical, pull_index),
                )

    def test_equal_arm_null_has_no_fixed_index_winner(self):
        records, _ = lab.run_experiment(
            [0.5] * 8,
            [8],
            trials=400,
            seed=20260713,
        )
        for algorithm in lab.ALGORITHMS:
            counts = [0] * 8
            for row in records:
                if row.algorithm == algorithm:
                    counts[row.selected_arm] += 1
            # This broad smoke guard catches deterministic/tie-index bugs; it
            # is not a formal uniformity test or a scientific acceptance gate.
            self.assertGreater(min(counts), 20, (algorithm, counts))
            self.assertLess(max(counts), 80, (algorithm, counts))


class ExperimentTests(unittest.TestCase):
    def test_reproducibility_pairing_and_validation(self):
        kwargs = dict(means=[0.60, 0.55, 0.50, 0.45], budgets=[4, 8], trials=25, seed=123)
        records_a, summaries_a = lab.run_experiment(**kwargs)
        records_b, summaries_b = lab.run_experiment(**kwargs)
        self.assertEqual(records_a, records_b)
        self.assertEqual(summaries_a, summaries_b)
        for row in summaries_a:
            self.assertEqual(row["paired_trials_vs_uniform"], 25)
            self.assertGreaterEqual(row["mean_simple_regret"], 0.0)
            self.assertTrue(0.0 <= row["probability_correct_selection"] <= 1.0)
        with self.assertRaises(ValueError):
            lab.run_experiment([0.6, 0.5], [2, 2], 2, 1)
        with self.assertRaises(ValueError):
            lab.run_experiment([0.6, 0.5], [2], 2, 1, algorithms=[])
        with self.assertRaises(ValueError):
            lab.run_experiment([0.6, 0.5], [2], 2, 1, algorithms=["uniform", "uniform"])

    def test_paired_fields_null_without_uniform(self):
        _, rows = lab.run_experiment(
            [0.60, 0.55, 0.50, 0.45],
            [4],
            trials=5,
            seed=9,
            algorithms=["kg_rank_risk"],
        )
        self.assertEqual(rows[0]["paired_trials_vs_uniform"], 0)
        self.assertIsNone(rows[0]["paired_mean_regret_delta_vs_uniform"])

    def test_manifest_contract_hash_is_config_deterministic(self):
        kwargs = dict(
            experiment_id="test",
            execution_mode="synthetic_screening",
            resolved_config={"seed": 1, "budgets": [4, 8]},
            repo_root=REPO_ROOT,
            source_paths=[SCENARIO_BANK],
            argv=["test"],
            assumptions=["iid"],
            prohibited_inferences=["mcts transfer"],
        )
        first = experiment_manifest.build_run_manifest(started_at="A", **kwargs)
        second = experiment_manifest.build_run_manifest(started_at="B", **kwargs)
        self.assertEqual(first["run_contract_hash"], second["run_contract_hash"])
        self.assertEqual(first["sources"], second["sources"])

    def test_cli_quick_smoke_and_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            command = [
                sys.executable,
                str(SCRIPT),
                "--scenario-bank",
                str(SCENARIO_BANK),
                "--scenarios",
                "top2_near_tie_k8",
                "--quick",
                "--seed",
                "17",
                "--output-dir",
                str(output),
            ]
            proc = subprocess.run(command, cwd=Path(tmp), capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["claim_status"], "synthetic_screening_only")
            self.assertEqual(len(manifest["artifacts"]), 4)
            for row in manifest["artifacts"]:
                path = output / row["path"]
                self.assertEqual(experiment_manifest.file_sha256(path), row["sha256"])
            with gzip.open(output / "trials.jsonl.gz", "rt", encoding="utf-8") as handle:
                trial_rows = [json.loads(line) for line in handle]
            self.assertEqual(len(trial_rows), 50 * 2 * 4 * len(lab.ALGORITHMS))
            self.assertIn("selected_canonical_arm", trial_rows[0])


if __name__ == "__main__":
    unittest.main()
