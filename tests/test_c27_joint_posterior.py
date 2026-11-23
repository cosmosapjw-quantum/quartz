"""C27 semantic regression tests; expectations derive from frozen exact fixtures.

Breaks caught: replacing joint conditioning by independent marginals; allowing fold
leakage; charging microseconds as nanoseconds; hiding nondeterministic repeat traces;
accepting duplicate observations or unsupported states.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/research/quartz_general_joint_posterior_20261123"
PATH = DOC / "coding/c27_campaign.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


h = load(PATH, "c27_harness_test") if PATH.exists() else None
theory = load(DOC / "theory/exact_joint_oracle.py", "c27_exact_test")


class SemanticRegression(unittest.TestCase):
    def test_information_query_value_is_109_over_200(self):
        if h is None:
            # Actual pre-C27 product closure is the meaningful RED baseline.
            F = theory.F
            laws = ((F(1, 2), F(1, 2), F(0)), (F(1, 2), F(0), F(1, 2)))
            value = float(
                theory.bellman(
                    theory.joint(laws, F(0)), (F(1, 100), F(3, 10)), F(1, 2), (0, 1), 2
                )
            )
        else:
            dist = h.joint_worlds(((0.5, 0.5, 0), (0.5, 0, 0.5)), 0.8)
            scores = h.world_scores(dist, {0: 0.01, 1: 0.3}, 0.5, [0, 1], 2)
            value = 0.5 + max(0, *(s["k2"] for s in scores))
        self.assertAlmostEqual(value, 109 / 200, places=13)

    def test_reveal_changes_other_marginal_to_082(self):
        if h is None:
            posterior = 0.5
        else:
            dist = h.joint_worlds(((0.5, 0.5, 0), (0.5, 0, 0.5)), 0.8)
            posterior = h.world_marginal(h.observe_worlds(dist, 0, 0.5), 1)[2]
        self.assertAlmostEqual(posterior, 0.82, places=13)


class BoundaryTests(unittest.TestCase):
    def test_rho_zero_preserves_unobserved_marginal(self):
        dist = h.joint_worlds(((0.2, 0.3, 0.5), (0.6, 0.1, 0.3)), 0)
        post = h.observe_worlds(dist, 0, 1)
        for a, b in zip(h.world_marginal(post, 1), (0.6, 0.1, 0.3)):
            self.assertAlmostEqual(a, b, places=13)

    def test_unsupported_observation_is_rejected(self):
        with self.assertRaises(ValueError):
            h.observe_worlds(h.joint_worlds(((1, 0, 0),), 0.9), 0, 1)

    def test_known_win_and_nonpositive_gain_stop(self):
        dist = h.joint_worlds(((0.5, 0, 0.5),), 0.8)
        self.assertIsNone(h.pick_score(h.world_scores(dist, {0: 0.01}, 1, [0], 2), 2))
        self.assertIsNone(h.pick_score(h.world_scores(dist, {0: 0.5}, 0, [0], 1), 1))

    def test_tie_uses_ascending_move(self):
        entries = [
            {"move": 4, "k1": 0.2, "k2": 0.3},
            {"move": 2, "k1": 0.2, "k2": 0.3 + 5e-13},
        ]
        self.assertEqual(h.pick_score(entries, 2), 2)
        chain = [
            {"move": i, "k1": 0.2 + i * 0.9e-12, "k2": 0.2 + i * 0.9e-12}
            for i in range(3)
        ]
        self.assertEqual(h.pick_score(chain, 2), 1)

    def test_projected_and_joint_differ_after_second_observation(self):
        dist = h.joint_worlds(((0.5, 0, 0.5),) * 3, 0.8)
        after_first = h.observe_worlds(dist, 0, 1)
        projected = h.joint_worlds(
            [h.world_marginal(after_first, i) for i in range(3)], 0
        )
        joint_p = h.world_marginal(h.observe_worlds(after_first, 1, 1), 2)[2]
        frozen_p = h.world_marginal(h.observe_worlds(projected, 1, 1), 2)[2]
        self.assertGreater(joint_p, frozen_p)

    def test_price_is_nanoseconds_per_node(self):
        price = h.price_from_measurements(
            [
                {"elapsed_ns": 3000, "solve_nodes": 100},
                {"elapsed_ns": 1000, "solve_nodes": 100},
            ]
        )
        self.assertEqual(price["exact_ns_per_node"], 20)
        self.assertAlmostEqual(price["lambda_time_ns"], (1 / 1024) / 20)

    def test_zero_price_denominator_rejected(self):
        with self.assertRaises(ValueError):
            h.price_from_measurements([{"elapsed_ns": 0, "solve_nodes": 0}])

    def test_fold_assignments_are_root_disjoint_and_stratified(self):
        sources = [
            {
                "id": f"r{ply}-{i}",
                "ply": ply,
                "split": "TRAIN",
                "source_hash": f"{i:064x}",
            }
            for ply in (4, 5, 6)
            for i in range(30)
        ]
        folds = h.assign_folds(list(reversed(sources)))
        self.assertEqual(
            [sum(v == f for v in folds.values()) for f in range(3)], [30, 30, 30]
        )
        for ply in (4, 5, 6):
            self.assertEqual(folds[f"r{ply}-0"], 0)
            self.assertEqual(folds[f"r{ply}-10"], 1)
            self.assertEqual(folds[f"r{ply}-29"], 2)

    def test_fold_refit_does_not_use_heldout_labels(self):
        sources = [
            {
                "id": f"r{ply}-{i}",
                "ply": ply,
                "split": "TRAIN",
                "source_hash": f"{i:064x}",
            }
            for ply in (4, 5, 6)
            for i in range(30)
        ]
        rows = [
            {
                "id": s["id"],
                "source_hash": s["source_hash"],
                "response": {
                    "candidates": [
                        {
                            "move": 0,
                            "remaining": 2,
                            "counts": [1, 2, 1],
                            "u": 0.0,
                            "solve_nodes": 3,
                            "terminal_known": False,
                        }
                    ]
                },
            }
            for s in sources
        ]
        folds = h.assign_folds(sources)
        before, provenance = h.fit_fold(rows, folds, 0)
        for row in rows:
            if folds[row["id"]] == 0:
                row["response"]["candidates"][0]["u"] = 1.0
        after, changed = h.fit_fold(rows, folds, 0)
        self.assertEqual(before, after)
        self.assertEqual(provenance, changed)
        self.assertEqual(len(provenance["sources"]), 60)

    def test_likelihood_rho_zero_exactly_zero(self):
        self.assertEqual(
            h.log_ratio(((0.2, 0.3, 0.5), (0.6, 0.1, 0.3)), (1, 0.5), 0), 0.0
        )

    def test_product_likelihood_ratio_matches_explicit_world(self):
        laws = ((0.2, 0.3, 0.5), (0.6, 0.1, 0.3))
        dist = h.joint_worlds(laws, 0.8)
        prob = sum(p for w, p in dist if w == (1, 0.5))
        import math

        self.assertAlmostEqual(
            h.log_ratio(laws, (1, 0.5), 0.8), math.log(prob / (0.5 * 0.1)), places=13
        )

    def test_runtime_request_does_not_forward_offline_fields(self):
        source = {
            "id": "fixture",
            "history": [0, 1],
            "cue_seed": 1,
            "u": [0, 1],
            "offline_oracle": "forbidden",
        }
        request = h.run_request(source, {"rho": 0.8, "model": {"toy": True}}, 0)
        self.assertNotIn("u", request)
        self.assertNotIn("offline_oracle", request)
        self.assertEqual(request["additional"], 2)

    def test_repeat_identity_ignores_only_times(self):
        a = {
            "policy": "JOINT_H2",
            "repeat": 0,
            "elapsed_ns": 100,
            "decisions": [{"chosen_move": 2, "queried_u": 0.5}],
        }
        b = {
            "policy": "JOINT_H2",
            "repeat": 1,
            "elapsed_ns": 200,
            "decisions": [{"chosen_move": 2, "queried_u": 0.5}],
        }
        h.assert_repeat_identity([a, b])
        b["decisions"][0]["queried_u"] = 1
        with self.assertRaises(AssertionError):
            h.assert_repeat_identity([a, b])

    def test_bootstrap_constant_difference_has_exact_endpoints(self):
        result = h.root_bootstrap([0.125] * 6, draws=100, seed=2709)
        self.assertEqual(result["mean"], 0.125)
        self.assertEqual(result["ci95"], [0.125, 0.125])

    def test_exact_fraction_fixture_matches_float_worlds(self):
        F = theory.F
        laws = ((F(1, 3), F(1, 3), F(1, 3)), (F(1, 2), F(0), F(1, 2)))
        exact = dict(theory.joint(laws, F(3, 4)))
        actual = dict(h.joint_worlds([[float(x) for x in law] for law in laws], 0.75))
        for world, probability in exact.items():
            self.assertAlmostEqual(
                actual[tuple(float(x) for x in world)], float(probability), places=13
            )


class AggregationTests(unittest.TestCase):
    def test_seed_medians_precede_root_average_and_nanosecond_charge(self):
        # Fails if timing repeats become roots, means replace medians, or ns are
        # silently treated as us. Fixture timings have a deliberate outlier.
        records = []
        for seed in range(4):
            runs = []
            for repeat, elapsed in enumerate([1000, 2000, 3000, 4000, 100000]):
                for policy in h.POLICIES:
                    runs.append(
                        {
                            "policy": policy,
                            "repeat": repeat,
                            "execution_slot": (
                                h.POLICIES.index(policy) - (seed + repeat)
                            )
                            % 5,
                            "initial_u": 0.5,
                            "final_u": 0.5,
                            "charged_nodes": 20,
                            "cue_nodes": 10,
                            "initial_nodes": 10,
                            "additional_nodes": 0,
                            "additional_queries": 0,
                            "query_moves": [],
                            "predicted_additional_nodes": 0.0,
                            "elapsed_ns": elapsed + 10 * seed,
                            "cue_ns": 100,
                            "initial_ns": 100,
                            "query_ns": 0,
                            "controller_ns": elapsed + 10 * seed - 200,
                            "decisions": [],
                        }
                    )
            records.append(
                {"id": "toy", "ply": 4, "seed": seed, "response": {"runs": runs}}
            )
        seed_rows, roots, dispersion = h.aggregate_records(
            records, {"toy": {"root_u": 1}}, {"lambda_time_ns": 0.00001}
        )
        self.assertEqual(len(seed_rows), 20)
        self.assertEqual(len(roots), 5)
        self.assertEqual(len(dispersion), 20)
        joint = next(row for row in roots if row["policy"] == "JOINT_H2")
        self.assertEqual(joint["metrics"]["elapsed_ns"], 3015)
        self.assertAlmostEqual(joint["metrics"]["net_wall"], 0.46985)
        self.assertEqual(joint["metrics"]["regret"], 0.5)

    def test_diagnostics_pair_predictors_within_identical_exposure(self):
        common = {
            "id": "toy",
            "seed": 0,
            "generating_policy": "JOINT_H2",
            "history_index": 0,
            "stage": "initial",
            "move": 2,
            "selected": True,
            "predicted_direct_gain": 0.1,
            "realized_direct_gain": 0.0,
            "selected_gain_residual": 0.1,
        }
        records = [
            {**common, "predictor": predictor, "log_loss": loss, "brier": brier}
            for predictor, loss, brier in [
                ("joint", 0.2, 0.1),
                ("product", 0.5, 0.4),
                ("initial_projected", 0.2, 0.1),
            ]
        ]
        result = h.summarize_diagnostics(records)
        paired = result["matched_history_differences"][0]
        self.assertEqual(paired["exposures"], 1)
        self.assertAlmostEqual(paired["joint_minus_product_log_loss"], -0.3)
        self.assertAlmostEqual(paired["joint_minus_product_brier"], -0.3)

    def test_mutated_native_trace_is_rejected_by_independent_reference(self):
        import copy

        sources, _, model = h.old_artifacts()
        source = next(s for s in sources if s["split"] == "TRAIN" and s["ply"] == 4)
        reference = h.reference_run(source, model, 0.8, 0, "JOINT_H2")
        mutated = copy.deepcopy(reference)
        mutated["final_u"] = 0.0 if reference["final_u"] == 1.0 else 1.0
        with self.assertRaises(AssertionError):
            h.same(mutated, reference)

    def test_old_training_root_labels_are_held_out_from_runtime_payload(self):
        sources, _, model = h.old_artifacts()
        source = next(s for s in sources if s["split"] == "TRAIN")
        request = h.run_request(source, {"model": model, "rho": 0.8}, 3)
        self.assertEqual(
            set(request),
            {
                "id",
                "history",
                "cue_seed",
                "policy_seed",
                "mode",
                "additional",
                "repeats",
                "warmup",
                "trace",
                "model",
                "rho",
            },
        )
        self.assertEqual(request["history"], source["history"])


if __name__ == "__main__":
    unittest.main()
