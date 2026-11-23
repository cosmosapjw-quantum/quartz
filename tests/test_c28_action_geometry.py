"""C28 native contract and independent finite-world scientific checks.

Run from the repository root:
  C28_BINARY=/absolute/path/quartz_c28_action_geometry \
    python3 -m unittest discover -s tests -p 'test_c28_action_geometry.py' -v

Missing C28_BINARY is an ERROR/NOT_RUN, never a successful skip. Tests use only
the stdlib, actual native executions, and the immutable Fraction world oracle.
"""
from __future__ import annotations

import copy
from fractions import Fraction as F
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/research/quartz_c28_action_geometry_20261123/coding/c28_campaign.py"
SPEC = importlib.util.spec_from_file_location("c28_campaign_under_test", SCRIPT)
c28 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = c28
SPEC.loader.exec_module(c28)


class OracleAndGameControlTests(unittest.TestCase):
    def test_exact_constructive_fixtures(self):
        ref = c28.oracle()
        for law, value, choice in ((ref.family(), F(3, 5), 0),
                                   (ref.family(sign=1), F(109, 200), 0),
                                   (ref.family(r=F(0)), F(1, 2), None)):
            _, actual, selected = c28.exact_synthetic(law, F(0), F(1, 2))
            self.assertEqual((actual, selected), (value, choice))
        self.assertEqual(c28.exact_synthetic(ref.family(c=F(2, 25)), F(0), F(1, 2))[1], F(53, 100))
        self.assertIsNone(c28.exact_synthetic(ref.family(c=F(2, 25), sign=1), F(0), F(1, 2))[2])

    def test_independent_tictactoe_controls_and_root_sign(self):
        board, turn = c28.replay([])
        self.assertEqual(c28.minimax(board, turn, turn)[0], 0)
        # Both X and O have a nonterminal root with their own immediate win.
        for history, winning_move in (([0, 3, 1, 4], 2), ([0, 3, 1, 4, 8], 5)):
            board, root = c28.replay(history)
            child = c28.play(board, winning_move, root)
            self.assertEqual(c28.minimax(child, -root, root), (1, 1))
            self.assertEqual(c28.minimax(child, -root, -root), (-1, 1))
        with self.assertRaises(AssertionError):
            c28.replay([0, 3, 1, 4, 2, 7])

    def test_cost_aggregation_divides_iterations_once(self):
        request = {"mode": "synthetic", "id": "units", "repeats": 3, "iterations": 8}
        response = {"timings": [
            {"repeat": 0, "direct_ns": 800, "lifted_ns": 640, "iterations": 8, "order": "direct_first"},
            {"repeat": 1, "direct_ns": 880, "lifted_ns": 720, "iterations": 8, "order": "lifted_first"},
            {"repeat": 2, "direct_ns": 720, "lifted_ns": 560, "iterations": 8, "order": "direct_first"}]}
        row = c28.timing_row(request, response)
        self.assertEqual(row["direct_median_ns"], 100)
        self.assertEqual(row["lifted_median_ns"], 80)
        summary = c28.aggregate([row])
        self.assertEqual(summary["paired_mean_direct_minus_lifted_ns"], 20)
        self.assertEqual(summary["ratio_of_paired_means_lifted_over_direct"], .8)


class NativeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        binary = os.environ.get("C28_BINARY")
        if not binary:
            raise RuntimeError("NOT_RUN: set C28_BINARY to an actual built C28 executable")
        cls.client = c28.NativeClient(binary)
        cls.addClassCleanup(cls.client.close)
        cls.contract, cls.sources, cls.model = c28.frozen_inputs()

    def valid_synthetic(self, model=None, m=F(0), b=F(1, 2), horizon=2, **kwargs):
        model = model if model is not None else c28.oracle().family()
        request = c28.synthetic_request(model, m, b, horizon=horizon, **kwargs)
        response = self.client.request(request)
        self.assertEqual(response.get("id"), request["id"])
        evidence = c28.check_synthetic(response, model, m, b, horizon)
        c28.check_timings(response, request.get("repeats", 0), request.get("iterations", 1))
        return response, evidence

    def reject(self, request):
        response = self.client.request(request)
        self.assertIn("error", response)
        self.assertIsInstance(response["error"], str)
        self.assertTrue(response["error"])
        self.assertNotIn("direct", response)

    def test_signed_common_product_and_cost_crossing(self):
        ref = c28.oracle()
        fixtures = ((ref.family(), .6, 0, [.1, .005, .005]),
                    (ref.family(sign=1), .545, 0, [.045, -.05, -.05]),
                    (ref.family(r=F(0)), .5, None, None),
                    (ref.family(c=F(2, 25)), .53, 0, None),
                    (ref.family(c=F(2, 25), sign=1), .5, None, None))
        for law, value, choice, scores in fixtures:
            with self.subTest(value=value, loadings=law.r, costs=law.cost):
                response, _ = self.valid_synthetic(law)
                for method in ("direct", "lifted"):
                    c28.close_float(response[method]["value"], value, "fixture value")
                    self.assertEqual(response[method]["choice"], choice)
                    if scores:
                        for actual, expected in zip(response[method]["scores"], scores):
                            c28.close_float(actual, expected, "fixture score")

    def test_exact_and_near_action_boundaries(self):
        ref = c28.oracle()
        for sign in (-1, 1):
            for delta in (F(-1, 10**8), F(-1, 10**13), F(0), F(1, 10**13), F(1, 10**8)):
                with self.subTest(sign=sign, delta=delta):
                    m = sign * F(19, 24) + delta
                    response, evidence = self.valid_synthetic(ref.family(), m)
                    self.assertEqual(response["direct"]["choice"], evidence["epsilon_choice"])
                    if delta == 0:
                        self.assertEqual(response["direct"]["choice"], 0)
                        c28.close_float(response["direct"]["value"], F(1, 2) + F(13, 120), "boundary gain")

    def test_empty_unary_zero_mass_point_mass_and_endpoints(self):
        for ident, law, m, b, horizon in c28.synthetic_cases(random_count=0):
            if ident.startswith("edge-"):
                with self.subTest(case=ident):
                    self.valid_synthetic(law, m, b, horizon, ident=ident)

    def test_fixed_seed_rational_models_against_complete_worlds(self):
        for ident, law, m, b, horizon in c28.synthetic_cases():
            if ident.startswith("rational-"):
                with self.subTest(case=ident):
                    self.valid_synthetic(law, m, b, horizon, ident=ident)

    def test_epsilon_choice_is_separate_from_raw_optimal_value(self):
        ref = c28.oracle()
        # A strictly positive raw gain below the declared STOP threshold.
        gain = F(1, 2 * 10**12)
        law = ref.Model(((F(0), F(0), F(1)),), (F(0),), (F(1, 2) - gain,))
        response, _ = self.valid_synthetic(law, horizon=1)
        for method in ("direct", "lifted"):
            self.assertIsNone(response[method]["choice"])
            self.assertGreater(response[method]["value"], .5)
            c28.close_float(response[method]["value"], F(1, 2) + gain, "raw value despite STOP")

    def test_probability_normalization_inside_admitted_tolerance(self):
        request = {"mode": "synthetic", "id": "roundoff-normalize",
                   "probabilities": [[.5, .5 + 5e-13, 0]], "loadings": [0], "costs": [0],
                   "m": 0, "b": 0, "horizon": 1}
        response = self.client.request(request)
        self.assertNotIn("error", response)
        expected = F(str(request["probabilities"][0][1])) / sum(map(lambda x: F(str(x)), request["probabilities"][0])) / 2
        for method in ("direct", "lifted"):
            c28.close_float(response[method]["value"], expected, "normalized probability expectation")

    def test_invalid_synthetic_domain_and_resource_bounds(self):
        base = c28.synthetic_request(c28.oracle().family())
        mutations = [
            {"costs": [-.01, .3, .3]}, {"loadings": [1, .8, -.8]},
            {"loadings": [-1, .8, -.8]}, {"loadings": [.8]}, {"costs": []},
            {"probabilities": [[.4, .5, 0], [.5, 0, .5], [.5, 0, .5]]},
            {"probabilities": [[-.01, 1.01, 0], [.5, 0, .5], [.5, 0, .5]]},
            {"probabilities": [[.5, .5], [.5, 0, .5], [.5, 0, .5]]},
            {"m": 1.001}, {"m": -1.001}, {"b": -.01}, {"b": 1.01},
            {"horizon": 0}, {"horizon": 3}, {"horizon": 1.5}, {"iterations": 0},
            {"iterations": 10001}, {"repeats": 1001}, {"warmup": 1001}, {"repeats": -1},
            {"oracle_utility": [0, .5, 1]}, {"unknown_field": True},
            {"probabilities": [[1, 0, 0]] * 65, "loadings": [0] * 65, "costs": [0] * 65},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.reject({**base, **mutation})

    def test_nonfinite_json_and_invalid_line_stream_recovery(self):
        for line in ("{", "[]", "null", '{"mode":"missing"}',
                     c28.encode(c28.synthetic_request(c28.oracle().family())).replace('"m":0.0', '"m":NaN'),
                     c28.encode(c28.synthetic_request(c28.oracle().family())).replace('"m":0.0', '"m":1e309')):
            with self.subTest(line=line[:40]):
                response = self.client.request_raw(line)
                self.assertIn("error", response)
                self.valid_synthetic()  # Same resident stream after every invalid request.

    def test_synthetic_timing_shape_and_alternating_order(self):
        self.valid_synthetic(repeats=3, warmup=1, iterations=4)

    def test_known_terminal_winning_children_for_both_root_players(self):
        for history, winning_move in (([0, 3, 1, 4], 2), ([0, 3, 1, 4, 8], 5)):
            request = c28.game_request({"id": "immediate-win", "history": history, "cue_seed": 23}, self.model)
            response = self.client.request(request)
            c28.check_game(response, request)
            self.assertEqual(response["direct"]["initial_move"], winning_move)
            self.assertEqual(response["direct"]["utility"], 1)
            self.assertEqual(response["direct"]["query_moves"], [])
            self.assertEqual(response["direct"]["initial_nodes"], 0)

    def test_frozen_public_game_traces_and_independent_costs(self):
        # A deterministic smoke sample; the CLI verify game suite uses all 300.
        indices = (0, 1, 49, 100, 149, 200, 249, 299)
        for index in indices:
            with self.subTest(source_index=index):
                request = c28.game_request(self.sources[index], self.model)
                response = self.client.request(request)
                self.assertEqual(response.get("id"), request["id"])
                c28.check_game(response, request)
                c28.check_timings(response, 0, 1)

    def test_game_timing_shape_and_trace_free_summary(self):
        request = c28.game_request(self.sources[0], self.model, trace=False, repeats=3, warmup=1)
        response = self.client.request(request)
        c28.check_game(response, request, check_scores=False)
        c28.check_timings(response, 3, 1)
        self.assertEqual(response["direct"]["decisions"], [])
        self.assertEqual(response["lifted"]["decisions"], [])

    def test_omitted_active_decision_trace_is_rejected(self):
        request = c28.game_request(self.sources[0], self.model)
        response = self.client.request(request)
        c28.check_game(response, request)
        self.assertEqual(len(response["direct"]["decisions"]), 2)
        candidates = {row["move"]: row for row in c28.game_candidates(
            request["history"], request["cue_seed"], request["model"])}
        cue_nodes = sum(row["cue_nodes"] for row in candidates.values())
        # Construct otherwise coherent prematurely truncated results: omit the
        # initial H2 decision, then separately omit the required H1 decision.
        for keep in (0, 1):
            malformed = copy.deepcopy(response)
            for method in ("direct", "lifted"):
                row = malformed[method]
                row["final_m"] = row["decisions"][keep]["m"]
                row["decisions"] = row["decisions"][:keep]
                row["query_moves"] = row["query_moves"][:keep]
                row["additional_queries"] = keep
                row["additional_nodes"] = sum(candidates[move]["solve_nodes"] for move in row["query_moves"])
                row["charged_nodes"] = cue_nodes + row["initial_nodes"] + row["additional_nodes"]
                row["final_move"], row["utility"] = row["initial_move"], row["initial_utility"]
                for move in row["query_moves"]:
                    if candidates[move]["utility"] > row["utility"]:
                        row["final_move"], row["utility"] = move, float(candidates[move]["utility"])
                self.assertLess(row["utility"], 1)
            with self.subTest(retained_decisions=keep):
                with self.assertRaisesRegex(AssertionError, "Incomplete decision trace"):
                    c28.check_game(malformed, request)

    def test_invalid_histories_terminal_roots_and_oracle_fields(self):
        base = c28.game_request(self.sources[0], self.model)
        histories = ([0, 0], [9], [-1], [0, 3, 1, 4, 2], [0, 3, 1, 4, 2, 7],
                     [0, 1, 2, 4, 3, 5, 7, 6, 8], [False])
        for history in histories:
            with self.subTest(history=history):
                self.reject({**base, "history": history})
        for field in ("oracle", "oracle_labels", "utilities", "outcomes", "observations", "source_hash"):
            with self.subTest(field=field):
                self.reject({**base, field: [1, 0, .5]})
        for mutation in ({"rho": -.01}, {"rho": 1}, {"cue_seed": -1}, {"cue_seed": 2**64},
                         {"repeats": 1001}, {"warmup": 1001}):
            self.reject({**base, **mutation})
        bad_model = copy.deepcopy(self.model)
        bad_model["oracle_labels"] = []
        self.reject({**base, "model": bad_model})
        for field, value in (("global_cost", -1), ("global_probs", [0, .5, .5]),
                             ("lambda_nodes", .1), ("tau", .1)):
            bad_model = {**self.model, field: value}
            self.reject({**base, "model": bad_model})


if __name__ == "__main__":
    unittest.main()
