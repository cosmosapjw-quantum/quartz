"""C26 independent finite oracle, public boundary and statistical checks.

These are verification fixtures, never the TRAIN/CAL/TEST policy cohorts.
"""

import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = (
    ROOT
    / "docs/research/quartz_general_game_metareasoning_20261123/coding/c26_campaign.py"
)
spec = importlib.util.spec_from_file_location("c26_campaign", MODULE)
c26 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c26)


def flat_model():
    return {
        "schema": "c26-categorical-v1",
        "feature_probs": {},
        "remaining_probs": {},
        "global_probs": [0.2, 0.3, 0.5],
        "remaining_costs": {},
        "global_cost": 10.0,
        "lambda_nodes": 2**-10,
        "tau": 2**-5,
    }


class PopulationOracleTests(unittest.TestCase):
    def test_complete_reachable_population_stops_at_terminals(self):
        states = c26.enumerate_reachable()
        self.assertEqual(len(states), 5478)
        self.assertEqual(sum(not c26.terminal(b) for b in states), 4520)
        self.assertTrue(all(c26.replay(h) == b for b, h in states.items()))

    def test_empty_board_exact_value_and_full_traversal_count(self):
        self.assertEqual(c26.oracle((0,) * 9, 1), (0, 549946))

    def test_terminal_root_frame_signs_and_one_node(self):
        x_win = c26.replay([0, 3, 1, 4, 2])
        o_win = c26.replay([0, 3, 1, 4, 8, 5])
        draw = c26.replay([0, 1, 2, 6, 4, 8, 3, 5, 7])
        self.assertEqual(c26.oracle(x_win, 1), (1, 1))
        self.assertEqual(c26.oracle(x_win, -1), (-1, 1))
        self.assertEqual(c26.oracle(o_win, -1), (1, 1))
        self.assertEqual(c26.oracle(o_win, 1), (-1, 1))
        self.assertEqual(c26.oracle(draw, 1), (0, 1))
        self.assertEqual(c26.legal_moves(x_win), [])

    def test_replay_rejects_illegal_and_terminal_continuation(self):
        for history in ([0, 0], [-1], [9], [True], [0, 3, 1, 4, 2, 6]):
            with self.subTest(history=history):
                with self.assertRaises(ValueError):
                    c26.replay(history)

    def test_source_selection_is_canonical_disjoint_and_blinded(self):
        public, offline, population = c26.build_sources()
        self.assertEqual(population["reachable_states"], 5478)
        self.assertEqual(
            population["canonical_nonterminal_population"],
            {"4": 108, "5": 153, "6": 183},
        )
        self.assertEqual(len(public), 300)
        self.assertEqual(len(offline), 300)
        self.assertEqual(len({s["id"] for s in public}), 300)
        for split, n in [("TRAIN", 90), ("CAL", 60), ("TEST", 150)]:
            self.assertEqual(sum(s["split"] == split for s in public), n)
        self.assertFalse(
            any("u" in s or "oracle" in s or "candidates" in s for s in public)
        )
        for s, truth in zip(public, offline):
            board = c26.replay(s["history"])
            self.assertEqual(tuple(truth["board"]), board)
            canonical, _ = c26.canonicalize(board, tuple(s["history"]))
            self.assertEqual(canonical, board)
            self.assertEqual(s["id"], truth["id"])

    def test_d4_rotation_has_same_canonical_representative(self):
        history = [0, 4, 1, 8]
        clockwise = [2, 5, 8, 1, 4, 7, 0, 3, 6]
        rotated = [clockwise[x] for x in history]
        a, _ = c26.canonicalize(c26.replay(history), tuple(history))
        b, hb = c26.canonicalize(c26.replay(rotated), tuple(rotated))
        self.assertEqual(a, b)
        self.assertEqual(c26.replay(hb), a)

    def test_splitmix64_literal_vectors(self):
        rng = c26.SplitMix64(0)
        self.assertEqual(
            [rng.next_u64() for _ in range(3)],
            [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F],
        )
        with self.assertRaises(ValueError):
            rng.index(0)

    def test_terminal_cue_is_paid_once(self):
        candidates = c26.cue_candidates(c26.replay([0, 3, 1, 4]), 123)
        win = next(x for x in candidates if x["move"] == 2)
        self.assertEqual(
            (win["counts"], win["cue_nodes"], win["solve_nodes"]), ([0, 0, 4], 1, 1)
        )
        self.assertEqual(win["remaining"], 0)
        self.assertEqual(win["u"], 1.0)


class ModelMechanismTests(unittest.TestCase):
    def test_train_only_smoothing_and_fallback_costs(self):
        rows = [
            {
                "id": "fixture",
                "source_hash": "fixture",
                "candidates": [
                    {
                        "move": 0,
                        "terminal_known": False,
                        "remaining": 3,
                        "counts": [0, 2, 2],
                        "u": 0.5,
                        "solve_nodes": 9,
                    },
                    {
                        "move": 1,
                        "terminal_known": False,
                        "remaining": 3,
                        "counts": [0, 2, 2],
                        "u": 0.5,
                        "solve_nodes": 11,
                    },
                    {
                        "move": 2,
                        "terminal_known": True,
                        "remaining": 0,
                        "counts": [0, 0, 4],
                        "u": 1,
                        "solve_nodes": 1,
                    },
                ],
            }
        ]
        model, provenance = c26.fit_model(rows)
        self.assertEqual(model["feature_probs"]["3:0:2:2"], [0.2, 0.6, 0.2])
        self.assertEqual(model["remaining_costs"]["3"], 10)
        self.assertEqual(
            c26.model_prediction(model, 3, [4, 0, 0]), ([0.2, 0.6, 0.2], 10)
        )
        self.assertEqual(
            c26.model_prediction(model, 7, [4, 0, 0]), ([0.2, 0.6, 0.2], 10)
        )
        self.assertEqual(provenance["nonterminal_candidates"], 2)
        self.assertEqual(provenance["global"]["cost_sum"], 20)
        with self.assertRaises(ValueError):
            c26.fit_model([])

    def test_reservation_index_piecewise_boundaries(self):
        for probs, cost, expected in [
            ([0.2, 0.3, 0.5], 0.125, 0.75),
            ([0.2, 0.3, 0.5], 0.25, 0.5),
            ([0.2, 0.3, 0.5], 0.45, 0.25),
            ([0.2, 0.3, 0.5], 0.65, 0.0),
            ([0.2, 0.3, 0.5], 0.75, -0.1),
            ([1.0, 0.0, 0.0], 0.2, -0.2),
            ([0.0, 1.0, 0.0], 0.25, 0.25),
        ]:
            with self.subTest(probs=probs, cost=cost):
                z = c26.reservation_index(probs, cost)
                self.assertAlmostEqual(z, expected, places=13)
                excess = sum(p * max(u - z, 0) for p, u in zip(probs, [0, 0.5, 1]))
                self.assertAlmostEqual(excess, cost, places=13)

    def test_gibbs_stability_shift_and_joint_scale(self):
        scores = [0.0, -0.2, 0.1]
        p = c26.choice_probabilities(scores, 0.03125)
        shifted = c26.choice_probabilities([x + 1000 for x in scores], 0.03125)
        scaled = c26.choice_probabilities([3 * x for x in scores], 3 * 0.03125)
        for a, b, c in zip(p, shifted, scaled):
            self.assertAlmostEqual(a, b, places=11)
            self.assertAlmostEqual(a, c, places=14)
        extreme = c26.choice_probabilities([0, -1e6, 1e6], 0.03125)
        self.assertEqual(extreme, [0.0, 0.0, 1.0])
        self.assertTrue(all(math.isfinite(x) for x in extreme))

    def test_gibbs_actual_sampling_frequency_fixed_tolerance(self):
        probabilities = c26.choice_probabilities([0, -0.02, 0.04], 0.03125)
        rng = c26.SplitMix64(2600923)
        counts = [0, 0, 0]
        for _ in range(100000):
            counts[c26.sample_probability(probabilities, rng.uniform())] += 1
        # Fixed before sampling; <=0.007 absolute error in every category.
        self.assertTrue(
            all(abs(n / 100000 - p) <= 0.007 for n, p in zip(counts, probabilities))
        )

    def test_public_request_cannot_carry_oracle_values(self):
        source = {
            "id": "fixture",
            "history": [0, 4, 1, 8],
            "cue_seed": 17,
            "u": 999,
            "oracle": "must not pass through",
        }
        request = c26.public_request(source, model=flat_model())
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
            },
        )

    def test_known_terminal_win_skips_initial_query_and_stops_rational_policies(self):
        for policy in [
            "INITIAL_STOP",
            "RANDOM_ORDER",
            "GREEDY_VOC",
            "PANDORA",
            "GIBBS_VOC",
        ]:
            run = c26.expected_run([0, 3, 1, 4], 123, 0, 9, policy, flat_model())
            self.assertEqual(run["selected_move"], 2)
            self.assertEqual(run["initial_solve_move"], None)
            self.assertEqual(run["initial_nodes"], 0)
            self.assertEqual(run["additional_queries"], 0)
            self.assertEqual(run["final_u"], 1)

    def test_parent_bootstrap_keeps_source_as_unit(self):
        result = c26.paired_bootstrap({4: [0.25, 0.25], 5: [0.25], 6: [0.25, 0.25]})
        self.assertEqual(result["parents"], 5)
        self.assertEqual(result["draws"], 2000)
        self.assertEqual(result["mean"], 0.25)
        self.assertEqual(result["ci95"], [0.25, 0.25])


class NativeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = Path(
            os.environ.get(
                "C26_BINARY", str(ROOT / "target/release/quartz_c26_game_metareasoning")
            )
        )
        if not cls.binary.is_file():
            raise unittest.SkipTest("native release binary not built yet")
        cls.client = c26.NativeClient(cls.binary)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "client"):
            cls.client.close()

    def test_invalid_json_history_and_model_continue_processing(self):
        source = {"id": "valid-fixture", "history": [0, 4, 1, 8], "cue_seed": 123}
        good = c26.public_request(source, mode="survey", model=None)
        invalid = ["not-json", "[]", "{}"]
        for key, value in [
            ("id", ""),
            ("id", "x" * 129),
            ("history", [0, 0]),
            ("history", [-1]),
            ("history", [9]),
            ("history", [True]),
            ("history", [0, 3, 1, 4, 2]),
            ("history", [0, 3, 1, 4, 2, 6]),
            ("history", [0, 1, 2, 6, 4, 8, 3, 5, 7]),
            ("additional", 0),
            ("additional", 10),
            ("repeats", 0),
            ("repeats", 16),
            ("warmup", 3),
            ("mode", "oracle"),
            ("cue_seed", -1),
            ("oracle_values", [1, 1, 1]),
        ]:
            bad = copy.deepcopy(good)
            bad[key] = value
            invalid.append(json.dumps(bad))
        run = c26.public_request(source, model=flat_model(), repeats=1, warmup=0)
        bad = copy.deepcopy(run)
        bad["model"] = None
        invalid.append(json.dumps(bad))
        for change in [
            {"unknown": True},
            {"global_probs": [0, 0.5, 0.5]},
            {"global_probs": [0.2, 0.3, 0.4]},
            {"global_cost": 0},
            {"global_cost": 1000001},
            {"lambda_nodes": 0.001},
            {"tau": 0},
            {"remaining_costs": {"9": 3}},
            {"feature_probs": {"3:1:1:1": [0.2, 0.3, 0.5]}},
        ]:
            bad = copy.deepcopy(run)
            bad["model"].update(change)
            invalid.append(json.dumps(bad))
        for text in invalid:
            with self.subTest(request=text[:120]):
                response, _ = self.client.send_raw(text)
                self.assertIn("error", response)
        response, _ = self.client.request(good)
        c26.validate_survey(source, response)

    def test_native_traces_all_policies_caps_and_players(self):
        histories = [
            [0, 4, 1, 8],
            [0, 4, 1, 8, 6],
            [0, 3, 1, 4],
            [0, 1, 2, 6, 4, 8, 3, 5],
        ]
        for i, history in enumerate(histories):
            source = {
                "id": "trace-fixture-" + str(i),
                "history": history,
                "cue_seed": 2600923 + i,
            }
            for cap in [1, 2, 9]:
                request = c26.public_request(
                    source,
                    model=flat_model(),
                    cap=cap,
                    policy_seed=1,
                    trace=True,
                    repeats=1,
                    warmup=0,
                )
                response, _ = self.client.request(request)
                c26.validate_response(
                    source, request, response, flat_model(), require_trace=True
                )

    def test_nine_repetitions_match_separate_trace(self):
        source = {"id": "timing-fixture", "history": [0, 4, 1, 8], "cue_seed": 2600923}
        request = c26.public_request(source, model=flat_model(), cap=9, policy_seed=3)
        response, transport = self.client.request(request)
        signature = c26.validate_response(source, request, response, flat_model())
        self.assertEqual(len(response["runs"]), 54)
        self.assertGreaterEqual(
            response["execution_elapsed_us"],
            response["warmup_elapsed_us"]
            + sum(r["elapsed_us"] for r in response["runs"]),
        )
        self.assertGreaterEqual(
            transport["roundtrip_us"], response["execution_elapsed_us"]
        )
        trace_request = c26.public_request(
            source, model=flat_model(), cap=9, policy_seed=3, trace=True
        )
        trace, _ = self.client.request(trace_request)
        trace_signature = c26.validate_response(
            source, trace_request, trace, flat_model(), require_trace=True
        )
        c26.assert_same(trace_signature, signature)

    def test_all_reachable_nonterminal_root_signs_and_node_counts(self):
        evidence = (
            ROOT
            / "docs/research/quartz_general_game_metareasoning_20261123/evidence/DATA_ALL_REACHABLE.json"
        )
        summary = c26.audit_all_reachable(self.client)
        self.assertEqual(summary["roots"], 4520)
        self.assertEqual(summary["root_players"], [-1, 1])
        self.assertEqual(summary["root_utilities"], [0.0, 0.5, 1.0])
        c26.write_json(evidence, summary)


class AggregationTests(unittest.TestCase):
    def test_median_then_seed_mean_then_equal_parent_contrast(self):
        records = []
        for parent, offset, ply in [("a", 0, 4), ("b", 100, 5)]:
            for seed in range(4):
                for policy in ["INITIAL_STOP", "PANDORA"]:
                    focus = policy == "PANDORA"
                    utility = 1.0 if focus and parent == "a" else 0.5
                    for repeat, elapsed in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 1000]):
                        elapsed += offset + seed * 10 + (3 if focus else 0)
                        records.append(
                            {
                                "id": parent,
                                "ply": ply,
                                "cap": 9,
                                "seed": seed,
                                "policy": policy,
                                "repeat": repeat,
                                "oracle_u": 1.0,
                                "initial_move": 0,
                                "initial_solve_move": 0,
                                "initial_u": 0.5,
                                "selected_move": 1 if utility == 1 else 0,
                                "final_u": utility,
                                "cue_nodes": 2,
                                "initial_nodes": 3,
                                "additional_nodes": 2 if focus else 0,
                                "charged_nodes": 7 if focus else 5,
                                "additional_queries": 1 if focus else 0,
                                "query_moves": [1] if focus else [],
                                "predicted_additional_nodes": 2 if focus else 0,
                                "stop_reason": "exhausted" if focus else "initial_stop",
                                "elapsed_us": elapsed,
                                "cue_us": elapsed / 4,
                                "initial_us": elapsed / 4,
                                "query_us": elapsed / 4,
                                "controller_us": elapsed / 4,
                            }
                        )
        seeds = c26.summarize_seed_runs(records)
        parents = c26.summarize_parents(seeds)
        lookup = {(r["id"], r["policy"]): r["metrics"] for r in parents}
        self.assertEqual(lookup[("a", "INITIAL_STOP")]["elapsed_us"], 20)
        self.assertEqual(lookup[("b", "INITIAL_STOP")]["elapsed_us"], 120)
        contrast = c26.comparison(parents, 9, "PANDORA", "INITIAL_STOP")
        self.assertEqual(contrast["metrics"]["elapsed_us"]["mean"], 3)
        self.assertEqual(contrast["metrics"]["utility"]["mean"], 0.25)
        self.assertEqual(contrast["metrics"]["net_nodes"]["mean"], 0.248046875)
        self.assertEqual(contrast["metrics"]["net_nodes"]["parents"], 2)
        with self.assertRaises(AssertionError):
            c26.summarize_seed_runs(records[:-1])
        with self.assertRaises(AssertionError):
            c26.summarize_parents(seeds[:-1])

    def test_multiclass_brier_diagnostics_use_exact_minimax_labels(self):
        sources = [
            {
                "id": "calibration-fixture",
                "ply": 4,
                "history": [0, 4, 1, 8],
                "cue_seed": 17,
            }
        ]
        report, rows = c26.calibration(sources, flat_model())
        literal_brier = {0.0: 0.98, 0.5: 0.78, 1.0: 0.38}
        for row in rows:
            self.assertAlmostEqual(row["brier"], literal_brier[row["u"]], places=14)
        self.assertEqual(report["pooled_candidates"]["candidates"], len(rows))
        self.assertAlmostEqual(
            report["pooled_candidates"]["brier"],
            sum(literal_brier[row["u"]] for row in rows) / len(rows),
        )


if __name__ == "__main__":
    unittest.main()
