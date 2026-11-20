"""Independent fixed examples for actual action utility and loss comparisons."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_connect4_utility import score_sign, join_root


class Connect4DecisionUtilityTests(unittest.TestCase):
    def setUp(self):
        self.scores = [3, -2, 0, -1, 1, 2, -1]
        self.exact = {
            "sequence":"4433226223",
            "position_id":"C4_TEST",
            "scores":self.scores,
            "status":"EXACT_EXTERNAL_SOLVER",
            "best_score":3,
        }
        self.result = {
            "schema":"quartz_connect4_resident_mcts_v1",
            "history":"4433226223",
            "ply":10,
            "side_to_move":1,
            "board_bottom_up":[0]*42,
            "root_hash":"abc",
            "legal_columns_0based":[0,1,2,3,4,5,6],
            "eval_mode":"uniform_no_neural_network",
            "fresh16":{"resident_match":True,"best_action":2,"evaluator_calls":3,"root_visits":16},
            "steps":[
                {"budget":8,"root_visits":8,"best_action":1,"root_hash":"abc","evaluator_calls":2,"additional_evaluator_calls":1,"step_wall_us":10},
                {"budget":16,"root_visits":16,"best_action":2,"root_hash":"abc","evaluator_calls":3,"additional_evaluator_calls":1,"step_wall_us":20},
                {"budget":32,"root_visits":32,"best_action":4,"root_hash":"abc","evaluator_calls":5,"additional_evaluator_calls":2,"step_wall_us":30},
                {"budget":64,"root_visits":64,"best_action":0,"root_hash":"abc","evaluator_calls":7,"additional_evaluator_calls":2,"step_wall_us":40},
            ]
        }

    def test_sign_utility_and_ordinality(self):
        self.assertEqual([score_sign(x) for x in self.scores[:3]], [1,-1,0])
        with self.assertRaises(ValueError):
            score_sign(-1000)

    def test_join_reports_exact_signed_gain_and_regret(self):
        row=join_root(self.exact,self.result,expected_board=[0]*42)
        self.assertEqual(row["action_utility_by_budget"],[-1,0,1,1])
        self.assertEqual(row["utility_delta_by_transition"],[1,1,0])
        self.assertEqual(row["game_regret_by_budget"],[2,1,0,0])
        self.assertEqual(row["oracle_optimal_WDL"],1)
        self.assertEqual(row["extra_eval_calls"],[1,2,2])

    def test_rejects_selecting_unavailable_column(self):
        altered=dict(self.result)
        altered["steps"]=[dict(step) for step in self.result["steps"]]
        altered["steps"][1]["best_action"]=7
        with self.assertRaisesRegex(ValueError,"illegal"):
            join_root(self.exact,altered,expected_board=[0]*42)

    def test_rejects_fake_fresh16_or_wrong_root(self):
        altered=dict(self.result)
        altered["fresh16"]={"resident_match":False}
        with self.assertRaisesRegex(ValueError,"fresh16"):
            join_root(self.exact,altered,expected_board=[0]*42)
        altered=dict(self.result)
        altered["steps"]=[dict(x) for x in self.result["steps"]]
        altered["steps"][2]["root_hash"]="other"
        with self.assertRaisesRegex(ValueError,"root"):
            join_root(self.exact,altered,expected_board=[0]*42)


if __name__=="__main__":
    unittest.main()
