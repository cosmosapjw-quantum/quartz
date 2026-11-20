"""Contract tests for real resident-root paired cost accounting."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resident_fork_cpu import validate_pair, summarize_rows


def mk(visits, calls, additional, move, root="abc001", policy=None):
    return {
        "root_visits": visits, "evaluator_calls": calls,
        "additional_evaluator_calls": additional,
        "additional_root_visits": 8 if visits in (8, 16) else 0,
        "root_hash": root, "best_move": move,
        "policy": [[move, 1.0]] if policy is None else policy,
    }


class ResidentContractTests(unittest.TestCase):
    def setUp(self):
        self.position = {
            "id": "P0003", "board": [0] * 49,
            "reference_best": 3, "oracle_best": 3,
        }
        self.first = mk(8, 9, 8, 2)
        self.second = mk(16, 17, 8, 3)
        self.fresh = mk(16, 17, 16, 3)
        self.fresh["additional_root_visits"] = 16  # Fresh engine started at zero

    def evaluate(self):
        return validate_pair(self.position,self.first,self.second,self.fresh,
                             python_calls=(9,8,17), wall_ms=(2.0,3.0,4.0),
                             checkpoint="C01_gen_5",seed=17)

    def test_visits_calls_and_reference_gain(self):
        row=self.evaluate()
        self.assertEqual(row["delta_ref"], 1)
        self.assertEqual(row["delta_oracle"], 1)
        self.assertEqual(row["delta_eval_calls"], 8)
        self.assertEqual(row["root_visits"], [8,16])
        self.assertTrue(row["fresh16_match"])
        self.assertEqual(row["NN_call_identity"], "exact")

    def test_reject_root_change(self):
        b=copy.deepcopy(self.second);b["root_hash"]="different"
        with self.assertRaisesRegex(ValueError,"root"):
            validate_pair(self.position,self.first,b,self.fresh,
                python_calls=(9,8,17),wall_ms=(2,3,4),checkpoint="C01",seed=1)

    def test_reject_evaluator_count_mismatch(self):
        with self.assertRaisesRegex(ValueError,"call"):
            validate_pair(self.position,self.first,self.second,self.fresh,
                python_calls=(9,7,17),wall_ms=(2,3,4),checkpoint="C01",seed=1)

    def test_reject_visits_not_exact(self):
        b=copy.deepcopy(self.second);b["root_visits"]=14
        with self.assertRaisesRegex(ValueError,"visit"):
            validate_pair(self.position,self.first,b,self.fresh,
                python_calls=(9,8,17),wall_ms=(2,3,4),checkpoint="C01",seed=1)

    def test_reject_illegal_action(self):
        p=copy.deepcopy(self.position);p["board"][2]=1
        with self.assertRaisesRegex(ValueError,"illegal"):
            validate_pair(p,self.first,self.second,self.fresh,
                python_calls=(9,8,17),wall_ms=(2,3,4),checkpoint="C01",seed=1)

    def test_fresh_divergence_is_preserved_not_forced_equal(self):
        b=copy.deepcopy(self.fresh);b["best_move"]=4;b["policy"]=[[4,1.0]]
        row=validate_pair(self.position,self.first,self.second,b,
                python_calls=(9,8,17),wall_ms=(2,3,4),checkpoint="C01",seed=1)
        self.assertFalse(row["fresh16_match"])

    def test_summary_keeps_negative_signed_outcomes(self):
        r=self.evaluate()
        r2=copy.deepcopy(r);r2["delta_ref"]=-1;r2["delta_oracle"]=-1;r2["fresh16_match"]=False
        ans=summarize_rows([r,r2])
        self.assertEqual(ans["signed_ref"],{"-1":1,"0":0,"1":1})
        self.assertEqual(ans["fresh16_mismatches"],1)
        self.assertEqual(ans["delta_eval_calls_total"],16)


if __name__=="__main__":
    unittest.main()
