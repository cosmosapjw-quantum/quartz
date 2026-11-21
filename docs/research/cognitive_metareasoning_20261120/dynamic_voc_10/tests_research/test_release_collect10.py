"""External exact signed-WDL join after Rust full search, no illegal defaults."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collect_release10 import labels_from_full

class ReleaseCollectorTests(unittest.TestCase):
    def test_explicit_negative_signed_gain_and_joint_horizon(self):
        raw={'root_hash':'root','history':'444444','side_to_move':1,
             'legal_columns_0based':[0,1,2,4,5,6],
             'root_initial_evaluator_calls':1,
             'steps':[
                 {'budget':8,'root_visits':8,'best_action':0,
                  'visit_policy':[[0,1.0]],'evaluator_calls':4,'step_wall_us':300},
                 {'budget':16,'root_visits':16,'best_action':1,
                  'visit_policy':[[0,1.0]],'evaluator_calls':6,'step_wall_us':400},
                 {'budget':32,'root_visits':32,'best_action':4,
                  'visit_policy':[[0,1.0]],'evaluator_calls':8,'step_wall_us':500},
                 {'budget':64,'root_visits':64,'best_action':5,
                  'visit_policy':[[0,1.0]],'evaluator_calls':10,'step_wall_us':600}]}
        ref=labels_from_full(raw,[2,-2,0,-1000,1,0,-1])
        self.assertEqual(ref['utility'],[1,-1,1,0])
        self.assertEqual(ref['gain_one'],[-2,2,-1])
        self.assertEqual(ref['gain_two'],[0,1])
        self.assertEqual(ref['cost_one_ms'],[.4,.5,.6])
        self.assertEqual(ref['cost_two_ms'],[.9,1.1])
    def test_illegal_selected_action_rejected(self):
        raw={'steps':[{'budget':8,'best_action':3}]}
        with self.assertRaises(ValueError):labels_from_full(raw,[-1000]*7)

if __name__=='__main__':unittest.main()
