"""Bounded current-root signed-gain controller and matched-cost replay contract."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_gain08 import (
    current_root_features, continuation_gain_labels, fit_stagewise_ridge,
    policy_stop_budget, classify_outcome, default_features_schema
)

class CostedVoc08Tests(unittest.TestCase):
    def setUp(self):
        self.root={"history":"1212", "side_to_move":1,"ply":4,
            "legal_columns_0based":[0,1,2,3,4,5,6],
            "root_initial_evaluator_calls":1,
            "steps":[
                {"budget":8,"root_visits":8,
                 "visit_policy":[[0,0.75],[1,0.25]],
                 "best_action":0,"evaluator_calls":5,"step_wall_us":500},
                {"budget":16,"root_visits":16,
                 "visit_policy":[[0,0.85],[1,0.15]],
                 "best_action":1,"evaluator_calls":10,"step_wall_us":850},
                {"budget":32,"root_visits":32,
                 "visit_policy":[[0,0.8],[1,0.2]],
                 "best_action":0,"evaluator_calls":18,"step_wall_us":1800},
                {"budget":64,"root_visits":64,
                 "visit_policy":[[0,0.95],[1,0.05]],
                 "best_action":0,"evaluator_calls":31,"step_wall_us":4000}
            ]}
        self.scores=[2,-5,-2,-1,-3,-2,-1]

    def test_features_are_current_only_and_bounded(self):
        x=current_root_features(self.root,0)
        self.assertEqual(len(x),9)
        self.assertEqual(x[0],1)
        self.assertEqual(x[1],0.125)
        self.assertAlmostEqual(x[3],0.75)
        self.assertAlmostEqual(x[4],0.5)
        self.assertAlmostEqual(x[6],5/9)
        self.assertEqual(x[8],1.0)
        with self.assertRaises(ValueError):
            current_root_features(dict(self.root,steps=[dict(self.root['steps'][0],
                visit_policy=[[0,float('nan')]])]),0)

    def test_loss_labels_are_signed_and_chained_only_after_search(self):
        row=continuation_gain_labels(self.root,self.scores)
        # 8->16: +1 to -1 => -2. 16->32 recovers => +2.
        self.assertEqual(row['utility'],[1,-1,1,1])
        self.assertEqual(row['gains'],[-2,2,0])
        self.assertEqual(row['next_wall_ms'],[0.85,1.8,4.0])

    def test_ridge_expected_gain_and_stopping(self):
        rows=[]
        for i in range(15):
            r={"features": [[1.0,0.125,0.1,0.7,0.5,1.0,0.6,0.1,1.0]]*3,
               "gains": [2 if i%2==0 else -2,0,0],
               "next_wall_ms":[1,2,3]}
            rows.append(r)
        fitted=fit_stagewise_ridge(rows,alpha=1.0)
        self.assertEqual(len(fitted['coefficients']),3)
        self.assertEqual(len(fitted['coefficients'][0]),9)
        self.assertEqual(fitted['forecast_cost_ms'],[1,2,3])
        stop_policy={**fitted,'schema':"quartz_cog02_signed_gain_surrogate_v1",
                     'features':'root_visits_v1','lambda_wdl_per_ms':1.0}
        self.assertEqual(policy_stop_budget(stop_policy,self.root),8)
        stop_policy['lambda_wdl_per_ms']=0.0
        self.assertIn(policy_stop_budget(stop_policy,self.root),(8,16,32,64))

    def test_invalid_model_and_exact_wdl_boundary(self):
        self.assertEqual(classify_outcome(-1),-1)
        self.assertEqual(classify_outcome(0),0)
        self.assertEqual(classify_outcome(12),1)
        with self.assertRaises(ValueError):
            classify_outcome(-1000)

class TimeCapContractTests(unittest.TestCase):
    def test_checkpoint_time_cap_is_not_exact_per_iteration_hard_cap(self):
        from calibrate_time_cap08 import selected_budget_and_cost
        row={'search_stage_wall_ms':[2.0,3.0,4.0,5.0]}
        self.assertEqual(selected_budget_and_cost(row,0.25),(8,2.0))
        self.assertEqual(selected_budget_and_cost(row,3.0),(16,5.0))
        self.assertEqual(selected_budget_and_cost(row,100.0),(64,14.0))

if __name__=='__main__':unittest.main()
