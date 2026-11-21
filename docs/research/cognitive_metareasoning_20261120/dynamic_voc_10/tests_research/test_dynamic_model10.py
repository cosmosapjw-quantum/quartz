"""One-step vs two-step gain and strictly positive release-cost surrogate."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dynamic_model10 import (
    train_ridge_with_intercept, fit_positive_cost, predict_option,
    simulate_bounded_policy, fit_dynamic_from_train,
)

class DynamicTenModelTests(unittest.TestCase):
    def test_simple_ridge_preserves_unpenalized_intercept(self):
        X=np.array([[1.,0.],[1.,1.],[1.,2.],[1.,3.]])
        y=np.full(4,2.0)
        w=train_ridge_with_intercept(X,y,alpha=1.0)
        self.assertAlmostEqual(w[0],2.0)
        self.assertAlmostEqual(w[1],0.0)

    def test_positive_cost_prediction_remains_finite(self):
        X=np.array([[1.,0.],[1.,1.],[1.,2.],[1.,3.]])
        y=np.array([0.1,0.3,0.2,0.5])
        coefs,scale=fit_positive_cost(X,y,alpha=1.0)
        p=np.exp(np.clip(X@coefs,-20,10))*scale
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue((p>0).all())
        self.assertAlmostEqual(np.mean(p),np.mean(y))

    def test_actual_joint_option_commits_across_negative_one_step(self):
        model={
           'schema':'quartz_cog02_dynamic_two_step_v1',
           'features':'root_visits_v1',
           'one_step_gain_coefficients':[[0.0]*9 for _ in range(3)],
           'two_step_gain_coefficients':[[0.0]*9 for _ in range(2)],
           'one_step_cost_log_coefficients':[[0.0]*9 for _ in range(3)],
           'two_step_cost_log_coefficients':[[0.0]*9 for _ in range(2)],
           'one_step_cost_scale':[.1]*3,
           'two_step_cost_scale':[.2]*2,
           'lambda_wdl_per_ms':.5
        }
        model['two_step_gain_coefficients'][0][0]=1.0
        model['one_step_gain_coefficients'][1][0]=10.0
        model['one_step_gain_coefficients'][2][0]=-1.0
        x=[1.0,.125,.5,.6,.2,1.0,.6,.3,1.0]
        result=predict_option(model,0,x,variant='dynamic')
        self.assertEqual(result['action'],'commit_two')
        self.assertLess(result['net_single'],0.0)
        observed={
            'features':[x]*3,
            'utilities':[0,-1,1,-1],
            'search_stage_wall_ms':[.1,.1,.1,.1],
        }
        out=simulate_bounded_policy(model,observed,variant='dynamic')
        self.assertEqual(out['visits'],32)
        self.assertEqual(out['skipped_stages'],[16])
        self.assertEqual(out['utility'],1)

    def test_fit_declines_missing_holdout_separation(self):
        with self.assertRaises(ValueError):
            fit_dynamic_from_train([],alpha=1.0)

if __name__=='__main__':unittest.main()
