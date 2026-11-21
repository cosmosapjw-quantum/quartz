"""Freeze validation-only nonmyopic vs myopic randomized CPU match."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mix_plan11 import (
    select_validation_cost_bracket,
    deterministic_hash_uniform,
    balanced_group_assignments,
    apply_myopic_lambda,
)

class MixtureContractTests(unittest.TestCase):
    def test_bracket_costs_not_regret_and_probability(self):
        src=[{'lambda':1.0,'cost_ms':0.16525520833333332},
             {'lambda':0.5,'cost_ms':0.17953125},
             {'lambda':0.0,'cost_ms':0.203421875}]
        a=select_validation_cost_bracket(src,target_ms=0.17375)
        self.assertEqual(a['lower_cpu_lambda'],1.0)
        self.assertEqual(a['higher_cpu_lambda'],0.5)
        self.assertAlmostEqual(a['expected_mean_ms'],0.17375,places=12)
        self.assertTrue(0<a['higher_cpu_probability']<1)
    def test_balanced_assignment_seed_and_cohort_quotas(self):
        modes={'uniform':{'lower_cpu_lambda':1.0,'higher_cpu_lambda':0.5,
                          'higher_cpu_probability':0.595}}
        cases=[{'id':f'x_{g}_F{f:02d}_R{k:02d}',
                'group':g,'family':f+(12 if g=='B' else 0),
                'history':f'1432{f+1}{k+1}',
                'prefix':f'{f:05d}'}
                for g in ('A','B') for f in range(12) for k in range(16)]
        one=balanced_group_assignments(cases,modes,seed=62011001)
        two=balanced_group_assignments(cases,modes,seed=62011001)
        three=balanced_group_assignments(cases,modes,seed=62011002)
        self.assertEqual(one,two)
        self.assertNotEqual(one,three)
        self.assertEqual(sum(v=='higher' for k,v in one['uniform'].items() if '_A_' in k),114)
        self.assertEqual(sum(v=='higher' for k,v in one['uniform'].items() if '_B_' in k),114)
        self.assertEqual(len(one['uniform']),384)
    def test_deterministic_hash_distinct_seeds(self):
        self.assertTrue(0<=deterministic_hash_uniform(12,'position')<1)
        self.assertNotEqual(deterministic_hash_uniform(12,'position'),
                            deterministic_hash_uniform(13,'position'))
    def test_prior_myopic_has_no_joint_gain_when_lambda_updated(self):
        p={'schema':'quartz_cog02_dynamic_two_step_v1',
           'two_step_gain_coefficients':[[0.0]*9 for _ in range(2)],
           'lambda_wdl_per_ms':1.0}
        q=apply_myopic_lambda(p,0.5)
        self.assertEqual(q['lambda_wdl_per_ms'],0.5)
        self.assertEqual(p['lambda_wdl_per_ms'],1.0)
        self.assertEqual(q['two_step_gain_coefficients'],[[0.0]*9 for _ in range(2)])
        with self.assertRaises(ValueError):
            apply_myopic_lambda({**p,'two_step_gain_coefficients':[[1.]*9]*2},0.5)

if __name__=='__main__':
    unittest.main()
