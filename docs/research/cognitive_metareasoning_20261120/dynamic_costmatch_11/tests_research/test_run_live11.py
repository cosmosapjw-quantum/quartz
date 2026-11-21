"""True Rust online single-step mixture request and predeclared randomized estimands."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_live11 import mixture_wdl_expectation, make_locked_mixture_policy, mean_time_match

class LockedCostMatchedRunners(unittest.TestCase):
    def test_costmix_model_one_step_only_and_selected_before_search(self):
        source={
            'schema':'quartz_cog02_dynamic_two_step_v1',
            'features':'root_visits_v1',
            'lambda_wdl_per_ms':1.0,
            'two_step_gain_coefficients':[[0.]*9 for _ in range(2)]
        }
        row={'id':'C11_new_A_F00_R00'}
        plan={'uniform':{
            'lower_cpu_lambda':1.0,'higher_cpu_lambda':0.5,
            'higher_cpu_probability':0.595}}
        assignments={'uniform':{'C11_new_A_F00_R00':'higher'}}
        out=make_locked_mixture_policy(row,source,plan,assignments,'uniform')
        self.assertEqual(out['lambda_wdl_per_ms'],0.5)
        self.assertEqual(out['two_step_gain_coefficients'],[[0.]*9 for _ in range(2)])
        self.assertEqual(source['lambda_wdl_per_ms'],1.0)
        with self.assertRaises(ValueError):
            make_locked_mixture_policy(row,source,plan,{'uniform':{'C11_new_A_F00_R00':'unknown'}},'uniform')
    def test_ex_ante_random_mixture_conditional_expectation(self):
        # low CPU regret=2, high CPU regret=0; p_high=0.25
        result=mixture_wdl_expectation({'regret':2,'search_ms':.1},
                                       {'regret':0,'search_ms':.3},.25)
        self.assertAlmostEqual(result['E_regret'],1.5)
        self.assertAlmostEqual(result['E_search_ms'],.15)
    def test_cost_gate_cannot_hide_large_time_mismatch(self):
        self.assertEqual(mean_time_match(.1,.12),'NOT_MATCHED_10_PERCENT')
        self.assertEqual(mean_time_match(.2,.215),'MATCHED_10_PERCENT')
        with self.assertRaises(ValueError):mean_time_match(0,.1)

if __name__=='__main__':unittest.main()
