"""Model-vs-uniform sampled WDL exact comparison contract."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark_cpu import per_root_metrics, summarize_modes

class BudgetComparisonTests(unittest.TestCase):
    def setUp(self):
        self.scores=[1,-5,0,-1000,2,4,-1]
        self.baseline={'history':'x', 'eval_mode':'uniform_no_neural_network',
            'root_hash':'hash',
            'steps':[{'budget':b,'root_visits':b,'best_action':a,'evaluator_calls':c,'step_wall_us':u,'root_hash':'hash'}
                     for b,a,c,u in [(8,1,3,100),(16,2,5,200),(32,4,7,400),(64,5,9,800)]],
            'fresh16':{'resident_match':True}}
    def test_exact_utility_negative_and_zero_and_cost(self):
        m=per_root_metrics(self.baseline,self.scores)
        self.assertEqual(m['utility'],[-1,0,1,1])
        self.assertEqual(m['regret'],[2,1,0,0])
        self.assertEqual(m['delta_utility'],[1,1,0])
        self.assertEqual(m['cumulative_eval_calls'],[3,5,7,9])
    def test_invalid_action_must_not_be_coerced(self):
        other=dict(self.baseline)
        other['steps']=[dict(x) for x in self.baseline['steps']]
        other['steps'][0]['best_action']=3
        with self.assertRaisesRegex(ValueError,'illegal'):
            per_root_metrics(other,self.scores)
    def test_two_mode_paired_comparison(self):
        u=per_root_metrics(self.baseline,self.scores)
        model=dict(self.baseline)
        model['steps']=[dict(s) for s in self.baseline['steps']]
        model['steps'][0]['best_action']=4
        m=per_root_metrics(model,self.scores)
        d=summarize_modes([{'id':'r1','uniform':u,'trained':m}],budgets=[8,16,32,64])
        self.assertEqual(d['by_budget']['8']['trained_minus_uniform_mean_utility'],2)
        self.assertEqual(d['by_budget']['16']['trained_minus_uniform_mean_utility'],0)
        self.assertEqual(d['n_paired_roots'],1)

if __name__=='__main__':unittest.main()
