"""Posthoc exact-WDL comparison, never online candidate selection."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_live12 import (exact_regret_for_action,
                        identify_decisive_omissions,
                        paired_cluster_interval,
                        decision_cost_match)

class ReopeningWDLTests(unittest.TestCase):
    def test_posthoc_decisive_excluded_columns(self):
        legal=[0,1,2,3,4,5,6]
        score=[2,1,-2,0,0,-1,1]
        initial=[3,2,4]
        d=identify_decisive_omissions(score,legal,initial)
        self.assertEqual(d,[0,1,6])
        self.assertEqual(exact_regret_for_action(score,legal,3),1)
        self.assertEqual(exact_regret_for_action(score,legal,0),0)
    def test_illegal_external_sentinel_never_treated_as_loss(self):
        with self.assertRaises(ValueError):
            exact_regret_for_action([-1000]*7,[],2)
    def test_cluster_bootstrap_12_not_root384_iid(self):
        a=[{'family':j//16,'regret':0} for j in range(192)]
        b=[{'family':j//16,'regret':1} for j in range(192)]
        result=paired_cluster_interval(a,b,seed=17)
        self.assertEqual(result['n_families'],12)
        self.assertEqual(result['n_roots'],192)
        self.assertEqual(result['mean_regret_delta'],-1)
        self.assertEqual(result['CI95'],[-1,-1])
    def test_end_to_end_cost_gate(self):
        self.assertEqual(decision_cost_match(1,1.05),'MATCHED_10_PERCENT')
        self.assertEqual(decision_cost_match(1,1.15),'NOT_MATCHED_10_PERCENT')
        with self.assertRaises(ValueError):decision_cost_match(0,1)

if __name__=='__main__':unittest.main()
