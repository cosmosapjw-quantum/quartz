"""Exact signed WDL and matched physical CPU and family uncertainty for resident root."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_resident13 import (
    exact_regret,decisive_omitted,physical_mean_cost_match,
    paired_cluster_ci,validate_resident_transition
)

class ResidentC13ScoringTests(unittest.TestCase):
    def test_exact_game_action_utility_and_omissions(self):
        s=[2,1,-2,0,0,-1,1]
        a=list(range(7));closed=[3,2,4]
        self.assertEqual(exact_regret(s,a,0),0)
        self.assertEqual(exact_regret(s,a,3),1)
        self.assertEqual(decisive_omitted(s,a,closed),[0,1,6])
    def test_missing_illegal_action_cannot_fake_wdl_score(self):
        with self.assertRaises(ValueError):
            exact_regret([-1000]*7,[],0)
        with self.assertRaises(ValueError):
            decisive_omitted([-1000]*7,[],[0])
    def test_physical_cpu_cost_threshold(self):
        self.assertEqual(physical_mean_cost_match(.2,.21),'MATCHED_10_PERCENT')
        self.assertEqual(physical_mean_cost_match(.2,.25),'NOT_MATCHED_10_PERCENT')
        with self.assertRaises(ValueError):
            physical_mean_cost_match(0,.2)
    def test_cluster_bootstrap_root_variance_not_IID(self):
        a=[{'family':i//16,'regret':0} for i in range(192)]
        b=[{'family':i//16,'regret':1} for i in range(192)]
        q=paired_cluster_ci(a,b)
        self.assertEqual(q['n_family_clusters'],12)
        self.assertEqual(q['n_roots'],192)
        self.assertEqual(q['CI95'],[-1,-1])
    def test_history_preserved_after_activation(self):
        e={'first_search':{'edge_stats':[{'action':0,'prior':.4,'visits':0,'w':0.,'m2':0.},
                                           {'action':1,'prior':.6,'visits':8,'w':2.5,'m2':.1}],
                           'root_visits':8,'edge_visit_sum':8,
                           'root_evaluator_calls':9},
           'activation_edge_stats':[{'action':0,'prior':.4,'visits':0,'w':0.,'m2':0.},
                                    {'action':1,'prior':.6,'visits':8,'w':2.5,'m2':.1}],
           'root_hash_before':'x','root_hash_after':'x',
           'resident_same_root_node':True,
           'root_evaluator_calls_identical_during_activation':True,
           'prior_and_edge_stats_identical_immediately_after_activation':True,
           'root_visits_immediately_after_activation':8,
           'original_full_root_edges':2,
           'second_search':{'root_visits':24,'additional_root_visits':16,
                            'edge_visit_sum':24,
                            'root_evaluator_calls':24,
                            'edge_stats':[{'action':0,'prior':.4,'visits':16,'w':1.0,'m2':0.1},
                                          {'action':1,'prior':.6,'visits':8,'w':2.5,'m2':.1}]}}
        validate_resident_transition(e)
        e['activation_edge_stats'][0]['visits']=1
        with self.assertRaises(ValueError):
            validate_resident_transition(e)
if __name__=='__main__':unittest.main()
