import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_highbranch14 import validate_real_root, pair_summary, choose_immediate_win_utility

class ActualHighbranch14ScientificControls(unittest.TestCase):
 def test_local_terminal_in_one_decision_utility(self):
  b=[0]*225
  for a in (31,32,33,34):b[a]=1
  self.assertEqual(choose_immediate_win_utility(b,1,30),1)
  self.assertEqual(choose_immediate_win_utility(b,1,35),1)
  self.assertEqual(choose_immediate_win_utility(b,1,112),0)
 def test_same_root_counters_are_physically_preserved(self):
  r={'first_stage':{'root_visits':8,'edge_visit_sum':8,'edge_stats':[[0,8,.5,1.2,.3],[1,0,.5,0.,0.]]},
     'activation_edge_stats':[[0,8,.5,1.2,.3],[1,0,.5,0.,0.]],
     'second_stage':{'root_visits':32,'additional_root_visits':24,'edge_visit_sum':32,'edge_stats':[[0,21,.5,1.2,.3],[1,11,.5,2.5,.1]]},
     'same_root_identity_verified':True,
     'edge_stats_preserved_at_activation':True,
     'no_root_evaluator_call_on_mask_change':True,
     'root_prior_preserved':True,
     'root_hash_before':'test','root_hash_after':'test'}
  validate_real_root(r)
  r['activation_edge_stats'][0][1]=9
  with self.assertRaises(ValueError):validate_real_root(r)
 def test_same_cpu_gate_and_single_stratum_paired_summary(self):
  outcomes=[{'stratum':'neutral','evidence':{'is_immediate_win':False,'teacher_agree':False,'cost_us':100},
             'full32':{'is_immediate_win':False,'teacher_agree':True,'cost_us':105}},
            {'stratum':'neutral','evidence':{'is_immediate_win':True,'teacher_agree':True,'cost_us':105},
             'full32':{'is_immediate_win':False,'teacher_agree':False,'cost_us':100}}]
  s=pair_summary(outcomes,'evidence','full32')
  self.assertEqual(s['n'],2)
  self.assertEqual(s['local_immediate_win_difference'],1)
  self.assertEqual(s['teacher_exact_agreement_difference'],0)
  self.assertEqual(s['cpu_10pct_match'],'MATCHED_10_PERCENT')
if __name__=='__main__':unittest.main()
