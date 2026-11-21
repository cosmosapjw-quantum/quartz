"""New exact root evidence mapping must not feed game oracle to Rust."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from oracle_resident13 import teacher_input,source_status

class ExactTeacherSourceContract(unittest.TestCase):
 def test_teacher_root_mapping_preserves_nonterminal_identity(self):
  x={'id':'C13_resident_A_F00_R00','family':0,'group':'resident_A',
     'history':'121213','plies':6,'current_player':1,
     'legal_columns':[0,1,2,3,4,5,6]}
  y=teacher_input(x)
  self.assertEqual(y['cohort'],'resident_A')
  self.assertEqual(y['side_to_move'],1)
  self.assertNotIn('exact_action_scores',y)
 def test_timeout_is_never_converted_into_draw(self):
  s=source_status([{'status':'OK'},{'status':'TIMEOUT'}])
  self.assertEqual(s['counts']['TIMEOUT'],1)
  self.assertFalse(s['all_384_OK'])
  self.assertTrue(s['no_missing_label_imputation'])
if __name__=='__main__':unittest.main()
