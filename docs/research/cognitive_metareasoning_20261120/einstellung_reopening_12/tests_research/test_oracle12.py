"""External exact-source mapping preserves missing statuses without imputation."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from oracle12 import teacher_input,complete_status

class COG03ExternalTeacherTest(unittest.TestCase):
    def test_convert_new_root_without_exact_solver_injection(self):
        row={'id':'C12_reopen_A_F00_R00','family':0,'group':'reopen_A',
             'history':'121213','plies':6,'current_player':1,
             'legal_columns':[0,1,2,3,4,5,6]}
        result=teacher_input(row)
        self.assertEqual(result['cohort'],'reopen_A')
        self.assertEqual(result['ply'],6)
        self.assertEqual(result['side_to_move'],1)
        self.assertNotIn('exact_action_scores',result)
    def test_missing_external_score_remains_explicit_blocker(self):
        r=complete_status([{'status':'OK'},{'status':'TIMEOUT'}])
        self.assertFalse(r['all_OK'])
        self.assertEqual(r['statuses']['TIMEOUT'],1)
        self.assertTrue(r['missing_labels_not_imputed'])

if __name__=='__main__':unittest.main()
