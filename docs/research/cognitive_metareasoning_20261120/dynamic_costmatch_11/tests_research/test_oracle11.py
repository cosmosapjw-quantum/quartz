"""Source-only teacher mapping and preservation of missing external oracle statuses."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from oracle11 import map_to_external_teacher, classify_teacher_status

class HeldoutTeacherTests(unittest.TestCase):
    def test_map_frozen_root_without_future_label(self):
        source={'id':'C11_new_A_F00_R00','family':0,'group':'new_A',
                'history':'12343','plies':5,'current_player':-1,
                'legal_columns':[0,1,2,3,4,5,6]}
        out=map_to_external_teacher(source)
        self.assertEqual(out['cohort'],'new_A')
        self.assertEqual(out['history'],'12343')
        self.assertEqual(out['ply'],5)
        self.assertEqual(out['side_to_move'],-1)
        self.assertNotIn('exact_action_scores',out)
    def test_statuses_do_not_impute_timeout_as_draw(self):
        rows=[{'status':'OK','root_wdl':1},{'status':'TIMEOUT'},
              {'status':'GLOBAL_CAP_NOT_ATTEMPTED'}]
        result=classify_teacher_status(rows)
        self.assertFalse(result['all_ok'])
        self.assertEqual(result['counts']['OK'],1)
        self.assertEqual(result['counts']['TIMEOUT'],1)
        self.assertEqual(result['counts']['GLOBAL_CAP_NOT_ATTEMPTED'],1)
        self.assertFalse(result['usable_in_full_scientific_test'])

if __name__=='__main__':unittest.main()
