"""COG07 teacher/source contracts: no inferred labels and no split leakage."""
from collections import Counter
from pathlib import Path
import json
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from label_1024 import (
    verify_cohort, exact_wdl_target, assert_complete_teacher,
)

class COG07TeacherTests(unittest.TestCase):
    def test_cohort_and_group_splits(self):
        d=Path(__file__).resolve().parents[1]
        cohort=json.loads((d/"COHORT_1024_FROZEN.json").read_text())
        result=verify_cohort(cohort)
        self.assertEqual(result['families'],64)
        self.assertEqual(result['splits'],{'train':640,'val':192,'test':192})
        self.assertEqual(result['colors'],{'black':512,'white':512})

    def test_teacher_optimal_action_set_includes_ties(self):
        row=exact_wdl_target([3,7,7,-1000,-8,0,-1],[0,1,2,4,5,6])
        self.assertEqual(row['root_wdl'],1)
        self.assertEqual(row['optimal_actions'],[0,1,2])
        self.assertAlmostEqual(row['target_policy'][0],1/3)
        self.assertEqual(row['target_policy'][3],0)
        self.assertEqual(sum(row['target_policy']),1.0)

    def test_missing_rows_are_not_implicitly_draws(self):
        with self.assertRaisesRegex(RuntimeError,"missing"):
            assert_complete_teacher([
                {'status':'OK','id':'a','root_wdl':0},
                {'status':'TIMEOUT','id':'b'}
            ],2)

    def test_cannot_accept_unsupported_label_scores(self):
        with self.assertRaisesRegex(ValueError,"sentinel"):
            exact_wdl_target([-1000,0,0,0,0,0,0],list(range(7)))

if __name__=='__main__':unittest.main()
