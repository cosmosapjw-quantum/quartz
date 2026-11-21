import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from freeze_gomoku14 import build_cohort, ordered_center_moves, replay

class FrozenGomoku14Tests(unittest.TestCase):
    def test_both_distinct_predeclared_strata(self):
        cohort=build_cohort()
        self.assertEqual(len(cohort['rows']),24)
        self.assertEqual(len([r for r in cohort['rows'] if r['stratum']=='neutral']),12)
        self.assertEqual(len([r for r in cohort['rows'] if r['stratum']=='tactical_stress']),12)
        self.assertEqual(len(set(tuple(r['history']) for r in cohort['rows'])),24)
        for row in cohort['rows']:
            state=replay(row['history'])
            self.assertFalse(state['terminal'])
            self.assertEqual(state['side_to_move'],row['side_to_move'])
            self.assertEqual(len(state['legal']),225-len(row['history']))
            rank=ordered_center_moves(state['legal'])
            self.assertEqual(len(rank),len(state['legal']))
            self.assertEqual(len(rank[:16]),16)
        self.assertTrue(cohort['source_generated_before_Rapfi_labels'])
    def test_tactical_stress_has_off_center_legal_win(self):
        cohort=build_cohort()
        for row in cohort['rows']:
            if row['stratum']=='tactical_stress':
                self.assertTrue(row['construction_has_omitted_immediate_win'])
                self.assertGreater(row['constructed_immediate_win_count'],0)
if __name__=='__main__':unittest.main()
