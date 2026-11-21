"""Strictly new root candidates and original CPU model byte invariants."""
import sys
from pathlib import Path
import collections
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from generate_cohort12 import generate_frozen_new_roots

class NewCognitiveRootsTests(unittest.TestCase):
    def test_two_new_mirror_distinct_twelve_family_groups(self):
        cohort=generate_frozen_new_roots()
        self.assertEqual(len(cohort['families']),24)
        self.assertEqual(len(cohort['rows']),384)
        self.assertEqual(collections.Counter(r['group'] for r in cohort['rows']),
                         {'reopen_A':192,'reopen_B':192})
        self.assertEqual(collections.Counter(r['current_player'] for r in cohort['rows']),
                         {1:192,-1:192})
        self.assertEqual(len({r['history'] for r in cohort['rows']}),384)
        self.assertTrue(cohort['prior_all_cohorts_excluded_modulo_horizontal_mirror'])

if __name__=='__main__':unittest.main()
