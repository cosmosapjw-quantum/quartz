"""New 24-family root suite must be disjoint from original C12 and all earlier cohorts."""
import collections
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from generate_resident_cohort13 import generate_cohort

class FreshResidentFamilyTests(unittest.TestCase):
    def test_new_384_roots_and_twenty_four_families(self):
        src=generate_cohort()
        self.assertEqual(len(src['families']),24)
        self.assertEqual(len(src['rows']),384)
        self.assertEqual(collections.Counter(x['group'] for x in src['rows']),{'resident_A':192,'resident_B':192})
        self.assertEqual(collections.Counter(x['current_player'] for x in src['rows']),{1:192,-1:192})
        self.assertEqual(len({x['history'] for x in src['rows']}),384)
        self.assertTrue(src['previous_C12_and_earlier_mirror_sources_excluded'])
        self.assertEqual(src['seeds'],{'resident_A':202611210321,'resident_B':202611210322})
    def test_root_lookup_has_unique_family_and_mirror_tags(self):
        x=generate_cohort()
        self.assertEqual(len({family['canonical_prefix'] for family in x['families']}),24)
        self.assertEqual(len({r['id'] for r in x['rows']}),384)
if __name__=='__main__':unittest.main()
