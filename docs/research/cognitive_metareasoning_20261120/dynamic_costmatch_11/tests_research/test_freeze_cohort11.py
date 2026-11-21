"""New family and pre-oracle randomized myopic assignment source locks."""
from pathlib import Path
import sys
import json
import unittest
import collections
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from freeze_cohort11 import build_new_cohort, build_locked_assignments

class FrozenNewCohortTests(unittest.TestCase):
    def test_two_new_twentyfour_families_without_prior_training_or_tests(self):
        c=build_new_cohort()
        self.assertEqual(len(c['families']),24)
        self.assertEqual(len(c['rows']),384)
        self.assertEqual(collections.Counter(r['group'] for r in c['rows']),{'new_A':192,'new_B':192})
        self.assertEqual(collections.Counter(r['current_player'] for r in c['rows']),{1:192,-1:192})
        self.assertEqual(len({r['history'] for r in c['rows']}),384)
        self.assertTrue(c['all_prior_cohorts_excluded_modulo_mirror'])
    def test_three_preassigned_block_randomizations_before_oracle(self):
        c=build_new_cohort()
        d=Path(__file__).resolve().parents[1]
        m=json.loads((d/'MIXTURE_PLAN_FROZEN.json').read_text())
        p=build_locked_assignments(c,m)
        self.assertEqual(len(p['seed_assignments']),3)
        for key,entry in p['seed_assignments'].items():
            for group in ('new_A','new_B'):
                cases=[r for r in c['rows'] if r['group']==group]
                a=entry['uniform']
                self.assertEqual(sum(a[r['id']]=='higher' for r in cases),114)
                self.assertEqual(sum(a[r['id']]=='lower' for r in cases),78)
            self.assertTrue(all(x=='lower' for x in entry['conv'].values()))
            self.assertTrue(all(x=='lower' for x in entry['new_mlp'].values()))
        self.assertNotEqual(p['seed_assignments'][str(m['assignment_seeds'][0])]['uniform'],
                            p['seed_assignments'][str(m['assignment_seeds'][1])]['uniform'])

if __name__=='__main__':unittest.main()
