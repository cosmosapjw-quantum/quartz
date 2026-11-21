"""Source-split and exact combinatorial nonmyopic information synergy."""
import hashlib
import json
from pathlib import Path
from fractions import Fraction
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from freeze_c10 import freeze
from prove_xor import evaluate

class SourceLockTests(unittest.TestCase):
    def test_exact_two_step_joint_information_synergy(self):
        r=evaluate()
        self.assertEqual(r['after_B1_optimal_utility'],'1/2')
        self.assertEqual(r['after_B2_optimal_utility'],'1/2')
        self.assertEqual(r['after_B1_and_B2_optimal_utility'],'1')
        self.assertEqual(Fraction(r['one_step_net_gain']),Fraction(-1,10))
        self.assertEqual(Fraction(r['two_step_net_gain']),Fraction(3,10))
    def test_frozen_384_cohort_byte_determinism(self):
        d=Path(__file__).resolve().parents[1]
        actual=(d/'COHORT_384_FROZEN.json').read_bytes()
        recreated=(json.dumps(freeze(),sort_keys=True,indent=2,ensure_ascii=False)+'\n').encode()
        self.assertEqual(hashlib.sha256(actual).hexdigest(),hashlib.sha256(recreated).hexdigest())
        co=json.loads(actual)
        self.assertEqual(len(co['families']),24)
        self.assertEqual(len(co['rows']),384)
        self.assertEqual(len([x for x in co['rows'] if x['group']=='fresh_A']),192)
        self.assertEqual(len([x for x in co['rows'] if x['group']=='fresh_B']),192)
if __name__=='__main__':unittest.main()
