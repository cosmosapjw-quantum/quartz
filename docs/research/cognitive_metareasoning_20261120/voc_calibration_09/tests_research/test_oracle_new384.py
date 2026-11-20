"""New-family calibration source identity and strict no-imputation tests."""
from pathlib import Path
import hashlib
import json
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from generate_new_cohorts import freeze,root_modules
from oracle_new384 import evaluate_rows

class Calibration09SourceTests(unittest.TestCase):
    def test_frozen_input_is_byte_reproducible_before_oracle(self):
        root=Path(__file__).resolve().parents[1]
        actual=(root/"COHORT_384_FROZEN.json").read_bytes()
        res=json.dumps(freeze(),sort_keys=True,ensure_ascii=False,indent=2).encode()+b"\n"
        self.assertEqual(hashlib.sha256(actual).hexdigest(),hashlib.sha256(res).hexdigest())
    def test_global_budget_does_not_create_fake_exact_draw(self):
        cases=[{'id':'a','family':0,'cohort':'replica_A','history':'12345',
                'ply':5,'side_to_move':-1,'legal_columns':list(range(7))}]
        rows,_=evaluate_rows(cases,Path('/does-not-exist'),Path('/does-not-exist'),cap_seconds=-1)
        self.assertEqual(rows[0]['status'],'GLOBAL_CAP_NOT_ATTEMPTED')
        self.assertNotIn('exact_action_scores',rows[0])

if __name__=='__main__':unittest.main()
