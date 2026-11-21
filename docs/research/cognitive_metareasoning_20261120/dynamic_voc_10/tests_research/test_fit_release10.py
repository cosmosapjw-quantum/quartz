"""Val-only selection and registered budget grid, keep frozen test untouched."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fit_release10 import select_validation_candidate, checkpoint_cap

class ValidationSelectionTenTests(unittest.TestCase):
    def test_validation_objective_selects_lowest_real_cost_on_tie(self):
        table=[
          {'lambda':0.0,'val':{'objective':0.5,'mean_search_ms':0.3}},
          {'lambda':1.0,'val':{'objective':0.5,'mean_search_ms':0.1}},
          {'lambda':2.0,'val':{'objective':0.5,'mean_search_ms':0.1}},
          {'lambda':4.0,'val':{'objective':0.49,'mean_search_ms':0.01}}
        ]
        result=select_validation_candidate(table)
        self.assertEqual(result['lambda'],2.0)

    def test_checkpoint_clock_actual_search_overshoot_allowed(self):
        stages=[0.02,0.03,0.04,0.05]
        self.assertEqual(checkpoint_cap(stages,.01),(8,.02))
        self.assertAlmostEqual(checkpoint_cap(stages,.04)[1],.05)
        self.assertEqual(checkpoint_cap(stages,10.)[0],64)

if __name__=='__main__':unittest.main()
