"""Release validation-only clock calibration, no oracle, no test leakage."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from release_validate09 import choose_cap, stop_from_full


class ReleaseCostTests(unittest.TestCase):
    def test_clock_cap_checks_only_completed_stage_prefix(self):
        stage_us=[100,200,300,400]
        self.assertEqual(stop_from_full(stage_us,0.05),(8,0.1))
        result=stop_from_full(stage_us,0.25)
        self.assertEqual(result[0],16)
        self.assertAlmostEqual(result[1],0.3)
        self.assertEqual(stop_from_full(stage_us,3.0),(64,1.0))

    def test_validation_only_cap_selection_matches_mean_cost(self):
        sample=[
            {'steps':[{'step_wall_us':100},{'step_wall_us':200},
                       {'step_wall_us':300},{'step_wall_us':400}]},
            {'steps':[{'step_wall_us':100},{'step_wall_us':200},
                       {'step_wall_us':300},{'step_wall_us':400}]}
        ]
        result=choose_cap(sample,0.3)
        self.assertAlmostEqual(result['mean_search_ms'],0.3)
        self.assertEqual(result['stop_visits'],{'16':2})
        self.assertTrue(0.1<result['cap_ms']<=0.3)


if __name__=='__main__':
    unittest.main()
