"""Family-cluster confidence and exact signed gain calibration bin boundaries."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_live10 import paired_family_ci, forecast_bin, cost_match_status

class DynamicLiveEvaluationTests(unittest.TestCase):
    def test_family_bootstrap_resamples_twelve_families_not_24_positions(self):
        a=[{'family':i//2,'regret':0} for i in range(24)]
        b=[{'family':i//2,'regret':1} for i in range(24)]
        r=paired_family_ci(a,b)
        self.assertEqual(r['n_roots'],24)
        self.assertEqual(r['n_family_clusters'],12)
        self.assertEqual(r['mean_regret_difference'],-1.0)
        self.assertEqual(r['conditional_family_bootstrap_95'],[-1.0,-1.0])
    def test_signed_gain_bins_and_cost_match_rules(self):
        self.assertEqual([forecast_bin(x) for x in [-1,-.2,-.05,0,.05,.2,.21]],[0,0,1,2,3,4,5])
        self.assertEqual(cost_match_status(1.0,1.08),'MATCHED_10_PERCENT')
        self.assertEqual(cost_match_status(1.0,1.15),'NOT_MATCHED_10_PERCENT')
        with self.assertRaises(ValueError):forecast_bin(float('nan'))

if __name__=='__main__':unittest.main()
