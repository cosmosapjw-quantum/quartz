"""Signed-gain calibration bins and paired family inference guardrails."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_new384_release09 import forecast_bin, paired_cluster_summary


class NewFamilyCalibrationTests(unittest.TestCase):
    def test_locked_signed_gain_bins_include_all_real_values(self):
        self.assertEqual(forecast_bin(-1.0),0)
        self.assertEqual(forecast_bin(-0.2),0)
        self.assertEqual(forecast_bin(-0.05),1)
        self.assertEqual(forecast_bin(0),2)
        self.assertEqual(forecast_bin(0.05),3)
        self.assertEqual(forecast_bin(0.2),4)
        self.assertEqual(forecast_bin(0.21),5)
        with self.assertRaises(ValueError):forecast_bin(float('nan'))

    def test_family_resampling_not_per_root(self):
        a=[{'family':i//2,'regret':1} for i in range(24)]
        b=[{'family':i//2,'regret':0} for i in range(24)]
        output=paired_cluster_summary(a,b,seed=17)
        self.assertEqual(output['n_families'],12)
        self.assertEqual(output['n_roots'],24)
        self.assertEqual(output['mean_left_minus_right_regret'],1.0)
        self.assertEqual(output['conditional_95_interval'],[1.0,1.0])

if __name__=='__main__': unittest.main()
