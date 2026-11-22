"""C18 TDD: finite source-only revision forecasts, independent of game."""
import math
import unittest
import numpy as np
from quartz.experiments import c18_observation as model

class C18ObservationTests(unittest.TestCase):
    def rows(self):
        return [{"gap":0.0,"n_actions":4,"label":1},
                {"gap":1.0,"n_actions":4,"label":0},
                {"gap":0.2,"n_actions":3,"label":1},
                {"gap":0.9,"n_actions":3,"label":0}]*5

    def test_generic_features_no_domain_or_label(self):
        x=model.features(0.25,4)
        self.assertEqual(len(x),3)
        self.assertAlmostEqual(x[0],1.0)
        self.assertAlmostEqual(x[1],0.25)
        self.assertAlmostEqual(x[2],math.log1p(4))
        with self.assertRaises(ValueError):model.features(-1.0,4)
        with self.assertRaises(ValueError):model.features(0.1,0)

    def test_fitted_logistic_monotonic_and_finite(self):
        rows=self.rows()
        w=model.fit_logistic(rows)
        self.assertEqual(len(w),3)
        self.assertTrue(np.isfinite(w).all())
        p=model.predict(w,rows)
        self.assertEqual(len(p),len(rows))
        self.assertTrue(all(0<p0<1 for p0 in p))
        self.assertLess(p[0],1.0)
        self.assertGreater(p[0],p[1])

    def test_calibration_is_intercept_only(self):
        rows=self.rows()
        w=model.fit_logistic(rows)
        delta=model.calibrate_intercept(w,rows)
        self.assertTrue(math.isfinite(delta))
        np.testing.assert_allclose(model.predict(w,rows,delta),
                                  model.predict(w+np.array([delta,0,0]),rows),
                                  atol=1e-12)

    def test_proper_scores_and_out_of_range_protection(self):
        report=model.score_predictions([0.5,0.5,0.9],[0,1,1])
        self.assertAlmostEqual(report["brier"],(0.25+0.25+0.01)/3)
        self.assertGreater(report["log_loss"],0)
        self.assertEqual(report["n"],3)
        self.assertGreaterEqual(report["ece10"],0)
        with self.assertRaises(ValueError):model.score_predictions([1.1],[0])
        with self.assertRaises(ValueError):model.score_predictions([0.5],[2])

    def test_null_intercept_and_duplicate_training(self):
        rows=[{"gap":0.2,"n_actions":2,"label":0}]*8
        w=model.fit_logistic(rows)
        out=model.predict(w,rows)
        self.assertTrue(all(0<p<0.5 for p in out))
        self.assertTrue(np.isfinite(model.calibrate_intercept(w,rows)))

if __name__=="__main__":
    unittest.main()
