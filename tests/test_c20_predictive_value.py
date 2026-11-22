import unittest, math
from fractions import Fraction
from quartz.experiments.c20_predictive_value import beta_evsi,sensor_evsi,generic_evsi

class ExactBayesEVSITests(unittest.TestCase):
    def test_beta_two_uniform_arms_one_each(self):
        a=beta_evsi([(1.0,1.0),(1.0,1.0)],1)
        self.assertAlmostEqual(a["evsi"],1/12,delta=1e-12)
        self.assertEqual(a["outcome_count"],4)
        self.assertAlmostEqual(a["total_predictive_mass"],1,delta=1e-12)
        self.assertGreaterEqual(a["evaluation_us"],0)

    def test_exact_sensor_binary_noisy_one_signal(self):
        r=sensor_evsi([0.5,0.5],[[0.75,0.25],[0.25,0.75]],
         [[1,0],[0,1]],1)
        self.assertAlmostEqual(r["evsi"],0.25,delta=1e-12)
        self.assertAlmostEqual(r["total_predictive_mass"],1,delta=1e-12)

    def test_shared_latent_correlated_sensor(self):
        r=sensor_evsi([.5,.5],[[1,0],[0,1]],[[1,0],[0,1]],2)
        self.assertAlmostEqual(r["evsi"],.5,delta=1e-12)
        self.assertEqual(r["outcome_count"],3)
        self.assertAlmostEqual(r["total_predictive_mass"],1,delta=1e-12)

    def test_no_information_and_zero_budget(self):
        r=sensor_evsi([.5,.5],[[.4,.6],[.4,.6]],[[1,0],[0,1]],2)
        self.assertAlmostEqual(r["evsi"],0,delta=1e-12)
        r2=beta_evsi([(3,4),(4,3)],0)
        self.assertAlmostEqual(r2["evsi"],0,delta=1e-12)

    def test_generic_mixture_aggregator(self):
        result=generic_evsi(.5,[(.5,[.75,.25]),(.5,[.25,.75])])
        self.assertAlmostEqual(result["evsi"],.25,delta=1e-12)
        with self.assertRaises(ValueError):
            generic_evsi(.5,[(.75,[1,0])])

    def test_probability_invalid_and_enumeration_limit(self):
        with self.assertRaises(ValueError):beta_evsi([(1,1),(1,1)],50)
        with self.assertRaises(ValueError):beta_evsi([(1,0),(1,1)],1)
        with self.assertRaises(ValueError):sensor_evsi([.5,.5],[[.5,.5],[.5,.6]],[[0,1],[1,0]],1)
        with self.assertRaises(ValueError):sensor_evsi([.5,.5],[[.5,.5],[.5,.5]],[[0,1],[1,0]],-1)

if __name__=="__main__":unittest.main()
