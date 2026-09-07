"""Focused mathematical checks and source-semantic counterexamples.

A passing test that reproduces a legacy defect does not validate legacy KG.
"""
import json
import math
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE),str(REPO),str(REPO/'prototype')]
from gaussian_batch_pilot import (D, NOISE, MEAN, gain, observe_cov,
                                 shift_variance, allocate, posterior, exact_bayes_regret)
from bqpp_prototype.kg import compute_kg_array, kg_gaussian_per_arm
from quartz.experiments.forked_voc import voc_proxy
from quartz.phase15_one_loop import one_loop_correction


def source_probes():
    leader_sd = 1/math.sqrt(2)
    n = 100
    v = 1/(n+4)
    honest_sd = v/math.sqrt(v+1)
    legacy = kg_gaussian_per_arm(0,n,1,.02,n,1)
    policies = [np.array([.9,.1]),np.array([.1,.9]),np.array([.9,.1])]
    corrected,_ = one_loop_correction(np.array([.8,.2,0.]),32)
    return {
      'leader_kg': {'legacy':compute_kg_array([.1,0],[0,0],[1,.01])[0],
                    'exact_two_action_gaussian':gain(.1,leader_sd),
                    'assumptions':'posterior covariance diag(1,.01), observe leader with noise variance 1'},
      'kg_scale': {'legacy_proxy':legacy,'exact_one_observation_kg':gain(.02,honest_sd),
                    'legacy_shift_sd':math.sqrt(2*v),'correct_shift_sd':honest_sd,
                    'assumptions':'independent posterior variances 1/104; one challenger observation noise variance 1'},
      'voc_churn': {'legacy_proxy':voc_proxy([8,16,32],policies),
                    'true_utilities':[1,0],'decisions':[0,1,0],
                    'signed_realized_utility_changes':[-1,1], 'net_utility_change':0},
      'omitted_arm': {'input':[.8,.2,0.],'output':corrected.tolist(),
                       'omitted_arm_can_be_best':True},
      'acyclic_hessian': {'hessian':[[2,-1],[-1,2]],'determinant':3,'product_of_diagonals':4},
      'claim_ceiling':'source-semantic counterexamples and conditional Gaussian identities only'
    }


class ResearchMath(unittest.TestCase):
    def test_kg_matches_independent_integral(self):
        for delta,sd in [(0,.3),(.2,.4),(2,.7),(.03,.01)]:
            got,_ = quad(lambda z:max(sd*z-delta,0)*math.exp(-z*z/2)/math.sqrt(2*math.pi),
                         delta/sd,np.inf,epsabs=1e-13)
            self.assertAlmostEqual(gain(delta,sd),got,places=12)

    def test_covariance_update_matches_precision(self):
        s = np.array([[.4,.1],[.1,.3]])
        expected = np.linalg.inv(np.linalg.inv(s)+np.diag([0,1/.2]))
        np.testing.assert_allclose(observe_cov(s,1,.2),expected,rtol=1e-13,atol=1e-15)

    def test_common_mode_has_zero_decision_gain(self):
        s = np.array([[1.,.8],[.8,1.]])
        h = np.ones(2)
        u = s@h/math.sqrt(h@s@h+.5)
        self.assertEqual(float(D@u),0)
        self.assertGreater(float(u@u),0)
        self.assertEqual(gain(.1,abs(float(D@u))),0)

    def test_leader_has_positive_value_source_disagrees(self):
        p = source_probes()['leader_kg']
        self.assertEqual(p['legacy'],0)
        self.assertGreater(p['exact_two_action_gaussian'],.2)

    def test_current_uncertainty_is_not_innovation(self):
        p = source_probes()['kg_scale']
        self.assertGreater(p['legacy_shift_sd']/p['correct_shift_sd'],14)
        self.assertGreater(p['legacy_proxy'],p['exact_one_observation_kg'])

    def test_voc_proxy_cannot_determine_utility_sign(self):
        p = source_probes()['voc_churn']
        self.assertAlmostEqual(p['legacy_proxy'],1.6)
        self.assertEqual(p['net_utility_change'],0)
        self.assertLess(p['signed_realized_utility_changes'][0],0)

    def test_posthoc_cannot_reopen_zero_support(self):
        self.assertEqual(source_probes()['omitted_arm']['output'][2],0)

    def test_free_energy_hessian_and_contrast_identity(self):
        beta = 1.7
        mu = np.array([.2,-.1,.5])
        prior = np.array([.2,.5,.3])
        pi = np.exp(np.log(prior)+beta*mu-logsumexp(np.log(prior)+beta*mu))
        cov = np.array([[.2,.03,.01],[.03,.1,.02],[.01,.02,.3]])
        hess = beta*(np.diag(pi)-np.outer(pi,pi))
        pair = sum(pi[a]*pi[b]*(cov[a,a]+cov[b,b]-2*cov[a,b]) for a in range(3) for b in range(3))
        self.assertAlmostEqual(.5*np.trace(hess@cov),beta*pair/4,places=13)
        f = lambda x:logsumexp(np.log(prior)+beta*x)/beta
        eps = 1e-4
        numerical = np.empty((3,3))
        for a in range(3):
          for b in range(3):
            ea,eb = np.eye(3)[a]*eps,np.eye(3)[b]*eps
            numerical[a,b]=(f(mu+ea+eb)-f(mu+ea-eb)-f(mu-ea+eb)+f(mu-ea-eb))/(4*eps**2)
        np.testing.assert_allclose(hess,numerical,rtol=1e-6,atol=1e-8)

    def test_tree_hessian_need_not_be_diagonal(self):
        h = np.array([[2.,-1.],[-1.,2.]])
        self.assertAlmostEqual(np.linalg.det(h),3)
        self.assertNotEqual(np.linalg.det(h),np.prod(h.diagonal()))

    def test_serial_pending_matches_stale_and_counts(self):
        s = .16*np.array([[1.,.9],[.9,1.]])
        for budget in [8,31,128]:
            a,pa = allocate(s,budget,1,'stale_batch')
            b,pb = allocate(s,budget,1,'pending_greedy')
            np.testing.assert_array_equal(a,b)
            np.testing.assert_allclose(pa,pb)
            self.assertEqual(int(a.sum()),budget)
            self.assertGreater(np.linalg.eigvalsh(pa).min(),0)

    def test_posterior_zero_data_and_exact_risk(self):
        s = np.diag([.16,.16])
        means,post = posterior(s,np.array([0,0]),np.zeros((1,2)))
        np.testing.assert_allclose(means[0],MEAN)
        np.testing.assert_allclose(post,s)
        self.assertAlmostEqual(exact_bayes_regret(s,s),gain(.1,math.sqrt(D@s@D)))
        self.assertAlmostEqual(exact_bayes_regret(s,np.zeros((2,2))),0)

    def test_exact_risk_nonincreasing_with_observations(self):
        s = .16*np.array([[1.,-.6],[-.6,1.]])
        post = s.copy()
        old = exact_bayes_regret(s,post)
        for arm in [0,1,1,0,1]*20:
            post = observe_cov(post,arm,NOISE[arm])
            new = exact_bayes_regret(s,post)
            self.assertLessEqual(new,old+1e-14)
            self.assertGreaterEqual(new,-1e-14)
            old = new


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--probes':
        print(json.dumps(source_probes(),indent=2))
    else:
        unittest.main(verbosity=2)
