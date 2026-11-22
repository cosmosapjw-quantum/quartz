import unittest
from quartz.experiments.c22_active_inputs import public_task,workload_bucket,make_case

class C22SourceOnlyInputs(unittest.TestCase):
 def test_beta_altering_offline_future_labels_cannot_alter_public(self):
  w={"id":"arms-001","family":"beta_arms","split":"train",
     "K":2,"initial_draws_per_arm":3,"extra_draws_per_arm":5,
     "theta_offline_true_arm_means":[.5,.7]}
  row={"group_id":"arms-001","family":"beta_arms","split":"train","rep":0,
     "initial_observations":[[1,0,0],[0,1,1]],
     "extra_observations":[[0]*5,[1]*5],"signed_gain":0.5,"label":1}
  p=public_task(row,w)
  assert p=={"family":"beta_arms","posterior":[[3.,4.],[4.,3.]],"extra_per_arm":[5,5]}
  self.assertEqual(workload_bucket(p),"beta_K2_n5")
  q=dict(row,extra_observations="arbitrary future fake",signed_gain=-.8,label=0)
  self.assertEqual(public_task(q,w),p)
  c=make_case(q,w)
  self.assertNotIn("theta_offline_arm_means",c["public"])
  self.assertIn("theta_offline_arm_means",c["offline"])
 def test_sensor_beta_posterior_initial_only_and_planned_budget(self):
  w={"id":"sensor-006","family":"finite_sensor","split":"train",
    "initial_observations":2,"extra_observations":3,
    "initial_prior":[.25]*4,"theta_offline_true_state":1,
    "observation_likelihood_state_outcome_PUBLIC":[[.7,.2,.1],[.1,.7,.2],[.2,.1,.7],[.2,.4,.4]],
    "utility_state_action_matrix_PUBLIC":[[.8,.2,.3],[.2,.8,.3],[.3,.2,.8],[.4,.6,.2]]}
  row={"group_id":"sensor-006","family":"finite_sensor","split":"train","rep":0,
       "initial_observations":[1,2],"extra_observations":[0,1,2]}
  p=public_task(row,w)
  self.assertEqual(workload_bucket(p),"sensor_n3")
  self.assertAlmostEqual(sum(p["prior"]),1)
  self.assertNotIn("theta_offline_true_state",p)
  q=dict(row,extra_observations=[])
  self.assertEqual(p,public_task(q,w))
  with self.assertRaises(ValueError):public_task(dict(row,initial_observations=[1]),w)
if __name__=="__main__":unittest.main()
