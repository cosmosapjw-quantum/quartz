import unittest
from quartz.experiments.c20_task_adapter import initial_only,eval_initial,resource_bucket

class C20TaskAdapterTests(unittest.TestCase):
 def test_beta_initial_only_no_label(self):
  source={"family":"beta_arms","initial_observations":[[1,0,1],[0,0,0]],
   "theta_offline_true_arm_means":[.8,.2],"signed_gain":-.1,
   "label":1,"post_action":0,"actual_extra_cost_us":12}
  input_data=initial_only(source)
  self.assertNotIn("label",input_data)
  self.assertNotIn("theta_offline_true_arm_means",input_data)
  self.assertEqual(resource_bucket(input_data),"beta_arms_K2")
  v=eval_initial(input_data)
  self.assertGreaterEqual(v["evsi"],0)
  self.assertAlmostEqual(v["total_predictive_mass"],1,delta=1e-10)
  with self.assertRaises(ValueError):
   eval_initial(dict(input_data,label=1))

 def test_sensor_initial_only_no_future_and_normal(self):
  source={"family":"finite_sensor","initial_observations":[0,1],
   "prior_PUBLIC":[.25,.25,.25,.25],
   "likelihood_PUBLIC":[[.7,.3],[.3,.7],[.6,.4],[.4,.6]],
   "utilities_PUBLIC":[[1,0],[0,1],[.7,.3],[.3,.7]],"theta_offline_true_state":0,
   "signed_gain":-.2,"label":1}
  # The real frozen C20 sensor contract has 3 actions, not 2.
  with self.assertRaises(ValueError):
   initial_only(source)
  source["utilities_PUBLIC"]=[[1,0,0],[0,1,0],[.7,.3,.2],[.3,.7,.2]]
  a=initial_only(source)
  self.assertEqual(resource_bucket(a),"finite_sensor_K3")
  self.assertNotIn("theta_offline_true_state",a)
  self.assertAlmostEqual(eval_initial(a)["total_predictive_mass"],1,delta=1e-10)
  with self.assertRaises(ValueError):eval_initial(dict(a,post_action=1))
if __name__=="__main__":unittest.main()
