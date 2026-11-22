"""C19 scientific TDD, externally observable signed-value and source boundary."""
import unittest
import math
import numpy as np
from quartz.experiments.c19_signed_value import (
    train_signed, calibrate_signed, shadow_predict, score_signed)

class SignedValueTests(unittest.TestCase):
    def training(self):
        out=[]
        for i in range(50):
            g=(i%10)/10.0
            changed=int(i%3!=0)
            outcome=(0.15 if i%2 else -0.04)*changed
            out.append({"gap":g,"n_actions":2+i%3,
                        "label":changed,"signed_gain":outcome})
        return out

    def test_two_part_model_and_clamped_values(self):
        model=train_signed(self.training())
        self.assertEqual(len(model["q_coefficients"]),3)
        self.assertEqual(len(model["m_coefficients"]),3)
        self.assertGreater(model["n_train_flips"],5)
        for x in model["q_coefficients"]+model["m_coefficients"]:
            self.assertTrue(math.isfinite(x))
    def test_calibration_can_only_update_two_intercepts(self):
        model=train_signed(self.training())
        correction=calibrate_signed(model,self.training()[0:20])
        self.assertIn("q_offset",correction)
        self.assertIn("m_offset",correction)
        self.assertTrue(math.isfinite(correction["q_offset"]))
        self.assertTrue(math.isfinite(correction["m_offset"]))
    def test_generic_initial_only_shadow_and_cost_units(self):
        model=train_signed(self.training())
        row=[{"gap":0.1,"n_actions":2,"compute_kind":"beta_arms"}]
        result=shadow_predict(model,row,resource_price=0.004,
           cost_by_kind={"beta_arms":10.0,"finite_sensor":4.0})[0]
        self.assertTrue(0<result["flip_probability"]<1)
        self.assertTrue(-1<=result["conditional_signed_value"]<=1)
        self.assertAlmostEqual(result["gross_signed_value"],
                               result["flip_probability"]*result["conditional_signed_value"])
        self.assertAlmostEqual(result["predicted_net_value"],
                               result["gross_signed_value"]-0.04)
        self.assertIn(result["shadow_choice"],("COMPUTE","STOP"))
        with self.assertRaises(ValueError):
            shadow_predict(model,[dict(row[0],label=1)],resource_price=0.004,
              cost_by_kind={"beta_arms":10.0})
    def test_signed_gain_loss_and_y_zero_invariant(self):
        output=score_signed([0.1,0.1,0.1],[0.0,0.2,-0.1],[0,1,1])
        self.assertAlmostEqual(output["mse"],(0.01+0.01+0.04)/3)
        self.assertEqual(output["n"],3)
        self.assertEqual(output["flips"],2)
        with self.assertRaises(ValueError):
            score_signed([0.2],[0.1],[0])
    def test_price_units_and_positive_prediction(self):
        model=train_signed(self.training())
        row=[{"gap":0.2,"n_actions":4,"compute_kind":"beta_arms"}]
        p=shadow_predict(model,row,resource_price=0.001,cost_by_kind={"beta_arms":2.0})[0]
        q=shadow_predict(model,row,resource_price=0.000001,cost_by_kind={"beta_arms":2000.0})[0]
        self.assertAlmostEqual(p["predicted_net_value"],q["predicted_net_value"],places=12)
        with self.assertRaises(ValueError):
            shadow_predict(model,row,resource_price=0.001,cost_by_kind={"beta_arms":-1})
if __name__=="__main__":unittest.main()
