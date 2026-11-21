"""C5 actual public-interface finite metalevel decision tests. TDD first."""
import copy
import json
import unittest
from pathlib import Path
from quartz.experiments.metalevel_decision import FiniteProblem

REPO=Path(__file__).resolve().parents[1]
CASES={x["id"]:x for x in json.loads(
    (REPO/"configs/metalevel_reference_cases.v1.json").read_text())["cases"]}

class FiniteDecisionControlTests(unittest.TestCase):
    def model(self,name):
        return FiniteProblem.from_mapping(copy.deepcopy(CASES[name]))

    def test_heterogeneous_cost_counterexample(self):
        f=self.model("heterogeneous_complete_vs_partial")
        self.assertAlmostEqual(f.terminal_value(),0.5,places=11)
        self.assertAlmostEqual(f.risk(),0.5,places=11)
        self.assertAlmostEqual(f.evsi("partial_state_3"),0.2,places=11)
        self.assertAlmostEqual(f.evsi("reveal_all"),0.5,places=11)
        r=f.one_step()
        self.assertEqual(r["ratio_choice"],"partial_state_3")
        self.assertEqual(r["net_choice"],"reveal_all")
        self.assertAlmostEqual(r["net_gains"]["partial_state_3"],0.15,places=11)
        self.assertAlmostEqual(r["net_gains"]["reveal_all"],0.30,places=11)
        self.assertEqual(r["should_stop"],False)
        self.assertAlmostEqual(f.bellman(0)[0],0.5,places=11)
        for h in (1,2,3):
            v, choice=f.bellman(h)
            self.assertAlmostEqual(v,0.2,places=11)
            self.assertEqual(choice,"reveal_all")

    def test_information_cannot_create_positive_value_without_choice_change(self):
        f=self.model("one_action_always_dominant")
        self.assertAlmostEqual(f.risk(),0.0,places=12)
        self.assertAlmostEqual(f.evsi("perfect_state_reveal"),0.0,places=12)
        self.assertTrue(f.one_step()["should_stop"])
        self.assertEqual(f.bellman(2)[1],"STOP")

    def test_equal_cost_both_prioritize_more_information(self):
        f=self.model("equal_cost_complete_and_partial")
        self.assertEqual(f.one_step()["ratio_choice"],"reveal_all")
        self.assertEqual(f.one_step()["net_choice"],"reveal_all")

    def test_signed_payoff_and_noisy_sensor_valid(self):
        for name in ("nonidentity_signed_utility","two_state_noisy_sensor"):
            f=self.model(name)
            self.assertGreaterEqual(f.risk(),0)
            self.assertTrue(all(f.evsi(e.name)>=-1e-12 for e in f.experiments))
            self.assertGreaterEqual(f.bellman(0)[0]+1e-12,f.bellman(1)[0])
            self.assertGreaterEqual(f.bellman(1)[0]+1e-12,f.bellman(2)[0])

    def test_action_permutation_and_observation_impossible_case(self):
        row=copy.deepcopy(CASES["heterogeneous_complete_vs_partial"])
        f=FiniteProblem.from_mapping(row)
        v=f.risk()
        row["utilities"]=[[u[2],u[0],u[1]] for u in row["utilities"]]
        p=FiniteProblem.from_mapping(row)
        self.assertAlmostEqual(p.risk(),v,places=12)
        self.assertAlmostEqual(p.evsi("partial_state_3"),f.evsi("partial_state_3"),places=12)
        row["experiments"].append({"name":"always_same","cost":"1/15",
           "likelihood":[["1","0"],["1","0"],["1","0"]]})
        q=FiniteProblem.from_mapping(row)
        events=list(q.posterior_outcomes("always_same"))
        self.assertEqual(len(events),1)
        self.assertAlmostEqual(events[0][0],1,places=12)
        self.assertAlmostEqual(q.evsi("always_same"),0,places=12)

    def test_cost_units_and_calibrated_bellman(self):
        row=copy.deepcopy(CASES["heterogeneous_complete_vs_partial"])
        f=FiniteProblem.from_mapping(row)
        old=f.bellman(2)
        for exp in row["experiments"]:
            exp["cost"]=str(float(eval_fraction(exp["cost"]))*1000)
        row["lambda"]="0.001"
        g=FiniteProblem.from_mapping(row)
        new=g.bellman(2)
        self.assertAlmostEqual(old[0],new[0],places=11)
        self.assertEqual(old[1],new[1])

    def test_invalid_probability_or_negative_resource_cost_rejected(self):
        row=copy.deepcopy(CASES["heterogeneous_complete_vs_partial"])
        row["experiments"][0]["likelihood"][0]=["1/2","1/4"]
        with self.assertRaises(ValueError):
            FiniteProblem.from_mapping(row)
        row=copy.deepcopy(CASES["heterogeneous_complete_vs_partial"])
        row["experiments"][0]["cost"]="-1/20"
        with self.assertRaises(ValueError):
            FiniteProblem.from_mapping(row)
        row=copy.deepcopy(CASES["heterogeneous_complete_vs_partial"])
        row["posterior"][0]="NaN"
        with self.assertRaises(ValueError):
            FiniteProblem.from_mapping(row)

def eval_fraction(value):
    from fractions import Fraction
    return Fraction(value)

if __name__=="__main__":
    unittest.main()
