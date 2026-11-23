"""Behavioral C25 source, retained-decision and TRAIN predictor tests."""
import copy
import importlib.util
import json
import math
import pathlib
import sys
import unittest
from fractions import Fraction as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/research/quartz_general_incumbent_price_20261123"
CODE = DOC / "coding"
sys.path.insert(0, str(CODE))

def module(name):
    spec = importlib.util.spec_from_file_location(name, CODE / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

sources = module("c25_sources")
predictor = module("c25_fit_predictor")
verify = module("c25_verify")

def retention_fixture():
    b0 = [{"lo":.1,"hi":.6,"ell_hat":.15},{"lo":.05,"hi":.8,"ell_hat":.1},{"lo":0.0,"hi":.4,"ell_hat":.05}]
    b1 = [{"lo":.19,"hi":.5,"ell_hat":.18},{"lo":.15,"hi":.65,"ell_hat":.2},{"lo":.08,"hi":.12,"ell_hat":.09}]
    threshold = math.nextafter(12 * 2.0**-30, -math.inf)
    decisions = []
    for checkpoint, bounds, charge, reason in [(0,b0,60,"CONTINUE"),(1,b1,168,"BUDGET")]:
        decisions.append({"checkpoint":checkpoint,"incumbent_before":0,"incumbent_after":0,
            "gain_upper":math.nextafter(max(x["hi"] for x in bounds[1:])-bounds[0]["lo"],math.inf),
            "price_threshold":threshold,"cost_kind":"CELL_STENCIL","reason":reason,
            "switch_proof":None,"executed_blocks":checkpoint,"charged_cell_stencils":charge})
    row = {"policy":"RETAIN_WORK","rep":0,"selected":0,"initial_selected":0,
        "certified":False,"reason":"BUDGET","iterations":[12]*3,"solver_matvecs":36,
        "fast_evaluations":6,"outward_evaluations":0,"ordinary_evaluations":0,"bound_evaluations":6,
        "initial_bounds":b0,"final_bounds":b1,"decision_bounds":b1,
        "trace":[{"iterations":[4]*3,"fresh_bounds":b0,"bounds":b0},
                 {"iterations":[12]*3,"fresh_bounds":b1,"bounds":b1}],
        "c25":{"executed_blocks":1,"continuation_us":3.0,"charged_cell_stencils":168,
            "final_gain_upper":decisions[-1]["gain_upper"],"weakly_optimal":False,"price_stop_certified":False,"decisions":decisions}}
    return row, {"side":2,"rho":1}, [F(1,5),F(2,5),F(1,10)]

def predictor_fixture():
    specs = [{"id":name,"side":side,"split":"TRAIN"} for name,side in
             [("a",8),("b",8),("inactive",8),("c",32),("d",96)]]
    cfg = {"sources":specs, "repeats":2, "primary_additional":32,
           "predictor":{"fit_split":"TRAIN","fit_additional":32,"fit_policy":"RETAIN_BLOCK","group":"side"}}
    envelopes = []
    for name,ratios in [("a",[2.,10.]),("b",[20.,22.]),("inactive",[0.,0.]),("c",[5.,7.]),("d",[11.,13.])]:
        blocks = 0 if name=="inactive" else 2
        rows = [{"policy":"RETAIN_BLOCK","rep":rep,"c25":{"executed_blocks":blocks,"continuation_us":ratio*blocks}}
                for rep,ratio in enumerate(ratios)]
        envelopes.append({"source_id":name,"split":"TRAIN","mode":"timing","additional":32,
                          "native":{"results":rows}})
    return cfg,envelopes

class SourcesTest(unittest.TestCase):
    def test_frozen_c25_configuration_is_accepted(self):
        cfg=json.loads((DOC/"SOURCE_SPECS.json").read_text())
        old=json.loads((ROOT/"docs/research/quartz_general_cheap_certificates_20261123/SOURCE_SPECS.json").read_text())
        sources.validate_configuration(cfg,old)

    def test_unit_seeds_are_fresh_c25_seeds(self):
        self.assertEqual([x["seed"] for x in sources.UNIT_SPECS[:4]],
                         [25090001,25090002,25090003,25090004])

class PredictorTest(unittest.TestCase):
    def test_uses_parent_mean_then_side_median_and_ignores_inactive(self):
        cfg,envs=predictor_fixture()
        got=predictor.fit_predictor(cfg,envs)
        self.assertIn("by_side",got)
        self.assertEqual(got["by_side"]["8"]["predicted_block_us"],13.5)
        self.assertEqual(got["by_side"]["8"]["active_parent_ids"],["a","b"])
        self.assertEqual(got["by_side"]["32"]["predicted_block_us"],6.0)
        self.assertEqual(got["by_side"]["96"]["predicted_block_us"],12.0)

    def test_rejects_calibration_or_test_input(self):
        cfg,envs=predictor_fixture();envs[0]["split"]="TEST"
        with self.assertRaises((AssertionError,ValueError)):
            predictor.fit_predictor(cfg,envs)

    def test_rejects_side_without_active_training_parent(self):
        cfg,envs=predictor_fixture()
        for env in envs:
            if env["source_id"]=="d":
                for row in env["native"]["results"]:
                    row["c25"].update(executed_blocks=0,continuation_us=0.)
        with self.assertRaises((AssertionError,ValueError)):
            predictor.fit_predictor(cfg,envs)

class RetentionTest(unittest.TestCase):
    def test_valid_retained_trace_is_accepted(self):
        row,task,q=retention_fixture()
        counts={}
        verify.verify_retention(row,task,q,8,1.0,counts)
        self.assertEqual(counts["retention_checkpoint_gain_checks"],2)
        self.assertEqual(counts["admitted_blocks_with_fresh_readout_floor"],1)

    def check_rejected(self, mutate):
        row,task,q=retention_fixture();mutate(row)
        with self.assertRaises((AssertionError,ValueError)):
            verify.verify_retention(row,task,q,8,1.0,{})

    def test_rejects_uncertified_incumbent_switch(self):
        def mutate(row):
            d=row["c25"]["decisions"][0]
            d.update(incumbent_after=1,switch_proof={"from":0,"to":1,
                "challenger_lower":.05,"incumbent_upper":.6})
        self.check_rejected(mutate)

    def test_rejects_understated_gain_bound(self):
        self.check_rejected(lambda row:row["c25"]["decisions"][0].update(gain_upper=0.0))

    def test_rejects_missing_fresh_readout_charge(self):
        self.check_rejected(lambda row:row["c25"]["decisions"][1].update(charged_cell_stencils=60))

    def test_rejects_false_positive_price_stop(self):
        self.check_rejected(lambda row:row["c25"]["decisions"][0].update(reason="PRICE_WORK"))

    def test_rejects_widened_historical_intersection(self):
        def mutate(row):
            row["trace"][1]["bounds"][1]["hi"]=.9
        self.check_rejected(mutate)


class AnalysisTest(unittest.TestCase):
    def fixture(self):
        policies=["PLAIN_STOP","PLAIN_ALWAYS","FAST_BLOCK","RETAIN_BLOCK","RETAIN_WORK","RETAIN_TIME"]
        cfg={"additional":[32],"repeats":9,"bootstrap_seed":2500923,"bootstrap_resamples":2000,
             "primary_additional":32,"primary_policy":"RETAIN_TIME","primary_baseline":"PLAIN_STOP",
             "primary_price":1e-6,"primary_price_exact":"1/1000000","prices":[0,1e-8,1e-7,1e-6,1e-5],
             "work_price_exact":"1/1073741824","policies":policies}
        tiny=F(1,10**400);truth={};envs=[]
        for ident,offset in [("a",0.),("b",10.)]:
            truth[ident]={"id":ident,"u_exact":["0",str(tiny),"0"],
                "spec":{"side":8,"rho":1,"target_gap":"tight","stratum":0}}
            rr=[]
            for p in policies:
                for rep,t in enumerate([1,9,2,8,3,7,4,6,5]):
                    retain=p.startswith("RETAIN")
                    row={"policy":p,"rep":rep,"selected":0 if p=="PLAIN_STOP" else 1,
                         "initial_selected":0,"certified":False,"reason":"BUDGET",
                         "total_us":t+offset+(0 if p=="PLAIN_STOP" else 3),
                         "validation_us":.1,"initial_us":1.,"solve_us":2.,"bounds_us":2.,
                         "iterations":[2,2,2],"solver_matvecs":6,"fast_evaluations":0 if p.startswith("PLAIN") else 6,
                         "outward_evaluations":0,"ordinary_evaluations":3 if p=="PLAIN_STOP" else 6 if p=="PLAIN_ALWAYS" else 0,
                         "bound_evaluations":0 if p.startswith("PLAIN") else 6,"residual_goal_met":None,
                         "c25":{"executed_blocks":1,"continuation_us":2. if retain else None,
                                "charged_cell_stencils":64*(9 if p=="PLAIN_STOP" else 12),
                                "final_gain_upper":1. if retain else None,"weakly_optimal":False if retain else None,
                                "final_ordinary_selected":1,"price_stop_certified":False}}
                    rr.append(row)
            envs.append({"source_id":ident,"additional":32,"mode":"timing","split":"UNIT","native":{"results":rr}})
        return cfg,truth,envs

    def test_parent_medians_and_exact_wrong_counts_survive_underflow(self):
        analysis=module("c25_analyze");cfg,truth,envs=self.fixture()
        parents=analysis.parent_metrics(truth,envs,cfg)
        self.assertEqual(len(parents),2)
        self.assertEqual(parents[0]["policies"]["PLAIN_STOP"]["median_total_us"],5.)
        self.assertEqual(parents[1]["policies"]["RETAIN_TIME"]["median_total_us"],18.)
        self.assertTrue(all(p["policies"]["PLAIN_STOP"]["wrong_decision"] for p in parents))

    def test_primary_is_time_minus_stop_at_independent_parent_level(self):
        analysis=module("c25_analyze");cfg,truth,envs=self.fixture()
        result=analysis.summarize(analysis.parent_metrics(truth,envs,cfg),cfg,"UNIT")
        self.assertEqual(result["primary_baseline"],"PLAIN_STOP")
        self.assertEqual(result["unique_parents"],2)
        contrast=result["contrasts"][0]
        self.assertEqual(contrast["policy"],"RETAIN_TIME")
        self.assertEqual(contrast["baseline"],"PLAIN_STOP")
        self.assertAlmostEqual(contrast["mean_net_difference"],-3e-6,places=17)
        self.assertEqual(contrast["paired_parent_bootstrap_net_difference_95"],[-3e-6,-3e-6])

class TransportAndSourceTest(unittest.TestCase):
    def test_manufacture_arithmetic_is_byte_reused(self):
        import ast
        old=(ROOT/"docs/research/quartz_general_cheap_certificates_20261123/coding/c24_sources.py").read_text()
        new=(CODE/"c25_sources.py").read_text()
        def functions(text):
            return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
        a,b=functions(old),functions(new)
        for name in ("integer_stencil","shape","make_source"):
            self.assertEqual(a[name],b[name])

    def test_oracle_matches_hand_solved_identity_operator(self):
        task,truth=sources.make_source(sources.UNIT_SPECS[0])
        self.assertEqual(task["rho"],0)
        for b,u in zip(task["rhs"],truth["u_exact"]):
            self.assertEqual(sum(F(v)**2 for v in b)/F(task["scale"]),F(u))
        self.assertEqual(verify.verify_sources({task["id"]:task},{task["id"]:truth})["parents"],1)

    def test_decimal_wire_roundtrips_and_rejects_oracle(self):
        run=module("c25_run");task,_=sources.make_source(sources.UNIT_SPECS[1])
        wire=run.to_wire(task)
        self.assertEqual([[run.bits(float(v)) for v in b] for b in wire["rhs"]],
                         [[run.bits(v) for v in b] for b in task["rhs"]])
        task["u_exact"]=["0","0","0"]
        with self.assertRaises(ValueError):
            run.to_wire(task)

    def test_missing_oracle_action_is_rejected(self):
        task,truth=sources.make_source(sources.UNIT_SPECS[0]);truth["x_num"].pop()
        with self.assertRaises(AssertionError):
            verify.verify_sources({task["id"]:task},{task["id"]:truth})

    def test_historical_bad_source_is_unit_regression_only(self):
        cfg=json.loads((DOC/"SOURCE_SPECS.json").read_text())
        self.assertEqual(sources.UNIT_SPECS[-1]["id"],"c24_test_34_1")
        self.assertNotIn("c24_test_34_1",{s["id"] for s in cfg["sources"]})
        self.assertEqual(len(sources.configured_specs(cfg,"UNIT")),5)

if __name__=="__main__":
    unittest.main()
