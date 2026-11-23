"""C24 source characterization and behavioral contracts at real Python seams."""
import copy
import gzip
import importlib.util
import json
import math
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
CODE = ROOT / "docs/research/quartz_general_cheap_certificates_20261123/coding"
POLICIES = ["PLAIN_STOP", "PLAIN_ALWAYS", "FAST_ONCE", "FAST_BLOCK", "OUTWARD_BLOCK", "RESIDUAL"]

def module(name):
    spec = importlib.util.spec_from_file_location(name, CODE / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

sources = module("c24_sources")
verify = module("c24_verify")

def write_rows(path, rows):
    with gzip.open(path, "wt") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")

def exact_solve(task, b):
    side = task["side"]; n = side * side; rho = F(task["rho"])
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            adjacent = abs(i // side-j // side)+abs(i % side-j % side) == 1
            row.append(1+4*rho if i == j else -rho if adjacent else F(0))
        matrix.append(row+[F(b[i])])
    for i in range(n):
        pivot = matrix[i][i]
        matrix[i] = [v/pivot for v in matrix[i]]
        for j in range(n):
            if i != j:
                z = matrix[j][i]
                matrix[j] = [a-z*c for a,c in zip(matrix[j],matrix[i])]
    return [row[-1] for row in matrix]

def fixture():
    task, oracle = sources.make_source({"id":"c24_unit", "side":2, "rho":1,
        "target_gap":"medium", "seed":2400003, "split":"UNIT", "stratum":0})
    bound = [{"lo":0.0, "hi":1.0, "ell_hat":0.0} for _ in range(3)]
    results = []
    for name in POLICIES:
        plain = name in ["PLAIN_STOP","PLAIN_ALWAYS","FAST_ONCE"]
        its = [4]*3 if name == "PLAIN_STOP" else [12]*3
        fast = 3 if name == "FAST_ONCE" else 6 if name in ["FAST_BLOCK","RESIDUAL"] else 0
        outward = 6 if name == "OUTWARD_BLOCK" else 0
        ordinary = 3 if name in ["PLAIN_STOP","FAST_ONCE"] else 6 if name == "PLAIN_ALWAYS" else 0
        trace = [{"iterations":[4]*3, "fresh_bounds":copy.deepcopy(bound), "bounds":copy.deepcopy(bound)}]
        if name != "PLAIN_STOP":
            trace.append({"iterations":its, "fresh_bounds":copy.deepcopy(bound), "bounds":copy.deepcopy(bound)})
        results.append({"policy":name,"rep":0,"selected":0,"initial_selected":0,"certified":False,
            "reason":"budget_exhausted","iterations":its,"solver_matvecs":sum(its),
            "total_us":10.0,"validation_us":1.0,"initial_us":2.0,"solve_us":3.0,"bounds_us":2.0,
            "initial_bounds":copy.deepcopy(bound),"final_bounds":copy.deepcopy(bound),
            "decision_bounds":copy.deepcopy(bound),"trace":trace,"final_x":[[0.0]*4 for _ in range(3)],
            "fast_evaluations":fast,"outward_evaluations":outward,"ordinary_evaluations":ordinary,
            "bound_evaluations":fast+outward,"residual_relative":[1.0]*3,
            "residual_upper":None if plain else [1.0]*3,"residual_goal_met":None if plain else False})
    def bits(x):return struct.unpack("<Q",struct.pack("<d",x))[0]
    native = {"id":task["id"],"additional":8,"results":results,
        "input_rhs_bits":[[bits(x) for x in row] for row in task["rhs"]],
        "input_scale_bits":bits(task["scale"])}
    env = {"source_id":task["id"],"split":"UNIT","mode":"trace","additional":8,"native":native}
    return task,oracle,env

class SourceReuseTest(unittest.TestCase):
    def test_oracle_matches_independent_exact_inverse(self):
        task, oracle, _ = fixture()
        self.assertEqual(set(task),{"id","side","rho","scale","rhs"})
        for b,q in zip(task["rhs"],oracle["u_exact"]):
            x = exact_solve(task,b)
            self.assertEqual(sum(F(v)*w for v,w in zip(b,x))/F(task["scale"]),F(q))
        self.assertEqual(verify.verify_sources({task["id"]:task},{task["id"]:oracle})["parents"],1)

    def test_true_tie_has_distinct_rhs(self):
        task, oracle = sources.make_source({"id":"tie","side":3,"rho":64,"target_gap":"tie",
            "seed":2400023,"split":"UNIT","stratum":1})
        q=list(map(F,oracle["u_exact"])); best=[i for i,v in enumerate(q) if v==max(q)]
        self.assertEqual(len(best),2);self.assertNotEqual(task["rhs"][best[0]],task["rhs"][best[1]])

    def test_missing_oracle_action_is_rejected(self):
        # C23 zip truncation could accept a missing action silently.
        task, oracle, _ = fixture(); oracle["x_num"].pop()
        with self.assertRaises((AssertionError,ValueError)):
            verify.verify_sources({task["id"]:task},{task["id"]:oracle})

class NativeContractTest(unittest.TestCase):
    def check_rejected(self, change):
        task,oracle,env=fixture();change(env["native"]["results"])
        with tempfile.TemporaryDirectory() as tmp:
            path=pathlib.Path(tmp)/"trace.jsonl.gz";write_rows(path,[env])
            with self.assertRaises((AssertionError,ValueError)):
                verify.verify_native({task["id"]:task},{task["id"]:oracle},path)

    def test_plain_cannot_claim_a_certified_residual(self):
        def change(rr): rr[0].update(residual_upper=[1.0]*3,residual_goal_met=False)
        self.check_rejected(change)

    def test_noncertified_fallback_ell_identity_is_required(self):
        def change(rr): rr[2]["final_bounds"][0]["ell_hat"]=0.125
        self.check_rejected(change)

    def test_duplicate_policy_rep_is_rejected(self):
        self.check_rejected(lambda rr:rr.append(copy.deepcopy(rr[-1])))

    def test_fresh_residual_upper_covers_exact_stored_state(self):
        def change(rr): rr[-1].update(residual_upper=[0.0]*3,residual_goal_met=True)
        self.check_rejected(change)

class AnalysisContractTest(unittest.TestCase):
    def run_fixture(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);doc=pathlib.Path(tmp.name)
        (doc/"coding").mkdir();(doc/"evidence").mkdir()
        shutil.copyfile(CODE/"c24_analyze.py",doc/"coding/c24_analyze.py")
        # Exact positive regret underflows binary64; wrong decisions must stay exact.
        tiny=F(1,10**400)
        cfg={"additional":[32],"repeats":9,"bootstrap_seed":2400923,"bootstrap_resamples":2000,
            "primary_additional":32,"primary_policy":"FAST_BLOCK","primary_baseline":"PLAIN_ALWAYS",
            "primary_price":1e-6,"prices":[0,1e-8,1e-7,1e-6,1e-5],"policies":POLICIES,
            "sources":[{"id":"a","split":"UNIT"},{"id":"b","split":"UNIT"}]}
        (doc/"SOURCE_SPECS.json").write_text(json.dumps(cfg))
        truths=[];envs=[]
        for parent,offset in [("a",0),("b",10)]:
            truths.append({"id":parent,"u_exact":["0",str(tiny),"0"],"spec":{"side":8,"rho":1,
                "target_gap":"tight","stratum":0}})
            rr=[]
            for p in POLICIES:
                for rep,t in enumerate([1,9,2,8,3,7,4,6,5]):
                    rr.append({"policy":p,"rep":rep,"selected":0 if p=="PLAIN_STOP" else 1,
                        "initial_selected":0,"certified":p in ["FAST_ONCE","FAST_BLOCK","OUTWARD_BLOCK"],
                        "total_us":t+offset+(0 if p=="PLAIN_STOP" else 5 if p=="PLAIN_ALWAYS" else 3),
                        "bounds_us":2.0,"solve_us":2.0,"initial_us":1.0,"validation_us":0.1,
                        "iterations":[2,2,2],"solver_matvecs":6,
                        "bound_evaluations":0 if p.startswith("PLAIN") else 6,
                        "fast_evaluations":0 if p.startswith("PLAIN") or p=="OUTWARD_BLOCK" else 6,
                        "outward_evaluations":6 if p=="OUTWARD_BLOCK" else 0,
                        "ordinary_evaluations":3 if p=="PLAIN_STOP" else 6 if p=="PLAIN_ALWAYS" else 0,
                        "residual_goal_met":None if p.startswith("PLAIN") else False})
            envs.append({"source_id":parent,"additional":32,"split":"UNIT","mode":"timing",
                "native":{"id":parent,"additional":32,"results":rr}})
        ev=doc/"evidence";write_rows(ev/"C24_ORACLE_TEST.jsonl.gz",truths)
        write_rows(ev/"C24_TEST_TIMING.jsonl.gz",envs)
        completed=subprocess.run([sys.executable,str(doc/"coding/c24_analyze.py"),"--split","TEST","--fixture"],
            text=True,capture_output=True)
        self.assertEqual(completed.returncode,0,completed.stderr)
        return json.loads((ev/"C24_TEST_SUMMARY.json").read_text()),json.loads((ev/"C24_TEST_PARENT_METRICS.json").read_text())

    def test_exact_wrong_count_survives_float_underflow(self):
        summary,_=self.run_fixture()
        self.assertEqual(summary["summaries"][0]["policies"]["PLAIN_STOP"]["wrong_decision_parents"],2)

    def test_outward_counts_both_ordinary_and_interval_stencils(self):
        # Full outward response runs its interval stencil and the ordinary score stencil.
        summary,parents=self.run_fixture()
        value=parents[0]["policies"]["OUTWARD_BLOCK"]
        self.assertIn("ordinary_stencil_evaluations",value)
        self.assertEqual(value["ordinary_stencil_evaluations"],12)
        self.assertEqual(value["interval_stencil_evaluations"],6)
        self.assertEqual(value["total_stencil_evaluations"],18)
        self.assertEqual(value["operator_response_passes"],12)
        self.assertEqual(summary["summaries"][0]["policies"]["OUTWARD_BLOCK"]["mean_total_stencil_evaluations"],18)

    def test_break_even_is_descriptive_and_certificate_is_not_early(self):
        summary,parents=self.run_fixture();policy=summary["summaries"][0]["policies"]["FAST_BLOCK"]
        self.assertEqual(policy["certified_parents"],2)
        self.assertEqual(policy["actual_early_stop_parents"],0)
        self.assertEqual(policy["mean_parent_median_us"],13.0)
        self.assertIn("break_even_vs_PLAIN_STOP",policy)
        self.assertTrue(all("regret_exact" in p["policies"]["PLAIN_STOP"] for p in parents))

if __name__=="__main__":
    unittest.main()
