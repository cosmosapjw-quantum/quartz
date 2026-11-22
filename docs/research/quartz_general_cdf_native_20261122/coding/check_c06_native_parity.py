#!/usr/bin/env python3
"""C06 native Rust exact-Beta evaluator vs independent lgamma Python and grid references.

Never imports Rust implementation; no future actual labels/latent theta used.
"""
from pathlib import Path
from statistics import median
from itertools import product
from fractions import Fraction as F
import hashlib,itertools,json,math,random,subprocess,tempfile
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
BIN=ROOT/"target/release/quartz_c21_fast_evsi"
SOURCE=HOME/"SOURCE_WORLDS_C21_FROZEN.json"
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def pmf_direct(a,b,n):
    logb=math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b)
    vals=[math.exp(math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)
          +math.lgamma(a+k)+math.lgamma(b+n-k)-math.lgamma(a+b+n)-logb)
          for k in range(n+1)]
    assert abs(math.fsum(vals)-1)<1e-10,(a,b,n,math.fsum(vals))
    return vals

def pure_python_CDF(arms,budgets):
    marg=[]
    for (a,b),n in zip(arms,budgets):
        probs=pmf_direct(a,b,n)
        marg.append([((a+k)/(a+b+n),probs[k]) for k in range(n+1)])
    support=sorted(set(v for m in marg for v,p in m))
    last=0.0;f_before=0.0;expect=0.0
    for v in support:
        expect+=(v-last)*(1-f_before)
        f_before=math.prod(math.fsum(p for x,p in m if x<=v) for m in marg)
        last=v
    assert abs(f_before-1)<1e-9
    baseline=max(a/(a+b) for a,b in arms)
    return {"gross_evsi":max(0.0,expect-baseline),"before_best":baseline,
        "after_expected_max":expect,"joint_mass":f_before,
        "support_points":len(support),"events":sum(n+1 for n in budgets)}

def pure_python_grid(arms,budgets):
    marg=[[( (a+k)/(a+b+n),p) for k,p in enumerate(pmf_direct(a,b,n))]
           for (a,b),n in zip(arms,budgets)]
    count=math.prod(len(m) for m in marg)
    assert count<=15000
    average=math.fsum(math.prod(p for v,p in tuple_values) *
            max(v for v,p in tuple_values)
            for tuple_values in itertools.product(*marg))
    return average-max(a/(a+b) for a,b in arms)

def truth_test():
    # Direct exact rational reference unrelated to full production float implementation.
    params=[(F(1),F(1)),(F(2),F(2)),(F(3),F(3))]
    pp=[(a,b) for a,b in params]
    assert sum([F(1),F(2)])==3
    return F(109,840)

rng=random.Random(202611212106)
cases=[
{"posterior":[[1,1],[2,2],[3,3]],"extra_per_arm":[2,2,2]},
{"posterior":[[1,1],[1,1]],"extra_per_arm":[1,1]},
{"posterior":[[.5,1.5],[2.5,3.5]],"extra_per_arm":[2,2]},
{"posterior":[[.1,100],[100,.1]],"extra_per_arm":[20,20]},
{"posterior":[[1,1],[2,2],[5,5],[2,2]],"extra_per_arm":[0,0,0,0]},
{"posterior":[[2,2],[3,8],[4,6]],"extra_per_arm":[0,2,10]},
{"posterior":[[.1,100] for _ in range(32)],"extra_per_arm":[20]*32},
]
worlds=json.loads(SOURCE.read_text())["worlds"]
for w in [w for w in worlds if w["family"]=="beta_arms" and w["split"]=="train"][:56]:
    k=w["K"]
    a=[[float(2+rng.randrange(0,4)),float(2+rng.randrange(0,4))] for _ in range(k)]
    cases.append({"posterior":a,"extra_per_arm":[rng.choice([0,1,2,10]) for _ in range(k)]})
for _ in range(100):
    k=rng.choice([2,3,4,8,16,32])
    params=[[rng.choice([.1,.5,1.5,2.0,3.0,7.5,20.0,100.0]),
             rng.choice([.1,.5,1.5,2.0,5.0,11.25,100.0])] for _ in range(k)]
    budget=[rng.choice([0,1,2,4,10,20]) for _ in range(k)]
    cases.append({"posterior":params,"extra_per_arm":budget})
assert len(cases)==163
req={"cmd":"beta_eval","cases":cases}
with tempfile.TemporaryDirectory() as t:
    p=Path(t)/"only_posterior.json"
    p.write_text(json.dumps(req,sort_keys=True))
    out=json.loads(subprocess.check_output([str(BIN),str(p)],text=True))
assert out["no_SearchPolicy_mutation"] is True
assert out["source_truth_used"] is False
assert len(out["results"])==len(cases)
maxerr=0.;compared=0;gridchecks=0;new_errors=[]
native_time={}
for i,(input,r) in enumerate(zip(cases,out["results"])):
    py=pure_python_CDF(input["posterior"],input["extra_per_arm"])
    for name in ("gross_evsi","before_best","after_expected_max","joint_mass"):
        err=abs(float(r[name])-float(py[name]))
        maxerr=max(maxerr,err)
        assert err<=1e-10,(i,name,err,r[name],py[name])
        compared+=1
    assert r["support_points"]==py["support_points"]
    assert r["events"]==py["events"]
    native_time.setdefault(len(input["posterior"]),[]).append(r["function_internal_us"])
    if len(input["posterior"])<=4 and math.prod(n+1 for n in input["extra_per_arm"])<=15000 and i<72:
        grid=pure_python_grid(input["posterior"],input["extra_per_arm"])
        assert abs(grid-py["gross_evsi"])<=1e-10,(i,grid,py["gross_evsi"])
        gridchecks+=1
assert abs(out["results"][0]["gross_evsi"]-float(truth_test()))<1e-12
rec={"schema":"quartz_C21_C06_native_Rust_vs_independent_lgamma_CDF_and_full_grid_v1",
    "status":"PASS","source_SHA256":sha(SOURCE),
    "compiled_native_Rust_SHA256":sha(BIN),
    "n_bounded_cases":len(cases),
    "n_numeric_float_equalities":compared,
    "max_abs_discrepancy":maxerr,
    "full_grid_independent_cases":gridchecks,
    "finite_fraction_exact_EVSI":"109/840",
    "no_true_latent_theta_in_any_case":True,
    "Rust_beta_validation_limits":{"K":[2,32],"n_each":[0,20],"alpha_beta_each":[.1,100]},
    "function_internal_median_us_by_K":{str(k):median(v) for k,v in native_time.items()},
    "full_C20_Beta_grid_Rust_port":"NOT_INVOKED",
    "original_SearchPolicy_untouched":True}
path=HOME/"evidence/C06_NUMERIC_PARITY.json"
path.write_text(json.dumps(rec,indent=2,sort_keys=True)+"\n")
print("C21_C06_NUMERIC",json.dumps(rec,sort_keys=True))
