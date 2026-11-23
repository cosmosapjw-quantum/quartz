"""Prespecified parent-level C24 costs, exact regrets and paired uncertainty.

Timing repetitions are reduced to one median per parent/policy/budget before any
between-parent mean or bootstrap. Prices and primary contrasts come from the
frozen source specification. Break-even prices are descriptive, never controls.
"""
import argparse
import gzip
import hashlib
import json
import math
import pathlib
import statistics
import sys
from fractions import Fraction as F
import numpy as np

POLICIES=("PLAIN_STOP","PLAIN_ALWAYS","FAST_ONCE","FAST_BLOCK","OUTWARD_BLOCK","RESIDUAL")
COSTS=("total","validation","initial","solve","bounds")
COUNTS=("solver_matvecs","bound_evaluations","fast_evaluations","outward_evaluations","ordinary_evaluations")

def read(path):
    with gzip.open(path,"rt") as stream:
        for line in stream:yield json.loads(line)

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def parent_metrics(truth,envelopes,cfg):
    parents=[];seen=set()
    for env in envelopes:
        ident=env["source_id"];budget=env["additional"];key=(ident,budget)
        assert key not in seen,"duplicate parent/budget"
        seen.add(key);oracle=truth[ident];q=list(map(F,oracle["u_exact"]))
        grouped={p:[] for p in POLICIES}
        for row in env["native"]["results"]:grouped[row["policy"]].append(row)
        record={"id":ident,"additional":budget,"side":oracle["spec"]["side"],"rho":oracle["spec"]["rho"],
            "gap":oracle["spec"]["target_gap"],"stratum":oracle["spec"]["stratum"],"policies":{}}
        for name,rr in grouped.items():
            assert len(rr)==cfg["repeats"] and {r["rep"] for r in rr}==set(range(cfg["repeats"])),"rep coverage"
            signatures={(r["selected"],r["initial_selected"],r["certified"],tuple(r["iterations"]),
                *(r[c] for c in COUNTS),r["residual_goal_met"]) for r in rr}
            assert len(signatures)==1,"scientific policy outcome changed across timing repetitions"
            first=rr[0];u=q[first["selected"]];initial=q[first["initial_selected"]];regret=max(q)-u
            value={"selected":first["selected"],"initial_selected":first["initial_selected"],
                "certified":first["certified"],"utility":float(u),"utility_exact":str(u),
                "regret":float(regret),"regret_exact":str(regret),"wrong_decision":regret>0,
                "gain":float(u-initial),"gain_exact":str(u-initial),"iterations":sum(first["iterations"]),
                "iterations_by_action":first["iterations"],"residual_goal_met":first["residual_goal_met"],
                "all_times_us":[r["total_us"] for r in rr]}
            for cost in COSTS:
                times=[r[cost+"_us"] for r in rr]
                assert all(math.isfinite(t) and t>=0 for t in times)
                value["median_"+cost+"_us"]=statistics.median(times)
            for counter in COUNTS:value[counter]=first[counter]
            assert value["bound_evaluations"]==value["fast_evaluations"]+value["outward_evaluations"]
            value["operator_response_passes"]=value["solver_matvecs"]+value["bound_evaluations"]+value["ordinary_evaluations"]
            value["ordinary_stencil_evaluations"]=value["solver_matvecs"]+value["fast_evaluations"]+value["ordinary_evaluations"]+value["outward_evaluations"]
            value["interval_stencil_evaluations"]=value["outward_evaluations"]
            value["total_stencil_evaluations"]=value["ordinary_stencil_evaluations"]+value["interval_stencil_evaluations"]
            record["policies"][name]=value
        for name,value in record["policies"].items():
            base=record["policies"]["PLAIN_ALWAYS"];stop=record["policies"]["PLAIN_STOP"]
            value["actual_early_stop"]=value["solver_matvecs"]<base["solver_matvecs"]
            value["solver_matvecs_saved_vs_PLAIN_ALWAYS"]=base["solver_matvecs"]-value["solver_matvecs"]
            du=F(value["utility_exact"])-F(stop["utility_exact"])
            dt=value["median_total_us"]-stop["median_total_us"]
            value["utility_difference_vs_PLAIN_STOP_exact"]=str(du)
            value["additional_time_vs_PLAIN_STOP_us"]=dt
            value["break_even_price_vs_PLAIN_STOP"]=float(du)/dt if dt>0 else None
            value["break_even_price_vs_PLAIN_STOP_exact"]=str(du/F(dt)) if dt>0 else None
        parents.append(record)
    assert seen=={(ident,b) for ident in truth for b in cfg["additional"]},"parent/budget population coverage"
    return parents

def summarize(parents,cfg,split):
    rng=np.random.default_rng(cfg["bootstrap_seed"]);summaries=[]
    mean=lambda values:float(np.mean(values))
    for budget in cfg["additional"]:
        group=[p for p in parents if p["additional"]==budget];n=len(group);assert n>0
        indices=rng.integers(0,n,(cfg["bootstrap_resamples"],n))
        def ci(values):
            return [float(x) for x in np.percentile(np.asarray(values,dtype=float)[indices].mean(axis=1),[2.5,97.5])]
        policies={}
        for name in POLICIES:
            vals=[p["policies"][name] for p in group]
            base=[p["policies"]["PLAIN_ALWAYS"] for p in group]
            stop=[p["policies"]["PLAIN_STOP"] for p in group]
            times=[b["median_total_us"]-v["median_total_us"] for v,b in zip(vals,base)]
            du_exact=[F(v["utility_exact"])-F(b["utility_exact"]) for v,b in zip(vals,base)]
            du=list(map(float,du_exact))
            extra_stop=[v["median_total_us"]-s["median_total_us"] for v,s in zip(vals,stop)]
            gain_stop_exact=[F(v["utility_exact"])-F(s["utility_exact"]) for v,s in zip(vals,stop)]
            gain_stop=list(map(float,gain_stop_exact))
            regret_exact=sum((F(v["regret_exact"]) for v in vals),F(0))/n
            entry={"parents":n,"mean_parent_median_us":mean([v["median_total_us"] for v in vals]),
                "mean_regret":float(regret_exact),"mean_regret_exact":str(regret_exact),
                "wrong_decision_parents":sum(v["wrong_decision"] for v in vals),
                "certified_parents":sum(v["certified"] for v in vals),
                "actual_early_stop_parents":sum(v["actual_early_stop"] for v in vals),
                "mean_solver_iterations":mean([v["iterations"] for v in vals]),
                "solver_iterations_saved":sum(b["iterations"]-v["iterations"] for v,b in zip(vals,base)),
                "solver_matvecs_saved":sum(v["solver_matvecs_saved_vs_PLAIN_ALWAYS"] for v in vals),
                "mean_time_saved_vs_PLAIN_ALWAYS_us":mean(times),
                "paired_parent_bootstrap_time_saved_vs_PLAIN_ALWAYS_95_us":ci(times),
                "mean_utility_difference_vs_PLAIN_ALWAYS":mean(du),
                "mean_utility_difference_vs_PLAIN_ALWAYS_exact":str(sum(du_exact,F(0))/n),
                "mean_additional_time_vs_PLAIN_STOP_us":mean(extra_stop),
                "mean_utility_difference_vs_PLAIN_STOP_exact":str(sum(gain_stop_exact,F(0))/n),
                "residual_goal_verified_true_parents":sum(v["residual_goal_met"] is True for v in vals),
                "residual_goal_verified_false_parents":sum(v["residual_goal_met"] is False for v in vals),
                "residual_goal_unverified_parents":sum(v["residual_goal_met"] is None for v in vals),
                "priced":[]}
            for cost in COSTS:entry["mean_"+cost+"_us"]=mean([v["median_"+cost+"_us"] for v in vals])
            for counter in COUNTS:
                entry["mean_"+counter]=mean([v[counter] for v in vals])
                entry["total_"+counter]=sum(v[counter] for v in vals)
            for counter in ("operator_response_passes","ordinary_stencil_evaluations","interval_stencil_evaluations","total_stencil_evaluations"):
                entry["mean_"+counter]=mean([v[counter] for v in vals])
                entry["total_"+counter]=sum(v[counter] for v in vals)
            thresholds=[v["break_even_price_vs_PLAIN_STOP"] for v in vals if v["break_even_price_vs_PLAIN_STOP"] is not None]
            delta_time_exact=sum(map(F,extra_stop),F(0))/n;delta_time=float(delta_time_exact);delta_utility=sum(gain_stop_exact,F(0))/n
            entry["break_even_vs_PLAIN_STOP"]={
                "role":"descriptive only; no price fitting, policy tuning, or checkpoint selection",
                "definition":"delta exact utility / delta parent-median elapsed us, only where delta time > 0",
                "parents_with_positive_extra_time":len(thresholds),
                "parents_with_nonpositive_extra_time":n-len(thresholds),
                "median_parent_break_even_price":statistics.median(thresholds) if thresholds else None,
                "parent_break_even_price_range":[min(thresholds),max(thresholds)] if thresholds else None,
                "cohort_mean_break_even_price":float(delta_utility)/delta_time if delta_time>0 else None,
                "cohort_mean_break_even_price_exact":str(delta_utility/delta_time_exact) if delta_time_exact>0 else None}
            for price in cfg["prices"]:
                vs_base=[d+price*t for d,t in zip(du,times)]
                vs_stop=[g-price*t for g,t in zip(gain_stop,extra_stop)]
                entry["priced"].append({"lambda_utility_per_us":price,
                    "mean_net_gain_from_initial":mean([v["gain"]-price*v["median_total_us"] for v in vals]),
                    "mean_net_difference_vs_PLAIN_ALWAYS":mean(vs_base),
                    "paired_parent_bootstrap_net_difference_vs_PLAIN_ALWAYS_95":ci(vs_base),
                    "mean_net_difference_vs_PLAIN_STOP":mean(vs_stop),
                    "paired_parent_bootstrap_net_difference_vs_PLAIN_STOP_95":ci(vs_stop)})
            policies[name]=entry
        subgroups=[]
        for field,levels in [("side",[8,32,96]),("rho",[1,16,64]),("gap",["wide","medium","tight","tie"])]:
            for level in levels:
                sub=[p for p in group if p[field]==level]
                if not sub:continue
                metrics={}
                for name in POLICIES:
                    vals=[p["policies"][name] for p in sub]
                    reg=sum((F(v["regret_exact"]) for v in vals),F(0))/len(sub)
                    metrics[name]={"mean_parent_median_us":mean([v["median_total_us"] for v in vals]),
                        "mean_regret":float(reg),"mean_regret_exact":str(reg),
                        "wrong_decision_parents":sum(v["wrong_decision"] for v in vals),
                        "certified":sum(v["certified"] for v in vals),
                        "actual_early_stop_parents":sum(v["actual_early_stop"] for v in vals),
                        "solver_matvecs_saved":sum(v["solver_matvecs_saved_vs_PLAIN_ALWAYS"] for v in vals),
                        "mean_bounds_us":mean([v["median_bounds_us"] for v in vals])}
                subgroups.append({"field":field,"value":level,"parents":len(sub),"policies":metrics})
        summaries.append({"additional":budget,"policies":policies,"subgroups":subgroups})
    primary=next(s for s in summaries if s["additional"]==cfg["primary_additional"])
    contrasts=[]
    for name,role in [(cfg["primary_policy"],"primary"),("FAST_ONCE","secondary"),("OUTWARD_BLOCK","secondary")]:
        p=primary["policies"][name]
        priced=next(x for x in p["priced"] if x["lambda_utility_per_us"]==cfg["primary_price"])
        contrasts.append({"role":role,"policy":name,"baseline":"PLAIN_ALWAYS",
            "additional":cfg["primary_additional"],"price":cfg["primary_price"],
            "mean_time_saved_us":p["mean_time_saved_vs_PLAIN_ALWAYS_us"],
            "paired_parent_bootstrap_time_saved_95_us":p["paired_parent_bootstrap_time_saved_vs_PLAIN_ALWAYS_95_us"],
            "mean_net_difference":priced["mean_net_difference_vs_PLAIN_ALWAYS"],
            "paired_parent_bootstrap_net_difference_95":priced["paired_parent_bootstrap_net_difference_vs_PLAIN_ALWAYS_95"]})
    return {"schema":"quartz.c24.summary.v1","split":split,
        "primary_additional":cfg["primary_additional"],"primary_price":cfg["primary_price"],
        "primary_policy":cfg["primary_policy"],"primary_baseline":"PLAIN_ALWAYS",
        "unique_parents":len({p["id"] for p in parents}),"bootstrap_resamples":cfg["bootstrap_resamples"],
        "bootstrap_seed":cfg["bootstrap_seed"],"timing_statistic":"median of nine physical runs per parent, then arithmetic parent mean",
        "resampling_unit":"paired independent parent, same resample indices for all policy contrasts within a budget",
        "prices":cfg["prices"],"clock":"wall elapsed microseconds","contrasts":contrasts,"summaries":summaries,
        "counter_semantics":"Ordinary and interval stencil calls are separate typed counts; full outward response has both. Counts do not imply equal cost; wall time is primary.",
        "interpretation":"Exact rational utility and wrong counts; finite paired percentile bootstrap; no multiplicity-adjusted confirmatory claim"}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split",choices=["TRAIN","CAL","TEST","UNIT"],default="TEST")
    ap.add_argument("--evidence-dir",type=pathlib.Path)
    ap.add_argument("--fixture",action="store_true",help="explicit small unverified analysis-only fixture; never a cohort claim")
    args=ap.parse_args()
    doc=pathlib.Path(__file__).resolve().parents[1];ev=args.evidence_dir or doc/"evidence"
    cfg=json.loads((doc/"SOURCE_SPECS.json").read_text())
    oracle_path=ev/f"C24_ORACLE_{args.split}.jsonl.gz";timing_path=ev/f"C24_{args.split}_TIMING.jsonl.gz"
    summary_path=ev/f"C24_{args.split}_SUMMARY.json";parents_path=ev/f"C24_{args.split}_PARENT_METRICS.json"
    if summary_path.exists() or parents_path.exists():raise FileExistsError("preserved analysis output exists")
    verification=ev/f"C24_VERIFY_{args.split}_TIMING.json"
    inputs=[doc/"SOURCE_SPECS.json",oracle_path,timing_path]
    if not args.fixture:
        v=json.loads(verification.read_text());assert v["status"]=="PASS"
        assert v["input_sha256"][timing_path.name]==sha(timing_path)
        assert v["input_sha256"][oracle_path.name]==sha(oracle_path)
        inputs.append(verification)
    truth={}
    for row in read(oracle_path):
        assert row["id"] not in truth,"duplicate oracle source ID"
        truth[row["id"]]=row
    if not args.fixture and args.split!="UNIT":
        expected={s["id"] for s in cfg["sources"] if s["split"]==args.split}
        assert truth.keys()==expected,"frozen analysis parent coverage"
    parents=parent_metrics(truth,read(timing_path),cfg)
    result=summarize(parents,cfg,args.split)
    result.update({"status":"FIXTURE_ONLY" if args.fixture else "VERIFIED_INPUTS_ANALYZED",
        "argv":sys.argv,"python":sys.version,"numpy":np.__version__,
        "input_sha256":{p.name:sha(p) for p in inputs},"analyzer_sha256":sha(pathlib.Path(__file__))})
    with parents_path.open("x") as out:out.write(json.dumps(parents,indent=2,allow_nan=False)+"\n")
    result["parent_metrics_sha256"]=sha(parents_path)
    with summary_path.open("x") as out:out.write(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],"split":args.split,"unique_parents":result["unique_parents"],
        "summary":str(summary_path),"parent_metrics":str(parents_path),"contrasts":result["contrasts"]}))
if __name__=="__main__":
    main()
