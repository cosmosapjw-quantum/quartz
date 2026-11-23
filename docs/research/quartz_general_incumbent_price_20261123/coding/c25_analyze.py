"""Frozen C25 source-paired utility, wall-cost, charged-work and predictor analysis.

Nine repeats are reduced to one median per source/policy/budget before averaging
or paired bootstrap. Sensitivity prices revalue the same fixed policy decisions.
"""
import argparse
import json
import math
import pathlib
import statistics
import sys
from fractions import Fraction as F
import numpy as np
from c25_io import manifest_files, rows, semantic_signature, sha

POLICIES=("PLAIN_STOP","PLAIN_ALWAYS","FAST_BLOCK","RETAIN_BLOCK","RETAIN_WORK","RETAIN_TIME")
RETAIN=("RETAIN_BLOCK","RETAIN_WORK","RETAIN_TIME")
COSTS=("total","validation","initial","solve","bounds")
COUNTS=("solver_matvecs","bound_evaluations","fast_evaluations","outward_evaluations","ordinary_evaluations")
WORKS=("operator_response_passes","ordinary_stencil_evaluations","interval_stencil_evaluations",
       "total_stencil_evaluations","charged_cell_stencils")

def parent_metrics(truth,envelopes,cfg):
    parents=[];seen=set()
    for env in envelopes:
        ident=env["source_id"];budget=env["additional"];key=(ident,budget)
        assert key not in seen,"duplicate source/budget"
        seen.add(key);oracle=truth[ident];q=list(map(F,oracle["u_exact"]))
        grouped={p:[] for p in POLICIES}
        for row in env["native"]["results"]:
            grouped[row["policy"]].append(row)
        record={"id":ident,"additional":budget,"side":oracle["spec"]["side"],"rho":oracle["spec"]["rho"],
                "gap":oracle["spec"]["target_gap"],"stratum":oracle["spec"]["stratum"],"policies":{}}
        for name,rr in grouped.items():
            assert len(rr)==cfg["repeats"] and {r["rep"] for r in rr}==set(range(cfg["repeats"])),"repeat coverage"
            assert len({semantic_signature(r) for r in rr})==1,"scientific outcome changed across timing repetitions"
            first=rr[0];meta=first["c25"];u=q[first["selected"]];initial=q[first["initial_selected"]]
            regret=max(q)-u;ordinary=q[meta["final_ordinary_selected"]]
            value={"selected":first["selected"],"initial_selected":first["initial_selected"],
                   "certified":first["certified"],"weakly_optimal":meta["weakly_optimal"],
                   "price_stop_certified":meta["price_stop_certified"],"reason":first["reason"],
                   "utility":float(u),"utility_exact":str(u),"regret":float(regret),
                   "regret_exact":str(regret),"wrong_decision":regret>0,
                   "gain":float(u-initial),"gain_exact":str(u-initial),
                   "iterations":sum(first["iterations"]),"iterations_by_action":first["iterations"],
                   "residual_goal_met":first["residual_goal_met"],"all_times_us":[r["total_us"] for r in rr],
                   "executed_blocks":meta["executed_blocks"],"final_gain_upper":meta["final_gain_upper"],
                   "final_ordinary_selected":meta["final_ordinary_selected"],
                   "final_ordinary_utility_exact":str(ordinary),
                   "missed_improvement_vs_own_final_ordinary_exact":str(max(F(0),ordinary-u)),
                   "protected_utility_vs_own_final_ordinary_exact":str(max(F(0),u-ordinary))}
            for cost in COSTS:
                values=[r[cost+"_us"] for r in rr]
                assert all(math.isfinite(v) and v>=0 for v in values)
                value["median_"+cost+"_us"]=statistics.median(values)
            if name in RETAIN:
                continuation=[r["c25"]["continuation_us"] for r in rr]
                assert all(math.isfinite(v) and v>=0 for v in continuation)
                value["median_continuation_us"]=statistics.median(continuation)
                blocks=value["executed_blocks"]
                value["mean_continuation_per_block_us"]=statistics.mean(t/blocks for t in continuation) if blocks else None
            else:
                value["median_continuation_us"]=None
                value["mean_continuation_per_block_us"]=None
            for counter in COUNTS:
                value[counter]=first[counter]
            assert value["bound_evaluations"]==value["fast_evaluations"]+value["outward_evaluations"]
            value["operator_response_passes"]=value["solver_matvecs"]+value["bound_evaluations"]+value["ordinary_evaluations"]
            value["ordinary_stencil_evaluations"]=value["solver_matvecs"]+value["fast_evaluations"]+value["ordinary_evaluations"]+value["outward_evaluations"]
            value["interval_stencil_evaluations"]=value["outward_evaluations"]
            value["total_stencil_evaluations"]=value["ordinary_stencil_evaluations"]+value["interval_stencil_evaluations"]
            value["charged_cell_stencils"]=meta["charged_cell_stencils"]
            assert value["charged_cell_stencils"]==record["side"]**2*value["total_stencil_evaluations"]
            record["policies"][name]=value
        stop=record["policies"]["PLAIN_STOP"];always=record["policies"]["PLAIN_ALWAYS"];fast=record["policies"]["FAST_BLOCK"]
        for name,value in record["policies"].items():
            utility=F(value["utility_exact"]);du=utility-F(stop["utility_exact"])
            dt=F(value["median_total_us"])-F(stop["median_total_us"])
            value["actual_early_stop"]=value["solver_matvecs"]<always["solver_matvecs"]
            value["solver_matvecs_saved_vs_PLAIN_ALWAYS"]=always["solver_matvecs"]-value["solver_matvecs"]
            value["readout_calls_saved_vs_PLAIN_ALWAYS"]=(always["bound_evaluations"]+always["ordinary_evaluations"])-(value["bound_evaluations"]+value["ordinary_evaluations"])
            value["charged_cell_stencils_saved_vs_PLAIN_ALWAYS"]=always["charged_cell_stencils"]-value["charged_cell_stencils"]
            value["utility_difference_vs_PLAIN_STOP_exact"]=str(du)
            value["additional_time_vs_PLAIN_STOP_us"]=float(dt)
            value["break_even_price_vs_PLAIN_STOP"]=float(du/dt) if dt>0 else None
            value["break_even_price_vs_PLAIN_STOP_exact"]=str(du/dt) if dt>0 else None
            value["utility_decrease_vs_PLAIN_STOP"]=du<0
            for label,base in (("PLAIN_ALWAYS",always),("FAST_BLOCK",fast)):
                missed=max(F(0),F(base["utility_exact"])-utility)
                value["missed_improvement_vs_"+label]=missed>0
                value["missed_improvement_vs_"+label+"_exact"]=str(missed)
            value["protected_plain_always_regression"]=F(always["utility_exact"])<F(stop["utility_exact"])<=utility
        parents.append(record)
    assert seen=={(ident,b) for ident in truth for b in cfg["additional"]},"source/budget population coverage"
    return parents

def predictor_diagnostics(group,predictor):
    if predictor is None:
        return {"status":"NO_FROZEN_PREDICTOR_FOR_THIS_SPLIT"}
    results=[]
    for side in sorted({p["side"] for p in group}):
        fit=predictor["by_side"][str(side)]["predicted_block_us"]
        active=[p for p in group if p["side"]==side and p["policies"]["RETAIN_BLOCK"]["mean_continuation_per_block_us"] is not None]
        observed=[p["policies"]["RETAIN_BLOCK"]["mean_continuation_per_block_us"] for p in active]
        errors=[fit-v for v in observed]
        results.append({"side":side,"predicted_block_us":fit,"active_parents":len(active),
                        "inactive_parents":sum(p["side"]==side for p in group)-len(active),
                        "median_observed_parent_mean_block_us":statistics.median(observed) if observed else None,
                        "mean_prediction_minus_observed_us":statistics.mean(errors) if errors else None,
                        "underprediction_parents":sum(e<0 for e in errors),"overprediction_parents":sum(e>0 for e in errors),
                        "equal_prediction_parents":sum(e==0 for e in errors),
                        "observations":[{"id":p["id"],"mean_block_us":v,"prediction_minus_observed_us":e}
                                        for p,v,e in zip(active,observed,errors)]})
    return {"status":"DESCRIPTIVE_NO_REFIT","policy":"RETAIN_BLOCK",
            "observation":"within-parent mean continuation_us/executed_blocks over repetitions",
            "interpretation":"Prediction error diagnostic, not a pathwise cost lower-bound test","by_side":results}

def summarize(parents,cfg,split,predictor=None):
    rng=np.random.default_rng(cfg["bootstrap_seed"]);summaries=[]
    mean=lambda values:float(np.mean(values))
    for budget in cfg["additional"]:
        group=[p for p in parents if p["additional"]==budget];n=len(group);assert n>0
        indices=rng.integers(0,n,(cfg["bootstrap_resamples"],n))
        def ci(values):
            return [float(v) for v in np.percentile(np.asarray(values,dtype=float)[indices].mean(axis=1),[2.5,97.5])]
        def exact_mean(values):
            return sum(values,F(0))/n
        policies={}
        for name in POLICIES:
            vals=[p["policies"][name] for p in group]
            always=[p["policies"]["PLAIN_ALWAYS"] for p in group]
            stop=[p["policies"]["PLAIN_STOP"] for p in group]
            time_saved=[b["median_total_us"]-v["median_total_us"] for v,b in zip(vals,always)]
            du_stop=[F(v["utility_exact"])-F(s["utility_exact"]) for v,s in zip(vals,stop)]
            du_always=[F(v["utility_exact"])-F(b["utility_exact"]) for v,b in zip(vals,always)]
            dt_stop=[F(v["median_total_us"])-F(s["median_total_us"]) for v,s in zip(vals,stop)]
            dt_always=[F(v["median_total_us"])-F(b["median_total_us"]) for v,b in zip(vals,always)]
            regrets=[F(v["regret_exact"]) for v in vals]
            entry={"parents":n,"mean_parent_median_us":mean([v["median_total_us"] for v in vals]),
                   "mean_regret":float(exact_mean(regrets)),"mean_regret_exact":str(exact_mean(regrets)),
                   "wrong_decision_parents":sum(v["wrong_decision"] for v in vals),
                   "certified_parents":sum(v["certified"] for v in vals),
                   "weak_optimal_parents":sum(v["weakly_optimal"] is True for v in vals),
                   "certified_work_price_stop_parents":sum(v["price_stop_certified"] for v in vals),
                   "heuristic_time_price_stop_parents":sum(v["reason"]=="PRICE_TIME" for v in vals),
                   "reason_counts":{reason:sum(v["reason"]==reason for v in vals) for reason in sorted({v["reason"] for v in vals})},
                   "actual_early_stop_parents":sum(v["actual_early_stop"] for v in vals),
                   "utility_decrease_vs_PLAIN_STOP_parents":sum(v["utility_decrease_vs_PLAIN_STOP"] for v in vals),
                   "protected_plain_always_regression_parents":sum(v["protected_plain_always_regression"] for v in vals),
                   "mean_solver_iterations":mean([v["iterations"] for v in vals]),
                   "solver_matvecs_saved":sum(v["solver_matvecs_saved_vs_PLAIN_ALWAYS"] for v in vals),
                   "readout_calls_saved":sum(v["readout_calls_saved_vs_PLAIN_ALWAYS"] for v in vals),
                   "charged_cell_stencils_saved":sum(v["charged_cell_stencils_saved_vs_PLAIN_ALWAYS"] for v in vals),
                   "mean_time_saved_vs_PLAIN_ALWAYS_us":mean(time_saved),
                   "paired_parent_bootstrap_time_saved_vs_PLAIN_ALWAYS_95_us":ci(time_saved),
                   "mean_additional_time_vs_PLAIN_STOP_us":float(exact_mean(dt_stop)),
                   "mean_utility_difference_vs_PLAIN_STOP_exact":str(exact_mean(du_stop)),
                   "mean_utility_difference_vs_PLAIN_ALWAYS_exact":str(exact_mean(du_always)),
                   "residual_goal_verified_true_parents":sum(v["residual_goal_met"] is True for v in vals),
                   "residual_goal_verified_false_parents":sum(v["residual_goal_met"] is False for v in vals),
                   "residual_goal_unverified_parents":sum(v["residual_goal_met"] is None for v in vals),"priced":[]}
            for cost in COSTS:
                entry["mean_"+cost+"_us"]=mean([v["median_"+cost+"_us"] for v in vals])
            for counter in COUNTS+WORKS:
                entry["mean_"+counter]=mean([v[counter] for v in vals])
                entry["total_"+counter]=sum(v[counter] for v in vals)
            for base in ("PLAIN_ALWAYS","FAST_BLOCK"):
                entry["missed_improvement_vs_"+base+"_parents"]=sum(v["missed_improvement_vs_"+base] for v in vals)
                entry["mean_missed_improvement_vs_"+base+"_exact"]=str(exact_mean([F(v["missed_improvement_vs_"+base+"_exact"]) for v in vals]))
            entry["missed_improvement_vs_own_final_ordinary_parents"]=sum(F(v["missed_improvement_vs_own_final_ordinary_exact"])>0 for v in vals)
            entry["mean_missed_improvement_vs_own_final_ordinary_exact"]=str(exact_mean([F(v["missed_improvement_vs_own_final_ordinary_exact"]) for v in vals]))
            for price in cfg["prices"]:
                exact_price=F(str(price))
                net_stop=[d-exact_price*t for d,t in zip(du_stop,dt_stop)]
                net_always=[d-exact_price*t for d,t in zip(du_always,dt_always)]
                entry["priced"].append({"lambda_utility_per_us":price,"lambda_exact":str(exact_price),
                    "mean_net_difference_vs_PLAIN_STOP":float(exact_mean(net_stop)),
                    "mean_net_difference_vs_PLAIN_STOP_exact":str(exact_mean(net_stop)),
                    "paired_parent_bootstrap_net_difference_vs_PLAIN_STOP_95":ci(list(map(float,net_stop))),
                    "positive_net_vs_PLAIN_STOP_parents":sum(v>0 for v in net_stop),
                    "negative_net_vs_PLAIN_STOP_parents":sum(v<0 for v in net_stop),
                    "mean_net_difference_vs_PLAIN_ALWAYS":float(exact_mean(net_always)),
                    "mean_net_difference_vs_PLAIN_ALWAYS_exact":str(exact_mean(net_always)),
                    "paired_parent_bootstrap_net_difference_vs_PLAIN_ALWAYS_95":ci(list(map(float,net_always)))})
            work_price=F(cfg["work_price_exact"])
            work_stop=[d-work_price*(v["charged_cell_stencils"]-s["charged_cell_stencils"]) for d,v,s in zip(du_stop,vals,stop)]
            work_always=[d-work_price*(v["charged_cell_stencils"]-b["charged_cell_stencils"]) for d,v,b in zip(du_always,vals,always)]
            entry["work_priced"]={"lambda_utility_per_cell_stencil_exact":str(work_price),
                "mean_net_difference_vs_PLAIN_STOP":float(exact_mean(work_stop)),
                "mean_net_difference_vs_PLAIN_STOP_exact":str(exact_mean(work_stop)),
                "paired_parent_bootstrap_net_difference_vs_PLAIN_STOP_95":ci(list(map(float,work_stop))),
                "mean_net_difference_vs_PLAIN_ALWAYS_exact":str(exact_mean(work_always)),
                "positive_net_vs_PLAIN_STOP_parents":sum(v>0 for v in work_stop),
                "negative_net_vs_PLAIN_STOP_parents":sum(v<0 for v in work_stop)}
            thresholds=[v["break_even_price_vs_PLAIN_STOP"] for v in vals if v["break_even_price_vs_PLAIN_STOP"] is not None]
            md_time=exact_mean(dt_stop);md_utility=exact_mean(du_stop)
            entry["break_even_vs_PLAIN_STOP"]={"role":"descriptive only; no policy/price/model fitting",
                "parents_with_positive_extra_time":len(thresholds),"parents_with_nonpositive_extra_time":n-len(thresholds),
                "median_parent_break_even_price":statistics.median(thresholds) if thresholds else None,
                "cohort_mean_break_even_price_exact":str(md_utility/md_time) if md_time>0 else None}
            policies[name]=entry
        subgroups=[]
        for field in ("side","rho","gap"):
            for level in sorted({p[field] for p in group}):
                subset=[p for p in group if p[field]==level];metrics={}
                for name in POLICIES:
                    vals=[p["policies"][name] for p in subset]
                    reg=sum((F(v["regret_exact"]) for v in vals),F(0))/len(vals)
                    metrics[name]={"mean_parent_median_us":mean([v["median_total_us"] for v in vals]),
                                   "mean_regret_exact":str(reg),"wrong_decision_parents":sum(v["wrong_decision"] for v in vals),
                                   "certified_parents":sum(v["certified"] for v in vals),
                                   "missed_improvement_vs_PLAIN_ALWAYS_parents":sum(v["missed_improvement_vs_PLAIN_ALWAYS"] for v in vals),
                                   "mean_charged_cell_stencils":mean([v["charged_cell_stencils"] for v in vals])}
                subgroups.append({"field":field,"value":level,"parents":len(subset),"policies":metrics})
        summaries.append({"additional":budget,"policies":policies,"subgroups":subgroups,
                          "predictor_diagnostics":predictor_diagnostics(group,predictor) if budget==32 else {"status":"PRIMARY_BUDGET_ONLY"}})
    primary=next(s for s in summaries if s["additional"]==cfg["primary_additional"])
    contrasts=[]
    selections=[(cfg["primary_policy"],cfg["primary_baseline"],"primary"),
                ("RETAIN_BLOCK","PLAIN_STOP","secondary"),("RETAIN_WORK","PLAIN_STOP","secondary"),
                ("RETAIN_TIME","PLAIN_ALWAYS","secondary"),("FAST_BLOCK","PLAIN_ALWAYS","secondary")]
    for name,baseline,role in selections:
        policy=primary["policies"][name];priced=next(p for p in policy["priced"] if p["lambda_utility_per_us"]==cfg["primary_price"])
        estimate=priced["mean_net_difference_vs_"+baseline];interval=priced["paired_parent_bootstrap_net_difference_vs_"+baseline+"_95"]
        contrasts.append({"role":role,"policy":name,"baseline":baseline,"additional":cfg["primary_additional"],
                          "price":cfg["primary_price"],"mean_net_difference":estimate,
                          "mean_net_difference_exact":priced["mean_net_difference_vs_"+baseline+"_exact"],
                          "paired_parent_bootstrap_net_difference_95":interval,
                          "positive_mean":estimate>0,"ci_strictly_positive":interval[0]>0,
                          "ci_strictly_negative":interval[1]<0})
    return {"schema":"quartz.c25.summary.v1","split":split,"primary_additional":cfg["primary_additional"],
            "primary_price":cfg["primary_price"],"primary_policy":cfg["primary_policy"],"primary_baseline":cfg["primary_baseline"],
            "unique_parents":len({p["id"] for p in parents}),"bootstrap_resamples":cfg["bootstrap_resamples"],
            "bootstrap_seed":cfg["bootstrap_seed"],"timing_statistic":"within-source median of nine physical runs, then equal-source mean",
            "resampling_unit":"paired independent source parent; shared indices across policy contrasts at each budget",
            "prices":cfg["prices"],"sensitivity_semantics":"revalue fixed primary-price decisions; no price reoptimization",
            "clock":"full native wall elapsed microseconds","contrasts":contrasts,"summaries":summaries,
            "counter_semantics":"actual typed solver/readout calls and n-scaled cell-stencil charges; unequal hardware costs are not equated",
            "interpretation":"Exact rational wrong counts/utility; finite paired percentile bootstrap; charged-work theorem and empirical wall-time result remain separate"}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split",choices=["TRAIN","CAL","TEST","UNIT"],default="TEST")
    ap.add_argument("--evidence-dir",type=pathlib.Path)
    ap.add_argument("--fixture",action="store_true",help="small analysis-only input; not a cohort claim")
    args=ap.parse_args()
    doc=pathlib.Path(__file__).resolve().parents[1];ev=(args.evidence_dir or doc/"evidence").resolve()
    cfg=json.loads((doc/"SOURCE_SPECS.json").read_text())
    oracle_path=ev/f"C25_ORACLE_{args.split}.jsonl.gz";timing_path=ev/f"C25_{args.split}_TIMING.manifest.json"
    summary_path=ev/f"C25_{args.split}_SUMMARY.json";parents_path=ev/f"C25_{args.split}_PARENT_METRICS.json"
    if summary_path.exists() or parents_path.exists():
        raise FileExistsError("preserved analysis output exists")
    inputs=[doc/"SOURCE_SPECS.json",oracle_path]+manifest_files(timing_path)
    if not args.fixture:
        verification=ev/f"C25_VERIFY_{args.split}_TIMING.json";verified=json.loads(verification.read_text())
        assert verified["status"]=="PASS" and verified["input_sha256"][timing_path.name]==sha(timing_path)
        assert verified["input_sha256"][oracle_path.name]==sha(oracle_path)
        inputs.append(verification)
    truth={}
    for row in rows(oracle_path):
        assert row["id"] not in truth,"duplicate oracle source"
        truth[row["id"]]=row
    if not args.fixture and args.split!="UNIT":
        assert truth.keys()=={s["id"] for s in cfg["sources"] if s["split"]==args.split}
    predictor=None
    if args.split in ("CAL","TEST"):
        predictor_path=ev/cfg["predictor"]["artifact"];predictor=json.loads(predictor_path.read_text());inputs.append(predictor_path)
    before={str(p.relative_to(ev)) if p.is_relative_to(ev) else p.name:sha(p) for p in inputs}
    parents=parent_metrics(truth,rows(timing_path),cfg)
    result=summarize(parents,cfg,args.split,predictor)
    assert before=={str(p.relative_to(ev)) if p.is_relative_to(ev) else p.name:sha(p) for p in inputs},"analysis inputs changed"
    result.update({"status":"FIXTURE_ONLY" if args.fixture else "VERIFIED_INPUTS_ANALYZED","argv":sys.argv,
                   "python":sys.version,"numpy":np.__version__,"input_sha256":before,"analyzer_sha256":sha(pathlib.Path(__file__))})
    with parents_path.open("x") as stream:
        stream.write(json.dumps(parents,indent=2,allow_nan=False)+"\n")
    result["parent_metrics_sha256"]=sha(parents_path)
    with summary_path.open("x") as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],"split":args.split,"unique_parents":result["unique_parents"],
                      "summary":str(summary_path),"contrasts":result["contrasts"]}))

if __name__=="__main__":
    main()
