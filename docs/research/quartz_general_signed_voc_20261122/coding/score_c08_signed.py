#!/usr/bin/env python3
"""C8 prespecified heldout signed-value/physical-cost SHADOW counterfactual study.
All TRAIN/CAL results are frozen/published before this TEST score.
No live runtime policy is changed: every extra observation was actually produced.
"""
from pathlib import Path
import hashlib,json,random,math,statistics,subprocess
from collections import defaultdict
from quartz.experiments.c19_signed_value import shadow_predict,score_signed
from quartz.experiments.c18_observation import score_predictions
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
MODEL=HOME/"SHADOW_SIGNED_MODEL_FROZEN.json"
TEST=HOME/"evidence/C08_TEST/RAW_SIGNED_CALCULATIONS.jsonl"
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def check_source():
 if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT).strip():
    raise RuntimeError("heldout scoring requires all source and model actually committed/published")
 head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
 remote=subprocess.check_output(["git","ls-remote","--heads","origin",branch],cwd=ROOT,text=True).split()[0]
 assert head==remote
 return head
def avg(nums):
 return sum(nums)/len(nums)
def grouped_ci(rows, paired_deltas, seed):
 groups=defaultdict(list)
 for i,r in enumerate(rows):groups[r["group_id"]].append(float(paired_deltas[i]))
 assert len(groups)==24, len(groups)
 # This is descriptive independent-PARENT bootstrap, not finite-sample coverage.
 rng=random.Random(seed)
 keys=sorted(groups)
 byparent={g:avg(groups[g]) for g in keys}
 sample=[]
 for _ in range(500):
  sample.append(avg([byparent[rng.choice(keys)] for _ in keys]))
 sample.sort()
 return {"n_parent_groups":len(keys),"n_resamples":500,
   "mean_parent_group_delta":avg(list(byparent.values())),
   "descriptive_percentile95":[sample[12],sample[487]]}
def prepare_forecasts(rows,record):
 core=record["trained_model"]
 cal=record["calibration_only_intercepts"]
 price=record["fixed_primary_price_utility_per_microsecond"]
 costs=record["predicted_incremental_cost_us_by_compute_kind"]
 initial=[{"gap":r["gap"],"n_actions":r["n_actions"],
           "compute_kind":r["compute_kind"]} for r in rows]
 forecasts=shadow_predict(core,initial,resource_price=price,cost_by_kind=costs,calibration=cal)
 return forecasts,initial
def compare_native(rows,initial,forecast,model):
 native=ROOT/"target/release/quartz_c19_shadow"
 req={"q_coefficients":model["trained_model"]["q_coefficients"],
   "m_coefficients":model["trained_model"]["m_coefficients"],
   "q_offset":model["calibration_only_intercepts"]["q_offset"],
   "m_offset":model["calibration_only_intercepts"]["m_offset"],
   "resource_price":model["fixed_primary_price_utility_per_microsecond"],
   "rows":[{"gap":r["gap"],"n_actions":r["n_actions"],
     "estimated_cost_us":model["predicted_incremental_cost_us_by_compute_kind"][r["compute_kind"]]} for r in initial]}
 reqfile=HOME/"evidence/C08_TEST/NO_LABEL_NATIVE_SHADOW_REQUEST.json"
 reqfile.write_text(json.dumps(req,sort_keys=True,indent=2)+"\n")
 result=json.loads(subprocess.check_output([str(native),str(reqfile)],text=True))
 assert result["no_search_or_decision_mutation"] is True
 assert result["true_utility_input_used"] is False
 assert len(result["predictions"])==len(forecast)
 differences=[]
 for na,py in zip(result["predictions"],forecast):
  assert na["shadow_choice"]==py["shadow_choice"]
  for key in ("flip_probability","conditional_signed_value","gross_signed_value",
             "predicted_net_value","estimated_incremental_cost_us"):
   differences.append(abs(na[key]-py[key]))
 assert max(differences)<=1e-10
 return max(differences),sha(native),sha(reqfile)
def main():
 head=check_source()
 source=json.loads((HOME/"SOURCE_WORLDS_FROZEN.json").read_text())
 model=json.loads(MODEL.read_text())
 assert model["TEST_labels_seen_or_used"] is False
 assert model["source_world_SHA256"]==sha(HOME/"SOURCE_WORLDS_FROZEN.json")
 assert source["experiment_primary_utility_per_microsecond_price"]==0.004
 rows=[json.loads(s) for s in TEST.read_text().splitlines() if s.strip()]
 assert len(rows)==144 and all(r["split"]=="test" for r in rows)
 assert len({r["group_id"] for r in rows})==48
 assert not ({r["group_id"] for r in rows if r["family"]=="beta_arms"}
             &{r["group_id"] for r in rows if r["family"]=="finite_sensor"})
 assert all(r["label"]==int(r["pre_action"]!=r["post_action"]) and
     r["actual_extra_cost_us"]>0 for r in rows)
 forecast,initial=prepare_forecasts(rows,model)
 diff,native_sha,request_sha=compare_native(rows,initial,forecast,model)
 constant_p=model["TRAIN_global_constant_probability"]
 global_m=model["TRAIN_global_conditional_signed_gain_if_flip"]
 global_z=model["TRAIN_global_constant_signed_gain"]
 res={}
 prices=model["fixed_price_sensitivity"]
 assert prices==[0.001,0.004,0.01]
 for family in ("beta_arms","finite_sensor"):
  ix=[i for i,r in enumerate(rows) if r["family"]==family]
  src=[rows[i] for i in ix]
  p=[forecast[i] for i in ix]
  y=[r["label"] for r in src]
  z=[r["signed_gain"] for r in src]
  estimated=[x["gross_signed_value"] for x in p]
  qonly=[x["flip_probability"]*global_m for x in p]
  zeros=[0.0]*len(src)
  mse={
   "TRAIN_constant":score_signed([global_z]*len(src),z,y),
   "flip_probability_times_TRAIN_conditional_mean":score_signed(qonly,z,y),
   "two_part_signed":score_signed(estimated,z,y)}
  proper={
   "TRAIN_constant_q":score_predictions([constant_p]*len(src),y),
   "two_part_calibrated_q":score_predictions([x["flip_probability"] for x in p],y)}
  cost=[float(r["actual_extra_cost_us"]) for r in src]
  assert all(c>0 for c in cost)
  utility_summary={
   "n_actual_computations":len(src),
   "n_independent_parent_groups":len(set(r["group_id"] for r in src)),
   "n_observed_flips":sum(y),
   "n_positive_signed_utility":sum(v>0 for v in z),
   "n_negative_signed_utility":sum(v<0 for v in z),
   "mean_true_signed_gain_if_always":avg(z),
   "mean_measured_extra_wall_us":avg(cost),
   "mean_predicted_extra_wall_us_from_TRAIN":avg([x["estimated_incremental_cost_us"] for x in p]),
   "probability_calibration":proper,
   "signed_gain_prediction":mse,
   "prices":{}}
  for price in prices:
    always=[g-price*c for g,c in zip(z,cost)]
    # All actual additional computations executed. These are *offline*
    # counterfactual policy-accounting outcomes; hardware time not saved.
    qchoose=[x["flip_probability"]>=0.20 for x in p]
    choose=[x["gross_signed_value"]-price*x["estimated_incremental_cost_us"]>0 for x in p]
    qresult=[v if active else 0.0 for v,active in zip(always,qchoose)]
    sresult=[v if active else 0.0 for v,active in zip(always,choose)]
    bypol={"NEVER":[0.0]*len(src),"ALWAYS":always,
            "FLIP_ONLY_Q_GE_0.20":qresult,"SIGNED_NET_SHADOW":sresult}
    entry={
     "mean_paid_net_utility_gain_vs_STOP":{name:avg(v) for name,v in bypol.items()},
     "shadow_compute_proposals":sum(choose),
     "shadow_avoided_us_accounting_not_executed_savings":
       sum(c for c,sel in zip(cost,choose) if not sel),
     "shadow_missed_positive_net_events":
       sum(not s and v>0 for s,v in zip(choose,always)),
     "shadow_computed_negative_net_events":
       sum(s and v<0 for s,v in zip(choose,always)),
     "paired_CI_SHADOW_minus_ALWAYS":grouped_ci(src,[s-a for s,a in zip(sresult,always)],
           19120000+int(price*1e6)+len(src)),
     "paired_CI_SHADOW_minus_NEVER":grouped_ci(src,sresult,
           19121000+int(price*1e6)+len(src)),
     "paired_CI_FLIP_ONLY_minus_ALWAYS":grouped_ci(src,[s-a for s,a in zip(qresult,always)],
           19122000+int(price*1e6)+len(src))}
    utility_summary["prices"][str(price)]=entry
  res[family]=utility_summary
 report={"schema":"quartz_general_c19_C08_heldout_signed_value_v1",
 "test_git_source_published_before_evaluation":head,
 "two_domain_source_SHA256":sha(HOME/"SOURCE_WORLDS_FROZEN.json"),
 "model_SHA256_frozen_before_TEST":sha(MODEL),
 "test_raw_SHA256":sha(TEST),
 "native_shadow_readonly_compiled_SHA256":native_sha,
 "native_initial_only_request_SHA256":request_sha,
 "n_test_parent_groups_total":48,"n_actual_utility_scored_extra_computation_paths":len(rows),
 "max_Rust_Python_numeric_error":diff,
 "TEST_fitted_any_coefficients":False,
 "NO_active_search_computation_or_stop_policy":True,
 "offline_utility_known_from_task_law_not_from_MCTS":True,
 "all_additional_computations_were_actually_executed":True,
 "avoided_us_only_counterfactual_accounting_not_hardware_measured_saving":True,
 "primary_lambda_utility_per_measured_us":0.004,
 "scientific_scope":"two synthetic-known-utility task families with actual stochastic computations, not real neural MCTS or universal transfer",
 "heldout_domains":res}
 out=HOME/"evidence/C08_TEST/C08_HELDOUT_SIGNED_RESULTS.json"
 out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 compact={f:{"flip":r["n_observed_flips"],"n":r["n_actual_computations"],
      "observed_mean_z":r["mean_true_signed_gain_if_always"],
      "signed_mse":r["signed_gain_prediction"]["two_part_signed"]["mse"],
      "constant_mse":r["signed_gain_prediction"]["TRAIN_constant"]["mse"],
      "net_primary":r["prices"]["0.004"]["mean_paid_net_utility_gain_vs_STOP"],
      "CI_shadow_minus_always":r["prices"]["0.004"]["paired_CI_SHADOW_minus_ALWAYS"]["descriptive_percentile95"]}
     for f,r in res.items()}
 print("C19_HELDOUT",json.dumps({"scope":report["scientific_scope"],
   "n_parent_groups":48,"n_records":len(rows),
   "native_vs_python_maxerr":diff,"domains":compact},sort_keys=True))
if __name__=="__main__":main()
