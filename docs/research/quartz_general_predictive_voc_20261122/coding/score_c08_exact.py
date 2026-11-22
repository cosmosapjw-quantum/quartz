#!/usr/bin/env python3
"""C20 prospectively frozen heldout exact Bayes predictive computation-value study.

Physical extra tasks are ALWAYS executed to obtain all potential outcomes.
All STOP/COMPUTE selection and time saving results are counterfactual SHADOW.
Exact model evaluator costs are separately measured and priced for feasibility.
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import hashlib,json,math,random,statistics,subprocess,time,tempfile
from quartz.experiments.c20_task_adapter import initial_only,eval_initial,resource_bucket
from quartz.experiments.c19_signed_value import shadow_predict as old_shadow_predict
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
SOURCE=HOME/"SOURCE_WORLDS_C20_FROZEN.json"
RAW=HOME/"evidence/C08_TEST/RAW_SIGNED_CALCULATIONS.jsonl"
COST_MODEL=HOME/"FROZEN_C20_RESOURCE_PRICE.json"
C19_MODEL=ROOT/"docs/research/quartz_general_signed_voc_20261122/SHADOW_SIGNED_MODEL_FROZEN.json"
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def assert_published():
 if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT).strip():
  raise RuntimeError("C20 heldout scorer must start from a clean published Git HEAD")
 head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
 remote=subprocess.check_output(["git","ls-remote","--heads","origin",branch],cwd=ROOT,text=True).split()[0]
 assert head==remote
 return head
def mean(x):return math.fsum(x)/len(x)
def group_ci(rows,deltas,seed):
 groups=defaultdict(list)
 for i,row in enumerate(rows):groups[row["group_id"]].append(float(deltas[i]))
 assert len(groups)==24
 parent=[mean(groups[g]) for g in sorted(groups)]
 rng=random.Random(seed)
 n=len(parent)
 draws=sorted(mean([parent[rng.randrange(n)] for _ in range(n)]) for _ in range(500))
 return {"n_independent_parent_groups":n,"n_bootstrap":500,
    "mean_delta_over_parent_groups":mean(parent),
    "descriptive_95pct_parent_bootstrap":[draws[12],draws[487]]}
def as_sensor_rust(initial):
 b=initial["prior"][:]
 L=initial["likelihood"]
 for y in initial["initial_observations"]:
  weights=[b[t]*L[t][y] for t in range(len(b))]
  norm=math.fsum(weights)
  b=[w/norm for w in weights]
 return {"prior":b,"likelihood":L,
    "utilities":initial["utilities"],"extra":initial["extra_observation_budget"]}
def main():
 head=assert_published()
 model=json.loads(COST_MODEL.read_text())
 c19=json.loads(C19_MODEL.read_text())
 assert model["source_bank_SHA256"]==sha(SOURCE)
 assert model["C19_historical_coarse_model_SHA256"]==sha(C19_MODEL)
 assert model["test_outcomes_or_labels_accessed"] is False
 assert model["fixed_price_sensitivity"]==[.001,.004,.01]
 rows=[json.loads(s) for s in RAW.read_text().splitlines() if s]
 assert len(rows)==144 and all(r["split"]=="test" for r in rows)
 assert len({r["group_id"] for r in rows})==48
 assert all(r["label"]==int(r["pre_action"]!=r["post_action"]) for r in rows)
 assert all(r["actual_extra_cost_us"]>0 for r in rows)
 # Only initial-only observation interface used to evaluate value of compute.
 forecasts=[];rust_cases=[];sensor_expected=[]
 for r in rows:
  inp=initial_only(r)
  prediction=eval_initial(inp)
  assert prediction["total_predictive_mass"]>=1-1e-9
  assert prediction["true_hidden_state_was_used"] is False
  if r["family"]=="finite_sensor":
   rust_cases.append(as_sensor_rust(inp));sensor_expected.append(prediction)
  bucket=resource_bucket(inp)
  summary=model["cost_and_predictor_overhead_medians_TRAIN_only"][bucket]
  c19input={"gap":r["gap"],"n_actions":r["n_actions"],"compute_kind":r["family"]}
  started=time.perf_counter_ns()
  coarse=old_shadow_predict(c19["trained_model"],[c19input],
         resource_price=.004,
         cost_by_kind=c19["predicted_incremental_cost_us_by_compute_kind"],
         calibration=c19["calibration_only_intercepts"])[0]
  old_eval_us=max(0,time.perf_counter_ns()-started)/1000
  forecasts.append({
    "group_id":r["group_id"],"family":r["family"],"rep":r["rep"],
    "resource_bucket":bucket,
    "evsi_exact":prediction["evsi"],
    "before_best":prediction["before_best"],
    "predictive_mass":prediction["total_predictive_mass"],
    "predictive_outcomes":prediction["outcome_count"],
    "predicted_extra_task_cost_us_TRAIN_median":summary["task_incremental_median_us"],
    "predicted_exact_evaluator_cost_us_TRAIN_median":summary["predictor_evaluation_median_us"],
    "measured_exact_evaluation_us":prediction["evaluation_us"],
    "predicted_coarse_evaluator_cost_us_TRAIN_median":summary["coarse_predictor_evaluation_median_us"],
    "measured_coarse_evaluation_us":old_eval_us,
    "C19_frozen_coarse_qm":coarse["gross_signed_value"],
    "no_true_hidden_state_or_future_action_in_predictor":True})
 # Actually compiled Rust finite hidden-state model on all TEST sensor initial-only cases.
 request={"cases":rust_cases}
 path=HOME/"evidence/C08_TEST/RUST_SHADOW_NO_LABEL_REQUEST.json"
 path.write_text(json.dumps(request,indent=2,sort_keys=True)+"\n")
 rust_bin=ROOT/"target/release/quartz_c20_shadow"
 native=json.loads(subprocess.check_output([str(rust_bin),str(path)],text=True))
 assert native["no_SearchPolicy_mutation"] is True
 assert native["beta_binomial_Rust_port"] is False
 diff=[]
 for a,b in zip(native["predictions"],sensor_expected):
  for field in ("evsi","before_best","after_expected_best","total_predictive_mass"):
   diff.append(abs(a[field]-b[field]))
  assert a["outcome_count"]==b["outcome_count"]
 assert len(rust_cases)==len(sensor_expected)==72
 assert max(diff)<=1e-10
 (HOME/"evidence/C08_TEST/FROZEN_SHADOW_FORECASTS.jsonl").write_text(
    "".join(json.dumps(v,sort_keys=True)+"\n" for v in forecasts))
 domains={}
 for family in ("beta_arms","finite_sensor"):
  idx=[i for i,r in enumerate(rows) if r["family"]==family]
  sub=[rows[i] for i in idx];pre=[forecasts[i] for i in idx]
  assert len(sub)==72 and len({r["group_id"] for r in sub})==24
  actual=[float(r["signed_gain"]) for r in sub]
  g=[float(v["evsi_exact"]) for v in pre]
  old=[float(v["C19_frozen_coarse_qm"]) for v in pre]
  costs=[float(r["actual_extra_cost_us"]) for r in sub]
  extra_eval=[float(v["measured_exact_evaluation_us"]) for v in pre]
  old_eval=[float(v["measured_coarse_evaluation_us"]) for v in pre]
  predicted_task=[v["predicted_extra_task_cost_us_TRAIN_median"] for v in pre]
  predicted_e=[v["predicted_exact_evaluator_cost_us_TRAIN_median"] for v in pre]
  predicted_old_e=[v["predicted_coarse_evaluator_cost_us_TRAIN_median"] for v in pre]
  pred_mse=[(x-y)**2 for x,y in zip(g,actual)]
  old_mse=[(x-y)**2 for x,y in zip(old,actual)]
  bydomain={"n_actual_extra_stochastic_task_computations":72,
   "n_independent_test_parents":24,
   "n_decision_revisions":sum(r["label"] for r in sub),
   "n_negative_realized_utility_gains":sum(x<0 for x in actual),
   "mean_realized_signed_terminal_gain":mean(actual),
   "mean_exact_predicted_EVSI":mean(g),
   "mean_C19_coarse_predicted_gain":mean(old),
   "exact_EVSI_prediction_MSE":mean(pred_mse),
   "C19_frozen_coarse_prediction_MSE":mean(old_mse),
   "exact_minus_coarse_MSE_descriptive_bootstrap":group_ci(sub,[a-b for a,b in zip(pred_mse,old_mse)],2026112011+len(sub)),
   "mean_actual_underlying_task_extra_us":mean(costs),
   "mean_actual_exact_predictor_eval_us":mean(extra_eval),
   "mean_actual_C19_coarse_eval_us":mean(old_eval),
   "mean_predicted_task_extra_us_TRAIN_median":mean(predicted_task),
   "mean_predicted_exact_eval_us_TRAIN_median":mean(predicted_e),
   "model_channel_claim":"correct for these simulator families, no real neural MCTS",
   "model_cost_caveat":"microsecond wall clocks include scheduler and Python interpreter; predicted group median may drift",
   "price_sensitivity":{}}
  for price in model["fixed_price_sensitivity"]:
   always=[z-price*c for z,c in zip(actual,costs)]
   q_coarse=[old[i]>price*predicted_task[i] for i in range(len(sub))]
   q_exact=[g[i]>price*predicted_task[i] for i in range(len(sub))]
   q_exactcharged=[g[i]>price*(predicted_task[i]+predicted_e[i]) for i in range(len(sub))]
   q_coarsecharged=[old[i]>price*(predicted_task[i]+predicted_old_e[i]) for i in range(len(sub))]
   # Operational forecasters are paid even when they choose STOP.
   net={
    "NEVER":[0.]*len(sub),
    "ALWAYS":always,
    "C19_COARSE_UNCHARGED":[v if active else 0. for v,active in zip(always,q_coarse)],
    "C20_EXACT_UNCHARGED":[v if active else 0. for v,active in zip(always,q_exact)],
    "C19_COARSE_CHARGED":[(v if active else 0.)-price*e
                          for v,active,e in zip(always,q_coarsecharged,old_eval)],
    "C20_EXACT_CHARGED":[(v if active else 0.)-price*e
                          for v,active,e in zip(always,q_exactcharged,extra_eval)]}
   record={
    "policies_mean_paid_net_utility_vs_STOP":{k:mean(v) for k,v in net.items()},
    "n_C20_EXACT_UNCHARGED_recommend_COMPUTE":sum(q_exact),
    "n_C20_EXACT_CHARGED_recommend_COMPUTE":sum(q_exactcharged),
    "n_C19_COARSE_CHARGED_recommend_COMPUTE":sum(q_coarsecharged),
    "C20_EXACT_CHARGED_missed_positive_net_task_opportunities":
      sum(not x and a>0 for x,a in zip(q_exactcharged,always)),
    "C20_EXACT_CHARGED_recommends_negative_net_task":
      sum(x and a<0 for x,a in zip(q_exactcharged,always)),
    "paired_CI_EXACT_UNCHARGED_minus_ALWAYS":group_ci(sub,[a-b for a,b in zip(net["C20_EXACT_UNCHARGED"],always)],
       2026112120+int(price*1e6)),
    "paired_CI_EXACT_CHARGED_minus_ALWAYS":group_ci(sub,[a-b for a,b in zip(net["C20_EXACT_CHARGED"],always)],
       2026112320+int(price*1e6)),
    "paired_CI_EXACT_CHARGED_minus_C19_CHARGED":group_ci(sub,
       [a-b for a,b in zip(net["C20_EXACT_CHARGED"],net["C19_COARSE_CHARGED"])],
       2026112420+int(price*1e6))
    }
   bydomain["price_sensitivity"][str(price)]=record
  domains[family]=bydomain
 report={"schema":"quartz_general_c20_C08_heldout_exact_predictive_EVSI_v1",
  "source_git_published_before_test_scoring":head,
  "prospective_C20_world_SHA256":sha(SOURCE),
  "frozen_C07A_RESOURCE_MODEL_SHA256":sha(COST_MODEL),
  "C08A_TEST_raw_SHA256":sha(RAW),
  "frozen_C19_coarse_reference_SHA256":sha(C19_MODEL),
  "n_distinct_test_parent_worlds":48,"n_actual_extra_stochastic_computations":144,
  "max_Rust_Python_sensor_EVSI_abs_error":max(diff),
  "n_Rust_Python_sensor_initial_only_cases":len(rust_cases),
  "native_standalone_Rust_compiled_sha256":sha(rust_bin),
  "price_units":"terminal expected utility per measured microsecond",
  "actual_hardware_saved_by_controller":"NOT_TESTED: source task extra computations always executed",
  "C20_controlled_live_policy_status":"SHADOW_ONLY",
  "true_neural_MCTS_signed_utility":"NOT_ESTIMABLE",
  "same_latent_parent_resampling_unit":True,
  "admission":"two simulator models with exact declared likelihood and utility, not production adaptive search",
  "heldout_domains":domains}
 out=HOME/"evidence/C08_TEST/C08_HELDOUT_EXACT_EVSI.json"
 out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 compact={family:{
   "mean_z":v["mean_realized_signed_terminal_gain"],
   "mean_G":v["mean_exact_predicted_EVSI"],
   "exact_mse":v["exact_EVSI_prediction_MSE"],
   "coarse_mse":v["C19_frozen_coarse_prediction_MSE"],
   "task_us":v["mean_actual_underlying_task_extra_us"],
   "evaluator_us":v["mean_actual_exact_predictor_eval_us"],
   "operational_primary":v["price_sensitivity"]["0.004"]["policies_mean_paid_net_utility_vs_STOP"],
   "CI_exact_charged_minus_always":v["price_sensitivity"]["0.004"]["paired_CI_EXACT_CHARGED_minus_ALWAYS"]["descriptive_95pct_parent_bootstrap"]
 } for family,v in domains.items()}
 print("C20_HELDOUT",json.dumps({"RUST_PYTHON_SENSOR_PARITY":max(diff),
    "n":144,"domains":compact},sort_keys=True))
if __name__=="__main__":main()
