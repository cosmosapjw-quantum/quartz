#!/usr/bin/env python3
"""C22 prospective TEST: actual in-process STOP and real native paired ALWAYS.

C22 C07 algorithm/score and C07A TRAIN-only price are frozen/pushed before
TEST raw is opened. Every active STOP skips REAL extra draws in Rust; a separate
actual ALWAYS control is executed under the same seed and latent source.
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,itertools,json,math,random,statistics,subprocess,time
from quartz.experiments.c22_active_inputs import build_request,public_task,workload_bucket
from quartz.experiments.c20_predictive_value import sensor_evsi
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
BANK=HOME/"SOURCE_WORLDS_C22_FROZEN.json"
TEST=HOME/"evidence/C08_TEST/RAW_SIGNED_CALCULATIONS.jsonl"
COSTS=HOME/"FROZEN_C22_RESOURCE_PRICE.json"
BIN=ROOT/"target/release/quartz_c22_active_gate"
EXPECTED_SOURCE="f7bffc98710bae151d0ac909ddeb19fc04f74d51a4273f6b16f7e7583f61398d"
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def preflight():
 if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT).strip():
  raise RuntimeError("must score from clean published C08A with frozen C07 science code")
 head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
 remote=subprocess.check_output(["git","ls-remote","--heads","origin",branch],cwd=ROOT,text=True).split()[0]
 assert head==remote
 return head
def mean(xs):return math.fsum(xs)/len(xs)
def group_bootstrap(rows,deltas,seed):
 byparent=defaultdict(list)
 for row,d in zip(rows,deltas):byparent[row["group_id"]].append(float(d))
 assert len(byparent)==24
 group=[mean(byparent[k]) for k in sorted(byparent)]
 rng=random.Random(seed)
 n=len(group)
 bs=sorted(mean([group[rng.randrange(n)] for _ in range(n)]) for _ in range(500))
 return {"n_independent_parent_groups":n,"bootstrap_draws":500,
  "parent_group_mean_delta":mean(group),
  "descriptive_percentile95":[bs[12],bs[487]]}
def source_reference_beta_evsi(posterior,budgets):
 # Independently written Pólya-urn predictive distributions, not native Rust source.
 # The repeated Bernoulli observations share arm theta, so counts are Beta-Binomial.
 arms=[]
 for (a,b),n in zip(posterior,budgets):
  p=math.prod((b+i)/(a+b+i) for i in range(n))
  probs=[p]
  for k in range(n):
   p*=((n-k)/(k+1))*((a+k)/(b+n-k-1))
   probs.append(p)
  assert abs(math.fsum(probs)-1)<1e-9
  arms.append([((a+k)/(a+b+n),probs[k]) for k in range(n+1)])
 support=sorted(set(v for m in arms for v,p in m))
 last=0.;prev=0.;expected=0.
 for v in support:
  expected+=(v-last)*(1-prev)
  prev=math.prod(math.fsum(p for x,p in arm if x<=v) for arm in arms)
  last=v
 assert abs(prev-1)<1e-8
 M=max(a/(a+b) for a,b in posterior)
 return max(0.,expected-M)
def run():
 head=preflight()
 assert sha(BANK)==EXPECTED_SOURCE
 worlds={w["id"]:w for w in json.loads(BANK.read_text())["worlds"]}
 price=json.loads(COSTS.read_text())
 assert price["source_bank_SHA256"]==sha(BANK)
 assert price["TEST_outcomes_opened"] is False
 assert price["precommitted_primary_lambda"]==0.004
 assert price["precommitted_sensitivity_lambda"]==[0.001,0.004,0.02,0.1]
 assert sha(BIN)==price["native_Rust_active_compiled_sha256"]
 raw=[json.loads(s) for s in TEST.read_text().splitlines() if s.strip()]
 assert len(raw)==144 and all(x["split"]=="test" for x in raw)
 assert len({r["group_id"] for r in raw})==48
 assert len({r["group_id"] for r in raw if r["family"]=="beta_arms"})==24
 assert len({r["group_id"] for r in raw if r["family"]=="finite_sensor"})==24
 files=HOME/"evidence/C08_TEST"
 all_rows=[]
 outcomes={}
 max_reference_error=0.;reference_checks=0
 for iprice,lam in enumerate(price["precommitted_sensitivity_lambda"]):
  req=build_request(raw,worlds,lam,price)
  for item in req["cases"]:
   pub=item["public"]
   assert not (set(pub)&{"theta_offline_arm_means","theta_offline_state","signed_gain",
      "label","extra_observations","actual_extra_cost_us"})
  reqfile=files/("NATIVE_ACTIVE_INPUT_LAMBDA_"+str(lam).replace(".","p")+".json")
  reqfile.write_text(json.dumps(req,sort_keys=True,separators=(",",":"))+"\n")
  t0=time.perf_counter_ns()
  result=json.loads(subprocess.check_output([str(BIN),str(reqfile)],text=True))
  host_total=(time.perf_counter_ns()-t0)/1000
  assert result["schema"]=="quartz_C22_ACTIVE_INPROCESS_v1"
  assert result["number_cases"]==144
  assert result["original_SearchPolicy_unchanged"] is True
  active_by_id={x["id"]:x for x in result["rows"]}
  assert len(active_by_id)==144
  byfamily={}
  for fam in ("beta_arms","finite_sensor"):
   pairs=[]
   for row in raw:
    if row["family"]!=fam:continue
    key=row["group_id"]+"-rep"+str(row["rep"])
    x=active_by_id[key]
    assert x["group_id"]==row["group_id"] and x["rep"]==row["rep"]
    t=x["results"];base=t["ALWAYS"];gate=t["GATED"]
    pub=public_task(row,worlds[row["group_id"]])
    budget=sum(pub["extra_per_arm"]) if fam=="beta_arms" else pub["extra"]
    assert base["actual_future_draw_count"]==budget
    for policy,part in t.items():
     if not part["COMPUTE"]:
      assert part["actual_future_draw_count"]==0
      assert part["before_action"]==part["after_action"]
      assert part["true_signed_terminal_utility_change"]==0.
      assert part["actual_future_task_us"]==0.
     else:
      assert part["actual_future_draw_count"]==budget
      assert part["after_action"]==base["after_action"]
      assert abs(part["true_signed_terminal_utility_change"]-base["true_signed_terminal_utility_change"])<1e-12
    if gate["certified_zero_information"]:
     assert gate["decision_reason"]=="CERTIFIED_ZERO"
     assert base["after_action"]==base["before_action"]
     assert abs(base["true_signed_terminal_utility_change"])<1e-12
    # Always exact Bayesian expected value computed separately from Rust model.
    if lam==0.004:
     if fam=="beta_arms":
      exact=source_reference_beta_evsi(pub["posterior"],pub["extra_per_arm"])
     else:
      exact=sensor_evsi(pub["prior"],pub["likelihood"],pub["utilities"],pub["extra"])["evsi"]
     err=abs(exact-t["FULL_EXACT"]["gross_predictive_EVSI"])
     max_reference_error=max(max_reference_error,err)
     assert err<=1e-10,(key,err)
     reference_checks+=1
    pairs.append((row,x))
   assert len(pairs)==72 and len({row["group_id"] for row,x in pairs})==24
   # Resampling is over original latent WORLD groups, not within repeated observations.
   rec=[{"group_id":r["group_id"]} for r,x in pairs]
   names=["NEVER","ALWAYS","CERTIFIED_ONLY","GATED","FULL_EXACT"]
   per={name:[x["results"][name] for row,x in pairs] for name in names}
   measured_net={name:mean([p["net_utility_change_after_measured_cost"] for p in arr])
      for name,arr in per.items()}
   measured_total={name:mean([p["active_total_internal_us"] for p in arr])
      for name,arr in per.items()}
   draws={name:sum(p["actual_future_draw_count"] for p in arr) for name,arr in per.items()}
   true_utility={name:mean([p["true_signed_terminal_utility_change"] for p in arr])
      for name,arr in per.items()}
   baseline=per["ALWAYS"];gated=per["GATED"];full=per["FULL_EXACT"]
   byfamily[fam]={
    "n_original_independent_test_parents":24,"n_actual_active_task_cases":72,
    "policy_mean_realized_net_gain_vs_free_STOP":measured_net,
    "policy_mean_true_terminal_signed_gain":true_utility,
    "policy_mean_actual_native_elapsed_us":measured_total,
    "policy_total_ACTUALLY_EXECUTED_future_draws":draws,
    "GATED_ACTUALLY_SKIPPED_future_draw_count":draws["ALWAYS"]-draws["GATED"],
    "GATED_n_true_STOP":sum(not p["COMPUTE"] for p in gated),
    "GATED_n_CERTIFIED_ZERO":sum(p["decision_reason"]=="CERTIFIED_ZERO" for p in gated),
    "GATED_n_PRICE_UPPER_SCREEN":sum(p["decision_reason"]=="PRICE_UPPER_SCREEN" for p in gated),
    "GATED_n_EXACT_STOP":sum(p["decision_reason"]=="EXACT_STOP" for p in gated),
    "GATED_STOP_missed_positive_utility":sum(
       (not v["COMPUTE"]) and b["true_signed_terminal_utility_change"]>0
       for v,b in zip(gated,baseline)),
    "GATED_STOP_missed_positive_priced_control_net":sum(
       (not v["COMPUTE"]) and b["net_utility_change_after_measured_cost"]>0
       for v,b in zip(gated,baseline)),
    "C22_actual_gate_cost_us_mean":mean([p["cheap_gate_us"] for p in gated]),
    "C22_actual_exact_estimator_cost_us_mean":mean([p["exact_evaluator_us"] for p in gated]),
    "CI_GATED_minus_ALWAYS_net":group_bootstrap(rec,[
       x["net_utility_change_after_measured_cost"]-y["net_utility_change_after_measured_cost"]
       for x,y in zip(gated,baseline)],220010+iprice*100),
    "CI_GATED_minus_NEVER_net":group_bootstrap(rec,[
       x["net_utility_change_after_measured_cost"] for x in gated],220020+iprice*100),
    "CI_GATED_minus_FULL_EXACT_net":group_bootstrap(rec,[
       x["net_utility_change_after_measured_cost"]-y["net_utility_change_after_measured_cost"]
       for x,y in zip(gated,full)],220030+iprice*100),
    "CI_GATED_ACTUAL_total_us_saved_vs_ALWAYS":group_bootstrap(rec,[
       y["active_total_internal_us"]-x["active_total_internal_us"]
       for x,y in zip(gated,baseline)],220040+iprice*100),
    "source_task_payoff":"known latent expected terminal utility, not single realized Bernoulli reward",
    "runtime_scope":"same Rust inprocess policy + actual task with separately timed ALWAYS control, excludes one-off host JSON invoke",
   }
  outcomes[str(lam)]={"lambda_utility_per_microsecond":lam,
   "host_binary_launch_json_and_score_us_TOTAL_NOT_IN_PER_CASE":host_total,
   "host_launch_scope":"one CLI batch of 144 cases, NOT per-computation server boundary",
   "families":byfamily,
   "total_n_gated_activated_STOPS":result["GATED_true_STOP_count"],
   "total_gated_ACTUALLY_avoided_future_draws":result["GATED_actually_unexecuted_future_draws"]}
  all_rows.extend([{"lambda":lam,**x} for x in result["rows"]])
 output=files/"C22_ACTIVE_ALL_POLICIES.jsonl"
 output.write_text("".join(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n" for row in all_rows))
 result={"schema":"quartz_C22_C08_PROSPECTIVE_ACTUAL_COMPUTE_STOP_TWO_FAMILIES_v1",
 "score_started_on_clean_published_C08A_Git_sha":head,
 "new_frozen_source_world_SHA256":sha(BANK),
 "frozen_TRAIN_only_price_SHA256":sha(COSTS),
 "C08A_raw_TEST_SHA256":sha(TEST),
 "C22_native_ACTUAL_POLICY_result_records_SHA256":sha(output),
 "C22_compiled_Rust_active_executable_sha256":sha(BIN),
 "n_independent_TEST_original_parent_worlds":48,
 "n_TEST_original_examples":144,
 "n_price_scenarios":len(outcomes),
 "C22_source_utility_calibration":"simulation correct prior and likelihood, independent Beta arms and finite hidden sensor",
 "n_independent_model_value_vs_C20_reference_checked":reference_checks,
 "max_abs_model_EVSI_reference_error":max_reference_error,
 "actual_active_STOP_physically_not_sampled":True,
 "additional_ALWAYS_control_physically_executed_and_timed_separately":True,
 "physical_GPU_or_live_PUCT_savings_claim":"NOT_TESTED",
 "original_MCTS_utility_signed":"NOT_ESTIMABLE",
 "price_scenarios":outcomes}
 path=files/"C08_ACTUAL_ACTIVE_RESULTS.json"
 path.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
 compact={l:{fam:{
   "n_gated_STOP":d["GATED_n_true_STOP"],"n_CERT":d["GATED_n_CERTIFIED_ZERO"],
   "n_skipped_future_draws":d["GATED_ACTUALLY_SKIPPED_future_draw_count"],
   "net_by_policy":d["policy_mean_realized_net_gain_vs_free_STOP"],
   "mean_us_by_policy":d["policy_mean_actual_native_elapsed_us"],
   "CI_net_GATE_minus_ALWAYS":d["CI_GATED_minus_ALWAYS_net"]["descriptive_percentile95"]}
   for fam,d in scenario["families"].items()}
   for l,scenario in outcomes.items()}
 print("C22_C08_ACTUAL",json.dumps({"n_parent_TEST":48,"n_examples":144,
  "max_reference_error":max_reference_error,
  "source_sha256":sha(BANK),"cost_sha256":sha(COSTS),
  "families":compact},sort_keys=True))
if __name__=="__main__":run()
