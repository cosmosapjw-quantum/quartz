#!/usr/bin/env python3
"""C8 frozen game-neutral metalevel experiment: exact enumeration plus paired MC.
All observations and utilities are task-defined. No game or SearchPolicy adapters.
"""
from __future__ import annotations
from pathlib import Path
from quartz.experiments.metalevel_decision import FiniteProblem
from collections import Counter
import hashlib,json,math,random,subprocess
ROOT=next(x for x in Path(__file__).resolve().parents if (x/".git").exists())
HOME=Path(__file__).resolve().parent.parent
TASK=ROOT/"configs/metalevel_reference_cases.v1.json"
TASK_SHA="46d69d13ea327c23e02b76c53a0a78aadc1f77ee9e405dbef1c203a6d0b471dd"
POLICIES=("stop","ratio","net","bellman")
N_TRIALS=512
SEED=202611220817

def digest(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def stable_uniform(*parts):
 data=":".join(map(str,parts)).encode("utf-8")
 # Uniform in [0,1), independent of Python hash randomization and arm ordering.
 return int.from_bytes(hashlib.sha256(data).digest()[:8],"big")/2**64

def choose_discrete(probabilities,u):
 c=0.0
 for i,p in enumerate(probabilities):
  c+=p
  if u < c:return i
 return len(probabilities)-1

def action_choice(problem,posterior,policy,h):
 if h==0 or policy=="stop":return "STOP"
 if policy=="ratio":return problem.one_step(posterior)["ratio_choice"]
 if policy=="net":return problem.one_step(posterior)["net_choice"]
 if policy=="bellman":return problem.bellman(h,posterior)[1]
 raise ValueError(policy)

def realized_loss(problem,hidden,posterior):
 action=problem.best_action(posterior)
 return max(problem.utilities[hidden])-problem.utilities[hidden][action]

def exact_expected_loss(problem,policy):
 h=problem.horizon
 def recurse(posterior,hidden,remain):
  action=action_choice(problem,posterior,policy,remain)
  if action=="STOP":return realized_loss(problem,hidden,posterior)
  exp=problem.experiment(action)
  channel=exp.likelihood[hidden]
  total=problem.lambda_cost*exp.cost
  for y,p in enumerate(channel):
   if p<=0:continue
   after=problem.posterior_after(action,y,posterior)
   total+=p*recurse(after,hidden,remain-1)
  return total
 return math.fsum(p*recurse(problem.prior,i,h) for i,p in enumerate(problem.prior) if p>0)

def one_trial(problem,case,policy,index):
 hidden=choose_discrete(problem.prior,stable_uniform("state",SEED,case,index))
 posterior=problem.prior
 counters=Counter()
 cost=0.0
 for remaining in range(problem.horizon,0,-1):
  action=action_choice(problem,posterior,policy,remaining)
  if action=="STOP":break
  exp=problem.experiment(action)
  draw=stable_uniform("obs",SEED,case,index,action,counters[action])
  counters[action]+=1
  outcome=choose_discrete(exp.likelihood[hidden],draw)
  posterior=problem.posterior_after(action,outcome,posterior)
  cost+=problem.lambda_cost*exp.cost
 return {
   "task":case,"trial":index,"policy":policy,
   "state_index":hidden,
   "decision_loss":realized_loss(problem,hidden,posterior),
   "priced_compute_cost":cost,
   "n_computations":sum(counters.values()),
   "chosen_action":problem.best_action(posterior),
 }

def authoritative():
 if subprocess.check_output(["git","-C",str(ROOT),"status","--porcelain"]).strip():
  raise RuntimeError("C8 requires frozen published clean source")
 head=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
 branch=subprocess.check_output(["git","-C",str(ROOT),"branch","--show-current"],text=True).strip()
 remote=subprocess.check_output(["git","-C",str(ROOT),"ls-remote","--heads","origin",branch],text=True).split()[0]
 if head!=remote:raise RuntimeError("current C8 script not published by GitHub")
 if digest(TASK)!=TASK_SHA:raise RuntimeError("C4 task bank bytes changed")
 return head

def main():
 head=authoritative()
 task=json.loads(TASK.read_text())
 assert len(task["cases"])==5
 out=HOME/"evidence/C08"
 out.mkdir(exist_ok=False,parents=True)
 manifest={"schema":"quartz_meta17_C08_exact_and_paired_MC_v1",
    "source_commit_published_before_any_outcome":head,
    "source_code_sha256":digest(Path(__file__)),
    "five_task_bank_sha256":digest(TASK),
    "trial_count_per_case_and_policy":N_TRIALS,
    "seed":SEED,"policies":list(POLICIES),
    "expected_science_scope":"finite synthetic calibrated conditional channels only",
    "games_or_neural_MCTS_tested":False,
    "treatment_costs_charged":True}
 (out/"EXECUTION_CONTRACT.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
 rows=[];summary=[]
 for task_data in task["cases"]:
  case=task_data["id"]
  problem=FiniteProblem.from_mapping(task_data)
  for policy in POLICIES:
   exact=exact_expected_loss(problem,policy)
   if policy=="bellman":
    assert math.isclose(exact,problem.bellman(problem.horizon)[0],rel_tol=0,abs_tol=1e-10),(case,exact)
   tr=[one_trial(problem,case,policy,i) for i in range(N_TRIALS)]
   score=[z["decision_loss"]+z["priced_compute_cost"] for z in tr]
   assert len(score)==N_TRIALS
   mean=math.fsum(score)/N_TRIALS
   variance=math.fsum((s-mean)**2 for s in score)/(N_TRIALS-1)
   stderr=math.sqrt(variance/N_TRIALS)
   # Only an execution diagnostic, no data-dependent adjustment.
   assert abs(mean-exact)<=6*stderr+0.04,(case,policy,mean,exact,stderr)
   rows.extend(tr)
   summary.append({"task":case,"family":task_data["family"],"policy":policy,
     "n":len(tr),"exact_expected_total_loss":exact,
     "MC_mean_total_loss":mean,"MC_standard_error":stderr,
     "abs_empirical_exact_diff":abs(mean-exact),
     "mean_decision_loss":math.fsum(z["decision_loss"] for z in tr)/N_TRIALS,
     "mean_priced_compute_cost":math.fsum(z["priced_compute_cost"] for z in tr)/N_TRIALS,
     "mean_number_of_computations":math.fsum(z["n_computations"] for z in tr)/N_TRIALS})
  print("C08_CASE_COMPLETED",case,flush=True)
 raw=out/"PAIRED_MC_10240.jsonl"
 raw.write_text("".join(json.dumps(v,sort_keys=True,separators=(",",":"))+"\n" for v in rows))
 report={"schema":"quartz_meta17_C08_finite_value_experiment_v1",
 "source_contract":manifest,
 "total_actual_simulated_paths":len(rows),
 "n_distinct_task_families":5,
 "n_independent_seeded_trial_indices_per_case":N_TRIALS,
 "fixed_world_and_observation_draws_keyed_per_trial_case":True,
 "actual_Rust_reference_binary_not_used_for_MC":"C06 independent Rust policy numerical parity already performed",
 "original_frozen_thesis_modified":False,
 "raw_SHA256":digest(raw),
 "per_arm":summary,
 "claim":"exact-model one-step and finite-horizon synthetic reference comparison only; no MCTS/CPU/Elo"}
 final=out/"C08_RESULTS.json"
 final.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print("C08_EXACT_PAIRED_MC",json.dumps({
   "status":"PASS","cases":5,"n_rows":len(rows),
   "raw_SHA256":digest(raw),
   "max_exact_empirical_diff":max(z["abs_empirical_exact_diff"] for z in summary),
   "heterogeneous_case":{z["policy"]:z["exact_expected_total_loss"] for z in summary if z["task"]=="heterogeneous_complete_vs_partial"}
 },sort_keys=True))
if __name__=="__main__":main()
