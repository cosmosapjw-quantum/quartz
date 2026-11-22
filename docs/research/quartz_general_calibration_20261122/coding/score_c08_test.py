#!/usr/bin/env python3
"""C8 TEST readout: frozen C07A shadow model, independent parent-family metrics.

The runner was source frozen BEFORE collection or reading any C18 TEST outcomes.
No model re-fit, no threshold/feature adjustment on this script's results.
"""
from pathlib import Path
from collections import defaultdict
import hashlib,json,random,subprocess,statistics
from quartz.experiments.c18_observation import predict,score_predictions
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
NATIVE=ROOT/"target/release/quartz_c18_shadow"
MODEL=HOME/"SHADOW_MODEL_FROZEN.json"
SOURCE=HOME/"SOURCE_GROUPS_FROZEN.json"
RAWT=HOME/"evidence/C08_TEST/SOURCE_RAW.jsonl"
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def published():
 if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT).strip():
  raise RuntimeError("heldout scoring requires clean published C07A model")
 head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
 remote=subprocess.check_output(["git","ls-remote","--heads","origin",branch],cwd=ROOT,text=True).split()[0]
 assert head==remote
 return head
def cluster_bootstrap(rows,pred,baseline,seed):
 groups=defaultdict(list)
 for i,r in enumerate(rows):groups[r["group_id"]].append(i)
 rng=random.Random(seed)
 keys=list(sorted(groups))
 draws=[]
 for _ in range(250):
  idx=[i for k in (rng.choice(keys) for _ in keys) for i in groups[k]]
  diff=sum((pred[i]-rows[i]["label"])**2
           -(baseline[i]-rows[i]["label"])**2 for i in idx)/len(idx)
  draws.append(diff)
 draws.sort()
 return {"n_parent_source_groups":len(keys),
  "calibrated_minus_train_constant_Brier_95pct_group_bootstrap":[draws[6],draws[243]],
  "number_bootstrap_draws":250}
def main():
 head=published()
 frozen=json.loads(MODEL.read_text())
 assert frozen["schema"]=="quartz_general_c18_shadow_train_cal_frozen_v1"
 assert frozen["heldout_test_labels_seen"] is False
 assert sha(SOURCE)==frozen["source_bank_sha256"]
 rows=[json.loads(x) for x in RAWT.read_text().splitlines() if x]
 assert rows and all(r["split"]=="test" for r in rows)
 assert len({r["group_id"] for r in rows if r["domain"]=="real_Rust_MCTS"})==24
 assert len({r["group_id"] for r in rows if r["domain"]=="non_game_Bernoulli"})==24
 theta=frozen["weights"];offset=frozen["cal_offset"]
 forecast=predict(theta,rows,offset)
 rawforecast=predict(theta,rows,0.0)
 baseline=[frozen["train_baseline_constant_by_domain"][r["domain"]] for r in rows]
 request={"theta":theta,"offset":offset,
      "rows":[{"gap":r["gap"],"n_actions":r["n_actions"]} for r in rows]}
 path=HOME/"evidence/C08_TEST/SHADOW_NO_LABEL_REQUEST.json"
 path.write_text(json.dumps(request,sort_keys=True,indent=2)+"\n")
 native=json.loads(subprocess.check_output([str(NATIVE),str(path)],text=True))
 assert native["online_actions_mutated"] is False
 errors=[abs(p-q) for p,q in zip(native["prediction"],forecast)]
 assert len(errors)==len(rows) and max(errors)<=1e-10
 results={}
 for domain in ("real_Rust_MCTS","non_game_Bernoulli"):
  sub=[r for r in rows if r["domain"]==domain]
  ix=[i for i,r in enumerate(rows) if r["domain"]==domain]
  pc=[float(forecast[i]) for i in ix]
  pr=[float(rawforecast[i]) for i in ix]
  b=[float(baseline[i]) for i in ix]
  y=[int(r["label"]) for r in sub]
  # NOTE: threshold is a SHADOW decision flag, never actuated STOP or source request.
  shadow_stop=[p<0.20 for p in pc]
  missed=sum(int(s and t==1) for s,t in zip(shadow_stop,y))
  outcome={"n":len(sub),"n_parent_groups":len(set(r["group_id"] for r in sub)),
    "n_actual_flip_events":sum(y),
    "calibrated":score_predictions(pc,y),
    "uncalibrated":score_predictions(pr,y),
    "train_constant":score_predictions(b,y),
    "shadow_proposed_STOP_count":sum(shadow_stop),
    "shadow_STOP_missed_revisions":missed,
    "shadow_STOP_empirical_wrong_revision_fraction":missed/max(1,sum(shadow_stop)),
    "source_group_bootstrap":cluster_bootstrap(sub,pc,b,2026111800+len(sub)),
    "mean_observed_source_wall_us":statistics.mean(
      r["observed_total_native_decision_wall_us"] if domain=="real_Rust_MCTS"
      else r["observed_extra_cost_us"] for r in sub),
    "wall_type":"C16 full active 20ms Rust root search" if domain=="real_Rust_MCTS"
     else "Python actual additional Bernoulli draws only"}
  if domain=="non_game_Bernoulli":
   deltas=[r["signed_true_utility_delta_OFFLINE"] for r in sub]
   outcome["actual_offline_mean_signed_utility_delta"]=statistics.mean(deltas)
   outcome["negative_signed_change_count"]=sum(x<0 for x in deltas)
   outcome["positive_signed_change_count"]=sum(x>0 for x in deltas)
   outcome["no_revision_but_posterior_updates_possible"]=sum(r["label"]==0 for r in sub)
  else:outcome["actual_game_signed_utility_delta"]="NOT_ESTIMABLE"
  results[domain]=outcome
 report={"schema":"quartz_general_c18_C08_heldout_v1",
   "model_frozen_before_test_sha256":sha(MODEL),
   "source_bank_sha256":sha(SOURCE),
   "test_raw_sha256":sha(RAWT),
   "source_git_published_before_scoring":head,
   "n_independent_test_parent_groups_each_domain":24,
   "n_actual_test_observations":len(rows),
   "max_rust_python_shadow_probability_difference":max(errors),
   "no_active_controller_or_game_change":True,
   "all_model_coefficients_frozen_on_CAL_only":True,
   "model_is_not_calibrated_utility_VOC":True,
   "heldout_per_domain":results,
   "claim_scope":"real MCTS decision revision calibration and non-game decision revision; no MCTS regret or CPU savings"}
 out=HOME/"evidence/C08_TEST/C08_HELDOUT_RESULTS.json"
 out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print("C18_TEST_SCORES",json.dumps(report,sort_keys=True))
if __name__=="__main__":main()
