#!/usr/bin/env python3
"""C19 source-only actual stochastic computations: BETA_ARMS and FINITE_SENSOR.

COLLECTION contracts source/runner/model identity and phase. TRAIN/CAL and TEST
are run separately with a GitHub-published frozen source before observations.
True latent θ is only present in source data for offline scientific auditing.
"""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json,random,subprocess,time
ROOT=next(x for x in Path(__file__).resolve().parents if (x/".git").exists())
HOME=Path(__file__).resolve().parent.parent
SOURCE=HOME/"SOURCE_WORLDS_C21_FROZEN.json"
SOURCE_SHA="036b7c2e77d998558de46ff80845e4b0be382ff4a35d365b03a7174d6c98dad8"
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def remote_clean():
 if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT).strip():
  raise RuntimeError("C19 outcomes require committed, clean and published source")
 head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
 remote=subprocess.check_output(["git","ls-remote","--heads","origin",branch],cwd=ROOT,text=True).split()[0]
 if head!=remote:raise RuntimeError("GitHub remote published HEAD mismatch")
 return head
def keyed_seed(world,rep):
 return int.from_bytes(hashlib.sha256(
    (world["id"]+":"+str(world["seed"])+":rep:"+str(rep)).encode()).digest()[:8],"big")
def argmax(values):
 return max(range(len(values)),key=lambda i:(values[i],-i))
def gap(values):
 z=sorted(values,reverse=True)
 g=z[0]-z[1]
 if not -1e-12<=g<=1+1e-12:raise ValueError("initial posterior utility gap out of range")
 return max(0.0,min(1.0,float(g)))
def draw_categorical(prob,rng):
 u=rng.random()
 s=0.
 for i,x in enumerate(prob):
  s+=x
  if u<s:return i
 return len(prob)-1
def beta_case(world,rep):
 rng=random.Random(keyed_seed(world,rep))
 true=world["theta_offline_true_arm_means"]
 k=world["K"]
 init_draws=[]
 successes=[0]*k
 for j in range(k):
  draws=[int(rng.random()<true[j]) for _ in range(world["initial_draws_per_arm"])]
  init_draws.append(draws);successes[j]=sum(draws)
 means=[(2.0+s)/(4+world["initial_draws_per_arm"]) for s in successes]
 start_action=argmax(means)
 initial_gap=gap(means)
 start=time.perf_counter_ns()
 later_draws=[]
 for j in range(k):
  draws=[int(rng.random()<true[j]) for _ in range(world["extra_draws_per_arm"])]
  later_draws.append(draws);successes[j]+=sum(draws)
 final_means=[(2.0+s)/(4+world["initial_draws_per_arm"]+world["extra_draws_per_arm"]) for s in successes]
 final_action=argmax(final_means)
 elapsed_us=max(1,(time.perf_counter_ns()-start)//1000)
 delta=true[final_action]-true[start_action]
 return {"schema":"quartz_general_c21_observation_v1","family":world["family"],
   "group_id":world["id"],"rep":rep,"split":world["split"],
   "source_kind":"genuine_Bernoulli_sampled_computations",
   "gap":initial_gap,"n_actions":k,"compute_kind":world["family"],
   "pre_action":start_action,"post_action":final_action,
   "label":int(start_action!=final_action),
   "signed_gain":float(delta),"actual_extra_cost_us":int(elapsed_us),
   "initial_observations":init_draws,"extra_observations":later_draws,
   "initial_posterior_expected_utilities":means,
   "final_posterior_expected_utilities":final_means,
   "theta_offline_true_arm_means":true,
   "posterior_prior":"Beta(2,2) correctly matched to latent worlds",
   "incremental_CPU_timing":"extra actual draws+posterior+new terminal choice",
   "online_predictor_observed_latent_theta":False}
def post_update(posterior,likelihood,obs):
 raw=[p*row[obs] for p,row in zip(posterior,likelihood)]
 s=sum(raw)
 if s<=0:raise ValueError("impossible posterior observation")
 return [x/s for x in raw]
def sensor_expected(posterior,utilities):
 return [sum(posterior[t]*utilities[t][a] for t in range(4)) for a in range(3)]
def sensor_case(world,rep):
 rng=random.Random(keyed_seed(world,rep))
 prior=world["initial_prior"]
 theta=world["theta_offline_true_state"]
 u=world["utility_state_action_matrix_PUBLIC"]
 L=world["observation_likelihood_state_outcome_PUBLIC"]
 posterior=prior[:];start_draws=[]
 for _ in range(world["initial_observations"]):
  y=draw_categorical(L[theta],rng)
  start_draws.append(y)
  posterior=post_update(posterior,L,y)
 initial_expected=sensor_expected(posterior,u)
 initial_action=argmax(initial_expected)
 initial_gap=gap(initial_expected)
 start=time.perf_counter_ns()
 extra=[]
 for _ in range(world["extra_observations"]):
  y=draw_categorical(L[theta],rng)
  extra.append(y)
  posterior=post_update(posterior,L,y)
 final_expected=sensor_expected(posterior,u)
 final_action=argmax(final_expected)
 elapsed_us=max(1,(time.perf_counter_ns()-start)//1000)
 delta=u[theta][final_action]-u[theta][initial_action]
 return {"schema":"quartz_general_c21_observation_v1","family":world["family"],
   "group_id":world["id"],"rep":rep,"split":world["split"],
   "source_kind":"genuine_categorical_sensor_samples",
   "gap":initial_gap,"n_actions":3,"compute_kind":world["family"],
   "pre_action":initial_action,"post_action":final_action,
   "label":int(initial_action!=final_action),
   "signed_gain":float(delta),"actual_extra_cost_us":int(elapsed_us),
   "initial_observations":start_draws,"extra_observations":extra,
   "initial_posterior_expected_utilities":initial_expected,
   "final_posterior_expected_utilities":final_expected,
   "theta_offline_true_state":theta,
   "prior_PUBLIC":prior,"utilities_PUBLIC":u,"likelihood_PUBLIC":L,
   "incremental_CPU_timing":"extra actual sensor observations+Bayes updates+new terminal choice",
   "online_predictor_observed_latent_theta":False}
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--phase",choices=("train_cal","test"),required=True)
 args=ap.parse_args()
 head=remote_clean()
 if sha(SOURCE)!=SOURCE_SHA:raise RuntimeError("C04 frozen world identity changed")
 worlds=json.loads(SOURCE.read_text())["worlds"]
 allowed={"train","cal"} if args.phase=="train_cal" else {"test"}
 selected=[w for w in worlds if w["split"] in allowed]
 assert len(selected)==(192 if args.phase=="train_cal" else 48)
 # Source is first-keyed and family count is fixed before any sampled outcome.
 raw=[]
 for world in selected:
  for rep in range(world["replications"]):
   row=beta_case(world,rep) if world["family"]=="beta_arms" else sensor_case(world,rep)
   assert 0<=row["gap"]<=1 and row["actual_extra_cost_us"]>0
   assert row["label"]==int(row["pre_action"]!=row["post_action"])
   if row["label"]==0 and abs(row["signed_gain"])>1e-12:
    raise AssertionError("unchanged action has nonzero utility gain")
   if abs(row["signed_gain"])>1+1e-12:raise AssertionError("non-normalized terminal utility")
   raw.append(row)
 assert len(raw)==(576 if args.phase=="train_cal" else 144)
 assert all(r["split"] in allowed for r in raw)
 dir=HOME/("evidence/C07_TRAIN_CAL" if args.phase=="train_cal" else "evidence/C08_TEST")
 dir.mkdir(parents=True,exist_ok=False)
 out=dir/"RAW_SIGNED_CALCULATIONS.jsonl"
 out.write_text("".join(json.dumps(r,sort_keys=True,separators=(",",":"))+"\n" for r in raw))
 report={"schema":"quartz_general_c21_source_collection_v1",
  "phase":args.phase,"source_published_HEAD_before_generation":head,
  "source_worlds_SHA256":sha(SOURCE),"raw_sha256":sha(out),
  "total_actual_stochastic_computations":len(raw),
  "independent_parent_world_groups":len(selected),
  "source_families":{f:{
     "parents":sum(w["family"]==f for w in selected),
     "actual_transcripts":sum(r["family"]==f for r in raw),
     "flips":sum(r["label"] for r in raw if r["family"]==f),
     "mean_signed_gain":sum(r["signed_gain"] for r in raw if r["family"]==f)/sum(r["family"]==f for r in raw),
     "mean_extra_wall_us":sum(r["actual_extra_cost_us"] for r in raw if r["family"]==f)/sum(r["family"]==f for r in raw),
    } for f in ("beta_arms","finite_sensor")},
  "training_features":["initial utility-margin normalized","number of terminal actions"],
  "true_utility_only_offline":True,"source_group_split_before_outcomes":True,
  "contains_future_test_labels":args.phase=="test",
  "no_game_Rust_MCTS_or_true_search_regret_claim":True}
 (dir/"COLLECTION_MANIFEST.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
 print("C21_ACTUAL_SOURCE_RESULTS",json.dumps(report,sort_keys=True))
if __name__=="__main__":main()
