#!/usr/bin/env python3
"""C9 independent replay of ALL C19 TEST source samples, signed utility and net.

No import of C19 source collector, model fitter or TEST scorer.
A separate implementation reconstructs the seeded draws and posterior law.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,math,random,statistics
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
WORLDS=HOME/"SOURCE_WORLDS_FROZEN.json"
TEST=HOME/"evidence/C08_TEST/RAW_SIGNED_CALCULATIONS.jsonl"
TRAIN=HOME/"evidence/C07_TRAIN_CAL/RAW_SIGNED_CALCULATIONS.jsonl"
MODEL=HOME/"SHADOW_SIGNED_MODEL_FROZEN.json"
HELDOUT=HOME/"evidence/C08_TEST/C08_HELDOUT_SIGNED_RESULTS.json"
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
bank=json.loads(WORLDS.read_text())
worlds={x["id"]:x for x in bank["worlds"]}
train=[json.loads(s) for s in TRAIN.read_text().splitlines() if s]
test=[json.loads(s) for s in TEST.read_text().splitlines() if s]
model=json.loads(MODEL.read_text())
reported=json.loads(HELDOUT.read_text())
assert sha(WORLDS)=="8fce353c5efd3a5ac2a643e5a799d46f8d69c6c429cdff0fb5418a43ea43292e"
assert sha(TEST)==reported["test_raw_SHA256"]
assert sha(MODEL)==reported["model_SHA256_frozen_before_TEST"]
assert len(test)==144
assert len({r["group_id"] for r in test})==48
assert not {r["group_id"] for r in test}&{r["group_id"] for r in train}
def seed(world,rep):
 return int.from_bytes(hashlib.sha256(
 (world["id"]+":"+str(world["seed"])+":rep:"+str(rep)).encode()).digest()[:8],"big")
def pick(values):
 return max(range(len(values)),key=lambda j:(values[j],-j))
def drawcat(vec,rng):
 v=rng.random();s=0.0
 for j,p in enumerate(vec):
  s+=p
  if v<s:return j
 return len(vec)-1
def bayes_update(b,likelihood,y):
 raw=[p*row[y] for p,row in zip(b,likelihood)]
 s=sum(raw)
 return [v/s for v in raw]
def expected(posterior,U):
 return [sum(posterior[t]*U[t][a] for t in range(4)) for a in range(3)]
def replay(r,w):
 rng=random.Random(seed(w,r["rep"]))
 if r["family"]=="beta_arms":
  k=w["K"];mu=w["theta_offline_true_arm_means"]
  initial=[[int(rng.random()<mu[j]) for _ in range(3)] for j in range(k)]
  m0=[(2.0+sum(v))/(4+3) for v in initial]
  a0=pick(m0)
  extra=[[int(rng.random()<mu[j]) for _ in range(10)] for j in range(k)]
  m1=[(2.0+sum(initial[j])+sum(extra[j]))/(4+3+10) for j in range(k)]
  a1=pick(m1)
  delta=mu[a1]-mu[a0]
 else:
  U=w["utility_state_action_matrix_PUBLIC"]
  L=w["observation_likelihood_state_outcome_PUBLIC"]
  theta=w["theta_offline_true_state"]
  b=w["initial_prior"][:]
  initial=[]
  for _ in range(2):
   y=drawcat(L[theta],rng);initial.append(y)
   b=bayes_update(b,L,y)
  m0=expected(b,U);a0=pick(m0)
  extra=[]
  for _ in range(8):
   y=drawcat(L[theta],rng);extra.append(y)
   b=bayes_update(b,L,y)
  m1=expected(b,U);a1=pick(m1)
  delta=U[theta][a1]-U[theta][a0]
 assert initial==r["initial_observations"],("initial RNG stream",r["group_id"],r["rep"])
 assert extra==r["extra_observations"],("extra RNG stream",r["group_id"],r["rep"])
 assert a0==r["pre_action"] and a1==r["post_action"]
 assert r["label"]==int(a0!=a1)
 assert abs(delta-r["signed_gain"])<1e-12
 assert abs((sorted(m0,reverse=True)[0]-sorted(m0,reverse=True)[1])-r["gap"])<1e-12
 assert all(abs(x-y)<1e-12 for x,y in zip(m0,r["initial_posterior_expected_utilities"]))
 assert all(abs(x-y)<1e-12 for x,y in zip(m1,r["final_posterior_expected_utilities"]))
 assert r["actual_extra_cost_us"]>0
 return delta
for r in test:
 w=worlds[r["group_id"]]
 assert w["split"]=="test" and r["split"]=="test"
 assert w["family"]==r["family"] and w["K"]==r["n_actions"]
 replay(r,w)
assert model["TEST_labels_seen_or_used"] is False
# Independently reconstruct training source-kind cost medians and global baselines.
train_rows=[r for r in train if r["split"]=="train"]
assert len(train_rows)==432
median={x:statistics.median(r["actual_extra_cost_us"] for r in train_rows if r["family"]==x)
 for x in ("beta_arms","finite_sensor")}
assert median==model["predicted_incremental_cost_us_by_compute_kind"]
unconditional=statistics.mean(r["signed_gain"] for r in train_rows)
assert abs(unconditional-model["TRAIN_global_constant_signed_gain"])<1e-12
qcoeff=model["trained_model"]["q_coefficients"]
mcoeff=model["trained_model"]["m_coefficients"]
qoff=model["calibration_only_intercepts"]["q_offset"]
moff=model["calibration_only_intercepts"]["m_offset"]
def sigmoid(x):return 1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x))
def estimate(r):
 x=[1.0,r["gap"],math.log1p(r["n_actions"])]
 q=sigmoid(sum(a*b for a,b in zip(qcoeff,x))+qoff)
 q=max(1e-6,min(1-1e-6,q))
 m=sum(a*b for a,b in zip(mcoeff,x))+moff
 m=max(-1.0,min(1.0,m))
 return q,q*m
independent={}
for kind in ("beta_arms","finite_sensor"):
 subset=[r for r in test if r["family"]==kind]
 basep=model["TRAIN_global_constant_probability"]
 preds=[estimate(r) for r in subset]
 brier=sum((r["label"]-q)**2 for r,(q,g) in zip(subset,preds))/len(subset)
 brier_const=sum((r["label"]-basep)**2 for r in subset)/len(subset)
 mse=sum((r["signed_gain"]-g)**2 for r,(q,g) in zip(subset,preds))/len(subset)
 mse_const=sum((r["signed_gain"]-unconditional)**2 for r in subset)/len(subset)
 original=reported["heldout_domains"][kind]
 for a,b in [(brier,original["probability_calibration"]["two_part_calibrated_q"]["brier"]),
             (brier_const,original["probability_calibration"]["TRAIN_constant_q"]["brier"]),
             (mse,original["signed_gain_prediction"]["two_part_signed"]["mse"]),
             (mse_const,original["signed_gain_prediction"]["TRAIN_constant"]["mse"])]:
  assert abs(a-b)<1e-12,("score discrepancy",kind,a,b)
 for price in (0.001,0.004,0.01):
  gain=[r["signed_gain"]-price*r["actual_extra_cost_us"] for r in subset]
  shadow=[g if (pg-price*median[kind]>0) else 0.0
          for pg,g in zip([x[1] for x in preds],gain)]
  always=statistics.mean(gain)
  mean=statistics.mean(shadow)
  record=original["prices"][str(price)]["mean_paid_net_utility_gain_vs_STOP"]
  assert abs(always-record["ALWAYS"])<1e-12
  assert abs(mean-record["SIGNED_NET_SHADOW"])<1e-12
 independent[kind]={"n_test_rows":len(subset),
  "flips":sum(r["label"] for r in subset),
  "heldout_Brier":brier,"heldout_signed_gain_MSE":mse,
  "paired_primary_mean_shadow_minus_always":
    original["prices"]["0.004"]["paired_CI_SHADOW_minus_ALWAYS"]["mean_parent_group_delta"]}
report={"schema":"quartz_general_c19_C09_independent_source_and_signed_utility_audit_v1",
 "status":"PASS","all_test_random_streams_replayed":len(test),
 "independent_latent_parent_groups":48,
 "separate_posterior_update_algorithms":"PASS",
 "source_and_test_group_split":"PASS",
 "train_only_cost_medians":"PASS",
 "train_only_baselines":"PASS",
 "Brier_MSE_and_three_price_net_values":"PASS",
 "source_world_sha256":sha(WORLDS),"train_cal_raw_sha256":sha(TRAIN),
 "untouched_test_raw_sha256":sha(TEST),"frozen_model_sha256":sha(MODEL),
 "heldout_report_sha256":sha(HELDOUT),
 "domains":independent,
 "actual_native_MCTS_game_regret":"NOT_TESTED_NOT_ESTIMABLE",
 "counterfactual_avoided_CPU_not_live_runtime":"TRUE"}
p=HOME/"evidence/C09_INDEPENDENT_AUDIT.json"
p.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print("C19_C09_INDEPENDENT",json.dumps(report,sort_keys=True))
