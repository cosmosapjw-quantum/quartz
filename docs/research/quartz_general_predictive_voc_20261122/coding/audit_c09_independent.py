#!/usr/bin/env python3
"""C09 independent source observation replay and alternative predictive value formulas.

Never imports C20 source generator, adapter, exact predictor, or C08 scorer.
Reconstructs raw stochastic paths and independently checks selected exact EVSI.
"""
from pathlib import Path
from collections import defaultdict
import itertools,hashlib,json,math,random,statistics
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
WORLD=HOME/"SOURCE_WORLDS_C20_FROZEN.json"
TRAIN=HOME/"evidence/C07_TRAIN_CAL/RAW_SIGNED_CALCULATIONS.jsonl"
TEST=HOME/"evidence/C08_TEST/RAW_SIGNED_CALCULATIONS.jsonl"
PRED=HOME/"evidence/C08_TEST/FROZEN_SHADOW_FORECASTS.jsonl"
REPORT=HOME/"evidence/C08_TEST/C08_HELDOUT_EXACT_EVSI.json"
COST=HOME/"FROZEN_C20_RESOURCE_PRICE.json"
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def load_rows(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
worlds={x["id"]:x for x in json.loads(WORLD.read_text())["worlds"]}
test=load_rows(TEST);train=load_rows(TRAIN)
forecasts=load_rows(PRED)
report=json.loads(REPORT.read_text());cost=json.loads(COST.read_text())
assert sha(WORLD)=="d6cf96c26d9ac913963f702d9b0418f0002038979f6a470da3865999368e4ae7"
assert sha(TEST)==report["C08A_TEST_raw_SHA256"]
assert sha(COST)==report["frozen_C07A_RESOURCE_MODEL_SHA256"]
assert len(test)==len(forecasts)==144
assert len({r["group_id"] for r in test})==48
assert not {r["group_id"] for r in test}&{r["group_id"] for r in train}
def fresh_rng(w,r):
 seedtext=f"{w['id']}:{w['seed']}:rep:{r['rep']}".encode()
 return random.Random(int.from_bytes(hashlib.sha256(seedtext).digest()[:8],"big"))
def pick(values):
 return max(range(len(values)),key=lambda j:(values[j],-j))
def sensor_sample(Lrow,rng):
 u=rng.random();s=0.
 for j,p in enumerate(Lrow):
  s+=p
  if u<s:return j
 return len(Lrow)-1
def update(b,L,y):
 nums=[b[i]*L[i][y] for i in range(len(b))]
 s=math.fsum(nums)
 return [v/s for v in nums]
def action_means(b,U):
 return [math.fsum(b[i]*U[i][a] for i in range(len(b))) for a in range(len(U[0]))]
def replay(row,w):
 rng=fresh_rng(w,row)
 if row["family"]=="beta_arms":
  true=w["theta_offline_true_arm_means"]
  initial=[[int(rng.random()<true[j]) for _ in range(3)] for j in range(w["K"])]
  means=[(2+sum(x))/7 for x in initial]
  a0=pick(means)
  extra=[[int(rng.random()<true[j]) for _ in range(10)] for j in range(w["K"])]
  after=[(2+sum(initial[j])+sum(extra[j]))/17 for j in range(w["K"])]
  a1=pick(after)
  value=true[a1]-true[a0]
 else:
  theta=w["theta_offline_true_state"]
  L=w["observation_likelihood_state_outcome_PUBLIC"]
  U=w["utility_state_action_matrix_PUBLIC"]
  b=w["initial_prior"][:]
  initial=[]
  for _ in range(2):
   y=sensor_sample(L[theta],rng);initial.append(y);b=update(b,L,y)
  means=action_means(b,U)
  a0=pick(means)
  extra=[]
  for _ in range(8):
   y=sensor_sample(L[theta],rng);extra.append(y);b=update(b,L,y)
  after=action_means(b,U);a1=pick(after)
  value=U[theta][a1]-U[theta][a0]
 assert row["initial_observations"]==initial
 assert row["extra_observations"]==extra
 assert row["pre_action"]==a0 and row["post_action"]==a1
 assert row["label"]==int(a1!=a0)
 assert abs(row["signed_gain"]-value)<1e-12
 assert abs(row["gap"]-(sorted(means,reverse=True)[0]-sorted(means,reverse=True)[1]))<1e-12
 assert all(abs(a-b)<1e-12 for a,b in zip(row["initial_posterior_expected_utilities"],means))
 assert all(abs(a-b)<1e-12 for a,b in zip(row["final_posterior_expected_utilities"],after))
 assert row["actual_extra_cost_us"]>0
def beta_probs_recurrent(a,b,n):
 p0=math.prod((b+t)/(a+b+t) for t in range(n))
 probs=[p0]
 for k in range(n):
  probs.append(probs[-1]*((n-k)/(k+1))*((a+k)/(b+n-k-1)))
 assert abs(sum(probs)-1)<1e-10
 return probs
def independent_beta_evsi(row):
 initial=row["initial_observations"]
 ab=[(2+sum(seq),2+len(seq)-sum(seq)) for seq in initial]
 ps=[beta_probs_recurrent(a,b,10) for a,b in ab]
 before=max(a/(a+b) for a,b in ab)
 total=0.;mass=0.
 for ks in itertools.product(range(11),repeat=len(ab)):
  p=math.prod(ps[j][k] for j,k in enumerate(ks))
  value=max((ab[j][0]+k)/(sum(ab[j])+10) for j,k in enumerate(ks))
  total+=p*value;mass+=p
 assert abs(mass-1)<1e-9
 return max(0.,total-before)
def independent_sensor_evsi(row,w):
 b=w["initial_prior"][:]
 L=w["observation_likelihood_state_outcome_PUBLIC"]
 U=w["utility_state_action_matrix_PUBLIC"]
 for y in row["initial_observations"]:b=update(b,L,y)
 before=max(action_means(b,U))
 # Independent alternative: enumerate all ordered 3^8 sequences, NOT sufficient counts.
 total=0.;mass=0.
 for observations in itertools.product(range(3),repeat=8):
  likelihood=[b[t]*math.prod(L[t][y] for y in observations) for t in range(4)]
  p=sum(likelihood)
  mass+=p
  total+=max(sum(likelihood[t]*U[t][a] for t in range(4)) for a in range(3))
 assert abs(mass-1)<1e-9
 return max(0.,total-before)
checked_replay=0;exact_beta=0;exact_sensor=0;max_err=0.
for row,p in zip(test,forecasts):
 w=worlds[row["group_id"]]
 assert w["split"]=="test" and w["family"]==row["family"]
 assert p["group_id"]==row["group_id"] and p["rep"]==row["rep"]
 replay(row,w);checked_replay+=1
 assert abs(p["predictive_mass"]-1)<1e-9
 assert p["evsi_exact"]>=0
 # All K2 Beta instances; one K4 to exercise expensive joint outcomes.
 if row["family"]=="beta_arms" and (row["n_actions"]==2 or (row["n_actions"]==4 and exact_beta==0)):
  value=independent_beta_evsi(row)
  max_err=max(max_err,abs(value-p["evsi_exact"]))
  assert abs(value-p["evsi_exact"])<1e-10,(value,p["evsi_exact"])
  exact_beta+=1
 # One per independent sensor parent (24), full ordered sequence reference 3^8.
 if row["family"]=="finite_sensor" and row["rep"]==0:
  value=independent_sensor_evsi(row,w)
  max_err=max(max_err,abs(value-p["evsi_exact"]))
  assert abs(value-p["evsi_exact"])<1e-10,(value,p["evsi_exact"])
  exact_sensor+=1
assert checked_replay==144 and exact_sensor==24 and exact_beta>=24
# Independently recompute gross-MSE and charged shadow paid policy score.
for fam in ("beta_arms","finite_sensor"):
 idx=[i for i,row in enumerate(test) if row["family"]==fam]
 subset=[test[i] for i in idx];pred=[forecasts[i] for i in idx]
 m=report["heldout_domains"][fam]
 exact_mse=statistics.mean((r["signed_gain"]-x["evsi_exact"])**2 for r,x in zip(subset,pred))
 coarse_mse=statistics.mean((r["signed_gain"]-x["C19_frozen_coarse_qm"])**2 for r,x in zip(subset,pred))
 assert abs(exact_mse-m["exact_EVSI_prediction_MSE"])<1e-12
 assert abs(coarse_mse-m["C19_frozen_coarse_prediction_MSE"])<1e-12
 for price in (.001,.004,.01):
  always=[r["signed_gain"]-price*r["actual_extra_cost_us"] for r in subset]
  charged=[(v if x["evsi_exact"]>price*(x["predicted_extra_task_cost_us_TRAIN_median"]+
    x["predicted_exact_evaluator_cost_us_TRAIN_median"]) else 0.)-
    price*x["measured_exact_evaluation_us"] for v,x in zip(always,pred)]
  scored=m["price_sensitivity"][str(price)]["policies_mean_paid_net_utility_vs_STOP"]
  assert abs(statistics.mean(always)-scored["ALWAYS"])<1e-12
  assert abs(statistics.mean(charged)-scored["C20_EXACT_CHARGED"])<1e-12
result={"schema":"quartz_general_c20_C09_independent_stochastic_replay_and_numeric_audit_v1",
 "status":"PASS","all_actual_TEST_RNG_observation_paths_replayed":checked_replay,
 "independent_Beta_Polya_urn_recurrence_exact_EVSI_paths":exact_beta,
 "independent_sensor_ordered_sequence_3_pow_8_EVSI_paths":exact_sensor,
 "max_reference_EVSI_abs_error":max_err,
 "independent_true_terminal_utility_gain_and_initial_posterior":"PASS",
 "independent_price_0_001_0_004_0_010_shadow_net_and_MSE":"PASS",
 "TEST_parent_disjoint":True,"C20_source_worlds_SHA256":sha(WORLD),
 "frozen_price_model_SHA256":sha(COST),"TEST_raw_SHA256":sha(TEST),
 "C08_heldout_SHA256":sha(REPORT),
 "source_law_validity":"ONLY_TWO_CORRECT_FINITE_SIMULATOR_MODELS",
 "real_adaptive_MCTS_signed_regret":"NOT_ESTIMABLE",
 "operational_CPU_savings":"NOT_OBSERVED"}
out=HOME/"evidence/C09_INDEPENDENT_AUDIT.json"
out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print("C20_C09_INDEPENDENT",json.dumps(result,sort_keys=True))
