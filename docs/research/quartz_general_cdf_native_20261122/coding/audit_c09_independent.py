#!/usr/bin/env python3
"""C09 independent scientific replayer and exact rational CDF/full-grid controls.

No imports of C21 predictor, collector, initial-only adapter, cost fitter or C08
scorer. Calls immutable Rust task sampler ONLY for deterministic native-source
replay; does not independently implement StdRng random generator.
"""
from pathlib import Path
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
import hashlib,json,math,random,statistics,subprocess
R=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
H=Path(__file__).resolve().parent.parent
BANK=H/"SOURCE_WORLDS_C21_FROZEN.json"
TRAIN=H/"evidence/C07_TRAIN_CAL/RAW_SIGNED_CALCULATIONS.jsonl"
TEST=H/"evidence/C08_TEST/RAW_SIGNED_CALCULATIONS.jsonl"
NATIVE=H/"evidence/C08_TEST/NATIVE_TEST_SHADOW_RECORDS.jsonl"
RESULT=H/"evidence/C08_TEST/C08_HELDOUT_NATIVE_EVSI_RESULTS.json"
PRICE=H/"FROZEN_C21_RESOURCE_PRICE.json"
BINARY=R/"target/release/quartz_c21_fast_evsi"
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def lines(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
worlds={w["id"]:w for w in json.loads(BANK.read_text())["worlds"]}
train=lines(TRAIN);rows=lines(TEST);pred=lines(NATIVE)
report=json.loads(RESULT.read_text());cost=json.loads(PRICE.read_text())
assert sha(BANK)=="036b7c2e77d998558de46ff80845e4b0be382ff4a35d365b03a7174d6c98dad8"
assert sha(RESULT)=="5feecc957bf679b75b6feccc64e7fcbdb70d206ceca37ef477b19b74a67cdd16"
assert sha(TEST)==report["TEST_raw_source_SHA256"]
assert sha(PRICE)==report["TRAIN_only_price_model_SHA256"]
assert len(rows)==len(pred)==144
# Original Python source rows interleave families; C08 predictions group Beta then sensor.
# Match the frozen original source identity, never index-align unrelated rows.
by_key={(r["group_id"],r["rep"],r["family"]):r for r in rows}
assert len(by_key)==len(rows)
assert set(by_key)=={(p["group_id"],p["rep"],p["family"]) for p in pred}
paired=[(by_key[(p["group_id"],p["rep"],p["family"])],p) for p in pred]
assert len({r["group_id"] for r in rows})==48
assert not {r["group_id"] for r in rows}&{r["group_id"] for r in train}

def seed(w,r):
 msg=f"{w['id']}:{w['seed']}:rep:{r['rep']}".encode()
 return int.from_bytes(hashlib.sha256(msg).digest()[:8],"big")
def pick(v):return max(range(len(v)),key=lambda i:(v[i],-i))
def samplecat(probs,rng):
 u=rng.random();c=0.
 for i,p in enumerate(probs):
  c+=p
  if u<c:return i
 return len(probs)-1
def update(prior,L,y):
 weights=[prior[i]*L[i][y] for i in range(len(prior))]
 z=sum(weights)
 return [x/z for x in weights]
def means(b,U):
 return [sum(b[i]*U[i][a] for i in range(len(b))) for a in range(len(U[0]))]

def replay_python_source(r,w):
 rng=random.Random(seed(w,r))
 if r["family"]=="beta_arms":
  true=w["theta_offline_true_arm_means"]
  init=[[int(rng.random()<true[j]) for _ in range(3)] for j in range(w["K"])]
  before=[(2+sum(v))/7 for v in init];a0=pick(before)
  new=[[int(rng.random()<true[j]) for _ in range(10)] for j in range(w["K"])]
  after=[(2+sum(init[j])+sum(new[j]))/17 for j in range(w["K"])]
  a1=pick(after)
  gain=true[a1]-true[a0]
 else:
  theta=w["theta_offline_true_state"]
  L=w["observation_likelihood_state_outcome_PUBLIC"]
  U=w["utility_state_action_matrix_PUBLIC"]
  b=w["initial_prior"][:];init=[]
  for _ in range(2):
   y=samplecat(L[theta],rng);init.append(y);b=update(b,L,y)
  before=means(b,U);a0=pick(before)
  new=[]
  for _ in range(8):
   y=samplecat(L[theta],rng);new.append(y);b=update(b,L,y)
  after=means(b,U);a1=pick(after)
  gain=U[theta][a1]-U[theta][a0]
 assert init==r["initial_observations"]
 assert new==r["extra_observations"]
 assert a0==r["pre_action"] and a1==r["post_action"]
 assert r["label"]==int(a0!=a1)
 assert abs(gain-r["signed_gain"])<1e-12
 assert abs(sorted(before,reverse=True)[0]-sorted(before,reverse=True)[1]-r["gap"])<1e-12
 assert all(abs(a-b)<1e-12 for a,b in zip(before,r["initial_posterior_expected_utilities"]))
 assert all(abs(a-b)<1e-12 for a,b in zip(after,r["final_posterior_expected_utilities"]))
 assert r["actual_extra_cost_us"]>0
for r,p in paired:
 w=worlds[r["group_id"]]
 assert r["split"]==w["split"]=="test"
 assert r["group_id"]==p["group_id"] and r["rep"]==p["rep"]
 replay_python_source(r,w)

def betabinomial_fraction(a,b,n):
 p=F(1)
 for t in range(n):p*=F(b+t,a+b+t)
 probs=[p]
 for k in range(n):
  p*=F(n-k,k+1)*F(a+k,b+n-k-1)
  probs.append(p)
 assert sum(probs)==1
 return probs
def exact_beta_joint(r):
 ab=[(2+sum(x),2+len(x)-sum(x)) for x in r["initial_observations"]]
 probs=[betabinomial_fraction(a,b,10) for a,b in ab]
 total=F(0)
 for ks in product(range(11),repeat=len(ab)):
  chance=math.prod(probs[j][k] for j,k in enumerate(ks))
  val=max(F(ab[j][0]+k,ab[j][0]+ab[j][1]+10)
          for j,k in enumerate(ks))
  total+=chance*val
 return total-max(F(a,a+b) for a,b in ab)
def full_ordered_sensor_EVSI(r,w):
 L=w["observation_likelihood_state_outcome_PUBLIC"]
 U=w["utility_state_action_matrix_PUBLIC"]
 b=w["initial_prior"][:]
 for y in r["initial_observations"]:b=update(b,L,y)
 before=max(means(b,U))
 mean=0.; mass=0.
 for ys in product(range(3),repeat=8):
  posterior_mass=[b[t]*math.prod(L[t][y] for y in ys) for t in range(4)]
  prob=sum(posterior_mass)
  mass+=prob
  mean+=max(sum(posterior_mass[t]*U[t][a] for t in range(4)) for a in range(3))
 assert abs(mass-1)<1e-9
 return max(0.,mean-before)
rational_checks=0;sensor_checks=0;maxerror=0.
for r,p in paired:
 if r["rep"]!=0:continue
 if r["family"]=="beta_arms":
  e=float(exact_beta_joint(r))
  rational_checks+=1
 else:
  e=full_ordered_sensor_EVSI(r,worlds[r["group_id"]])
  sensor_checks+=1
 err=abs(e-p["predicted_gross_EVSI"])
 maxerror=max(maxerror,err)
 assert err<1e-10,(r["group_id"],err)
assert rational_checks==sensor_checks==24

# Offline native action path replay via same compiled Rust StdRng sampler.
# It confirms reproducibility/source identity but is not independent RNG code.
native_replayed=0
proc=subprocess.Popen([str(BINARY),"--server"],stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1)
try:
 for r,p in paired:
  if r["family"]!="beta_arms":continue
  posterior=[[float(2+sum(x)),float(2+len(x)-sum(x))] for x in r["initial_observations"]]
  t=r["theta_offline_true_arm_means"]
  text=f"C21-NATIVE-TASK:{r['group_id']}:{r['rep']}:{r['split']}".encode()
  seed_native=int.from_bytes(hashlib.sha256(text).digest()[:8],"big")
  req={"cmd":"bench_task","cases":[{"posterior":posterior,"extra_per_arm":[10]*len(t),
      "theta_offline_means":t,"seed":seed_native}]}
  proc.stdin.write(json.dumps(req)+"\n");proc.stdin.flush()
  out=json.loads(proc.stdout.readline())
  assert "error" not in out
  a1=int(out["results"][0]["sampled_terminal_action"])
  z=t[a1]-t[r["pre_action"]]
  assert abs(z-p["observed_signed_true_utility"])<1e-12
  assert int(a1!=r["pre_action"])==p["observed_decision_revision"]
  assert p["observed_task_us"]>0 and p["meta_native_function_us"]>0
  native_replayed+=1
finally:
 proc.stdin.write('{"cmd":"quit"}\n');proc.stdin.flush()
 proc.stdin.close();proc.wait(timeout=15)
assert native_replayed==72

def avg(x):return sum(x)/len(x)
for family in ("beta_arms","finite_sensor"):
 subset=[p for p in pred if p["family"]==family]
 assert len(subset)==72 and len({p["group_id"] for p in subset})==24
 section=report["domains"][family]
 mse=sum((p["predicted_gross_EVSI"]-p["observed_signed_true_utility"])**2
         for p in subset)/72
 mse0=sum((p["historical_C19_coarse_signed_gross"]-p["observed_signed_true_utility"])**2
         for p in subset)/72
 assert abs(mse-section["exact_EVSI_MSE_against_realized_utility"])<1e-12
 assert abs(mse0-section["historical_C19_coarse_MSE"])<1e-12
 for price in (0.001,0.004,0.01):
  always=[p["observed_signed_true_utility"]-price*p["observed_task_us"] for p in subset]
  chosen_internal=[];chosen_boundary=[]
  for p,a in zip(subset,always):
   lock=cost["price_by_declared_workload_bucket"][p["resource_bucket"]]
   gi=p["predicted_gross_EVSI"]
   internal=gi>price*(lock["train_task_median_us"]+lock["train_predictor_function_median_us"])
   boundary=gi>price*(lock["train_task_median_us"]+lock["train_predictor_boundary_median_us"])
   chosen_internal.append((a if internal else 0)-price*p["meta_native_function_us"])
   chosen_boundary.append((a if boundary else 0)-price*p["meta_native_server_roundtrip_us"])
  target=section["prices"][str(price)]["mean_net_utility_gain_vs_immediate_STOP"]
  assert abs(avg(always)-target["ALWAYS"])<1e-12
  assert abs(avg(chosen_internal)-target["C21_NATIVE_FUNCTION_CHARGED"])<1e-12
  assert abs(avg(chosen_boundary)-target["C21_NATIVE_HOST_BOUNDARY_CHARGED"])<1e-12
record={"schema":"quartz_C21_C09_independent_SOURCE_LAW_and_SHADOW_AUDIT_v1",
 "status":"PASS",
 "actual_Python_TEST_stochastic_sequence_replays":144,
 "source_beta_exact_Fraction_joint_10_per_arm_checks":rational_checks,
 "sensor_independent_ordered_3_pow_8_checks":sensor_checks,
 "max_separate_EXACT_EVSI_discrepancy":maxerror,
 "native_Rust_StdRng_task_DETERMINISTIC_replays":native_replayed,
 "native_RNG_independent_algorithm":"NOT_IMPLEMENTED; same compiled Rust source replay, Python source RNG independently replayed",
 "independent_price_cost_score_and_six_signed_MSE_checks":"PASS",
 "C04_new_source_SHA256":sha(BANK),
 "C07A_price_SHA256":sha(PRICE),
 "C08A_TEST_raw_SHA256":sha(TEST),
 "C08_native_actual_records_SHA256":sha(NATIVE),
 "C08_scientific_results_SHA256":sha(RESULT),
 "all_48_original_parent_groups_TEST_disjoint":"PASS",
 "real_neural_MCTS_true_utility":"NOT_ESTIMABLE",
 "live_ACTUATOR_savings":"NOT_MEASURED"}
out=H/"evidence/C09_INDEPENDENT_AUDIT.json"
out.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
print("C21_C09_INDEPENDENT",json.dumps(record,sort_keys=True))
