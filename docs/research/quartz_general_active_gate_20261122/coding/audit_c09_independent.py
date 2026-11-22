#!/usr/bin/env python3
"""C09 independent scientific audit: source RNG, exact Fraction EVSI, real STOP.

Deliberately imports no C22 source generator, adapter, native decision, cost fitter
or C08 scoring function. Rust same-StdRng replay explicitly classified as not
an independent algorithm.
"""
from pathlib import Path
from fractions import Fraction as F
from collections import defaultdict
from itertools import product
import hashlib,json,math,random,statistics,subprocess,tempfile

ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
H=Path(__file__).resolve().parent.parent
WORLD=H/"SOURCE_WORLDS_C22_FROZEN.json"
TRAIN=H/"evidence/C07_TRAIN_CAL/RAW_SIGNED_CALCULATIONS.jsonl"
TEST=H/"evidence/C08_TEST/RAW_SIGNED_CALCULATIONS.jsonl"
ACTIVE=H/"evidence/C08_TEST/C22_ACTIVE_ALL_POLICIES.jsonl"
REPORT=H/"evidence/C08_TEST/C08_ACTUAL_ACTIVE_RESULTS.json"
PRICE=H/"FROZEN_C22_RESOURCE_PRICE.json"
BIN=ROOT/"target/release/quartz_c22_active_gate"
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def readlines(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
worlds={w["id"]:w for w in json.loads(WORLD.read_text())["worlds"]}
train=readlines(TRAIN);test=readlines(TEST);active=readlines(ACTIVE)
science=json.loads(REPORT.read_text())
model=json.loads(PRICE.read_text())
assert sha(WORLD)=="f7bffc98710bae151d0ac909ddeb19fc04f74d51a4273f6b16f7e7583f61398d"
assert sha(TEST)=="807fff205d6b7c4b71d9cbd9e57e05f472635ee609f547b8ac68f9588d769dd0"
assert sha(REPORT)=="b91d6fff56466047b56361ce4cc87b8f60dad365ed4137bf4e3baea7bbb29c64"
assert sha(PRICE)=="7814df9043a2814dfc198f0407ca35fc5808236506e2347ea7983e59db0bdab1"
assert len(test)==144 and len(active)==576
assert len({r["group_id"] for r in test})==48
assert not {r["group_id"] for r in test}&{r["group_id"] for r in train}
def seed(w,rep):
 msg=f"{w['id']}:{w['seed']}:rep:{rep}".encode()
 return int.from_bytes(hashlib.sha256(msg).digest()[:8],"big")
def argmax(x):return max(range(len(x)),key=lambda k:(x[k],-k))
def categorical(probs,rng):
 u=rng.random();z=0.
 for j,p in enumerate(probs):
  z+=p
  if u<z:return j
 return len(probs)-1
def updated(b,L,y):
 val=[b[j]*L[j][y] for j in range(len(b))]
 z=sum(val)
 return [p/z for p in val]
def scores(b,U):
 return [sum(b[t]*U[t][a] for t in range(len(b))) for a in range(len(U[0]))]
def replay_source(r,w):
 rng=random.Random(seed(w,r["rep"]))
 if r["family"]=="beta_arms":
  theta=w["theta_offline_true_arm_means"]
  n0=w["initial_draws_per_arm"];ne=w["extra_draws_per_arm"]
  init=[[int(rng.random()<theta[j]) for _ in range(n0)] for j in range(w["K"])]
  m0=[(2+sum(v))/(4+n0) for v in init]
  a0=argmax(m0)
  additional=[[int(rng.random()<theta[j]) for _ in range(ne)] for j in range(w["K"])]
  m1=[(2+sum(init[j])+sum(additional[j]))/(4+n0+ne) for j in range(w["K"])]
  a1=argmax(m1)
  z=theta[a1]-theta[a0]
  assert ne==r["planned_extra_per_arm"]
 else:
  prior=w["initial_prior"][:]
  L=w["observation_likelihood_state_outcome_PUBLIC"]
  U=w["utility_state_action_matrix_PUBLIC"]
  theta=w["theta_offline_true_state"]
  init=[]
  for _ in range(w["initial_observations"]):
   y=categorical(L[theta],rng);init.append(y);prior=updated(prior,L,y)
  m0=scores(prior,U);a0=argmax(m0)
  additional=[]
  for _ in range(w["extra_observations"]):
   y=categorical(L[theta],rng);additional.append(y);prior=updated(prior,L,y)
  m1=scores(prior,U);a1=argmax(m1)
  z=U[theta][a1]-U[theta][a0]
  assert w["extra_observations"]==r["planned_extra"]
 assert init==r["initial_observations"]
 assert additional==r["extra_observations"]
 assert a0==r["pre_action"] and a1==r["post_action"]
 assert r["label"]==int(a0!=a1)
 assert abs(z-r["signed_gain"])<=1e-12
 assert all(abs(x-y)<1e-12 for x,y in zip(m0,r["initial_posterior_expected_utilities"]))
 assert all(abs(x-y)<1e-12 for x,y in zip(m1,r["final_posterior_expected_utilities"]))
 assert abs(sorted(m0,reverse=True)[0]-sorted(m0,reverse=True)[1]-r["gap"])<1e-12
for r in test:
 assert r["split"]=="test" and worlds[r["group_id"]]["split"]=="test"
 replay_source(r,worlds[r["group_id"]])

def beta_fraction_pmf(a,b,n):
 a=F(a);b=F(b);p=F(1)
 for t in range(n):p*=(b+t)/(a+b+t)
 out=[p]
 for k in range(n):
  p*=F(n-k,k+1)*(a+k)/(b+n-k-1)
  out.append(p)
 assert sum(out)==1
 return out
def beta_fraction_EVSI(row,world):
 init=row["initial_observations"]
 n=world["extra_draws_per_arm"]
 arms=[(F(2+sum(seq)),F(2+len(seq)-sum(seq))) for seq in init]
 weights=[]
 for a,b in arms:
  probs=beta_fraction_pmf(a,b,n)
  weights.append([((a+k)/(a+b+n),probs[k]) for k in range(n+1)])
 support=sorted({v for z in weights for v,p in z})
 before=F(0);expected=F(0);last=F(0)
 for v in support:
  curr=math.prod(sum(p for score,p in z if score<=v) for z in weights)
  expected+=(v-before)*(1-last)
  before=v;last=curr
 assert last==1
 initial=max(a/(a+b) for a,b in arms)
 return expected-initial
def sensor_ordered_EVSI(row,world):
 L=world["observation_likelihood_state_outcome_PUBLIC"]
 U=world["utility_state_action_matrix_PUBLIC"]
 b=world["initial_prior"][:]
 for y in row["initial_observations"]:b=updated(b,L,y)
 m0=max(scores(b,U))
 n=world["extra_observations"]
 value=0.;mass=0.
 for ys in product(range(3),repeat=n):
  weights=[b[t]*math.prod(L[t][y] for y in ys) for t in range(4)]
  mass+=sum(weights)
  value+=max(sum(weights[t]*U[t][a] for t in range(4)) for a in range(3))
 assert abs(mass-1)<1e-9
 return max(0.,value-m0)

lookup={(r["lambda"],r["group_id"],r["rep"]):r for r in active}
assert len(lookup)==len(active)
maxerror=0.;beta_exact_count=0;sensor_exact_count=0
for r in test:
 if r["rep"]!=0:continue
 world=worlds[r["group_id"]]
 v=lookup[(0.004,r["group_id"],r["rep"])]["results"]["FULL_EXACT"]["gross_predictive_EVSI"]
 assert v is not None
 if r["family"]=="beta_arms":
  exact=float(beta_fraction_EVSI(r,world));beta_exact_count+=1
 else:
  exact=sensor_ordered_EVSI(r,world);sensor_exact_count+=1
 maxerror=max(maxerror,abs(exact-v))
 assert abs(exact-v)<1e-10,(r["group_id"],exact,v)
assert beta_exact_count==sensor_exact_count==24

for lam in (0.001,0.004,0.02,0.1):
 for fam in ("beta_arms","finite_sensor"):
  entries=[r["results"] for r in active if r["lambda"]==lam and r["source_family"]==fam]
  assert len(entries)==72
  recorded=science["price_scenarios"][str(lam)]["families"][fam]
  for policy in ("NEVER","ALWAYS","CERTIFIED_ONLY","GATED","FULL_EXACT"):
   measured=[x[policy] for x in entries]
   resultmean=statistics.mean(v["net_utility_change_after_measured_cost"] for v in measured)
   timemean=statistics.mean(v["active_total_internal_us"] for v in measured)
   draws=sum(v["actual_future_draw_count"] for v in measured)
   assert abs(resultmean-recorded["policy_mean_realized_net_gain_vs_free_STOP"][policy])<1e-12
   assert abs(timemean-recorded["policy_mean_actual_native_elapsed_us"][policy])<1e-12
   assert draws==recorded["policy_total_ACTUALLY_EXECUTED_future_draws"][policy]
  gated=[e["GATED"] for e in entries]
  assert sum(not p["COMPUTE"] for p in gated)==recorded["GATED_n_true_STOP"]
  assert sum(p["decision_reason"]=="CERTIFIED_ZERO" for p in gated)==recorded["GATED_n_CERTIFIED_ZERO"]
  assert sum(p["decision_reason"]=="PRICE_UPPER_SCREEN" for p in gated)==recorded["GATED_n_PRICE_UPPER_SCREEN"]
  assert sum(p["decision_reason"]=="EXACT_STOP" for p in gated)==recorded["GATED_n_EXACT_STOP"]
  assert all(p["actual_future_draw_count"]==0 and p["true_signed_terminal_utility_change"]==0. for p in gated if not p["COMPUTE"])
  for baseline,p in zip([e["ALWAYS"] for e in entries],gated):
   if p["certified_zero_information"]:
    assert baseline["before_action"]==baseline["after_action"]
    assert baseline["true_signed_terminal_utility_change"]==0.
   if p["COMPUTE"]:
    assert baseline["after_action"]==p["after_action"]
    assert abs(baseline["true_signed_terminal_utility_change"]-p["true_signed_terminal_utility_change"])<1e-12

# Deterministic replay using the SAME compiled Rust StdRng engine.
# Separate math/source replay above is independent; same-code RNG replay is provenance only.
path=H/"evidence/C08_TEST/NATIVE_ACTIVE_INPUT_LAMBDA_0p004.json"
replay=json.loads(subprocess.check_output([str(BIN),str(path)],text=True))
assert replay["number_cases"]==144
replayed=0
for r in replay["rows"]:
 ref=lookup[(0.004,r["group_id"],r["rep"])]
 assert r["id"]==ref["id"]
 for name in ("NEVER","ALWAYS","CERTIFIED_ONLY","GATED","FULL_EXACT"):
  expected=ref["results"][name]
  actual=r["results"][name]
  for key in ("COMPUTE","decision_reason","actual_future_draw_count","before_action","after_action"):
   assert expected[key]==actual[key]
  assert abs(expected["true_signed_terminal_utility_change"] -
             actual["true_signed_terminal_utility_change"])<1e-12
 replayed+=1
assert replayed==144
report={"schema":"quartz_C22_C09_independent_source_and_actual_native_stop_audit_v1",
 "status":"PASS","original_TEST_python_future_RNG_replays":144,
 "original_latent_source_groups_TEST":48,
 "Beta_exact_Fraction_CDF_n20_bounded_parent_checks":beta_exact_count,
 "sensor_independent_ordered_likelihood_parent_checks":sensor_exact_count,
 "max_abs_independent_gross_EVSI_error":maxerror,
 "all_four_price_policy_net_elapsed_draws_independently_aggregated":True,
 "Rust_same_engine_StdRng_control_path_deterministic_replays":replayed,
 "Rust_RNG_independent_algorithm":"NOT_IMPLEMENTED",
 "certified_zero_bound_no_action_change_vs_always_CONTROL":"PASS",
 "actual_STOP_all_zero_additional_draws_and_zero_signed_change":"PASS",
 "measured_NEVER_native_wrapper_cost_distinct_from_free_mathematical_STOP":True,
 "mathematical_free_immediate_STOP_reference_net":0.,
 "C04_new_source_bank_sha256":sha(WORLD),
 "C07A_train_cost_lock_sha256":sha(PRICE),
 "C08A_test_source_raw_sha256":sha(TEST),
 "C08_native_results_sha256":sha(ACTIVE),
 "C08_final_science_sha256":sha(REPORT),
 "original_adaptive_MCTS_true_utility":"NOT_ESTIMABLE",
 "overall_experiment_cpu_savings":"NOT_CLAIMED_separate_controls_executed"}
p=H/"evidence/C09_INDEPENDENT_AUDIT.json"
p.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print("C22_C09_INDEPENDENT",json.dumps(report,sort_keys=True))
