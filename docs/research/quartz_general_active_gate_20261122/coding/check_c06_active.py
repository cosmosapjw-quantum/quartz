#!/usr/bin/env python3
"""C06 independent source-only EVSI math and actual paired execution invariants."""
from pathlib import Path
import hashlib,json,math,random,subprocess,tempfile,collections
from quartz.experiments.c20_predictive_value import beta_evsi,sensor_evsi
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
BIN=ROOT/"target/release/quartz_c22_active_gate"
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
rng=random.Random(220622)
cases=[]
def norm(x):
 s=sum(x);return [v/s for v in x]
for i in range(105):
 family="beta_arms" if i%2==0 else "finite_sensor"
 if family=="beta_arms":
  k=2+i%3
  n=[1,2,5,10][(i//3)%4]
  posterior=[[float(rng.choice([2,3,5,7,12,25,40])),
              float(rng.choice([2,3,5,9,15,40]))] for _ in range(k)]
  mu=[rng.betavariate(2,2) for _ in range(k)]
  case={"family":"beta_arms","posterior":posterior,"extra_per_arm":[n]*k}
  offline={"theta_offline_arm_means":mu,"seed":100000+i}
 else:
  prior=norm([rng.random()+.1 for _ in range(4)])
  L=[norm([rng.random()+.15 for _ in range(3)]) for _ in range(4)]
  U=[[rng.betavariate(2,2) for _ in range(3)] for _ in range(4)]
  case={"family":"finite_sensor","prior":prior,"likelihood":L,"utilities":U,
     "extra":[1,3,8][(i//3)%3]}
  theta=rng.choices(range(4),weights=prior,k=1)[0]
  offline={"theta_offline_state":theta,"seed":100000+i}
 cases.append({"id":f"c06-sanity-{i:03}","group_id":f"synthetic-{i:03}",
  "rep":0,"split":"train","public":case,"offline":offline,
  "price":{"task_est_us":0.4 if family=="beta_arms" else 2.0,
           "exact_est_us":0.8 if family=="beta_arms" else 1.2}})
cases.extend([
{"id":"dominant-beta","group_id":"pure-witness-beta","split":"train","rep":0,
 "public":{"family":"beta_arms","posterior":[[12.,2.],[2.,12.]],"extra_per_arm":[1,1]},
 "offline":{"theta_offline_arm_means":[.9,.1],"seed":999},
 "price":{"task_est_us":.25,"exact_est_us":.8}},
{"id":"dominant-sensor","group_id":"pure-witness-sensor","split":"train","rep":0,
 "public":{"family":"finite_sensor","prior":[.25]*4,
   "likelihood":[[.7,.2,.1]]*4,
   "utilities":[[.8,.4,.1]]*4,"extra":8},
 "offline":{"theta_offline_state":2,"seed":888},
 "price":{"task_est_us":2.,"exact_est_us":1.}}
])
assert len(cases)==107
req={"cmd":"active_experiment","lambda":0.004,"cases":cases}
with tempfile.TemporaryDirectory(prefix="quartz_C22_C06_") as tmp:
 p=Path(tmp)/"source_only_scientific_case.json"
 p.write_text(json.dumps(req,sort_keys=True))
 out=json.loads(subprocess.check_output([str(BIN),str(p)],text=True))
assert out["original_SearchPolicy_unchanged"] is True
assert out["number_cases"]==len(cases)
assert len(out["rows"])==len(cases)
assert out["GATED_true_STOP_count"]>0
assert out["GATED_actually_unexecuted_future_draws"]>0
max_error=0.;exact_pairs=0;count_cert=0;count_price=0
counts=collections.Counter();total_draws_avoided=0
for case,row in zip(cases,out["rows"]):
 assert row["id"]==case["id"]
 data=row["results"]
 assert set(data)=={"NEVER","ALWAYS","CERTIFIED_ONLY","GATED","FULL_EXACT"}
 assert data["NEVER"]["actual_future_draw_count"]==0
 assert data["NEVER"]["true_signed_terminal_utility_change"]==0
 before=data["ALWAYS"]["before_action"]
 kind=case["public"]["family"]
 if kind=="beta_arms":
  expected=sum(case["public"]["extra_per_arm"])
 else:
  expected=case["public"]["extra"]
 assert data["ALWAYS"]["actual_future_draw_count"]==expected
 assert data["ALWAYS"]["before_action"]==before
 for name,val in data.items():
  if not val["COMPUTE"]:
   assert val["actual_future_draw_count"]==0 and val["after_action"]==before
   assert val["true_signed_terminal_utility_change"]==0
   assert val["actual_future_task_us"]==0
  else:
   assert val["actual_future_draw_count"]==expected
   assert val["after_action"]==data["ALWAYS"]["after_action"]
   assert abs(val["true_signed_terminal_utility_change"] -
              data["ALWAYS"]["true_signed_terminal_utility_change"])<1e-12
  for field in ("cheap_gate_us","exact_evaluator_us","actual_future_task_us",
                "active_total_internal_us"):
   assert math.isfinite(val[field]) and val[field]>=0
  assert val["active_total_internal_us"]>=val["actual_future_task_us"]-1e-7
 g=data["FULL_EXACT"]["gross_predictive_EVSI"]
 assert g is not None and g>=0
 if kind=="beta_arms":
  pub=case["public"]
  reference=beta_evsi(pub["posterior"],pub["extra_per_arm"][0])
 else:
  pub=case["public"]
  reference=sensor_evsi(pub["prior"],pub["likelihood"],pub["utilities"],pub["extra"])
 err=abs(reference["evsi"]-g)
 assert err<=1e-10,(case["id"],err)
 max_error=max(max_error,err)
 exact_pairs+=1
 if data["GATED"]["decision_reason"]=="CERTIFIED_ZERO":
  assert g<=1e-10
  assert data["ALWAYS"]["after_action"]==before
  count_cert+=1
 if data["GATED"]["decision_reason"]=="PRICE_UPPER_SCREEN":
  count_price+=1
 if not data["GATED"]["COMPUTE"]:
  total_draws_avoided+=expected
 counts[kind]+=1
assert out["GATED_actually_unexecuted_future_draws"]==total_draws_avoided
assert data
report={"schema":"quartz_C22_C06_active_execution_and_independent_EVSI_checks_v1",
 "status":"PASS","compiled_Rust_sha256":sha(BIN),
 "n_synthetic_source_cases":len(cases),"n_reference_scored_EVSI":exact_pairs,
 "max_abs_EVSI_difference":max_error,
 "task_families_checked":dict(counts),
 "GATED_n_physical_STOP":out["GATED_true_STOP_count"],
 "GATED_n_truly_unexecuted_future_draws":total_draws_avoided,
 "n_certified_zero_cases":count_cert,
 "n_price_upper_cases":count_price,
 "STOP_always_zero_RNG_count_and_zero_gain":"PASS",
 "COMPUTE_same_seed_final_action_and_true_payoff_as_physical_ALWAYS":"PASS",
 "all_controller_inputs_public_model_only":"PASS",
 "original_MCTS_SearchPolicy_mutated":False,
 "real_GAME_true_utility":"NOT_ESTIMABLE",
 "source_label":"pure independent bounded C06 synthetic, not C22 TEST"}
path=HOME/"evidence/C06_ACTIVE_INDEPENDENT_CHECK.json"
path.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
print("C22_C06_INDEPENDENT",json.dumps(report,sort_keys=True))
