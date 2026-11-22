#!/usr/bin/env python3
"""Prospectively freeze two structurally distinct source-world populations.

Latent truth is locked here but inaccessible to predictor rows at runtime.
No outcomes / source revision or gain labels are generated before C7.
"""
from pathlib import Path
import hashlib,json,random
ROOT=next(p for p in Path(__file__).resolve().parents if (p/".git").exists())
HOME=Path(__file__).resolve().parent.parent
def sha(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def discrete(v,u):
 tot=0.
 for i,p in enumerate(v):
  tot+=p
  if u<tot:return i
 return len(v)-1
def norm(xs):
 s=sum(xs)
 return [x/s for x in xs]
worlds=[]
for i in range(120):
    split="train" if i<72 else ("cal" if i<96 else "test")
    # World family A: independent Bernoulli utility arms, true means hidden online.
    seed_a=20010020+i*6361
    rng=random.Random(seed_a)
    k=2+i%3
    true_mu=[rng.betavariate(2.0,2.0) for _ in range(k)]
    worlds.append({"id":f"arms-{i:03d}","family":"beta_arms",
        "split":split,"seed":seed_a,
        "theta_offline_true_arm_means":true_mu,
        "K":k,"initial_draws_per_arm":3,"extra_draws_per_arm":10,
        "replications":3,"reward_beta_prior":[2.0,2.0],
        "true_payoff_function":"U(action,theta)=theta[action]"})
    # World family B: fixed but unknown latent discrete state and known noisy sensors.
    seed_b=20020020+i*9871
    rng=random.Random(seed_b)
    prior=norm([rng.gammavariate(2.0,1.0) for _ in range(4)])
    hidden=discrete(prior,rng.random())
    utility=[[max(0.05,min(0.95,rng.betavariate(2.0,2.0))) for _ in range(3)] for _ in range(4)]
    channel=[]
    for theta in range(4):
        noise=[rng.gammavariate(1.0,1.0) for _ in range(3)]
        signal=[0.13+0.74*(int(j==theta%3))+0.12*noise[j] for j in range(3)]
        channel.append(norm(signal))
    worlds.append({"id":f"sensor-{i:03d}","family":"finite_sensor",
       "split":split,"seed":seed_b,
       "theta_offline_true_state":hidden,
       "initial_prior":prior,"utility_state_action_matrix_PUBLIC":utility,
       "observation_likelihood_state_outcome_PUBLIC":channel,
       "K":3,"initial_observations":2,"extra_observations":8,
       "replications":3,
       "true_payoff_function":"U(action,theta)=PUBLIC_matrix[offline_true_state][action]"})
assert len(worlds)==240
assert {k:sum(r["split"]==k for r in worlds if r["family"]=="beta_arms") for k in ("train","cal","test")}=={"train":72,"cal":24,"test":24}
assert {k:sum(r["split"]==k for r in worlds if r["family"]=="finite_sensor") for k in ("train","cal","test")}=={"train":72,"cal":24,"test":24}
assert all(r["K"]>=2 for r in worlds)
record={"schema":"quartz_general_c20_prospective_two_domain_sources_v1",
"provenance":"independent parent latent worlds fixed before any future computation observations",
"source_parent_C18":"097255bd00ffa436c3d43704ed201815e7cbb8ea",
"experiment_primary_utility_per_microsecond_price":0.004,
"prespecified_sensitivity_prices":[0.001,0.004,0.010],
"sample_splits_by_parent":{"train":72,"cal":24,"test":24},
"replications_per_parent":3,
"families":["beta_arms","finite_sensor"],
"worlds":worlds,
"trained_forecast_must_not_read_offline_truth":True,
"source_id_group_disjoint":True,
"no_observation_labels_yet":True}
out=HOME/"SOURCE_WORLDS_C20_FROZEN.json"
out.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
print("C20_C04_SOURCE_LOCK",json.dumps({"parents":len(worlds),"each_family_parents":120,
"replications":3,"source_SHA256":sha(out),"C19_lambda":0.004},sort_keys=True))
