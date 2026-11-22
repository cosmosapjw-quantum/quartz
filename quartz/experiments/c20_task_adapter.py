"""C20 source observation adapter. Splits OFFLINE raw labels from online pre-observation inputs.

Only this adapter knows the generative task law; exact generic EVSI consumes
posterior and public utility/channel. Future samples, realized utility, hidden
theta and realised wall time are never supplied to its inputs.
"""
from __future__ import annotations
import math
from typing import Mapping
from quartz.experiments.c20_predictive_value import beta_evsi,sensor_evsi

def initial_only(source:Mapping)->dict:
    kind=source["family"]
    init=source["initial_observations"]
    if kind=="beta_arms":
        if not (2<=len(init)<=4) or any(len(v)!=3 for v in init):
            raise ValueError("Beta initial sample count mismatch")
        if any(bit not in (0,1) for arm in init for bit in arm):
            raise ValueError("non-Bernoulli initial outcome")
        return {"family":"beta_arms","n_actions":len(init),
           "initial_observations":[list(arm) for arm in init],
           "extra_observation_budget":10}
    if kind=="finite_sensor":
        if len(init)!=2 or any(y not in (0,1,2) for y in init):
            raise ValueError("sensor initial observation contract violated")
        if len(source["utilities_PUBLIC"])!=4 or any(len(row)!=3 for row in source["utilities_PUBLIC"]):
            raise ValueError("expected three terminal actions for frozen sensor task")
        return {"family":"finite_sensor","n_actions":3,
         "initial_observations":list(init),
         "prior":list(source["prior_PUBLIC"]),
         "likelihood":[list(v) for v in source["likelihood_PUBLIC"]],
         "utilities":[list(v) for v in source["utilities_PUBLIC"]],
         "extra_observation_budget":8}
    raise ValueError("unknown task adapter")

def eval_initial(initial:Mapping)->dict:
    kind=initial["family"]
    if kind=="beta_arms":
        if set(initial)!={"family","n_actions","initial_observations","extra_observation_budget"}:
            raise ValueError("oracle/future fields forbidden")
        alpha_beta=[]
        for seq in initial["initial_observations"]:
            s=sum(seq)
            alpha_beta.append((2.0+s,2.0+len(seq)-s))
        if len(alpha_beta)!=initial["n_actions"]:raise ValueError("action count mismatch")
        return beta_evsi(alpha_beta,int(initial["extra_observation_budget"]))
    if kind=="finite_sensor":
        if set(initial)!={"family","n_actions","initial_observations",
                          "prior","likelihood","utilities","extra_observation_budget"}:
            raise ValueError("oracle/future fields forbidden")
        prior=list(initial["prior"])
        L=initial["likelihood"]
        for y in initial["initial_observations"]:
            evidence=[prior[t]*L[t][y] for t in range(len(prior))]
            z=math.fsum(evidence)
            if z<=0:raise ValueError("impossible initial observation")
            prior=[v/z for v in evidence]
        return sensor_evsi(prior,L,initial["utilities"],
                           int(initial["extra_observation_budget"]))
    raise ValueError("unknown task adapter")

def resource_bucket(initial:Mapping)->str:
    if initial["family"]=="beta_arms":
        return "beta_arms_K"+str(initial["n_actions"])
    if initial["family"]=="finite_sensor":
        return "finite_sensor_K3"
    raise ValueError("unknown declared computation family")
