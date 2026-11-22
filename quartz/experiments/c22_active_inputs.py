"""C22 source-only public decision adapter, distinct from the offline secret sampler.

Only initial evidence and parent-fixed public task law feed the native selector.
Source theta is exported under a separate offline-only key, never inside public.
The original Python future observations are NOT used to choose native actions.
"""
from __future__ import annotations
import hashlib,math
from typing import Mapping,Sequence

def public_task(row:Mapping,world:Mapping)->dict:
    if row["group_id"]!=world["id"] or row["family"]!=world["family"]:
        raise ValueError("source original parent identity mismatch")
    init=row["initial_observations"]
    if row["family"]=="beta_arms":
        k=world["K"]
        n0=world["initial_draws_per_arm"]
        if len(init)!=k or any(len(x)!=n0 for x in init):
            raise ValueError("malformed original initial Bernoulli evidence")
        if any(bit not in (0,1) for seq in init for bit in seq):
            raise ValueError("not an observed Bernoulli bit")
        posterior=[[float(2+sum(seq)),float(2+len(seq)-sum(seq))] for seq in init]
        return {"family":"beta_arms","posterior":posterior,
                "extra_per_arm":[int(world["extra_draws_per_arm"])]*k}
    if row["family"]=="finite_sensor":
        n0=world["initial_observations"]
        if len(init)!=n0 or any(y not in (0,1,2) for y in init):
            raise ValueError("malformed public categorical initial evidence")
        L=world["observation_likelihood_state_outcome_PUBLIC"]
        U=world["utility_state_action_matrix_PUBLIC"]
        b=list(world["initial_prior"])
        for y in init:
            weights=[b[t]*L[t][y] for t in range(4)]
            total=math.fsum(weights)
            if total<=0:raise ValueError("impossible categorical observation")
            b=[v/total for v in weights]
        return {"family":"finite_sensor","prior":b,"likelihood":L,
                "utilities":U,"extra":int(world["extra_observations"])}
    raise ValueError("unknown publicly specified source family")

def workload_bucket(public:Mapping)->str:
    if public["family"]=="beta_arms":
        n=public["extra_per_arm"][0]
        assert all(k==n for k in public["extra_per_arm"])
        return "beta_K"+str(len(public["posterior"]))+"_n"+str(n)
    if public["family"]=="finite_sensor":
        return "sensor_n"+str(public["extra"])
    raise ValueError("unknown workload bucket")

def source_secret(row:Mapping,world:Mapping)->dict:
    if row["group_id"]!=world["id"] or row["split"]!=world["split"]:
        raise ValueError("offline source mismatch")
    seedtext=f"C22-ACTIVE:{world['id']}:{row['rep']}".encode()
    seed=int.from_bytes(hashlib.sha256(seedtext).digest()[:8],"big")
    if world["family"]=="beta_arms":
        return {"seed":seed,"theta_offline_arm_means":list(world["theta_offline_true_arm_means"])}
    if world["family"]=="finite_sensor":
        return {"seed":seed,"theta_offline_state":world["theta_offline_true_state"]}
    raise ValueError("source unknown")

def make_case(row:Mapping,world:Mapping,prices:Mapping|None=None)->dict:
    pub=public_task(row,world)
    if prices is None:
        # C07A TRAIN-only calibration trial. This price is NOT used at TEST.
        cost={"task_est_us":0.5,"exact_est_us":1.0}
    else:
        bucket=workload_bucket(pub)
        details=prices["price_by_bucket"][bucket]
        cost={"task_est_us":details["task_native_median_us"],
              "exact_est_us":details["exact_native_median_us"]}
    if cost["task_est_us"]<=0 or cost["exact_est_us"]<=0:
        raise ValueError("positive precommitted cost forecasts required")
    return {"id":row["group_id"]+"-rep"+str(row["rep"]),
            "group_id":row["group_id"],"split":row["split"],"rep":row["rep"],
            "public":pub,"offline":source_secret(row,world),"price":cost}

def build_request(rows:Sequence[Mapping],worlds:Mapping,lam:float,
                  prices:Mapping|None=None)->dict:
    if not rows or not math.isfinite(lam) or lam<=0:raise ValueError("nonempty priced request required")
    cases=[make_case(row,worlds[row["group_id"]],prices) for row in rows]
    if len({x["id"] for x in cases})!=len(cases):raise ValueError("duplicate original parent/rep")
    return {"cmd":"active_experiment","lambda":lam,"cases":cases}
