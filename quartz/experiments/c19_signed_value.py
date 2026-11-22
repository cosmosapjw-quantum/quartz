"""Game-neutral signed computation-value predictor and SHADOW-only cost selector.

Correct task law and true theta/terminal utilities are available to the *offline
source generator and scorer only*. No game-specific heuristic, real MCTS utility
claim, operational STOP actuation, or data-dependent policy retuning.
"""
from __future__ import annotations
import math
from typing import Mapping,Sequence
import numpy as np
from quartz.experiments.c18_observation import (
    features,fit_logistic,calibrate_intercept,predict)

RIDGE=1.0
CAL_RIDGE=1.0
SIGNED_BOUND=1.0
TOL=1e-10
INITIAL_ONLY={"gap","n_actions","compute_kind"}

def _check_training(rows):
    if not rows:raise ValueError("no train rows")
    for r in rows:
        z=float(r["signed_gain"])
        y=int(r["label"])
        if y not in (0,1) or not math.isfinite(z) or not -1-TOL<=z<=1+TOL:
            raise ValueError("signed utility must lie in [-1,1], label binary")
        if y==0 and abs(z)>TOL:
            raise ValueError("no action change cannot have signed action gain")
        features(float(r["gap"]),int(r["n_actions"]))

def train_signed(rows:Sequence[Mapping])->dict:
    """Fit revision logit and changed-only signed-utility regression on TRAIN."""
    _check_training(rows)
    qcoeff=np.asarray(fit_logistic(rows),dtype=float)
    flips=[r for r in rows if int(r["label"])==1]
    if len(flips)<5:
        raise ValueError("conditional signed change not estimable with fewer than 5 flip observations")
    x=np.stack([features(float(r["gap"]),int(r["n_actions"])) for r in flips])
    y=np.asarray([float(r["signed_gain"]) for r in flips],dtype=float)
    # One ridge coefficient penalty per model parameter, fixed before data.
    reg=RIDGE*np.identity(3)
    mcoeff=np.linalg.solve(x.T@x+reg,x.T@y)
    if not np.isfinite(qcoeff).all() or not np.isfinite(mcoeff).all():
        raise ArithmeticError("nonfinite fit")
    return {"schema":"quartz_general_c19_two_part_signed_v1",
      "q_coefficients":qcoeff.tolist(),
      "m_coefficients":mcoeff.tolist(),
      "n_train":len(rows),"n_train_flips":len(flips),
      "train_mean_signed_value":float(np.mean([r["signed_gain"] for r in rows])),
      "train_conditional_mean_if_flip":float(np.mean(y)),
      "ridge":RIDGE,"feature_keys":["1","normalized_gap","log1p_actions"],
      "reads_latent_truth_or_domain":False}

def calibrate_signed(model:Mapping,rows:Sequence[Mapping])->dict:
    """CAL-only correction to two additive intercepts, no slope optimization."""
    _check_training(rows)
    qcoeff=np.asarray(model["q_coefficients"],dtype=float)
    mcoeff=np.asarray(model["m_coefficients"],dtype=float)
    qoffset=float(calibrate_intercept(qcoeff,rows))
    flips=[r for r in rows if r["label"]==1]
    if len(flips)<1:
        raise ValueError("no calibration flips: conditional effect offset NOT_ESTIMABLE")
    x=np.stack([features(float(r["gap"]),int(r["n_actions"])) for r in flips])
    y=np.asarray([float(r["signed_gain"]) for r in flips])
    # Frozen ridge shrinkage on one scalar intercept, no other learned coefficients.
    m_offset=float(np.sum(y-x@mcoeff)/(len(y)+CAL_RIDGE))
    return {"q_offset":qoffset,"m_offset":m_offset,
      "n_cal":len(rows),"n_cal_flips":len(flips),
      "calibration_only_intercepts":True}

def shadow_predict(model:Mapping, initial_only_rows:Sequence[Mapping],
                   *,resource_price:float,cost_by_kind:Mapping[str,float],
                   calibration:Mapping|None=None)->list[dict]:
    if not math.isfinite(resource_price) or resource_price<0:
        raise ValueError("invalid utility-per-microsecond exchange rate")
    if not initial_only_rows:return []
    qoff=float((calibration or {}).get("q_offset",0.0))
    moff=float((calibration or {}).get("m_offset",0.0))
    if not math.isfinite(qoff) or not math.isfinite(moff):
        raise ValueError("invalid calibration offsets")
    qcoeff=np.asarray(model["q_coefficients"],dtype=float)
    mcoeff=np.asarray(model["m_coefficients"],dtype=float)
    if qcoeff.shape!=(3,) or mcoeff.shape!=(3,):
        raise ValueError("model weights require three dimensions")
    answer=[]
    for r in initial_only_rows:
        if set(r)-INITIAL_ONLY:
            raise ValueError("shadow forecast must not receive labels, true utility or post-compute outcome")
        if set(r)!=INITIAL_ONLY:raise ValueError("incomplete initial-only row")
        kind=r["compute_kind"]
        if not isinstance(kind,str) or kind not in cost_by_kind:
            raise ValueError("unknown declared computation type")
        cost=float(cost_by_kind[kind])
        if not math.isfinite(cost) or cost<=0:
            raise ValueError("positive frozen predicted marginal compute time required")
        x=features(float(r["gap"]),int(r["n_actions"]))
        q=float(predict(qcoeff,[r],qoff)[0])
        m=float(np.clip(x@mcoeff+moff,-SIGNED_BOUND,SIGNED_BOUND))
        gross=q*m
        net=gross-resource_price*cost
        answer.append({"flip_probability":q,
            "conditional_signed_value":m,
            "gross_signed_value":gross,
            "estimated_incremental_cost_us":cost,
            "predicted_net_value":net,
            "shadow_choice":"COMPUTE" if net>0 else "STOP",
            "runtime_search_unchanged":True})
    return answer

def score_signed(predicted:Sequence[float], actual:Sequence[float],flipped:Sequence[int])->dict:
    if not predicted or len(predicted)!=len(actual) or len(actual)!=len(flipped):
        raise ValueError("nonempty equal sized outcome vectors")
    p=np.asarray(predicted,dtype=float)
    z=np.asarray(actual,dtype=float)
    y=np.asarray(flipped,dtype=int)
    if not np.isfinite(p).all() or not np.isfinite(z).all() or not np.all(np.isin(y,[0,1])):
        raise ValueError("invalid numeric predictions or binary changed outcomes")
    if np.any(abs(z)>1+TOL) or np.any((y==0)&(abs(z)>TOL)):
        raise ValueError("signed utility must be bounded and zero on unchanged action")
    return {"n":len(z),"flips":int(np.sum(y)),
      "mse":float(np.mean((p-z)**2)),
      "mae":float(np.mean(abs(p-z))),
      "actual_mean_signed_gain":float(np.mean(z)),
      "predicted_mean_signed_gain":float(np.mean(p))}
