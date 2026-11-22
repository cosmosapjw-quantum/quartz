"""C18 game-neutral shadow forecast for decision revision, not a live stop rule.

No domain IDs, terminal winner, future visit information or utility labels are
accepted as predictors. Actual adaptive MCTS results require separately tested
source adapters and do NOT become calibrated posterior utility models here.
"""
from __future__ import annotations
import math
from typing import Mapping, Sequence
import numpy as np

EPS=1e-6
L2=0.5
CAL_L2=0.25

def features(gap:float,n_actions:int)->np.ndarray:
    if (not isinstance(n_actions,int) or n_actions<2 or
        not isinstance(gap,(int,float)) or not math.isfinite(float(gap)) or not 0<=gap<=1):
        raise ValueError("initial-only gap in[0,1] and action count>=2 required")
    return np.array([1.0,float(gap),math.log1p(n_actions)],dtype=float)

def _design(rows:Sequence[Mapping])->tuple[np.ndarray,np.ndarray]:
    if not rows:raise ValueError("empty source-group rows")
    x=np.stack([features(float(r["gap"]),int(r["n_actions"])) for r in rows])
    y=np.asarray([r["label"] for r in rows],dtype=float)
    if not np.all((y==0)|(y==1)):raise ValueError("binary revision label required")
    return x,y

def _sigmoid(z):
    z=np.asarray(z,dtype=float)
    return np.where(z>=0,1/(1+np.exp(-np.minimum(z,600))),
                    np.exp(np.maximum(z,-600))/(1+np.exp(np.maximum(z,-600))))

def fit_logistic(rows:Sequence[Mapping])->np.ndarray:
    """Full-batch deterministic L2 Newton, TRAIN data only, no domain coefficient."""
    x,y=_design(rows)
    theta=np.zeros(3,float)
    penalty=np.diag([0.0,L2,L2])
    for _ in range(60):
        z=x@theta
        p=_sigmoid(z)
        grad=x.T@(p-y)+penalty@theta
        hess=(x.T*(p*(1-p)))@x+penalty
        delta=np.linalg.solve(hess,grad)
        # Subtract a Newton step; fixed objective, no TEST-dependent tuning.
        theta-=delta
        if np.linalg.norm(delta)<=1e-9 or np.linalg.norm(grad)<=1e-7:
            break
    if not np.isfinite(theta).all():
        raise ArithmeticError("fitted nonfinite logistic coefficients")
    return theta

def calibrate_intercept(theta:Sequence[float],cal_rows:Sequence[Mapping])->float:
    """Pool CAL parent groups for one strictly shadow additive logit shift."""
    x,y=_design(cal_rows)
    theta=np.asarray(theta,dtype=float)
    if theta.shape!=(3,) or not np.isfinite(theta).all():
        raise ValueError("invalid fixed fitted coefficients")
    shift=0.0
    for _ in range(60):
        p=_sigmoid(x@theta+shift)
        grad=float(np.sum(p-y)+CAL_L2*shift)
        hess=float(np.sum(p*(1-p))+CAL_L2)
        step=grad/hess
        shift-=step
        if abs(step)<1e-10:break
    if not math.isfinite(shift):raise ArithmeticError("invalid calibration adjustment")
    return shift

def predict(theta:Sequence[float],rows:Sequence[Mapping],offset:float=0.0)->np.ndarray:
    theta=np.asarray(theta,dtype=float)
    if theta.shape!=(3,) or not np.isfinite(theta).all() or not math.isfinite(offset):
        raise ValueError("invalid frozen shadow model")
    if not rows:return np.asarray([],dtype=float)
    x=np.stack([features(float(r["gap"]),int(r["n_actions"])) for r in rows])
    return np.clip(_sigmoid(x@theta+offset),EPS,1-EPS)

def score_predictions(predictions:Sequence[float],labels:Sequence[int])->dict:
    p=np.asarray(predictions,dtype=float)
    y=np.asarray(labels,dtype=float)
    if p.ndim!=1 or p.shape!=y.shape or p.size==0:
        raise ValueError("matching nonempty vectors required")
    if not np.isfinite(p).all() or not np.isfinite(y).all() or (p<0).any() or (p>1).any():
        raise ValueError("invalid prediction/label")
    if not np.all((y==0)|(y==1)):
        raise ValueError("binary revision labels only")
    clipped=np.clip(p,EPS,1-EPS)
    brier=float(np.mean((p-y)**2))
    log_loss=float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped)))
    ece=0.0
    for i in range(10):
        m=(p>=i/10)&((p<(i+1)/10) if i<9 else (p<=1))
        if m.any():
            ece+=float(np.mean(m)*abs(np.mean(p[m])-np.mean(y[m])))
    return {"n":len(p),"brier":brier,"log_loss":log_loss,
            "ece10":ece,"event_rate":float(np.mean(y)),
            "mean_prediction":float(np.mean(p))}

def initial_rust_snapshot(output:Mapping)->dict:
    """Strictly source-initial-only feature and visit-modal action readout."""
    original=output["first8_edge_stats"]
    first=int(output["first8_root_visits"])
    legal=output["legal_actions"]
    if first!=8 or len(original)!=len(legal) or len(legal)<2:
        raise ValueError("invalid actual native root witness")
    rows=[(int(r[0]),int(r[1]),float(r[2])) for r in original]
    if set(a for a,_,_ in rows)!=set(legal) or sum(n for _,n,_ in rows)!=first:
        raise ValueError("invalid first8 edge conservation")
    order=sorted(rows,key=lambda r:(-r[1],-r[2],r[0]))
    return {"gap":float(order[0][1]-order[1][1])/first,
            "n_actions":len(legal),"initial_action":order[0][0],
            "initial_visit_count":first,
            "initial_margin_raw":order[0][1]-order[1][1]}
