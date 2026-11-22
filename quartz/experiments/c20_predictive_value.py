"""Exact model-conditional one-step expected value of sample information (EVSI).

Task-specific source likelihood remains in adapter functions. The metalevel
selector consumes only their generic probabilistic payoff summary and priced
CPU resource. All future outcomes are integrated out. Hidden true task theta,
realised future observations and future actions are NOT inputs.
"""
from __future__ import annotations
import itertools
import math
import time
from typing import Iterable,Sequence

NORM=1e-9
MAX_OUTCOMES=50000
MAX_N=10

def _row(v:Sequence[float],*,probability:bool)->list[float]:
    if not v:raise ValueError("empty vector")
    z=[float(a) for a in v]
    if any(not math.isfinite(a) for a in z):
        raise ValueError("nonfinite scalar")
    if probability:
        if any(a<0 or a>1 for a in z) or abs(math.fsum(z)-1)>NORM:
            raise ValueError("invalid probability simplex")
    else:
        if any(a<0 or a>1 for a in z):raise ValueError("utility outside [0,1]")
    return z

def _finish(before:float,after:float,mass:float,support:int,start:int)->dict:
    if not all(math.isfinite(v) for v in (before,after,mass)):
        raise ValueError("nonfinite exact computation")
    if abs(mass-1)>NORM:raise ValueError(f"predictive mass not 1: {mass}")
    evsi=after-before
    if evsi < -NORM or evsi > 1-before+NORM:
        raise ArithmeticError(f"Bayes information bound violated: {evsi}")
    return {"before_best":float(before),"after_expected_best":float(after),
       "evsi":float(max(evsi,0.0)),
       "total_predictive_mass":float(mass),"outcome_count":int(support),
       "evaluation_us":float(max(0,time.perf_counter_ns()-start))/1000,
       "true_hidden_state_was_used":False,
       "interpretation":"model-conditional exact predictive EVSI, not true-game MCTS utility"}

def generic_evsi(before_best:float,weighted_future:Sequence[tuple[float,Sequence[float]]])->dict:
    """Generic finite predictive outcome distribution over posterior action means."""
    begin=time.perf_counter_ns()
    if not math.isfinite(before_best) or not 0<=before_best<=1:
        raise ValueError("invalid current expected utility")
    if not weighted_future or len(weighted_future)>MAX_OUTCOMES:
        raise ValueError("invalid outcome support")
    mass=[];nextu=[]
    n_actions=None
    for p,vals in weighted_future:
        p=float(p)
        if not math.isfinite(p) or not 0<=p<=1:raise ValueError("invalid outcome weight")
        utilities=_row(vals,probability=False)
        if n_actions is None:n_actions=len(utilities)
        if len(utilities)!=n_actions:raise ValueError("inconsistent terminal action set")
        mass.append(p);nextu.append(p*max(utilities))
    return _finish(before_best,math.fsum(nextu),math.fsum(mass),
                   len(weighted_future),begin)

def _beta_binomial_probs(alpha:float,beta:float,n:int)->list[float]:
    if alpha<=0 or beta<=0:raise ValueError("improper beta posterior")
    logB=math.lgamma(alpha)+math.lgamma(beta)-math.lgamma(alpha+beta)
    ans=[]
    for k in range(n+1):
        logp=math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)
        logp+=math.lgamma(alpha+k)+math.lgamma(beta+n-k)
        logp-=math.lgamma(alpha+beta+n)
        logp-=logB
        ans.append(math.exp(logp))
    mass=math.fsum(ans)
    if abs(mass-1)>NORM:raise ArithmeticError("Beta predictive distribution unnormalized")
    return ans

def beta_evsi(posterior:Sequence[Sequence[float]],extra:int)->dict:
    """Full Beta-Binomial sufficient-count joint law, one batch per arm."""
    start=time.perf_counter_ns()
    if isinstance(extra,bool) or not isinstance(extra,int) or not 0<=extra<=MAX_N:
        raise ValueError("invalid bounded Bernoulli batch")
    if not (2<=len(posterior)<=4) or (extra+1)**len(posterior)>MAX_OUTCOMES:
        raise ValueError("unsupported source action count or exponential count budget")
    params=[]
    for pair in posterior:
        if len(pair)!=2:raise ValueError("Beta(a,b) requires two parameters")
        a,b=float(pair[0]),float(pair[1])
        if not math.isfinite(a) or not math.isfinite(b) or a<=0 or b<=0:
            raise ValueError("proper finite Beta posterior required")
        params.append((a,b))
    before=max(a/(a+b) for a,b in params)
    prob=[_beta_binomial_probs(a,b,extra) for a,b in params]
    powers=[[(a+k)/(a+b+extra) for k in range(extra+1)] for a,b in params]
    total=[];weights=[]
    for path in itertools.product(range(extra+1),repeat=len(params)):
        p=math.prod(prob[j][k] for j,k in enumerate(path))
        u=max(powers[j][k] for j,k in enumerate(path))
        weights.append(p);total.append(p*u)
    return _finish(before,math.fsum(total),math.fsum(weights),len(weights),start)

def _counts(kinds:int,total:int):
    if kinds==1:
        yield (total,)
    else:
        for v in range(total+1):
            for rem in _counts(kinds-1,total-v):
                yield (v,)+rem

def sensor_evsi(prior:Sequence[float],likelihood:Sequence[Sequence[float]],
                utilities:Sequence[Sequence[float]],extra:int)->dict:
    """Mixture of multinomials with one fixed latent theta across all new samples."""
    start=time.perf_counter_ns()
    if isinstance(extra,bool) or not isinstance(extra,int) or not 0<=extra<=MAX_N:
        raise ValueError("invalid bounded sensor batch")
    b=_row(prior,probability=True)
    if len(b)<2 or len(b)>8:raise ValueError("unsupported latent support")
    if len(likelihood)!=len(b) or len(utilities)!=len(b):
        raise ValueError("latent dimension mismatch")
    L=[_row(x,probability=True) for x in likelihood]
    U=[_row(x,probability=False) for x in utilities]
    m=len(L[0]);a_count=len(U[0])
    if not (2<=m<=4) or not (2<=a_count<=8):
        raise ValueError("unsupported bounded sensor outcome/action set")
    if any(len(x)!=m for x in L) or any(len(x)!=a_count for x in U):
        raise ValueError("mismatched sensor channel/terminal action set")
    size=math.comb(extra+m-1,m-1)
    if size>MAX_OUTCOMES:raise ValueError("too many predictive count outcomes")
    baseline=[math.fsum(b[t]*U[t][a] for t in range(len(b))) for a in range(a_count)]
    before=max(baseline)
    totals=[];prob_mass=[]
    for counts in _counts(m,extra):
        mult=math.factorial(extra)//math.prod(math.factorial(k) for k in counts)
        latent=[b[t]*math.prod(L[t][y]**k for y,k in enumerate(counts))
                  for t in range(len(b))]
        mass=mult*math.fsum(latent)
        weighted_best=mult*max(math.fsum(latent[t]*U[t][a] for t in range(len(b)))
                                   for a in range(a_count))
        prob_mass.append(mass);totals.append(weighted_best)
    return _finish(before,math.fsum(totals),math.fsum(prob_mass),size,start)
