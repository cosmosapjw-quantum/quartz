"""Finite joint utility-gap estimator; no game labels at inference."""
from itertools import product
from math import exp, fsum, log, isfinite
STATES=tuple(sorted({(b-a,c-a) for a,b,c in product(range(3),repeat=3)}))
INDEX={s:i for i,s in enumerate(STATES)}
HVALUES=(None,1.,.5,.25)
assert len(STATES)==19
def feature(x):
    if len(x)!=10 or not all(isfinite(v) and abs(v)<=1.000001 for v in x):
        raise ValueError('invalid feature')
def predict(train,x,h):
    feature(x)
    if h not in HVALUES or not train:raise ValueError('invalid bandwidth or empty training set')
    c=[1/19]*19;w=[]
    for r in train:
        z=r['x'];feature(z);y=tuple(r['y'])
        if y not in INDEX:raise ValueError('infeasible gap state')
        a=1. if h is None else exp(-fsum((u-v)**2 for u,v in zip(x,z,strict=True))/(2*h*h))
        c[INDEX[y]]+=a;w.append(a)
    sw=fsum(w);s2=fsum(a*a for a in w);q=[a/(sw+1) for a in c]
    assert min(q)>0 and abs(fsum(q)-1)<1e-12
    return q,sw*sw/s2 if s2 else 0.
def means(q):
    if len(q)!=19 or min(q)<0 or not all(isfinite(v) for v in q) or abs(fsum(q)-1)>1e-12:
        raise ValueError('invalid probability vector')
    return [0.,fsum(v*s[0]/2 for v,s in zip(q,STATES)),fsum(v*s[1]/2 for v,s in zip(q,STATES))]
def decide(q):
    m=means(q)
    return max(range(3),key=m.__getitem__)
def metrics(q,y):
    m=means(q);i=INDEX[tuple(y)]
    if q[i]<=0:raise ValueError('zero observed mass')
    risk=fsum(v*max(0,s[0]/2,s[1]/2) for v,s in zip(q,STATES))-max(m)
    assert risk>=-1e-12
    return dict(mean=m,risk=max(0.,risk),log=-log(q[i]),brier=fsum((v-int(j==i))**2 for j,v in enumerate(q)),gap_mse=fsum((m[j+1]-y[j]/2)**2 for j in range(2))/2)
