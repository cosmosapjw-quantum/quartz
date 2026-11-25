"""Coherent empirical joint observation model with one latent marginal."""
from itertools import product
from math import exp,fsum,isfinite,log
STATES=tuple((a,b) for a,b in product(range(-2,3),repeat=2) if max(0,a,b)-min(0,a,b)<=2)
VALUES=tuple((0.,a/2,b/2) for a,b in STATES)
INDEX={s:i for i,s in enumerate(STATES)}
CHANNELS=tuple(f'{a}:{n}' for n in (32,64) for a in ('K16N','K16BAL','K17EXP'))
HVALUES=(None,1.,.5,.25)
PRICES=(.01,0.,.002,.02)
def valid(x):
    if len(x)!=6 or not all(isfinite(a) and abs(a)<=1.000001 for a in x):raise ValueError('bad features')
def estimate(train,x,h):
    valid(x)
    if not train or h not in HVALUES:raise ValueError('bad training/bandwidth')
    w=[];s=[]
    for row in train:
        valid(row['x']);s.append(INDEX[tuple(row['y'])])
        w.append(1. if h is None else exp(-fsum((a-b)**2 for a,b in zip(x,row['x']))/(2*h*h)))
        if set(row['obs'])!=set(CHANNELS):raise ValueError('incomplete paired channels')
    sw=fsum(w);assert sw>0
    p=[(1/19+fsum(a for a,i in zip(w,s) if i==k))/(sw+1) for k in range(19)]
    mu=[fsum(a*v[i] for a,v in zip(p,VALUES)) for i in range(3)]
    H=fsum(a*max(v) for a,v in zip(p,VALUES));V=max(mu);stop=max(range(3),key=mu.__getitem__)
    out={}
    for c in CHANNELS:
        J=[[1/57 for _ in range(3)] for _ in STATES];cost=[]
        for a,i,r in zip(w,s,train):
            o=r['obs'][c];z=o['z'];t=o['cost']
            if type(z)!=int or not 0<=z<3 or not isfinite(t) or t<0:raise ValueError('bad channel observation')
            J[i][z]+=a;cost.append(a*t)
        J=[[a/(sw+1) for a in r] for r in J]
        assert all(abs(fsum(r)-v)<1e-12 for r,v in zip(J,p))
        pi=[fsum(r[z] for r in J) for z in range(3)]
        t=[[fsum(r[z]*v[i] for r,v in zip(J,VALUES)) for i in range(3)] for z in range(3)]
        choices=[max(range(3),key=r.__getitem__) for r in t]
        vc=fsum(max(r) for r in t);delta=vc-V
        assert -1e-12<=delta<=H-V+1e-12
        assert all(abs(fsum(r[i] for r in t)-mu[i])<1e-12 for i in range(3))
        out[c]=dict(J=J,pi=pi,choices=choices,delta=max(0.,delta),cost=fsum(cost)/sw,risk=max(0.,H-vc))
    return dict(p=p,mu=mu,H=H,V=V,risk=H-V,stop=stop,channels=out,ess=sw*sw/fsum(a*a for a in w))
def choose(q,price):
    if not isfinite(price) or price<0:raise ValueError('invalid price')
    best='STOP';value=0.
    for c in CHANNELS:
        a=q['channels'][c];v=a['delta']-price*a['cost']
        if v>value+1e-15:best=c;value=v
    return best,value
def joint_log(q,y,c,z):return -log(q['channels'][c]['J'][INDEX[tuple(y)]][z])
