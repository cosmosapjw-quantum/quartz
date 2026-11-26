from fractions import Fraction as F
from itertools import product
from random import Random
def sample_stable(ids,seed):
    n=len(ids);k=min(n,max(2,(n+3)//4))
    return sorted(Random(seed).sample(sorted(ids),k))
def estimate_absolute(N,strata):
    assert N>0 and sum(n for n,x in strata)==N
    total=F(0);var=F(0);weights=[]
    for n,x in strata:
        k=len(x)
        if n==0:assert k==0;continue
        assert 1<=k<=n and (k==n or k>=2)
        a=sum(x,F(0))/k;total+=n*a
        if k<n:
            v=sum((y-a)**2 for y in x)/(k-1)
            var+=F(n,N)**2*(1-F(k,n))*v/k
        weights.extend([F(n,k)]*k)
    assert sum(weights)==N
    return dict(mean=total/N,variance=var,ess=F(N*N)/sum(w*w for w in weights))
def delta(a,b,u):
    if a==b:return F(0)
    assert u is not None
    return u[str(a)]-u[str(b)]
def fit_table(rows):
    n=len(rows);assert n
    agg=[{h:[0,F(0)] for h in (80,128)} for _ in range(8)]
    for r in rows:
        for h in (80,128):
            agg[r['bin']][h][0]+=r['cost'][h];agg[r['bin']][h][1]+=r['gain'][h]
    feasible=[]
    for tab in product((80,128),repeat=8):
        cost=sum(agg[b][h][0] for b,h in enumerate(tab))
        g=sum(agg[b][h][1] for b,h in enumerate(tab))
        if g>=-F(n,40):feasible.append((cost,-g,tab))
    cost,ng,tab=min(feasible)
    return dict(table=list(tab),cost=cost,gain=-ng/n,n=n,feasible=len(feasible),bin_n=[sum(r['bin']==b for r in rows) for b in range(8)])
