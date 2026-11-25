from itertools import product,permutations
from fractions import Fraction as F
from random import Random
from math import ceil
r=Random(2026112504);n=0
# Exact exchangeable rank and inclusive-tie coverage for arbitrary fixed multisets.
for m in range(1,10):
    for _ in range(100):
        s=[r.randrange(6) for _ in range(m+1)]
        a=F(r.randrange(1,10),10);k=ceil((m+1)*(1-a))
        hit=sum(k>m or s[i]<=sorted(s[:i]+s[i+1:])[k-1] for i in range(m+1))
        assert F(hit,m+1)>=1-a;n+=1
# Exact positive rational weights act as pi*exp(beta*g); DP equals enumeration.
for _ in range(500):
    N=4;K=r.randrange(0,13)
    q=[[0,r.randrange(1,6),r.randrange(1,8)] for i in range(N)]
    w=[[F(r.randrange(1,10),r.randrange(1,10)) for j in range(3)] for i in range(N)]
    v=[F(1)]*(K+1)
    for i in range(N):
        v=[sum(w[i][j]*v[b-q[i][j]] for j in range(3) if q[i][j]<=b) for b in range(K+1)]
    total=F(0)
    for a in product(range(3),repeat=N):
        if sum(q[i][j] for i,j in enumerate(a))<=K:
            t=F(1)
            for i,j in enumerate(a):t*=w[i][j]
            total+=t
    assert total==v[K];n+=1
# Simultaneous envelope and ceil rounding imply all selected feasible plans safe on the event.
for _ in range(500):
    d=[[F(r.randrange(1,10),3) for j in range(3)] for i in range(4)]
    mult=F(r.randrange(3,10),3);h=F(1,2);K=r.randrange(0,30)
    c=[[mult*x*F(r.randrange(0,11),10) for x in row] for row in d]
    q=[[0]+[ceil(mult*x/h) for x in row] for row in d]
    for a in product(range(4),repeat=4):
        if sum(q[i][j] for i,j in enumerate(a))<=K:
            assert sum(c[i][j-1] for i,j in enumerate(a) if j)<=K*h
            n+=1
print('Exact rational/rank/partition/feasibility assertions',n)
