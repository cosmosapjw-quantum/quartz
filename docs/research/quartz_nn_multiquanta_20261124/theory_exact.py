from fractions import Fraction as F
from itertools import product
from random import Random
from math import ceil,floor
r=Random(31124); ncheck=0
for _ in range(700):
    n=4; J=3; delta=F(1,4); h=F(1,2); eps=F(1,10); B=F(r.randint(8,28),2); K=floor(B/h)
    c=[ [F(0)]+[F(r.randint(2,20),4) for j in range(J-1)] for i in range(n)]
    d=[[F(0)]+[x+F(r.randint(-1,1),4) for x in row[1:]] for row in c]
    g=[[F(0)]+[F(r.randint(-10,10),10) for j in range(J-1)] for i in range(n)]
    est=[[F(0)]+[x+F(r.randint(-1,1),10) for x in row[1:]] for row in g]
    q=[[0]+[ceil((x+delta)/h) for x in row[1:]] for row in d]
    val=lambda a,x:sum(x[i][j] for i,j in enumerate(a))
    choices=list(product(range(J),repeat=n))
    feasible=[a for a in choices if val(a,q)<=K]
    assert all(val(a,c)<=B for a in feasible)
    v=[F(0)]*(K+1)
    for i in range(n):v=[max(v[b-q[i][j]]+est[i][j] for j in range(J) if q[i][j]<=b) for b in range(K+1)]
    best=max(feasible,key=lambda a:val(a,est));assert val(best,est)==v[K]
    lam=F(r.randint(0,10),10)
    assert v[K]<=lam*K+sum(max(est[i][j]-lam*q[i][j] for j in range(J)) for i in range(n))
    slack=n*(2*delta+h)+h
    if B>=slack:
        tight=[a for a in choices if val(a,c)<=B-slack]
        assert all(a in feasible for a in tight)
        assert max(val(a,g) for a in tight)-val(best,g)<=2*n*eps
        ncheck+=2
    ncheck+=3
print('Exact enumeration assertions:',ncheck)
