from random import Random
from fractions import Fraction as F
r=Random(261129)
for _ in range(8192):
    L=r.randrange(2,8);u=[[F(r.randrange(3),2) for _ in range(L)] for i in range(8)]
    p=[F(r.randrange(1,8)) for i in u];p=[w/sum(p) for w in p]
    z=[[r.randrange(L),r.randrange(L)] for i in u]
    m=[sum(w*v[a] for w,v in zip(p,u)) for a in range(L)]
    J=[[sum(w*v[a]*F(zz.count(b),2) for w,v,zz in zip(p,u,z)) for a in range(L)] for b in range(L)]
    assert [sum(row[a] for row in J) for a in range(L)]==m
    V=sum(max(row) for row in J);G=V-max(m);D=sum(row[b] for b,row in enumerate(J))-m[0]
    assert 0<=G<=sum(w*max(v) for w,v in zip(p,u))-max(m)
    assert D==G+(max(m)-m[0])-(V-sum(row[b] for b,row in enumerate(J)))
    b=[r.randrange(8) for i in range(r.randrange(1,9))]
    unique=set(b);assert len(b)==len(unique)+len(b)-len(unique)
print('8192 exact rational joint marginals, Bayes gap/EVPI and batch cardinality checks')
