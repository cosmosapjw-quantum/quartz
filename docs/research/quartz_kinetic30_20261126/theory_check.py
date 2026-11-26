from fractions import Fraction as F
from random import Random
r=Random(261130)
for _ in range(8192):
    L=r.randrange(2,8);Z=2*L
    u=[[F(r.randrange(3),2) for j in range(L)] for i in range(7)]
    w=[F(r.randrange(1,9)) for i in u];w=[x/sum(w) for x in w]
    z=[[r.randrange(Z),r.randrange(Z)] for i in u]
    m=[sum(x*v[a] for x,v in zip(w,u)) for a in range(L)]
    p=[sum(x*F(v.count(b),2) for x,v in zip(w,z)) for b in range(Z)]
    j=[[sum(x*v[a]*F(y.count(b),2) for x,v,y in zip(w,u,z)) for a in range(L)] for b in range(Z)]
    a=F(r.randrange(17),16);q=[[(1-a)*p[b]*m[c]+a*j[b][c] for c in range(L)] for b in range(Z)]
    assert [sum(row[c] for row in q) for c in range(L)]==m
    G=sum(max(row) for row in q)-max(m);G1=sum(max(row) for row in j)-max(m)
    assert 0<=G<=a*G1
    D=sum(row[b//2] for b,row in enumerate(q))-m[0]
    D0=sum(p[b]*m[b//2] for b in range(Z))-m[0];D1=sum(row[b//2] for b,row in enumerate(j))-m[0]
    assert D==(1-a)*D0+a*D1
    guard=F(r.randrange(5),8)
    v=0
    for b,row in enumerate(q):
        v+=max(row) if p[b]>0 and a>0 and max(row)-row[b//2]>guard*p[b] else row[b//2]
    assert v>=D+m[0] and v<=G+max(m)
print('8192 exact rational marginal/convex EVSI/affine native/guard identities')
