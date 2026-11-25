from itertools import product
from fractions import Fraction as F
from random import Random
r=Random(2026112507);n=0
I=[(a,b) for a in (-1,0,1) for b in (-1,0,1) if a<=b]
for intervals in product(I,repeat=3):
    lo=max(a for a,b in intervals);hi=max(b for a,b in intervals)
    R=[F(max(0,max(intervals[j][1] for j in range(3) if j!=i)-intervals[i][0]),2) for i in range(3)]
    for vals in product(*(range(a,b+1) for a,b in intervals)):
        T=max(vals);assert lo<=T<=hi;n+=1
        for i,v in enumerate(vals):
            assert F(T-v,2)<=R[i];n+=1
            if intervals[i][1]<lo:assert v<T;n+=1
            if R[i]==0:assert v==T;n+=1
    if lo==hi:assert any(a==hi for a,b in intervals);n+=1
    for refined in product(*[[j for j in I if a<=j[0]<=j[1]<=b] for a,b in intervals]):
        assert lo<=max(a for a,b in refined)<=max(b for a,b in refined)<=hi;n+=1
        for i in range(3):
            x=F(max(0,max(refined[j][1] for j in range(3) if j!=i)-refined[i][0]),2)
            assert x<=R[i];n+=1
# Random acyclic graph, independent recursive truth, random registration order.
for _ in range(500):
    edges={i:r.sample(range(i+1,12),r.randrange(1,min(4,12-i))) for i in range(8)}
    truth={i:r.randrange(-1,2) for i in range(8,12)}
    for i in reversed(range(8)):truth[i]=max(-truth[j] for j in edges[i])
    iv={i:(-1,1) for i in range(12)};reg=set()
    order=list(range(12));r.shuffle(order)
    for i in order:
        reg.add(i)
        if i>=8:iv[i]=(truth[i],truth[i])
        for j in reversed(range(8)):
            if j in reg:iv[j]=(max(-iv[k][1] for k in edges[j]),max(-iv[k][0] for k in edges[j]))
        assert all(a<=truth[k]<=b for k,(a,b) in iv.items());n+=12
    assert all(iv[k]==(v,v) for k,v in truth.items());n+=12
print('Exact interval, regret and DAG assertions',n)
