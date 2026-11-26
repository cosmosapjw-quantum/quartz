from random import Random
from fractions import Fraction as F
r=Random(261127)
for _ in range(8192):
    B=r.randrange(2,33);p=r.randrange(1,B);free=[B-p,p];spent=[0,0];pending=[[0,0] for i in range(8)]
    for _ in range(64):
        w=r.randrange(8);j=r.randrange(2);k=r.randrange(1,3);op=r.randrange(3)
        if op==0 and free[j]>=k:free[j]-=k;pending[w][j]+=k
        if op==1 and pending[w][j]>=k:pending[w][j]-=k;spent[j]+=k
        if op==2 and pending[w][j]>=k:pending[w][j]-=k;free[j]+=k
        assert all(free[j]+spent[j]+sum(x[j] for x in pending)==[B-p,p][j] for j in range(2))
    for j in range(2):free[j]+=sum(x[j] for x in pending)
    assert sum(free)+sum(spent)==B
    b=sorted([F(r.randrange(-4,9),4) for _ in range(B)],reverse=True)
    v=sorted([F(r.randrange(-4,9),4) for _ in range(B)],reverse=True)
    G=[sum(b[:B-k])+sum(v[:k]) for k in range(B+1)]
    d=[y-x for x,y in zip(G,G[1:])]
    assert d==sorted(d,reverse=True)
    k=sum(x>0 for x in d);assert G[k]==max(G)
print('8192 exact split-optimum cases; 524288 protected-pool transition checks and release identities')
