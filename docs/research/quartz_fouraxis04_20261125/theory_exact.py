from itertools import product
from fractions import Fraction as F
from random import Random
r=Random(2026112508);n=0
for q in range(1,6):
    K=4*q
    for b in product(range(K+1),repeat=4):
        if sum(b)>K:continue
        S=sum(max(0,q-i) for i in b);E=sum(max(0,i-q) for i in b);U=K-sum(b)
        assert S-E==U and E<=S;n+=1
# Exhaustive risk upper bound and zero witness for three legal actions.
I=[(a,b) for a in (-1,0,1) for b in (-1,0,1) if a<=b]
for iv in product(I,repeat=3):
    risk=[max(0,max(iv[j][1] for j in range(3) if j!=i)-iv[i][0]) for i in range(3)]
    for v in product(*(range(a,b+1) for a,b in iv)):
        for i in range(3):
            assert max(v)-v[i]<=risk[i];n+=1
            if risk[i]==0:assert v[i]==max(v);n+=1
# Finite resource exact optimization with certificate contraction.
# Independent deterministic potential tables, optimal allocator can ignore work.
for _ in range(600):
    q=3;K=6;cert=r.randrange(4)
    y=[F(r.randrange(3),2) for j in range(7)]
    x=[F(r.randrange(3),2) for j in range(cert)]+[F(1)]*(7-cert)
    full=max(x[i]+y[j] for i in range(7) for j in range(7-i))
    cut=max(x[i]+y[j] for i in range(cert+1) for j in range(7-i))
    assert full==cut;n+=1
print('Exact block accounting, risk and STOP contraction assertions',n)
