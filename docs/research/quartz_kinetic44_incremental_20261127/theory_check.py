from itertools import product
from collections import Counter
n=3;k=3
W=[tuple((r+i*dr)*n+c+i*dc for i in range(k)) for r in range(n) for c in range(n) for dr,dc in ((1,0),(0,1),(1,1),(1,-1)) if 0<=r+(k-1)*dr<n and 0<=c+(k-1)*dc<n]
inc=[[w for w in W if m in w] for m in range(n*n)]
def contribution(b,w):
    e=tuple(m for m in w if b[m]==0)
    if len(e) not in (1,2):return None
    v={b[m] for m in w if b[m]}
    return (next(iter(v)),e) if len(v)==1 else None
def counts(b):
    return Counter(c for w in W if (c:=contribution(b,w)) is not None)
ct=Counter()
for b0 in product(range(3),repeat=9):
    b=list(b0);a=counts(b);ct['boards']+=1
    for m in range(9):
        if b[m]:continue
        for p in (1,2):
            c=a.copy()
            for w in inc[m]:
                x=contribution(b,w)
                if x is not None:c[x]-=1
            b[m]=p
            for w in inc[m]:
                x=contribution(b,w)
                if x is not None:c[x]+=1
            assert +c==counts(b)
            opp=3-p
            assert sum(v for (i,e),v in c.items() if i==opp)==sum(v for (i,e),v in a.items() if i==opp and m not in e)
            for w in inc[m]:
                x=contribution(b,w)
                if x is not None:c[x]-=1
            b[m]=0
            for w in inc[m]:
                x=contribution(b,w)
                if x is not None:c[x]+=1
            assert +c==a
            ct['make_unmake']+=1
print(dict(ct))
