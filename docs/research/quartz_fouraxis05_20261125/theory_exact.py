from itertools import product
from random import Random
from collections import deque
from fractions import Fraction as F
rng=Random(2026112509);n=0
I=[(l,u) for l in (-1,0,1) for u in (-1,0,1) if l<=u]
def risk(a):return min(max(0,max((v[1] for j,v in enumerate(a) if j!=i),default=-1)-l) for i,(l,u) in enumerate(a))
def width(a):return sum(u-l for l,u in a)
for a in product(I,repeat=3):
    for i,(l,u) in enumerate(a):
        for x in range(l,u+1):
            b=list(a);b[i]=(x,x);assert risk(b)<=risk(a) and width(b)<=width(a);n+=1
    assert (risk(a)==0)==any(l>=max((v[1] for j,v in enumerate(a) if j!=i),default=-1) for i,(l,u) in enumerate(a));n+=1
for _ in range(600):
    edges={i:rng.sample(range(i+1,13),rng.randrange(1,min(4,13-i))) for i in range(9)}
    parents={j:[i for i,a in edges.items() if j in a] for j in range(13)}
    iv={i:rng.choice(I) for i in range(9,13)}
    def f(i,m):return (max(-m[j][1] for j in edges[i]),max(-m[j][0] for j in edges[i]))
    for i in reversed(range(9)):iv[i]=f(i,iv)
    a=[(-iv[j][1],-iv[j][0]) for j in edges[0]]
    for s in range(9,13):
        for x in range(iv[s][0],iv[s][1]+1):
            m=dict(iv);m[s]=(x,x);todo=deque(parents[s])
            while todo:
                i=todo.popleft();z=f(i,m)
                if z!=m[i]:
                    assert m[i][0]<=z[0]<=z[1]<=m[i][1];n+=1
                    m[i]=z;todo.extend(parents[i])
            b=dict(iv);b[s]=(x,x)
            for i in reversed(range(9)):b[i]=f(i,b)
            assert m==b;n+=1
            aa=[(-m[j][1],-m[j][0]) for j in edges[0]]
            assert risk(aa)<=risk(a) and width(aa)<=width(a);n+=1
for _ in range(1200):
    y=[F(rng.randrange(3),2) for j in range(8)]
    assert sum(b-a for a,b in zip(y,y[1:]))==y[-1]-y[0];n+=1
print('Exact interval, DAG overlay, signed-gain assertions',n)
