from fractions import Fraction as F
from itertools import product
from random import Random
from math import prod
rng=Random(241124); n=0
for _ in range(1000):
    w=[rng.randrange(1,20) for _ in range(5)];z=sum(w);p=[F(x,z) for x in w]
    d=[F(-1),F(-1,2),F(0),F(1,2),F(1)]
    g=sum(x*y for x,y in zip(p,d)); pp=p[3]+p[4];pm=p[0]+p[1]
    assert pp/2-pm<=g<=pp-pm/2 and abs(g)<=pp+pm
    if pp+pm:
        assert g==(pp+pm)*(pp/(pp+pm)*(p[3]/2+p[4])/pp-pm/(pp+pm)*(p[0]+p[1]/2)/pm)
    n+=2
for _ in range(200):
    options=[[(0,F(0))]+[(rng.randrange(1,5),F(rng.randrange(-3,7),5)) for _ in range(2)] for _ in range(4)]
    B=rng.randrange(1,10); v=[F(0)]*(B+1)
    for row in options:
        v=[max(v[b-c]+g for c,g in row if c<=b) for b in range(B+1)]
    assert v[B]==max(sum(g for c,g in choice) for choice in product(*options) if sum(c for c,g in choice)<=B)
    n+=1
for u in [F(0),F(1,2),F(1)]:
    for v in [F(0),F(1,2),F(1)]:
        for a in [F(0),F(1,2),F(1)]:
            assert min(u,v,a)-u<=a-u<=max(u,v,a)-u
            if u==max(u,v,a):assert a-u<=0
            n+=1
print('Exact Fraction exposure/decomposition/knapsack/headroom checks',n)
