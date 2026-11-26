from fractions import Fraction as F
from random import Random
r=Random(261133)
for _ in range(8192):
    u=[r.randrange(-1,2) for i in range(r.randrange(2,50))]
    l=[r.randrange(-1,x+1) for x in u];h=[r.randrange(x,2) for x in u]
    for i,x in enumerate(l):
        if x>=max(h[:i]+h[i+1:]):assert u[i]==max(u)
    k=[r.randrange(65) for i in u];cert=[r.randrange(2) for i in u]
    if sum(k):
        saved=F(sum(a*b for a,b in zip(k,cert)),sum(k));assert 0<=saved<=1
        assert 1-F(sum(a*(1-b) for a,b in zip(k,cert)),sum(k))==saved
    a,b=F(r.randrange(3),2),F(r.randrange(3),2);assert 0<=(a+b)/2<=1
print('8192 exact interval witness, weighted saving and paired score checks')
