from itertools import product
from fractions import Fraction as F
vals=range(-2,3);n=0
for v in product(vals,repeat=3):
    for est in product(vals,repeat=3):
        a=max(range(3),key=est.__getitem__);b=max(range(3),key=v.__getitem__);e=max(abs(x-y) for x,y in zip(v,est))
        assert 0<=v[b]-v[a]<=2*e
        n+=1
ints=[(a,b) for a in range(3) for b in range(a,3)]
m=0
for I in product(ints,repeat=3):
    for u in product(*(range(l,r+1) for l,r in I)):
        for d in range(3):
            assert I[d][0]-I[0][1]<=u[d]-u[0]<=I[d][1]-I[0][0]
        assert max(u)-u[0]<=max(0,max(r-I[0][0] for l,r in I[1:]))
        if max(r for l,r in I)<=I[0][0]:assert max(u)==u[0]
        m+=1
for counts in product(range(3),repeat=5):
    if not sum(counts):continue
    p=[F(c,sum(counts)) for c in counts]
    assert sum(F(i,2)*x for i,x in zip(vals,p))==p[3]/2+p[4]-p[1]/2-p[0]
print('selection',n,'interval utility assignments',m,'signed histogram identity PASS')
