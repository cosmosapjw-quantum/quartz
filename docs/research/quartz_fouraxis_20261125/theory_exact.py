from fractions import Fraction as F
from random import Random
from itertools import product
r=Random(2026112505);n=0
def subs(x):return [y for y in range(16) if y&x==y]
for _ in range(500):
    f=[F(r.randrange(-10,11),10) for j in range(16)]
    m=[sum((-1)**((t^a).bit_count())*f[a] for a in subs(t)) for t in range(16)]
    for a in range(16):
        assert sum(m[t] for t in subs(a))==f[a];n+=1
    phi=[sum(m[t]/t.bit_count() for t in range(1,16) if t&(1<<i)) for i in range(4)]
    assert sum(phi)==f[15]-f[0];n+=1
for _ in range(1000):
    x=F(r.randrange(-10,11),10);y=x;raw=x;prod0=F(1)
    for l in range(r.randrange(1,12)):
        a=F(r.randrange(0,5),16);m=F(r.randrange(-10,11),10)
        z=(1-a)*y+a*m
        assert -1<=z<=1 and abs(z-y)<=2*a
        assert (1-a)*(-y)+a*(-m)==-z
        y=-z;raw=-raw;prod0*=1-a
        assert abs(y-raw)<=2*(1-prod0);n+=4
print('Exact Fraction factorial and backup assertions',n)
