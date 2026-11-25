from fractions import Fraction as F
from itertools import product
from math import exp,log,sqrt,isclose
import random
def gib(x,tau):
    z=max(x);w=[exp((a-z)/tau) for a in x];s=sum(w);return z+tau*log(s/len(x)),[a/s for a in w]
def kl(a,b):return sum(x*log(x/y) for x,y in zip(a,b) if x)
r=random.Random(2026112621);n=0;err=0.
for _ in range(4096):
    m=[r.uniform(-.4,.4) for _ in range(3)];v=[r.uniform(-.5,.5) for _ in range(3)]
    t=r.choice([.05,.1,.25,.5]);f,p=gib(m,t)
    vals=[];ks=[];hard=[];xs=[]
    for sign in (-1,1):
        y=[a+sign*b for a,b in zip(m,v)];g,q=gib(y,t);b=sum(a*sign*b for a,b in zip(p,v));k=t*kl(p,q)
        err=max(err,abs(g-f-b-k));assert abs(g-f-b-k)<1e-12
        vals.append(g);ks.append(k);hard.append(max(y));xs.append(y)
    G=sum(vals)/2-f;D=sum(hard)/2-max(m);a=max(range(3),key=m.__getitem__)
    B=sum((sqrt((m[a]-m[b])**2+(v[b]-v[a])**2)-(m[a]-m[b]))/2 for b in range(3) if b!=a)
    assert -1e-12<=D<=B+1e-12 and abs(D-G)<=t*log(3)+1e-12
    assert abs(G-sum(ks)/2)<1e-12 and G<=(max(v)-min(v))**2/(8*t)+1e-12
    n+=1
# exact finite hard contrast inequality without floats via equivalent squared inequality
m=0
for a,b,c in product(range(-2,3),repeat=3):
    if a+b+c:continue
    for d in range(5):
        E=sum((max(0,x-d) for x in (a,b,c)),F(0))/3
        S=sum((x*x for x in (a,b,c)),F(0))/3
        assert (2*E+d)**2<=d*d+S
        m+=1
print('finite random two-outcome coherent checks',n,'exact contrast cases',m,'max Bregman residual',err)
