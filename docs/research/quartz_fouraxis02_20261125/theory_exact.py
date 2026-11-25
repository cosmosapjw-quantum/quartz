from fractions import Fraction as F
from itertools import product
from random import Random
r=Random(2026112506);n=0
for _ in range(1500):
    t=[F(r.randrange(-1,2)) for i in range(4)]
    h=[F(r.randrange(-10,11),10) for i in range(4)]
    v=F(r.randrange(-10,11),10);k=r.sample(range(4),2);a=F(r.randrange(11),10)
    T=max(t);Tk=max(t[i] for i in k);Hk=max(h[i] for i in k)
    assert (1-a)*v+a*Hk-T==(1-a)*(v-T)+a*((Tk-T)+(Hk-Tk));n+=1
    assert Tk<=T and abs(Hk-Tk)<=max(abs(h[i]-t[i]) for i in k);n+=1
    known=r.sample(range(4),r.randrange(5));lo=max(t[i] if i in known else -1 for i in range(4));hi=max(t[i] if i in known else 1 for i in range(4))
    assert lo<=T<=hi and ((lo==hi)==(any(t[i]==1 for i in known) or len(known)==4));n+=1
for _ in range(1200):
    e=[F(r.randrange(-10,11),10) for i in range(7)];f=[F(r.randrange(-10,11),10) for i in range(7)]
    s=sum(x*x for x in e)/7;t=sum(x*x for x in f)/7;u=sum(x*y for x,y in zip(e,f))/7;D=s+t-2*u
    assert D>=0;n+=1
    a=F(r.randrange(5),16);m=sum(((1-a)*x+a*y)**2 for x,y in zip(e,f))/7
    assert m==s+2*a*(u-s)+a*a*D;n+=1
    if D:
        opt=min(F(1,4),max(F(0),(s-u)/D))
        assert s+2*opt*(u-s)+opt*opt*D<=m;n+=1
    else:assert e==f;n+=1
assert sum(max(x,y) for x,y in product((-1,1),repeat=2))/4==.5
print('Exact finite assertions',n+1)
