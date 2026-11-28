from fractions import Fraction as F
from itertools import product
n=0
for a,b,f,t,k in product(range(3),range(3),range(2),range(2),range(5)):
    a,b=F(a,2),F(b,2);vf=(a,b)[f];vt=(a,b)[t];o=max(a,b)
    assert vt-vf==(o-vf)-(o-vt)
    assert vt-vf-F(k,500)==(o-vf)-(o-vt)-F(k,500)
    n+=1
for v,w,k in product(range(5),range(-8,9),(1,2)):
    w=F(w,2)
    if abs(w)>v:continue
    for z in product((-1,F(-1,2),0,F(1,2),1),repeat=k):
        q=(w-sum(z))/(v+k)
        assert -1<=q<=1 and (w-k)/(v+k)<=q<=(w+k)/(v+k)
        n+=1
for a,b,x in product(range(-4,5),range(-4,5),range(11)):
    x=F(x,10);assert x*a+(1-x)*b>=min(a,b);n+=1
print('exact cases',n,'Wolfram 3 True')
