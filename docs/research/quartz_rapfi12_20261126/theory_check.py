from fractions import Fraction as F
from itertools import product
n=0
for l in range(1,9):
    for h in range(l+1,12):
        for t in range(l,h+1):
            q=F(t-l,h-l);assert 0<=q<=1 and (1-q)*l+q*h==t;n+=1
for p in [F(i,100) for i in range(101)]:
    for s in range(32,257,16):
        assert (1-p)*F(s,2)+p*2*s==s*(F(1,2)+F(3,2)*p);n+=1
        assert ((1-p)*F(s,2)+p*2*s<=s)==(p<=F(1,3));n+=1
for c0,c1,g0,g1 in product(range(1,5),range(1,5),range(3),range(3)):
    for q in [F(i,4) for i in range(5)]:
        c=(1-q)*c0+q*c1;g=(1-q)*F(g0,2)+q*F(g1,2)
        for lam in [F(i,4) for i in range(5)]:
            assert g<=lam*c+max(F(g0,2)-lam*c0,F(g1,2)-lam*c1);n+=1
for a,b,m,k in product(range(1,6),repeat=4):
    assert F(a+b,m+k)==(m*F(a,m)+k*F(b,k))/(m+k);n+=1
print('Exact finite assertions',n)
