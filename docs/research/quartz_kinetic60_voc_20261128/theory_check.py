from itertools import product
from fractions import Fraction as F
n=0
for a,b,k,l in product(range(3),range(3),range(5),range(5)):
    g=F(b-a,2);c=l-k;lam=F(1,500)
    assert g-lam*c==(F(b,2)-lam*l)-(F(a,2)-lam*k)
    assert g==max(g,0)-max(-g,0)
    n+=1
for h in (4,8,12,16):
    for d in range(21):
        for agree in (0,1):
            i=(h//4-1)*4+2*int(2*d>=20-h)+agree
            assert 0<=i<16
            n+=1
assert F(1,2)-F(1,500)>0 and (F(1,2)-F(1,2))/2-F(1,500)<0
print('exact checks',n,'Wolfram 3 True; cell-average counterexample')
