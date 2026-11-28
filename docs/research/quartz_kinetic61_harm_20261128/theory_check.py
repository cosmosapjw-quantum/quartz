from fractions import Fraction as F
from itertools import product,combinations
n=0
for a,b,k in product(range(3),range(3),range(21)):
    g=F(b-a,2);gp=max(g,0);gm=max(-g,0)
    assert g==gp-gm and gp+gm==abs(g)
    assert gp-F(1,2)*gm-F(k,500)==g+F(1,2)*gm-F(k,500)
    n+=1
for N in range(1,8):
    z=list(range(N))
    for m in range(1,N+1):
        ss=list(combinations(z,m));assert sum(F(N,m)*sum(s) for s in ss)/len(ss)==sum(z)
        n+=len(ss)
assert F(1,8)-F(1,500)>0 and -F(1,500)<0
print('exact cases',n,'Wolfram 3 True')
