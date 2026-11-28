from itertools import product,combinations
from fractions import Fraction as F
n=0
for a,b,t in product(range(3),repeat=3):
    a,b,t=F(a,2),F(b,2),F(t,2)
    o=max(a,b);r=o-a
    assert r==max(b-a,0) and o-t==r-(t-a)
    for v in (a,b):assert 0<=o-v<=1 and v-a==r-(o-v)
    n+=1
for a,b,z1,z2 in product(range(5),range(-4,5),range(-2,3),range(-2,3)):
    if abs(b)>a:continue
    v=(F(b)-F(z1,2)-F(z2,2))/(a+2)
    assert -1<=v<=1
    n+=1
for N in range(1,8):
    for m in range(1,N+1):
        ss=list(combinations(range(N),m));assert sum(F(N,m)*sum(s) for s in ss)/len(ss)==sum(range(N));n+=len(ss)
print('exact cases',n,'Wolfram 3 True')
