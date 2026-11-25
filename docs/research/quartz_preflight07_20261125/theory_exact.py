from fractions import Fraction as F
from itertools import product
n=0
for m in range(1,9):
    for x in product((-1,0,1),repeat=m):
        W=F(0);M=F(0)
        for k,y in enumerate(x):
            if k:M+=(F(y)-W/k)*(F(y)-(W+y)/(k+1))
            W+=y
            assert M==sum((F(v)-W/(k+1))**2 for v in x[:k+1]);n+=1
            assert 0<=M<=k+1-W**2/(k+1);n+=1
        for k in range(m):
            assert (-1)**(m-k)*x[-1]==(-1)**(m-k-1)*(-x[-1]);n+=1
I=[(a,b) for a in (-1,0,1) for b in (-1,0,1) if a<=b]
for a in product(I,repeat=4):
    hi=sorted(((u,i) for i,(l,u) in enumerate(a)),reverse=True)
    ref=[max(0,max(u for j,(l,u) in enumerate(a) if i!=j)-l) for i,(l,u) in enumerate(a)]
    calc=[max(0,(hi[1][0] if hi[0][1]==i else hi[0][0])-l) for i,(l,u) in enumerate(a)]
    assert ref==calc;n+=1
for t in range(226):
    for i in range(t):
        assert sum(i<t-j for j in range(8))==min(8,t-i);n+=1
print('Exact finite assertions',n)
