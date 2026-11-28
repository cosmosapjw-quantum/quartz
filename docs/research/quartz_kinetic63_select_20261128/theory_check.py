from fractions import Fraction as F
from itertools import product,combinations
n=0
for a,b,f,t,k in product(range(3),range(3),range(2),range(2),range(5)):
    a,b=F(a,2),F(b,2);vf=(a,b)[f];vt=(a,b)[t];o=max(a,b)
    assert vt-a==(o-a)-(o-vt) and vf-a==(o-a)-(o-vf)
    assert vt-a-F(k,500)-(vf-a)==vt-vf-F(k,500)
    assert max(F(),vf-a,vt-a-F(k,500))>=max(F(),vf-a)
    n+=1
for N in range(1,8):
    for m in range(1,N+1):
        ss=list(combinations(range(N),m));assert sum(F(N,m)*sum(s) for s in ss)/len(ss)==sum(range(N));n+=len(ss)
print('exact cases',n,'Wolfram 3 True')
