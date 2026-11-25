from fractions import Fraction as F
from itertools import product
from math import log
S=sorted({(b-a,c-a) for a,b,c in product(range(3),repeat=3)})
assert len(S)==19
assert S==[(k,l) for k,l in product(range(-2,3),repeat=2) if max(0,k,l)-min(0,k,l)<=2]
D=[(F(0),F(k,2),F(l,2)) for k,l in S]
def mean(p):return tuple(sum(t*d[i] for t,d in zip(p,D)) for i in range(3))
def dist(i,j,x):
    a=[F(0)]*19;a[i]+=x;a[j]+=1-x;return a
count=0
for i,j in product(range(19),repeat=2):
    p=dist(i,j,F(2,5))
    for k,l in product(range(19),repeat=2):
        q=dist(k,l,F(3,5));mp,mq=mean(p),mean(q);a=max(range(3),key=mp.__getitem__);b=max(range(3),key=mq.__getitem__)
        assert 0<=mp[a]-mp[b]<=min(F(1),sum(abs(x-y) for x,y in zip(p,q)))
        count+=1
# proper score identity using rational distributions on 3 outcomes
p=[F(1,2),F(1,3),F(1,6)];q=[F(1,3)]*3
def br(x,y):return sum(t*sum((z-int(i==j))**2 for j,z in enumerate(y)) for i,t in enumerate(x))
assert br(p,q)-br(p,p)==sum((x-y)**2 for x,y in zip(p,q))
def kl(p,q):return p*log(p/q)+(1-p)*log((1-p)/(1-q))
assert kl(.51,.49)<kl(.51,.90)
print('exact TV decision inequalities',count,'support',len(S),'Brier identity and binary counterexample PASS')
