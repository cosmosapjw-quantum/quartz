from fractions import Fraction as F
from itertools import product
import math,json,random
from publish import D
n=0
u=(F(0),F(1,2),F(1))
for a,b,c,d in product(u,repeat=4):
    assert (b-a)+(c-b)==c-a
    assert (c-a)-(d-a)==c-d
    assert -1<=c-d<=1;n+=3
for y,m,t in product((-1,F(-1,2),0,F(1,2),1),repeat=3):
    for k in range(1,33):
        a=min(F(1,4),F(k,k+4));z=(1-a)*y+a*m
        assert -1<=z<=1
        assert (z-t)**2-(y-t)**2==2*a*(y-t)*(m-y)+a*a*(m-y)**2;n+=2
for d in map(F,range(1001)):
    c=2*(1+F(1,2)*d/(1+d));assert 2<=c<3;n+=1
rng=random.Random(2026112510)
def key(b,p):return sum(x<<(2*i) for i,x in enumerate(b))|((p==1)<<450)
seen={}
for j in range(2048):
    b=[rng.randrange(3) for _ in range(225)];p=rng.choice((-1,1));k=key(b,p)
    assert k>>451==0
    assert [(k>>(2*i))&3 for i in range(225)]==b and (1 if (k>>450)&1 else -1)==p
    assert k not in seen or seen[k]==(b,p);seen[k]=(b,p);n+=3
for L in range(1,226):
    p=list(range(L,0,-1));m=sum(p);k=next(i for i in range(1,L+1) if 4*sum(p[:i])>=3*m)
    for old,base in [(0,1),(L//2,2),(L,1)]:
        w=min(L,max(k,base,old));assert w>=old and w>=k and 4*sum(p[:w])>=3*m;n+=3
print('Exact finite assertions',n)
print('Wolfram symbolic telescoping, failure decomposition and blend expansion equality returned {0,0,0,16}; finite key encode/decode test, not collision proof beyond algebra.')
