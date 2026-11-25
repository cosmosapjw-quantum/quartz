from fractions import Fraction as F
from itertools import product
import math,random
n=0;up=0
# Exact rational depletion/transport, bounded response, counterexample search.
for N in product(range(4),repeat=2):
 for d in product(range(3),repeat=2):
  M=tuple(a+b for a,b in zip(N,d))
  for x,y in product(range(1,8),repeat=2):
   q=(F(x,8),1-F(x,8));p=(F(y,8),1-F(y,8))
   A=sum(v/F(k+4) for v,k in zip(q,N));B=sum(v/F(k+4) for v,k in zip(p,M))
   D=sum(v*(1/F(k+4)-1/F(j+4)) for v,k,j in zip(q,N,M))
   T=sum((v-w)/F(k+4) for v,w,k in zip(p,q,M))
   assert B-A==-D+T and D>=0 and 0<A<=F(1,4) and 0<B<=F(1,4);n+=2
   up+=B>A and sum(d)>0
   for c in (0,1,4):
    r=4*(A-B)/(1+F(c,16));assert -1<=r<=1;n+=1
assert up>0
rng=random.Random(2026112613)
def gibbs(pi,u,b=4):
 z=[p*math.exp(b*x) for p,x in zip(pi,u)];s=sum(z);return [x/s for x in z],math.log(s)/b
err=0.
for j in range(2000):
 k=rng.randrange(2,226);p=[rng.random()+1e-8 for _ in range(k)];p=[x/sum(p) for x in p]
 u=[rng.uniform(-1,1) for _ in p];v=[rng.uniform(-1,1) for _ in p];q,P=gibbs(p,u);r,Q=gibbs(p,v)
 kl=sum(x*math.log(x/y) for x,y in zip(q,r))
 e=abs(Q-P-sum(x*(y-z) for x,y,z in zip(q,v,u))-kl/4)
 err=max(err,e);assert e<1e-12 and kl>=-1e-12;n+=2
 m=rng.uniform(-1,1);y=rng.uniform(-1,1);M=(m+y)/2
 assert abs((M-m)-(y-m)/2)<1e-15;n+=1
print('Assertions',n,'rational positive-A cases despite added visits',up,'max Gibbs identity residual',err)
