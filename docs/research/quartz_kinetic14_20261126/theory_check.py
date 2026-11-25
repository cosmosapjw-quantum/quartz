from fractions import Fraction as F
from itertools import product
import math,random
n=0;neg=pos=0
def kl(a,b):return sum(x*math.log(x/y) for x,y in zip(a,b) if x)
def H(p):return -sum(x*math.log(x) for x in p if x)
for counts in product(range(3),repeat=3):
 N=sum(counts)
 if not N:continue
 for dn in product(range(3),repeat=3):
  M=sum(dn)
  if not M:continue
  r=[F(x,N) for x in counts];j=[F(x,M) for x in dn];g=F(M,N+M);p=[F(x+y,N+M) for x,y in zip(counts,dn)]
  J=[v-u for u,v in zip(r,p)]
  assert J==[g*(v-u) for u,v in zip(r,j)] and sum(J)==0;n+=2
  assert sum(map(abs,J))==g*sum(abs(v-u) for u,v in zip(r,j));n+=1
  pi=[1/6,2/6,3/6];js=(1-float(g))*kl(r,p)+float(g)*kl(j,p)
  assert -1e-12<=js<=H([float(g),1-float(g)])+1e-12
  d=kl(p,pi)-kl(r,pi)
  assert abs(d-float(g)*(kl(j,pi)-kl(r,pi))+js)<1e-12;n+=2;neg+=d<0;pos+=d>0
  for u in (F(0),F(1,2),F(1)):
   known=J[0]*u;lo=known+sum(min(0,z) for z in J[1:]);hi=known+sum(max(0,z) for z in J[1:])
   v=[known+sum(a*b for a,b in zip(J[1:],t)) for t in product((F(0),F(1)),repeat=2)]
   assert min(v)==lo and max(v)==hi;n+=2
for y in product((F(0),F(1,2),F(1)),repeat=4):
 a,b,c,d=y;assert (a-b-c+d)==(a-c)-(b-d);n+=1
 for x in range(4):
  for panel in ((x,),tuple(range(4))):
   for V in ((x,),panel):
    assert max(y[i] for i in panel)-y[x]==max(y[i] for i in panel)-max(y[i] for i in V)+max(y[i] for i in V)-y[x];n+=1
rng=random.Random(2026112614);err=0
for _ in range(400):
 y=[rng.uniform(-1,1) for i in range(25)]
 f=[int(i//5==1)+int(i%5==1) for i in range(25)]
 def value(e):
  w=[math.exp(e*z) for z in f];return sum(x*z for x,z in zip(y,w))/sum(w)
 exact=sum(a*b for a,b in zip(y,f))/25-sum(y)*sum(f)/625
 diff=(value(1e-5)-value(-1e-5))/(2e-5)
 err=max(err,abs(exact-diff));assert abs(exact-diff)<1e-9;n+=1
 a=[rng.random()+.1 for i in range(5)];a=[v/sum(a) for v in a]
 b=[]
 for i in range(5):
  t=[rng.random()+.1 for j in range(5)];b.append([v/sum(t) for v in t])
 path=[a[i]*b[i][j] for i in range(5) for j in range(5)]
 assert abs(kl(path,[1/25]*25)-kl(a,[.2]*5)-sum(a[i]*kl(b[i],[.2]*5) for i in range(5)))<1e-12;n+=1
print('Finite rational/numeric assertions',n,'KL decreases/increases',neg,pos,'response residual',err)
