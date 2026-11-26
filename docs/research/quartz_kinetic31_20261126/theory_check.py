from random import Random
from fractions import Fraction as F
r=Random(261131)
for _ in range(8192):
 y=[F(r.randrange(-2,3),2) for i in range(9)]+[F(0)]
 m=[F(r.randrange(-10,11),10) for i in range(9)]+[F(0)]
 a,b=r.randrange(10),r.randrange(10)
 z=[m[a]-m[b]+9*((y[j]-m[j])*(int(j==a)-int(j==b))) for j in range(9)]
 assert sum(z)/9==y[a]-y[b]
 if a==b:assert all(x==0 for x in z)
 u=[r.randrange(-1,2) for i in range(r.randrange(2,8))]
 lo=[r.randrange(-1,x+1) for x in u];hi=[r.randrange(x,2) for x in u]
 l2=[r.randrange(l,x+1) for l,x in zip(lo,u)];h2=[r.randrange(x,h+1) for x,h in zip(u,hi)]
 def risk(l,h):return min(max(0,max(h[j] for j in range(len(u)) if j!=i)-l[i]) for i in range(len(u)))
 assert risk(l2,h2)<=risk(lo,hi)
 for i,x in enumerate(u):
  z=max(0,max(hi[j] for j in range(len(u)) if j!=i)-lo[i])
  assert max(u)-x<=z
  if z==0:assert x==max(u)
 assert max(-x for x in hi)<=max(-x for x in u)<=max(-x for x in lo)
print('8192 exact rational known-propensity contrasts and integer interval/witness/negamax cases')
