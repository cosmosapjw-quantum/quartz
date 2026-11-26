from random import Random
from fractions import Fraction as F
r=Random(261136)
for _ in range(8192):
 n=[r.randrange(50) for j in range(r.randrange(2,31))];L=max(range(len(n)),key=lambda j:(n[j],j))
 d=min(n[L]-n[j]+int(j<L) for j in range(len(n)) if j!=L)
 h=r.randrange(80);v=n.copy()
 for i in range(h):v[r.randrange(len(n))]+=1
 if d>h:assert max(range(len(v)),key=lambda j:(v[j],j))==L
 else:
  j=next(j for j in range(len(n)) if j!=L and n[L]-n[j]+int(j<L)==d)
  v=n.copy();v[j]+=d;assert max(range(len(v)),key=lambda j:(v[j],j))!=L
 u,v=F(r.randrange(-2,3),2),F(r.randrange(-2,3),2)
 assert abs((u+v)/2)<=max(abs(u),abs(v))
 a,b=r.randrange(257),r.randrange(257);p=F(r.randrange(101),100)
 assert min(a,b)<=p*a+(1-p)*b<=max(a,b)
for n in range(32):
 for k in range(1,9):
  if n+k>=32:
   assert n+k<=39 and n+k+1<80-(n+k)
print('8192 exact visit reachability, dual label and weighted stratum checks; probe-before-lock bound')
