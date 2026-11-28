from fractions import Fraction as F
from itertools import product
from math import exp
n=0
ps=[tuple(F(i,4) for i in x) for x in product(range(5),repeat=5) if sum(x)==4]
d=[F(-1),F(-1,2),F(0),F(1,2),F(1)]
for p in ps:
 g=sum(x*y for x,y in zip(p,d))
 for q in ps:
  h=sum(x*y for x,y in zip(q,d))
  regret=max(F(0),g)-(g if h>0 else F(0))
  assert 0<=regret<=abs(g-h)<=sum(abs(a-b) for a,b in zip(p,q))
  b=sum(p[k]*sum((F(int(j==k))-q[j])**2 for j in range(5)) for k in range(5))
  assert b==1-sum(x*x for x in p)+sum((x-y)**2 for x,y in zip(p,q));n+=1
def gain(t):
 z=[exp(x/t) for x in [-4,4,0,-4,3]];return sum(float(a)*b for a,b in zip(d,z))/sum(z)
assert gain(.5)<0<gain(10)
print('exact finite checks',n,'temperature sign counterexample',gain(.5),gain(10),'Wolfram signed mean, squared gap, contrast verified')
