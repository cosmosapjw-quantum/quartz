from itertools import product
from fractions import Fraction as F
from collections import Counter
C=Counter()
for n in product(range(4),repeat=3):
 a=max(range(3),key=lambda i:(n[i],-i))
 d=min(n[a]-n[j]+int(j>a) for j in range(3) if j!=a)
 for r in range(5):
  assert (d>r)==all(max(range(3),key=lambda i:(n[i]+x[i],-i))==a for x in product(range(r+1),repeat=3) if sum(x)<=r)
  C['vote_lock']+=1
for gains in product(range(-2,3),repeat=3):
 for costs in product(range(4),repeat=3):
  tot=sum(costs);cap=tot//4
  feas=[(sum(gains[i] for i in range(3) if m>>i&1),-m.bit_count(),-m) for m in range(8) if sum(costs[i] for i in range(3) if m>>i&1)<=cap]
  g,_,neg=max(feas);m=-neg
  assert g>=0 and sum(costs[i] for i in range(3) if m>>i&1)<=cap
  if tot:
   p=F(sum(costs[i] for i in range(3) if m>>i&1),tot)
   assert p*tot==sum(costs[i] for i in range(3) if m>>i&1)
  C['allocation_cases']+=1
print(dict(C))
