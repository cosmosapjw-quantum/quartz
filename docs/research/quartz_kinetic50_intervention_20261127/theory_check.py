from itertools import product
from fractions import Fraction as F
from collections import Counter
C=Counter()
for e,a,y0,y1 in product((0,1),(0,1),range(3),range(3)):
 if a>e or (not a and y0!=y1):continue
 assert y1-y0==e*(y1-y0)==a*(y1-y0)
 for x in (0,1):
  assert (y1 if x and e else y0)==y0+x*e*(y1-y0);C['outcome_cases']+=1
for l,dq,db,pre in product(range(4),range(-3,4),range(-3,4),range(4)):
 q0=pre+4;b0=4;q1=q0+l+dq;b1=b0+db
 assert q1-q0==l+dq and q1+b1-q0-b0==l+dq+db;C['cost_cases']+=1
for g in product(range(-2,3),repeat=3):
 for cost in product(range(4),repeat=3):
  cap=sum(cost)//4
  v=[(sum(g[i] for i in range(3) if m>>i&1),-m.bit_count(),-m) for m in range(8) if sum(cost[i] for i in range(3) if m>>i&1)<=cap]
  assert max(v)[0]>=0
  m=-max(v)[2];c=sum(cost[i] for i in range(3) if m>>i&1)
  if sum(cost):assert F(c,sum(cost))*sum(cost)==c
  C['mask_cases']+=1
print(dict(C))
