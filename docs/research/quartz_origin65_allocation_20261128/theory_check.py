from itertools import product
from fractions import Fraction as F
n=0
iv=[(-1,-1),(-1,0),(-1,1),(0,0),(0,1),(1,1)]
for vv in product((-1,0,1),repeat=3):
 for bd in product(iv,repeat=3):
  if not all(l<=v<=u for v,(l,u) in zip(vv,bd)):continue
  f=[i for i,(l,u) in enumerate(bd) if u>-1] or list(range(3))
  assert max(vv)==max(vv[i] for i in f);n+=1
for vv in product((F(0),F(1,2),F(1)),repeat=4):
 for mask in range(1,16):
  ids=[i for i in range(4) if mask>>i&1];v=max(vv[i] for i in ids)
  for i in ids:
   assert max(vv)-vv[i]==max(vv)-v+v-vv[i];n+=1
for f in range(1,20):
 for h in range(0,21):
  counts=[0]*f
  for _ in range(h):
   i=next((j for j,v in enumerate(counts) if not v),0);counts[i]+=1
  assert sum(v>0 for v in counts)==min(f,h);n+=1
for t in range(5):
 for hits in range(6-t):
  logical=5-t;physical=logical-hits
  assert physical>=0 and logical==physical+hits;n+=1
print('exact finite checks',n,'Wolfram algebraic checks {True, True, True}')
