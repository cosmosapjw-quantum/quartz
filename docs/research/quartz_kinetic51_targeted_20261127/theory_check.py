from itertools import product
from fractions import Fraction as F
from collections import Counter
C=Counter()
for l,u,m,v in product(range(-1,2),repeat=4):
 if l>u or m>v:continue
 for a,b in product(range(-1,2),repeat=2):
  if not(l<=a<=u and m<=b<=v):continue
  assert m-u<=b-a<=v-l
  if l>=v:assert a>=b
  C['wdl_pairs']+=1
 for x,y in product(range(-4,5),repeat=2):
  a=max(l,min(u,F(x,4)));b=max(m,min(v,F(y,4)))
  assert m-u<=b-a<=v-l;C['clipped_pairs']+=1
for a,b in product(range(6),repeat=2):
 assert 2+10-a-b<=12 and 2+10-a-b>=2;C['cost_cases']+=1
for xs in product(range(-1,2),repeat=5):
 for mask in range(32):
  a=[x for j,x in enumerate(xs) if mask>>j&1];b=[x for j,x in enumerate(xs) if not mask>>j&1]
  assert F(sum(a)+sum(b),5)==F(sum(xs),5);C['backup_cases']+=1
print(dict(C))
