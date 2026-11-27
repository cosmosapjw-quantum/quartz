from itertools import product
from collections import Counter
c=Counter();V=(-2,-1,0,1,2)
for x,y,z in product(V,repeat=3):
 assert max(abs(x-y),abs(y-z),abs(x-z))==max(x,y,z)-min(x,y,z);c['diameter']+=1
for l,u,x,y in product(V,repeat=4):
 if l>u:continue
 a=max(l,min(u,x));b=max(l,min(u,y));assert abs(a-b)<=abs(x-y) and (x<=y)==(x<=y and a<=b);c['clip']+=1
for g in product((-1,0,1),repeat=6):
 for cost in product(range(3),repeat=2):
  masks=[m for m in range(4) if sum(cost[i] for i in range(2) if m>>i&1)<=sum(cost)//4]
  def objective(m,H):return min(sum(g[2*h+i] for i in range(2) if m>>i&1) for h in H)
  a=max(objective(m,(0,1)) for m in masks);b=max(objective(m,(0,1,2)) for m in masks)
  assert 0<=b<=a;c['robust']+=1
print(dict(c))
