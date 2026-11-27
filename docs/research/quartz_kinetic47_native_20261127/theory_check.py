from itertools import product
from collections import Counter
C=Counter()
for n in product(range(5),repeat=3):
 a=max(range(3),key=lambda i:(n[i],-i))
 d=min(n[a]-n[j]+int(j>a) for j in range(3) if j!=a)
 for r in range(6):
  possible=[max(range(3),key=lambda i:(n[i]+x[i],-i)) for x in product(range(r+1),repeat=3) if sum(x)<=r]
  assert (all(j==a for j in possible))==(d>r);C['vote_cases']+=1
for l,u,m,v in product(range(-1,2),repeat=4):
 if l>u or m>v:continue
 for x,y in product(range(-8,9),repeat=2):
  a=max(l,min(u,x/8));b=max(m,min(v,y/8))
  assert l-v<=a-b<=u-m;C['clipped_gain']+=1
print(dict(C))
