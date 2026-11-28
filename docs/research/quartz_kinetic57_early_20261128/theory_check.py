from fractions import Fraction as F
from itertools import product
from collections import Counter
c=Counter()
for a,b,y,z in product(range(3),repeat=4):
 if a==b and y!=z:continue
 assert z-y==int(a!=b)*(z-y);c['factor']+=1
for x,y,u,v in product(range(-2,3),repeat=4):
 assert F(max(x,y)+max(u,v),2)>=max(F(x+u,2),F(y+v,2));c['jensen']+=1
for n in (4,8,12,16):
 for d in range(1,n+2):
  for a,b in product(range(3),repeat=2):
   if d>20-n:assert not(d<=20-n)
   assert (a-b)==(a-b)*1;c['boundary']+=1
for n,m,h,f in product(range(4),repeat=4):
 if n==m+h+f:assert n-f==m+h;c['cache']+=1
print(dict(c))
