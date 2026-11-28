from itertools import product
from fractions import Fraction as F
from collections import Counter
c=Counter()
def best(g,k,l):return max(range(3),key=lambda i:(g[i]-l*k[i],-i))
for a,b,u,v in product(range(-2,3),range(-2,3),range(4),range(4)):
 g=[0,a,b];k=[0,u,v];ls=[F(i,4) for i in range(25)]
 cs=[k[best(g,k,l)] for l in ls];assert all(x>=y for x,y in zip(cs,cs[1:]))
 cuts={F(0)}
 for i in range(3):
  for j in range(i):
   if k[i]!=k[j]:
    z=F(g[i]-g[j],k[i]-k[j])
    if z>=0:cuts.add(z)
 cuts=sorted(cuts);mid=[(x+y)/2 for x,y in zip(cuts,cuts[1:])]
 ops={best(g,k,l) for l in cuts+mid+[cuts[-1]+1]}
 assert {best(g,k,l) for l in ls}<=ops;c['price']+=1
for a,b,x,y in product(range(-2,3),repeat=4):
 assert (a+b*x-y)**2+16*b*b>=0
 assert (a+b*x)**2+16*b*b>0 or (a==b==0);c['positive']+=1
print(dict(c))
