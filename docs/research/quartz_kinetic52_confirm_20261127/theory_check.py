from itertools import product
from fractions import Fraction as F
from collections import Counter
c=Counter()
for n,k,q,z in product(range(1,21),range(1,6),range(-4,5),range(-4,5)):
 q,z=F(q,4),F(z,4);m=(n*q+k*z)/(n+k)
 assert m-z==F(n,n+k)*(q-z) and abs(m-z)<=F(2*n,n+k);c['reuse']=c['reuse']+1
for ns in product(range(4),repeat=3):
 a=max(range(3),key=lambda i:(ns[i],i));d=min(ns[a]-ns[j]+int(j<a) for j in range(3) if j!=a)
 for r in range(5):
  assert (d>r)==all(max(range(3),key=lambda i:(ns[i]+x[i],i))==a for x in product(range(r+1),repeat=3) if sum(x)<=r);c['vote']+=1
for a,b in product(range(6),repeat=2):
 assert 2+10-a-b<=12;c['physical']+=1
print(dict(c))
