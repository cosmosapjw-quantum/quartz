from fractions import Fraction as F
from itertools import product
from collections import Counter
c=Counter()
for n,k,q,z in product(range(1,21),range(1,6),range(-4,5),range(-4,5)):
 q,z=F(q,4),F(z,4);w=n*q;x=k*z
 assert (w+x)/(n+k)-z==F(n,n+k)*(q-z)
 assert (w+x-w)/k==z;c['reuse']+=1
for seq in product(range(3),repeat=8):
 seen=set();m=h=0
 for x in seq:
  if x in seen:h+=1
  else:m+=1;seen.add(x)
 assert m==len(set(seq)) and m+h==len(seq);c['cache']+=1
for a,b,d in product(range(3),repeat=3):
 assert d-a==(d-b)+(b-a);c['telescoping']+=1
for init1,init2,t1,t2 in product(range(2),range(2),range(6),range(6)):
 assert 0<=10+init1+init2-t1-t2<=12;c['calls']+=1
print(dict(c))
