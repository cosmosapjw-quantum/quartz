from fractions import Fraction as F
from itertools import product
from collections import Counter
c=Counter()
for n in range(1,21):
 for w in range(-n,n+1):
  for v in (-1,0,1):
   a=F(w,n);b=F(w+v,n+1)
   assert b-a==F(v-a,n+1) and abs(b-a)<=F(2,n+1);c['update']+=1
for n,m in product(range(1,8),repeat=2):
 for w,z in product(range(-n,n+1),range(-m,m+1)):
  assert (F(w,n)>=F(z,m))==(w*m>=z*n);c['cross']+=1
for a,b,d,e in product(range(1,8),repeat=4):
 assert (1-F(a,b))-(1-F(d,e))==F(d,e)-F(a,b);c['ratio']+=1
for h in (4,8,12,16):
 assert (h%8==4)!= (h%8==0)
 c['geometry']+=1
assert F(4,10)/4>F(1,10)/2 and F(-6,10)/5<F(1,10)/2
print(dict(c))
