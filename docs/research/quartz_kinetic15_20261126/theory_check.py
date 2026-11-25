from fractions import Fraction as F
from itertools import product
import math
n=0
def norm(x):return tuple(F(v,sum(x)) for v in x)
def tv(x,y):return sum(abs(a-b) for a,b in zip(x,y))/2
def winner(x):return max(range(len(x)),key=lambda i:(x[i],i))
def kl(x,y):return sum(float(a)*math.log(float(a/b)) for a,b in zip(x,y) if a)
def H(x):return -sum(float(a)*math.log(float(a)) for a in x if a)
for N in product(range(3),repeat=3):
 if not sum(N):continue
 for f in product(range(2),repeat=3):
  for g in product(range(2),repeat=3):
   base=tuple(a+c for a,c in zip(N,g));m=tuple(a+b for a,b in zip(base,f))
   r,s=norm(m),norm(base);w=F(sum(f),sum(m))
   assert tv(r,s)<=w;n+=1
   if sum(f):
    t=norm(f);assert tv(r,s)==w*tv(t,s);n+=1
   for u in product((0,1),repeat=3):
    assert abs(sum((a-b)*c for a,b,c in zip(r,s,u)))<=tv(r,s);n+=1
   p=(F(1,6),F(2,6),F(3,6));parts=[x for x in (N,f,g) if sum(x)]
   ws=[F(sum(x),sum(m)) for x in parts]
   q=[norm(x) for x in parts];I=sum(float(a)*kl(b,r) for a,b in zip(ws,q))
   assert -1e-12<=I<=H(ws)+1e-12
   assert abs(sum(float(a)*kl(b,p) for a,b in zip(ws,q))-kl(r,p)-I)<1e-12;n+=2
  a=tuple(x+y for x,y in zip(N,f));L=winner(a)
  for b in range(5):
   lock=all((a[L],L)>(a[j]+b,j) for j in range(3) if j!=L)
   real=all(winner(tuple(x+y for x,y in zip(a,g)))==L for g in product(range(b+1),repeat=3) if sum(g)==b)
   assert lock==real;n+=1
assert winner((12,20))!=winner((12,4))
print('finite assertions',n,'exact forced count reversal verified')
