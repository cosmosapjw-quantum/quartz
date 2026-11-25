from fractions import Fraction as F
from itertools import product
n=0
d=tuple(map(F,(-1,-.5,0,.5,1)))
def risk(w):
 z=sum(w);mu=sum(x*y for x,y in zip(d,w))/z
 return (sum(abs(x)*y for x,y in zip(d,w))/z-abs(mu))/2
for w in product(range(3),repeat=5):
 if not sum(w):continue
 z=sum(w);mu=sum(x*y for x,y in zip(d,w))/z
 assert risk(w)==min(sum(max(x,0)*y for x,y in zip(d,w))/z,sum(max(-x,0)*y for x,y in zip(d,w))/z);n+=1
 for mask in range(1,31):
  a=tuple(w[i] if mask>>i&1 else 0 for i in range(5));b=tuple(x-y for x,y in zip(w,a))
  if not sum(a) or not sum(b):continue
  m=lambda x:sum(t*q for t,q in zip(d,x))/sum(x)
  E=(sum(a)*risk(a)+sum(b)*risk(b))/z
  assert E<=risk(w)
  assert risk(w)-E==(sum(a)*abs(m(a))+sum(b)*abs(m(b))-z*abs(mu))/(2*z);n+=2
for a,b,c in product(range(-4,5),repeat=3):
 assert (a+c)-(b+c)==a-b;n+=1
for m in range(5):
 for s in range(-m,m+1):
  x=F(s,m+4)
  for k in range(5):
   for t in range(-k,k+1):
    y=(t+4*x)/(k+4);assert -1<=y<=1
    assert (k!=0 or y==x);n+=2
print('exact assertions',n)
