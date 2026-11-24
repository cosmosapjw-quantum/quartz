from fractions import Fraction as F
from itertools import combinations,product
from random import Random
assert F(1,2)*(F(1,5)-F(4,5)*F(1,2))==-F(1,10)
assert F(1,4)*F(1,2)==F(1,8)
def top(g,k):return set(sorted(range(len(g)),key=lambda i:(-g[i],i))[:k])
def subset_value(g,s):return sum(g[i] for i in s)
def coeff(w,k):
 d=[F(1)]+[F(0)]*k
 for x in w:
  for j in range(k,0,-1):d[j]+=x*d[j-1]
 return d[k]
rng=Random(241124);n=0
for _ in range(400):
 g=[F(rng.randrange(-10,11),10) for i in range(6)]
 e=[F(rng.randrange(-2,3),10) for i in range(6)]
 h=[a+b for a,b in zip(g,e)]
 w=[F(rng.randrange(1,8),rng.randrange(1,8)) for i in range(6)]
 for k in range(7):
  combos=list(combinations(range(6),k))
  s=top(g,k); sh=top(h,k)
  assert subset_value(g,s)==max(subset_value(g,t) for t in combos)
  gap=subset_value(g,s)-subset_value(g,sh)
  assert 0<=gap<=sum(abs(e[i]) for i in s^sh)<=2*k*max(map(abs,e))
  assert coeff(w,k)==sum(__import__('math').prod(w[i] for i in t) for t in combos)
  n+=3
assert max((sum(g for i,(c,g) in enumerate([(3,5),(2,3),(2,3)]) if mask>>i&1) for mask in range(8) if sum(c for i,(c,g) in enumerate([(3,5),(2,3),(2,3)]) if mask>>i&1)<=4))==6
print('Fraction exchange/regret/partition assertions',n,'plus 3 counterexample checks')
