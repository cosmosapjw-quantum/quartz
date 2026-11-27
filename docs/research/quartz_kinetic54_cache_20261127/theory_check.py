from itertools import product
from fractions import Fraction as F
from collections import Counter
c=Counter()
for q,b in product(range(1,21),repeat=2):
 for hq,hb in product(range(q+1),range(b)):
  a=1-F(q-hq,b-hb)-(1-F(q,b))
  assert a==F(b*hq-q*hb,b*(b-hb))
  assert (a>=0)==(F(hq,q)>=F(hb,b));c['fair_cache']+=1
for xs in product(range(3),repeat=8):
 seen=set();hits=0
 for x in xs:
  hits+=x in seen;seen.add(x)
 assert len(seen)+hits==len(xs);c['trace']+=1
for z,e,ch,y0,y1 in product((0,1),(0,1),(0,1),range(3),range(3)):
 if e>z or ch>e or (not ch and y0!=y1):continue
 assert y1-y0==e*(y1-y0)==ch*(y1-y0);c['conditional']+=1
print(dict(c))
