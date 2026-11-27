from itertools import product
from fractions import Fraction as F
from collections import Counter
c=Counter()
for a,b,d,g,h in product(range(3),repeat=5):
 vals=(a,b,d)
 assert vals[g]-a==(vals[g]-vals[h])+(vals[h]-a);c['pair']+=1
for g1,g2,c1,c2 in product(range(-2,3),range(-2,3),range(1,5),range(1,5)):
 assert F(g1+g2,c1+c2)==F(c1,c1+c2)*F(g1,c1)+F(c2,c1+c2)*F(g2,c2);c['ratio']+=1
for gs in product(range(-1,2),repeat=4):
 for cs in product(range(1,3),repeat=4):
  vals=[]
  for t in product(range(3),repeat=2):
   g=sum(gs[2*i+x-1] for i,x in enumerate(t) if x);co=sum(cs[2*i+x-1] for i,x in enumerate(t) if x)
   if co<=2:vals.append((g,-co,-sum(x>0 for x in t),tuple(-x for x in t)))
  assert max(vals)[0]>=0;c['feasible']+=1
print(dict(c))
