"""Exact finite cross-check of C25 contract; not a substitute for its proof."""
from fractions import Fraction as F
from itertools import product
import json,math,pathlib,random,time
start=time.perf_counter()
grid=[F(k,4) for k in range(5)]
ints=[(a,b) for a in grid for b in grid if a<=b]
counts={'boxes':0,'incumbents':0,'corners':0,'strict_switches':0,'nested_checks':0,'charged_comparisons':0,'rounded_checks':0}
def gap(bs,i):return max(F(0),max(b for j,(a,b) in enumerate(bs) if j!=i)-bs[i][0])
for bs in product(ints,repeat=3):
 counts['boxes']+=1
 for i in range(3):
  counts['incumbents']+=1
  j=max(range(3),key=lambda k:(bs[k][0],-k))
  dest=j if bs[j][0]>bs[i][1] else i
  g=gap(bs,i);gn=gap(bs,dest);assert gn<=g
  assert not any(bs[k][0]>bs[dest][1] for k in range(3))
  if dest!=i:
   counts['strict_switches']+=1
   assert gn<=g-(bs[dest][0]-bs[i][0])
  regrets=[]
  for u in product(*[(a,b) for a,b in bs]):
   counts['corners']+=1
   assert u[dest]>=u[i]
   regrets.append(max(u)-u[i]);assert regrets[-1]<=g
   if dest!=i:assert u[dest]>u[i]
   for charge in [F(1,8),F(1),F(2)]:
    price=F(1,2);floor=F(1,8)
    if gn<=price*floor:
     assert max(u)-u[dest]-price*charge<=0
     counts['charged_comparisons']+=1
  assert max(regrets)==g
  narrow=tuple((a+(b-a)/4,b-(b-a)/4) for a,b in bs)
  assert gap(narrow,dest)<=gn;counts['nested_checks']+=1
rng=random.Random(2500923)
for _ in range(20000):
 l=rng.random();u=rng.random();exact=max(F(0),F(u)-F(l))
 gu=0. if u<=l else math.nextafter(u-l,math.inf)
 n=rng.randrange(4,16385);k=rng.randrange(2,9);c=n*k
 p=math.ldexp(1.,-30);td=max(0.,math.nextafter(p*c,-math.inf))
 assert F(gu)>=exact and F(td)<=F(p)*c
 if gu<=td: assert exact<=F(p)*c
 counts['rounded_checks']+=1
# Predictive cost can reverse the economic decision even without a free switch.
assert F(1,10)-F(1,100)==F(9,100)>0
# Max lower bound does not necessarily minimize worst-case incumbent regret.
b=((F(1,2),F(1)),(F(3,5),F(7,10)))
assert gap(b,0)==F(1,5)<gap(b,1)==F(2,5)
result={'status':'PASS','method':'exhaustive rational interval boxes/corners plus deterministic binary64 directed-rounding cases','counts':counts,'elapsed_seconds':time.perf_counter()-start,'proof_scope':'finite cross-check of exact published lemmas','seed':2500923}
path=pathlib.Path(__file__).resolve().parents[1]/'evidence/R07_EXACT_CROSSCHECK.json'
assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
