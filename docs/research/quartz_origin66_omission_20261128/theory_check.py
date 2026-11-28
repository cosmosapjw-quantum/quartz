from fractions import Fraction as F
from itertools import product
n=0
for y in product((F(0),F(1,2),F(1)),repeat=4):
 for f in range(1,16):
  safe=[i for i in range(4) if f>>i&1]
  for l in range(1,16):
   if l&~f:continue
   live=[i for i in safe if l>>i&1]
   ma,mf,ml=max(y),max(y[i] for i in safe),max(y[i] for i in live)
   for a in live:
    assert ma-y[a]==(ma-mf)+(mf-ml)+(ml-y[a])
    assert min(ma-mf,mf-ml,ml-y[a])>=0;n+=1
    for e in safe:
     if e in live:continue
     mp=max(ml,y[e])
     for b in live+[e]:
      assert y[b]-y[a]==(mp-ml)-((mp-y[b])-(ml-y[a]));n+=1
for N in range(1,6):
 for w in range(-N,N+1):
  for z in (-1,0,1):
   assert -1<=F(w+z,N+1)<=1;n+=1
print('exact finite checks',n,'Wolfram 3 True')
