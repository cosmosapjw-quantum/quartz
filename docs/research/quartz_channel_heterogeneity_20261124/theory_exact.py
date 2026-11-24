from fractions import Fraction as F
from itertools import product
import random
def moment(atoms,n):
    v=[w*r[0]**n[0]*r[1]**n[1] for u,r,w in atoms]
    return sum(v),sum(u*x for (u,r,w),x in zip(atoms,v))
def exhaustive(atoms,b,c,h,n=(0,0)):
    m,a=moment(atoms,n); stop=max(b,a/m)
    if not h:return stop
    return max(stop,-c+sum(moment(atoms,nn)[0]/m*exhaustive(atoms,b,c,h-1,nn) for nn in [(n[0]+1,n[1]),(n[0],n[1]+1)] if moment(atoms,nn)[0]))
def worlds(atoms,b,c,h):
    stop=max(b,sum(u*w for u,r,w in atoms)/sum(w for u,r,w in atoms))
    if not h:return stop
    total=sum(w for u,r,w in atoms)
    vals=[]
    for y in range(2):
        ws=[(u,r,w*r[y]) for u,r,w in atoms]
        den=sum(w for u,r,w in ws)
        if den: vals.append(den/total*worlds(ws,b,c,h-1))
    return max(stop,sum(vals)-c)
a=[(F(0),(F(1,10),F(9,10)),F(1,4)),(F(0),(F(9,10),F(1,10)),F(1,4)),(F(1),(F(1,2),F(1,2)),F(1,2))]
f=[(F(0),(F(1,2),F(1,2)),F(1,2)),a[-1]]
b,c=F(1,2),F(1,100)
assert exhaustive(a,b,c,1)==b and exhaustive(a,b,c,2)==F(14,25)
assert exhaustive(f,b,c,2)==b
rng=random.Random(418); count=0
for _ in range(300):
    atoms=[]
    for j in range(4):
        p=F(rng.randrange(1,10),10)
        atoms.append((F(rng.randrange(3),2),(p,1-p),F(1,4)))
    for h in range(3):
        assert exhaustive(atoms,b,c,h)==worlds(atoms,b,c,h);count+=1
    for n in [(0,0),(1,0),(2,1)]:
        m,v=moment(atoms,n); children=[moment(atoms,(n[0]+1,n[1])),moment(atoms,(n[0],n[1]+1))]
        assert sum(x[0] for x in children)==m and sum(x[1] for x in children)==v
        count+=2
print('exact Fraction assertions',count+3,'A H1',exhaustive(a,b,c,1),'A H2',exhaustive(a,b,c,2),'F H2',exhaustive(f,b,c,2))
