from random import Random
from itertools import product
from fractions import Fraction as F
r=Random(261134);N=225
dirs=((0,1),(1,0),(1,1),(1,-1))
lines=[tuple(15*(i+t*di)+j+t*dj for t in range(5)) for i,j in product(range(15),repeat=2) for di,dj in dirs if 0<=i+4*di<15 and 0<=j+4*dj<15]
assert len(lines)==572
def incidence(b,p):
    w=set();q=[set() for _ in range(N)]
    for l in lines:
        a=[i for i in l if b[i]==0]
        if any(b[i]==3-p for i in l):continue
        if len(a)==1:w.add(a[0])
        elif len(a)==2:x,y=a;q[x].add(y);q[y].add(x)
    return w,q
def win(b,a,p):
    if b[a]:return False
    i,j=divmod(a,15)
    for di,dj in dirs:
        n=1
        for s in (-1,1):
            x,y=i+s*di,j+s*dj
            while 0<=x<15 and 0<=y<15 and b[15*x+y]==p:
                n+=1;x+=s*di;y+=s*dj
        if n>=5:return True
    return False
def winning(b,p):return {a for a in range(N) if win(b,a,p)}
cases=0
for t in range(256):
    b=[0]*N
    for i,a in enumerate(r.sample(range(N),r.randrange(4,75))):b[a]=i%2+1
    if t%4==0:
        for a in (110,111,113):b[a]=1
        b[112]=0
    for p in (1,2):
        w,q=incidence(b,p);v,_=incidence(b,3-p)
        assert w==winning(b,p) and v==winning(b,3-p)
        for a in r.sample([i for i in range(N) if not b[i]],8):
            b[a]=p
            assert (w-{a})|q[a]==winning(b,p)
            assert v-{a}==winning(b,3-p)
            b[a]=0;cases+=1
for _ in range(8192):
    n=[r.randrange(20) for i in range(r.randrange(2,24))]
    L=max(range(len(n)),key=lambda i:(n[i],i));d=min(n[L]-n[i]+int(i<L) for i in range(len(n)) if i!=L)
    H=r.randrange(24)
    delta=[0]*len(n)
    for i in range(H):delta[r.randrange(len(n))]+=1
    if d>H:assert max(range(len(n)),key=lambda i:(n[i]+delta[i],i))==L
    if d<=H:
        j=next(i for i in range(len(n)) if i!=L and n[L]-n[i]+int(i<L)==d);nn=n.copy();nn[j]+=d
        assert max(range(len(n)),key=lambda i:(nn[i],i))!=L
print(cases,'independent direct winning-set comparisons; 8192 exact vote reachability checks')
