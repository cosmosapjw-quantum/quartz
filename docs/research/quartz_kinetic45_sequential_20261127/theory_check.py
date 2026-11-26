from itertools import product
from functools import lru_cache
L=(7,56,448,73,146,292,273,84);F=511
def cells(e):
    while e:
        m=e&-e;yield m;e^=m
def won(a):return any(a&l==l for l in L)
def imm(a,b):return tuple(m for m in cells(F^(a|b)) if won(a|m))
@lru_cache(None)
def mm(a,b):
    if won(b):return -1
    if a|b==F:return 0
    return max(-mm(b,a|m) for m in cells(F^(a|b)))
@lru_cache(None)
def attack(a,b,d):
    if d<=0 or won(b) or a|b==F:return False
    return any(after(a,b,m,d) for m in cells(F^(a|b)))
def after(a,b,m,d):
    x=a|m
    if won(x):return d>=1
    if d<3 or imm(b,x):return False
    q=imm(x,b)
    if len(q)>=2:return True
    return len(q)==1 and attack(x,b|q[0],d-2)
seen=set();stack=[(0,0)];facts=gains=monotone=0
while stack:
    a,b=stack.pop()
    if (a,b) in seen:continue
    seen.add((a,b))
    p=[attack(a,b,d) for d in range(1,8)]
    assert p==sorted(p)
    for d,q in enumerate(p,1):
        if q:assert mm(a,b)==1
        monotone+=1
    if won(b) or a|b==F:continue
    E=list(cells(F^(a|b)));truth={m:-mm(b,a|m) for m in E};bd={m:(-1,1) for m in E}
    for d in (3,5,7):
        for m in E:
            l,u=bd[m];x=a|m
            if after(a,b,m,d):l=u=1
            elif x|b==F:l=u=0
            elif attack(b,x,d-1):l=u=-1
            assert l<=truth[m]<=u and bd[m][0]<=l<=u<=bd[m][1];bd[m]=(l,u);facts+=1
        for i,j in product(E,repeat=2):
            l,u=bd[i];p,q=bd[j];lo,hi=(0,0) if i==j else (l-q,u-p)
            if all(l>=bd[m][1] for m in E if m!=i):lo=max(lo,0)
            if all(p>=bd[m][1] for m in E if m!=j):hi=min(hi,0)
            assert lo<=truth[i]-truth[j]<=hi
            gains+=1
    stack.extend((b,a|m) for m in E)
print('states',len(seen),'monotone depth checks',monotone,'root fact checks',facts,'gain interval checks',gains)
