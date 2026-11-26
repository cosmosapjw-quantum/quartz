from functools import lru_cache
from itertools import product
LINES=(7,56,448,73,146,292,273,84);FULL=511
def won(a):return any(a&x==x for x in LINES)
def cells(e):
    while e:
        t=e&-e;yield t;e^=t
@lru_cache(None)
def mm(a,b):
    if won(b):return -1
    if a|b==FULL:return 0
    return max(-mm(b,a|m) for m in cells(FULL^(a|b)))
@lru_cache(None)
def P(a,b,t):
    if won(b):return False
    if a|b==FULL:return t==0
    return any(not P(b,a|m,1-t) for m in cells(FULL^(a|b)))
n=acts=0;seen=set();stack=[(0,0)]
while stack:
    a,b=stack.pop()
    if (a,b) in seen:continue
    seen.add((a,b));n+=1
    assert all(P(a,b,t)==(mm(a,b)>=t) for t in (0,1))
    if won(b) or a|b==FULL:continue
    E=list(cells(FULL^(a|b)));truth={m:-mm(b,a|m) for m in E}
    lo={m:-1 for m in E};hi={m:1 for m in E};choice=None
    for t in (1,0):
        for m in E:
            z=not P(b,a|m,1-t)
            if z:lo[m]=max(lo[m],t);choice=m;break
            hi[m]=min(hi[m],t-1)
        if choice is not None:break
    if choice is None:choice=E[0]
    assert choice==min(m for m,v in truth.items() if v==max(truth.values()))
    assert all(lo[m]<=truth[m]<=hi[m] for m in E)
    assert lo[choice]>=max(hi.values()) and all(hi[m]<lo[choice] for m in E if m<choice)
    acts+=len(E);stack.extend((b,a|m) for m in E)
print('reachable states',n,'threshold predicates',2*n,'root action checks',acts)
