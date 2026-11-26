from functools import lru_cache
lines=[7,56,448,73,146,292,273,84];FULL=511
def cells(x):
    while x:
        t=x&-x;yield t;x^=t
def won(a):return any(a&l==l for l in lines)
def imm(a,b):return [t for t in cells(FULL^(a|b)) if won(a|t)]
@lru_cache(None)
def brute(a,b):
    if won(b):return -1
    if a|b==FULL:return 0
    return max(-brute(b,a|t) for t in cells(FULL^(a|b)))
@lru_cache(None)
def fast(a,b):
    if won(b):return -1
    if a|b==FULL:return 0
    if imm(a,b):return 1
    o=imm(b,a)
    if len(o)>=2:return -1
    if o:return -fast(b,a|o[0])
    if all(l&b for l in lines) and all(l&a for l in lines):return 0
    best=-1
    for t in cells(FULL^(a|b)):
        best=max(best,-fast(b,a|t))
        if best==1:return 1
    return best
seen=set();stack=[(0,0)];checks=0;actions=0
while stack:
    a,b=stack.pop()
    if (a,b) in seen:continue
    seen.add((a,b))
    assert fast(a,b)==brute(a,b);checks+=1
    if won(b) or a|b==FULL:continue
    for t in cells(FULL^(a|b)):
        assert -fast(b,a|t)==-brute(b,a|t);actions+=1;stack.append((b,a|t))
print('exact reachable 3x3 states',checks,'action checks',actions)
assert checks>5000
