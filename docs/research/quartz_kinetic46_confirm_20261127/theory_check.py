from itertools import product
from collections import Counter
V=(-1,0,1);I=[(l,u) for l in V for u in V if l<=u]
def S(a,b,oa,ob,same=False):
    return {(x,y) for x in V for y in V if a[0]<=x<=a[1] and b[0]<=y<=b[1] and (not oa or x>=y) and (not ob or x<=y) and (not same or x==y)}
def interval(s):
    return min(x-y for x,y in s),max(x-y for x,y in s)
c=Counter()
for a,b,oa,ob,same in product(I,I,(False,True),(False,True),(False,True)):
    s=S(a,b,oa,ob,same)
    if not s:continue
    l,u=interval(s);c['feasible_pairs']+=1
    if same:assert (l,u)==(0,0)
    for aa,bb in product(I,I):
        if not (a[0]<=aa[0]<=aa[1]<=a[1] and b[0]<=bb[0]<=bb[1]<=b[1]):continue
        ss=S(aa,bb,oa,ob,same)
        if not ss:continue
        ll,uu=interval(ss);assert ss<=s and l<=ll<=uu<=u;c['nested']+=1
    for x,y in s:
        for t in (0,1):
            aa=(max(a[0],t),a[1]) if x>=t else (a[0],min(a[1],t-1))
            assert (x,y) in S(aa,b,oa,ob,same);c['threshold_updates']+=1
print(dict(c))
