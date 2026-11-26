from functools import lru_cache
L=(7,56,448,73,146,292,273,84);F=511
def cells(e):
    while e:
        m=e&-e;yield m;e^=m
def win(a):return any(a&l==l for l in L)
def im(a,b):return tuple(m for m in cells(F^(a|b)) if win(a|m))
@lru_cache(None)
def mm(a,b):
    if win(b):return -1
    if a|b==F:return 0
    return max(-mm(b,a|m) for m in cells(F^(a|b)))
@lru_cache(None)
def attack(a,b,d):
    if d<=0 or win(b) or a|b==F:return False
    return any(after(a,b,m,d) for m in cells(F^(a|b)))
def after(a,b,m,d):
    x=a|m
    if win(x):return d>=1
    if d<3 or im(b,x):return False
    q=im(x,b)
    if len(q)>=2:return True
    return len(q)==1 and attack(x,b|q[0],d-2)
seen=set();todo=[(0,0)];ct=[0,0,0,0]
while todo:
    a,b=todo.pop()
    if (a,b) in seen:continue
    seen.add((a,b));ct[0]+=1
    if win(b) or a|b==F:continue
    E=list(cells(F^(a|b)));truth={m:-mm(b,a|m) for m in E}
    for d in (1,3,5,7):
        bd={}
        for m in E:
            l,u=-1,1
            if after(a,b,m,d):l=u=1
            elif a|b|m==F:l=u=0
            elif attack(b,a|m,d-1):l=u=-1
            assert l<=truth[m]<=u
            bd[m]=(l,u);ct[1]+=1
        for m,(l,u) in bd.items():
            if all(l>=x[1] for k,x in bd.items() if k!=m):assert truth[m]==max(truth.values());ct[2]+=1
            if u==-1:
                assert all(v>=truth[m] for v in truth.values());ct[3]+=1
    todo.extend((b,a|m) for m in E)
print('states, bounded action facts, optimum certificates, weak replacements',ct)
