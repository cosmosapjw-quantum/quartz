from budget import solve,F
from itertools import product
def brute(p,B):
    zs=list(p);best=None
    for choices in product(*(list(p[z]) for z in zs)):
        c=sum(p[z][h][0] for z,h in zip(zs,choices));g=sum(p[z][h][1] for z,h in zip(zs,choices))
        if c<=B:best=g if best is None else max(best,g)
        for i,z in enumerate(zs):
            h0=choices[i];k0,g0=p[z][h0]
            for h1,(k1,g1) in p[z].items():
                if k1==k0:continue
                q=F(B-c,k1-k0)
                if 0<=q<=1:
                    val=g+q*(g1-g0);best=val if best is None else max(best,val)
    return best
n=0
for a,b,c,d in product(range(-2,3),repeat=4):
    p={0:{20:(0,0),26:(1,a),32:(3,b)},1:{20:(0,0),26:(2,c),32:(3,d)}}
    for B in range(7):
        r=solve(p,B);assert r['gain']==brute(p,B);n+=1
assert solve({0:{20:(0,0),26:(0,2),32:(3,1)}},0)['gain']==2
assert not solve({0:{20:(0,0)}},-1)['feasible']
print('exact multi-choice vertex comparisons',n,'plus zero-cost/infeasible cases')
