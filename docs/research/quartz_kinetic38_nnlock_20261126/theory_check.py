from itertools import product
from fractions import Fraction as F
def winner(n): return max(range(len(n)),key=lambda i:(n[i],-i))
def distance(n):
    l=winner(n)
    return min((n[l]-n[i]+int(i>l) for i in range(len(n)) if i!=l),default=float('inf'))
def run():
    checks=0;certificates=0;sharp=0
    for n in product(range(5),repeat=4):
        l=winner(n);d=distance(n)
        for r in range(6):
            possible=all(winner(tuple(a+b for a,b in zip(n,x)))==l for x in product(range(r+1),repeat=4) if sum(x)<=r)
            assert possible==(d>r),(n,r,d,possible)
            checks+=1;certificates+=possible;sharp+=not possible
    assert distance((1,0))==2 and distance((0,1))==1
    for p in map(lambda x:F(x,10),range(1,11)):
        for b in range(1,6):
            for l in range(b+1):
                assert 1-p*l/(p*b)==1-F(l,b)
    print(dict(simplex_cases=checks,locked=certificates,reachable_relaxation_countercases=sharp,cost_identity_cases=200))
if __name__=='__main__':run()
