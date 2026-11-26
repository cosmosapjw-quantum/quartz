from itertools import product
from fractions import Fraction as F
def leader(n):return max(range(len(n)),key=lambda i:(n[i],-i))
def d(n):
    a=leader(n)
    return min((n[a]-x+int(i>a) for i,x in enumerate(n) if i!=a),default=10**9)
def stop(seq,H):
    n=[0]*3
    for t in range(H+1):
        if d(n)>H-t or t==H:return t,leader(n)
        n[seq[t]]+=1
def run():
    paths=0
    for seq in product(range(3),repeat=8):
        lo,a=stop(seq,4);hi,b=stop(seq,8)
        assert lo<=hi and a==leader([seq[:4].count(i) for i in range(3)]) and b==leader([seq.count(i) for i in range(3)])
        paths+=1
    tables=0
    for z in product(range(-2,3),repeat=4):
        scores=[sum(z[i] for i in range(4) if m>>i&1) for m in range(16)]
        m=sum(1<<i for i,g in enumerate(z) if g>0)
        assert scores[m]==max(scores) and scores[m]>=max(0,sum(z))
        tables+=1
    mixes=0
    for v in product((-1,0,1),repeat=5):
        flips=[x for x in v if x]
        mu=F(sum(v),len(v))
        assert mu==(F(len(flips),len(v))*F(sum(flips),len(flips)) if flips else 0)
        assert abs(mu)<=F(len(flips),len(v))
        mixes+=1
    print(dict(prefix_paths=paths,tables=tables,signed_mixtures=mixes))
if __name__=='__main__':run()
