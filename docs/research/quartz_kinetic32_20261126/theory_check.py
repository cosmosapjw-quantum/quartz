from random import Random
from fractions import Fraction as F
r=Random(261132)
def close(children,term):
    v={}
    for i in reversed(range(len(children))):
        if i in term:v[i]=(term[i],term[i])
        elif children[i]:v[i]=(max(-v[j][1] for j in children[i]),max(-v[j][0] for j in children[i]))
        else:v[i]=(-1,1)
    return v
for _ in range(8192):
    n=r.randrange(4,17);truth={i:r.randrange(-1,2) for i in range(n//2,n)}
    aa=[sorted(r.sample(range(i+1,n),r.randrange(1,min(n-i-1,4)+1))) if i<n//2 else [] for i in range(n)]
    va=close(aa,truth)
    p={i:x for i,x in truth.items() if r.randrange(2)};q={i:x for i,x in truth.items() if r.randrange(2)}
    v0=close(aa,p);v1=close(aa,p|q)
    assert v1==close(aa,q|p)==close(aa,p|q|p)
    assert all(v0[i][0]<=v1[i][0]<=va[i][0]==va[i][1]<=v1[i][1]<=v0[i][1] for i in range(n))
    a=aa[0];lo=[-v1[j][1] for j in a];hi=[-v1[j][0] for j in a];u=[-va[j][0] for j in a]
    assert all(hi[i]>=max(lo) for i,x in enumerate(u) if x==max(u))
    for i,x in enumerate(u):
        b=max(0,max((hi[j] for j in range(len(a)) if j!=i),default=-1)-lo[i])
        assert F(max(u)-x,2)<=F(b,2)
        if b==0:assert x==max(u)
    T=r.randrange(1,65);P=r.randrange(T);S=T-P;B=E=0
    for j in range(128):
        if r.randrange(3)==0:S+=P;P=0
        elif r.randrange(2)==0 and S:
            S-=1
            if r.randrange(2):S+=1
            else:B+=1
        else:
            k=r.randrange(3)
            if k<=S+P:
                z=min(k,P);P-=z;S-=k-z;E+=k
        assert S>=0 and P>=0 and S+P+B+E==T
print('8192 DAG union/closure and witness checks; 1048576 exact integer budget transitions')
