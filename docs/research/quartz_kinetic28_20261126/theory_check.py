from random import Random
from fractions import Fraction as F
from math import exp,log
rng=Random(261128)
for B in range(2,65):
    for p in range(B):
        for s in range(B-p+1):
            for k in range(3):
                h=min(p,k)
                assert (s>=k-h)==(s+p>=k)
                if k<=p:assert s>=k-h
                if s>=k-h:
                    assert s-k+h>=0 and p-h>=0 and s-k+h+p-h+k==s+p
for _ in range(8192):
    B=rng.randrange(2,65);r=rng.randrange(B);s,p=B-r,r;R=[0]*8;C=0
    for _ in range(64):
        i=rng.randrange(8);k=rng.randrange(3);a=rng.randrange(4)
        if a==0 and s:s-=1;R[i]+=1
        elif a==1 and s+p>=k:h=min(p,k);p-=h;s-=k-h;R[i]+=k
        elif a==2 and R[i]>=k:R[i]-=k;C+=k
        elif a==3 and not any(R):s+=p;p=0
        assert min(s,p,C)>=0 and s+p+sum(R)+C==B
    C+=sum(R);assert s+p+C==B
    x=[rng.random() for _ in range(5)];y=[rng.random() for _ in x];t=.25
    def phi(v):return t*log(sum(exp(a/t) for a in v)/len(v))
    def q(v):
        w=[exp(a/t) for a in v];return [a/sum(w) for a in w]
    u,v=q(x),q(y);g=t*sum(a*log(a/b) for a,b in zip(u,v))
    assert abs(phi(y)-phi(x)-sum(a*(b-c) for a,b,c in zip(u,y,x))-g)<1e-12 and g>=-1e-12
print('Exhaustive B=2..64 feasible pairs and k=0..2; 524288 interleavings; 8192 Gibbs decompositions')
