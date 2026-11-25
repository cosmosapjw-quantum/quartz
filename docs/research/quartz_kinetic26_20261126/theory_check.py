from fractions import Fraction as F
from random import Random
r=Random(261126)
# Independent interleavings with tokens, calls and refunds.
for _ in range(4096):
    B=r.randrange(1,17);free=B;spent=0;pending=[0]*r.randrange(1,9)
    for _ in range(100):
        w=r.randrange(len(pending));k=r.randrange(1,4);op=r.randrange(3)
        if op==0 and free>=k:free-=k;pending[w]+=k
        if op==1 and pending[w]>=k:pending[w]-=k;spent+=k
        if op==2 and pending[w]>=k:pending[w]-=k;free+=k
        assert free>=0 and spent<=B and free+sum(pending)+spent==B
    free+=sum(pending);assert free+spent==B
    p=[F(r.randrange(1,6)) for _ in range(4)];p=[x/sum(p) for x in p]
    q=[[F(r.randrange(-4,5),4) for j in range(6)] for i in range(4)]
    fb=sum(w*max(a) for w,a in zip(p,q));st=max(sum(w*a[j] for w,a in zip(p,q)) for j in range(6))
    assert fb>=st
    assert (fb==st)==any(all(a[j]==max(a) for a in q) for j in range(6))
print('4096 exact rational feedback equality and random token invariant histories, 409600 transitions')
