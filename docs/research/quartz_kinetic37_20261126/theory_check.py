from itertools import combinations
from fractions import Fraction as F
from random import Random
r=Random(261137)
for _ in range(1024):
    N=r.randrange(2,9);k=r.randrange(2,N+1);a=[F(r.randrange(9),8) for j in range(N)]
    m=sum(a)/N;S2=sum((x-m)**2 for x in a)/(N-1)
    est=[];var=[]
    for idx in combinations(range(N),k):
        x=[a[j] for j in idx];v=sum(x)/k;est.append(v)
        var.append((1-F(k,N))*sum((z-v)**2 for z in x)/(k-1)/k)
    assert sum(est)/len(est)==m
    assert sum((x-m)**2 for x in est)/len(est)==(1-F(k,N))*S2/k==sum(var)/len(var)
    rows=[(r.randrange(4),r.randrange(4)) for i in range(N)]
    lab=[[F(r.randrange(9),8) for j in range(4)] for i in range(N)]
    d=[lab[i][b]-lab[i][a] for i,(a,b) in enumerate(rows)]
    assert sum(d)==sum(x for (a,b),x in zip(rows,d) if a!=b)
print('1024 exhaustive SRSWOR subset variance/HT and structural-zero contrast checks')
