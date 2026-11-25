import math,random,itertools
from fractions import Fraction as F
r=random.Random(221126)
U=list(itertools.product((0,.5,1),repeat=3))
def val(p):return max(sum(w*u[a] for w,u in zip(p,U)) for a in range(3))
def stats(J):
    p=[sum(a) for a in J];z=[sum(a[i] for a in J) for i in range(len(J[0]))]
    v=sum(val([a[i] for a in J]) for i in range(len(z)))
    mi=sum(x*math.log(x/(p[j]*z[i])) for j,a in enumerate(J) for i,x in enumerate(a) if x)
    return v,mi,val(p)
for _ in range(2048):
    a=[[r.randrange(6) for _ in range(6)] for u in U];s=sum(map(sum,a));J=[[x/s for x in row] for row in a]
    K=[[row[0]+row[1],row[2]+row[3],row[4]+row[5]] for row in J]
    v,i,b=stats(J);w,j,c=stats(K)
    assert -1e-12<=v-b<=math.sqrt(i/2)+1e-12
    assert -1e-12<=v-w<=math.sqrt(max(0,i-j)/2)+1e-12 and j<=i+1e-12
for _ in range(4096):
    e=[F(r.randrange(1,6)) for i in range(3)];e=[x/sum(e) for x in e]
    y=[F(r.randrange(-4,5),4) for i in range(3)];m=[F(r.randrange(-4,5),4) for i in range(3)]
    a,b=r.randrange(3),r.randrange(3);d=[int(i==a)-int(i==b) for i in range(3)]
    q=sum(x*z for x,z in zip(d,m));psi=[q+d[i]*(y[i]-m[i])/e[i] for i in range(3)]
    target=y[a]-y[b];assert sum(w*x for w,x in zip(e,psi))==target
    v=sum(w*(x-target)**2 for w,x in zip(e,psi));rr=[d[i]*(y[i]-m[i]) for i in range(3)]
    assert v==sum(x*x/w for x,w in zip(rr,e))-sum(rr)**2
    if a==b:assert psi==[0]*3
print('2048 finite compression/MI/EVSI checks; 4096 exact rational AIPW contrast expectation and variance cases; all assertions passed')
