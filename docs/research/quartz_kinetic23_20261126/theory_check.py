from fractions import Fraction as F
from random import Random
rng=Random(2026112623)
for _ in range(4096):
    y=[F(rng.randrange(-4,5),4) for _ in range(9)]
    e=[F(rng.randrange(1,6)) for _ in range(3)];e=[x/sum(e) for x in e]
    f=[[F(rng.randrange(1,6)) for _ in range(3)] for _ in range(3)];f=[[x/sum(z) for x in z] for z in f]
    g=[e[a]*f[a][b] for a in range(3) for b in range(3)]
    p,q=rng.randrange(9),rng.randrange(9);d=[int(k==p)-int(k==q) for k in range(9)]
    delta=y[p]-y[q];w=[x*x/z for x,z in zip(d,g)]
    def ips(m):return [d[k]*(y[k]-m)/g[k] for k in range(9)]
    def var(x):return sum(z*(v-delta)**2 for z,v in zip(g,x))
    m=F(rng.randrange(-4,5),4);best=sum(x*z for x,z in zip(w,y))/sum(w) if sum(w) else F(0)
    assert sum(z*x for z,x in zip(g,ips(m)))==delta
    assert var(ips(m))-var(ips(best))==sum(w)*(m-best)**2
    m1=[F(rng.randrange(-4,5),4) for _ in range(3)];m2=[F(rng.randrange(-4,5),4) for _ in range(9)]
    def dr(k):
        a,b=divmod(k,3);vals=[]
        for i in range(3):
            for j in range(3):
                r1=F(int(i==a))/e[i];r2=F(int(j==b))/f[i][j]
                vals.append(m1[a]+r1*(m2[3*i+b]-m1[i])+r1*r2*(y[3*i+j]-m2[3*i+j]))
        return vals
    v=[x-z for x,z in zip(dr(p),dr(q))]
    assert sum(z*x for z,x in zip(g,v))==delta
    means=[sum(f[a][b]*v[3*a+b] for b in range(3)) for a in range(3)]
    assert var(v)==sum(e[a]*((means[a]-delta)**2+sum(f[a][b]*(v[3*a+b]-means[a])**2 for b in range(3))) for a in range(3))
print('4096 exact rational nonuniform sequential assignment cases: unbiasedness, optimum/excess variance, DR, conditional variance identity')
