from fractions import Fraction as F
from itertools import product
from random import Random
from math import log

U=(F(0),F(1,2),F(1))
RNG=Random(20261123)

def factors(ps):
    out=[]
    for p in ps:
        mu=sum(q*u for q,u in zip(p,U))
        s=max(abs(u-mu) for q,u in zip(p,U) if q)
        out.append(tuple((u-mu)/s if q and s else F(0) for q,u in zip(p,U)))
    return out

def worlds(ps,rs,m):
    fs=factors(ps)
    out=[]
    for ids in product(*(tuple(k for k,p in enumerate(row) if p) for row in ps)):
        a=(1+m)/2; z=(1-m)/2
        for i,k in enumerate(ids):
            a*=ps[i][k]*(1+rs[i]*fs[i][k])
            z*=ps[i][k]*(1-rs[i]*fs[i][k])
        out.append((tuple(U[k] for k in ids),a+z))
    assert sum(w for _,w in out)==1
    return out

def raw_scores(ps,rs,ds,m,b):
    ww=worlds(ps,rs,m); n=len(ps); out=[]
    for i in range(n):
        v=-ds[i]+sum(w*max(F(0),uu[i]-b) for uu,w in ww)
        for u in U:
            branch=[(uu,w) for uu,w in ww if uu[i]==u]
            if not branch: continue
            bu=max(b,u)
            choices=[F(0)]
            for j in range(n):
                if j!=i:
                    choices.append(sum(w*(max(F(0),uu[j]-bu)-ds[j]) for uu,w in branch))
            v+=max(choices)
        out.append(v)
    return out

def lifted_scores(ps,rs,ds,m,b):
    fs=factors(ps)
    def ab(i,bx):
        return (sum(q*max(F(0),u-bx) for q,u in zip(ps[i],U))-ds[i],
                rs[i]*sum(q*f*max(F(0),u-bx) for q,f,u in zip(ps[i],fs[i],U)))
    out=[]
    for i in range(len(ps)):
        a,z=ab(i,b); value=a+m*z
        for k,u in enumerate(U):
            if not ps[i][k]: continue
            t=rs[i]*fs[i][k]; opts=[F(0)]
            for j in range(len(ps)):
                if i!=j:
                    aa,bb=ab(j,max(b,u))
                    opts.append((1+m*t)*aa+(m+t)*bb)
            value+=ps[i][k]*max(opts)
        out.append(value)
    return out

def selected(scores):
    vals=[F(0)]+scores
    return vals.index(max(vals))-1

def reference(sigma,r,c,d,m=F(0)):
    return ([(F(1,2),F(1,2),F(0)),(F(1,2),F(0),F(1,2)),(F(1,2),F(0),F(1,2))],
            [r,r,sigma*r],[c,d,d],m,F(1,2))

checked=0
for z in range(120):
    n=z%5; ps=[]
    for i in range(n):
        weights=[RNG.randrange(5) for _ in U]
        if not any(weights): weights[0]=1
        if z%10==1: weights=[1,0,0]
        den=sum(weights); ps.append(tuple(F(w,den) for w in weights))
    rs=[RNG.choice([F(-4,5),F(-1,2),F(0),F(1,2),F(4,5)]) for _ in range(n)]
    ds=[RNG.choice([F(0),F(1,100),F(1,10),F(3,10),F(1,2),F(1)]) for _ in range(n)]
    m=RNG.choice([F(-1),F(-4,5),F(0),F(1,3),F(1)])
    b=RNG.choice([F(0),F(1,4),F(1,2),F(2,3),F(1)])
    assert raw_scores(ps,rs,ds,m,b)==lifted_scores(ps,rs,ds,m,b)
    ww=worlds(ps,rs,m); fs=factors(ps)
    for i in range(n):
        for j in range(i+1,n):
            ei=sum(w*uu[i] for uu,w in ww); ej=sum(w*uu[j] for uu,w in ww)
            cov=sum(w*uu[i]*uu[j] for uu,w in ww)-ei*ej
            di=sum(q*u*f for q,u,f in zip(ps[i],U,fs[i]))
            dj=sum(q*u*f for q,u,f in zip(ps[j],U,fs[j]))
            assert cov==rs[i]*rs[j]*di*dj*(1-m*m)
    if n>=2:
        for a,k in product(range(3),repeat=2):
            if not ps[0][a] or not ps[1][k]: continue
            t=rs[0]*fs[0][a]; v=rs[1]*fs[1][k]
            post=lambda mm,tt:(mm+tt)/(1+mm*tt)
            assert post(post(m,t),v)==post(post(m,v),t)
            lp=(1+m)*(1+t)*(1+v); lm=(1-m)*(1-t)*(1-v)
            assert post(post(m,t),v)==(lp-lm)/(lp+lm)
    checked+=1
print('RANDOM_EXACT_MODELS',checked)

phase=0
for r in [F(0),F(1,4),F(1,2),F(4,5),F(99,100)]:
    y=r*r/4
    for x in sorted({F(-1,2),-y,-y/3,F(0),y,F(1,4),F(-1,20)}):
        if x>F(1,4): continue
        common_boundary=(y-abs(x))/2
        contrast_boundary=min(x+y,(y-x)/2)
        costs={F(0),F(1,100),F(2,25),F(1,2)}
        for edge in [common_boundary,contrast_boundary]:
            if edge>=0: costs.add(edge)
        for c in costs:
            for sigma in [1,-1]:
                params=reference(sigma,r,c,F(1,4)-x)
                raw=raw_scores(*params)
                if sigma==1:
                    expected=[-c+(max(x+y,F(0))+max(x-y,F(0)))/2,x+max(x-y,F(0))/2]
                    unique=(2*c+abs(x)<y)
                    assert (expected[1]>0)==(x>0)
                else:
                    expected=[-c+max(x+y,F(0)),x+max(x+y,F(0))/2]
                    unique=(c<min(x+y,(y-x)/2))
                    assert (expected[1]>0)==(3*x+y>0)
                assert raw==[expected[0],expected[1],expected[1]]
                assert unique==(raw[0]>max(F(0),raw[1],raw[2]))
                phase+=1
print('EXACT_COST_PARTITION_CASES',phase)

beliefs=sorted({F(k,96) for k in range(-96,97)}|{F(11,16),F(-11,16),F(7,8),F(-7,8),F(19,24),F(-19,24)})
for m in beliefs:
    vals=raw_scores(*reference(-1,F(4,5),F(1,100),F(3,10),m))
    z=abs(m)
    a=F(1,10) if z<=F(11,16) else (F(9,200)+F(2,25)*z if z<=F(7,8) else -F(3,50)+z/5)
    target=F(1,200)+F(3,25)*z if z<=F(11,16) else -F(1,20)+z/5
    assert vals[0]==a and max(vals[1:])==target
    expected_action=2 if m<F(-19,24) else (1 if m>F(19,24) else 0)
    assert selected(vals)==expected_action
print('EXACT_BELIEF_STATES',len(beliefs))
for m in [F(-1),F(-19,24),F(0),F(19,24),F(1)]:
    print('REFERENCE_SCORES',str(m),[str(v) for v in raw_scores(*reference(-1,F(4,5),F(1,100),F(3,10),m))])

for sigma in [1,-1]:
    ps,rs,ds,m,b=reference(sigma,F(4,5),F(1,100),F(3,10))
    ww=worlds(ps,rs,m)
    pa={}; pbc={}
    for uu,w in ww:
        pa[uu[0]]=pa.get(uu[0],F(0))+w
        pbc[uu[1:]]=pbc.get(uu[1:],F(0))+w
    mi=sum(float(w)*log(float(w/(pa[uu[0]]*pbc[uu[1:]]))) for uu,w in ww if w)
    print('MUTUAL_INFORMATION_NATS',sigma,format(mi,'.15f'))
    if sigma==1: common=dict(ww)
    else:
        for (a,bb,c),w in ww:
            assert w==common[(a,bb,1-c)]

ps=[(F(1,2),F(0),F(1,2))]*3; rs=[F(1,2)]*3
ww=worlds(ps,rs,F(1))
covs=[]
for i,j in [(0,1),(0,2),(1,2)]:
    covs.append(sum(w*uu[i]*uu[j] for uu,w in ww)-sum(w*uu[i] for uu,w in ww)*sum(w*uu[j] for uu,w in ww))
print('ENDPOINT_COVARIANCE_SIGN_COUNTEREXAMPLE',list(map(str,covs)))
assert covs==[0,0,0]
print('ALL_ASSERTIONS_PASSED')
