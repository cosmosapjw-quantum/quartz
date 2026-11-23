#!/usr/bin/env python3
"""C29 independent finite-world unnormalized oracle and normalized factor filter.
Fraction-only; no imports from the Rust or C28 planner. All worlds have fixed U.
"""
from __future__ import annotations
from fractions import Fraction as F
from itertools import product
from dataclasses import dataclass
import hashlib,json,random,time
from pathlib import Path
SUPPORT=(F(0),F(1,2),F(1))
ZS=(-1,1)

def frac(x):
    return x if isinstance(x,F) else F(str(x))

def model(probs,loads,channels,costs,b=F(0),m=F(0),known_only=False):
    p=tuple(tuple(map(frac,pi)) for pi in probs)
    r=tuple(map(frac,loads))
    ch=tuple(tuple(tuple(map(frac,row)) for row in channel) for channel in channels)
    n=len(p)
    assert len(r)==len(ch)==len(costs)==n and n<=6
    assert -1<=frac(m)<=1 and 0<=frac(b)<=1
    for i in range(n):
        assert len(p[i])==3 and sum(p[i])==1 and all(v>=0 for v in p[i])
        assert abs(r[i])<1 and len(ch[i])==3
        k=len(ch[i][0])
        assert k>0 and all(len(row)==k and sum(row)==1 and all(v>=0 for v in row) for row in ch[i])
        if known_only:
            assert k==3 and all(row[y]==int(u==y) for u,row in enumerate(ch[i]) for y in range(3))
    assert all(frac(c)>=0 for c in costs)
    return {'p':p,'r':r,'ch':ch,'c':tuple(map(frac,costs)),'b':frac(b),'m':frac(m),'known_only':known_only}

def components(spec):
    q=((1-spec['m'])/2,(1+spec['m'])/2)
    locals_=[]
    for pi,ri in zip(spec['p'],spec['r']):
        mu=sum(u*a for u,a in zip(SUPPORT,pi))
        s=max((abs(u-mu) for u,a in zip(SUPPORT,pi) if a>0),default=F(0))
        features=[(u-mu)/s if s and a else F(0) for u,a in zip(SUPPORT,pi)]
        locals_.append(tuple(tuple(a*(1+z*ri*f) for a,f in zip(pi,features)) for z in ZS))
    return tuple(q),tuple(locals_)

@dataclass(frozen=True)
class Factor:
    q: tuple
    loc: tuple
    known: tuple=()
    def mass(self):
        return sum(self.q)
    def posterior_means(self):
        return tuple(sum(self.q[z]*sum(SUPPORT[u]*self.loc[i][z][u] for u in range(3)) for z in range(2))/self.mass() for i in range(len(self.loc)))
    def update(self,spec,i,y):
        likelihood=tuple(sum(self.loc[i][z][u]*spec['ch'][i][u][y] for u in range(3)) for z in range(2))
        d=sum(self.q[z]*likelihood[z] for z in range(2))
        if d==0:return None,F(0)
        nextq=tuple(self.q[z]*likelihood[z]/d for z in range(2))
        nextloc=list(self.loc);local_new=[]
        for z in range(2):
            if likelihood[z]:
                local_new.append(tuple(self.loc[i][z][u]*spec['ch'][i][u][y]/likelihood[z] for u in range(3)))
            else:local_new.append(self.loc[i][z])
        nextloc[i]=tuple(local_new)
        known=self.known
        if spec['known_only']:
            assert i not in [j for j,u in known]
            known=tuple(sorted(known+((i,SUPPORT[y]),)))
        return Factor(nextq,tuple(nextloc),known),d

def terminal(spec,state):
    if spec['known_only']:
        return max((spec['b'],)+(tuple(u for i,u in state.known)))
    return max((spec['b'],)+state.posterior_means())

def factor_value(spec,state,h):
    t=terminal(spec,state)
    best=t;choice=None
    if h:
        for i,ci in enumerate(spec['c']):
            if spec['known_only'] and i in [j for j,u in state.known]:continue
            val=-ci
            for y in range(len(spec['ch'][i][0])):
                nxt,p=state.update(spec,i,y)
                if p:val+=p*factor_value(spec,nxt,h-1)[0]
            if val>best:
                best,choice=val,i
    return best,choice

def world_prior(spec):
    q,loc=components(spec)
    worlds={}
    for indices in product(range(3),repeat=len(spec['p'])):
        w=sum(q[z]*__import__('math').prod(loc[i][z][u] for i,u in enumerate(indices)) for z in range(2))
        if w:worlds[indices]=w
    return worlds

def world_value(spec,worlds,h,known=()):
    z=sum(worlds.values())
    if not z:return F(0),None
    if spec['known_only']:t=max((spec['b'],)+tuple(u for i,u in known))*z
    else:t=max([spec['b']*z]+[sum(w*SUPPORT[state[i]] for state,w in worlds.items()) for i in range(len(spec['p']))])
    best=t;choice=None
    if h:
        for i,ci in enumerate(spec['c']):
            if spec['known_only'] and i in [j for j,u in known]:continue
            val=-ci*z
            for y in range(len(spec['ch'][i][0])):
                wy={s:w*spec['ch'][i][s[i]][y] for s,w in worlds.items() if w*spec['ch'][i][s[i]][y]>0}
                nextknown=tuple(sorted(known+((i,SUPPORT[y]),))) if spec['known_only'] and wy else known
                val+=world_value(spec,wy,h-1,nextknown)[0]
            if val>best:
                best,choice=val,i
    return best,choice

def compare(spec,h):
    fact=Factor(*components(spec));v,choice=factor_value(spec,fact,h)
    world=world_prior(spec);gv,oracle_choice=world_value(spec,world,h)
    assert sum(world.values())==1
    assert v==gv and choice==oracle_choice, (spec,h,v,gv,choice,oracle_choice)
    return v,choice

def check(seed=2907,count=80):
    started=time.perf_counter();rng=random.Random(seed);checks=0
    acc=F(3,4);bsc=tuple(((acc,1-acc),(F(1,2),F(1,2)),(1-acc,acc))) # for {0,.5,1} BSC
    independent=model([(F(1,2),F(0),F(1,2))]*2,[0,0],[bsc,bsc],[F(1,20)]*2,b=0)
    assert compare(independent,1)==(F(23,40),0);checks+=1
    s=Factor(*components(independent));ns,_=s.update(independent,0,1)
    assert s.q==ns.q==(F(1,2),F(1,2))
    assert terminal(independent,s)==F(1,2) and terminal(independent,ns)==F(3,4);checks+=1
    one=model([(F(1,2),F(0),F(1,2))],[0],[bsc],[F(1,100)],b=F(3,5))
    assert compare(one,1)==(F(133,200),0)
    assert compare(one,2)==(F(543,800),0);checks+=2
    after,_=Factor(*components(one)).update(one,0,1)
    assert factor_value(one,after,1)==(F(311,400),0);checks+=1
    noinfo=tuple((F(1,2),F(1,2)) for _ in SUPPORT)
    unin=model([(F(1,2),F(0),F(1,2))],[0],(noinfo,),(F(1,100),),b=F(3,5))
    assert compare(unin,2)==(F(3,5),None);checks+=1
    exact=tuple(tuple(F(int(i==j)) for j in range(3)) for i in range(3))
    comp=model([(F(1,3),F(1,3),F(1,3))]*3,[F(1,2),F(0),F(-1,2)],[exact]*3,[F(1,100)]*3,b=F(1,2),known_only=True)
    compare(comp,2);checks+=1
    cases=[]
    for k in range(count):
        n=rng.randint(1,4)
        pp=[]
        for i in range(n):
            a=[rng.randint(0,4) for _ in range(3)]
            if not any(a):a[0]=1
            pp.append(tuple(F(x,sum(a)) for x in a))
        ll=[rng.choice([-1,0,1])*rng.choice([F(0),F(1,4),F(3,4)]) for _ in range(n)]
        cc=[]
        for i in range(n):
            sensor=[]
            for u in range(3):
                row=[rng.randint(0,4) for _ in range(rng.choice([2,3]))] if False else []
            length=rng.choice([2,3])
            for u in range(3):
                a=[rng.randint(0,5) for _ in range(length)]
                if not any(a):a[0]=1
                sensor.append(tuple(F(x,sum(a)) for x in a))
            cc.append(tuple(sensor))
        spec=model(pp,ll,cc,[rng.choice([F(0),F(1,100),F(1,20),F(1,5)]) for _ in range(n)],b=rng.choice([F(0),F(1,2),F(3,5),F(1)]),m=rng.choice([F(-1),F(-1,2),F(0),F(1,2),F(1)]))
        for h in (0,1,2):
            compare(spec,h);checks+=1
        cases.append({'n':n,'horizon':2,'loading':list(map(str,ll))})
    return {'status':'PASS_EXACT_RATIONAL','seed':seed,'random_models':count,'assertion_groups':checks,'nonmyopic_h1':'133/200','nonmyopic_h2':'543/800','nonmyopic_gain':'11/800','factor_world_residual':'0 exact Fraction','duration_seconds':time.perf_counter()-started,'cases':cases}

if __name__=='__main__':
    import sys
    result=check()
    if len(sys.argv)==2:
        path=Path(sys.argv[1]);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
