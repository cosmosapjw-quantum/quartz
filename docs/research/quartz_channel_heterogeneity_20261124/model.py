"""Offline TRAIN-only exact board labels and finite moment lookup."""
import json, math, hashlib
from functools import lru_cache
from pathlib import Path
from population import outcome,apply,replay,D
from fractions import Fraction as Q
from collections import defaultdict
EPS=Q(1,32)
def hist(n):
    return [(i,j,n-i-j) for i in range(n+1) for j in range(n-i+1)]
CUES=hist(4); ADDS=sum([hist(n) for n in range(3)],[])
def plus(a,b):return tuple(x+y for x,y in zip(a,b))
def add(h,y):return tuple(x+(i==y) for i,x in enumerate(h))
@lru_cache(None)
def truth(b,p,root):
    o=outcome(b)
    if o is not None:
        v=(o*root+1)//1
        return v,1,tuple(Q(i==v) for i in range(3)),Q(1)
    children=[truth(apply(b,i,p),-p,root) for i in range(9) if b[i]==0]
    n=len(children)
    return ((max if p==root else min)(v[0] for v in children),1+sum(v[1] for v in children),
            tuple(sum(v[2][j] for v in children)/n for j in range(3)),1+sum(v[3] for v in children)/n)
def rows(source):
    b,p=replay(source['history'])
    for i in range(9):
        if b[i]:continue
        child=apply(b,i,p)
        v,n,rate,cost=truth(child,-p,p)
        yield dict(move=i,rem=child.count(0),terminal=outcome(child) is not None,u=v,solve_nodes=n,rate=rate,rollout_cost=cost)
def calc(atoms,cue,added,arm):
    # atoms (utility index, probability vector), equal weight; class moments.
    def mass(h):
        a=[Q(0)]*3
        for u,r in atoms:
            a[u]+=math.prod(p**n for p,n in zip(r,h))
        return a
    m=mass(cue)
    cm=[[mass(add(cue,y))[u]/m[u] for y in range(3)] for u in range(3)]
    if arm==2:
        n=plus(cue,added);v=mass(n)
        den=sum(v)
        pr=[sum(mass(add(n,y)))/den for y in range(3)] if sum(added)<2 else [0]*3
    else:
        if arm==0:
            mm=mass((0,0,0))
            cm=[[mass(add((0,0,0),y))[u]/mm[u] for y in range(3)] for u in range(3)]
        v=[m[u]*math.prod(cm[u][y]**added[y] for y in range(3)) for u in range(3)]
        den=sum(v)
        pr=[sum(v[u]*cm[u][y] for u in range(3))/den for y in range(3)] if sum(added)<2 else [0]*3
    return [float((v[1]/2+v[2])/den),*map(float,pr)]
def fit():
    sources=[json.loads(l) for l in (D/'sources.jsonl').read_text().splitlines()]
    by=defaultdict(list); costs=defaultdict(list); evidence=[]
    for s in sources:
        if s['split']!='TRAIN':continue
        for c in rows(s):
            if c['terminal']:continue
            r=tuple((1-EPS)*x+EPS/3 for x in c['rate'])
            by[c['rem']].append((c['u'],r))
            costs[c['rem']].append(c['rollout_cost'])
            evidence.append(dict(id=s['id'],move=c['move'],rem=c['rem'],u=c['u'],rate=list(map(str,c['rate'])),solve_nodes=c['solve_nodes'],rollout_cost=str(c['rollout_cost'])))
    assert evidence
    pool=sum(by.values(),[]); cp=sum(costs.values(),[])
    table=[]; cost=[]
    for rem in range(9):
        atoms=by.get(rem,pool)+[(u,(Q(1,3),)*3) for u in range(3)]
        # cache exact sufficient moments by distinct integer histograms (0..7).
        @lru_cache(None)
        def mass(h):
            acc=[Q(0)]*3
            for u,r in atoms:acc[u]+=math.prod(p**n for p,n in zip(r,h))
            return tuple(acc)
        base=mass((0,0,0))
        global_chan=[[mass(add((0,0,0),y))[u]/base[u] for y in range(3)] for u in range(3)]
        arms=[[],[],[]]
        for cue in CUES:
            m=mass(cue)
            local=[[mass(add(cue,y))[u]/m[u] for y in range(3)] for u in range(3)]
            for arm in range(3):
                rr=[]
                for h in ADDS:
                    if arm==2:
                        n=plus(cue,h);v=mass(n);den=sum(v)
                        pr=[sum(mass(add(n,y)))/den for y in range(3)] if sum(h)<2 else [0]*3
                    else:
                        chan=global_chan if arm==0 else local
                        v=[m[u]*math.prod(chan[u][y]**h[y] for y in range(3)) for u in range(3)];den=sum(v)
                        pr=[sum(v[u]*chan[u][y] for u in range(3))/den for y in range(3)] if sum(h)<2 else [0]*3
                    rr.append([float((v[1]/2+v[2])/den),*map(float,pr)])
                arms[arm].append(rr)
        table.append(arms); cc=costs.get(rem,cp);cost.append(float(sum(cc)/len(cc)))
    model={'schema':'quartz-channel-table-v1','costs':cost,'table':table}
    (D/'model.json').write_text(json.dumps(model,separators=(',',':'))+'\n')
    (D/'TRAIN_ATOMS.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in evidence))
    (D/'TRAIN_FIT.json').write_text(json.dumps({'train_roots':sum(s['split']=='TRAIN' for s in sources),'atoms':len(evidence),'atoms_by_remaining':{str(k):len(v) for k,v in by.items()},'fallback_remaining':[k for k in range(9) if k not in by],'epsilon':'1/32','model_sha256':hashlib.sha256((D/'model.json').read_bytes()).hexdigest()},indent=2)+'\n')
    print((D/'TRAIN_FIT.json').read_text())
if __name__=='__main__':fit()
