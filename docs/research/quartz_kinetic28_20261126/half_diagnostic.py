"""Post-lock mechanism diagnostic: one-stage static baselines vs frozen stopping. No policy refit."""
from trial import *
FROZEN=read('FROZEN.json');assert FROZEN['budgets']['64']['selected']=='stop'
outdir=BIN.parent
path=outdir/'HALF_DIAGNOSTIC.jsonl.gz';assert not path.exists()
ll=labelmap('TEST');rr=[r for r in read('COHORT.json')['roots'] if r['id'] in ll]
assert len(rr)==163
with Server() as srv,gzip.open(path,'wt',compresslevel=3) as fd:
    for r in rr:
        oo=[(a,j) for a in (0,2) for j in range(2)]
        random.Random(SEED+771001+r['id']).shuffle(oo);res={}
        for a,j in oo:
            x=req(srv,r,64,path=[a,STOP]);assert x['stopped']
            res[str(a)+':'+str(j)]=x
        emit(fd,dict(root=r,order=oo,results=res))
# Actual two repeats per root, avoid treating repeats as independent.
a0={}
for t in rows('TEST_LIVE',64):
    if t['root']['id'] in ll:a0[t['root']['id']]=t['results']
v={}
for t in raw(path):v[t['root']['id']]=t['results']
assert set(v)==set(a0)==set(ll)
ids=sorted(ll)
def get(k,i,j):
    return v[i][k+':'+str(j)] if k in ('0','2') else a0[i][k+str(j)]
def pair(k,m):
    d=[];nn=[];ne=[];w=[]
    for i in ids:
        u=ll[i];aa=[get(k,i,j) for j in range(2)];bb=[get(m,i,j) for j in range(2)]
        d.append(S.mean(float(util(a,u)-util(b,u)) for a,b in zip(aa,bb)))
        nn.append(S.mean(a['NN_calls']-b['NN_calls'] for a,b in zip(aa,bb)))
        ne.append(S.mean(float(net(a,u)-net(b,u)) for a,b in zip(aa,bb)))
        w.append(S.mean((a['wall_ns']-b['wall_ns'])/1e6 for a,b in zip(aa,bb)))
    def ci(x):
        rng=random.Random(SEED+917);bs=sorted(S.mean(rng.choices(x,k=len(x))) for _ in range(4000))
        return [bs[99],bs[3899]]
    return dict(n=len(d),gain=S.mean(d),gain_ci=ci(d),net_gain=S.mean(ne),net_ci=ci(ne),NN_delta=S.mean(nn),NN_ci=ci(nn),better=sum(x>0 for x in d),worse=sum(x<0 for x in d),wall_delta_descriptive=S.mean(w))
def mean(k):
    x=[(get(k,i,j),ll[i]) for i in ids for j in range(2)]
    return dict(n=len(ids),utility=S.mean(float(util(a,u)) for a,u in x),net=S.mean(float(net(a,u)) for a,u in x),NN=S.mean(a['NN_calls'] for a,u in x),regret=S.mean(float(max(u.values())-util(a,u)) for a,u in x))
out=dict(scope='post-freeze TEST diagnostic, actual one-stage serial/tree4 matched target vs frozen STOP, two repeats averaged per root; no selection or refit',sha256=sha(path),requests=len(ids)*4,NN=sum(a['NN_calls'] for z in v.values() for a in z.values()),visits=sum(a['visits'] for z in v.values() for a in z.values()),means={k:mean(k) for k in ('0','2','serial','tree4','stop','continue')},pairs={k+':'+m:pair(k,m) for k,m in [('stop','0'),('stop','2'),('stop','serial'),('0','serial'),('2','tree4'),('continue','serial')]})
(outdir/'HALF_DIAGNOSTIC.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
print(json.dumps(out,sort_keys=True),flush=True)
