"""Descriptive horizon effects under equal-weight central/whole generators, independent cell family sampling."""
import collections as C,json,numpy as np
from k58_plan import D,T,load,sha
from experiment58 import rows
ARMS=('control','stop','stable')
def total(g):
    a=np.zeros(6,dtype=np.int64);a[0]=g['score2'];a[1]=1
    for r in g['rows']:
        k=2 if r['arm']=='Q' else 4;a[k]+=1;a[k+1]+=r['actual_nn']
    return a
def met(x):
    return x[...,0]/(2*x[...,1]),1-(x[...,3]/x[...,2])/(x[...,5]/x[...,4])
def report(a,b):
    p0,s0=met(a);p1,s1=met(b);return p1-p0,s1-s0
def main():
    g=C.defaultdict(dict)
    for z in rows('NN'):
        j=z['control']['job'];k=(j['horizon'],j['geometry']);f=j['id']
        if f not in g[k]:g[k][f]=np.zeros((3,6),dtype=np.int64)
        g[k][f]+=np.array([total(z[a]) for a in ARMS])
    assert set(g)=={(h,k) for h in (4,8,12,16) for k in ('central','whole')}
    rng=np.random.default_rng(load('PLAN.json')['bootstrap_seed']+58);agg={};boot={};diag={}
    for k,v in g.items():
        a=np.array([v[i] for i in sorted(v)]);n=len(a);assert n in (8,32) and np.all(a[:,0,1]==14)
        agg[k]=a.mean(axis=0);boot[k]=a[rng.integers(0,n,(8000,n))].mean(axis=1)
        diag[str(k)]={'families':n,'effects':{a0:{'score':float(report(agg[k][0],agg[k][i])[0]),'saving':float(report(agg[k][0],agg[k][i])[1])} for i,a0 in enumerate(ARMS) if i}}
    out={}
    for h in (4,8,12,16,'all'):
        ks=[k for k in sorted(g) if h=='all' or k[0]==h];a=np.mean([agg[k] for k in ks],axis=0);b=np.mean([boot[k] for k in ks],axis=0)
        o={}
        for i,arm in enumerate(ARMS):
            if not i:continue
            p,q=report(a[0],a[i]);pp,qq=report(b[:,0],b[:,i]);assert np.isfinite(pp).all() and np.isfinite(qq).all()
            o[arm]=dict(delta_score=float(p),delta_saving=float(q),score_CI=[float(np.sort(pp)[199]),float(np.sort(pp)[7799])],saving_CI=[float(np.sort(qq)[199]),float(np.sort(qq)[7799])])
        out[str(h)]=o
    obj=dict(cells=diag,standardized=out,scope='descriptive equal weight per generator and assigned horizon, 8000 family resamples within each cell; not primary NI, no per-cell superiority claim, whole generator overlaps central')
    check=json.loads((T/'STREAM_CHECK.json').read_text());lk=load('NN_LOCK.json')
    assert check['counts']==lk['counts'] and check['sha256']=={x['file']:x['sha256'] for x in lk['files']}
    obj['stream_check']=dict(counts=check['counts'],hashes=check['sha256'],panels=check['panels'],source_sha256=sha(T/'stream58.py'),scope=check['scope'])
    (T/'STANDARDIZED.json').write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');print(json.dumps(obj),flush=True)
if __name__=='__main__':main()
