"""Supplementary diagnostic: never interpret signed-cost ratio with nonpositive resample denominator."""
import collections as C,json
import numpy as np
from k51_plan import D,T,load
from experiment51 import rows
def main():
    x=list(rows('NN'));rng=np.random.default_rng(load('PLAN.json')['bootstrap_seed']);out={}
    for pan,rr in [('all',x),('old',[r for r in x if r['control']['seed'] in (101,102,103,105)]),('new',[r for r in x if r['control']['seed'] in (41,42,43)])]:
        ix=rng.integers(0,48,(8000,48));out[pan]={}
        for arm in ('global','target'):
            groups=C.defaultdict(list)
            for r in rr:groups[r['pairs'][arm]['job']].append(r['pairs'][arm])
            keys=('delta2','local_nn','delta_Q','delta_total')
            a=np.array([[sum(r[k] for r in groups[j]) for k in keys] for j in sorted(groups)],dtype=float)
            z=a[ix].sum(axis=1);tot=a.sum(axis=0);out[pan][arm]={}
            for i,k in enumerate(keys[1:],1):
                bad=int((z[:,i]<=0).sum());d=dict(nonpositive_resamples=bad,zero_resamples=int((z[:,i]==0).sum()),point=float(tot[0]/(2*tot[i])) if tot[i]>0 else None,admit_ratio=bool(bad==0 and tot[i]>0))
                if d['admit_ratio']:
                    v=np.sort(z[:,0]/(2*z[:,i]));d['CI']=[float(v[199]),float(v[7799])]
                out[pan][arm][k]=d
    (T/'RATIO_DIAGNOSTIC.json').write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print('RATIO',json.dumps(out),flush=True)
if __name__=='__main__':main()
