"""Independent family-cluster resampling: Python integer aggregates and Fraction ratios."""
import random,collections as C,gzip,json,pathlib
from fractions import Fraction
from k47_plan import D,T,load,sha
OLD={101,102,103,105};NEW={41,42,43}
def read(mode):
    lk=load(mode+'_LOCK.json')
    panels=C.defaultdict(lambda:C.defaultdict(lambda:[0]*10))
    for fi in lk['files']:
        p=D/fi['file'];assert sha(p)==fi['sha256']
        with gzip.open(p,'rt') as f:
            for g in map(json.loads,f):
                pan='Rapfi' if mode=='R_ARENA' else 'old' if g['seed'] in OLD else 'new'
                z=panels[pan][g['job']['id']];z[0]+=g['score2'];z[1]+=1
                for r in g['rows']:
                    k=2 if r['arm']=='Q' else 6
                    z[k]+=1;z[k+1]+=r['nn'];z[k+2]+=r['cpu_ns'];z[k+3]+=r['wall_ns']
    if mode!='R_ARENA':panels['all']={i:[a+b for a,b in zip(panels['old'][i],panels['new'][i])] for i in panels['old']}
    return panels
def sumv(ar):return [sum(x[i] for x in ar) for i in range(10)]
def metr(x):
    return dict(score=float(Fraction(x[0],2*x[1])),NN_reduction=float(1-Fraction(x[3]*x[6],x[7]*x[2])),CPU_reduction=float(1-Fraction(x[4]*x[6],x[8]*x[2])),wall_reduction=float(1-Fraction(x[5]*x[6],x[9]*x[2])))
def ci(a):
    a=sorted(a);return dict(CI=[a[199],a[7799]],lower95=a[399])
def run():
    modes=('NN_LOW','R_ARENA');r={m:read(m) for m in modes};s=load('SUMMARY.json');points={m:{p:metr(sumv(list(z.values()))) for p,z in x.items()} for m,x in r.items()}
    for m,z in points.items():
        for p,x in z.items():
            for k,y in x.items():assert abs(y-s['arena'][m][p]['metrics'][k])<1e-14
    rng=random.Random(202611274700+911);b={m:{p:C.defaultdict(list) for p in z} for m,z in r.items()};diff={p:C.defaultdict(list) for p in ('old','new','all')}
    ids=sorted(r['NN_LOW']['all']);ri=sorted(r['R_ARENA']['Rapfi']);assert len(ids)==48 and len(ri)==64
    for i in range(8000):
        ix=rng.choices(ids,k=48);ir=rng.choices(ri,k=64);z={}
        for m,pan in r.items():
            for p,x in pan.items():
                q=metr(sumv([x[j] for j in (ir if m=='R_ARENA' else ix)]));z[m,p]=q
                for k,v in q.items():b[m][p][k].append(v)
    out=dict(replicates=8000,seed=202611275611,scope='Independent Python family resampling and exact aggregate ratios; same family shared across panels/policies, not independent individual games',arena={m:{p:{k:ci(v) for k,v in x.items()} for p,x in pan.items()} for m,pan in b.items()})
    for p in ('old','new'):
        x=out['arena']['NN_LOW'][p];assert x['score']['lower95']>=.45 and x['NN_reduction']['lower95']>=.30 and points['NN_LOW'][p]['CPU_reduction']>0
    f=T/'STATS_INDEPENDENT.json';assert not f.exists();f.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print('INDEPENDENT_BOOTSTRAP',json.dumps({p:out['arena']['NN_LOW'][p] for p in ('old','new','all')}),flush=True)
if __name__=='__main__':run()
