"""Separate integer aggregate and Python family bootstrap."""
import random,collections as C,gzip,json,math
from fractions import Fraction
from k48_plan import D,T,load,sha,SEED
OLD={101,102,103,105}
def read(mode):
    lk=load(mode+'_LOCK.json');p=C.defaultdict(lambda:C.defaultdict(lambda:[0]*10))
    for fi in lk['files']:
        f=D/fi['file'];assert sha(f)==fi['sha256']
        with gzip.open(f,'rt') as z:
            for g in map(json.loads,z):
                names=['Rapfi'] if mode.startswith('R') else ['all','old' if g['seed'] in OLD else 'new']
                for name in names:
                    a=p[name][g['job']['id']];a[0]+=g['score2'];a[1]+=1
                    for r in g['rows']:
                        k=2 if r['arm']=='Q' else 6
                        a[k]+=1;a[k+1]+=r['nn'];a[k+2]+=r['cpu_ns'];a[k+3]+=r['wall_ns']
    return p
def metric(a):
    return dict(score=float(Fraction(a[0],2*a[1])),NN_reduction=float(1-Fraction(a[3]*a[6],a[7]*a[2])),CPU_reduction=float(1-Fraction(a[4]*a[6],a[8]*a[2])),wall_reduction=float(1-Fraction(a[5]*a[6],a[9]*a[2])))
def sums(xs):return [sum(x[i] for x in xs) for i in range(10)]
def ci(a):
    a=sorted(a);return dict(CI=[a[199],a[7799]],lower95=a[399])
def main():
    s=load('SUMMARY.json');modes=list(s['arena']);r={m:read(m) for m in modes};rng=random.Random(SEED+917)
    point={m:{p:metric(sums(list(x.values()))) for p,x in y.items()} for m,y in r.items()}
    for m,y in point.items():
        for p,z in y.items():
            for k,v in z.items():assert abs(v-s['arena'][m][p]['metrics'][k])<1e-14
    bs={m:{p:C.defaultdict(list) for p in y} for m,y in r.items()}
    dif={m:{p:C.defaultdict(list) for p in y} for m,y in s['paired'].items()}
    ids=sorted(r['NN_FIXED']['all']);ri=sorted(r['R128']['Rapfi']);assert len(ids)==len(ri)==32
    for _ in range(8000):
        ix=rng.choices(ids,k=32);ir=rng.choices(ri,k=32);z={}
        for m,y in r.items():
            for p,x in y.items():
                a=metric(sums([x[j] for j in (ir if m.startswith('R') else ix)]));z[m,p]=a
                for k,v in a.items():bs[m][p][k].append(v)
        for m,y in dif.items():
            for p,x in y.items():
                b=s['paired'][m][p]['reference']
                for k,v in z[m,p].items():x[k].append(v-z[b,p][k])
    out=dict(replicates=8000,seed=SEED+917,arena={m:{p:{k:ci(v) for k,v in x.items()} for p,x in y.items()} for m,y in bs.items()},paired={m:{p:{k:ci(v) for k,v in x.items()} for p,x in y.items()} for m,y in dif.items()})
    # Independent repeat of gate decisions; report discrepancies rather than suppress.
    out['gate']={}
    for m,y in out['arena'].items():
        out['gate'][m]={p:dict(score=x['score']['lower95']>=.45,NN=x['NN_reduction']['lower95']>=.30,CPU=point[m][p]['CPU_reduction']>0) for p,x in y.items()}
    print('INDEPENDENT',json.dumps(out['gate']),flush=True)
    print('LOWERS',[(m,p,x['score']['lower95'],x['NN_reduction']['lower95']) for m,y in out['arena'].items() for p,x in y.items()],flush=True)
    (T/'STATS_INDEPENDENT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
