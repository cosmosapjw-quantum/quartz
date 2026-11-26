"""Independent family-cluster bootstrap and matched-family contrast vs common baseline.
No Q-v-Q games. Two separate policies against identical B32 on same family/model/color.
"""
import collections as C,random,json,gzip,math
from fractions import Fraction as F
from k45_plan import D,SEED,load,save,sha
def famstats(mode):
    out=C.defaultdict(C.Counter);games={};lk=load(mode+'_LOCK.json')
    for f in lk['files']:
        p=D/f['file'];assert sha(p)==f['sha256']
        with gzip.open(p,'rt') as fd:
            for line in fd:
                g=json.loads(line);i=g['job']['id'];k=i,g['seed'],g['color'];assert k not in games
                games[k]=g;z=out[i];z['games']+=1;z['score2']+=g['score2']
                for r in g['rows']:
                    t='Q' if r['arm']=='Q' else 'B';z[t+'moves']+=1;z[t+'nn']+=r['nn'];z[t+'cpu']+=r['cpu_ns'];z[t+'wall']+=r['wall_ns']
    assert len(out)==48 and all(v['games']==(2 if mode=='R_ARENA' else 8) for v in out.values())
    return out,games
def measure(z):
    return dict(score=float(F(z['score2'],2*z['games'])),NN_reduction=float(1-F(z['Qnn']*z['Bmoves'],z['Bnn']*z['Qmoves'])),CPU_reduction=float(1-F(z['Qcpu']*z['Bmoves'],z['Bcpu']*z['Qmoves'])),wall_reduction=float(1-F(z['Qwall']*z['Bmoves'],z['Bwall']*z['Qmoves'])))
def sumz(xs):return sum(xs,C.Counter())
def quant(x,p):
    x=sorted(x);i=(len(x)-1)*p;j=int(i);return x[j]+(i-j)*(x[min(j+1,len(x)-1)]-x[j])
def main():
    modes=('NN_FIXED','NN_LOW','R_ARENA');ss={m:famstats(m) for m in modes};a,b=ss['NN_FIXED'][1],ss['NN_LOW'][1]
    assert set(a)==set(b) and all(a[k]['job']==b[k]['job'] for k in a)
    same=sum(a[k]['history']==b[k]['history'] for k in a);ds=C.Counter(b[k]['score2']-a[k]['score2'] for k in a)
    obs={m:measure(sumz(ss[m][0].values())) for m in modes};boot={m:C.defaultdict(list) for m in modes};diff=C.defaultdict(list)
    rng=random.Random(SEED+4591);N=8000
    ids=sorted(ss['NN_FIXED'][0]);assert ids==sorted(ss['NN_LOW'][0]);rids=sorted(ss['R_ARENA'][0])
    for _ in range(N):
        sample=rng.choices(ids,k=48);r_sample=rng.choices(rids,k=48);x={}
        for m in modes:
            z=measure(sumz(ss[m][0][i] for i in (r_sample if m=='R_ARENA' else sample)));x[m]=z
            for k,v in z.items():boot[m][k].append(v)
        for k in x['NN_FIXED']:diff[k].append(x['NN_LOW'][k]-x['NN_FIXED'][k])
    out=dict(replicates=N,seed=SEED+4591,identical_histories=same,paired_score2_difference={str(k):v for k,v in sorted(ds.items())},estimates=obs,independent_CI={m:{k:[quant(v,.025),quant(v,.975)] for k,v in b.items()} for m,b in boot.items()},paired_difference={k:dict(point=obs['NN_LOW'][k]-obs['NN_FIXED'][k],CI=[quant(v,.025),quant(v,.975)]) for k,v in diff.items()},scope='separate Q vs identical B games matched by opening family/model/color; independent Python random family bootstrap, not direct Q-v-Q match or proof of equivalence; CPU times from separate actual runs')
    save('PAIRED.json',out);print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
