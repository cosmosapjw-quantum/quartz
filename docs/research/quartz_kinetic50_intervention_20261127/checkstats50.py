"""Independent integer aggregate and stdlib family bootstrap."""
import collections as C,gzip,json,random,math
from fractions import Fraction as F
from k50_plan import D,T,load,sha,SEED
OLD={101,102,103,105}
def raw(mode):
    for f in load(mode+'_LOCK.json')['files']:
        p=D/f['file'];assert sha(p)==f['sha256']
        with gzip.open(p,'rt') as fd:yield from map(json.loads,fd)
def add(a,b):
    for i,x in enumerate(b):a[i]+=x
def sums(a):
    return [sum(x[i] for x in a) for i in range(len(a[0]))]
def game(g):
    a=[g['score2'],1,0,0,0,0,0,0,0,0]
    for r in g['rows']:
        k=2 if r['arm']=='Q' else 6;a[k]+=1;a[k+1]+=r['nn'];a[k+2]+=r['cpu_ns'];a[k+3]+=r['wall_ns']
    return a
def gm(a):
    return dict(score=float(F(a[0],2*a[1])),NN_reduction=float(1-F(a[3]*a[6],a[7]*a[2])),CPU_reduction=float(1-F(a[4]*a[6],a[8]*a[2])),wall_reduction=float(1-F(a[5]*a[6],a[9]*a[2])))
def pm(a):
    return dict(tau=float(F(a[0],2*a[1])),local_nn=float(F(a[2],a[1])),delta_Q=float(F(a[3],a[1])),delta_B=float(F(a[4],a[1])),delta_total=float(F(a[5],a[1])))
def ci(a):
    a=sorted(a);return dict(CI=[a[199],a[7799]],lower95=a[399])
def main():
    s=load('SUMMARY.json');rng=random.Random(SEED+867)
    nn={m:C.defaultdict(lambda:C.defaultdict(lambda:[0]*26)) for m in ('TRAIN','CAL','TEST','ALL')}
    r=C.defaultdict(lambda:C.defaultdict(lambda:[0]*10))
    for m in ('TRAIN','CAL','TEST'):
        for x in raw(m):
            c,t,z=x['control'],x['extend'],x['pair'];a,b=game(c),game(t);j=c['job']['id']
            v=[t['score2']-c['score2'],1,z['local_nn'],b[3]-a[3],b[7]-a[7],b[3]+b[7]-a[3]-a[7]]+a+b
            for mode in (m,'ALL'):
                for pan in ('all','old' if c['seed'] in OLD else 'new'):add(nn[mode][pan][j],v)
    for g in raw('R96'):
        for pan in ('all',g['job']['stratum']):add(r[pan][g['job']['id']],game(g))
    out=dict(nn={},rapfi={},replicates=8000,seed=SEED+867)
    for m,pan in nn.items():
        ids=sorted(pan['all']);b={p:{a:C.defaultdict(list) for a in ('effect','control','extend')} for p in pan}
        for p,x in pan.items():
            q=sums(list(x.values()));v=pm(q[:6]);a=gm(q[6:16]);bb=gm(q[16:26])
            for k,val in v.items():assert abs(val-s['paired'][m][p]['metrics'][k])<1e-14
            for arm,g in [('control',a),('extend',bb)]:
                for k,val in g.items():assert abs(val-s['nn'][m][p][arm]['metrics'][k])<1e-14
        for _ in range(8000):
            ix=rng.choices(ids,k=len(ids))
            for p,x in pan.items():
                q=sums([x[j] for j in ix])
                for arm,z in [('effect',pm(q[:6])),('control',gm(q[6:16])),('extend',gm(q[16:26]))]:
                    for k,val in z.items():b[p][arm][k].append(val)
        out['nn'][m]={p:{a:{k:ci(v) for k,v in z.items()} for a,z in x.items()} for p,x in b.items()}
    for p,x in r.items():
        ids=sorted(x);a=gm(sums(list(x.values())))
        for k,v in a.items():assert abs(v-s['rapfi'][p]['metrics'][k])<1e-14
        b=C.defaultdict(list)
        for _ in range(8000):
            z=gm(sums([x[j] for j in rng.choices(ids,k=len(ids))]))
            for k,v in z.items():b[k].append(v)
        out['rapfi'][p]={k:ci(v) for k,v in b.items()}
    print('INDEPENDENT ITT',out['nn']['ALL']['all']['effect'],flush=True)
    print('INDEPENDENT RAPFI',out['rapfi'],flush=True)
    (T/'STATS_INDEPENDENT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
