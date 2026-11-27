"""Independent stdlib integer aggregate and family bootstrap."""
import collections as C,gzip,json,random,math
from fractions import Fraction as F
from k51_plan import D,T,load,sha,SEED
OLD={101,102,103,105}
def raw(mode):
    for f in load(mode+'_LOCK.json')['files']:
        p=D/f['file'];assert sha(p)==f['sha256']
        with gzip.open(p,'rt') as fd:yield from map(json.loads,fd)
def archived():
    d=D.parent/'quartz_kinetic50_intervention_20261127';p=d/'R96_LOCK.json';assert sha(p)==load('PLAN.json')['historical_rapfi_lock_sha256']
    for f in json.loads(p.read_text())['files']:
        q=d/f['file'];assert sha(q)==f['sha256']
        with gzip.open(q,'rt') as fd:yield from map(json.loads,fd)
def add(a,b):
    for i,x in enumerate(b):a[i]+=x
def sums(a):return [sum(x[i] for x in a) for i in range(len(a[0]))]
def game(g):
    a=[g['score2'],1,0,0,0,0,0,0,0,0]
    for r in g['rows']:
        k=2 if r['arm']=='Q' else 6;a[k]+=1;a[k+1]+=r['nn'];a[k+2]+=r['cpu_ns'];a[k+3]+=r['wall_ns']
    return a
def gm(a):
    return dict(score=float(F(a[0],2*a[1])),NN_reduction=float(1-F(a[3]*a[6],a[7]*a[2])),CPU_reduction=float(1-F(a[4]*a[6],a[8]*a[2])))
def pm(a):
    return dict(tau=float(F(a[0],2*a[1])),local_nn=float(F(a[2],a[1])),delta_Q=float(F(a[3],a[1])),delta_B=float(F(a[4],a[1])),delta_total=float(F(a[5],a[1])))
def ci(a):
    a=sorted(a);return dict(CI=[a[199],a[7799]],lower95=a[399])
def main():
    s=load('SUMMARY.json');rng=random.Random(SEED+901)
    nn=C.defaultdict(lambda:C.defaultdict(lambda:[0]*42))
    for x in raw('NN'):
        c,g,t=x['control'],x['global'],x['target'];a,b,d=game(c),game(g),game(t);j=c['job']['id']
        q=[]
        for arm,bb in [('global',b),('target',d)]:
            z=x['pairs'][arm];q.extend([bb[0]-a[0],1,z['local_nn'],bb[3]-a[3],bb[7]-a[7],bb[3]+bb[7]-a[3]-a[7]])
        for pan in ('all','old' if c['seed'] in OLD else 'new'):add(nn[pan][j],q+a+b+d)
    out=dict(nn={},rapfi={},replicates=8000,seed=SEED+901)
    ids=sorted(nn['all']);assert len(ids)==48
    for p,x in nn.items():
        a=sums(list(x.values()))
        for arm,j in [('global',0),('target',6)]:
            for k,v in pm(a[j:j+6]).items():assert abs(v-s['nn'][p]['effects'][arm]['metrics'][k])<1e-14
        for arm,j in [('control',12),('global',22),('target',32)]:
            for k,v in gm(a[j:j+10]).items():assert abs(v-s['nn'][p]['arms'][arm]['metrics'][k])<1e-14
        b={a:C.defaultdict(list) for a in ('global_effect','target_effect','control','global','target','target_minus_global')}
        for _ in range(8000):
            q=sums([x[j] for j in rng.choices(ids,k=len(ids))]);v={a:gm(q[j:j+10]) for a,j in [('control',12),('global',22),('target',32)]}
            v.update(global_effect=pm(q[:6]),target_effect=pm(q[6:12]),target_minus_global={k:v['target'][k]-v['global'][k] for k in v['target']})
            for arm,z in v.items():
                for k,y in z.items():b[arm][k].append(y)
        out['nn'][p]={a:{k:ci(v) for k,v in z.items()} for a,z in b.items()}
    f96=list(raw('R_F96'));f128=list(raw('R_F128'))
    for cohort,aa,bb in [('historical',list(archived()),list(raw('R_H128'))),('fresh',f96,[g for g in f96 if g['job']['stratum']=='compact']+f128)]:
        out['rapfi'][cohort]={}
        for pan in ('all','compact','outer'):
            x=C.defaultdict(lambda:[0]*20)
            for arm,gs in enumerate((aa,bb)):
                for g in gs:
                    if pan!='all' and g['job']['stratum']!=pan:continue
                    j=g['job']['id'];v=game(g);a=x[j]
                    for i,y in enumerate(v):a[10*arm+i]+=y
            ids=sorted(x);q=sums(list(x.values()));a=gm(q[:10]);b=gm(q[10:])
            for arm,z in [('a',a),('b',b)]:
                for k,v in z.items():assert abs(v-s['rapfi'][cohort][pan][arm]['metrics'][k])<1e-14
            bs={a:C.defaultdict(list) for a in ('a','b','delta')}
            for _ in range(8000):
                q=sums([x[j] for j in rng.choices(ids,k=len(ids))]);a=gm(q[:10]);b=gm(q[10:])
                for arm,z in [('a',a),('b',b),('delta',{k:b[k]-a[k] for k in a if cohort!='historical' or k!='CPU_reduction'})]:
                    for k,v in z.items():bs[arm][k].append(v)
            out['rapfi'][cohort][pan]={a:{k:ci(v) for k,v in z.items()} for a,z in bs.items()}
    (T/'STATS_INDEPENDENT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print('INDEPENDENT NN',out['nn']['all']['target_effect'],flush=True)
    print('INDEPENDENT RAPFI',out['rapfi'],flush=True)
if __name__=='__main__':main()
