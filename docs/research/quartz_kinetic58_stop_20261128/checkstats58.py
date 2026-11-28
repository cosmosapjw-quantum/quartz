"""Independent standard-library family bootstrap; exact point ratios."""
import collections as C,gzip,json,random
from fractions import Fraction as F
from k58_plan import D,T,load,sha,SEED
ARMS=('control','stop','stable')
def rows(mode):
    for x in load(mode+'_LOCK.json')['files']:
        p=D/x['file'];assert sha(p)==x['sha256']
        with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def game(g):
    a=[g['score2'],1,0,0,0,0,0,0,0,0]
    for r in g['rows']:
        k=2 if r['arm']=='Q' else 6
        a[k]+=1;a[k+1]+=r.get('actual_nn',r['nn']);a[k+2]+=r['nn'];a[k+3]+=r['cpu_ns']
    return a
def summ(xs):
    return [sum(x) for x in zip(*xs)]
def add(x,y):
    for i,v in enumerate(y):x[i]+=v
def metrics(a):
    assert a[2]>0 and a[6]>0 and a[7]>0 and a[9]>0
    return dict(score=float(F(a[0],2*a[1])),NN_reduction=float(1-F(a[3]*a[6],a[7]*a[2])),logical_NN_reduction=float(1-F(a[4]*a[6],a[8]*a[2])),CPU_reduction=float(1-F(a[5]*a[6],a[9]*a[2])))
def ci(a):
    a=sorted(a);return dict(CI=[a[199],a[7799]],lower95=a[399])
def main():
    s=load('SUMMARY.json');rng=random.Random(SEED+989);x=C.defaultdict(lambda:[0]*30);j={}
    for r in rows('NN'):
        g=r['control'];i=g['job']['id'];j[i]=g['job'];add(x[i],sum((game(r[a]) for a in ARMS),[]))
    assert len(x)==160
    out=dict(nn={},rapfi={},replicates=8000,seed=SEED+989)
    for split in ('REPLICATION','CROSSOVER','all'):
        ids=sorted(i for i in x if split=='all' or j[i]['split']==split);v=summ([x[i] for i in ids]);g=[metrics(v[10*k:10*k+10]) for k in range(3)]
        for a,q in zip(ARMS,g):
            for k,val in q.items():
                z=s['nn'][split]['arms'][a]
                ref=z['logical']['metrics']['NN_reduction'] if k=='logical_NN_reduction' else z['physical']['metrics'][k]
                assert abs(val-ref)<1e-14
        strata=[[i for i in ids if j[i]['horizon']==h] for h in (4,8,12,16)]
        bs={a:C.defaultdict(list) for a in ARMS[1:]}
        for _ in range(8000):
            v=summ([x[i] for st in strata for i in rng.choices(st,k=len(st))]);g=[metrics(v[10*k:10*k+10]) for k in range(3)]
            for a,q in zip(ARMS[1:],g[1:]):
                bs[a]['delta_score'].append(q['score']-g[0]['score']);bs[a]['delta_saving'].append(q['NN_reduction']-g[0]['NN_reduction']);bs[a]['score'].append(q['score']);bs[a]['saving'].append(q['NN_reduction'])
        out['nn'][split]={a:{k:ci(v) for k,v in b.items()} for a,b in bs.items()}
    aa={(g['job']['id'],g['color']):g for g in rows('R_96')};bb={(g['job']['id'],g['color']):g for g in rows('R_T')};assert set(aa)==set(bb) and len(aa)==192
    x=C.defaultdict(lambda:[0]*20)
    for k,a in aa.items():add(x[k[0]],game(a)+game(bb[k]))
    ids=sorted(x);assert len(ids)==96;v=summ(list(x.values()));g=[metrics(v[k:k+10]) for k in (0,10)]
    for a,q in zip(('96','T'),g):
        for k,val in q.items():
            if k!='logical_NN_reduction':assert abs(val-s['rapfi']['arms'][a]['metrics'][k])<1e-14
    bs={a:C.defaultdict(list) for a in ('96','T','effect')}
    for _ in range(8000):
        v=summ([x[i] for i in rng.choices(ids,k=len(ids))]);g=[metrics(v[k:k+10]) for k in (0,10)]
        for a,q in zip(('96','T'),g):
            for k in ('score','NN_reduction'):bs[a][k].append(q[k])
        bs['effect']['score'].append(g[1]['score']-g[0]['score'])
    out['rapfi']={a:{k:ci(v) for k,v in b.items()} for a,b in bs.items()}
    (T/'STATS_INDEPENDENT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print('INDEPENDENT',json.dumps(out),flush=True)
if __name__=='__main__':main()
