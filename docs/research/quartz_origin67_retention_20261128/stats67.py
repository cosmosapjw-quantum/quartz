from plan67 import D,T,ARMS,sha,read
import gzip,json,collections as C,random,math
from fractions import Fraction as F
import numpy as np
KEYS=('score2','games','Qnn','Qmoves','Bnn','Bmoves','Qcpu','Bcpu','wins','draws','losses','targets','switches','Qlogical','Blogical')
def calc(a):
    i={k:a[...,j] for j,k in enumerate(KEYS)}
    return dict(score=i['score2']/(2*i['games']),saving=1-(i['Qnn']/i['Qmoves'])/(i['Bnn']/i['Bmoves']),cpu_saving=1-(i['Qcpu']/i['Qmoves'])/(i['Bcpu']/i['Bmoves']),q_rate=i['Qnn']/i['Qmoves'],b_rate=i['Bnn']/i['Bmoves'],utility=(i['score2']/2-i['Qnn']/500)/i['games'],q_cost_per_game=i['Qnn']/i['games'])
def low(x):return float(np.sort(x)[399])
def interval(x):return [float(np.sort(x)[199]),float(np.sort(x)[7799])]
def arena(mode):
    with gzip.open(T/(mode+'_AUDIT.json.gz'),'rt') as f:aud=json.load(f)
    rows=aud['rows'];N=24 if mode=='MAIN' else 12;M=7 if mode=='MAIN' else 3;assert len(rows)==2*N*M
    ids=sorted({r['id'] for r in rows});ii={j:i for i,j in enumerate(ids)};fdata=np.zeros((N,9,len(KEYS)),dtype=np.int64);cells=C.defaultdict(list);byid={}
    for r in rows:
        i=ii[r['id']];byid[i]=r['geometry']
        for k,arm in enumerate(ARMS):fdata[i,k]+=np.array([r['arms'][arm].get(x,0) for x in KEYS],dtype=np.int64)
    for i,c in sorted(byid.items()):cells[c].append(i)
    assert len(cells)==2 and all(len(v)==N//2 for v in cells.values()) and (fdata[:,:,1]==2*M).all()
    rng=np.random.default_rng(202611286700+["MAIN","ARCHIVE","NOVEL","TRANSFER"].index(mode));ix=np.concatenate([rng.choice(v,(8000,len(v)),replace=True) for v in cells.values()],axis=1)
    agg=fdata.sum(axis=0);boot=fdata[ix].sum(axis=1);point=calc(agg);bs=calc(boot)
    out={'mode':mode,'arms':{},'contrasts':{},'audit_sha256':sha(T/(mode+'_AUDIT.json.gz'))}
    for k,arm in enumerate(ARMS):
        p={n:float(v[k]) for n,v in point.items()};b={n:low(v[:,k]) for n,v in bs.items()}
        out['arms'][arm]=dict(point=p,lower95=b,counts={x:int(v) for x,v in zip(KEYS,agg[k])})
    for t,b in [(1,0),(2,1),(3,2),(4,2),(5,2),(6,3),(7,4),(8,5),(6,2),(7,2),(8,2),(6,1),(7,1),(8,1)]:
        x={n:float(v[t]-v[b]) for n,v in point.items()};y={n:low(v[:,t]-v[:,b]) for n,v in bs.items()}
        gates=dict(score_NI=y['score']>=-.02,absolute_saving=out['arms'][ARMS[t]]['lower95']['saving']>=.85,relative_saving=y['saving']>=-.05,cpu_positive=bool(point['cpu_saving'][t]>0),signed_utility=y['utility']>0)
        out['contrasts'][ARMS[t]+'-'+ARMS[b]]=dict(point=x,lower95=y,CI={n:interval(v[:,t]-v[:,b]) for n,v in bs.items()},gates=gates,performance=all(gates[k] for k in ('score_NI','absolute_saving','relative_saving','cpu_positive')),benefit=all(gates.values()))
        if b==0:
            p=[r['pairs'][ARMS[t]] for r in rows]
            out['contrasts'][ARMS[t]+'-H4']['pairs']=dict(diverged=sum(z['diverged'] for z in p),improved=sum(z['delta2']>0 for z in p),worsened=sum(z['delta2']<0 for z in p))
    data=fdata.tolist();rng=random.Random(202611286710+["MAIN","ARCHIVE","NOVEL","TRANSFER"].index(mode));vv=C.defaultdict(list)
    def single(s):return F(s[0],2*s[1]),1-F(s[2]*s[5],s[3]*s[4]),F(s[0]*250-s[2],500*s[1])
    for _ in range(8000):
        sel=[rng.choice(v) for v in cells.values() for j in v];a=[single([sum(data[i][k][j] for i in sel) for j in range(6)]) for k in range(9)]
        for t,b in [(6,3),(7,4),(8,5),(6,2),(7,2),(8,2)]:
            for i,key in enumerate(('score','saving','utility')):vv[ARMS[t]+'-'+ARMS[b],key].append(float(a[t][i]-a[b][i]))
    out['promotion']={k:out['contrasts'][k]['benefit'] for k in ('CALV-VOTE','CALF-FOCUS','CALX-EXPAND')}
    out['independent']={f'{a}:{k}':sorted(v)[399] for (a,k),v in vv.items()}
    (T/(mode+'_SUMMARY.json')).write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(mode,json.dumps(out),flush=True)
if __name__=='__main__':
    import sys
    arena(sys.argv[1])
