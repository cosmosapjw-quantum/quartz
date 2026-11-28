"""Family-stratified paired whole-game inference and independent exact resampling."""
from plan61 import D,T,ARMS,sha,read
import gzip,json,collections as C,random
from fractions import Fraction as F
import numpy as np
KEYS=('score2','games','Qnn','Qmoves','Bnn','Bmoves','Qcpu','Bcpu','wins','draws','losses','events','attempts','Qlogical','Blogical')
def calc(a):
    i={k:a[...,j] for j,k in enumerate(KEYS)}
    return dict(score=i['score2']/(2*i['games']),saving=1-(i['Qnn']/i['Qmoves'])/(i['Bnn']/i['Bmoves']),cpu_saving=1-(i['Qcpu']/i['Qmoves'])/(i['Bcpu']/i['Bmoves']),q_rate=i['Qnn']/i['Qmoves'],b_rate=i['Bnn']/i['Bmoves'],utility=(i['score2']/2-i['Qnn']/500)/i['games'],q_cost_per_game=i['Qnn']/i['games'])
def low(x):return float(np.sort(x)[399])
def interval(x):return [float(np.sort(x)[199]),float(np.sort(x)[7799])]
def main():
    with gzip.open(T/'ARENA_AUDIT.json.gz','rt') as f:aud=json.load(f)
    rows=aud['rows'];assert len(rows)==672
    fdata=np.zeros((48,4,len(KEYS)),dtype=np.int64);cells=C.defaultdict(list);byid={}
    for r in rows:
        byid[r['id']]=r['geometry']
        for k,arm in enumerate(ARMS):fdata[r['id'],k]+=np.array([r['arms'][arm].get(x,0) for x in KEYS],dtype=np.int64)
    for i,c in sorted(byid.items()):cells[c].append(i)
    assert len(cells)==2 and all(len(v)==24 for v in cells.values()) and (fdata[:,:,1]==14).all()
    rng=np.random.default_rng(202611286100);ix=np.concatenate([rng.choice(v,(8000,len(v)),replace=True) for v in cells.values()],axis=1)
    agg=fdata.sum(axis=0);boot=fdata[ix].sum(axis=1);point=calc(agg);bs=calc(boot)
    out={'arms':{},'contrasts':{},'cells':{},'audit_sha256':sha(T/'ARENA_AUDIT.json.gz'),'adaptive_cells':aud['cells']}
    for k,arm in enumerate(ARMS):
        p={n:float(v[k]) for n,v in point.items()};b={n:low(v[:,k]) for n,v in bs.items()}
        out['arms'][arm]=dict(point=p,lower95=b,counts={x:int(v) for x,v in zip(KEYS,agg[k])})
    for t in (1,2,3):
        b=0;x={n:float(v[t]-v[b]) for n,v in point.items()};y={n:low(v[:,t]-v[:,b]) for n,v in bs.items()}
        counts=aud['cells'].get(ARMS[t],{});nstop=sum(v for k,v in counts.items() if k.endswith(':stop'));nextend=sum(v for k,v in counts.items() if k.endswith(':extend'))
        gates=dict(score_NI=y['score']>=-.02,absolute_saving=out['arms'][ARMS[t]]['lower95']['saving']>=.85,relative_saving=y['saving']>=-.05,cpu_positive=out['arms'][ARMS[t]]['point']['cpu_saving']>0,mixed=nstop>0 and nextend>0,signed_utility=y['utility']>0)
        out['contrasts'][ARMS[t]+'-H4']=dict(point=x,lower95=y,CI={n:interval(v[:,t]-v[:,b]) for n,v in bs.items()},gates=gates,performance=all(gates[k] for k in ('score_NI','absolute_saving','relative_saving','cpu_positive')),adaptive_benefit=all(gates.values()),decisions=dict(stop=nstop,extend=nextend))
        pairs=[r['pairs'][ARMS[t]] for r in rows];out['contrasts'][ARMS[t]+'-H4']['pairs']=dict(diverged=sum(z['diverged'] for z in pairs),improved=sum(z['delta2']>0 for z in pairs),worsened=sum(z['delta2']<0 for z in pairs))
    for cell,ids in cells.items():
        z=calc(fdata[ids].sum(axis=0));out['cells'][str(cell)]={ARMS[t]:dict(delta_score=float(z['score'][t]-z['score'][0]),delta_saving=float(z['saving'][t]-z['saving'][0]),delta_utility=float(z['utility'][t]-z['utility'][0])) for t in (1,2,3)}
    data=fdata.tolist();r=random.Random(202611286103);ds=[];dv=[];du=[]
    def single(s):
        return F(s[0],2*s[1]),1-F(s[2]*s[5],s[3]*s[4]),F(s[0]*250-s[2],500*s[1])
    a,b=map(single,[agg[0].tolist(),agg[1].tolist()])
    for i,k in enumerate(('score','saving','utility')):assert abs(float(b[i]-a[i])-out['contrasts']['SELECT-H4']['point'][k])<1e-12
    for _ in range(8000):
        sel=[r.choice(v) for v in cells.values() for j in v]
        aa=[[sum(data[i][k][j] for i in sel) for j in range(6)] for k in (0,1)]
        u,v=map(single,aa);ds.append(float(v[0]-u[0]));dv.append(float(v[1]-u[1]));du.append(float(v[2]-u[2]))
    out['independent']=dict(score_lower95=sorted(ds)[399],saving_lower95=sorted(dv)[399],utility_lower95=sorted(du)[399])
    (T/'SUMMARY.json').write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(out,indent=2,sort_keys=True),flush=True)
if __name__=='__main__':main()
