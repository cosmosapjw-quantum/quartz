"""Family-stratified fixed-panel paired inference, plus independent stdlib resampling."""
from plan59 import D,T,ARMS,sha
import gzip,json,collections as C,random,math
from fractions import Fraction
import numpy as np
KEYS=('score2','games','Qnn','Qmoves','Bnn','Bmoves','Qcpu','Bcpu','wins','draws','losses','events','attempts','Qlogical','Blogical')
def calc(a):
    i={k:a[...,j] for j,k in enumerate(KEYS)}
    return dict(score=i['score2']/(2*i['games']),saving=1-(i['Qnn']/i['Qmoves'])/(i['Bnn']/i['Bmoves']),cpu_saving=1-(i['Qcpu']/i['Qmoves'])/(i['Bcpu']/i['Bmoves']),q_rate=i['Qnn']/i['Qmoves'],b_rate=i['Bnn']/i['Bmoves'])
def low(x):return float(np.sort(x)[399])
def interval(x):
    return [float(np.sort(x)[199]),float(np.sort(x)[7799])]
def main():
    with gzip.open(T/'AUDIT.json.gz','rt') as f:aud=json.load(f)
    rows=aud['episodes']; assert len(rows)==896
    fdata=np.zeros((64,3,len(KEYS)),dtype=np.int64);cells=C.defaultdict(list);byid={}
    for r in rows:
        byid[r['id']]=(r['horizon'],r['geometry'])
        for k,arm in enumerate(ARMS):
            fdata[r['id'],k]+=np.array([r['arms'][arm].get(x,0) for x in KEYS],dtype=np.int64)
    for i,c in sorted(byid.items()):cells[c].append(i)
    assert len(cells)==8 and all(len(v)==8 for v in cells.values()) and (fdata[:,:,1]==14).all()
    rng=np.random.default_rng(202611285900)
    ix=np.concatenate([rng.choice(v,(8000,len(v)),replace=True) for v in cells.values()],axis=1)
    agg=fdata.sum(axis=0);boot=fdata[ix].sum(axis=1)
    point=calc(agg);bs=calc(boot)
    out={'arms':{},'contrasts':{},'cells':{},'audit_sha256':sha(T/'AUDIT.json.gz')}
    for k,arm in enumerate(ARMS):
        p={n:float(v[k]) for n,v in point.items()}
        b={n:low(v[:,k]) for n,v in bs.items()}
        gate=dict(score=b['score']>=.45,NN=b['saving']>=.30,CPU=p['cpu_saving']>0)
        out['arms'][arm]=dict(point=p,lower95=b,gate=gate,counts={x:int(v) for x,v in zip(KEYS,agg[k])})
    for t,b in [(2,1),(2,0),(1,0)]:
        x={n:float(v[t]-v[b]) for n,v in point.items()}
        y={n:low(v[:,t]-v[:,b]) for n,v in bs.items()}
        out['contrasts'][ARMS[t]+'-'+ARMS[b]]=dict(point=x,lower95=y,CI={n:interval(v[:,t]-v[:,b]) for n,v in bs.items()},joint=y['score']>=-.02 and y['saving']>=.05 and all(out['arms'][ARMS[t]]['gate'].values()))
    for cell,ids in cells.items():
        z=calc(fdata[ids].sum(axis=0))
        out['cells'][str(cell)]=dict(delta_score=float(z['score'][2]-z['score'][1]),delta_saving=float(z['saving'][2]-z['saving'][1]))
    pairs=[r['pair'] for r in rows]
    out['pairs']={'diverged':sum(z['divergence'] is not None for z in pairs),'improved':sum(z['delta2']>0 for z in pairs),'worsened':sum(z['delta2']<0 for z in pairs),'extra_attempts':sum(z['extra_attempts'] for z in pairs),'extra_events':sum(z['extra_events'] for z in pairs),'common_local_nn':sum(z['common_local_nn'] for z in pairs),'delta_nn':sum(z['delta_nn'] for z in pairs)}
    # Independent stdlib bootstrap with separate RNG and exact integer family sums; no per-root resampling.
    data=fdata.tolist();r=random.Random(202611285901);ds=[];dv=[]
    def single(s):
        score=Fraction(s[0],2*s[1]);saving=1-Fraction(s[2]*s[5],s[3]*s[4]);return score,saving
    f0=single(agg[1].tolist());f1=single(agg[2].tolist())
    assert abs(float(f1[0]-f0[0])-out['contrasts']['REPEAT-FIRST']['point']['score'])<1e-12
    assert abs(float(f1[1]-f0[1])-out['contrasts']['REPEAT-FIRST']['point']['saving'])<1e-12
    for _ in range(8000):
        sel=[r.choice(v) for v in cells.values() for j in v]
        aa=[[sum(data[i][k][j] for i in sel) for j in range(6)] for k in (1,2)]
        u,v=map(single,aa);ds.append(float(v[0]-u[0]));dv.append(float(v[1]-u[1]))
    out['independent']=dict(score_lower95=sorted(ds)[399],saving_lower95=sorted(dv)[399])
    out['independent']['joint']=out['independent']['score_lower95']>=-.02 and out['independent']['saving_lower95']>=.05 and all(out['arms']['REPEAT']['gate'].values())
    (T/'SUMMARY.json').write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(out,indent=2,sort_keys=True),flush=True)
if __name__=='__main__':main()
