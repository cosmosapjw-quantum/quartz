"""Preregistered family-stratified NI and physical savings, separate exploratory stability and crossover."""
import collections as C,json,sys
import numpy as np
from k58_plan import D,T,load,save,sha
from experiment58 import rows
sys.path.append(str(D.parent/'quartz_kinetic54_cache_20261127'))
from analyze54 import cache_contrast,phys,game,pair_stats,contrast,rp
ARMS=('control','stop','stable')
def idx_strat(xx,rng):
    j={z['control']['job']['id']:z['control']['job']['horizon'] for z in xx}
    ids=sorted(j);pos={k:i for i,k in enumerate(ids)}
    return np.concatenate([rng.choice([pos[k] for k in ids if j[k]==h],(8000,sum(j[k]==h for k in ids)),replace=True) for h in (4,8,12,16)],axis=1)
def panel(xx,idx):
    a={k:cache_contrast([z[k] for z in xx],idx) for k in ARMS}
    p={k:pair_stats([z['pairs'][k] for z in xx],idx) for k in ARMS[1:]}
    cmp={}
    for k in ARMS[1:]:
        gs=[phys(z[k]) for z in xx];bs=[phys(z['control']) for z in xx]
        _,b=game(bs,idx);_,t=game(gs,idx);dv=t['NN_reduction']-b['NN_reduction'];ds=t['score']-b['score'];low=lambda x:float(np.sort(x)[399])
        assert abs(float(ds.mean())-p[k]['metrics']['tau'])<.01
        cmp[k]=dict(delta_saving=a[k]['physical']['metrics']['NN_reduction']-a['control']['physical']['metrics']['NN_reduction'],saving_lower95=low(dv),saving_CI=[float(np.sort(dv)[199]),float(np.sort(dv)[7799])],score_lower95=low(ds),score_CI=p[k]['CI']['tau'],criteria=dict(paired_NI=low(ds)>=-.02,extra_saving=low(dv)>=.05,absolute=all(a[k]['physical']['gate'].values())))
    return dict(arms=a,pairs=p,comparison=cmp)
def main():
    rng=np.random.default_rng(load('PLAN.json')['bootstrap_seed']);x=list(rows('NN'));assert len(x)==2240
    out={}
    for split in ('REPLICATION','CROSSOVER','all'):
        xx=[z for z in x if split=='all' or z['control']['job']['split']==split];n=len(xx)//14
        idx=idx_strat(xx,rng);assert idx.shape==(8000,n)
        q=panel(xx,idx);q['unstratified']=panel(xx,rng.integers(0,n,(8000,n)))['comparison'];out[split]=q
        print('NN',split,{a:(v['physical']['metrics'],v['physical']['lower95'],v['physical']['gate']) for a,v in q['arms'].items()},q['comparison'],flush=True)
    cells=C.defaultdict(C.Counter);hist=C.defaultdict(C.Counter)
    for z in x:
        j=z['control']['job'];key=(j['split'],str(j['horizon']),j['geometry'])
        for a in ARMS:
            g=z[a];hist[a]['episodes']+=1;hist[a]['attempts']+=g['attempt'] is not None;hist[a]['events']+=g['event'] is not None
            if g['event'] is not None:hist[a]['h'+str(g['rows'][g['event']]['early']['at']['visits'])]+=1
        for a in ARMS[1:]:
            r=z['pairs'][a];c=cells[key+(a,)];c['n']+=1;c['reached']+=r['reached'];c['changed']+=r['changed'];c['gain2']+=r['delta2'];c['local_nn']+=r['local_nn'];c['delta_Q']+=r['delta_Q'];c['delta_total']+=r['delta_total']
    aa={(g['job']['id'],g['color']):g for g in rows('R_96')};bb={(g['job']['id'],g['color']):g for g in rows('R_T')};assert set(aa)==set(bb) and len(aa)==192
    idx=rng.integers(0,96,(8000,96));ga=[aa[k] for k in sorted(aa)];gb=[bb[k] for k in sorted(bb)]
    rap=dict(arms={'96':game(ga,idx)[0],'T':game(gb,idx)[0]},paired=pair_stats([rp(aa[k],bb[k]) for k in sorted(aa)],idx),contrast=contrast(ga,gb,idx)['delta'])
    print('RAPFI',json.dumps(rap),flush=True)
    save('SUMMARY.json',dict(nn=out,cell_points={'|'.join(k):dict(v) for k,v in cells.items()},events={k:dict(v) for k,v in hist.items()},rapfi=rap,locks={m:sha(D/(m+'_LOCK.json')) for m in ('NN','R_96','R_T')},primary='REPLICATION STOP paired NI -0.02 and extra saving +.05 one-sided95 family-stratified; other contrasts diagnostic'))
if __name__=='__main__':main()
