"""Prospective physical-cost selection and screened tactical transfer."""
import collections as C,json,sys
import numpy as np
from k56_plan import D,T,SEEDS,load,save,sha
from experiment56 import rows
sys.path.append(str(D.parent/'quartz_kinetic54_cache_20261127'))
from analyze54 import cache_contrast,phys,game,pair_stats,contrast,rp
ARMS=('control','cold','reuse','voc')
def main():
    rng=np.random.default_rng(load('PLAN.json')['bootstrap_seed']);x=list(rows('NN'));assert len(x)==672
    out={}
    for p,xx in [('all',x),('old',[z for z in x if z['control']['seed'] in (101,102,103,105)]),('new',[z for z in x if z['control']['seed'] in (41,42,43)])]:
        idx=rng.integers(0,48,(8000,48));arms={a:[z[a] for z in xx] for a in ARMS}
        z=dict(arms={a:cache_contrast(v,idx) for a,v in arms.items()},pairs={a:pair_stats([z['pairs'][a] for z in xx],idx) for a in ARMS[1:]})
        for a in z['pairs']:
            z['pairs'][a]['baseline_relative']=contrast([phys(g) for g in arms['control']],[phys(g) for g in arms[a]],idx)['delta']
            z['pairs'][a]['logical_extra_total']=sum(r['pairs'][a]['logical_extra'] for r in xx)
        z['voc_vs_cold']=contrast([phys(g) for g in arms['cold']],[phys(g) for g in arms['voc']],idx)['delta']
        z['voc_vs_reuse']=contrast([phys(g) for g in arms['reuse']],[phys(g) for g in arms['voc']],idx)['delta']
        z['reuse_vs_cold']=contrast([phys(g) for g in arms['cold']],[phys(g) for g in arms['reuse']],idx)['delta']
        out[p]=z
        print('NN',p,{a:(v['physical']['metrics'],v['physical']['gate'],v['counts']) for a,v in z['arms'].items()},{a:(v['counts'],v['CI']['tau'],v['ratio']['local_nn']) for a,v in z['pairs'].items()},flush=True)
    from fit56 import features
    from predictor56 import predict
    m=load('FROZEN.json')['model'];means=load('FROZEN.json')['means'];diag=C.Counter();err=C.Counter();by_cell=C.defaultdict(C.Counter);by_seed=C.defaultdict(C.Counter)
    for z in x:
        g=z['control'];e=g['event'];p=z['pairs']['voc'];q=by_seed[g['seed']]
        for c in (diag,q):c['episodes']+=1;c['reached']+=e is not None;c['gain2']+=p['delta2'];c['physical']+=p['local_nn']
        if e is None:continue
        x0=features(g['rows'][e]);op,a,b=predict(m,x0);assert op==p['op']
        for c in (diag,q,by_cell[p['cell']]):
            c['events']+=1;c['op'+str(op)]+=1;c['pred_gain']+=a[op];c['pred_cost']+=b[op];c['observed_gain']+=p['delta2']/2;c['observed_cost']+=p['local_nn'];c['all_cold_cost']+=z['pairs']['cold']['local_nn']
        for i,arm in enumerate(('cold','reuse'),1):
            r=z['pairs'][arm]
            for v,y,pred,base in [('gain',r['delta2']/2,a[i],means['gain'][i-1]),('cost',r['local_nn'],b[i],means['cost'][i-1])]:
                err[v+str(i)+'_sq']+=(y-pred)**2;err[v+str(i)+'_const']+=(y-base)**2
    diag['actual_budget_fraction']=diag['observed_cost']/diag['all_cold_cost']
    mse={k:v/diag['events'] for k,v in err.items()}
    a=list(rows('R_96'));b=list(rows('R_T'));assert len(a)==len(b)==144
    rap={}
    for p in ('all','unresolved_tactical','supplement'):
        aa={(g['job']['id'],g['color']):g for g in a if p=='all' or g['job']['stratum']==p};bb={(g['job']['id'],g['color']):g for g in b if p=='all' or g['job']['stratum']==p};assert set(aa)==set(bb)
        n=len(aa)//2;idx=rng.integers(0,n,(8000,n));ga=[aa[k] for k in sorted(aa)];gb=[bb[k] for k in sorted(bb)]
        z=dict(arms={'96':game(ga,idx)[0],'T':game(gb,idx)[0]},paired=pair_stats([rp(aa[k],bb[k]) for k in sorted(aa)],idx),contrast=contrast(ga,gb,idx)['delta'])
        for g in gb:
            if g['color']==1:assert g['event']==0
        rap[p]=z;print('RAPFI',p,{a:(v['metrics'],v['lower95'],v['gate']) for a,v in z['arms'].items()},z['paired']['counts'],z['paired']['CI']['tau'],flush=True)
    save('SUMMARY.json',dict(nn=out,voc_diagnostic=dict(diag),voc_MSE=mse,by_cell={str(k):dict(v) for k,v in by_cell.items()},by_seed={str(k):dict(v) for k,v in by_seed.items()},rapfi=rap,locks={m:sha(D/(m+'_LOCK.json')) for m in ('NN','R_96','R_T')},frozen=sha(D/'FROZEN.json'),runtime=sha(D/'RUNTIME_FROZEN.json'),scope='8000 family bootstrap; physical gain/cost predictor diagnostic, two separately screened tactical cohorts'))
if __name__=='__main__':main()
