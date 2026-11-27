"""Frozen contextual policy evaluation with actual same-evaluator matches."""
import collections as C,json,sys
import numpy as np
from k55_plan import D,T,SEEDS,load,save,sha
from experiment55 import rows
sys.path.append(str(D.parent/'quartz_kinetic54_cache_20261127'))
from analyze54 import cache_contrast,phys,game,pair_stats,contrast,rp
ARMS=('control','cold','reuse','select','mix')
def main():
    rng=np.random.default_rng(load('PLAN.json')['bootstrap_seed']);x=list(rows('NN'));assert len(x)==896
    out={}
    for p,xx in [('all',x),('old',[z for z in x if z['control']['seed'] in (101,102,103,105)]),('new',[z for z in x if z['control']['seed'] in (41,42,43)])]:
        idx=rng.integers(0,64,(8000,64));arms={a:[z[a] for z in xx] for a in ARMS}
        z=dict(arms={a:cache_contrast(v,idx) for a,v in arms.items()},pairs={a:pair_stats([z['pairs'][a] for z in xx],idx) for a in ARMS[1:]})
        for a in z['pairs']:
            z['pairs'][a]['baseline_relative']=contrast([phys(g) for g in arms['control']],[phys(g) for g in arms[a]],idx)['delta']
            z['pairs'][a]['logical_extra_total']=sum(r['pairs'][a]['logical_extra'] for r in xx)
        z['mix_vs_cold']=contrast([phys(g) for g in arms['cold']],[phys(g) for g in arms['mix']],idx)['delta']
        z['mix_vs_reuse']=contrast([phys(g) for g in arms['reuse']],[phys(g) for g in arms['mix']],idx)['delta']
        out[p]=z
        print('NN',p,{a:(v['physical']['metrics'],v['physical']['gate'],v['counts']) for a,v in z['arms'].items()},{a:(v['counts'],v['CI']['tau'],v['ratio']['local_nn']) for a,v in z['pairs'].items()},flush=True)
    diag=C.defaultdict(C.Counter);coin=C.Counter()
    for z in x:
        coin['high']+=z['high'];coin['n']+=1
        for a in ARMS[1:]:
            p=z['pairs'][a];q=diag[(a,str(p['cell']))];q['n']+=1;q['gain2']+=p['delta2'];q['logical']+=p['logical_extra'];q['physical']+=p['local_nn'];q['active']+=p['op']!=0;q['changed']+=p['changed']
    a=list(rows('R_96'));b=list(rows('R_T'));assert len(a)==len(b)==256
    rap={}
    for p in ('all','compact','tactical'):
        aa={(g['job']['id'],g['color']):g for g in a if p=='all' or g['job']['stratum']==p};bb={(g['job']['id'],g['color']):g for g in b if p=='all' or g['job']['stratum']==p};assert set(aa)==set(bb)
        n=len(aa)//2;idx=rng.integers(0,n,(8000,n));ga=[aa[k] for k in sorted(aa)];gb=[bb[k] for k in sorted(bb)]
        z=dict(arms={'96':game(ga,idx)[0],'T':game(gb,idx)[0]},paired=pair_stats([rp(aa[k],bb[k]) for k in sorted(aa)],idx),contrast=contrast(ga,gb,idx)['delta'])
        rap[p]=z;print('RAPFI',p,{a:(v['metrics'],v['lower95'],v['gate']) for a,v in z['arms'].items()},z['paired']['counts'],z['paired']['CI']['tau'],flush=True)
    save('SUMMARY.json',dict(nn=out,diagnostic={a:{k:dict(v) for (b,k),v in diag.items() if b==a} for a in ARMS[1:]},coin=dict(coin),rapfi=rap,locks={m:sha(D/(m+'_LOCK.json')) for m in ('NN','R_96','R_T')},frozen=sha(D/'FROZEN.json'),mix=sha(D/'MIX_FROZEN.json'),scope='8000 family bootstrap, five actual cached Q-v-B arms; integer and expected-budget selector, two Rapfi distributions'))
if __name__=='__main__':main()
