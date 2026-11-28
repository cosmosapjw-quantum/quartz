"""Actual early policy effects, support and frozen heldout two-factor predictor."""
import collections as C,json,sys,math
import numpy as np
from k57_plan import D,T,load,save,sha
from experiment57 import rows
from fit57 import cell
sys.path.append(str(D.parent/'quartz_kinetic54_cache_20261127'))
from analyze54 import cache_contrast,phys,game,pair_stats,contrast,rp
ARMS=('control','stop','cold','reuse')
PAIRS=('stop','cold','reuse','cold_vs20','reuse_vs20')
def partition(x,idx):
    a={k:cache_contrast([z[k] for z in x],idx) for k in ARMS}
    p={k:pair_stats([z['pairs'][k] for z in x],idx) for k in PAIRS}
    for k,r in p.items():
        rs=[z['pairs'][k] for z in x]
        for v in ('target_extra','base_delta','logical_extra'):r['counts'][v]=sum(z[v] for z in rs)
        n=r['counts']['reached'];f=r['counts']['changed'];gain=r['counts']['delta2']/2
        if n:
            r['factor']=dict(p_flip=f/n,m=gain/f if f else None,mu=gain/n)
            assert abs(gain/n-(f/n*(gain/f) if f else 0))<1e-15
        else:r['factor']=None
    return dict(arms=a,pairs=p)
def main():
    rng=np.random.default_rng(load('PLAN.json')['bootstrap_seed']);x=list(rows('NN'));assert len(x)==896
    out={}
    for label,xx in [('all',x)]+[(s,[z for z in x if z['control']['job']['split']==s]) for s in ('TRAIN','CAL','TEST')]+[(str(h),[z for z in x if z['control']['job']['horizon']==h]) for h in (4,8,12,16)]:
        n=len(xx)//14;idx=rng.integers(0,n,(8000,n));out[label]=partition(xx,idx)
        print('NN',label,{a:(v['physical']['metrics']['score'],v['physical']['metrics']['NN_reduction']) for a,v in out[label]['arms'].items()},{a:(v['counts'],v['CI']['tau']) for a,v in out[label]['pairs'].items()},flush=True)
    fr=load('FROZEN.json');assert fr['lock']==sha(D/'NN_LOCK.json')
    pred={};support=C.defaultdict(lambda:C.defaultdict(set));hist=C.Counter();reasons=C.Counter()
    for z in x:
        g=z['control'];i=g['attempt']
        if i is None:reasons['no_noncert']=reasons.get('no_noncert',0)+1;continue
        e=g['rows'][i]['early'];reasons[e['reason']]+=1
        for r in e['snapshots']:
            n=r['visits'];support[n]['distance'].add(r['distance']);support[n]['prior_gap'].add(max(v[3] for v in r['edges'])-next(v[3] for v in r['edges'] if v[0]==r['move']))
            support[n]['leader_value'].add(next(v[2]/v[1] if v[1] else 0 for v in r['edges'] if v[0]==r['move']))
            hist[n]+=1
    for s in ('TRAIN','CAL','TEST'):
        xx=[z for z in x if z['control']['job']['split']==s];res={};replay=C.Counter()
        for a in ('cold','reuse'):
            u=C.Counter();bycell=C.defaultdict(C.Counter);m=fr['arms'][a]
            for z in xx:
                p=z['pairs'][a]
                if not p['reached']:continue
                k=cell(p);v=m['cells'][k];q=m['constant'];d=p['delta2']/2;f=int(p['changed'])
                for c in (u,bycell[k]):
                    c['n']+=1;c['flip']+=f;c['gain2']+=p['delta2'];c['pred_flip']+=v['p'];c['pred_gain']+=v['mu'];c['cost']+=p['local_nn'];c['brier']+=(f-v['p'])**2;c['brier0']+=(f-q['p'])**2;c['mse']+=(d-v['mu'])**2;c['mse0']+=(d-q['mu'])**2
            n=u['n'];res[a]=dict(counts=dict(u),metrics={k:u[k]/n for k in ('brier','brier0','mse','mse0')},cells={k:dict(v) for k,v in bycell.items()})
        for z in xx:
            replay['episodes']+=1
            if not z['pairs']['cold']['reached']:replay['stop']+=1;continue
            k=cell(z['pairs']['cold']);g=[0]+[fr['arms'][a]['cells'][k]['mu'] for a in ('cold','reuse')]
            a=('stop','cold','reuse')[max(range(3),key=lambda i:(g[i],-i))]
            replay[a]+=1
            if a!='stop':
                r=z['pairs'][a];replay['gain2']+=r['delta2'];replay['local_nn']+=r['local_nn'];replay['pred_gain']+=max(g)
        res['replay']=dict(replay);pred[s]=res
    a=list(rows('R_96'));b=list(rows('R_T'));assert len(a)==len(b)==96
    rap={}
    for s in ('all','compact','outer'):
        aa={(g['job']['id'],g['color']):g for g in a if s=='all' or g['job']['stratum']==s};bb={(g['job']['id'],g['color']):g for g in b if s=='all' or g['job']['stratum']==s};assert set(aa)==set(bb)
        n=len(aa)//2;idx=rng.integers(0,n,(8000,n));ga=[aa[k] for k in sorted(aa)];gb=[bb[k] for k in sorted(bb)]
        rap[s]=dict(arms={'96':game(ga,idx)[0],'T':game(gb,idx)[0]},paired=pair_stats([rp(aa[k],bb[k]) for k in sorted(aa)],idx),contrast=contrast(ga,gb,idx)['delta'])
    sr={str(n):{k:dict(unique=len(v),min=min(v),max=max(v)) for k,v in a.items()} for n,a in support.items()}
    save('SUMMARY.json',dict(nn=out,predictive=pred,support=sr,snapshot_counts=dict(hist),reasons=dict(reasons),rapfi=rap,locks={m:sha(D/(m+'_LOCK.json')) for m in ('NN','R_96','R_T')},frozen=sha(D/'FROZEN.json'),scope='actual paired early effects; heldout factorized predictor; offline replay not additional games; family bootstrap8000'))
    print('PREDICTIVE',json.dumps(pred),flush=True);print('SUPPORT',sr,dict(hist),dict(reasons),flush=True)
    print('RAPFI',json.dumps(rap),flush=True)
if __name__=='__main__':main()
