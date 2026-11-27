"""Actual same-evaluator games; paired family bootstrap, signed costs separately."""
import collections as C,json,sys,math
import numpy as np
from k53_plan import D,T,SEEDS,load,save,sha
from experiment53 import rows
sys.path.append(str(D.parent/'quartz_kinetic50_intervention_20261127'))
from analyze50 import game,ci
ARMS=('control','cold','reuse','multi')
def pair_stats(pairs,idx):
    keys=('n','delta2','local_nn','downstream_Q','downstream_B','delta_Q','delta_B','delta_total','reached','changed','same_history','interventions','base_interventions','improved','worsened')
    g=C.defaultdict(C.Counter);tot=C.Counter()
    for r in pairs:
        q=C.Counter(n=1,improved=int(r['delta2']>0),worsened=int(r['delta2']<0))
        for k in keys[1:-2]:q[k]=r[k]
        g[r['job']].update(q);tot.update(q)
    a=np.array([[g[j][k] for k in keys] for j in sorted(g)],dtype=float);assert idx.shape[1]==len(a)
    z=a[idx].sum(axis=1);v=a.sum(axis=0);out=dict(counts={k:int(tot[k]) for k in keys},metrics={},CI={},ratio={})
    for i,k in enumerate(keys[1:],1):
        f=.5 if k=='delta2' else 1.;name='tau' if k=='delta2' else k
        out['metrics'][name]=float(f*v[i]/v[0]);out['CI'][name]=ci(f*z[:,i]/z[:,0])
    for k in ('local_nn','delta_Q','delta_total'):
        i=keys.index(k);bad=int((z[:,i]<=0).sum())
        out['ratio'][k]=dict(nonpositive_resamples=bad,point=float(v[1]/(2*v[i])) if v[i]>0 else None,CI=ci(z[:,1]/(2*z[:,i])) if bad==0 and v[i]>0 else None)
    return out
def contrast(a,b,idx):
    aa,za=game(a,idx);bb,zb=game(b,idx)
    return dict(a=aa,b=bb,delta={k:dict(point=bb['metrics'][k]-aa['metrics'][k],CI=ci(zb[k]-za[k])) for k in za})
def rp(c,t):
    assert c['job']==t['job'] and c['color']==t['color'] and c['seed']==t['seed']
    e=t['event'];local=0;changed=False
    if e is None:assert c['history']==t['history'] and c['nn']==t['nn']
    else:
        a,b=c['rows'][e],t['rows'][e];assert c['history'][:a['ply']]==t['history'][:b['ply']] and a['votes']==b['votes'] and a['nn']==b['root_nn']
        assert a['move']==b['target']['incumbent'];local=b['target']['extra_nn'];changed=a['move']!=b['move']
        if not changed:assert c['history']==t['history'] and t['nn']-c['nn']==local
    cost=lambda g,tag:sum(r['nn'] for r in g['rows'] if r['arm']==tag)
    dq=cost(t,'Q')-cost(c,'Q');db=cost(t,'B')-cost(c,'B')
    return dict(job=c['job']['id'],delta2=t['score2']-c['score2'],local_nn=local,downstream_Q=dq-local,downstream_B=db,delta_Q=dq,delta_B=db,delta_total=dq+db,reached=e is not None,changed=changed,same_history=c['history']==t['history'],interventions=int(e is not None),base_interventions=0)
def main():
    rng=np.random.default_rng(load('PLAN.json')['bootstrap_seed']);x=list(rows('NN'));assert len(x)==672
    out={}
    for p,xx in [('all',x),('old',[z for z in x if z['control']['seed'] in (101,102,103,105)]),('new',[z for z in x if z['control']['seed'] in (41,42,43)])]:
        idx=rng.integers(0,48,(8000,48));arms={a:[z[a] for z in xx] for a in ARMS}
        z=dict(arms={a:game(v,idx)[0] for a,v in arms.items()},pairs={a:pair_stats([z['pairs'][a] for z in xx],idx) for a in ('cold','reuse','multi')})
        for a,b in [('cold','control'),('reuse','cold'),('multi','cold')]:
            z['pairs'][a]['baseline_relative']=contrast(arms[b],arms[a],idx)['delta']
        out[p]=z
        print('NN',p,{a:(v['metrics']['score'],v['metrics']['NN_reduction'],v['metrics']['CPU_reduction'],v['gate']) for a,v in z['arms'].items()},{a:(v['counts'],v['CI']['tau'],v['ratio']['local_nn']) for a,v in z['pairs'].items()},flush=True)
    d=C.Counter();freq=C.Counter();by_seed=C.defaultdict(C.Counter);by_bin=C.defaultdict(C.Counter)
    for z in x:
        g=z['reuse'];e=g['event'];freq[len(z['multi']['events'])]+=1
        for a in ('cold','reuse','multi'):
            p=z['pairs'][a];by_seed[p['seed']][a+'_delta2']+=p['delta2']
        if e is None:continue
        t=g['rows'][e]['target'];s=z['cold']['rows'][e]['target']
        d['reached']+=1;d['same_candidate']+=t['rival']==s['rival'];d['same_chosen']+=t['chosen']==s['chosen']
        for c in t['children']:
            if c is None:continue
            d['children']+=1;d['initial_eval']+=c['initial_eval'];d['reused']+=not c['initial_eval'] and not c['terminal'];d['nn']+=c['nn'];d['old_visits']+=sum(e[1] for e in c['pre_edges'])
        by_bin[g['rows'][e]['bin']]['n']+=1;by_bin[g['rows'][e]['bin']]['gain2']+=z['pairs']['reuse']['delta2']
    a=list(rows('R_96'));b=list(rows('R_T'));assert len(a)==len(b)==128
    aa={(g['job']['id'],g['color']):g for g in a};bb={(g['job']['id'],g['color']):g for g in b};assert set(aa)==set(bb)
    idx=rng.integers(0,64,(8000,64));r=dict(arms={'96':game(a,idx)[0],'T':game(b,idx)[0]},paired=pair_stats([rp(aa[k],bb[k]) for k in sorted(aa)],idx),contrast=contrast(a,b,idx)['delta'])
    print('RAPFI',r,flush=True)
    save('SUMMARY.json',dict(nn=out,reuse_receipts=dict(d),multi_event_hist=dict(freq),by_seed={str(k):dict(v) for k,v in by_seed.items()},by_bin={str(k):dict(v) for k,v in by_bin.items()},rapfi=r,cache=load('CACHE_LOCK.json')['counts'],locks={m:sha(D/(m+'_LOCK.json')) for m in ('NN','R_96','R_T')},scope='8000 family bootstrap, exploratory separate actual Q-v-B arms, paired policy effects and signed costs; no posthoc adoption'))
if __name__=='__main__':main()
