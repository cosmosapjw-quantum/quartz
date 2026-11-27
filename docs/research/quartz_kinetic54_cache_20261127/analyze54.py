"""Fair physical-vs-logical cost comparison and independent Rapfi confirmation."""
import collections as C,json,sys
import numpy as np
from k54_plan import D,T,SEEDS,load,save,sha
from experiment54 import rows,sem
sys.path.append(str(D.parent/'quartz_kinetic53_reuse_20261127'))
from analyze53 import pair_stats,contrast,rp
from analyze50 import game,ci
ARMS=('control','cold','conditional')
def phys(g):return dict(g,rows=[dict(r,nn=r['actual_nn']) for r in g['rows']],nn=g['actual_nn'])
def costs(gs):
    ct=C.Counter()
    for g in gs:
        for r in g['rows']:
            a=r['arm'];z=r['cache_receipt'];ct[a+'_logical']+=r['nn'];ct[a+'_physical']+=r['actual_nn'];ct[a+'_hits']+=z['hits'];ct[a+'_moves']+=1
            if 'target' in r:ct['target_logical']+=r['target']['extra_nn'];ct['target_physical']+=z['physical']-z['root_physical'];ct['target_events']+=1
    for a in ('Q','B'):assert ct[a+'_logical']==ct[a+'_physical']+ct[a+'_hits']
    return dict(ct)
def cache_contrast(gs,idx):
    z=contrast(gs,[phys(g) for g in gs],idx);c=costs(gs)
    a=z['a']['metrics']['NN_reduction'];b=z['b']['metrics']['NN_reduction']
    q=c['Q_logical']/c['Q_moves'];h=c['Q_hits']/c['Q_moves'];n=c['B_logical']/c['B_moves'];hh=c['B_hits']/c['B_moves']
    assert abs((b-a)-(n*h-q*hh)/(n*(n-hh)))<1e-14
    return dict(logical=z['a'],physical=z['b'],delta=z['delta']['NN_reduction'],counts=c)
def main():
    rng=np.random.default_rng(load('PLAN.json')['bootstrap_seed']);x=list(rows('NN'));assert len(x)==672
    out={}
    for p,xx in [('all',x),('old',[z for z in x if z['control']['seed'] in (101,102,103,105)]),('new',[z for z in x if z['control']['seed'] in (41,42,43)])]:
        idx=rng.integers(0,48,(8000,48));arms={a:[z[a] for z in xx] for a in ARMS}
        z=dict(arms={a:cache_contrast(v,idx) for a,v in arms.items()},pairs={a:pair_stats([z['pairs'][a] for z in xx],idx) for a in ('cold','conditional')})
        for a,b in [('cold','control'),('conditional','cold')]:z['pairs'][a]['baseline_relative']=contrast([phys(g) for g in arms[b]],[phys(g) for g in arms[a]],idx)['delta']
        for a in z['pairs']:z['pairs'][a]['logical_extra_total']=sum(r['pairs'][a]['logical_extra'] for r in xx)
        out[p]=z
        print('NN',p,{a:(v['physical']['metrics'],v['physical']['gate'],v['counts']) for a,v in z['arms'].items()},{a:(v['counts'],v['CI']['tau'],v['ratio']['local_nn']) for a,v in z['pairs'].items()},flush=True)
    unc=list(rows('UNCACHED'));assert len(unc)==112
    key=lambda g:(g['job']['id'],g['seed'],g['color'])
    d={key(g):g for g in unc};c={key(z['cold']):z['cold'] for z in x if z['cold']['job']['id']<8};assert set(d)==set(c)
    for k in d:
        a,b=d[k],c[k];assert a['history']==b['history'] and a['score2']==b['score2'] and a['nn']==b['nn']
        assert sem(a['rows'])==sem(b['rows'])
    idx=rng.integers(0,8,(8000,8));cached=[c[k] for k in sorted(c)];u=[d[k] for k in sorted(d)]
    rep=dict(n=len(unc),cached=cache_contrast(cached,idx),uncached=game(u,idx)[0],physical_contrast=contrast(u,[phys(g) for g in cached],idx)['delta'],scope='same games repeated for semantic and cost parity, not independent sample')
    print('REPLAY',rep,flush=True)
    freq=C.Counter(len(z['conditional']['events']) for z in x);assert set(freq)<=set(range(3))
    diag=C.Counter();by_seed=C.defaultdict(C.Counter)
    for z in x:
        g=z['conditional'];e=g['events'];p0=z['pairs']['conditional'];c=by_seed[g['seed']]
        for q in (diag,c):
            q['episodes']+=1;q['first']+=bool(e);q['second']+=len(e)==2;q['delta2']+=p0['delta2'];q['physical_extra']+=p0['local_nn'];q['logical_extra']+=p0['logical_extra']
            if e:
                t=g['rows'][e[0]]['target'];q['first_kept']+=t['incumbent']==t['chosen']
            if len(e)==2:
                t=g['rows'][e[1]]['target'];q['second_switch']+=t['incumbent']!=t['chosen']
    assert diag['second']==freq[2] and diag['first']==freq[1]+freq[2]
    assert diag['second']<=diag['first_kept']

    a=list(rows('R_96'));b=list(rows('R_T'));assert len(a)==len(b)==384
    aa={(g['job']['id'],g['color']):g for g in a};bb={(g['job']['id'],g['color']):g for g in b};assert set(aa)==set(bb)
    idx=rng.integers(0,192,(8000,192));r=dict(arms={'96':game(a,idx)[0],'T':game(b,idx)[0]},paired=pair_stats([rp(aa[k],bb[k]) for k in sorted(aa)],idx),contrast=contrast(a,b,idx)['delta'])
    r['confirmed']=all(r['arms']['T']['gate'].values())
    print('RAPFI',r,flush=True)
    save('SUMMARY.json',dict(nn=out,uncached_replay=rep,conditional_events=dict(freq),conditional_diagnostic=dict(diag),conditional_by_seed={str(k):dict(v) for k,v in by_seed.items()},rapfi=r,parity=load('PARITY_LOCK.json')['counts'],locks={m:sha(D/(m+'_LOCK.json')) for m in ('NN','UNCACHED','R_96','R_T')},scope='8000 family bootstrap, fair cached baseline physical counts, logical costs separately; Rapfi independent confirmation; conditional heuristic exploratory'))
if __name__=='__main__':main()
