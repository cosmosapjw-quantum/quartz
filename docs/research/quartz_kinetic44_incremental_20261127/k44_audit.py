"""Independent K43 integer proof verification on K44 records, cost/admission replay.
K43 archived source reconstructed by actual chronological prefix.
"""
import collections as C,gzip,json,math,pathlib,sys,importlib.util
from fractions import Fraction as F
from k44_plan import D,ROOT,T,SEEDS,TIERS,load,save,sha,family
OLD=D.parent/'quartz_kinetic43_partial_20261127'
sys.path.insert(0,str(OLD))
sp=importlib.util.spec_from_file_location('_audit43',OLD/'k43_audit.py')
v=importlib.util.module_from_spec(sp);sp.loader.exec_module(v)
from inc44 import semantic
CT=C.Counter()
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def locked(mode):
    lk=load(mode+'_LOCK.json')
    assert lk['plan']==sha(D/'PLAN.json') and lk['source']==sha(D/'inc44.cpp') and lk['lib']==sha(T/'inc44.so')
    assert lk['engine']==sha(D/'k44_engine.py') and lk['collector']==sha(D/'k44_experiment.py')
    for f in lk['files']:
        p=D/f['file'];assert sha(p)==f['sha256'];rr=list(raw(p))
        assert len(rr)==f['counts']['games'] and sum(g['nn'] for g in rr)==f['counts']['nn']
        assert sum(len(g['rows']) for g in rr)==f['counts']['moves']
        if mode.endswith(('CAL','TEST')):
            assert sum(len(g['atlas']) for g in rr)==f['counts']['atlas_roots']
            assert sum(a['L' if r['arm']=='B' else 'B']['nn'] for g in rr for r,a in zip([r for r in g['rows'] if r['kind']!='CERTIFICATE'],g['atlas']))==f['counts']['atlas_extra_nn']
        yield from rr
def geom_bin(h,n):
    a,b=v.orient(h,n);k,full,ls=v.G[n];e=full^(a|b)
    return int(any(not l&b and (l&e).bit_count()==2 for l in ls))+2*int(any(not l&a and (l&e).bit_count()==2 for l in ls))
def inspect(h,n,z,feature=False):
    assert z['mode']=='F' and z['bin']==geom_bin(h,n) and 0<=z['skipped']<=len(z['bounds'])
    if feature:
        assert z['depth']==3 and z['budget']==0 and z['queries']==0
        a,b=v.orient(h,n);bd,g=v.base(a,b,n);q={m:(l,u) for m,l,u in z['bounds']}
        assert q==bd and z['gate']==g
        opts=[m for m,(l,u) in q.items() if all(l>=x[1] for j,x in q.items() if j!=m)]
        assert z['cert']==(g if g is not None else min(opts) if opts else None)
        CT['feature_checks']+=1;return q
    q=v.inspect(h,n,z);CT['F_receipts']+=1;return q
def vote(h,n,r,lock,pre=None):
    if r['kind']=='PARTIAL44':r=dict(r,kind='PARTIAL43')
    v.vote(h,n,r,lock,pre)
def expected_option(a,d):
    f=a['feature']['cpu_ns'];L=a['L']
    if not d:return dict(cost=L['cpu_ns']+f,nn=L['nn'],cert=0,changed=0,replacement=0,proof_cpu=f,net=-f)
    x=a['tiers'][(3,5,7).index(d)];cert=x['receipt']['cert'] is not None
    c=f+x['cpu_ns']+(0 if cert else L['cpu_ns'])
    return dict(cost=c,nn=0 if cert else L['nn'],cert=int(cert),changed=int(x['move']!=L['move']),replacement=int(x['replacement']),proof_cpu=f+x['cpu_ns'],net=L['cpu_ns']-c)
def check_atlas(a,h,n,verify=True):
    assert a['history']==h;B,L=a['B'],a['L'];assert B['move']==L['move'] and L['nn']<=B['nn']
    if n==15:assert B['native']['history']==L['native']['history']==h
    inspect(h,n,a['feature']['receipt'],True)
    for (d,cap),x in zip(TIERS,a['tiers']):
        z=x['receipt'];assert (z['depth'],z['budget'])==(d,cap)
        q=inspect(h,n,z) if verify else {m:(l,u) for m,l,u in z['bounds']}
        cert=z['cert'] is not None
        mv=z['cert'] if cert else v.changed(L['move'],L['votes'],q,n)
        assert x['move']==mv and x['replacement']==(not cert and mv!=L['move']) and x['nn']==(0 if cert else L['nn'])
        assert x['total_cpu_ns']==x['cpu_ns']+(0 if cert else L['cpu_ns'])
def audit_bench():
    lk=load('BENCHMARK_LOCK.json');p=D/lk['file'];assert sha(p)==lk['sha256'] and lk['binary']==sha(T/'inc44.so')
    # Validate exact archived history and old receipt for all rows; all prior K43 facts already audited.
    br=iter(raw(p));ct=C.Counter();train=C.defaultdict(lambda:C.defaultdict(C.Counter))
    for mode,n,lane in [('NN_SOURCE',7,'NN'),('R_SOURCE',15,'R')]:
        ol=json.loads((OLD/(mode+'_LOCK.json')).read_text())
        for fi in ol['files']:
            path=OLD/fi['file'];assert sha(path)==fi['sha256']
            for g in raw(path):
                ai=iter(g['atlas'])
                for r in g['rows']:
                    if r['kind']=='CERTIFICATE':continue
                    old=next(ai);a=next(br);h=g['history'][:r['ply']]
                    assert a['lane']==lane and a['history']==h and a['seed']==g['seed'] and a['job']==g['job']['id']
                    assert a['B']==old['B'] and a['L']==old['L']
                    o,i,f=(a['runs'][k]['receipt'] for k in ('OLD','I','F'))
                    assert o==old['tiers'][-1]['receipt'] and semantic(o)==semantic(i)
                    if not o['exhausted']:assert all(o[k]==f[k] for k in ('gate','cert','bounds','exhausted'))
                    for q in a['low']:assert semantic(q['old'])==semantic(q['new']);ct['low']+=1
                    assert a['feature']['receipt']['bin']==geom_bin(h,n)
                    for d,cap in TIERS:
                        x=a['tiers'][(3,5,7).index(d)];z=x['receipt']
                        ref=old['tiers'][(3,5,7).index(d)]['receipt']
                        # All same facts already independently checked at K43; reprove only new/different facts.
                        if z['bounds']!=ref['bounds'] or z['cert']!=ref['cert']:inspect(h,n,z);ct['changed_tier_receipt']+=1
                        else:assert z['bin']==geom_bin(h,n);ct['inherited_exact_facts']+=1
                    for d in (0,3,5,7):train[lane][a['feature']['receipt']['bin']][d]+=expected_option(a,d)['net']
                    ct[lane]+=1
                assert next(ai,None) is None
    assert next(br,None) is None and ct['NN']==1130 and ct['R']==648 and ct['low']==448
    fr=load('FROZEN.json');assert fr['train_lock']==sha(D/'BENCHMARK_LOCK.json')
    for lane in ('NN','R'):
        for z in range(4):
            x=train[lane][z];best=max((0,3,5,7),key=lambda d:(x[d],-d))
            if x[best]<=0:best=0
            assert best==fr[lane][str(z)] and {str(d):x[d] for d in (0,3,5,7)}==fr['stats'][lane][str(z)]['net']
    return dict(ct)
def audit_mode(mode):
    v.CT.clear();CT.clear()
    s=load('SUMMARY.json');fr=load('FROZEN.json');ad=load('ADMISSION.json');p=load('PLAN.json')
    jobs={j['id']:j for j in p['jobs']};rjobs={j['id']:j for j in p['rapfi_jobs']}
    total=0;res={}
    n=7 if mode.startswith('NN') else 15;lane=mode.split('_')[0];source=mode.endswith(('CAL','TEST'));jj=jobs if n==7 else rjobs
    cnt=C.Counter();opts={d:C.Counter() for d in ('policy',0,3,5,7)};bins=C.Counter();pairs=C.defaultdict(list);seen=set()
    for g in locked(mode):
        j=g['job'];assert j==jj[j['id']] and j['split']==(mode.split('_')[1] if source else 'ARENA')
        key=j['id'],g['seed'],g['color'];assert key not in seen;seen.add(key)
        h=j['opening'].copy();ai=iter(g['atlas']);nn=0
        for r in g['rows']:
            player=1-2*(len(h)%2);q=player==g['color'];assert r['player']==player and r['ply']==len(h) and r['arm']==('Q' if q else 'B')
            a,b0=v.orient(h,n);bd,gate=v.base(a,b0,n);assert r['move'] in bd
            if source:
                vote(h,n,r,q)
                if gate is None:
                    x=next(ai);assert r==(x['L'] if q else x['B'])
                    vote(h,n,x['B'] if q else x['L'],not q)
                    check_atlas(x,h,n)
                    z=x['feature']['receipt']['bin'];bins[z]+=1
                    cnt['roots']+=1;cnt['B_nn']+=x['B']['nn'];cnt['L_nn']+=x['L']['nn'];cnt['B_cpu']+=x['B']['cpu_ns'];cnt['L_cpu']+=x['L']['cpu_ns']
                    for d in opts:opts[d].update(expected_option(x,fr[lane][str(z)] if d=='policy' else d))
            else:
                z=r.get('partial')
                if q and mode=='R_ADAPT':
                    f=r.get('feature')
                    if ad['R']:
                        assert f is not None;inspect(h,n,f['receipt'],True)
                        alloc=0 if gate is not None else fr['R'][str(f['receipt']['bin'])]
                        assert alloc==r['allocation']
                        assert bool(z)==bool(alloc and gate is None)
                    else:assert r['allocation']==0 and not f and not z
                elif q and mode=='R_FIXED':assert r['allocation']==7 and (z is not None)==(gate is None)
                elif q and mode=='NN_ARENA':assert (z is not None)==(gate is None)
                if z:
                    assert q and gate is None;bounds=inspect(h,n,z)
                    cnt['partial_attempts']+=1;cnt['queries']+=z['queries'];cnt['skipped']+=z['skipped']
                    if z['cert'] is not None:
                        assert r['kind']=='PARTIAL44' and r['move']==z['cert'] and not r['replacement'] and r['original_move'] is None
                        cnt['partial_cert']+=1;vote(h,n,r,True)
                    else:
                        vote(h,n,r,True,r['original_move'])
                        mv=v.changed(r['original_move'],r['votes'],bounds,n);assert mv==r['move'] and r['replacement']==(mv!=r['original_move'])
                        cnt['replacement']+=r['replacement']
                else:vote(h,n,r,q)
                tag='candidate' if q else 'baseline'
                for k in ('nn','cpu_ns','wall_ns'):cnt[tag+'_'+k]+=r[k]
                cnt[tag+'_moves']+=1
                if q:cnt['allocation_'+str(r.get('allocation',7 if z else 0))]+=1
            nn+=r['nn'];cnt['moves']+=1;cnt['common']+=gate is not None;h.append(r['move'])
        assert next(ai,None) is None and h==g['history']
        aa,bb,w=v.replay(h,n);assert w or (aa|bb)==v.G[n][1]
        assert w==g['winner'] and g['score2']==1+g['color']*w and nn==g['nn']
        cnt['games']+=1;cnt['nn']+=nn;cnt['score2']+=g['score2'];cnt['wins']+=g['score2']==2;cnt['draws']+=g['score2']==1;pairs[j['id'],g['seed']].append(g)
    expected=192 if n==7 and source else 24 if source else 512 if n==7 else 96
    assert len(seen)==expected
    for k in ('nn','moves','games'):assert cnt[k]==s['counts'][mode][k]
    total+=cnt['nn']+s['counts'][mode].get('atlas_extra_nn',0)
    if source:
        x=s['source'][mode];assert x['roots']==cnt['roots']
        for k,value in x['counts'].items():assert cnt[k]==value
        assert x['bins']=={str(k):value for k,value in bins.items()}
        for d,c in opts.items():assert dict(c)==x['options'][str(d)],(mode,d)
        if mode.endswith('CAL'):assert x==ad['cal'][mode]
        if mode=='R_CAL':assert ad['R']==(opts['policy']['net']>0)
        mix=x['matched_fixed_mixture']
        if mix:
            lo=opts[mix['low']];hi=opts[mix['high']];p0=mix['p_high']
            assert math.isclose((1-p0)*lo['cost']+p0*hi['cost'],opts['policy']['cost'],abs_tol=1e-4)
            for k in ('cert','nn','changed'):assert math.isclose((1-p0)*lo[k]+p0*hi[k],mix[k],abs_tol=1e-8)
    else:
        x=s['arena'][mode]
        for k in ('games','wins','draws','partial_attempts','partial_cert','replacement','queries','candidate_moves','candidate_nn','baseline_moves','baseline_nn'):assert x[k]==cnt[k],(mode,k)
        assert x['losses']==cnt['games']-cnt['wins']-cnt['draws'] and abs(x['score']-cnt['score2']/(2*cnt['games']))<1e-15
        for k,key in [('nn','NN_reduction'),('cpu_ns','CPU_reduction'),('wall_ns','wall_reduction')]:
            red=1-F(cnt['candidate_'+k]*cnt['baseline_moves'],cnt['baseline_'+k]*cnt['candidate_moves'])
            assert abs(float(red)-x[key])<1e-14
        assert all(len(y)==2 for y in pairs.values()) and x['identical_pairs']==sum(y[0]['history']==y[1]['history'] for y in pairs.values())
    res[mode]=dict(cnt);print('AUDITED',mode,dict(cnt),flush=True)
    return mode,res[mode],dict(v.CT),dict(CT),total

def main():
    s=load('SUMMARY.json');fr=load('FROZEN.json');ad=load('ADMISSION.json');p=load('PLAN.json')
    jobs={j['id']:j for j in p['jobs']};rjobs={j['id']:j for j in p['rapfi_jobs']}
    assert len(jobs)==160 and len(rjobs)==96 and len({family(j['opening']) for j in jobs.values()})==160
    for x in p['prior_locks']:
        path=ROOT/x['path'];assert sha(path)==x['sha256'];prev=json.loads(path.read_text())
        assert not {family(j['opening']) for j in prev['jobs']}&{family(j['opening']) for j in jobs.values()}
    for x in p['rapfi_prior_locks']:assert sha(ROOT/x['path'])==x['sha256']
    sys.path.insert(0,str(D.parent/'quartz_kinetic37_20261126'));import experiment as e
    assert len({e.family(j['opening']) for j in rjobs.values()})==96
    assert not {j['family'] for j in rjobs.values()}&{j['family'] for j in json.loads((OLD/'PLAN.json').read_text())['rapfi_jobs']}
    assert not {j['family'] for j in rjobs.values()}&{j['family'] for j in e.read('PLAN.json')['jobs']}
    b=audit_bench();print('BENCH_AUDIT',b,flush=True)
    total=0;res={}
    base_ct=C.Counter(v.CT);extra=C.Counter(CT)
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    modes=('NN_CAL','R_CAL','NN_TEST','R_TEST','NN_ARENA','R_FIXED','R_ADAPT')
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as ex:
        for mode,r,c,e,n in ex.map(audit_mode,modes):
            res[mode]=r;base_ct.update(c);extra.update(e);total+=n
    out=dict(benchmark=b,counts=dict(base_ct),extra_checks=dict(extra),modes=res,actual_nn=total,scope='all raw legal trajectories, feature bins, new forcing facts, exact action certificates, same-horizon votes and cost/conditional allocation; historical unchanged facts independently checked K43; CIs not independently recomputed')
    save('AUDIT.json',out);print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
