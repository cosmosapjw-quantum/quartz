"""Independent exact-rule, forcing, gain and raw chronology audit."""
import collections as C,gzip,json,math,pathlib,sys,importlib.util,subprocess
from fractions import Fraction
from k45_plan import D,ROOT,T,SEEDS,TIERS,load,save,sha,family
OLD=D.parent/'quartz_kinetic44_incremental_20261127'
K42=D.parent/'quartz_kinetic42_threshold_20261127';K41=D.parent/'quartz_kinetic41_exact_20261126'
V=D.parent/'quartz_kinetic43_partial_20261127'
sys.path.insert(0,str(V));sp=importlib.util.spec_from_file_location('_independent43',V/'k43_audit.py')
v=importlib.util.module_from_spec(sp);sp.loader.exec_module(v)
CT=C.Counter();NEW_EXACT={}
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def rows(mode,d=D):
    lk=json.loads((d/(mode+'_LOCK.json')).read_text())
    if d==D and mode!='TRAIN':
        assert lk['plan']==sha(D/'PLAN.json') and lk['library']==sha(T/'seq45.so') and lk['hybrid']==sha(D/'hybrid45.py')
        assert lk['source']==sha(D/('k45_low.py' if mode=='NN_LOW' else 'k45_experiment.py'))
        assert lk['engine']==sha(D/'k45_engine.py')
    for f in lk['files']:
        p=d/f['file'];assert sha(p)==f['sha256'];rr=list(raw(p))
        if d==D and mode!='TRAIN':
            assert len(rr)==f['counts']['games'] and sum(g['nn'] for g in rr)==f['counts']['nn']
            assert sum(len(g['rows']) for g in rr)==f['counts']['moves']
            if mode.endswith(('CAL','TEST')):
                assert sum(len(g['atlas']) for g in rr)==f['counts']['atlas_roots']
                assert sum(a['L']['32']['nn']+(a['B']['nn'] if r['arm']=='Q' else 0) for g in rr for r,a in zip([r for r in g['rows'] if r['kind']!='CERTIFICATE'],g['atlas']))==f['counts']['atlas_extra_nn']
        yield from rr
def bounds(h,n,z,prior=None,truth=None,late=False):
    a,b=v.orient(h,n);base,gate=v.base(a,b,n);prev=None
    rr=z['seq']['stages'];assert 1<=len(rr)<=3 and z['seq']['receipt']==rr[-1]
    for i,r in enumerate(rr):
        assert (r['depth'],r['budget'])==TIERS[i] and r['mode']=='SEQ' and r['gate']==gate
        assert r['queries']<=r['budget'] and r['scans']>=len(v.G[n][2])
        if r['exhausted']:assert r['queries']==r['budget']
        assert r['entries']<=r['queries'] and r['positive']+r['negative']>=r['entries']
        k,full,ls=v.G[n];e=full^(a|b)
        bn=int(any(not l&b and (l&e).bit_count()==2 for l in ls))+2*int(any(not l&a and (l&e).bit_count()==2 for l in ls))
        assert r['bin']==bn
        if prev:
            assert all(x[0]==y[0] and x[1]<=y[1]<=y[2]<=x[2] for x,y in zip(prev['bounds'],r['bounds']))
            for k in ('queries','scans','checks','skipped','hits','entries','positive','negative'):assert r[k]>=prev[k]
            assert prev['cert'] is None and prev['bin']!=0
        if prior is not None and r['bounds']==prior[i]['receipt']['bounds'] and r['cert']==prior[i]['receipt']['cert']:
            CT['inherited_stage_facts']+=1
        else:
            v.inspect(h,n,r);CT['independent_stages']+=1
        CT['stages']+=1;prev=r
    r=rr[-1];assert r['cert'] is not None or r['bin']==0 or len(rr)==3
    bd={m:(l,u) for m,l,u in r['bounds']};cert=r['cert'];ex=z['exact']
    assert (ex is not None)==(n==7 and 33<=len(h)<49 and cert is None)
    if ex:
        CT['exact_attempt']+=1;assert ex['nodes']<=25000
        if ex['complete']:
            ct={m:(l,u) for m,l,u in ex['bounds']};c=ex['best'];value=ex['value']
            assert set(ct)==set(bd) and ct[c]==(value,value)
            assert all(u<=value and (m>=c or u<value) for m,(l,u) in ct.items())
            cert=c
            for m,(l,u) in ct.items():
                x,y=bd[m];bd[m]=(max(l,x),min(u,y));assert bd[m][0]<=bd[m][1]
            CT['exact_complete']+=1
        else:assert 'best' not in ex and 'bounds' not in ex and ex['nodes']==25000
        if truth is None and not late:NEW_EXACT.setdefault((a,b),[]).append(z)
    assert z['bounds']==[[m,l,u] for m,(l,u) in bd.items()] and z['cert']==cert
    if cert is not None:assert all(bd[cert][0]>=u for m,(l,u) in bd.items() if m!=cert)
    if truth is not None:
        assert all(l<=truth[str(m)]<=u for m,(l,u) in bd.items())
        if cert is not None:assert truth[str(cert)]==max(truth.values())
    assert z['cpu_ns']==z['seq']['cpu_ns']+(ex['cpu_ns'] if ex else 0)
    CT['bundles']+=1;CT['cert']+=cert is not None;CT['queries']+=r['queries'];CT['hits']+=r['hits']
    return bd
def checkvote(h,n,r,lock,pre=None,cap=32):
    if n==15:
        v.vote(h,n,dict(r,kind='PARTIAL43') if r['kind']=='PARTIAL45' else r,lock,pre);return
    a,b=v.orient(h,n);bd,g=v.base(a,b,n);E=list(bd)
    assert r['nn']>=0 and r['visits']>=0 and r['cpu_ns']>=0 and r['wall_ns']>=0 and r['cap']==cap and r['lock']==lock
    if g is not None:
        assert r['kind']=='CERTIFICATE' and r['move']==g and r['nn']==r['visits']==0 and not r['votes'];CT['common']+=1;return
    if r['kind']=='PARTIAL45':
        assert lock and r['nn']==r['visits']==0 and not r['votes'];return
    vs=r['votes'];assert [m for m,x in vs]==E and all(type(x) is int and x>=0 for m,x in vs)
    lead=min(E,key=lambda m:(-dict(vs)[m],m));assert (r['move'] if pre is None else pre)==lead
    d=min((dict(vs)[lead]-x+int(m>lead) for m,x in vs if m!=lead),default=10**9)
    assert sum(x for m,x in vs)==r['visits']<=cap and r['nn']==1+r['visits']-r['terminal_leaves'] and r['distance']==d
    if r['kind']=='VOTE_LOCK':assert lock and d>cap-r['visits'] and r['visits']<cap
    else:assert r['kind']=='CAP' and r['visits']==cap
    CT['vote_rows']+=1;CT['legal_edges']+=len(vs)
def option(a,h,bd,verify_votes=True):
    B=a['B'];base=B['move'];L=a['L'];z=a['proof'];cert=z['cert']
    assert a['history']==h and L['32']['move']==base and L['20']['nn']<=L['26']['nn']<=L['32']['nn']<=B['nn']
    if verify_votes:checkvote(h,7,B,False)
    for k,r in L.items():
        if verify_votes:checkvote(h,7,r,True,cap=int(k))
        mv=cert if cert is not None else v.changed(r['move'],r['votes'],bd,7)
        if mv==base:lo=hi=0
        else:
            l,u=bd[mv];p,q=bd[base];lo=l-q;hi=u-p
            if cert is not None:lo=max(lo,0)
            if all(p>=uu for m,(ll,uu) in bd.items() if m!=base):hi=min(hi,0)
        x=a['options'][k];assert x['move']==mv and x['gain2']==[lo,hi] and lo<=hi
        assert x['nn']==(0 if cert is not None else r['nn']) and x['replacement']==(cert is None and mv!=r['move']) and x['cert']==(cert is not None)
        assert x['cost_cpu_ns']==z['cpu_ns']+(0 if cert is not None else r['cpu_ns'])
        if k=='32':assert lo>=0
        CT['gain_checks']+=1
def agg(rr):
    c=C.Counter();vv={str(h):C.Counter() for h in (20,26,32)}
    for a in rr:
        c['roots']+=1;c['B_nn']+=a['B']['nn'];c['B_cpu_ns']+=a['B']['cpu_ns']
        for k,x in a['options'].items():
            y=vv[k];lo,hi=x['gain2'];y['nn']+=x['nn'];y['gain2_lo']+=lo;y['gain2_hi']+=hi;y['cert']+=x['cert'];y['replacement']+=x['replacement'];y['changes']+=x['move']!=a['B']['move'];y['cost_cpu_ns']+=x['cost_cpu_ns'];y['exact_gain']+=lo==hi;y['strict_positive']+=lo>0;y['strict_negative']+=hi<0
    return dict(counts=dict(c),options={k:dict(y) for k,y in vv.items()})
def audit_bench():
    p=D/'BENCHMARK.jsonl.gz';lk=load('BENCHMARK_LOCK.json');assert sha(p)==lk['sha256'];it=iter(raw(p));ct=C.Counter();hist={}
    for lane,n in [('NN',7),('R',15)]:
        for split in ('CAL','TEST'):
            for g in rows(lane+'_'+split,OLD):
                for a in g['atlas']:
                    r=next(it);h=a['history'];assert r['history']==h and r['lane']==lane and r['job']==g['job']['id'] and r['seed']==g['seed'] and r['split']==split
                    assert r['B']==a['B'] and r['L']==a['L'] and r['cold']['receipt']==a['tiers'][-1]['receipt']
                    bd=bounds(h,n,r['proof'],a['tiers']);ct[lane]+=1
                    if n==7:hist[(r['split'],r['job'],r['seed'],tuple(h))]=r
    assert next(it,None) is None and ct=={'NN':1709,'R':772}
    tr=list(rows('TRAIN'));assert len(tr)==1709
    for a in tr:
        r=hist.pop((a['split'],a['job'],a['seed'],tuple(a['history'])))
        assert a['proof']==r['proof'] and a['B']==r['B'] and a['L']['32']['nn']==r['L']['nn']
        option(a,a['history'],{m:(l,u) for m,l,u in a['proof']['bounds']})
    assert not hist
    fr=load('FROZEN.json');x=agg(tr);assert fr['train']==x
    vv=x['options'];cap=next(h for h in (20,26,32) if vv[str(h)]['gain2_lo']>=0 and vv[str(h)]['nn']<=vv['32']['nn']);assert cap==fr['cap']
    print('BENCH_TRAIN_AUDIT',dict(ct),dict(CT),flush=True)
    return dict(ct)
def audit_late():
    p=D/'LATE.jsonl.gz';lk=load('LATE_LOCK.json');assert sha(p)==lk['sha256']
    for path,digest in lk['truth_sources'].items():assert sha(ROOT/path)==digest
    old={(r['a'],r['b']):r['values'] for r in raw(K41/'INDEPENDENT.jsonl.gz') if r.get('complete') and ('corpus_complete' in r['tags'] or 'corpus_censored' in r['tags'])}
    new={(r['a'],r['b']):r['values'] for r in raw(K42/'INDEPENDENT.jsonl.gz') if r.get('complete') and 'fresh' in r['tags']}
    ct=C.Counter()
    for r,j in zip(raw(p),raw(K42/'BENCHMARK.jsonl.gz'),strict=True):
        assert r['job']==j['job'] and (r['a'],r['b'])==(j['a'],j['b'])
        truth=(old if j['job']['frame']=='K41' else new).get((r['a'],r['b']));assert r['truth']==truth
        h=r['job']['opening'];z=r['proof'];bounds(h,7,z,truth=truth,late=True)
        ct['roots']+=1;ct['truth']+=truth is not None;ct['partial']+=z['seq']['receipt']['cert'] is not None;ct['hybrid']+=z['cert'] is not None
        if z['exact']:ct['exact_attempt']+=1;ct['exact_complete']+=z['exact']['complete'];ct['exact_nodes']+=z['exact']['nodes']
        if truth is not None and z['cert'] is not None:ct['verified_hybrid']+=1
    assert dict(ct)==lk['counts'];return dict(ct)
def audit_mode(mode):
    CT.clear();v.CT.clear();NEW_EXACT.clear()
    n=7 if mode.startswith('NN') else 15;source=mode.endswith(('CAL','TEST'))
    cap=20 if mode=='NN_LOW' else load('ADMISSION.json')['cap'] if mode=='NN_ADMIT' else 32
    pl=load('PLAN.json');jj={j['id']:j for j in pl['jobs' if n==7 else 'rapfi_jobs']}
    ct=C.Counter();seen=set();aa=[];pairs=C.defaultdict(list)
    for g in rows(mode):
        j=g['job'];assert j==jj[j['id']] and j['split']==(mode.split('_')[1] if source else 'ARENA')
        k=j['id'],g['seed'],g['color'];assert k not in seen;seen.add(k)
        h=j['opening'].copy();it=iter(g['atlas']);nn=0
        for r in g['rows']:
            side=1-2*(len(h)%2);q=side==g['color'];a,b=v.orient(h,n);bd,g0=v.base(a,b,n)
            assert r['move'] in bd and r['ply']==len(h) and r['player']==side and r['arm']==('Q' if q else 'B')
            if source:
                checkvote(h,n,r,q)
                if g0 is None:
                    a0=next(it);assert a0['history']==h and (r['move'],r['nn'])==(a0['L']['32']['move'],a0['L']['32']['nn']) if q else a0['B']==r
                    bds=bounds(h,n,a0['proof']);option(a0,h,bds);aa.append(a0)
            else:
                z=r.get('partial');assert bool(z)==bool(q and g0 is None)
                if z:
                    bds=bounds(h,n,z);r0=z['seq']['receipt'];ct['attempts']+=1;ct['queries']+=r0['queries'];ct['hits']+=r0['hits']
                    ct['exact_attempt']+=z['exact'] is not None;ct['exact_complete']+=bool(z['exact'] and z['exact']['complete'])
                    if z['cert'] is not None:
                        assert r['kind']=='PARTIAL45' and r['move']==z['cert'] and r['original_move'] is None and not r['replacement'];ct['cert']+=1;checkvote(h,n,r,True,cap=cap)
                    else:
                        checkvote(h,n,r,True,r['original_move'],cap=cap)
                        mv=v.changed(r['original_move'],r['votes'],bds,n);assert r['move']==mv and r['replacement']==(mv!=r['original_move']);ct['replacement']+=r['replacement']
                else:checkvote(h,n,r,q,cap=cap if q else 32)
                tag='candidate' if q else 'baseline'
                for k in ('nn','cpu_ns','wall_ns'):ct[tag+'_'+k]+=r[k]
                ct[tag+'_moves']+=1
            ct['moves']+=1;nn+=r['nn'];h.append(r['move'])
        assert next(it,None) is None and h==g['history']
        a,b,w=v.replay(h,n);assert w or (a|b)==v.G[n][1]
        assert w==g['winner'] and g['score2']==1+g['color']*w and nn==g['nn']
        ct['games']+=1;ct['nn']+=nn;ct['score2']+=g['score2'];ct['wins']+=g['score2']==2;ct['draws']+=g['score2']==1;pairs[j['id'],g['seed']].append(g)
    assert len(seen)==(160 if source else 384 if n==7 else 96)
    lk=load(mode+'_LOCK.json')
    for k in ('games','nn','moves'):assert ct[k]==lk['counts'][k]
    summary=load('LOW_SUMMARY.json') if mode=='NN_LOW' else load('SUMMARY.json')['arena'].get(mode)
    if source:
        x=agg(aa);assert x==load('SUMMARY.json')['source'][mode]
        if mode=='NN_CAL':
            ad=load('ADMISSION.json');f=load('FROZEN.json')['cap'];v0=x['options'];ok=v0[str(f)]['gain2_lo']>=0 and v0[str(f)]['nn']<=v0['32']['nn']
            assert ad['cal']==x and ad['selected']==f and ad['admitted']==ok and ad['cap']==(f if ok else 32)
    else:
        for k in ('games','wins','draws','attempts','cert','replacement','queries','hits','exact_attempt','exact_complete','candidate_moves','candidate_nn','baseline_moves','baseline_nn'):assert summary[k]==ct[k],(mode,k)
        assert summary['losses']==ct['games']-ct['wins']-ct['draws'] and summary['score']==ct['score2']/(2*ct['games'])
        for k,k2 in [('nn','NN_reduction'),('cpu_ns','CPU_reduction'),('wall_ns','wall_reduction')]:
            red=1-Fraction(ct['candidate_'+k]*ct['baseline_moves'],ct['baseline_'+k]*ct['candidate_moves']);assert abs(float(red)-summary[k2])<1e-14
        assert all(len(z)==2 for z in pairs.values())
    print('AUDITED',mode,dict(ct),flush=True)
    return mode,dict(ct),dict(CT),dict(v.CT),[(k,x) for k,x in NEW_EXACT.items()]
def main():
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    plan=load('PLAN.json');assert len(plan['jobs'])==128 and len(plan['rapfi_jobs'])==48
    fam={family(j['opening']) for j in plan['jobs']};assert len(fam)==128
    for x in plan['prior_locks']:
        p=ROOT/x['path'];assert sha(p)==x['sha256'];o=json.loads(p.read_text());assert not fam&{family(j['opening']) for j in o['jobs']}
    for x in plan['rapfi_prior_locks']:assert sha(ROOT/x['path'])==x['sha256']
    b=audit_bench();late=audit_late();ct=C.Counter(CT);ind=C.Counter(v.CT);ne=C.defaultdict(list,NEW_EXACT);res={}
    modes=['NN_CAL','NN_TEST','NN_FIXED','R_ARENA']+(['NN_ADMIT'] if (D/'NN_ADMIT_LOCK.json').exists() else [])+(['NN_LOW'] if (D/'NN_LOW_LOCK.json').exists() else [])
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as ex:
        for m,r,c,vv,ee in ex.map(audit_mode,modes):
            ct.update(c);ind.update(vv);res[m]=r
            for k,x in ee:ne[k].extend(x)
    truth_ct=C.Counter()
    if ne:
        binary=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic41_exact_backups/independent'
        assert sha(binary)=='a1e943f250eae145e10a12482fc575d11af08960dfd7ddf804c1f0c0ff14c7af'
        keys=list(ne);inp=''.join(f'{i} {a} {b}\n' for i,(a,b) in enumerate(keys))
        p=subprocess.run([str(binary)],input=inp,text=True,capture_output=True,check=True,timeout=120);rs=list(map(json.loads,p.stdout.splitlines()));assert len(rs)==len(keys)
        for r in rs:
            k=keys[r['id']];truth_ct['roots']+=1;truth_ct['complete']+=r['complete']
            if r['complete']:
                for z in ne[k]:
                    vv=r['values'];assert all(l<=vv[str(m)]<=u for m,l,u in z['bounds'])
                    if z['cert'] is not None:assert vv[str(z['cert'])]==max(vv.values())
        save('NATURAL_EXACT_AUDIT.json',dict(counts=dict(truth_ct),rows=[dict(a=keys[r['id']][0],b=keys[r['id']][1],**r) for r in rs]))
    total=load('TRAIN_LOCK.json')['counts']['nn']+sum(load(m+'_LOCK.json')['counts']['nn']+load(m+'_LOCK.json')['counts'].get('atlas_extra_nn',0) for m in modes)
    out=dict(bench=b,late=late,counts=dict(ct),independent=dict(ind),modes=res,physical_nn=total,natural_exact=dict(truth_ct),scope='all chronological legal games, all new proof facts checked by independent K43 integer geometry and forcing; historical identical stage facts reused audited K44; exact late labels independently known; signed gain and cost reconstruction; bootstrap not independently recomputed')
    save('AUDIT.json',out);print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
