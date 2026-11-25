"""Independent legacy rules, exact allocation/cost reconstruction, paired games and restore."""
import collections as C,hashlib,importlib.util,json,math,statistics as S,subprocess as SP,sys,tempfile
from fractions import Fraction
from pathlib import Path
from publish import ROOT,D,B,BASE,BR,git,sha
from experiment import *
from probe_rapfi import call
oldpath=ROOT/'docs/research/quartz_fouraxis06_20261125/rules.py'
spec=importlib.util.spec_from_file_location('old_ref',oldpath);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
cache={}
def old_rule(h):
    k=tuple(h)
    if k not in cache:
        b,p,w=ref.replay(h,'freestyle');cache[k]=(b,p,w,None if w is not None else ref.proof_reference(b,p,'freestyle'))
    return cache[k]
def old_family(h):
    b,p,w=ref.replay(h[:6],'freestyle');assert w is None
    return hashlib.sha256(ref.orbit_board(b,p,'freestyle')).hexdigest()
def independent_distance(r):
    rank={m:i for i,(m,v) in enumerate(r['candidates'])};n={m:0 for m in rank}
    for e in r['edges']:n[e['move']]=e['n']
    L=max(n,key=lambda m:(n[m],rank[m]));assert L==r['best'] and sum(n.values())==r['root_n']>0
    # Select the lexicographically largest rival, rather than evaluate every distance.
    rival=max((m for m in n if m!=L),key=lambda m:(n[m],rank[m]),default=None)
    if rival is None:return None
    delta=n[L]-n[rival]
    d=delta if rank[rival]>rank[L] else delta+1
    assert all((n[j]+d-1,rank[j])<(n[L],rank[L]) for j in n if j!=L)
    assert (n[rival]+d,rank[rival])>(n[L],rank[L])
    return d
def verify(a,h,arm,cap,metered=True):
    b,p,w,pr=old_rule(h)
    assert w is None and a['history']==h and a['board']==b and a['player']==p and a['proof']==pr
    legal=[i for i,v in enumerate(b) if not v];assert a['legal']==legal
    if arm in ARMS:p5check(a,h,arm,cap)
    else:check(a,h,arm,cap,17)
    if metered:
        m=a['measure'];assert all(type(m[k]) is int and m[k]>=0 for k in ('rust_ticks','rapfi_ticks','python_cpu_ns'))
        assert m['tick_hz']==100 and a['transport_ns']>0
    if arm in ARMS and a['resource']:
        r=a['resource'];n=a['visits'];d=independent_distance(a['result']);assert r['final_d']==d
        if r['locked']:
            b=r['selected_cap']-n;assert b>0 and (d is None or d>b)
            # Direct compare: even assigning every remaining visit to any rival
            # cannot beat the leader under actual tie order.
            v=a['result'];rank={m:i for i,(m,_) in enumerate(v['candidates'])}
            counts={m:0 for m in rank}
            counts.update({e['move']:e['n'] for e in v['edges']})
            L=v['best'];assert all((counts[m]+b,rank[m])<(counts[L],rank[L]) for m in counts if m!=L)
    return len(legal)
def game(g,policy):
    h=list(g['job']['opening']);c=C.Counter()
    for r in g['rows']:
        b,p,w,pr=old_rule(h);assert w is None
        arm,cap=policy(h,p);a=r['response']
        assert r['ply']==len(h) and r['player']==p and r['move']==a['best']
        c.update(plies=1,intervals=verify(a,h,arm,cap),NN=a['NN_calls'],visits=a['visits'])
        h.append(r['move'])
    _,_,w,_=old_rule(h);assert w is not None and w==g['winner_black'] and h==g['final_history'] and len(h)<=225
    return c
def audit():
    ident=identity();plan=read('C06_SOURCE_PLAN.json');fit=read('C07_FROZEN_MIX.json');summary=read('C08_SUMMARY.json')
    assert plan['identity']==fit['identity']==summary['identity']==ident
    files=[('C06_SOURCE_GAMES.jsonl.gz','C06_SOURCE_LOCK.json'),('C07_GRID_RAW.jsonl.gz','C07_GRID_LOCK.json'),('C07_CAL_RAW.jsonl.gz','C07_CAL_LOCK.json'),('C08_ARENA_RAW.jsonl.gz','C08_ARENA_LOCK.json')]
    for name,lock in files:
        lk=read(lock);assert lk['identity']==ident and lk['raw_sha256']==sha(D/name)
        if 'fit_sha256' in lk:assert lk['fit_sha256']==sha(D/'C07_FROZEN_MIX.json')
    c=C.Counter();prev,boards=historical()
    assert len(prev)==plan['excluded_families'] and len(boards)==plan['excluded_boards']
    fam=set();jobs={}
    for j in plan['jobs']:
        assert old_family(j['opening'])==j['family'] and j['family'] not in fam|prev
        fam.add(j['family']);jobs[j['id']]=j
    assert C.Counter(j['group'] for j in jobs.values())=={'TRAIN':4,'CAL':4,'TEST':8}
    sources=list(rows(files[0][0]));roots={}
    for g in sources:
        j=g['job'];assert j==jobs[j['id']] and j['group'] in ('TRAIN','CAL')
        z=game(g,lambda h,p:('STANDARD_PW',64));c.update({'source_'+k:v for k,v in z.items()})
        for r in g['rows']:
            a=r['response'];key=j['family'],r['ply']
            assert key not in roots
            roots[key]=dict(id=j['id'],family=j['family'],group=j['group'],history=a['history'],ply=r['ply'],player=r['player'])
    assert len(sources)==8 and len(roots)==113
    train={k:r for k,r in roots.items() if r['group']=='TRAIN'};cal={k:r for k,r in roots.items() if r['group']=='CAL'}
    assert len(train)==59 and len(cal)==54
    grid=list(rows(files[1][0]));gg=C.defaultdict(dict);sums=C.Counter();active_alloc={}
    for v in grid:
        r=v['root'];k=r['family'],r['ply'];a=v['response'];arm=a['arm'];cap=a['cap']
        assert r==train[k] and (arm,cap) not in gg[k]
        gg[k][arm,cap]=a; c['grid_intervals']+=verify(a,r['history'],arm,cap);sums[arm,cap]+=a['transport_ns']
        c['grid_NN']+=a['NN_calls'];c['grid_visits']+=a['visits']
        if arm=='P5ALLOC' and a['resource']:active_alloc[tuple(r['history'])]=a
    expected={('P5PW',128)}|{(a,b) for a in ARMS[1:] for b in GRID}
    for k,d in gg.items():
        assert set(d)==expected
        z=d['P5PW',128]
        for (arm,cap),a in d.items():
            assert a['root_eval']==z['root_eval'];c['grid_root_eval_equalities']+=int((arm,cap)!=('P5PW',128))
            if arm=='P5ALLOC' and a['resource']:assert (a['resource']['prefix_d'],a['resource']['prefix_best'])==(z['resource']['prefix_d'],z['resource']['prefix_best'])
    assert len(grid)==1475 and set(gg)==set(train)
    assert fit['grid_sum_ns']=={f'{a}/{b}':v for (a,b),v in sums.items()} and fit['grid_sha256']==sha(D/files[1][0])
    target=sums['P5PW',128];assert abs(fit['target_wall_ns']-target/len(train))<1e-8
    for arm in ARMS:
        f=fit['fit'][arm];lo,hi=sums[arm,f['low']],sums[arm,f['high']]
        assert lo==f['train_low_sum'] and hi==f['train_high_sum']
        assert f['bracket'] and Fraction(f['den']-f['num'],f['den'])*lo+Fraction(f['num'],f['den'])*hi==target
        if arm!='P5PW':
            vals=[sums[arm,k] for k in GRID]
            assert lo==max(v for v in vals if v<=target) and hi==min(v for v in vals if v>=target)
    def independent_choose(fam,ply,arm):
        x=fit['fit'][arm]
        data=(fit['mix_seed']+'|'+fam+'|'+str(ply)+'|'+arm).encode()
        u=Fraction(int(hashlib.sha256(data).hexdigest()[:16],16),2**64)
        return x['high'] if u<Fraction(x['num'],x['den']) else x['low']
    cr=list(rows(files[2][0]));cc=C.defaultdict(dict)
    for v in cr:
        r=v['root'];k=r['family'],r['ply'];a=v['response'];arm=a['arm']
        assert r==cal[k] and arm not in cc[k];cc[k][arm]=a
        cap=independent_choose(r['family'],r['ply'],arm);c['cal_intervals']+=verify(a,r['history'],arm,cap);c['cal_NN']+=a['NN_calls'];c['cal_visits']+=a['visits']
        if arm=='P5ALLOC' and a['resource']:active_alloc[tuple(r['history'])]=a
    assert len(cr)==216 and set(cc)==set(cal)
    for k,d in cc.items():
        assert set(d)==set(ARMS)
        assert all(a['root_eval']==d['P5PW']['root_eval'] for a in d.values());c['cal_root_eval_equalities']+=3
        if d['P5ALLOC']['resource']:assert d['P5ALLOC']['resource']['prefix_d']==d['P5PW']['resource']['prefix_d']
    # Recompute cost ratios directly from stored elapsed integer sums and tick counts.
    def means(v):
        return (Fraction(sum(a['transport_ns'] for a in v),len(v)),Fraction(sum(a['measure']['rust_ticks']+a['measure']['rapfi_ticks'] for a in v),len(v)))
    b=means([d['P5PW'] for d in cc.values()])
    for arm in ARMS[1:]:
        a=means([d[arm] for d in cc.values()]);z=summary['cal_ratios'][arm]
        assert math.isclose(float(a[0]/b[0]),z['wall_ratio']) and math.isclose(float(a[1]/b[1]),z['cpu_ratio'])
        assert z['wall_match']==(abs(a[0]/b[0]-1)<=Fraction(1,10)) and z['cpu_match']==(abs(a[1]/b[1]-1)<=Fraction(1,5))
    arena=list(rows(files[3][0]));seen=set();by=C.defaultdict(list);parity_samples={}
    for g in arena:
        j=g['job'];arm=g['candidate'];color=g['candidate_color']
        assert j==jobs[j['id']] and j['group']=='TEST' and arm in ARMS[1:] and color in (1,-1)
        k=j['id'],arm,color;assert k not in seen;seen.add(k)
        def policy(h,p):
            a=arm if p==color else 'P5PW';return a,independent_choose(j['family'],len(h),a)
        z=game(g,policy);c.update({'arena_'+k:v for k,v in z.items()})
        assert g['score']==(1+color*g['winner_black'])/2;by[arm].append(g)
        for row in g['rows']:
            a=row['response']
            if a['resource']:
                if a['arm']=='P5ALLOC':active_alloc[tuple(a['history'])]=a
                if a['resource']['locked'] and a['arm'] in ('P5PW','P5FULL','P5ALLOC'):
                    key=a['arm'],j['family']
                    if key not in parity_samples:parity_samples[key]=a
                if a['resource']['locked']:c['arena_certified_stops']+=1;c['arena_saved_visits']+=a['resource']['selected_cap']-a['visits']
    assert seen=={(j['id'],arm,color) for j in jobs.values() if j['group']=='TEST' for arm in ARMS[1:] for color in (1,-1)}
    for arm,gg0 in by.items():
        z=summary['comparisons'][arm];assert len(gg0)==16
        assert z['WDL']==[sum(g['score']==v for g in gg0) for v in (1,.5,0)]
        assert z['score']==S.mean(g['score'] for g in gg0)
        aa=[row['response'] for g in gg0 for row in g['rows'] if row['response']['arm']=='P5PW']
        bb=[row['response'] for g in gg0 for row in g['rows'] if row['response']['arm']==arm]
        x,y=means(aa),means(bb);r=z['arena_ratio']
        assert math.isclose(float(y[0]/x[0]),r['wall_ratio']) and math.isclose(float(y[1]/x[1]),r['cpu_ratio'])
        assert r['wall_match']==(abs(y[0]/x[0]-1)<=Fraction(1,10)) and r['cpu_match']==(abs(y[1]/x[1]-1)<=Fraction(1,5))
        assert z['resource_match']==(summary['host_gate'] and fit['fit'][arm]['bracket'] and all(z[stage][v] for stage in ('arena_ratio','cal_ratio') for v in ('wall_match','cpu_match')))
    # Actual independent prefix checks on every distinct active ALLOC state.
    # Replay a deterministic subset of actual certified stops to the full selected cap.
    parity=[];prefixes=[]
    with Server() as s:
        for h,a in sorted(active_alloc.items()):
            r=s.search(list(h),'STANDARD_PW',16,17);assert r['root_eval']==a['root_eval']
            d=independent_distance(r['result']);assert d==a['resource']['prefix_d'] and r['best']==a['resource']['prefix_best']
            prefixes.append(dict(history=list(h),min_d=d,best=r['best'],root=r['result']))
        for (arm,f),a in sorted(parity_samples.items()):
            cap=a['resource']['selected_cap'];std='STANDARD_FULL' if arm=='P5FULL' else 'STANDARD_PW'
            r=s.search(a['history'],std,cap,17)
            assert r['best']==a['best'] and r['root_eval']==a['root_eval']
            parity.append(dict(arm=arm,family=f,cap=cap,original_visits=a['visits'],history=a['history'],result=r['result']))
    n=D/'C09_PARITY_RAW.json.gz';assert not n.exists()
    n.write_bytes(gzip.compress((json.dumps(dict(prefixes=prefixes,parity=parity),separators=(',',':'),allow_nan=False)+'\n').encode(),mtime=0))
    c['actual_prefix_rechecks']=len(prefixes);c['actual_full_cap_parities']=len(parity)
    sm=read('C06_SMOKE.json');assert len(sm['rows'])==9 and len(sm['rejected'])==3
    for q in sm['rows']:
        for a in q['responses']:verify(a,a['history'],a['arm'],q['cap'],False)
        for a in q['parity']:
            std=a['standard'];assert std['best']==next(x['best'] for x in q['responses'] if x['arm']==a['arm'])
            c['smoke_cap_parities']+=1
    assert sha(B/'mcts_demo')==sha(BIN)
    return dict(identity=ident,source_games=len(sources),train_roots=len(train),cal_roots=len(cal),grid_requests=len(grid),cal_requests=len(cr),arena_games=len(arena),distinct_rule_histories=len(cache),parity_sha256=sha(n),**dict(c),scope='independent older rule reference, exact allocation and mixture reconstruction, actual cap parity and prefix re-search, actual resource and paired outcomes; no global concurrency or equal-time admission')
def restore():
    rec=json.loads((B/'LATEST.json').read_text());head=git('rev-parse','HEAD')
    assert head==rec['head']==git('ls-remote','origin',f'refs/heads/{BR}').split()[0] and git('rev-parse','HEAD^{tree}')==rec['tree']
    bundle=B/rec['bundle'];assert sha(bundle)==rec['sha256']
    receipts=[json.loads(x) for x in (B/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
    chain=git('rev-list','--reverse',f'{BASE}..{head}').splitlines()
    assert chain==[x['head'] for x in receipts]
    clean=[json.loads(x) for x in (B/'CLEANUP.jsonl').read_text().splitlines()]
    assert len(clean)==len(receipts)-1 and [p.name for p in B.glob('*.bundle')]==[bundle.name]
    files=git('diff','--name-only',BASE,head).splitlines()
    def gg(r,*a):return SP.check_output(['git','-C',str(r),*a])
    with tempfile.TemporaryDirectory(prefix='quartz-rapfi12-restore-') as tmp:
        r=Path(tmp)/'r.git';SP.run(['git','init','--bare','-q',str(r)],check=True)
        gg(r,'fetch','--depth=1','--no-tags','file://'+str(ROOT),BASE)
        assert not (r/'objects/info/alternates').exists()
        assert SP.run(['git','-C',str(r),'cat-file','-e',head],capture_output=True).returncode!=0
        gg(r,'bundle','verify',str(bundle));gg(r,'fetch',str(bundle),f'refs/heads/{BR}:refs/heads/restored')
        assert gg(r,'rev-parse','refs/heads/restored').decode().strip()==head
        assert gg(r,'rev-parse','refs/heads/restored^{tree}').decode().strip()==rec['tree']
        assert gg(r,'rev-list','--reverse',f'{BASE}..{head}').decode().splitlines()==chain
        for f in files:assert hashlib.sha256(gg(r,'show',f'{head}:{f}')).hexdigest()==sha(ROOT/f),f
        gg(r,'fsck','--connectivity-only','--no-dangling')
    assert sha(B/'mcts_demo')==sha(BIN)
    return dict(head=head,remote=rec['remote'],tree=rec['tree'],stage_count=len(receipts),cleanup_count=len(clean),restored_files=len(files),external_base=BASE,original_ssot='87c15dc828d6eea519c219fd8a412fa21dfc464d',bundle=bundle.name,bundle_sha256=sha(bundle),bytes=rec['bytes'],actual_restore=True,connectivity_fsck=True,identity=identity())
if __name__=='__main__':print(json.dumps(restore() if len(sys.argv)>1 and sys.argv[1]=='restore' else audit(),sort_keys=True,indent=2,allow_nan=False),flush=True)
