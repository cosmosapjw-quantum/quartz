"""Older independent rule reference and actual exact context backup replay."""
from experiment import *
from publish import B,BASE,BR,git
from probe_rapfi import call
import monitor_ref as mr
import struct,subprocess as SP,tempfile
p=ROOT/'docs/research/quartz_fouraxis06_20261125/rules.py'
sp=importlib.util.spec_from_file_location('old_ref',p);ref=importlib.util.module_from_spec(sp);sp.loader.exec_module(ref)
def key(b,p):return sum(v<<(2*i) for i,v in enumerate(b))|((p==1)<<450)
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
cache={}
def old_rule(h):
    k=tuple(h)
    if k not in cache:
        b,p,w=ref.replay(h,'freestyle');cache[k]=(b,p,w,None if w is not None else ref.proof_reference(b,p,'freestyle'))
    return cache[k]
def verify_root(a,h,arm,cap,seed):
    b,p,w,pr=old_rule(h)
    assert w is None and a['board']==b and a['player']==p and a['proof']==pr
    check(a,h,arm,cap,seed)
    c=a['measure'];assert all(type(c[k]) is int and c[k]>=0 for k in ('rust_cpu_ns','rapfi_cpu_ns','python_cpu_ns'))
    assert len(c['clock_res_s'])==2 and all(0<x<=1e-6 for x in c['clock_res_s'])
    return len(a['legal'])
def game(g,h,cap):
    h=list(h);c=C.Counter()
    for row in g['rows']:
        a=row['response'];b,p,w,pr=old_rule(h)
        assert w is None and row['ply']==len(h) and row['player']==p and row['move']==a['best']
        c.update(plies=1,intervals=verify_root(a,h,'STANDARD_PW',cap,17),NN=a['NN_calls'],visits=a['visits'])
        h.append(row['move'])
    _,_,w,_=old_rule(h);assert w is not None and h==g['final_history'] and w==g['winner_black']
    return c
def replay_backup(a):
    h=a['history'];b0,p0,w,_=old_rule(h);assert w is None
    trace=a['meta']['b_trace'];assert len(trace)==256
    phases=[a['meta']['prefix']]+a['meta']['steps'];ledger={};root=C.defaultdict(list);c=C.Counter();samples=[]
    for i,t in enumerate(trace):
        b=b0.copy();p=p0;winner=None;keys=[];path=t['path'];assert path
        for j,m in enumerate(path):
            assert winner is None and type(m) is int and 0<=m<225 and not b[m]
            par=key(b,p);b[m]=1 if p==1 else 2
            if ref.win_at(b,m,b[m],'freestyle'):winner=p
            elif len(h)+j+1==225:winner=0
            p=-p;keys.append((key(b,p),par,m));c['path_edges']+=1
        leaf=f32(t['leaf']);assert math.isfinite(leaf) and -1<=leaf<=1
        if winner is not None:
            assert t['proved'] and leaf==winner*p;c['terminal_paths']+=1
        elif t['proved']:
            pr=ref.proof_reference(b,p,'freestyle')
            assert (pr['own'] and leaf==1) or (len(pr['actions'])==1 and pr['interval']==[leaf,leaf])
            c['nonterminal_proof_paths']+=1
        elif a['arm']=='MBASE' and not samples:
            samples.append((h+path,leaf))
        raw=y=leaf;nb=0
        for child,par,m in reversed(keys):
            ctx=ledger.setdefault(child,{});k=(par,m)
            # Integer packing independent; sorted context summation matches Rust
            # ordering of little-endian words, not numeric order of packed int.
            words=lambda v:tuple((v>>(64*z))&((1<<64)-1) for z in range(8))
            other=sorted(((k0,v) for k0,v in ctx.items() if k0!=k),key=lambda z:(words(z[0][0]),z[0][1]))
            s=sum(v[0] for _,v in other);n=sum(v[1] for _,v in other)
            s0,n0=ctx.get(k,(0.,0));ctx[k]=(s0+raw,n0+1)
            if t['mode']==3 and not t['proved'] and n:
                eta=min(.25,n/(n+4));z=f32(max(-1.,min(1.,(1-eta)*y+eta*s/n)))
                nb+=1;c['changed']+=abs(f32(z-y))>1e-7;y=z
            raw=-raw;y=-y
        assert nb==t['blends'];c['blends']+=nb;c['paths']+=1
        root[path[0]].append(y)
        if (i+1)%16==0:
            z=phases[i//16];r=z['result'];cc=z['counters']
            assert cc['ledger_nodes']==len(ledger)
            assert cc['ledger_contexts']==sum(len(v) for v in ledger.values())
            assert cc['ledger_shared_children']==sum(len(v)>1 for v in ledger.values())
            assert cc['ledger_observations']==c['path_edges'] and cc['merge_blends']==c['blends'] and cc['backup_changed']==c['changed']
            for e in r['edges']:
                v=root[e['move']];assert e['n']==len(v)
                mu=S.mean(v) if v else 0.
                assert math.isclose(e['w'],sum(v),rel_tol=1e-7,abs_tol=1e-7)
                assert math.isclose(e['m2'],sum((x-mu)**2 for x in v),rel_tol=1e-7,abs_tol=1e-7)
                c['root_moment_checks']+=1
    return c,samples
def audit():
    plan=read('C06_SOURCE_PLAN.json');co=read('C06_COHORT.json');yp=read('C07_YOKE_PLAN.json')
    ident=identity();assert plan['identity']==co['identity']==yp['identity']==ident
    fam,boards=historical();assert len(fam)==plan['excluded_families'] and len(boards)==plan['excluded_boards']
    c=C.Counter();seen=set();rr={x['id']:x for x in co['roots']}
    assert len(rr)==16 and not co['missing']
    source=list(rows('C06_SOURCE_GAMES.jsonl.gz'));assert sha(D/'C06_SOURCE_GAMES.jsonl.gz')==co['source_sha256']
    for g,j in zip(source,plan['jobs'],strict=True):
        assert g['job']==j
        b,p,w=ref.replay(j['opening'],'freestyle')
        f=hashlib.sha256(ref.orbit_board(b,p,'freestyle')).hexdigest()
        assert f==j['family'] and f not in fam|seen;seen.add(f)
        z=game(g,j['opening'],128);c.update({'source_'+k:v for k,v in z.items()})
        elig=[]
        for row in g['rows']:
            a=row['response'];h=a['history'];b,p,w,pr=old_rule(h);k=ref.orbit_board(b,p,'freestyle')
            if not pr['own'] and not pr['opponent'] and k not in boards:elig.append((h,k))
        h,k=min(elig,key=lambda x:(abs(len(x[0])-18),len(x[0])))
        r=rr[j['id']];assert r==g['selected'] and r['history']==h and r['family']==f and r['control_seed']==j['control_seed']
        boards.add(k)
    raw=list(all_searches());assert len(raw)==64
    aa={};samples=[];need=set();modes=C.Counter()
    for x in raw:
        a=x['response'];r=x['root'];i=r['id'];arm=a['arm'];k=i,arm
        assert k not in aa and r==rr[i];aa[k]=a
        c['root_intervals']+=verify_root(a,r['history'],arm,256,r['control_seed'])
        sch=yp['mapping'][i]['schedule'] if arm=='YOKE' else None
        mr.verify(a,r['control_seed'],sch)
        c.update(blocks=len(a['meta']['steps']),visits=a['visits'],NN=a['NN_calls'])
        z,s=replay_backup(a);c.update(z);samples+=s
        need.update((i,m,cap) for m in (a['meta']['prefix']['result']['best'],a['best']) for cap in (128,512))
        for t in a['meta']['steps']:modes[r['group'],arm,t['channel']]+=1
    for i,r in rr.items():
        base=aa[i,'MBASE']
        for arm in ARMS:
            a=aa[i,arm];assert a['root_eval']==base['root_eval'] and a['meta']['prefix']['result']==base['meta']['prefix']['result']
            c['prefix_equalities']+=int(arm!='MBASE')
        donor=yp['mapping'][i]['donor'];assert donor!=i and rr[donor]['group']==r['group']
        assert yp['mapping'][i]['schedule']==[t['channel'] for t in aa[donor,'META']['meta']['steps']]
    for group in ('CAL','TEST'):
        ids=sorted(i for i,r in rr.items() if r['group']==group)
        assert [yp['mapping'][i]['donor'] for i in ids]==ids[1:]+ids[:1]
        for a in range(5):assert modes[group,'META',a]==modes[group,'YOKE',a]
    cont=list(rows('C08_CONT_RAW.jsonl.gz'));assert sha(D/'C08_CONT_RAW.jsonl.gz')==read('C08_CONT_LOCK.json')['raw_sha256']
    u={}
    for g in cont:
        r=g['root'];i=r['id'];k=i,g['action'],g['cap']
        assert r==rr[i] and k not in u and k in need
        z=game(g,r['history']+[g['action']],g['cap']);c.update({'continuation_'+k:v for k,v in z.items()})
        assert g['utility']==(1+r['player']*g['winner_black'])/2;u[k]=g['utility']
    assert set(u)==need
    for x in read('C08_METRICS.json'):
        i=x['id'];a=aa[i,x['arm']]
        assert x['final']==a['best'] and x['prefix']==a['meta']['prefix']['result']['best'] and x['baseline']==aa[i,'MBASE']['best']
        for cap in (128,512):
            v=[u[i,x[k],cap] for k in ('prefix','final','baseline')]
            assert x['utility'][str(cap)]==v and x['gain'][str(cap)]==v[1]-v[2]
    got=call([h for h,v in samples])
    for x,(h,v) in zip(got,samples,strict=True):
        assert x['history']==h and math.isclose(f32(f32(x['wdl'][0])-f32(x['wdl'][2])),v,abs_tol=1e-7)
    c['direct_leaf_NN']=len(samples)
    sm=read('C06_SMOKE.json')
    assert len(sm['rows'])==3 and len(sm['rejected'])==3
    for q in sm['rows']:
        for a in q['rows']:
            verify_root(a,a['history'],a['arm'],256,17);mr.verify(a,17,q['schedule'] if a['arm']=='YOKE' else None)
        if q['standard']:
            assert q['rows'][2]['result']==q['standard']['result'] and q['rows'][0]['result']==q['rows'][3]['result']
            c['smoke_standard_and_yoke_parity']+=2
    assert sha(B/'mcts_demo')==sha(BIN)
    return dict(identity=ident,source_games=len(source),roots=len(rr),requests=len(raw),continuations=len(cont),distinct_rule_histories=len(cache),**dict(c),scope='independent old rule and numerical observer; all actual exact backup paths, terminal/proof checks and root moments, direct NN spot checks, same-prefix and yoke; not whole-tree selection replay or biological/strength validation')
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
    with tempfile.TemporaryDirectory(prefix='quartz-metacog13-restore-') as tmp:
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
