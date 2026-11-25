"""Older independent rule reference and actual exact context backup replay."""
from kinetic_run import *
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
    trace=a['kinetic']['b_trace'];assert len(trace)==96
    phases=[a['kinetic']['prefix']]+a['kinetic']['steps'];ledger={};root=C.defaultdict(list);c=C.Counter();samples=[]
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
        elif a['arm']=='KINETIC' and all(t0['channel']==0 for t0 in a['kinetic']['steps']) and not samples:
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
        if i+1 in (16,48,80,96):
            z=phases[(16,48,80,96).index(i+1)];r=z['result'];cc=z['counters']
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
    import numpy as np
    plan=read('C06_SOURCE_PLAN.json');co=read('C06_COHORT.json');panel=read('C07_PANEL.json');models=read('C08_MODELS.json')
    ident=identity();assert plan['identity']==co['identity']==models['identity']==ident
    assert sha(D/'C06_SOURCE_GAMES.jsonl.gz')==co['source_sha256']
    assert sha(D/'C07_ATLAS_RAW.jsonl.gz')==models['atlas_sha256']==panel['atlas_sha256']
    assert sha(D/'C08_TRAIN_LABELS.jsonl.gz')==models['train_labels_sha256']
    assert sha(D/'C08_MODELS.json')==read('C08_LIVE_LOCK.json')['models_sha256']
    fam,boards=historical();assert len(fam)==plan['excluded_families'] and len(boards)==plan['excluded_boards']
    cc=C.Counter();seen=set();rr={x['id']:x for x in co['roots']}
    source=list(rows('C06_SOURCE_GAMES.jsonl.gz'))
    miss=[]
    for g,j in zip(source,plan['jobs'],strict=True):
        assert g['job']==j
        b,p,w=ref.replay(j['opening'],'freestyle')
        f=hashlib.sha256(ref.orbit_board(b,p,'freestyle')).hexdigest()
        assert f==j['family'] and f not in fam|seen;seen.add(f)
        z=game(g,j['opening'],128);cc.update({'source_'+k:v for k,v in z.items()})
        elig=[]
        for row in g['rows']:
            a=row['response'];h=a['history'];b,p,w,pr=old_rule(h);k=ref.orbit_board(b,p,'freestyle')
            if not pr['own'] and not pr['opponent'] and k not in boards:elig.append((h,k))
        elig=sorted(elig,key=lambda x:(abs(len(x[0])-18),len(x[0])))[:3]
        accepted=None
        for q,(h,k) in zip(g['probes'],elig):
            assert q['history']==h
            a=q['response'];verify_root(a,h,'STANDARD_PW',16,17);cc['source_probes']+=1
            d=distance(a['result']);assert q['distance']==d
            if j['stratum']=='UNFILTERED' or d<=4:
                accepted=dict(id=j['id'],family=f,group=j['group'],stratum=j['stratum'],history=h,player=a['player'],distance16=d)
                assert q is g['probes'][-1];boards.add(k);break
        assert accepted==g['selected']
        if accepted:assert accepted==rr[j['id']]
        else:assert len(g['probes'])==len(elig);miss.append(j['id'])
    assert miss==co['missing'] and len(rr)+len(miss)==len(source)
    raw=atlas_rows();assert len(raw)==len(rr)*25
    aa={};samples=[];need=set()
    for x in raw:
        r=x['root'];a=x['response'];i=r['id'];c,d=x['schedule'];k=i,c,d
        assert k not in aa and r==rr[i];aa[k]=a
        cc['root_intervals']+=verify_root(a,r['history'],'KINETIC',96,17)
        kr.verify(a,[c,d]);cc.update(blocks=3,visits=a['visits'],NN=a['NN_calls'])
        z,s=replay_backup(a);cc.update(z);samples+=s
        need.update((i,m,cap) for m in (a['kinetic']['prefix']['result']['best'],a['kinetic']['steps'][0]['result']['best'],a['best']) for cap in (128,512))
    for i,r in rr.items():
        a0=aa[i,0,0];pre=a0['kinetic']['prefix']['result'];assert distance(pre)==r['distance16']
        for c,d in PAIRS:
            a=aa[i,c,d];assert a['root_eval']==a0['root_eval'] and a['kinetic']['prefix']['result']==pre
            assert a['kinetic']['steps'][0]['result']==aa[i,c,0]['kinetic']['steps'][0]['result']
            cc['prefix_equalities']+=int((c,d)!=(0,0));cc['first_pulse_equalities']+=int(d!=0)
        ca=[z[0] for z in pre['candidates']];n=kr.counts(pre);L=pre['best'];ch=max((k for k,v in enumerate(n) if ca[k]!=L and v>0),key=lambda k:(n[k],k),default=None)
        alt=ca[ch] if ch is not None else next((m for m in ca if m!=L),None)
        hidden=ca[len(pre['edges'])] if len(pre['edges'])<len(ca) else None
        p=panel['panels'][i];assert p['root']==r and p['independent']==[x for x in (alt,hidden) if x is not None]
        need.update((i,m,k) for m in p['independent'] for k in (128,512))
        assert set(p['moves'])=={m for j,m,_ in need if j==i}
    u={};labs=[]
    for part in ('TRAIN','HELDOUT'):
        name='C08_'+part+'_LABELS.jsonl.gz'
        assert sha(D/name)==read('C08_'+part+'_LOCK.json')['raw_sha256']
        for g in rows(name):
            r=g['root'];i=r['id'];k=i,g['action'],g['cap']
            assert r==rr[i] and k not in u and k in need and ((r['group']=='TRAIN')==(part=='TRAIN'))
            z=game(g,r['history']+[g['action']],g['cap']);cc.update({'continuation_'+k:v for k,v in z.items()})
            assert g['utility']==(1+r['player']*g['winner_black'])/2;u[k]=g['utility'];labs.append(g)
    assert set(u)==need
    ids=sorted(i for i,r in rr.items() if r['group']=='TRAIN');assert models['train_families']==ids
    U=lambda i,c,d:S.mean(u[i,aa[i,c,d]['best'],k] for k in (128,512))
    maxres=0.
    for kind,arm in enumerate(ARMS):
        m=models['models'][arm]
        for stage in ('first','second'):
            for c in range(1,5):
                X=[];Y=[];C0=[]
                for i in ids:
                    if stage=='first':
                        X.append(aa[i,0,0]['kinetic']['prefix']['features'][kind])
                        Y.append(S.mean(U(i,c,d)-U(i,0,d) for d in range(5)))
                        C0.append(S.mean(cpu(aa[i,c,d])-cpu(aa[i,0,d]) for d in range(5)))
                    else:
                        for d in range(5):
                            X.append(aa[i,d,0]['kinetic']['steps'][0]['features'][kind]);Y.append(U(i,d,c)-U(i,d,0));C0.append(cpu(aa[i,d,c])-cpu(aa[i,d,0]))
                X=np.array(X)
                for key0,target in (('gain',Y),('cost',C0)):
                    w=np.array(m[stage+'_'+key0][c]);res=float(np.max(np.abs(X.T@(X@w-np.array(target))+4*w)))
                    assert res<1e-9;maxres=max(maxres,res);cc['ridge_normal_equations']+=1
    lv=list(rows('C08_LIVE_RAW.jsonl.gz'));assert sha(D/'C08_LIVE_RAW.jsonl.gz')==read('C08_LIVE_LOCK.json')['raw_sha256']
    seen=set()
    for x in lv:
        r=x['root'];i=r['id'];a=x['response'];arm=a['arm'];c,d=x['schedule'];assert r==rr[i] and r['group']!='TRAIN' and (i,arm) not in seen;seen.add((i,arm))
        verify_root(a,r['history'],arm,96,17);kr.verify(a,model=models['models'][arm])
        b=aa[i,c,d];assert a['result']==b['result'] and a['kinetic']['b_trace']==b['kinetic']['b_trace']
        for x0,y0 in zip([a['kinetic']['prefix']]+a['kinetic']['steps'],[b['kinetic']['prefix']]+b['kinetic']['steps'],strict=True):assert x0['result']==y0['result'] and x0['features']==y0['features']
        cc['live_replay_equalities']+=1
    assert seen=={(i,a) for i,r in rr.items() if r['group']!='TRAIN' for a in ARMS}
    got=call([h for h,v in samples])
    for x,(h,v) in zip(got,samples,strict=True):
        assert x['history']==h and math.isclose(f32(f32(x['wdl'][0])-f32(x['wdl'][2])),v,abs_tol=1e-7)
    cc['direct_leaf_NN']=len(samples)
    sm=read('C06_SMOKE.json');assert len(sm['rows'])==3 and len(sm['rejected'])==4
    for q in sm['rows']:
        for a in q['rows']:
            verify_root(a,a['history'],a['arm'],96,17)
            if a['arm']=='KINETIC':kr.verify(a,[[0,0],[1,3],[3,1]][q['rows'].index(a)])
            else:kr.verify(a,model=q['model'])
        if q['standard']:assert q['rows'][0]['result']==q['standard']['result'] and q['rows'][1]['result']==q['rows'][3]['result'];cc['smoke_parity']+=2
    assert sha(B/'mcts_demo')==sha(BIN)
    return dict(identity=ident,source_games=len(source),roots=len(rr),atlas_requests=len(raw),live_requests=len(lv),continuations=len(labs),distinct_rule_histories=len(cache),ridge_max_residual=maxres,**dict(cc),scope='Independent rule replay, exact backup paths, root moments, native feature/flow and live schedule parity, frozen ridge equation residuals; not an independent full selection-tree implementation')
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
    with tempfile.TemporaryDirectory(prefix='quartz-kinetic14-restore-') as tmp:
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
