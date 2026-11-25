"""Older independent rule reference and actual exact context backup replay."""
from response_run import *
from publish import B,BASE,BR,git
from probe_rapfi import call
import monitor_ref as mr
import response_ref as dr
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
def game(g,h,cap,arm="STANDARD_PW"):
    h=list(h);c=C.Counter()
    for row in g['rows']:
        a=row['response'];b,p,w,pr=old_rule(h)
        assert w is None and row['ply']==len(h) and row['player']==p and row['move']==a['best']
        c.update(plies=1,intervals=verify_root(a,h,arm,cap,17),NN=a['NN_calls'],visits=a['visits'])
        h.append(row['move'])
    _,_,w,_=old_rule(h);assert w is not None and h==g['final_history'] and w==g['winner_black']
    return c
def replay_backup(a):
    h=a['history'];b0,p0,w,_=old_rule(h);assert w is None
    trace=a['exposure']['b_trace'];assert len(trace)==a['visits']
    phases=[a['exposure']['prefix']]+a['exposure']['phases'];ledger={};root=C.defaultdict(list);c=C.Counter();samples=[]
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
        elif a['arm']=='K15N' and a['cap']==32 and not samples:
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
        if i+1 in {z['result']['root_n'] for z in phases}:
            z=next(z for z in phases if z['result']['root_n']==i+1);r=z['result'];cc=z['counters']
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
    plan=read('C06_SOURCE_PLAN.json');co=read('C06_COHORT.json');pan=read('C07_PANEL.json');fit=read('C08_FROZEN.json');summary=read('C08_SUMMARY.json')
    ident=identity();assert plan['identity']==co['identity']==fit['identity']==summary['identity']==ident
    assert sha(D/'C06_SOURCE_RAW.jsonl.gz')==co['source_sha256']
    assert sha(D/'C07_ATLAS_RAW.jsonl.gz')==pan['atlas_sha256']==fit['atlas_sha256']==read('C07_ATLAS_LOCK.json')['raw_sha256']
    assert sha(D/'C08_TRAIN_LABELS.jsonl.gz')==fit['train_labels_sha256']==read('C08_TRAIN_LOCK.json')['raw_sha256']
    cc=C.Counter();fam,boards=historical()
    assert len(fam)==plan['excluded_families'] and len(boards)==plan['excluded_boards']
    rr={r['id']:r for r in co['roots']};miss=[];seen=set()
    src=list(rows('C06_SOURCE_RAW.jsonl.gz'));assert len(src)==48
    for g,j in zip(src,plan['jobs'],strict=True):
        assert g['job']==j
        b,p,w=ref.replay(j['opening'],'freestyle');f=hashlib.sha256(ref.orbit_board(b,p,'freestyle')).hexdigest()
        assert f==j['family'] and f not in fam|seen;seen.add(f)
        z=game(g,j['opening'],128);cc.update({'source_'+k:v for k,v in z.items()})
        elig=[]
        for row in g['rows']:
            a=row['response'];h=a['history'];b,p,w,pr=old_rule(h);k=ref.orbit_board(b,p,'freestyle')
            if not pr['own'] and not pr['opponent'] and k not in boards:elig.append((h,k))
        elig=sorted(elig,key=lambda t:(abs(len(t[0])-14),len(t[0])))[:2 if j['stratum']=='SCREEN' else 1]
        accept=None
        for q,(h,k) in zip(g['attempts'],elig):
            assert q['history']==h;pre=q['prefix'];cc['prefix_intervals']+=verify_root(pre,h,'STANDARD_PW',8,17);cc['prefix_probes']+=1
            r=pre['result'];ca=[x[0] for x in r['candidates']];n=[0]*len(ca);idx={a:i for i,a in enumerate(ca)}
            for e in r['edges']:n[idx[e['move']]]=e['n']
            l=max(range(len(n)),key=lambda t:(n[t],t));rv=max((i for i in range(len(n)) if i!=l and n[i]>0),key=lambda i:(n[i],i),default=None)
            if rv is None:rv=next(i for i in range(len(n)) if i!=l)
            hidden=len(r['edges']) if len(r['edges'])<len(n) else None
            panel=list(dict.fromkeys(ca[i] for i in (l,rv,hidden) if i is not None));assert q['panel']==panel
            if j['stratum']=='SCREEN':
                assert [x['move'] for x in q['screen']]==panel
                for x in q['screen']:
                    z=game(x,h+[x['move']],32);cc.update({'screen_'+k:v for k,v in z.items()});cc['screen_games']+=1
                    assert x['utility']==(1+pre['player']*x['winner_black'])/2
            else:assert not q['screen']
            contrast=len({x['utility'] for x in q['screen']})>1
            accepted=j['stratum']=='UNFILTERED' or contrast
            assert q['contrast']==contrast and q['accepted']==accepted
            if accepted:
                assert q is g['attempts'][-1]
                accept=dict(id=j['id'],group=j['group'],stratum=j['stratum'],family=f,history=h,player=pre['player'],panel=panel);boards.add(k);break
        assert accept==g['selected']
        if accept:assert accept==rr[j['id']]
        else:assert len(g['attempts'])==len(elig);miss.append(j['id'])
    assert miss==co['missing'] and len(rr)+len(miss)==48
    raw=atlas_rows();assert len(raw)==len(rr)*20;aa={};samples=[];needs=C.defaultdict(set)
    for x in raw:
        r=x['root'];i=r['id'];cap=x['cap'];arm=x['arm'];a=x['response'];k=i,cap,arm
        assert k not in aa and r==rr[i];aa[k]=a
        cc['atlas_intervals']+=verify_root(a,r['history'],arm,cap,17)
        dr.verify(a);cc.update(atlas_requests=1,visits=a['visits'],NN=a['NN_calls'],forced=sum(a['exposure']['forced']),blocks=len(a['exposure']['phases']))
        z,s=replay_backup(a);cc.update(z);samples+=s
        needs[i].update(r['panel']);needs[i].update((a['best'],a['exposure']['sub_best']))
    for i,r in rr.items():
        pre=aa[i,32,'K15N']['exposure']['prefix']['result'];assert dr.panel(pre)
        for cap in CAPS:
            for arm in ARMS:
                a=aa[i,cap,arm];assert a['exposure']['prefix']['result']==pre and a['root_eval']==aa[i,32,'K15N']['root_eval'];cc['prefix_equalities']+=1
            a=aa[i,cap,'K15LOCK'];b=aa[i,cap,'K15N'];assert a['best']==b['best'];cc['lock_parity']+=1
            if a['exposure']['locked']:
                cc['locks']+=1;cc['saved_visits']+=cap-a['visits']
                r0=a['result'];ca=[x[0] for x in r0['candidates']];counts={e['move']:e['n'] for e in r0['edges']};l=ca.index(a['best']);b0=cap-a['visits']
                assert all((counts.get(m,0)+b0,j)<(counts.get(a['best'],0),l) for j,m in enumerate(ca) if j!=l)
        assert set(pan['panels'][i]['moves'])==needs[i]
    u={}
    for part in ('TRAIN','HELDOUT'):
        name='C08_'+part+'_LABELS.jsonl.gz';assert sha(D/name)==read('C08_'+part+'_LOCK.json')['raw_sha256']
        for g in rows(name):
            r=g['root'];i=r['id'];m=g['action'];arm=g['arm'];cap=g['cap'];k=i,m,arm
            assert r==rr[i] and m in needs[i] and k not in u and ((r['group']=='TRAIN')==(part=='TRAIN'))
            assert (arm,cap) in (('STANDARD_PW',128),('STANDARD_FULL',256))
            z=game(g,r['history']+[m],cap,arm);cc.update({'continuation_'+k:v for k,v in z.items()});cc['continuations']+=1
            assert g['utility']==(1+r['player']*g['winner_black'])/2;u[k]=g['utility']
    assert set(u)=={(i,m,arm) for i,moves in needs.items() for m in moves for arm in ('STANDARD_PW','STANDARD_FULL')}
    def value(i,m):return (u[i,m,'STANDARD_PW']+u[i,m,'STANDARD_FULL'])/2
    assert fit['train_families']==sorted(i for i,r in rr.items() if r['group']=='TRAIN')
    for g in ('TRAIN','CAL','TEST'):
        ids=sorted(i for i,r in rr.items() if r['group']==g)
        counts=C.Counter();details={}
        for i in ids:
            gain=[];sub=[]
            for cap in CAPS:
                b=value(i,aa[i,cap,'K15N']['best'])
                for arm in ARMS[1:]:
                    a=aa[i,cap,arm];gain.append(value(i,a['best'])-b);sub.append(value(i,a['exposure']['sub_best'])-b)
            t=dict(exposed=any(x!=0 for x in gain),positive=any(x>0 for x in gain),negative=any(x<0 for x in gain),sub_exposed=any(x!=0 for x in sub));counts.update(t);details[i]=t
        z=summary['groups'][g];assert dict(counts)==z['exposure']['counts'] and details==z['exposure']['details']
        if g=='TRAIN':
            gate=counts['exposed']>=6 and counts['positive']>=3 and counts['negative']>=3 and counts['sub_exposed']>=3
            assert fit['gate']['pass_gate']==gate and fit['gate']['details']==details
            assert not gate and fit['models']=={} and fit['model_status']=='NOT_FIT_EXPOSURE_GATE'
        for cap in CAPS:
            for arm in ARMS:
                a=[aa[i,cap,arm] for i in ids];b=[aa[i,cap,'K15N'] for i in ids];x=z['by_cap'][str(cap)]['arms'][arm]
                d=[value(i,p['best'])-value(i,q['best']) for i,p,q in zip(ids,a,b,strict=True)]
                s=[value(i,p['exposure']['sub_best'])-value(i,q['best']) for i,p,q in zip(ids,a,b,strict=True)]
                assert x['gain']['mean']==S.mean(d) and x['sub_gain']['mean']==S.mean(s)
                assert x['gain']['positive']==sum(v>0 for v in d) and x['gain']['negative']==sum(v<0 for v in d)
                assert x['raw_sub_changed']==sum(p['best']!=p['exposure']['sub_best'] for p in a)
                assert x['mean_visits']==S.mean(p['visits'] for p in a) and x['mean_NN']==S.mean(p['NN_calls'] for p in a)
                assert math.isclose(x['cpu_ms'],sum(p['measure']['rust_cpu_ns']+p['measure']['rapfi_cpu_ns'] for p in a)/(1e6*len(a)))
                cc['aggregate_checks']+=1
    got=call([h for h,v in samples])
    for x,(h,v) in zip(got,samples,strict=True):
        assert x['history']==h and math.isclose(f32(f32(x['wdl'][0])-f32(x['wdl'][2])),v,abs_tol=1e-7)
    cc['direct_leaf_NN']=len(samples)
    sm=read('C06_SMOKE.json');assert len(sm['rows'])==6 and len(sm['rejected'])==3
    for q in sm['rows']:
        for a in q['rows']:
            verify_root(a,a['history'],a['arm'],q['cap'],17);dr.verify(a)
        assert q['rows'][0]['result']==q['standard']['result'] and q['rows'][-1]['best']==q['standard']['best'];cc['smoke_parity']+=2
    assert sha(B/'mcts_demo')==sha(BIN)
    return dict(identity=ident,distinct_rule_histories=len(cache),source_games=len(src),roots=len(rr),**dict(cc),scope='Independent older-rule replay and full exact context backup, forced count/flow and lock, selection-screen and frozen outcome/admission reconstruction. Not independent full selection-tree replay or matched-time strength.')
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
    with tempfile.TemporaryDirectory(prefix='quartz-kinetic15-restore-') as tmp:
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
