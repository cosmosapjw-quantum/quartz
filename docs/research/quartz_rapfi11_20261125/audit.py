"""Independent old-rule reference and actual search/corpus/ledger audit."""
import collections as C,hashlib,importlib.util,json,math,statistics as S,struct,subprocess as SP,sys,tempfile
from pathlib import Path
from publish import ROOT,D,B,BASE,BR,git,sha
from experiment import *
from probe_rapfi import call
OLD=ROOT/'docs/research/quartz_fouraxis06_20261125/rules.py'
sp=importlib.util.spec_from_file_location('old_ref',OLD);ref=importlib.util.module_from_spec(sp);sp.loader.exec_module(ref)
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def eq(x,y,tol=1e-7):assert abs(x-y)<=tol*max(1,abs(x),abs(y)),(x,y)
def key(b,p):
    v=sum(x<<(2*i) for i,x in enumerate(b))|((p==1)<<450)
    return tuple((v>>(64*i))&((1<<64)-1) for i in range(8))
def board_key(h):
    b,p,w=ref.replay(h,'freestyle');return ref.orbit_board(b,p,'freestyle')
def root_rule(a,h):
    b,p,w=ref.replay(h,'freestyle');assert w is None and a['board']==b and a['history']==h and a['player']==p
    assert a['legal']==[i for i,v in enumerate(b) if not v]
    assert a['proof']==ref.proof_reference(b,p,'freestyle')
    return len(a['legal'])
def game(g,h,cap):
    h=list(h);n=nn=act=0
    for r in g['rows']:
        b,p,w=ref.replay(h,'freestyle');assert w is None
        a=r['response'];assert r['ply']==len(h) and r['player']==p and r['move']==a['best']
        check(a,h,'STANDARD_PW',cap,17)
        act+=root_rule(a,h);nn+=a['NN_calls'];n+=1;h.append(r['move'])
    _,_,w=ref.replay(h,'freestyle')
    assert w is not None and h==g['final_history'] and w==g['winner_black']
    return n,nn,act
def independent_reach(r):
    # Reconstruct from move identity, not the prefix-index arithmetic in Rust.
    rank={m:i for i,(m,_) in enumerate(r['candidates'])}
    n={m:0 for m in rank}
    for e in r['edges']:n[e['move']]=e['n']
    assert sum(n.values())==r['root_n']>0
    L=max(n,key=lambda m:(n[m],rank[m]));assert L==r['best']
    # Directly increment each competitor until it becomes leader; small cap.
    ds={}
    for m in n:
        if m==L:continue
        for b in range(n[L]+2):
            if (n[m]+b,rank[m])>(n[L],rank[L]):ds[m]=b;break
        else:raise AssertionError('unreachable')
    return L,ds,min(ds.values(),default=None)
def replay_b(a,p):
    dec=a['decision'];tr=dec['b_trace'];assert len(tr)==p+8
    ledger={};root=C.defaultdict(list);cnt=C.Counter();sample=[]
    phases={z['visits']:z for z in dec['phases']}
    for i,t in enumerate(tr):
        assert t['mode']==(5 if i<p else 3) and t['path']
        hh=list(a['history']);b,pl,w=ref.replay(hh,'freestyle');ks=[]
        for m in t['path']:
            par=key(b,pl);hh.append(m);b,pl,w=ref.replay(hh,'freestyle')
            ks.append((key(b,pl),par,m));cnt['path_edges']+=1
        leaf=f32(t['leaf']);assert math.isfinite(leaf) and -1<=leaf<=1
        if w is not None:assert t['proved'] and leaf==pl*w;cnt['terminal_trace']+=1
        else:
            assert not t['proved']
            if not sample:sample.append((hh,leaf))
        raw=y=leaf;nb=0
        for child,par,m in reversed(ks):
            ctx=ledger.setdefault(child,{});c=(par,m)
            other=sorted((k,v) for k,v in ctx.items() if k!=c)
            s=sum(v[0] for _,v in other);n=sum(v[1] for _,v in other)
            s0,n0=ctx.get(c,(0.,0));ctx[c]=(s0+raw,n0+1)
            if t['mode']==3 and not t['proved'] and n:
                eta=min(.25,n/(n+4.));z=f32(max(-1.,min(1.,(1-eta)*y+eta*s/n)))
                nb+=1;cnt['changed']+=abs(f32(z-y))>1e-7;y=z
            raw=-raw;y=-y
        assert nb==t['blends'];cnt['blends']+=nb;cnt['steps']+=1
        root[t['path'][0]].append(y)
        if i+1 in phases:
            z=phases[i+1];c=z['counters'];r=z['result']
            assert c['ledger_nodes']==len(ledger) and c['ledger_contexts']==sum(len(v) for v in ledger.values())
            assert c['ledger_shared_children']==sum(len(v)>1 for v in ledger.values())
            assert c['ledger_observations']==cnt['path_edges'] and c['merge_blends']==cnt['blends'] and c['backup_changed']==cnt['changed']
            for e in r['edges']:
                vv=root[e['move']];assert e['n']==len(vv);eq(e['w'],sum(vv))
                m=S.mean(vv) if vv else 0.;eq(e['m2'],sum((x-m)**2 for x in vv));cnt['root_moment_checks']+=1
    return cnt,sample
def audit():
    ident=identity();plan=read('C06_SOURCE_PLAN.json');co=read('C06_COHORT.json')
    assert plan['identity']==co['identity']==ident
    src=list(rows('C06_SOURCE_GAMES.jsonl.gz'));assert sha(D/'C06_SOURCE_GAMES.jsonl.gz')==co['source_sha256']
    oldfam,boards=historical();assert len(oldfam)==plan['excluded_families'] and len(boards)==plan['excluded_boards']
    rr={r['id']:r for r in co['roots']};assert len(rr)==24 and not co['missing']
    c=C.Counter();fam=set();selected=set()
    for g,j in zip(src,plan['jobs'],strict=True):
        assert g['job']==j and j['family']==hashlib.sha256(board_key(j['opening'])).hexdigest() and j['family'] not in oldfam|fam
        fam.add(j['family']);n,nn,act=game(g,j['opening'],64);c.update(source_plies=n,source_NN=nn,source_intervals=act)
        candidates=[]
        for row in g['rows']:
            a=row['response'];h=a['history'];b,p,w=ref.replay(h,'freestyle');k=board_key(h);pr=ref.proof_reference(b,p,'freestyle')
            if not pr['own'] and not pr['opponent'] and k not in boards:candidates.append(h)
        candidates=sorted(candidates,key=lambda h:(abs(len(h)-18),len(h)))[:3]
        assert g['probes']
        chosen=None
        for h,z in zip(candidates,g['probes']):
            assert h==z['history'];a=z['response'];check(a,h,'STANDARD_PW',8,17);root_rule(a,h)
            L,ds,md=independent_reach(a['result']);assert z['min_d']==md
            c['selection_probes']+=1
            if j['stratum']=='UNFILTERED' or md<=4:chosen=(h,md);break
        r=rr[j['id']];assert chosen is not None and chosen==(r['history'],r['min_d8']) and g['selected']==r
        assert r['family']==j['family'] and r['group']==j['group'] and r['stratum']==j['stratum']
        assert len(g['probes'])==candidates.index(chosen[0])+1
        k=board_key(r['history']);assert k not in selected;selected.add(k);boards.add(k)
    raw=list(rows('C07_RAW.jsonl.gz'));assert sha(D/'C07_RAW.jsonl.gz')==read('C07_RAW_LOCK.json')['raw_sha256']
    pp={};choices=set();samples=[];metrics={(r['id'],r['prefix'],r['arm']):r for r in read('C08_METRICS.json')}
    for x in raw:
        r=x['root'];p=x['prefix'];a=x['response'];arm=a['arm'];i=r['id'];k=i,p,arm
        assert k not in pp and r==rr[i];pp[k]=a;dscheck(a,r['history'],arm,p)
        c['root_intervals']+=root_rule(a,r['history']);c['visits']+=a['visits'];c['intervention_NN']+=a['NN_calls']
        ps=a['decision']['phases'];pre=ps[0]['result'];L,ds,md=independent_reach(pre)
        # Reconstruct target and actual first four additions by move rather than rank.
        t=a['decision']['target'];old={e['move']:e['n'] for e in pre['edges']}
        if t is not None:
            m=t['move'];after={e['move']:e['n'] for e in ps[1]['result']['edges']}
            assert after[m]-old.get(m,0)==4 and all(v==old.get(j,0)+(4 if j==m else 0) for j,v in after.items())
            if arm=='DSF':assert m!=L and old.get(m,0)>0 and ds[m]<=32-p
            elif arm=='DSX':assert m not in old and ds[m]<=32-p
            else:assert arm=='DSL' and m==L
            c['forced_blocks']+=1;c['forced_visits']+=4
        for z in ps:
            _,dd,md=independent_reach(z['result']);assert z['min_d']==md and z['locked']==(md is None or md>32-z['visits'])
            c['snapshots']+=1;c['distance_checks']+=len(dd)
        if arm=='DSSTOP' and a['decision']['stopped']:c['stops']+=1;c['saved_visits']+=32-a['visits']
        if arm=='DSB':
            ct,samp=replay_b(a,p);c.update(ct);samples+=samp
        moves=recommend(a,p);assert tuple(metrics[k]['moves'])==moves
        choices.update((i,m,cap) for m in moves for cap in (64,256))
    assert len(pp)==24*2*7
    for i in rr:
        for p in (8,16):
            z=pp[i,p,'DS0']
            for arm in ARMS:
                a=pp[i,p,arm];assert a['decision']['phases'][0]['result']==z['decision']['phases'][0]['result'] and a['root_eval']==z['root_eval']
                if arm!='DS0':c['prefix_equalities']+=1
            stop=pp[i,p,'DSSTOP'];assert stop['best']==z['best']
            assert all(x['result']==y['result'] for x,y in zip(stop['decision']['phases'],z['decision']['phases']))
            c['stop_parities']+=1
            assert pp[i,8,'DS0']['result']==pp[i,16,'DS0']['result']
    cont=list(rows('C08_CONTINUATIONS.jsonl.gz'));assert sha(D/'C08_CONTINUATIONS.jsonl.gz')==read('C08_CONT_LOCK.json')['raw_sha256'];u={}
    for g in cont:
        r=g['root'];i=r['id'];m=g['action'];cap=g['cap'];assert r==rr[i] and (i,m,cap) not in u
        n,nn,act=game(g,r['history']+[m],cap);assert g['utility']==(1+r['player']*g['winner_black'])/2
        u[i,m,cap]=g['utility'];c.update(continuation_plies=n,continuation_NN=nn,continuation_intervals=act)
    assert set(u)==choices
    for (i,p,arm),m in metrics.items():
        for cap in (64,256):
            v=[u[i,a,cap] for a in m['moves']];assert v==m['utilities'][str(cap)]
            a=metrics[i,p,'DS0'];assert m['final_gain'][str(cap)]==v[-1]-u[i,a['moves'][-1],cap]
    # One direct nonterminal raw leaf for each B request where available.
    got=call([h for h,_ in samples])
    for z,(h,v) in zip(got,samples,strict=True):
        assert z['history']==h;eq(f32(f32(z['wdl'][0])-f32(z['wdl'][2])),v)
    c['direct_NN_leaf_checks']=len(samples)
    cost=list(rows('C08_COST_RAW.jsonl.gz'));assert sha(D/'C08_COST_RAW.jsonl.gz')==read('C08_COST_LOCK.json')['raw_sha256']
    ck=set()
    for r in cost:
        k=r['id'],r['prefix'],r['rep'],r['arm'];assert k not in ck;ck.add(k)
        a=pp[r['id'],r['prefix'],r['arm']];assert r['best']==a['best'] and r['visits']==a['visits'] and r['nn']==a['NN_calls']
        assert r['tick_hz']>0 and all(r[v]>=0 for v in ('rust_ticks','rapfi_ticks','wall_ns','native_us'))
    c['cost_requests']=len(ck)
    sm=read('C06_SMOKE.json');assert len(sm['rows'])==42 and sm['standard_parity']==4 and len(sm['rejected'])==3
    for x in sm['rows']:
        a=x['response'];p=a['decision']['prefix'] if a['decision'] is not None else 8;dscheck(a,a['history'],a['arm'],p)
    assert sha(B/'mcts_demo')==sha(BIN)
    return dict(identity=ident,source_games=len(src),roots=len(rr),requests=len(raw),continuations=len(cont),**dict(c),scope='old independent rule reference, explicit competitor increment enumeration, B raw path replay, actual policy-conditioned outcomes, no minimax or strength claim')
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
    with tempfile.TemporaryDirectory(prefix='quartz-rapfi11-restore-') as tmp:
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
