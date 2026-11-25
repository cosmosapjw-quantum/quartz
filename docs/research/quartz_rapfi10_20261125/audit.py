"""Independent original rule replay and exact B path/state/context re-aggregation."""
import collections as C,hashlib,importlib.util,json,math,os,statistics as S,struct,subprocess as SP,sys,tempfile
from pathlib import Path
from publish import ROOT,D,B,BASE,BR,git,sha
from experiment import rows,read,identity,axischeck,historical,family,ARMS,MODES
from arena import check,BIN
from probe_rapfi import EXE,W,call
OLD=ROOT/'docs/research/quartz_fouraxis06_20261125/rules.py'
sp=importlib.util.spec_from_file_location('old_ref',OLD);ref=importlib.util.module_from_spec(sp);sp.loader.exec_module(ref)
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def key(b,p):
    k=sum(v<<(2*i) for i,v in enumerate(b))|((p==1)<<450)
    return tuple((k>>(64*i))&((1<<64)-1) for i in range(8))
def eq(x,y,tol=1e-6):assert abs(x-y)<=tol*max(1,abs(x),abs(y)),(x,y)
def root_rule(a,h):
    b,p,w=ref.replay(h,'freestyle');assert w is None
    assert a['history']==h and a['board']==b and a['player']==p
    assert a['legal']==[i for i,v in enumerate(b) if v==0]
    assert a['proof']==ref.proof_reference(b,p,'freestyle')
    return len(a['legal'])
def game(g,h):
    h=list(h);n=nn=act=0
    for r in g['rows']:
        b,p,w=ref.replay(h,'freestyle');assert w is None
        a=r['response'];assert r['ply']==len(h) and r['player']==p and r['move']==a['best']
        assert a['arm']=='STANDARD_PW' and a['cap']==64
        check(a,h,'STANDARD_PW',64,17)
        act+=root_rule(a,h);nn+=a['NN_calls'];n+=1
        h.append(r['move'])
    _,_,w=ref.replay(h,'freestyle');assert w is not None and h==g['final_history'] and w==g['winner_black']
    return n,nn,act
def replay_b(a):
    """Exact raw context stats, then reconstruct reverse backed y sample path."""
    h=a['history'];ax=a['axis'];tr=ax['b_trace'];assert len(tr)==64
    ledger={};root=C.defaultdict(list);cnt=C.Counter();sample=[]
    for i,t in enumerate(tr):
        assert t['mode']==(5 if i<32 else 3) and t['path']
        b,p,w=ref.replay(h,'freestyle');assert w is None
        hh=list(h);keys=[]
        for m in t['path']:
            k0=key(b,p)
            hh.append(m);b,p,w=ref.replay(hh,'freestyle')
            keys.append((key(b,p),k0,m));cnt['path_edges']+=1
        leaf=f32(t['leaf']);assert math.isfinite(leaf) and -1<=leaf<=1
        if w is not None:assert t['proved'] and leaf==w*p;cnt['terminal_trace']+=1
        elif t['proved']:
            # V3 certificate based on all legal terminal children or immediate win.
            pr=ref.proof_reference(b,p,'freestyle')
            assert pr['own'] and leaf==1 or len(pr['actions'])==1 and pr['interval']==[leaf,leaf]
            cnt['proved_nonterminal_trace']+=1
        elif a['arm']=='AXB' and len(sample)<1 and i%7==0:sample.append((hh,leaf))
        raw=y=leaf;nb=0
        for child,parent,m in reversed(keys):
            contexts=ledger.setdefault(child,{})
            c=(parent,m);other=sorted((k,v) for k,v in contexts.items() if k!=c)
            s=sum(v[0] for _,v in other);n=sum(v[1] for _,v in other)
            s0,n0=contexts.get(c,(0.,0));contexts[c]=(s0+raw,n0+1)
            if t['mode']==3 and not t['proved'] and n:
                eta=min(.25,n/(n+4.));z=f32(max(-1.,min(1.,(1-eta)*y+eta*s/n)))
                cnt['changed']+=abs(f32(z-y))>1e-7;nb+=1;y=z
            raw=-raw;y=-y
        assert nb==t['blends'];cnt['blends']+=nb;cnt['steps']+=1
        root[t['path'][0]].append(y)
        if i in (31,63):
            ph=ax['phases'][i//32];c=ph['counters'];r=ph['result']
            assert c['ledger_nodes']==len(ledger)
            assert c['ledger_contexts']==sum(len(v) for v in ledger.values())
            assert c['ledger_shared_children']==sum(len(v)>1 for v in ledger.values())
            assert c['ledger_observations']==sum(n for v in ledger.values() for s,n in v.values())==cnt['path_edges']
            assert c['merge_blends']==cnt['blends'] and c['backup_changed']==cnt['changed']
            for e in r['edges']:
                vv=root[e['move']];assert e['n']==len(vv);eq(e['w'],sum(vv),1e-7)
                m=S.mean(vv) if vv else 0.;eq(e['m2'],sum((x-m)**2 for x in vv),1e-7)
                cnt['root_edge_moment_checks']+=1
            eq(r['root_w'],sum(sum(v) for v in root.values()),1e-7)
    return cnt,sample
def audit():
    ident=identity();plan=read('C06_SOURCE_PLAN.json');co=read('C06_COHORT.json');assert plan['identity']==co['identity']==ident
    source=list(rows('C06_SOURCE_GAMES.jsonl.gz'));assert sha(D/'C06_SOURCE_GAMES.jsonl.gz')==co['source_sha256']
    oldfam,boards=historical();assert len(oldfam)==plan['excluded_families'] and len(boards)==plan['excluded_boards']
    rr={r['id']:r for r in co['roots']};assert len(rr)==16 and not co['missing']
    count=C.Counter();sample=[];fam=set();roots_seen=set()
    for g,j in zip(source,plan['jobs'],strict=True):
        assert g['job']==j and family(j['opening'])==j['family'] and j['family'] not in oldfam|fam
        fam.add(j['family'])
        n,nn,act=game(g,j['opening']);count['source_plies']+=n;count['source_NN']+=nn;count['source_action_intervals']+=act
        elig=[]
        for row in g['rows']:
            a=row['response'];b,p,w=ref.replay(a['history'],'freestyle');k=ref.orbit_board(b,p,'freestyle')
            pr=ref.proof_reference(b,p,'freestyle')
            if not pr['own'] and not pr['opponent'] and k not in boards:elig.append((a,k))
        desired=[v for v in elig if v[0]['player']==j['desired_player']]
        a,k=min(desired or elig,key=lambda v:(abs(len(v[0]['history'])-18),len(v[0]['history'])))
        r=rr[j['id']];assert r['history']==a['history'] and r['family']==j['family'] and r['group']==j['group'] and r['player']==a['player']
        assert k not in roots_seen;boards.add(k);roots_seen.add(k)
    raw=list(rows('C07_RAW.jsonl.gz'));assert sha(D/'C07_RAW.jsonl.gz')==read('C07_RAW_LOCK.json')['raw_sha256']
    p={};choices={}
    metrics={(x['id'],x['arm']):x for x in read('C08_METRICS.json')}
    for r in raw:
        root=r['root'];a=r['response'];i=root['id'];arm=a['arm'];assert root==rr[i] and (i,arm) not in p
        axischeck(a,root['history'],arm);count['root_action_intervals']+=root_rule(a,root['history'])
        count['intervention_NN']+=a['NN_calls'];count['intervention_visits']+=a['visits']
        p[i,arm]=a
        if arm in ('AXB','AXALL'):
            c,s=replay_b(a);count.update(c);sample+=s
        ch=[x['result']['best'] for x in a['axis']['phases']]
        choices.setdefault(i,set()).update(ch)
        assert ch==metrics[i,arm]['moves']
    for i in rr:
        aa=[p[i,arm] for arm in ARMS]
        assert all(a['axis']['phases'][0]['result']==aa[0]['axis']['phases'][0]['result'] and a['root_eval']==aa[0]['root_eval'] for a in aa)
        count['prefix_equalities']+=5
    assert len(p)==96
    cont=list(rows('C08_CONTINUATIONS.jsonl.gz'));assert sha(D/'C08_CONTINUATIONS.jsonl.gz')==read('C08_CONT_LOCK.json')['raw_sha256']
    u={}
    for g in cont:
        r=g['root'];i=r['id'];m=g['action'];assert r==rr[i] and (i,m) not in u and m in choices[i]
        n,nn,act=game(g,r['history']+[m])
        assert g['utility']==(1+r['player']*g['winner_black'])/2;u[i,m]=g['utility']
        count['continuation_plies']+=n;count['continuation_NN']+=nn;count['continuation_action_intervals']+=act
    assert set(u)=={(i,m) for i,v in choices.items() for m in v}
    for (i,arm),m in metrics.items():
        v=[u[i,a] for a in m['moves']];assert v==m['utilities']
        assert m['signed_total']==v[2]-v[0]==m['signed_1']+m['signed_2']
        ref0=metrics[i,'AX0'];assert m['paired_96']==v[2]-ref0['utilities'][2]
    # independent direct NN eval of actual B-only nonterminal raw leaf samples
    # from distinct requests (not assumed iid), same frozen model.
    got=call([h for h,_ in sample])
    for z,(h,v) in zip(got,sample,strict=True):
        assert z['history']==h;eq(f32(f32(z['wdl'][0])-f32(z['wdl'][2])),v,1e-7)
    count['independent_b_leaf_NN_rechecks']=len(sample)
    sm=read('C06_SMOKE.json');assert sm['standard_parity']==2 and len(sm['rows'])==18
    for a in sm['rows']:axischeck(a,a['history'],a['arm'])
    assert sha(B/'mcts_demo')==sha(BIN)
    return dict(identity=ident,source_games=len(source),roots=len(rr),requests=len(raw),continuations=len(cont),**dict(count),scope='older independent rule reference, independent B trace reverse replay, direct Rapfi leaf spot checks; not full internal tree or production validation',scientific_promotion=False)
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
    with tempfile.TemporaryDirectory(prefix='quartz-rapfi10-restore-') as tmp:
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
