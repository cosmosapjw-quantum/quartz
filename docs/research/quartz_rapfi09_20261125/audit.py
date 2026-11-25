"""Independent original rule reference, actual game and shared-evaluator audit."""
import collections as C,gzip,hashlib,importlib.util,json,math,statistics as S,subprocess as SP,sys,tempfile
from pathlib import Path
from publish import ROOT,D,B,BASE,BR,git,sha
from arena import identity,check,read,BIN
from probe_rapfi import EXE,W,call
p=ROOT/'docs/research/quartz_fouraxis06_20261125/rules.py'
sp=importlib.util.spec_from_file_location('old_reference',p);ref=importlib.util.module_from_spec(sp);sp.loader.exec_module(ref)
def audit():
    p=read('PAIR_PLAN.json');assert p['identity']==identity()
    co=read('SHARED_SMOKE.json');assert co['identity']==p['identity']
    by=C.defaultdict(list)
    for a in co['rows']:
        check(a,a['history'],a['arm'],a['cap'],a['seed'])
        b,side,w=ref.replay(a['history'],'freestyle')
        assert b==a['board'] and side==a['player'] and w is None
        assert ref.proof_reference(b,side,'freestyle')==a['proof']
        by[a['fixture']].append(a)
    assert len(co['rows'])==15 and len(by)==5 and len(co['rejected'])==3
    for rr in by.values():
        assert len(rr)==3 and len({r['arm'] for r in rr})==3
        assert rr[0]['root_eval']==rr[1]['root_eval']==rr[2]['root_eval']
        if rr[0]['root_eval'] is not None:
            assert rr[0]['initial']['candidates']==rr[1]['initial']['candidates']==rr[2]['initial']['candidates']
        else:assert len({x['best'] for x in rr})==1
    b,side,w=ref.replay(p['opening'],'freestyle')
    assert hashlib.sha256(ref.orbit_board(b,side,'freestyle')).hexdigest()==p['family'] and w is None
    games=[];arms=C.defaultdict(list);proofs=0;sample=[];checks=0
    for i,job in enumerate(p['games']):
        f=D/f'PAIR_GAME_{i}.json.gz'
        with gzip.open(f,'rt') as fp:g=json.load(fp)
        assert g['identity']==identity() and g['job']==job and g['opening']==p['opening'] and g['family']==p['family'] and g['plan_sha256']==sha(D/'PAIR_PLAN.json') and g['status']=='COMPLETED'
        h=list(g['opening']);w=g['winner_black']
        for row in g['rows']:
            a=row['response'];board,side,win=ref.replay(h,'freestyle');assert win is None
            arm=job['black'] if side==1 else job['white']
            assert row['ply']==len(h) and row['player']==side and row['z']==w*side and row['played']==a['best']
            assert a['board']==board and a['history']==h and a['player']==side and a['arm']==arm
            assert a['proof']==ref.proof_reference(board,side,'freestyle')
            proofs+=len(a['proof']['actions'])
            check(a,h,arm,p['cap'],p['seed'])
            if a['root_eval'] is not None and (len(sample)<6 or (len(h)%11==0 and len(sample)<12)):
                sample.append((list(h),a['root_eval']))
            if a['quartz'] is not None:checks+=len(a['quartz']['checks'])
            arms[arm].append(a);h.append(row['played'])
        _,_,win=ref.replay(h,'freestyle')
        assert h==g['final_history'] and win==w and len(h)<=225
        winner=job['black'] if w==1 else job['white'] if w==-1 else 'DRAW'
        games.append(dict(id=job['id'],winner=winner,winner_black=w,plies=len(h),searched_plies=len(g['rows']),sha256=sha(f),bytes=f.stat().st_size,seconds=g['seconds']))
    assert len(arms['STANDARD_PW'])+len(arms['QUARTZ'])==sum(g['searched_plies'] for g in games)
    actual=call([h for h,_ in sample])
    for a,(h,e) in zip(actual,sample,strict=True):
        assert a['history']==h and a['player']==e['player']
        assert all(abs(v-w)<1e-7 for v,w in zip(a['policy'],e['policy']))
        assert all(abs(e[k]-v)<1e-7 for k,v in zip(('win','draw','loss'),a['wdl']))
    resource={}
    for arm,aa in sorted(arms.items()):
        t=[a['transport_ns']/1e6 for a in aa]
        active=[a for a in aa if a['visits']>0]
        resource[arm]=dict(moves=len(aa),visits=sum(a['visits'] for a in aa),NN_calls=sum(a['NN_calls'] for a in aa),mean_ms=S.mean(t),median_ms=S.median(t),p95_ms=sorted(t)[math.ceil(.95*len(t))-1],mean_native_ms=S.mean(a['whole_us']/1000 for a in aa),active_mean_ms=S.mean(a['transport_ns']/1e6 for a in active),guarded=sum(a['cause']=='GUARDED' for a in aa),causes=dict(C.Counter(a['cause'] for a in aa)),controller_checks=sum(len(a['quartz']['checks']) for a in aa if a['quartz']),penalty_abs_sum=sum(a['quartz']['penalty_abs_sum'] for a in aa if a['quartz']),root_selects=sum(a['quartz']['root_selects'] for a in aa if a['quartz']))
    return dict(identity=identity(),opening_family=p['family'],independent_opening_families=1,games=games,resources=resource,root_action_intervals=proofs,independent_nn_rechecks=len(sample),smoke_root_parities=3,shared_guard_fixtures=2,controller_checks=checks,quartz_score=sum(1 if g['winner']=='QUARTZ' else .5 if g['winner']=='DRAW' else 0 for g in games),actual_games=len(games),training=False,equal_mean_time=False,strength_promotion=False)
def restore():
    rec=json.loads((B/'LATEST.json').read_text());head=git('rev-parse','HEAD')
    assert head==rec['head']==git('ls-remote','origin',f'refs/heads/{BR}').split()[0]
    bundle=B/rec['bundle'];assert sha(bundle)==rec['sha256']
    receipts=[json.loads(x) for x in (B/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
    chain=git('rev-list','--reverse',f'{BASE}..{head}').splitlines()
    assert chain==[x['head'] for x in receipts]
    clean=[json.loads(x) for x in (B/'CLEANUP.jsonl').read_text().splitlines()]
    assert len(clean)==len(receipts)-1 and [p.name for p in B.glob('*.bundle')]==[bundle.name]
    files=git('diff','--name-only',BASE,head).splitlines()
    def gg(r,*a):return SP.check_output(['git','-C',str(r),*a])
    with tempfile.TemporaryDirectory(prefix='quartz-rapfi09-restore-') as tmp:
        r=Path(tmp)/'r.git';SP.run(['git','init','--bare','-q',str(r)],check=True)
        gg(r,'fetch','--depth=1','--no-tags','file://'+str(ROOT),BASE)
        assert not (r/'objects/info/alternates').exists()
        assert SP.run(['git','-C',str(r),'cat-file','-e',head],capture_output=True).returncode!=0
        gg(r,'bundle','verify',str(bundle));gg(r,'fetch',str(bundle),f'refs/heads/{BR}:refs/heads/restored')
        assert gg(r,'rev-parse','refs/heads/restored').decode().strip()==head
        assert gg(r,'rev-list','--reverse',f'{BASE}..{head}').decode().splitlines()==chain
        for f in files:assert hashlib.sha256(gg(r,'show',f'{head}:{f}')).hexdigest()==sha(ROOT/f),f
        gg(r,'fsck','--connectivity-only','--no-dangling')
    assert sha(B/'mcts_demo')==sha(BIN)
    return dict(head=head,remote=rec['remote'],tree=rec['tree'],stage_count=len(receipts),cleanup_count=len(clean),restored_files=len(files),external_base=BASE,bundle=bundle.name,bundle_sha256=sha(bundle),actual_restore=True,connectivity_fsck=True,identity=identity())
if __name__=='__main__':print(json.dumps(restore() if len(sys.argv)>1 and sys.argv[1]=='restore' else audit(),sort_keys=True,indent=2,allow_nan=False),flush=True)
