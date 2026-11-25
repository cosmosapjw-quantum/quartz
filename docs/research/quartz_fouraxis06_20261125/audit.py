"""Independent grid, direct two-ply witness, trace and restore audit."""
import collections as C,gzip,hashlib,json,math,os,struct,subprocess as SP,tempfile,sys
from pathlib import Path
from publish import ROOT,D,B,BASE,BR,sha,git
from engine import BIN,MODEL,MODEL_SHA,identity
from collector import verify_store,validate_plan,collection_lock,digest,load_record
RULES={'freestyle':0,'standard':1,'omok':2,'renju':3,'caro':4}
ADMITTED=('freestyle','standard')
def read(n):return json.loads((D/n).read_text())
def win(b,m,p,rule):
    r,c=divmod(m,15)
    # Direct two-ray count, independently applied to actual child boards.
    for dr,dc in ((1,0),(0,1),(1,1),(1,-1)):
        n=1
        for sign in (-1,1):
            rr,cc=r+sign*dr,c+sign*dc
            while 0<=rr<15 and 0<=cc<15 and b[rr*15+cc]==p:n+=1;rr+=sign*dr;cc+=sign*dc
        if n>=5 and (rule=='freestyle' or n==5):return True
    return False
def replay(h,rule):
    b=[0]*225;w=None
    for i,m in enumerate(h):
        assert type(m) is int and 0<=m<225 and b[m]==0 and w is None
        c=1+i%2;b[m]=c
        if win(b,m,c,rule):w=1 if c==1 else -1
        elif i==224:w=0
    return b,1 if len(h)%2==0 else -1,w
def features_bytes(h):
    # Construct bitplanes by move age, not snapshot encoder implementation.
    p=1 if len(h)%2==0 else 2
    v=[0.]*3825
    for i,m in enumerate(h):
        color=1+i%2;plane=0 if color==p else 225
        for j in range(min(8,len(h)-i)):v[j*450+plane+m]=1.
    if p==1:v[-225:]=[1.]*225
    return struct.pack('<3825f',*v)
def check_identity(a,h):
    b=a['board'];p=a['player'];rule=a['rule'];k=int(a['rule_key'],16)
    assert len(a['rule_key'])==128 and k>>454==0
    assert [(k>>(2*i))&3 for i in range(225)]==b
    assert (1 if (k>>450)&1 else -1)==p and ((k>>451)&7)==RULES[rule]
    assert a['recent']==h[-7:]
    x=int(a['board_hash'],16)^0x9e3779b97f4a7c15
    for y in [RULES[rule]+1,len(a['recent'])]+[m+1 for m in a['recent']]:x=((x^y)*0x100000001b3)&((1<<64)-1)
    assert a['tt_hash']==f'{x:016x}'
CACHE={}
def audit_state(a,h,rule):
    check_identity(a,h)
    if rule not in ADMITTED:
        assert a['proof'] is None;return (0,0)
    b,p,w=replay(h,rule);assert a['board']==b and a['player']==p and a['terminal']==(w is not None)
    assert a['outcome']==(None if w is None else w*p)
    if 'features' in a:assert struct.pack('<3825f',*a['features'])==features_bytes(h)
    if w is not None:assert not a['legal'] and a['proof'] is None;return (0,0)
    legal=[i for i,v in enumerate(b) if v==0];assert a['legal']==legal
    z=(tuple(b),p,rule)
    if z not in CACHE:
        c=1 if p==1 else 2;opp=3-c;rows=[];own=[];op=[]
        for m in legal:
            if win(b,m,opp,rule):op.append(m)
            child=b.copy();child[m]=c
            if win(child,m,c,rule):own.append(m);rows.append([m,1,1,None]);continue
            if len(legal)==1:rows.append([m,0,0,None]);continue
            # Full direct child legal reply scan, not threat-removal lemma.
            reply=next((j for j in legal if j!=m and win(child,j,opp,rule)),None)
            rows.append([m,-1,-1,reply] if reply is not None else [m,-1,1,None])
        risks=[max(0,max((u for j,l,u,r in rows if j!=m),default=-1)-l) for m,l,_,_ in rows]
        CACHE[z]=dict(kind='rule_two_ply_v1',own=own,opponent=op,actions=rows,interval=[max(r[1] for r in rows),max(r[2] for r in rows)],witness=next((r[0] for r,v in zip(rows,risks) if v==0),None),risk2=min(risks))
    pr=CACHE[z];assert a['proof']==pr
    if 'visits' in a:
        assert a['best'] in legal and a['nn_rows']==a['evaluator_calls']
        if a['cause']=='CERTIFIED':
            assert a['arm']=='CERT' and pr['witness']==a['best'] and a['visits']==a['nn_rows']==0 and a['result'] is None
        else:
            r=a['result'];assert a['cause']=='VISITS' and r['root_hash']==a['tt_hash'] and r['root_n']==a['visits']==a['requested_visits']
            assert r['evaluator_calls']==a['nn_rows'] and a['root_nn_feature_sha256']==hashlib.sha256(features_bytes(h)).hexdigest()
            assert {m for m,p0 in r['candidates']}==set(legal)
            assert all(math.isfinite(p0) and p0>=0 for _,p0 in r['candidates'])
            assert math.isclose(sum(p0 for _,p0 in r['candidates']),1,rel_tol=1e-5,abs_tol=1e-5)
            assert [e['move'] for e in r['edges']]==[m for m,p0 in r['candidates'][:len(r['edges'])]]
            assert sum(e['n'] for e in r['edges'])==r['root_n']
            assert abs(sum(e['w'] for e in r['edges'])-r['root_w'])<1e-5
            assert all(e['vl']==0 and abs(e['vvalue'])<1e-6 and e['m2']>=-1e-7 and abs(e['w'])<=e['n']+1e-6 for e in r['edges'])
    return len(pr['actions']),sum(l==u for _,l,u,_ in pr['actions'])
def audit():
    ident=identity();sm=read('C06_NATIVE_SMOKE.json');assert sm['identity']==ident
    assert len(sm['inspect'])==19 and len(sm['search'])==6 and len(sm['rejected'])==8
    totals=C.Counter()
    for a in sm['inspect']+sm['search']:
        m,k=audit_state(a,a['history'],a['rule']);totals['action_intervals']+=m;totals['exact_actions']+=k
    for x in sm['rejected']:assert 'error' in x['response']
    probe=read('C08_CPU_PROBE.json');assert probe['identity']==ident
    a=probe['row'];m,k=audit_state(a,a['history'],a['rule']);totals['action_intervals']+=m;totals['exact_actions']+=k
    assert a['python_cpu_ns']>0 and a['rust_cpu_ticks']>=0 and a['tick_hz']>0
    totals['native_requests']=len(sm['search'])+1;totals['NN_rows']=sum(a['nn_rows'] for a in sm['search'])+probe['row']['nn_rows']
    root=D/'C07_SMOKE';mf=json.loads((root/'manifest.json').read_text());p=mf['plan'];assert mf['plan_sha256']==digest(p) and p['schema']=='quartz_long_smoke_plan_v1' and not p['generation_enabled']
    files=sorted(root.glob('shard_*/*.json.gz'));assert len(files)==2
    games=[]
    for f in files:
        raw=f.read_bytes();assert hashlib.sha256(raw).hexdigest()==f.name.split('.')[1]
        with gzip.open(f,'rt') as fp:r=json.load(fp)
        j=r['job'];h=list(j['opening']);w=r['winner_black'];assert r['record_type']=='SMOKE_FIXTURE'
        for t in r['moves']:
            b,pl,w0=replay(h,r['rule']);assert w0 is None and t['player']==pl and t['ply']==len(h)
            a=dict(t['native'],board=b,history=h,rule=r['rule']);m,k=audit_state(a,h,r['rule'])
            totals['action_intervals']+=m;totals['exact_actions']+=k;totals['NN_rows']+=a['nn_rows'];totals['native_requests']+=1
            assert t['z']==(None if w is None else w*pl) and t['played_move'] in a['legal']
            assert t['behavior_probability']==[1,1] and t['played_move']==a['best']
            if a['cause']=='CERTIFIED':assert t['policy_target'] is None
            else:
                counts=sorted([e['move'],e['n']] for e in a['result']['edges'] if e['n'])
                assert t['policy_target']==dict(counts=counts,total=sum(n for _,n in counts))
            h=h+[t['played_move']]
        b,pl,w0=replay(h,r['rule']);assert h==r['final_history'] and w0==w
        assert (r['status'],r['reason'])==(('CENSORED','PLY_CAP') if w is None else ('COMPLETED','RULE_TERMINAL'))
        if w is None:assert len(h)==j['max_ply']
        games.append(dict(id=j['game_id'],initial=len(j['opening']),final=len(h),searched=len(r['moves']),winner=w,status=r['status'],sha256=sha(f),bytes=f.stat().st_size))
    vv=verify_store(root);assert vv==read('C08_SMOKE_REVERIFY.json')
    pp=validate_plan(read('PLAN_DISABLED.json'));aa=validate_plan(read('ARENA_PLAN_DISABLED.json'));assert not pp['generation_enabled'] and not aa['generation_enabled'] and pp['games']==aa['games']==10000
    assert pp['collection_lock']==aa['collection_lock']==collection_lock()
    dry=read('C08_DRY_RUN.json');assert dry['plan_sha256']==digest(pp) and dry['planned_games']==10000 and dry['planned_splits']=={'TRAIN':6000,'CAL':2000,'TEST':2000}
    assert not dry['ready_for_authorized_launch'] and not dry['preflight']['disk_ok'] and not Path(dry['output']).exists()
    # Fresh disabled CLI rejection on the exact finalized plan, no filesystem mutation.
    with tempfile.TemporaryDirectory(prefix='quartz_fa06_gate_') as tmp:
        path=Path(tmp)/'must_not_exist'
        cp=SP.run([sys.executable,str(D/'collector.py'),'--execute','--output',str(path),'--authorize-hash',digest(pp),'--max-new-games','1'],capture_output=True,text=True)
        assert cp.returncode==2 and 'generation_enabled=false' in cp.stderr and not path.exists()
        cp=SP.run([sys.executable,str(D/'collector.py'),'--verify','--output',str(root),'--probe-manifest',str(path)],capture_output=True,text=True)
        assert cp.returncode==2 and not path.exists()
    old=Path('/home/cosmosapjw/Dropbox/personal_projects/quartz_nn_decision_backups/gomoku15_F1_latest.pt')
    assert sha(MODEL)==MODEL_SHA
    assets=dict(model_path=str(MODEL),model_sha=MODEL_SHA,prior_model_backup=str(old),prior_backup_verified=old.is_file() and sha(old)==MODEL_SHA,binary_sha=sha(BIN))
    assert sha(B/'mcts_demo')==sha(BIN)
    # Only exact path of this run; no mutation of unrelated processes.
    running=[v for v in SP.check_output(['ps','-eo','args='],text=True).splitlines() if v.startswith(str(BIN)+' ')]
    assert not running
    return dict(identity=ident,assets=assets,inspect_count=len(sm['inspect']),rejected_count=len(sm['rejected']),unique_nonterminal_rule_states=len(CACHE),**dict(totals),smoke_games=games,smoke_verify=vv,disabled_plan_sha=digest(pp),disabled_arena_plan_sha=digest(aa),production_games=0,production_directory_exists=False,own_workers_running=len(running),bulk_launch=False,training=False,scientific_performance_promotion=False)
def restore():
    rec=json.loads((B/'LATEST.json').read_text());head=git('rev-parse','HEAD')
    assert head==rec['head']==git('ls-remote','origin',f'refs/heads/{BR}').split()[0]
    assert git('rev-parse','HEAD^{tree}')==rec['tree'];bundle=B/rec['bundle'];assert sha(bundle)==rec['sha256']
    receipts=[json.loads(l) for l in (B/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
    chain=git('rev-list','--reverse',f'{BASE}..{head}').splitlines()
    assert chain==[x['head'] for x in receipts] and all(x['head']==x['remote'] for x in receipts)
    clean=[json.loads(l) for l in (B/'CLEANUP.jsonl').read_text().splitlines()]
    assert len(clean)==len(receipts)-1 and [p.name for p in B.glob('*.bundle')]==[bundle.name]
    files=git('diff','--name-only',BASE,head).splitlines()
    allowed=('src/games/gomoku15.rs','src/mcts/mod.rs','src/mcts/select.rs','src/mcts/four_axis.rs','src/mcts/four_axis_math.rs','src/mcts/proof_graph.rs','src/mcts_server.rs','src/mcts_server/four_axis_cpu_server.rs','src/mcts_server/four_axis_pool_plan.rs','src/mcts_server/four_axis_pool_cpu_server.rs','src/mcts_server/four_axis_local_cpu_server.rs','src/mcts_server/gomoku_long_cpu_server.rs','src/mcts_server/gomoku_long_proof.rs')
    assert all(p.startswith(str(D.relative_to(ROOT))+'/') or p in allowed for p in files)
    def gg(r,*a):return SP.check_output(['git','-C',str(r),*a])
    with tempfile.TemporaryDirectory(prefix='quartz-fouraxis06-restore-') as tmp:
        r=Path(tmp)/'r.git';SP.run(['git','init','--bare','-q',str(r)],check=True)
        gg(r,'fetch','--depth=1','--no-tags','file://'+str(ROOT),BASE)
        assert not (r/'objects/info/alternates').exists() and SP.run(['git','-C',str(r),'cat-file','-e',head],capture_output=True).returncode!=0
        gg(r,'bundle','verify',str(bundle));gg(r,'fetch',str(bundle),f'refs/heads/{BR}:refs/heads/restored')
        assert gg(r,'rev-parse','refs/heads/restored').decode().strip()==head and gg(r,'rev-parse','refs/heads/restored^{tree}').decode().strip()==rec['tree']
        assert gg(r,'rev-list','--reverse',f'{BASE}..{head}').decode().splitlines()==chain
        for p in files:assert hashlib.sha256(gg(r,'show',f'{head}:{p}')).hexdigest()==sha(ROOT/p),p
        gg(r,'fsck','--connectivity-only','--no-dangling')
    assert sha(B/'mcts_demo')==sha(BIN)==read('C06_BUILD_IDENTITY.json')['binary_sha256']
    return dict(head=head,tree=rec['tree'],remote=rec['remote'],stage_count=len(receipts),cleanup_count=len(clean),bundle=bundle.name,sha256=rec['sha256'],bytes=rec['bytes'],restored_files=len(files),external_base=BASE,actual_restore=True,connectivity_fsck=True,binary_sha256=sha(BIN),model_sha256=sha(MODEL))
if __name__=='__main__':print(json.dumps(restore() if len(sys.argv)>1 and sys.argv[1]=='restore' else audit(),sort_keys=True,indent=2),flush=True)
