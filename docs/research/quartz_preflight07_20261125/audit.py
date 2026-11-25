"""Independent old reference replay, quiescent statistics, corpus and restore."""
import collections as C,hashlib,importlib.util,json,math,os,statistics as S,struct,subprocess as SP,sys,tempfile
from pathlib import Path
from publish import ROOT,D,B,BASE,BR,git,sha
from engine import identity,BIN,MODEL,MODEL_SHA
from collector import verify_store,validate_plan,digest,load_record,job_stream,derive_seed,canonical,strict_json
OLD=ROOT/'docs/research/quartz_fouraxis06_20261125'
spec=importlib.util.spec_from_file_location('fa06_old_rules',OLD/'rules.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
def read(n):return json.loads((D/n).read_text())
def eq(a,b):assert abs(a-b)<=1e-6*max(1,abs(a),abs(b)),(a,b)
def snapshot(a,h,rule):
    b,p,w=ref.replay(h,rule)
    assert a['history']==h and a['board']==b and a['player']==p and a['rule']==rule and a['rule_key']==ref.rule_key(b,p,rule) and a['recent']==h[-7:]
    assert a['terminal']==(w is not None) and a['outcome']==(None if w is None else w*p)
    legal=[] if w is not None else [i for i,v in enumerate(b) if not v]
    assert a['legal']==legal
    if 'features' in a:assert a['features']==ref.features(h)
    if w is not None:assert a['proof'] is None;return 0
    pr=ref.proof_reference(b,p,rule);assert pr==a['proof']
    if 'visits' not in a:return len(legal)
    assert a['best'] in legal and a['nn_rows']==a['evaluator_calls']
    if a['cause']=='CERTIFIED':
        assert a['arm']=='CERT' and pr['witness']==a['best'] and a['visits']==a['nn_rows']==0 and a['result'] is None
    else:
        r=a['result'];assert r['root_hash']==a['tt_hash'] and r['root_n']==a['visits']==a['requested_visits']
        assert r['evaluator_calls']==a['nn_rows'] and a['root_nn_feature_sha256']==hashlib.sha256(struct.pack('<3825f',*ref.features(h))).hexdigest()
        assert len(r['candidates'])==len(legal) and {m for m,_ in r['candidates']}==set(legal)
        assert all(math.isfinite(p0) and p0>=0 for _,p0 in r['candidates']);eq(sum(p0 for _,p0 in r['candidates']),1)
        assert [e['move'] for e in r['edges']]==[m for m,p0 in r['candidates'][:len(r['edges'])]]
        assert sum(e['n'] for e in r['edges'])==r['root_n'];eq(sum(e['w'] for e in r['edges']),r['root_w'])
        if a['arm']=='BASE':assert a['best']==r['best']
        for e,(m,p0) in zip(r['edges'],r['candidates']):
            n,x,M=e['n'],e['w'],e['m2'];assert type(n) is int and n>=0 and abs(x)<=n+1e-6 and e['vl']==0 and abs(e['vvalue'])<1e-6
            assert -1e-7<=M<=(n-x*x/n if n else 0)+1e-5
            eq(e['prior'],p0)
    return len(legal)
def preflight_audit():
    sm=read('C06_NATIVE_SMOKE.json');rg=read('C08_NATIVE_REGRESSION.json');assert sm['identity']==rg['identity']==identity()
    count=0;nn=0
    for a in sm['inspect']+sm['search']:
        if a['rule'] in ref.ADMITTED:count+=snapshot(a,a['history'],a['rule'])
        nn+=a['nn_rows']
    for v in rg['serial']:
        count+=snapshot(v['new'],v['new']['history'],v['rule']);assert v['max_m2_delta']<1e-10;nn+=v['new']['nn_rows']
    for v in rg['parallel']:
        a=v['result'];count+=snapshot(a,a['history'],v['rule']);nn+=a['nn_rows']
    assert len(sm['inspect'])==19 and len(sm['search'])==6 and len(rg['serial'])==3 and len(rg['parallel'])==6
    assert len(sm['rejected'])==8 and all('error' in v['response'] for v in sm['rejected'])
    full=(D/'C04_FULL_RUST.log').read_text();assert '632 passed; 0 failed; 89 ignored' in full and full.rstrip().endswith('exit=0')
    verify=verify_store(D/'C07_SMOKE');assert verify['committed']==2
    assert not validate_plan(read('PLAN_DISABLED.json'))['generation_enabled'] and not validate_plan(read('ARENA_PLAN_DISABLED.json'))['generation_enabled']
    assert sha(B/'mcts_demo')==sha(BIN)
    return dict(identity=identity(),rule_action_intervals=count,actual_nn_rows=nn,serial_parity=3,parallel_searches=6,rust_passed=632,rust_ignored=89,smoke_records=verify['committed'],no_bulk=True,worktree_clean_before_audit=True)
def pilot_audit():
    root=D/'P2_CORPUS';p=validate_plan(read('P1_PLAN.json'));manifest=strict_json((root/'manifest.json').read_text());assert manifest['plan']==p and manifest['plan_sha256']==digest(p)
    jobs={j['game_id']:j for j in job_stream(p)}
    report=verify_store(root);assert report['committed']==p['games']==16 and report['missing']==0
    lengths=[];winners=C.Counter();actions=0;nn=0;visits=0;proofs=0;wall=[];files=[];families=C.Counter();at_risk=C.Counter();direct_replies=0
    for file in sorted(root.glob('shard_*/*.json.gz')):
        r=load_record(file);j=jobs[r['job']['game_id']];assert r['job']==j
        h=list(j['opening']);assert r['record_type']=='SELFPLAY';w=r['winner_black'];families[j['family']]+=1
        for k,row in enumerate(r['moves']):
            b,pl,w0=ref.replay(h,p['rule']);assert w0 is None and row['ply']==len(h) and row['player']==pl
            a=dict(row['native'],board=b,history=list(h),rule=p['rule']);actions+=snapshot(a,h,p['rule']);nn+=a['nn_rows'];visits+=a['visits'];wall.append(a['transport_ns']/1e6);proofs+=a['proof']['witness'] is not None
            assert a['arm']=='BASE' and a['requested_visits']==p['visits']==8 and a['seed']==derive_seed(j['seed'],len(h),'search')
            assert row['z']==(None if w is None else w*pl)
            cc=sorted([e['move'],e['n']] for e in a['result']['edges'] if e['n']>0);n=sum(x[1] for x in cc);assert row['policy_target']==dict(counts=cc,total=n)
            m=row['played_move'];assert m in a['legal']
            if len(h)>=p['temperature_plies']:assert m==a['best'] and row['behavior_probability']==[1,1]
            else:
                import random
                z=random.Random(derive_seed(j['seed'],len(h),'behavior')).randrange(n)
                for m0,n0 in cc:
                    if z<n0:assert m==m0 and row['behavior_probability']==[n0,n];break
                    z-=n0
            # Independent direct reply proof on first sampled ply of every game.
            if k==0:
                for m0,l,u,rep in a['proof']['actions']:
                    if rep is not None:
                        bb=b.copy();bb[m0]=1 if pl==1 else 2
                        assert ref.win_at(bb,rep,2 if pl==1 else 1,p['rule']) and l==u==-1
                        direct_replies+=1
            at_risk[len(h)]+=1;h.append(m)
        bb,pl,w0=ref.replay(h,p['rule']);assert h==r['final_history'] and w0==w
        assert (r['status'],r['reason'])==(('CENSORED','PLY_CAP') if w is None else ('COMPLETED','RULE_TERMINAL'))
        lengths.append(len(h));winners[str(w)]+=1;files.append(dict(id=j['game_id'],family=j['family'],split=j['split'],plies=len(h),searched=len(r['moves']),winner=w,status=r['status'],bytes=file.stat().st_size,sha256=sha(file)))
    assert len(files)==16 and len(families)==8 and set(families.values())=={2}
    assert nn==report['counts']['NN_rows'] and visits==8*report['counts']['searched_plies']
    assert report['counts']['compressed_bytes']==sum(x['bytes'] for x in files)
    wall.sort()
    return dict(identity=identity(),plan_sha=digest(p),games=files,completed=report['counts'].get('COMPLETED',0),censored=report['counts'].get('CENSORED',0),winners=dict(winners),family_count=len(families),mean_length=S.mean(lengths),median_length=S.median(lengths),min_length=min(lengths),max_length=max(lengths),lengths=lengths,at_risk={str(k):v for k,v in sorted(at_risk.items())},searched_plies=len(wall),native_visits=visits,NN_rows=nn,root_action_intervals=actions,root_witnesses=proofs,direct_reply_checks=direct_replies,mean_wall_ms=S.mean(wall),median_wall_ms=S.median(wall),p95_wall_ms=wall[math.ceil(.95*len(wall))-1],compressed_bytes=report['counts']['compressed_bytes'],bytes_per_searched_ply=report['counts']['compressed_bytes']/len(wall),verification=report,scientific_promotion=False,pilot_only=True)
def restore():
    rec=json.loads((B/'LATEST.json').read_text());head=git('rev-parse','HEAD')
    assert head==rec['head']==git('ls-remote','origin',f'refs/heads/{BR}').split()[0] and git('rev-parse','HEAD^{tree}')==rec['tree']
    bundle=B/rec['bundle'];assert sha(bundle)==rec['sha256']
    receipts=[json.loads(l) for l in (B/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
    chain=git('rev-list','--reverse',f'{BASE}..{head}').splitlines();assert chain==[x['head'] for x in receipts]
    clean=[json.loads(l) for l in (B/'CLEANUP.jsonl').read_text().splitlines()];assert len(clean)==len(receipts)-1 and [p.name for p in B.glob('*.bundle')]==[bundle.name]
    files=git('diff','--name-only',BASE,head).splitlines()
    def gg(r,*a):return SP.check_output(['git','-C',str(r),*a])
    with tempfile.TemporaryDirectory(prefix='quartz-preflight07-restore-') as tmp:
        r=Path(tmp)/'r.git';SP.run(['git','init','--bare','-q',str(r)],check=True)
        gg(r,'fetch','--depth=1','--no-tags','file://'+str(ROOT),BASE)
        assert not (r/'objects/info/alternates').exists() and SP.run(['git','-C',str(r),'cat-file','-e',head],capture_output=True).returncode!=0
        gg(r,'bundle','verify',str(bundle));gg(r,'fetch',str(bundle),f'refs/heads/{BR}:refs/heads/restored')
        assert gg(r,'rev-parse','refs/heads/restored').decode().strip()==head and gg(r,'rev-parse','refs/heads/restored^{tree}').decode().strip()==rec['tree']
        assert gg(r,'rev-list','--reverse',f'{BASE}..{head}').decode().splitlines()==chain
        for f in files:assert hashlib.sha256(gg(r,'show',f'{head}:{f}')).hexdigest()==sha(ROOT/f),f
        gg(r,'fsck','--connectivity-only','--no-dangling')
    assert sha(B/'mcts_demo')==sha(BIN)
    return dict(head=head,tree=rec['tree'],remote=rec['remote'],stage_count=len(receipts),cleanup_count=len(clean),bundle=bundle.name,sha256=rec['sha256'],bytes=rec['bytes'],restored_files=len(files),external_base=BASE,original_ssot='87c15dc828d6eea519c219fd8a412fa21dfc464d',actual_restore=True,connectivity_fsck=True,binary_sha256=sha(BIN),model_sha256=sha(MODEL))
if __name__=='__main__':
    print(json.dumps({'preflight':preflight_audit,'pilot':pilot_audit,'restore':restore}[sys.argv[1]](),indent=2,sort_keys=True),flush=True)
