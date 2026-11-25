"""Independent prior grouping-based replay, direct child proof, snapshot and restore."""
import collections as C,gzip,hashlib,importlib.util,json,math,os,statistics as S,struct,subprocess as SP,sys,tempfile
from pathlib import Path
from publish import ROOT,D,B,BASE,BR,git,sha
from engine import identity,BIN,MODELS
from experiment import rows,read,check,ST,targets,classify,FULL,LOCAL,CAL
OLD=ROOT/'docs/research/quartz_fouraxis06_20261125/rules.py'
spec=importlib.util.spec_from_file_location('old_reference',OLD);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
def fam(h):
    b,p,w=ref.replay(h[:6],'freestyle');assert w is None
    return hashlib.sha256(ref.orbit_board(b,p,'freestyle')).hexdigest()
def direct_proof(h):
    b,p,w=ref.replay(h,'freestyle');assert w is None
    color=1 if p==1 else 2;opp=3-color
    legal=[i for i,v in enumerate(b) if v==0]
    own=[i for i in legal if ref.win_at(b,i,color,'freestyle')]
    ot=[i for i in legal if ref.win_at(b,i,opp,'freestyle')]
    rr=[];calls=0
    for m in legal:
        bb=b.copy();bb[m]=color
        if ref.win_at(bb,m,color,'freestyle'):rr.append([m,1,1,None]);continue
        if len(legal)==1:rr.append([m,0,0,None]);continue
        found=None
        for j in legal:
            if j==m:continue
            calls+=1
            if ref.win_at(bb,j,opp,'freestyle'):found=j;break
        rr.append([m,-1,-1,found] if found is not None else [m,-1,1,None])
    d=[max(0,max((u for j,l,u,rep in rr if j!=m),default=-1)-l) for m,l,u,rep in rr]
    q=dict(kind='rule_two_ply_v1',own=own,opponent=ot,actions=rr,interval=[max(x[1] for x in rr),max(x[2] for x in rr)],witness=next((x[0] for x,z in zip(rr,d) if z==0),None),risk2=min(d))
    assert q==ref.proof_reference(b,p,'freestyle')
    return q,calls
def audit():
    co=read('C07_COHORT.json');priorfam=set();priorboards=set();priorrows={}
    for f in sorted((ROOT/'docs/research/quartz_preflight07_20261125/P2_CORPUS').glob('shard_*/*.json.gz')):
        rel=str(f.relative_to(ROOT));assert SP.check_output(['git','show',BASE+':'+rel])==f.read_bytes()
        with gzip.open(f,'rt') as fp:g=json.load(fp)
        h=list(g['job']['opening']);key=fam(h);assert key==g['job']['family'];priorfam.add(key)
        for row in g['moves']:
            b,p,w=ref.replay(h,'freestyle');assert w is None
            priorboards.add(ref.orbit_board(b,p,'freestyle'))
            priorrows[g['job']['game_id'],len(h)]=(list(h),key)
            h.append(row['played_move'])
        b,p,w=ref.replay(h,'freestyle');priorboards.add(ref.orbit_board(b,p,'freestyle'))
    assert len(priorfam)==8
    calc=C.Counter();cal=set()
    for r in co['CAL']:
        h,ff=priorrows[r['source_game'],r['source_ply']]
        assert r['history']==h and r['family']==ff and ff not in cal;cal.add(ff)
        b,p,w=ref.replay(h,r['rule']);assert p==r['player'] and classify(ref.proof_reference(b,p,r['rule']))==r['stratum']
        calc[r['stratum']]+=1
    assert cal==priorfam and set(calc.values())=={2}
    for h in ref.fixtures().values():
        if len(h)>=6:priorfam.add(fam(h))
        b,p,w=ref.replay(h,'freestyle');priorboards.add(ref.orbit_board(b,p,'freestyle'))
    assert len(priorfam)==co['excluded_families'] and len(priorboards)==co['excluded_boards']
    test=set();boards=set();ct=C.Counter();dp={};direct_calls=0;direct_acts=0
    for r in co['TEST']:
        h=r['history'];ff=fam(h);assert ff==r['family'] and ff not in priorfam|test;test.add(ff)
        b,p,w=ref.replay(h,'freestyle');key=ref.orbit_board(b,p,'freestyle')
        assert w is None and p==r['player'] and key not in priorboards|boards;boards.add(key)
        q,n=direct_proof(h);assert classify(q)==r['stratum'];dp[r['id']]=q;direct_calls+=n;direct_acts+=len(q['actions'])
        ct[r['stratum'],r['player']]+=1
    assert len(test)==16 and all(ct[s,p]==2 for s in ST for p in (1,-1))
    seen=set();prefix={};prior_eq=0;point_n=0;rule_n=0;win_edges=0;win_samples=0;zero=0;forced=0;nn=0;vis=0;request_counts=C.Counter();native_time_check=0
    basepoints={};metric={(r['id'],r['model'],r['arm'],r['budget']):r for r in read('C08_METRICS.json')}
    for group in ('CAL','TEST'):
        rr={r['id']:r for r in co[group]}
        for model in ('F1','F2'):
            name=f'C07_CAL_{model}.jsonl.gz' if group=='CAL' else f'C08_TEST_{model}.jsonl.gz'
            hr=read(name.replace('.jsonl.gz','_HOST.json'));assert hr['raw_sha256']==sha(D/name)
            local_n=0
            for a in rows(name):
                r=rr[a['id']];arm=a['arm']
                assert a['group']==group and a['stratum']==r['stratum'] and a['family']==r['family'] and a['history']==r['history']
                k=(group,model,r['id'],arm);assert k not in seen;seen.add(k);request_counts[group,model,arm]+=1;local_n+=1
                buds=CAL if group=='CAL' else FULL if arm=='BASE' else LOCAL if arm in ('EXPOSE','FOCUS') else (8,)
                assert arm=='BASE' or group=='TEST' and model=='F1'
                if arm in ('EXPOSE','FOCUS'):assert r['stratum'] in ('WIN','BLOCK')
                check(a,r['history'],r['rule'],arm,buds,17,model,ref=ref)
                if group=='TEST':assert a['proof']==dp[r['id']]
                rule_n+=len(a['proof']['actions']);vis+=a['visits'];nn+=a['nn_rows']
                if a['initial'] is not None:
                    key=(group,model,r['id']);c=a['initial']['candidates']
                    if key in prefix:assert prefix[key]==c;prior_eq+=1
                    else:prefix[key]=c
                else:zero+=1
                if arm=='FOCUS':forced+=1
                for x in a['points']:
                    point_n+=1;v=x['result'];t=set(a['targets']);p=a['proof']
                    iv={m:(l,u) for m,l,u,rep in p['actions']};best=x['best'];l,u=iv[best]
                    hi=max(0,max((v0[1] for m,v0 in iv.items() if m!=best),default=-1)-l)/2
                    lo=max(0,max((v0[0] for m,v0 in iv.items() if m!=best),default=-1)-u)/2
                    assert 0<=lo<=hi<=1
                    if v:
                        es=v['edges']
                        rank=next((j+1 for j,(m,pi) in enumerate(v['candidates']) if m in t),None)
                        if t:
                            E=any(e['move'] in t for e in es)
                            if arm in ('BASE','GUARD','CERT'):
                                first=1 if rank==1 else math.ceil(rank*rank/4)+1
                                assert E==(x['budget']>=first)
                        for e in es:
                            if e['move'] in p['own'] and e['n']>0:
                                assert abs(e['w']-e['n'])<1e-8 and abs(e['m2'])<1e-8;win_edges+=1;win_samples+=e['n']
                        if arm=='BASE':basepoints[group,model,r['id'],x['budget']]=v
                    else:assert hi==0
                    if group=='TEST':
                        m=metric[r['id'],model,arm,x['budget']]
                        assert m['regret_lo']==lo and m['regret_hi']==hi and m['best']==best and m['proof_optimal']==(hi==0)
                    native_time_check+=1
            assert local_n==hr['requests']
    for group in ('CAL','TEST'):
        for model in ('F1','F2'):
            for r in co[group]:
                assert (group,model,r['id'],'BASE') in seen
                if group=='TEST' and model=='F1':
                    assert all((group,model,r['id'],arm) in seen for arm in ('GUARD','CERT'))
                    assert all(((group,model,r['id'],arm) in seen)==(r['stratum'] in ('WIN','BLOCK')) for arm in ('EXPOSE','FOCUS'))
    assert len(seen)==96 and point_n==352 and forced==8
    # Guard/CERT fallback must exactly equal BASE8 including the visit trajectory.
    fallback=0;stop_loss=0;stop_guard=0
    for a in rows('C08_TEST_F1.jsonl.gz'):
        if a['arm'] not in ('GUARD','CERT'):continue
        if a['cause']=='VISITS':
            assert a['points'][0]['result']==basepoints['TEST','F1',a['id'],8];fallback+=1
        elif a['arm']=='CERT' and a['stratum']=='LOSS':stop_loss+=1
        elif a['arm']=='GUARD':stop_guard+=1
    assert (fallback,stop_loss,stop_guard)==(12,4,8)
    sm=read('C06_SMOKE.json');assert len(sm['rows'])==17 and len(sm['rejected'])==5 and all('error' in x['response'] for x in sm['rejected'])
    assert all(x['old']['result']==x['ladder']['result'] for x in sm['parity'])
    s=read('C08_SUMMARY.json');assert s['requests']==80 and s['snapshots']==304
    assert sha(B/'mcts_demo')==sha(BIN) and sha(B/'gomoku15_F2_latest.pt')==MODELS['F2'][1]
    return dict(source_cal_families=len(cal),source_test_families=len(test),test_direct_proof_actions=direct_acts,direct_child_reply_tests=direct_calls,actual_requests=len(seen),actual_snapshots=point_n,root_action_intervals=rule_n,prior_equalities=prior_eq,forced_first_requests=forced,zero_visit_stops=zero,fallback_exact_snapshot_equalities=fallback,cert_loss_stops=stop_loss,guard_stops=stop_guard,terminal_win_edge_snapshots=win_edges,cumulative_terminal_samples_repeated_across_checkpoints=win_samples,NN_rows=nn,visits=vis,native_time_checks=native_time_check,smoke_requests=len(sm['rows']),rejected=len(sm['rejected']),scientific_promotion=False,scope='CAL+TEST, independent older rule and feature implementation and direct two-ply TEST proof; no full-tree trace or independent oracle for unresolved states')
def restore():
    rec=json.loads((B/'LATEST.json').read_text());head=git('rev-parse','HEAD')
    assert head==rec['head']==git('ls-remote','origin',f'refs/heads/{BR}').split()[0] and git('rev-parse','HEAD^{tree}')==rec['tree']
    bundle=B/rec['bundle'];assert sha(bundle)==rec['sha256']
    receipts=[json.loads(l) for l in (B/'STAGE_RECEIPTS.jsonl').read_text().splitlines()]
    chain=git('rev-list','--reverse',f'{BASE}..{head}').splitlines();assert chain==[x['head'] for x in receipts]
    clean=[json.loads(l) for l in (B/'CLEANUP.jsonl').read_text().splitlines()];assert len(clean)==len(receipts)-1 and [p.name for p in B.glob('*.bundle')]==[bundle.name]
    files=git('diff','--name-only',BASE,head).splitlines()
    def gg(r,*a):return SP.check_output(['git','-C',str(r),*a])
    with tempfile.TemporaryDirectory(prefix='quartz-tactical08-restore-') as tmp:
        r=Path(tmp)/'r.git';SP.run(['git','init','--bare','-q',str(r)],check=True)
        gg(r,'fetch','--depth=1','--no-tags','file://'+str(ROOT),BASE)
        assert not (r/'objects/info/alternates').exists() and SP.run(['git','-C',str(r),'cat-file','-e',head],capture_output=True).returncode!=0
        gg(r,'bundle','verify',str(bundle));gg(r,'fetch',str(bundle),f'refs/heads/{BR}:refs/heads/restored')
        assert gg(r,'rev-parse','refs/heads/restored').decode().strip()==head and gg(r,'rev-parse','refs/heads/restored^{tree}').decode().strip()==rec['tree']
        assert gg(r,'rev-list','--reverse',f'{BASE}..{head}').decode().splitlines()==chain
        for f in files:assert hashlib.sha256(gg(r,'show',f'{head}:{f}')).hexdigest()==sha(ROOT/f),f
        gg(r,'fsck','--connectivity-only','--no-dangling')
    assert sha(B/'mcts_demo')==sha(BIN) and sha(B/'gomoku15_F2_latest.pt')==MODELS['F2'][1]
    return dict(head=head,tree=rec['tree'],remote=rec['remote'],stage_count=len(receipts),cleanup_count=len(clean),bundle=bundle.name,sha256=rec['sha256'],bytes=rec['bytes'],restored_files=len(files),external_base=BASE,original_ssot='87c15dc828d6eea519c219fd8a412fa21dfc464d',actual_restore=True,connectivity_fsck=True,binary_sha256=sha(BIN),models={m:sha(v[0]) for m,v in MODELS.items()})
if __name__=='__main__':print(json.dumps(restore() if len(sys.argv)>1 and sys.argv[1]=='restore' else audit(),sort_keys=True,indent=2),flush=True)
