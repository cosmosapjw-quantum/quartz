"""Independent readback: grid legality, raw path snapshots, fixed-design exact OPE."""
import pathlib,subprocess as SP,hashlib,json,gzip,random,collections as C,math,itertools,importlib.util,statistics as S
from fractions import Fraction as F
import numpy as np
D=pathlib.Path(__file__).resolve().parent;R=D.parents[2];P=D.parent
def read(n):return json.loads((D/n).read_text())
def lines(n):
    with gzip.open(D/n,'rt') as f:return [json.loads(x) for x in f]
def sha(x):return hashlib.sha256(x).hexdigest()
def fs(n):return sha((D/n).read_bytes())
def git(*a):return SP.check_output(['git','-C',str(R),*a])
def near(a,b):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-11)
spec=importlib.util.spec_from_file_location('grid17',P/'quartz_kinetic17_20261126/audit_k17.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
plan=read('PLAN.json');co=read('COHORT.json');f=read('FROZEN.json');su=read('SUMMARY.json')
locks=plan['K20_K21_K22_K23_locks']+plan['history_locks']+list(plan['prior_locks'].values())
for l in locks:assert sha(git('show',l['commit']+':'+l['path']))==l['sha256']
assert fs('PLAN.json')==co['plan'] and fs('SOURCE.jsonl.gz')==co['source']
assert fs('COHORT.json')==f['cohort'] and fs('TRAIN_CAL_ATLAS.jsonl.gz')==f['trace'] and fs('TRAIN_CAL_LABELS.json')==f['labels'] and fs('merge_trial.py')==f['code']
freeze='c489ee7fedf11b1f636e120e0f808b10c7e55984';prefix=str(D.relative_to(R))+'/'
assert sha(git('show',freeze+':'+prefix+'FROZEN.json'))==fs('FROZEN.json')
for n in ('TEST_ATLAS.jsonl.gz','TEST_LABELS.json'):
    assert SP.run(['git','-C',str(R),'cat-file','-e',freeze+':'+prefix+n],capture_output=True).returncode!=0
assert read('TEST_LOCK.json')['freeze']==fs('FROZEN.json')
roots={r['id']:r for r in co['roots']};families=set();boards=set()
for l in locks:
    if l['path'].endswith('COHORT.json'):
        obj=json.loads(git('show',l['commit']+':'+l['path']))
        for r in obj.get('roots',[]):
            h=r.get('history','')
            if isinstance(h,str) and h and all(c in '1234567' for c in h):boards.add(g.boardkey(g.replay(h)));families.add(g.canonical(h[:8]))
    if l['path'].endswith('PLAN.json'):
        obj=json.loads(git('show',l['commit']+':'+l['path']))
        for j in obj.get('jobs',[]):families.add(j['family'])
src=lines('SOURCE.jsonl.gz');assert len(src)==480;seen=set();counts=C.Counter();nn=plies=0
for j,t in zip(plan['jobs'],src):
    assert t['job']==j and j['family']==g.canonical(j['opening']) and j['family'] not in seen and j['family'] not in families
    seen.add(j['family']);h=j['opening'];board=g.replay(h);rng=random.Random(j['seed'])
    for step in t['steps']:
        side=len(h)%2+1;assert not g.won(board,1) and not g.won(board,2)
        legal=[i for i,c in enumerate(board) if len(c)<6]
        def wins(p):
            out=[]
            for i in legal:
                board[i].append(p)
                if g.won(board,p):out.append(i)
                board[i].pop()
            return out
        own,opp=wins(side),wins(3-side);m=step['move'];assert m in legal
        if own:assert (m,step['kind'])==(own[0],'WIN')
        elif len(opp)==1:assert (m,step['kind'])==(opp[0],'BLOCK')
        elif len(legal)==1:assert (m,step['kind'])==(legal[0],'ONLY')
        elif rng.random()<.75:assert step['kind']=='N32'
        else:assert (m,step['kind'])==(rng.choice(legal),'UNIFORM')
        counts[step['kind']]+=1;nn+=step['NN'];plies+=1;h+=str(m+1);board[m].append(side)
    assert h==t['history'] and (len(h)==30 or g.won(board,1) or g.won(board,2))
    r=t['selected'];assert (r is not None)==(j['id'] in roots)
    if r:
        assert r==roots[j['id']] and h.startswith(r['history'])
        b=g.replay(r['history']);key=g.boardkey(b);assert key not in boards and not any(g.canwin(b,s) for s in (1,2)) and len(r['history']) in (24,26,28,30)
        assert r['legal']==[i for i,c in enumerate(b) if len(c)<6] and len(r['legal'])>=3;boards.add(key)
assert (nn,plies,counts['N32'])==(co['NN'],co['plies'],co['queries'])
lab={};actions=nodes=0
for part in ('TRAIN_CAL','TEST'):
    l=read(part+'_LABELS.json');k=read(part+'_LOCK.json');assert l['atlas']==k['sha256']==fs(part+'_ATLAS.jsonl.gz')
    for z in l['rows']:
        r=z['root'];assert roots[r['id']]==r and z['agreement'] and z['status']=='EXACT'
        w={int(k):v for k,v in z['independent'].items()};assert set(w)==set(r['legal']) and all(v==np.sign(z['scores'][a]) for a,v in w.items())
        lab[r['id']]={a:F(v+1,2) for a,v in w.items()};actions+=len(w);nodes+=z['nodes']
assert lab.keys()==roots.keys()
# Independent chronological trace reconstruction. Python tuples represent exact
# boards (including ply/side determined by count); no native hash/ledger import.
import struct
def f32(v):return struct.unpack('f',struct.pack('f',v))[0]
def key(b):return tuple(tuple(c) for c in b)
def same(a,b):
    if a is None or b is None:assert a is b
    else:assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(a,b)
def moments(samples,snap):
    assert sum(map(len,samples.values()))==snap['root_n']
    for e in snap['edges']:
        v=samples[e['move']];assert len(v)==e['n']
        m=S.mean(v) if v else 0.
        same(sum(v),e['w']);same(sum((a-m)**2 for a in v),e['m2'])
    same(sum(sum(v) for v in samples.values()),snap['root_w'])
def verify(x,r):
    root=g.replay(r['history']);ledger={};samples=C.defaultdict(list);payloads={};received={};counter=C.Counter()
    assert len(x['trace'])==x['visits']
    def counts(t):
        return dict(children=len(ledger),contexts=sum(len(y) for y in ledger.values()),shared=sum(len(y)>1 for y in ledger.values()),observations=sum(len(v) for y in ledger.values() for v in y.values()),unique_context=sum(len(dict(v)) for y in ledger.values() for v in y.values()),iterations=t,actual_node_ids=len(ledger))
    nn=0;firstnn=0
    for i,t in enumerate(x['trace']):
        b=[c[:] for c in root];side=len(r['history'])%2+1;path=t['path'];keys=[]
        assert t['mode']==(0 if i<32 else x['mode'])
        for m in path:
            assert not g.won(b,1) and not g.won(b,2) and len(b[m])<6
            parent=key(b);b[m].append(side);side=3-side;keys.append((key(b),(parent,m)))
        terminal=g.won(b,1) or g.won(b,2) or sum(map(len,b))==42
        assert t['terminal']==terminal
        leaf=(key(b),terminal);raw=y=f32(t['leaf'])
        if terminal:assert y==(-1 if g.won(b,3-side) else 0)
        if leaf in payloads:assert payloads[leaf]==raw;counter['global_repeat']+=1
        else:payloads[leaf]=raw
        nn+=not terminal
        for j in reversed(range(len(path))):
            child,ctx=keys[j];dd=ledger.setdefault(child,{})
            own=dd.setdefault(ctx,[])
            other=[v for k,ls in dd.items() if k!=ctx for v in ls]
            distinct={}
            for l,v in other:
                if l in distinct:assert distinct[l]==v
                distinct[l]=v
            ownkeys={l for l,v in own};novel={l:v for l,v in distinct.items() if l not in ownkeys and l!=leaf}
            s=t['steps'][j];assert (s['raw_n'],s['unique_n'],s['novel_n'])==(len(other),len(distinct),len(novel))
            rm=S.mean(v for l,v in other) if other else None;dm=S.mean(novel.values()) if novel else None
            same(rm,s['raw_mean']);same(dm,s['novel_mean']);assert f32(s['incoming'])==y
            if not terminal and i>=32:
                counter['raw_opportunities']+=bool(other);counter['novel_opportunities']+=bool(novel);counter['duplicate_only']+=bool(other) and not novel
                counter['cross_raw_count']+=len(other);counter['cross_unique_count']+=len(distinct);counter['cross_novel_count']+=len(novel)
                counter['repeated_other_count']+=len(other)-len(distinct);counter['overlap_own_current']+=len(distinct)-len(novel)
                if t['mode']==2:
                    rr=received.setdefault((child,ctx),set());counter['novel_transfers_reoffered']+=len(rr&novel.keys());counter['novel_transfers_first']+=len(novel.keys()-rr);rr.update(novel)
            own.append((leaf,raw))
            if not terminal and t['mode']:
                n,m=(len(other),rm) if t['mode']==1 else (len(novel),dm)
                if n:
                    a=min(.25,n/(n+4));y=f32(max(-1.,min(1.,(1-a)*y+a*m)))
            raw=-raw;y=-y
            assert f32(s['backup'])==y,(i,j,s['backup'],y)
            if not terminal and i>=32:counter['changed_backup']+=abs(s['backup']+s['incoming'])>1e-7
        samples[path[0]].append(y)
        if i==31:
            firstnn=nn;assert counts(i+1)==x['prefix_counts'];moments(samples,x['prefix'])
    assert counts(len(x['trace']))==x['counts'];moments(samples,x['result'])
    assert nn-firstnn==x['incremental_nn']
    assert nn+1==x['NN_calls'],(nn,x['NN_calls'])
    counter['unique_leafs']=len(payloads);counter['iterations']=len(x['trace']);counter['edges']=x['counts']['observations']
    counter['shared']=x['counts']['shared'];return counter
records=[];tot=C.defaultdict(C.Counter);requests=0;parity=0;changes=C.Counter();part_tot=C.Counter()
for part in ('TRAIN_CAL','TEST'):
    for t in lines(part+'_ATLAS.jsonl.gz'):
        r=t['root'];u=lab[r['id']];seq=list(range(3));random.Random(2026112624+900000+r['id']).shuffle(seq)
        assert seq==t['order'] and seq[0]==t['logged'] and t['propensity']==[1/3]*3
        aa=t['arms'];pre=aa[0]['prefix'];pc=aa[0]['prefix_counts'];n=sorted(e['n'] for e in pre['edges']);bin0=2*int(pc['shared']>0)+int(n[-1]-n[-2]>=4)
        for i,x in enumerate(aa):
            requests+=1;assert x['mode']==i and x['bin']==bin0 and x['prefix']==pre and x['prefix_counts']==pc and x['trace'][:32]==aa[0]['trace'][:32]
            assert x['history']==r['history'] and x['best']==x['result']['best'] and x['best'] in u
            assert all(lo<=u[m]<=hi for m,lo,hi in x['proof'])
            v=verify(x,r);tot[(part,i)].update(v);part_tot[part]+=1
            if part=='TEST':
                changes[(i,'recommendation')]+=x['best']!=aa[0]['best'];changes[(i,'snapshot')]+=x['result']!=aa[0]['result']
        if part=='TEST':
            a=t['base'];b=t['live'];c=f['table'][bin0]
            assert all(a[k]==aa[0][k] for k in ('prefix','result','best','visits','incremental_nn'))
            assert all(b[k]==aa[c][k] for k in ('prefix','result','trace','counts','best','visits','incremental_nn'))
            parity+=1
        y=[u[a['best']]-F(a['incremental_nn'],500) for a in aa]
        records.append(dict(id=r['id'],group=r['group'],bin=bin0,y=y,u=[u[a['best']] for a in aa],n=[a['incremental_nn'] for a in aa],full=max(u.values())))
assert len(records)==len(roots)==199 and requests==597 and parity==52
tr=[r for r in records if r['group']=='TRAIN'];cal=[r for r in records if r['group']=='CAL'];te=[r for r in records if r['group']=='TEST']
assert [r['id'] for r in tr]==f['train_ids'] and [r['id'] for r in cal]==f['cal_ids']
for b in range(4):
    rr=[r for r in tr if r['bin']==b];q=[]
    for c in (1,2):
        v=[r['y'][c]-r['y'][0] for r in rr];n=len(v);mu=S.mean(v) if n else 0.;se=S.stdev(v)/math.sqrt(n) if n>1 else 0.
        z=f['stats'][b][c-1];assert n==z['n'] and sum(a>0 for a in v)==z['positive'] and sum(a<0 for a in v)==z['negative']
        same(mu,z['mean']);same(se,z['se']);same(mu-2*se,z['lower'])
        eligible=n>=8 and z['positive']>=3 and z['negative']>=1 and mu-2*se>0
        assert eligible==z['eligible']
        if eligible:q.append((mu-2*se,c))
    assert f['guarded'][b]==max(q+[(0,0)])[1]
for name,t in [('N',[0]*4),('guarded',f['guarded'])]:same(S.mean(r['y'][t[r['bin']]] for r in cal),f['calnet'][name])
assert f['selected']==max(('N','guarded'),key=f['calnet'].__getitem__)
matched=[r for r in te if r['n']==[64]*3];assert len(matched)==su['matched']==44
for i,name in enumerate(('K24N','K24M','K24D')):
    same(S.mean(r['u'][i] for r in te),su['full'][name]['utility']);same(S.mean(r['n'][i] for r in te),su['full'][name]['NN'])
    same(S.mean(r['u'][i]-r['u'][0] for r in matched),su['matched_pairs'][name]['gain'])
    for k in ('raw_opportunities','novel_opportunities','duplicate_only','cross_raw_count','cross_unique_count','cross_novel_count','repeated_other_count','overlap_own_current','changed_backup'):
        assert tot[('TEST',i)][k]==su['telemetry'][name]['totals'][k],(k,i,tot[('TEST',i)][k])
out=dict(roots=len(records),source_attempts=len(src),source_plies=plies,source_NN=nn,source_counts=dict(counts),actions=actions,oracle_nodes=nodes,source_locks=len(locks),fixed_native_paths=requests,base_live_parity=parity,trace_iterations=sum(t['iterations'] for t in tot.values()),trace_edges=sum(t['edges'] for t in tot.values()),matched=len(matched),audit={f'{part}:{i}':dict(v) for (part,i),v in tot.items()},changed_decisions={f'{i}:{k}':v for (i,k),v in changes.items()},scope='independent grid/chronological identity sets and raw floats, root moments, policy fit, source replay and stored WDL agreement; not a reimplementation of selection or independent CNN re-eval. Reoffered unique transfer items are counted per child/context, not consumed by ledger.')
(D/'C09_AUDIT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(out)
