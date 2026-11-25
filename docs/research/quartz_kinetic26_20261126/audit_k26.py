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
locks=plan['K20_K21_K22_K23_K24_K25_locks']+plan['history_locks']+list(plan['prior_locks'].values())
for l in locks:assert sha(git('show',l['commit']+':'+l['path']))==l['sha256']
assert fs('PLAN.json')==co['plan'] and fs('SOURCE.jsonl.gz')==co['source']
model=read('MODEL.json')
assert fs('COHORT.json')==f['cohort']==model['cohort'] and fs('TRAIN_CAL_ATLAS.jsonl.gz')==model['trace'] and fs('TRAIN_CAL_LABELS.json')==model['labels'] and fs('trial.py')==f['code']==model['code']
assert fs('MODEL.json')==f['model'] and fs('CAL_LIVE.jsonl.gz')==f['cal_sha']
prefix=str(D.relative_to(R))+'/'
freeze=git('log','-1','--format=%H','--',prefix+'FROZEN.json').decode().strip()
mf=git('log','-1','--format=%H','--',prefix+'MODEL.json').decode().strip()
assert sha(git('show',freeze+':'+prefix+'FROZEN.json'))==fs('FROZEN.json')
assert sha(git('show',mf+':'+prefix+'MODEL.json'))==fs('MODEL.json')
for commit,names in [(freeze,('TEST_ATLAS.jsonl.gz','TEST_LABELS.json')),(mf,('CAL_LIVE.jsonl.gz','TEST_ATLAS.jsonl.gz','FROZEN.json'))]:
    for n in names:assert SP.run(['git','-C',str(R),'cat-file','-e',commit+':'+prefix+n],capture_output=True).returncode!=0
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
src=lines('SOURCE.jsonl.gz');assert len(src)==300;seen=set();counts=C.Counter();nn=plies=0
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
lab={};statuses=C.Counter();actions=nodes=0;seen_labels=set()
for part in ('TRAIN_CAL','TEST'):
    l=read(part+'_LABELS.json');k=read(part+'_LOCK.json');assert l['atlas']==k['sha256']==fs(part+'_ATLAS.jsonl.gz')
    for z in l['rows']:
        r=z['root'];assert roots[r['id']]==r and r['id'] not in seen_labels;seen_labels.add(r['id']);statuses[z['status']]+=1
        if z['status']!='EXACT':continue
        assert z['agreement'];w={int(k):v for k,v in z['independent'].items()};assert set(w)==set(r['legal']) and all(v==np.sign(z['scores'][a]) for a,v in w.items())
        lab[r['id']]={a:F(v+1,2) for a,v in w.items()};actions+=len(w);nodes+=z['nodes']
assert seen_labels==roots.keys()
METHODS=['SERIAL','TREE2','TREE4','TREE8','WAVE4','ROOT2','ROOT2D'];MASKS=[0,1,2,4,8,15];PATHS=list(itertools.product(MASKS,repeat=2))
def same(a,b):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-8),(a,b)
def best(s):return s['moves'][max(range(len(s['n'])),key=lambda i:(s['n'][i],i))]
def bin0(s):
    n=sorted(s['n']);return 2*int(4*n[-1]>=3*s['visits'])+int(n[-1]-n[-2]>=4)
def js(a,b):
    x={m:n/a['visits'] for m,n in zip(a['moves'],a['n'])};y={m:n/b['visits'] for m,n in zip(b['moves'],b['n'])}
    return sum((p*math.log(2*p/(p+q)) if p else 0.)/2+(q*math.log(2*q/(p+q)) if q else 0.)/2 for m in x.keys()|y.keys() for p,q in [(x.get(m,0),y.get(m,0))])
def sc(a,j):return sum(z[0]['used_nn'] for z in a['stages'][j]['budgets'])
def matched(a):return sc(a,0)==sc(a,1)==8
def policy_choice(a,m,k,f):
    z=f['methods'][m]
    if k=='selected':k=z['selected']
    if k=='base':return [0,0]
    if k=='full':return [15,15]
    if k=='static':return z['static']
    t=z['table'];c=t['first'][a['prefix_bin']];return [c,t['second'][c][a['mid_bin']]]
requests=lanes=stages=visits=NN=0;tot=C.Counter();methods=C.defaultdict(C.Counter);full_exposure=C.defaultdict(C.Counter);tr=[];cal=[];te=[]
def verify(a,r,m,chosen,trace=False):
    global requests,lanes,stages,visits,NN
    requests+=1;visits+=a['visits'];NN+=a['NN_calls']
    assert a['schema']=='quartz_k26_c4_v1' and a['history']==r['history'] and a['method']==m and a['chosen']==list(chosen)
    assert a['legal']==r['legal'] and a['prefix']['visits']==8 and a['prefix_bin']==bin0(a['prefix'])
    # Independently recompute two-ply proof and selected native action.
    b=g.replay(r['history']);side=len(r['history'])%2+1;proof=[]
    for mv in r['legal']:
        b[mv].append(side)
        if g.won(b,side):lo=hi=1.
        elif sum(map(len,b))==42:lo=hi=.5
        elif g.canwin(b,3-side):lo=hi=0.
        else:lo,hi=0.,1.
        proof.append([mv,lo,hi]);b[mv].pop()
    assert proof==a['proof']
    ub=max(p[2] for p in proof);cert=next((p[0] for p in proof if p[1]>=ub),None)
    assert a['best']==(a['result']['best'] if cert is None else cert)
    if r['id'] in lab:assert all(lo<=lab[r['id']][mv]<=hi for mv,lo,hi in proof)
    k=2 if m.startswith('ROOT2') else 1;workers={'TREE2':2,'TREE4':4,'TREE8':8,'WAVE4':4}.get(m,1)
    last=a['prefix_lanes'];ph=a['prefix_hooks'];ps=a['prefix'];assert len(last)==len(ph)==k
    for l,h in zip(last,ph):
        assert l['visits']==8//k and l['nn']==1+h['hooks']['base_evals'] and h['ledger']['iterations']==l['visits']
        assert h['ledger']['transferred']==0
    for s in a['stages']:
        mask=s['profile'];assert len(s['lanes'])==len(s['hooks'])==len(s['budgets'])==k
        z=s['state'];assert s['bin']==2*bin0(z)+any(h['ledger']['shared']>0 for h in s['hooks'])
        for l,h,(bd,times),prev,old in zip(s['lanes'],s['hooks'],s['budgets'],last,ph):
            lanes+=1;stages+=1;lh=h['ledger'];old=old['ledger'];hh=h['hooks']
            assert bd['used_nn']==l['nn']-prev['nn']==hh['base_evals']+hh['extra_evals']
            assert bd['target']==8//k and bd['cap']==32//k and bd['target']==bd['used_nn']+bd['unused']
            assert 0<=bd['unused']<=bd['target'] and bd['visits']==l['visits']-prev['visits']==hh['base_evals']+hh['terminal_leaves']
            assert 0<bd['visits']<=bd['cap'] and (not bd['unused'] or bd['visits']==bd['cap'])
            assert lh['iterations']==l['visits'] and lh['eligible']==lh['novel']+lh['blocked_receipt']
            assert lh['transferred']==lh['receipt_items']>=old['transferred'] and lh['repeated_delivery']==0
            assert 1<=h['peak']<=workers and times[0]<=times[1]
            if not mask&1:assert hh['select_nodes']==0
            if not mask&2:assert hh['width_nodes']==h['width_realized']==0
            if not mask&4:assert hh['extra_evals']==hh['refinements']==bd['denied_refinement']==0
            if not mask&8:assert lh['transferred']==old['transferred']
            assert sum(l['n'])==l['visits'] and l['moves']==prev['moves']
            tot.update({'NN_target':int(bd['unused']==0),'cap':int(bd['unused']>0),'reserved_unused':bd['unused'],'denied':bd['denied_refinement'],'transferred':lh['transferred']-old['transferred'],'blocked':lh['blocked_receipt']-old['blocked_receipt'],'repeat':lh['repeated_delivery']})
            methods[m].update({'stage_lanes':1,'peak_'+str(h['peak']):1,'denied':bd['denied_refinement']})
        assert z['visits']-ps['visits']==sum(b[0]['visits'] for b in s['budgets']) and z['nn']-ps['nn']==sc(a,0 if ps==a['prefix'] else 1)
        last=s['lanes'];ph=s['hooks'];ps=z
    assert a['stages'][0]['profile']==chosen[0] and a['stages'][1]['profile']==chosen[1] and a['mid_bin']==a['stages'][0]['bin']
    assert a['result']==ps and a['NN_calls']==ps['nn'] and a['visits']==ps['visits']<=72
    for z,ls in [(a['prefix'],a['prefix_lanes'])]+[(s['state'],s['lanes']) for s in a['stages']]:
        n=[sum(l['n'][i] for l in ls) for i in range(len(z['n']))];assert z['n']==n and sum(n)==z['visits'] and z['best']==best(z)
        assert z['nn']==sum(l['nn'] for l in ls) and set(z['moves'])==set(r['legal'])
        for i,w in enumerate(z['w']):same(w,sum(l['w'][i] for l in ls))
    for l,s in zip(last,a['snapshots']):
        assert l['visits']==s['root_n'] and l['nn']==s['evaluator_calls']
        for i,mv in enumerate(l['moves'][:l['len']]):
            e=next((e for e in s['edges'] if e['move']==mv),{});assert e.get('n',0)==l['n'][i] and e.get('vl',0)==e.get('vvalue',0)==0
            same(e.get('w',0),l['w'][i]);same(e.get('m2',0),l['m2'][i])
    return a
def util(a,u):return u[a['best']]
def net(a,u):return util(a,u)-F(a['NN_calls'],500)
for part in ('TRAIN_CAL','TEST'):
    c0=requests;n0=NN;v0=visits
    for t in lines(part+'_ATLAS.jsonl.gz'):
        r=t['root'];assert roots[r['id']]==r;order=[(m,i) for m in METHODS for i in range(36)];random.Random(2026112626+900000+r['id']).shuffle(order);assert t['order']==[list(x) for x in order]
        for m in METHODS:
            aa=t['arms'][m];assert len(aa)==36
            for i,a in enumerate(aa):verify(a,r,m,PATHS[i])
            assert all(a['prefix']==aa[0]['prefix'] for a in aa)
            if m in METHODS[:5]:assert aa[0]['prefix']==t['arms']['SERIAL'][0]['prefix']
            if part=='TEST':
                for name,a in t['extra'][m].items():verify(a,r,m,policy_choice(a,m,name[:-1],f))
                a=aa[35];ct=[0]*4
                for stage in a['stages']:
                    for h in stage['hooks']:
                        ct[0]+=h['hooks']['select_changed'];ct[1]+=h['width_realized'];ct[2]+=h['hooks']['eval_changed']
                ct[3]=sum(h['ledger']['applied'] for h in a['stages'][1]['hooks'])
                for name,n in zip('SXVB',ct):full_exposure[m][name]+=int(n>0)
                full_exposure[m]['all4']+=int(all(n>0 for n in ct))
        if r['id'] in lab:
            if r['group']=='TRAIN':tr.append((t,lab[r['id']]))
            elif r['group']=='CAL':cal.append((r,lab[r['id']]))
            else:te.append((t,lab[r['id']]))
    lk=read(part+'_LOCK.json');assert (requests-c0,NN-n0,visits-v0)==(lk['requests'],lk['NN'],lk['visits'])
assert model['train_ids']==[t['root']['id'] for t,u in tr]
assert f['cal_ids']==[r['id'] for r,u in cal]
# Independent exact rational backward regression. Same data, no imported fitter.
for m in METHODS:
    z=model['methods'][m];q2={}
    for ai,i in enumerate(MASKS):
        for bj,j in enumerate(MASKS):
            rr=[(a['mid_bin'],util(a,u)-F(sc(a,1),500)) for t,u in tr for a in [t['arms'][m][PATHS.index((i,j))]]]
            pool=S.mean(y for b,y in rr)
            for b in range(8):
                v=[y for k,y in rr if k==b];q2[i,b,j]=(sum(v)+8*pool)/(len(v)+8);same(q2[i,b,j],z['q2'][ai][b][bj])
        for b in range(8):assert q2[i,b,z['table']['second'][i][b]]==max(q2[i,b,j] for j in MASKS)
    q1={}
    for ai,i in enumerate(MASKS):
        rr=[(a['prefix_bin'],max(q2[i,a['mid_bin'],j] for j in MASKS)-F(sc(a,0),500)) for t,u in tr for a in t['arms'][m] if a['chosen'][0]==i]
        p0=S.mean(y for b,y in rr)
        for b in range(4):
            v=[y for k,y in rr if k==b];q1[b,i]=(sum(v)+8*p0)/(len(v)+8);same(q1[b,i],z['q1'][b][ai])
    for b in range(4):assert q1[b,z['table']['first'][b]]==max(q1[b,i] for i in MASKS)
    means=[S.mean(net(t['arms'][m][i],u) for t,u in tr) for i in range(36)]
    near([float(v) for v in means],z['static_net']);assert means[PATHS.index(tuple(z['static']))]==max(means)
# Actual CAL requests are distinct from atlas and policy was selected on their outcomes.
c0=requests;n0=NN;v0=visits;cv=C.defaultdict(list)
for t in lines('CAL_LIVE.jsonl.gz'):
    r=t['root'];assert r['group']=='CAL';u=lab[r['id']]
    order=[(m,k) for m in METHODS for k in ('base','static','dynamic')];random.Random(2026112626+100000+r['id']).shuffle(order);assert t['order']==[list(x) for x in order]
    for m in METHODS:
        for k,a in t['results'][m].items():
            verify(a,r,m,policy_choice(a,m,k,model));cv[m,k].append(net(a,u))
assert (requests-c0,NN-n0,visits-v0)==(f['cal_requests'],f['cal_NN'],f['cal_visits'])
for m in METHODS:
    vals={k:S.mean(cv[m,k]) for k in ('base','static','dynamic')}
    for k,v in vals.items():same(v,f['methods'][m]['calnet'][k])
    assert vals[f['methods'][m]['selected']]==max(vals.values())
    M=su['methods'][m]
    for i in range(36):
        aa=[(t['arms'][m][i],u) for t,u in te]
        same(S.mean(util(a,u) for a,u in aa),M['paths'][i]['utility']);same(S.mean(net(a,u) for a,u in aa),M['paths'][i]['net'])
    for name,idx in [('base',0),('full',35),('selected',None),('dynamic',None)]:
        aa=[(t['arms'][m][idx] if idx is not None else t['extra'][m][name+'0'],u) for t,u in te]
        base=[(t['arms'][m][0],u) for t,u in te];bb=[(t['extra'][m][name+'1'],u) for t,u in te]
        z=M['results'][name]
        same(S.mean(util(a,u) for a,u in aa),z['utility']);same(S.mean(max(u.values())-util(a,u) for a,u in aa),z['regret']);same(S.mean(net(a,u) for a,u in aa),z['net'])
        same(S.mean(a['NN_calls'] for a,u in aa),z['NN']);assert sum(matched(a) for a,u in aa)==z['exact']
        same(S.mean(util(a,u)-util(b,u) for (a,u),(b,_) in zip(aa,base)),M['pairs'][name]['gain'])
        mm=[(x,y) for x,y in zip(aa,base) if matched(x[0]) and matched(y[0])];assert len(mm)==M['matched'][name]['n']
        if mm:same(S.mean(util(x[0],x[1])-util(y[0],x[1]) for x,y in mm),M['matched'][name]['gain'])
        w=M['repeats'][name];same(math.sqrt(S.mean((util(a,u)-util(b,u))**2/2 for (a,u),(b,_) in zip(aa,bb))),w['utility_sd']);same(math.sqrt(S.mean((a['NN_calls']-b['NN_calls'])**2/2 for (a,u),(b,_) in zip(aa,bb))),w['NN_sd'])
        same(S.mean(js(a['result'],b['result']) for (a,u),(b,_) in zip(aa,bb)),w['mean_js']);assert sum(a['best']!=b['best'] for (a,u),(b,_) in zip(aa,bb))==w['best_disagree']
# Independent chronology on 14 diagnostic traces, receipts persist at profile switch.
import struct
def f32(v):return struct.unpack('f',struct.pack('f',v))[0]
def boardkey(b):return tuple(tuple(c) for c in b)
trace_edges=trace_iters=trace_lanes=trace_receipts=trace_blocked=0
def verify_trace(t):
    global trace_edges,trace_iters,trace_lanes,trace_receipts,trace_blocked
    r=t['root'];a=t['response'];root=g.replay(r['history'])
    for k,trace_hook in enumerate(a['stages'][1]['hooks']):
        trace_lanes+=1;ledger={};payloads={};samples=C.defaultdict(list);c=C.Counter()
        pre=a['prefix_lanes'][k];mid=a['stages'][0]['lanes'][k];end=a['stages'][1]['lanes'][k]
        boundary={pre['visits']:(pre,a['prefix_hooks'][k]),mid['visits']:(mid,a['stages'][0]['hooks'][k]),end['visits']:(end,trace_hook)}
        trace=trace_hook['trace'];assert len(trace)==end['visits']
        def summary(i):
            return dict(iterations=i,edges=c['edges'],children=len(ledger),contexts=sum(len(x) for x in ledger.values()),shared=sum(len(x)>1 for x in ledger.values()),unique_base_leaf=len(payloads),eligible=c['eligible'],novel=c['novel'],blocked_receipt=c['blocked_receipt'],transferred=c['transferred'],receipt_items=sum(len(z[1]) for x in ledger.values() for z in x.values()),applied=c['applied'],changed=c['changed'],repeated_delivery=0)
        for i,z in enumerate(trace):
            trace_iters+=1;b=[v[:] for v in root];side=len(r['history'])%2+1;keys=[]
            assert z['mask']==(0 if i<pre['visits'] else a['chosen'][0] if i<mid['visits'] else a['chosen'][1])
            for mv in z['path']:
                assert not g.won(b,1) and not g.won(b,2) and len(b[mv])<6
                p0=boardkey(b);b[mv].append(side);side=3-side;keys.append((boardkey(b),(p0,mv)))
            terminal=g.won(b,1) or g.won(b,2) or sum(map(len,b))==42
            assert z['terminal']==terminal;leaf=(boardkey(b),terminal);raw=f32(z['raw']);y=f32(z['leaf'])
            if terminal:assert raw==y==(-1 if g.won(b,3-side) else 0)
            if not z['mask']&4:assert raw==y
            if leaf in payloads:assert payloads[leaf]==raw
            else:payloads[leaf]=raw
            for j in reversed(range(len(keys))):
                trace_edges+=1;c['edges']+=1;child,ctx=keys[j];cc=ledger.setdefault(child,{})
                own,received=cc.setdefault(ctx,({},{}));other={}
                for p0,(dd,rr) in cc.items():
                    if p0==ctx:continue
                    for key,val in dd.items():
                        if key in other:assert other[key]==val
                        other[key]=val
                other.pop(leaf,None);elig={k:v for k,v in other.items() if k not in own};new={k:v for k,v in elig.items() if k not in received}
                s=z['steps'][j];assert (s['eligible'],s['novel'],s['blocked'])==(len(elig),len(new),len(elig)-len(new))
                c['eligible']+=len(elig);c['novel']+=len(new);c['blocked_receipt']+=len(elig)-len(new)
                mean=S.mean(new.values()) if new else None
                if mean is None:assert s['mean'] is None
                else:same(mean,s['mean'])
                assert f32(s['incoming'])==y
                if leaf in own:assert own[leaf]==raw
                if leaf in received:assert received[leaf]==raw
                own[leaf]=raw
                if not terminal and z['mask']&8 and new:
                    assert not received.keys()&new.keys();received.update(new);c['transferred']+=len(new);c['applied']+=1
                    incoming=y;alpha=min(.25,len(new)/(len(new)+4));y=f32(max(-1.,min(1.,(1-alpha)*y+alpha*mean)));c['changed']+=abs(y-incoming)>1e-7
                raw=-raw;y=-y;assert f32(s['backup'])==y
            samples[z['path'][0]].append(y)
            if i+1 in boundary:
                l,h=boundary[i+1];assert summary(i+1)==h['ledger']
                for m,mv in enumerate(l['moves'][:l['len']]):
                    v=samples[mv];assert len(v)==l['n'][m];mu=S.mean(v) if v else 0.
                    same(sum(v),l['w'][m]);same(sum((x-mu)**2 for x in v),l['m2'][m])
        trace_receipts+=c['transferred'];trace_blocked+=c['blocked_receipt']
dd=lines('DIAGNOSTIC.jsonl.gz');assert len(dd)==14 and fs('DIAGNOSTIC.jsonl.gz')==read('DIAGNOSTIC_LOCK.json')['sha256']
for t in dd:verify_trace(t)
out=dict(roots=len(roots),source_attempts=len(src),source_plies=plies,source_NN=nn,source_counts=dict(counts),source_locks=len(locks),statuses=dict(statuses),exact_actions=actions,oracle_nodes=nodes,requests=requests,stage_lanes=stages,visits=visits,NN=NN,totals=dict(tot),method_totals={k:dict(v) for k,v in methods.items()},full_exposure={k:dict(v) for k,v in full_exposure.items()},trace_requests=len(dd),trace_lanes=trace_lanes,trace_iterations=trace_iters,trace_edges=trace_edges,trace_receipts=trace_receipts,trace_blocked=trace_blocked,scope='Independent source grid and WDL sign readback, exact rational backward regressions, CAL actual choices and TEST repeated statistics, raw per-lane budget and chronology. Not independent CNN or solver rerun.')
(D/'C09_AUDIT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(out)
