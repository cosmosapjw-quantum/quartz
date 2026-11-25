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
locks=plan['K20_K21_K22_K23_K24_locks']+plan['history_locks']+list(plan['prior_locks'].values())
for l in locks:assert sha(git('show',l['commit']+':'+l['path']))==l['sha256']
assert fs('PLAN.json')==co['plan'] and fs('SOURCE.jsonl.gz')==co['source']
assert fs('COHORT.json')==f['cohort'] and fs('TRAIN_CAL_ATLAS.jsonl.gz')==f['trace'] and fs('TRAIN_CAL_LABELS.json')==f['labels'] and fs('trial.py')==f['code']
freeze='75551aa9f5ff808e00ded9e36043d0422c3aafa2';prefix=str(D.relative_to(R))+'/'
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
src=lines('SOURCE.jsonl.gz');assert len(src)==240;seen=set();counts=C.Counter();nn=plies=0
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
# Independent reconstruction from persisted outcomes. No trial.py import.
METHODS=['SERIAL','TREE2','WAVE2','ROOT2']
NAMES='SXVB'
def same(a,b):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-8),(a,b)
def js(a,b):
    x={m:n/a['visits'] for m,n in zip(a['moves'],a['n'])};y={m:n/b['visits'] for m,n in zip(b['moves'],b['n'])}
    return sum((p*math.log(2*p/(p+q)) if p else 0.)/2+(q*math.log(2*q/(p+q)) if q else 0.)/2 for m in x.keys()|y.keys() for p,q in [(x.get(m,0),y.get(m,0))])
def best(a):return a['moves'][max(range(len(a['n'])),key=lambda i:(a['n'][i],i))]
requests=lanes=visits=NN=0;receipt_tot=C.Counter();exposure=C.Counter();repeats=0
def verify(a,r,m,profile,trace=False):
    global requests,lanes,visits,NN
    requests+=1;visits+=a['visits'];NN+=a['NN_calls']
    assert a['schema']=='quartz_k25_c4_v1' and a['history']==r['history'] and a['method']==m and a['profile']==profile
    assert a['legal']==r['legal'] and a['visits']==128 and a['prefix']['visits']==32
    assert a['best']==best(a['result']) and a['result']['best']==a['best']
    assert all(lo<=lab[r['id']][act]<=hi for act,lo,hi in a['proof'])
    for tag in ('prefix','result'):
        z=a[tag];assert sum(z['n'])==z['visits'] and z['best']==best(z) and set(z['moves'])==set(r['legal'])
    n=sorted(a['prefix']['n']);assert a['bin']==2*(n[-1]>=24)+(n[-1]-n[-2]>=4)
    k=2 if m=='ROOT2' else 1;assert len(a['lanes'])==k
    for tag,tt in [('prefix','prefix'),('result','end')]:
        z=a[tag]
        assert z['n']==[sum(l[tt]['n'][i] for l in a['lanes']) for i in range(len(z['moves']))]
        assert z['nn']==sum(l[tt]['nn'] for l in a['lanes'])
        for i,v in enumerate(z['w']):same(v,sum(l[tt]['w'][i] for l in a['lanes']))
    assert a['NN_calls']==a['result']['nn']
    for l in a['lanes']:
        lanes+=1
        pre,end,snap=l['prefix'],l['end'],l['snapshot'];p0=l['prefix_hooks'];p1=l['tail_hooks'];h0=p0['hooks'];h1=p1['hooks'];q=p1['ledger']
        assert pre['visits']==32//k and end['visits']==128//k and end['visits']==snap['root_n']
        assert sum(end['n'])==end['visits'] and q['iterations']==end['visits']
        assert pre['nn']==1+h0['base_evals']+h0['extra_evals'] and end['nn']-pre['nn']==h1['base_evals']+h1['extra_evals']
        assert p0['ledger']['transferred']==0 and q['eligible']==q['novel']+q['blocked_receipt']
        assert q['transferred']==q['receipt_items'] and q['repeated_delivery']==0
        assert q['applied']<=q['transferred'] and q['changed']<=q['applied']
        assert 1<=p1['peak']<=(2 if m in ('TREE2','WAVE2') else 1)
        for flag,key0 in [(1,'select_nodes'),(2,'width_nodes'),(4,'refinements')]:
            if not profile&flag:assert h1[key0]==0
        if not profile&2:assert p1['width_realized']==0
        if not profile&8:assert q['transferred']==0
        assert (p1['trace'] is not None)==trace
        for i,act in enumerate(end['moves'][:end['len']]):
            e=next((e for e in snap['edges'] if e['move']==act),{})
            assert e.get('n',0)==end['n'][i] and e.get('vl',0)==e.get('vvalue',0)==0
            same(e.get('w',0),end['w'][i]);same(e.get('m2',0),end['m2'][i])
        receipt_tot.update({z:q[z] for z in ('eligible','novel','blocked_receipt','transferred','applied','repeated_delivery')})
    return lab[r['id']][a['best']]-F(a['NN_calls'],500)
tr=[];cal=[];te=[]
for part in ('TRAIN_CAL','TEST'):
    count0=requests;nn0=NN;vis0=visits
    for t in lines(part+'_ATLAS.jsonl.gz'):
        r=t['root'];assert roots[r['id']]==r
        order=[(m,c) for m in METHODS for c in range(16)]
        random.Random(2026112625+900000+r['id']).shuffle(order)
        assert t['order']==[list(z) for z in order]
        vals={};bins={}
        for m in METHODS:
            arms=t['arms'][m];assert len(arms)==16
            vals[m]=[verify(a,r,m,c) for c,a in enumerate(arms)];bins[m]=arms[0]['bin']
            assert all(a['prefix']==arms[0]['prefix'] for a in arms)
            if m!='ROOT2':assert arms[0]['prefix']==t['arms']['SERIAL'][0]['prefix']
            if part=='TEST':
                a=t['live'][m];assert a['profile']==f['methods'][m]['table'][bins[m]]
                verify(a,r,m,a['profile'])
                for key,pr in [('base2',0),('full2',15),('live2',a['profile'])]:
                    z=t['repeat'][m][key];verify(z,r,m,pr);repeats+=1
                if m in ('SERIAL','ROOT2'):
                    assert a['result']==arms[a['profile']]['result']
                a=arms[15];hs=[l['tail_hooks'] for l in a['lanes']]
                x=[sum(l['hooks']['select_changed'] for l in hs),sum(l['width_realized'] for l in hs),sum(l['hooks']['eval_changed'] for l in hs),sum(l['ledger']['applied'] for l in hs)]
                for name,v in zip(NAMES,x):exposure[(m,name)]+=int(v>0)
                exposure[(m,'all4')]+=int(all(v>0 for v in x))
        q=dict(id=r['id'],y=vals,b=bins)
        if part=='TEST':te.append(t)
        elif r['group']=='TRAIN':tr.append(q)
        else:cal.append(q)
    lk=read(part+'_LOCK.json');assert (requests-count0,NN-nn0,visits-vis0)==(lk['requests'],lk['NN'],lk['visits'])
assert len(tr)==34 and len(cal)==17 and len(te)==19 and requests==4784
assert f['train_ids']==[r['id'] for r in tr] and f['cal_ids']==[r['id'] for r in cal]
for m in METHODS:
    mod=f['methods'][m];pool=[S.mean(r['y'][m][c]-r['y'][m][0] for r in tr) for c in range(16)]
    near([float(v) for v in pool],mod['pooled']);tab=[]
    for b in range(4):
        rr=[r for r in tr if r['b'][m]==b];z=[(sum(r['y'][m][c]-r['y'][m][0] for r in rr)+8*pool[c])/(len(rr)+8) for c in range(16)]
        near([float(v) for v in z],mod['bins'][b]['gains'])
        # Ties in exact arithmetic can differ by roundoff; confirm selected maximizer.
        selected=mod['tables']['conditional'][b];assert z[selected]==max(z)
        tab.append(selected)
    assert pool[mod['tables']['global'][0]]==max(pool)
    for name,tb in mod['tables'].items():
        v=S.mean(r['y'][m][tb[r['b'][m]]] for r in cal);same(v,mod['calnet'][name])
    assert mod['calnet'][mod['selected']]==max(mod['calnet'].values())
# Exact rational factorial and repeated variance reconstruction.
cross={};full_exposure={}
for m in METHODS:
    M=su['methods'][m];full_exposure[m]={k:v for (method,k),v in exposure.items() if method==m}
    for metric,key0 in [('utility','factorial_U'),('net','factorial_net')]:
        val=[]
        for c in range(16):
            v=[]
            for t in te:
                a=t['arms'][m][c];u=lab[t['root']['id']][a['best']]
                v.append(u if metric=='utility' else u-F(a['NN_calls'],500))
            val.append(S.mean(v));same(val[-1],M['arms'][c][metric])
        beta=[sum(v*(-1)**(k.bit_count()-(k&c).bit_count()) for c,v in enumerate(val))/16 for k in range(16)]
        near([float(v) for v in beta],M[key0]['beta'])
        for j,name in enumerate(NAMES):
            bit=1<<j;assert S.mean(val[c|bit]-val[c] for c in range(16) if not c&bit)==2*beta[bit]
            same(val[15]-val[15^bit],M[key0]['leaveout'][name])
    for name,c,rep in [('base',0,'base2'),('full',15,'full2'),('live',None,'live2')]:
        v=[];w=[];dd=[];jj=[];nnv=[]
        for t in te:
            u=lab[t['root']['id']];a=t['live'][m] if c is None else t['arms'][m][c];b=t['repeat'][m][rep]
            x=u[a['best']]-u[b['best']];dn=a['NN_calls']-b['NN_calls']
            v.append(x*x/2);w.append((x-F(dn,500))**2/2);dd.append(a['best']!=b['best']);jj.append(js(a['result'],b['result']));nnv.append(dn*dn/2)
        z=M['repeats'][name];same(math.sqrt(S.mean(v)),z['utility_sd']);same(math.sqrt(S.mean(w)),z['net_sd']);same(S.mean(jj),z['mean_js']);same(math.sqrt(S.mean(nnv)),z['NN_sd']);assert sum(dd)==z['best_disagree']
    cross[m]={}
    for name,c in [('base',0),('full',15),('live',None)]:
        a=[t['live'][m] if c is None else t['arms'][m][c] for t in te];b=[t['live']['SERIAL'] if c is None else t['arms']['SERIAL'][c] for t in te]
        cross[m][name]=dict(decision_disagree=sum(x['best']!=y['best'] for x,y in zip(a,b)),mean_js=S.mean(js(x['result'],y['result']) for x,y in zip(a,b)),NN_delta=S.mean(x['NN_calls']-y['NN_calls'] for x,y in zip(a,b)),utility_delta=float(S.mean(lab[t['root']['id']][x['best']]-lab[t['root']['id']][y['best']] for t,x,y in zip(te,a,b))))
# Independent chronology for diagnostic traces, exact boards with turn parity.
import struct
def f32(v):return struct.unpack('f',struct.pack('f',v))[0]
def boardkey(b):return tuple(tuple(c) for c in b)
trace_edges=trace_iters=trace_lanes=trace_receipts=trace_blocked=0
def verify_trace(t):
    global trace_edges,trace_iters,trace_lanes,trace_receipts,trace_blocked
    r=t['root'];a=t['response'];profile=a['profile'];root=g.replay(r['history'])
    for lane in a['lanes']:
        trace_lanes+=1;ledger={};payloads={};samples=C.defaultdict(list);c=C.Counter();base=lane['prefix']['visits'];end=lane['end'];trace=lane['tail_hooks']['trace']
        def summary(i):
            return dict(iterations=i,edges=c['edges'],children=len(ledger),contexts=sum(len(x) for x in ledger.values()),shared=sum(len(x)>1 for x in ledger.values()),unique_base_leaf=len(payloads),eligible=c['eligible'],novel=c['novel'],blocked_receipt=c['blocked_receipt'],transferred=c['transferred'],receipt_items=sum(len(z[1]) for x in ledger.values() for z in x.values()),applied=c['applied'],changed=c['changed'],repeated_delivery=0)
        def moments(lite):
            for i,m in enumerate(lite['moves'][:lite['len']]):
                v=samples[m];assert len(v)==lite['n'][i]
                mu=S.mean(v) if v else 0.;same(sum(v),lite['w'][i]);same(sum((q-mu)**2 for q in v),lite['m2'][i])
        for i,z in enumerate(trace):
            trace_iters+=1;b=[v[:] for v in root];side=len(r['history'])%2+1;keys=[]
            assert z['mask']==(0 if i<base else profile)
            for mv in z['path']:
                assert not g.won(b,1) and not g.won(b,2) and len(b[mv])<6
                p0=boardkey(b);b[mv].append(side);side=3-side;keys.append((boardkey(b),(p0,mv)))
            terminal=g.won(b,1) or g.won(b,2) or sum(map(len,b))==42
            assert z['terminal']==terminal
            leaf=(boardkey(b),terminal);raw=f32(z['raw']);y=f32(z['leaf'])
            if terminal:assert raw==y==(-1 if g.won(b,3-side) else 0)
            if not z['mask']&4:assert raw==y
            if leaf in payloads:assert payloads[leaf]==raw
            else:payloads[leaf]=raw
            for j in reversed(range(len(keys))):
                trace_edges+=1;c['edges']+=1;child,ctx=keys[j];cc=ledger.setdefault(child,{})
                own,received=cc.setdefault(ctx,({},{}));other={}
                for p0,(dd,rr) in cc.items():
                    if p0==ctx:continue
                    for k,v in dd.items():
                        if k in other:assert other[k]==v
                        other[k]=v
                other.pop(leaf,None)
                elig={k:v for k,v in other.items() if k not in own};new={k:v for k,v in elig.items() if k not in received}
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
                    assert not received.keys()&new.keys()
                    received.update(new);c['transferred']+=len(new);c['applied']+=1
                    incoming=y;alpha=min(.25,len(new)/(len(new)+4));y=f32(max(-1.,min(1.,(1-alpha)*y+alpha*mean)))
                    c['changed']+=abs(y-incoming)>1e-7
                raw=-raw;y=-y;assert f32(s['backup'])==y
            samples[z['path'][0]].append(y)
            if i+1==base:assert summary(base)==lane['prefix_hooks']['ledger'];moments(lane['prefix'])
        assert summary(len(trace))==lane['tail_hooks']['ledger'];moments(end)
        trace_receipts+=c['transferred'];trace_blocked+=c['blocked_receipt']
dd=lines('DIAGNOSTIC.jsonl.gz');assert len(dd)==16 and fs('DIAGNOSTIC.jsonl.gz')==read('DIAGNOSTIC_LOCK.json')['sha256']
for t in dd:verify_trace(t)
out=dict(roots=len(roots),source_attempts=len(src),source_plies=plies,source_NN=nn,source_counts=dict(counts),source_locks=len(locks),actions=actions,oracle_nodes=nodes,requests=requests,lanes=lanes,visits=visits,NN=NN,repeat_requests=repeats,receipt_totals=dict(receipt_tot),diagnostic_requests=len(dd),trace_lanes=trace_lanes,trace_iterations=trace_iters,trace_edges=trace_edges,trace_receipts=trace_receipts,trace_blocked=trace_blocked,full_exposure=full_exposure,cross_method=cross,scope='Independent source replay, stored WDL signs, 4784 request accounting, TRAIN/CAL rational contrasts, Walsh and repeated variance, chronological 16 trace receipt/float32/root moment reconstruction. No independent CNN or solver rerun.')
(D/'C09_AUDIT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(out)
