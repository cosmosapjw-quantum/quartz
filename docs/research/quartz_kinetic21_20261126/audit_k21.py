"""Independent readback: grid rules, exact-label agreement, fitting, policy and legacy readout.
No import of sequential.py or geometry.py; raw NN source choices are recorded, not re-searched.
"""
import collections as C, gzip, hashlib, importlib.util, json, math, pathlib, random, statistics as S, subprocess as SP
D=pathlib.Path(__file__).resolve().parent; R=D.parents[2]; P=D.parent
def read(n):return json.loads((D/n).read_text())
def lines(n):
    with gzip.open(D/n,'rt') as f:return [json.loads(l) for l in f]
def sha(b):return hashlib.sha256(b).hexdigest()
def fs(n):return sha((D/n).read_bytes())
def git(*a):return SP.check_output(['git','-C',str(R),*a])
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-11),(a,b)
def load(p,n):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
g=load(P/'quartz_kinetic17_20261126/audit_k17.py','grid17')
hist=read('HISTORY_INDEX.json')
assert len(hist['commits'])==len({x['sha'] for x in hist['commits']})==234
for x in hist['documents']:
    b=git('cat-file','blob',x['blob']);assert len(b)==x['bytes'] and sha(b)==x['sha256']
assert len(hist['documents'])==306
plan=read('PLAN.json');co=read('COHORT.json');f=read('FROZEN.json');summary=read('SUMMARY.json')
for l in plan['K20_locks']+plan['history_locks']+list(plan['prior_locks'].values()):
    assert sha(git('show',l['commit']+':'+l['path']))==l['sha256']
assert fs('PLAN.json')==co['plan'] and fs('SOURCE.jsonl.gz')==co['source']
assert fs('COHORT.json')==f['cohort'] and fs('TRAIN_CAL_TRACE.jsonl.gz')==f['trace'] and fs('TRAIN_CAL_LABELS.json')==f['labels'] and fs('sequential.py')==f['code']
freeze='ffd55801e38b2df744f91799eb2415688f044ff1'
prefix=str(D.relative_to(R))+'/'
assert sha(git('show',freeze+':'+prefix+'FROZEN.json'))==fs('FROZEN.json')
for n in ('TEST_ATLAS.jsonl.gz','TEST_LABELS.json'):
    assert SP.run(['git','-C',str(R),'cat-file','-e',freeze+':'+prefix+n],capture_output=True).returncode!=0
roots={r['id']:r for r in co['roots']}
families=set();boards=set()
for l in plan['history_locks']+plan['K20_locks']:
    if not l['path'].endswith('COHORT.json'):continue
    obj=json.loads(git('show',l['commit']+':'+l['path']))
    for r in obj.get('roots',[]):
        h=r.get('history','')
        if isinstance(h,str) and h and all(c in '1234567' for c in h):
            boards.add(g.boardkey(g.replay(h)));families.add(g.canonical(h[:8]))
    for j in obj.get('jobs',[]):
        if isinstance(j.get('family'),str):families.add(j['family'])
for l in plan['K20_locks']:
    if l['path'].endswith('PLAN.json'):
        families.update(j['family'] for j in json.loads(git('show',l['commit']+':'+l['path']))['jobs'])
source=lines('SOURCE.jsonl.gz');assert len(source)==480
seen=set();counts=C.Counter();nn=plies=0
for j,t in zip(plan['jobs'],source):
    assert t['job']==j and j['family']==g.canonical(j['opening']) and j['family'] not in seen and j['family'] not in families
    seen.add(j['family']);h=j['opening'];b=g.replay(h);rng=random.Random(j['seed'])
    for step in t['steps']:
        side=len(h)%2+1;assert not g.won(b,1) and not g.won(b,2)
        legal=[i for i,c in enumerate(b) if len(c)<6]
        def wins(p):
            out=[]
            for i in legal:
                b[i].append(p)
                if g.won(b,p):out.append(i)
                b[i].pop()
            return out
        own,opp=wins(side),wins(3-side)
        m=step['move'];assert m in legal
        if own:assert (m,step['kind'])==(own[0],'WIN')
        elif len(opp)==1:assert (m,step['kind'])==(opp[0],'BLOCK')
        elif len(legal)==1:assert (m,step['kind'])==(legal[0],'ONLY')
        elif rng.random()<.75:assert step['kind']=='N32' and step['NN']>=0
        else:assert (m,step['kind'])==(rng.choice(legal),'UNIFORM')
        counts[step['kind']]+=1;nn+=step['NN'];plies+=1;h+=str(m+1);b[m].append(side)
    assert h==t['history']
    assert len(h)==30 or g.won(b,1) or g.won(b,2)
    r=t['selected'];assert (r is not None)==(j['id'] in roots)
    if r:
        assert r==roots[j['id']] and h.startswith(r['history']) and r['family']==j['family']
        b=g.replay(r['history']);k=g.boardkey(b)
        assert k not in boards and len(r['history']) in (24,26,28,30) and not any(g.canwin(b,s) for s in (1,2))
        assert r['legal']==[i for i,c in enumerate(b) if len(c)<6] and len(r['legal'])>=3
        boards.add(k)
assert (nn,plies,counts['N32'])==(co['NN'],co['plies'],co['queries'])
labels={};nodes=C.Counter();actions=C.Counter()
for part in ('TRAIN_CAL','TEST'):
    lab=read(part+'_LABELS.json');assert lab['cohort']==fs('COHORT.json')
    assert lab['atlas']==fs('TRAIN_CAL_TRACE.jsonl.gz' if part=='TRAIN_CAL' else 'TEST_ATLAS.jsonl.gz')
    for z in lab['rows']:
        r=z['root'];i=r['id'];assert roots[i]==r and z['agreement'] and z['status']=='EXACT'
        w={int(a):v for a,v in z['independent'].items()};assert set(w)==set(r['legal'])
        for a,v in w.items():assert v==(z['scores'][a]>0)-(z['scores'][a]<0)
        labels[i]={a:(v+1)/2 for a,v in w.items()};nodes[part]+=z['nodes'];actions[part]+=len(w)
assert labels.keys()==roots.keys()
states=0;req=0;tot=C.Counter()
def check(a,r):
    global states,req
    req+=1;prev=None;prior_phi=.5
    assert a['history']==r['history'] and a['legal']==r['legal'] and a['best'] in r['legal']
    for m,lo,hi in a['proof']:assert lo<=labels[r['id']][m]<=hi
    for s in a['states']:
        states+=1;k=s['len'];n=s['n'][:k];w=s['w'][:k];moves=s['moves'][:k]
        assert set(moves)==set(r['legal']) and sum(n)==s['visits'] and all(abs(v)<=c+1e-6 for c,v in zip(n,w))
        x=[.5+.5*v/(c+4) for c,v in zip(n,w)]
        if prev is None:y=[.5]*k
        else:
            assert prev['moves']==s['moves'] and all(c>=d for c,d in zip(s['n'],prev['n'])) and s['nn']>=prev['nn']
            y=[.5+.5*v/(c+4) for c,v in zip(prev['n'][:k],prev['w'][:k])]
        z=sum(math.exp(4*v) for v in x);z0=sum(math.exp(4*v) for v in y)
        prob=[math.exp(4*v)/z0 for v in y];q=[math.exp(4*v)/z for v in x]
        div=sum(a*math.log(a/b) for a,b in zip(prob,q));dr=sum(a*(b-c) for a,b,c in zip(prob,x,y));phi=math.log(z/k)/4
        close(s['kl'],div);close(s['drift'],dr);close(s['phi'],phi);close(s['bregman'],phi-prior_phi-dr);close(s['bregman'],div/4)
        cnt=sorted(n,reverse=True);assert s['bin']==4*(cnt[0]/sum(n)>=.75)+2*(cnt[0]-cnt[1]>=2)+(div>=.02)
        prev=s;prior_phi=phi
    assert prev['nn']==a['NN_calls'] and prev['visits']==a['visits']==a['result']['root_n']
    assert prev['best']==a['result']['best']==a['best']
    tot['NN']+=a['NN_calls'];tot['visits']+=a['visits']
traces=lines('TRAIN_CAL_TRACE.jsonl.gz')
data={}
for t in traces:
    r=t['root'];i=r['id'];a=t['response'];assert r==roots[i] and r['group']!='TEST'
    check(a,r);assert len(a['states'])==8
    data[i]=[(s['bin'],labels[i][s['best']],s['nn']) for s in a['states']]
assert len(traces)==123 and set(data)==set(f['train_ids'])|set(f['cal_ids'])
price=f['price'];assert price==.002
def fit(mode):
    train=[data[i] for i in f['train_ids']];cal=[data[i] for i in f['cal_ids']]
    table=[[0]*8 for _ in range(7)]
    def stop(row,start=0):return next((j for j in range(start,7) if table[j][row[j][0]]<=0),7)
    for t in reversed(range(7)):
        vals=[]
        for row in train:
            j=stop(row,t+1);b,u,n=row[t];_,v,m=row[j];vals.append((b,v-u-price*(m-n)))
        mean=S.mean(v for _,v in vals)
        for k in range(8):
            a=[v for b,v in vals if (b==k if mode=='geometry' else b//2==k//2)]
            table[t][k]=(sum(a)+8*mean)/(len(a)+8)
            close(table[t][k],f['fits'][mode]['table'][t][k])
    close(S.mean(row[stop(row)][1]-price*row[stop(row)][2] for row in cal),f['fits'][mode]['cal_net'])
    return table
tables={m:fit(m) for m in ('basic','geometry')};assert f['selected']==max(tables,key=lambda m:f['fits'][m]['cal_net'])
test=lines('TEST_ATLAS.jsonl.gz');assert len(test)==44
allrows=[];match=0
for t in test:
    r=t['root'];i=r['id'];assert r==roots[i] and r['group']=='TEST';u=labels[i];row={}
    for rep in t['reps']:
        for a in rep.values():check(a,r)
        full=rep['normal']['states']
        for mode in tables:
            j=next((k for k in range(7) if tables[mode][k][full[k]['bin']]<=0),7)
            a=rep[mode];assert (a['NN_calls'],a['visits'],a['best'])==(full[j]['nn'],full[j]['visits'],full[j]['best'])
        assert rep['geometry']['NN_calls']==rep['match']['NN_calls'];match+=1
        for k,a in rep.items():row.setdefault(k,[]).append((u[a['best']],a['NN_calls'],a['visits'],a['cpu_ns']/1e6,a['wall_ns']/1e6))
    allrows.append(row)
for k in allrows[0]:
    rows=[x for r in allrows for x in r[k]]
    for j,n in enumerate(('u','NN','visits','cpu_ms','wall_ms')):close(S.mean(x[j] for x in rows),summary['arms'][k][n])
assert req==563 and match==88
# Actual legacy B13 one-loop readout, post-freeze diagnostic on ordinary native traces.
import numpy as np
b13=load(R/'quartz/phase15_one_loop.py','legacy_b13')
legacy=C.defaultdict(list);nonmono=C.Counter();flip=C.Counter()
normal=[t['response'] for t in traces]+[t['reps'][0]['normal'] for t in test]
for a in normal:
    for c in (.25,1.):
        old=None
        for s in a['states']:
            n=np.array(s['n'][:s['len']],dtype=float);p=n/n.sum()
            q,m=b13.one_loop_correction(p,s['visits'],curvature=c)
            assert abs(q.sum()-1)<1e-12 and np.all(q[p==0]==0)
            d=m['one_loop_effect_kl'];legacy[(c,s['visits'])].append((d,m['one_loop_top1_delta']))
            flip[c]+=not m['one_loop_argmax_preserved']
            if old is not None and d>old+1e-12:nonmono[c]+=1
            old=d
out=dict(history_blobs=306,history_commits=234,source_attempts=len(source),source_counts=dict(counts),source_NN=nn,source_plies=plies,roots=len(roots),labels=dict(actions),oracle_nodes=dict(nodes),native_requests=req,typed_states=states,exact_NN_match_reps=match,fit_cells=112,legacy_B13=dict(traces=len(normal),operator_sha256=sha((R/'quartz/phase15_one_loop.py').read_bytes()),argmax_changes=dict(flip),nonmonotone_steps=dict(nonmono),rows=[dict(curvature=k[0],visits=k[1],n=len(v),mean_KL=S.mean(x[0] for x in v),mean_top1_delta=S.mean(x[1] for x in v)) for k,v in sorted(legacy.items())]))
(D/'C09_AUDIT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out,indent=2,sort_keys=True))
