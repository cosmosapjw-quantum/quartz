"""Independent exact readback of locked K27 data and policy, not another CNN/solver execution."""
import pathlib,hashlib,subprocess as SP,importlib.util,json,gzip,random,math,statistics as S,collections as C
from fractions import Fraction as F
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2];P=D.parent
SEED=2026112628
A=[(0,w,0,0) for w in (1,2,4,8)]+[(15,w,0,0) for w in (1,2,4,8)]+[(15,w,1,4) for w in (1,2,4,8)]+[(15,w,2,4) for w in (1,2,4,8)]+[(4,8,2,4),(11,4,0,0),(15,8,2,2),(15,4,2,2)]
N=20;STOP=20;BS=(8,16,32,64);KINDS=('serial','tree4','continue','stop')
def read(n):return json.loads((D/n).read_text())
def raw(p):
    with gzip.open(p,'rt') as f:
        for l in f:yield json.loads(l)
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def git(*a):return SP.check_output(['git','-C',str(ROOT),*a])
def near(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(a,b)
def gridkey(b):return min(tuple(tuple(x) for x in b),tuple(tuple(x) for x in b[::-1]))
def flatkey(b):
    return gridkey([[1 if b[7*y+c]==1 else 2 for y in range(6) if b[7*y+c]!=0] for c in range(7)])
sp=importlib.util.spec_from_file_location('grid17',P/'quartz_kinetic17_20261126/audit_k17.py');g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)
plan=read('PLAN.json');co=read('COHORT.json');model=read('MODEL.json');frozen=read('FROZEN.json');summary=read('SUMMARY.json')
assert sha(D/'PLAN.json')==co['plan'] and sha(D/'SOURCE.jsonl.gz')==co['source']
assert sha(D/'MODEL.json')==frozen['model'] and sha(D/'CAL_LIVE_LOCK.json')==frozen['cal']
assert sha(D/'TRAIN_CAL_ATLAS_LOCK.json')==model['atlas'] and sha(D/'TRAIN_CAL_LABELS.json')==model['labels']
assert sha(D/'trial.py')==model['code']==frozen['code']
locks=plan['history_locks']+list(plan['prior_locks'].values())+plan['assets']['helpers']
for lk in locks:assert hashlib.sha256(git('show',lk['commit']+':'+lk['path'])).hexdigest()==lk['sha256']
prefix=str(D.relative_to(ROOT))+'/'
for name,absent in [('MODEL.json',['CAL_LIVE_LOCK.json','FROZEN.json','TEST_ATLAS_LOCK.json','TEST_LIVE_LOCK.json']),('FROZEN.json',['TEST_ATLAS_LOCK.json','TEST_LIVE_LOCK.json','TEST_LABELS.json'])]:
    c=git('log','-1','--format=%H','--',prefix+name).decode().strip()
    assert hashlib.sha256(git('show',c+':'+prefix+name)).hexdigest()==sha(D/name)
    for z in absent:assert SP.run(['git','-C',str(ROOT),'cat-file','-e',c+':'+prefix+z],capture_output=True).returncode!=0
families=set();boards=set()
for name,lk in plan['prior_locks'].items():
    if name.endswith('.gz') or name=='FA05/attempts':continue
    obj=json.loads(git('show',lk['commit']+':'+lk['path']))
    rows=obj['queries'] if name.endswith('C07_B_QUERIES.json') else (obj.get('rows') or obj.get('connect4'))
    for r in rows:
        if r.get('board') is not None:boards.add(flatkey(r['board']))
        else:boards.add(gridkey(g.replay(r.get('history',r.get('sequence')))))
obj=json.loads(git('show','9b8a4718f7f92d83db2fb00e5806c6c4bd8cbfc2:docs/research/quartz_fouraxis05_20261125/C07_COHORT.json'))
for r in obj['rows']:boards.add(flatkey(r['board']))
for lk in plan['history_locks']:
    if lk['path'].endswith('SOURCE.jsonl.gz'):continue
    obj=json.loads(git('show',lk['commit']+':'+lk['path']))
    for r in obj.get('roots',[]):
        if r.get('domain','C4')!='C4':continue
        h=r['history'];boards.add(gridkey(g.replay(h)))
        if len(h)>=8:families.add(g.canonical(h[:8]))
    for j in obj.get('jobs',obj.get('c4_attempts',[])):
        if 'family' in j and len(j['family'])==8:families.add(j['family'])
        elif len(j.get('history',''))>=8:families.add(g.canonical(j['history'][:8]))
assert len(boards)==plan['historical_boards'],(len(boards),plan['historical_boards'])
assert len(families)==plan['historical_families'],(len(families),plan['historical_families'])
roots={r['id']:r for r in co['roots']};src=0;source_NN=source_plies=queries=0
for j,t in zip(plan['jobs'],raw(D/'SOURCE.jsonl.gz'),strict=True):
    src+=1;assert t['job']==j and g.canonical(j['opening'])==j['family'] and j['family'] not in families
    families.add(j['family']);b=g.replay(j['opening']);h=j['opening'];rng=random.Random(j['seed']);eligible=[]
    def wins(side):
        out=[]
        for k,col in enumerate(b):
            if len(col)==6:continue
            col.append(side)
            if g.won(b,side):out.append(k)
            col.pop()
        return out
    def cand():
        if len(h) in (24,26,28,30) and not g.won(b,1) and not g.won(b,2) and len([v for v in b if len(v)<6])>=3 and not wins(1) and not wins(2):
            eligible.append(dict(history=h,legal=[i for i,c in enumerate(b) if len(c)<6]))
    for st in t['steps']:
        cand();side=len(h)%2+1;legal=[i for i,c in enumerate(b) if len(c)<6];own=wins(side);opp=wins(3-side);m=st['move'];assert m in legal and not any(g.won(b,k) for k in (1,2))
        if own:assert st['kind']=='WIN' and m==own[0]
        elif len(opp)==1:assert st['kind']=='BLOCK' and m==opp[0]
        elif len(legal)==1:assert st['kind']=='ONLY' and m==legal[0]
        elif rng.random()<.75:assert st['kind']=='N32'
        else:assert st['kind']=='UNIFORM' and m==rng.choice(legal)
        if st['kind']=='N32':queries+=1;assert st['NN']>0
        else:assert st['NN']==0
        source_NN+=st['NN'];source_plies+=1;b[m].append(side);h+=str(m+1)
    cand();assert t['history']==h and t['eligible']==eligible,(j['id'],t['eligible'],eligible)
    selected=None
    for a in eligible:
        key=gridkey(g.replay(a['history']))
        if key not in boards:
            boards.add(key);selected=dict(id=j['id'],group=j['group'],family=j['family'],history=a['history'],legal=a['legal'],depth=len(a['history']),tactical=False);break
    assert t['selected']==selected==roots.get(j['id'])
assert src==1800 and (source_NN,source_plies,queries)==(co['NN'],co['plies'],co['queries'])
lab={};statuses=C.Counter();actions=nodes=0
for part in ('TRAIN_CAL','TEST'):
    x=read(part+'_LABELS.json');assert x['atlas']==sha(D/(part+'_ATLAS_LOCK.json'))
    for z in x['rows']:
        r=z['root'];assert roots[r['id']]==r;statuses[z['status']]+=1
        if z['status']!='EXACT':continue
        assert z['agreement'];u={int(k):v for k,v in z['independent'].items()}
        assert set(u)==set(r['legal']) and all(v==(z['scores'][a]>0)-(z['scores'][a]<0) for a,v in u.items())
        lab[r['id']]={a:F(v+1,2) for a,v in u.items()};actions+=len(u);nodes+=z['nodes']
# Independently read raw stage and native recommendation/receipt accounting.
from concurrent.futures import ProcessPoolExecutor
def bin0(s):
    n=sorted(s['n']);return 2*(4*n[-1]>=3*s['visits'])+(n[-1]-n[-2]>=4)
def choice(a,k,m):
    if k=='serial':return [0,0]
    if k=='tree4':return [2,2]
    t=m['tables'][k];x=t['first'][a['prefix_bin']];return [x,t['second'][x][a['mid_bin']]]
def native_verify(a,r,path,B,tot,by):
    assert a['schema']=='quartz_k28_c4_v1' and a['history']==r['history'] and a['chosen']==list(path) and a['budget']==B
    assert a['legal']==r['legal'];pre=a['prefix'];h0=a['prefix_hooks'];assert pre['visits']==8 and pre['nn']==1+h0['hooks']['base_evals']
    assert h0['ledger']['transferred']==0 and h0['ledger']['iterations']==8 and a['prefix_bin']==bin0(pre)
    board=g.replay(r['history']);side=len(r['history'])%2+1;pr=[]
    for mv in r['legal']:
        board[mv].append(side)
        if g.won(board,side):lo=hi=1.
        elif sum(map(len,board))==42:lo=hi=.5
        elif g.canwin(board,3-side):lo=hi=0.
        else:lo,hi=0.,1.
        pr.append([mv,lo,hi]);board[mv].pop()
    assert pr==a['proof'];ub=max(z[2] for z in pr);certificate=next((m for m,l,u in pr if l>=ub),None)
    assert a['best']==(a['result']['best'] if certificate is None else certificate)
    assert a['mid_best']==(a['stages'][0]['state']['best'] if certificate is None else certificate)
    if r['id'] in lab:assert all(l<=lab[r['id']][m]<=u for m,l,u in pr)
    assert a['stopped']==(path[1]==STOP) and len(a['stages'])==(1 if a['stopped'] else 2)
    if a['stopped']:assert a['best']==a['mid_best'] and a['result']==a['stages'][0]['state']
    prev=pre;old=h0['ledger']
    for j,z in enumerate(a['stages']):
        x=path[j];mask,w,mode,res=A[x];s=z['state'];h=z['hooks'];b=z['budget'];l=h['ledger'];hh=h['hooks']
        assert (z['action'],z['mask'],z['workers'],z['mode'],z['reserve'])==(x,mask,w,mode,res)
        assert b['target']==B and b['cap']==4*B and b['reserve']==res
        assert s['nn']-prev['nn']==b['used_nn']==hh['base_evals']+hh['extra_evals']
        assert b['used_nn']+b['unused']==B and 0<=b['unused']<=B
        assert s['visits']-prev['visits']==b['visits']==hh['base_evals']+hh['terminal_leaves']
        assert 0<b['visits']<=4*B and (b['unused']==0 or b['visits']==4*B)
        assert b['cause']==('NN_TARGET' if not b['unused'] else 'VISIT_CAP')
        assert b['protected_extra']+b['released']==res and b['protected_extra']+b['borrowed']==b['pre_extra']
        assert 0<=b['protected_base']+b['borrowed']<=B-res
        assert b['pre_extra']<=hh['extra_evals'] and b['protected_base']<=hh['base_evals']
        assert 0<=b['protected_visits']<=b['visits'] and (res>0 or b['protected_base']==b['protected_visits']==0)
        if mode!=2:assert b['borrowed']==0
        if res and b['protected_visits']<b['cap']:assert b['protected_base']+b['borrowed']==B-res
        assert l['iterations']==s['visits'] and l['eligible']==l['novel']+l['blocked_receipt']
        assert l['transferred']==l['receipt_items']>=old['transferred'] and l['repeated_delivery']==0
        assert 1<=h['peak']<=w and z['bin']==2*bin0(s)+int(l['shared']>0)
        if j==0:assert a['mid_bin']==z['bin']
        if not mask&1:assert hh['select_nodes']==0
        if not mask&2:assert hh['width_nodes']==h['width_realized']==0
        if not mask&4:assert hh['refinements']==hh['extra_evals']==b['denied_refinement']==0
        if not mask&8:assert l['transferred']==old['transferred']
        assert sum(s['n'])==s['visits'] and s['moves']==prev['moves']
        # LiteRoot native tie rule already used in engine; check root accounting.
        lite=z['lite'];assert s['n']==lite['n'][:lite['len']] and s['moves']==lite['moves'][:lite['len']]
        c=C.Counter(stages=1,exact=int(b['unused']==0),cap=int(b['unused']>0),unused=b['unused'],denied=b['denied_refinement'],refinements=hh['refinements'],extra=hh['extra_evals'],reserve=res,released=b['released'],protected_extra=b['protected_extra'],borrowed=b['borrowed'],transferred=l['transferred']-old['transferred'],blocked=l['blocked_receipt']-old['blocked_receipt'],repeat=l['repeated_delivery'],peak=h['peak'],parallel_overlap=int(h['peak']>1))
        tot.update(c);by[x].update(c);prev=s;old=l
    assert a['result']==prev and a['NN_calls']==prev['nn'] and a['visits']==prev['visits']
    snap=a['snapshot'];lite=a['stages'][-1]['lite'];assert snap['root_n']==a['visits'] and snap['evaluator_calls']==a['NN_calls'] and snap['best']==prev['best']
    for k,mv in enumerate(prev['moves']):
        e=next((v for v in snap['edges'] if v['move']==mv),{})
        assert prev['n'][k]==e.get('n',0) and e.get('vl',0)==e.get('vvalue',0)==0
        near(prev['w'][k],e.get('w',0));near(lite['m2'][k],e.get('m2',0))
    return certificate
def rows(phase,B):
    lk=read(phase+'_LOCK.json');assert lk['cohort']==sha(D/'COHORT.json')
    for fi in lk['files']:
        if fi['budget']!=B:continue
        p=D/fi['file'];assert sha(p)==fi['sha256'];yield from raw(p)
def reward(a,u):return u[a['best']]-F(a['NN_calls'],500)
def exact(a):return len(a['stages'])==2 and all(z['budget']['used_nn']==a['budget'] for z in a['stages'])
def small(a,u):
    c=C.Counter()
    for z in a['stages']:
        h=z['hooks']['hooks'];b=z['budget']
        c.update(refinements=h['refinements'],denied_refinement=b['denied_refinement'],borrowed=b['borrowed'],reserve=b['reserve'],released=b['released'],protected_extra=b['protected_extra'])
    return dict(u=u[a['best']],opt=max(u.values()),net=reward(a,u),nn=a['NN_calls'],exact=exact(a),best=a['best'],result=a['result'],chosen=a['chosen'],counts=c,stopped=a['stopped'])
def root_means(v):
    r=C.defaultdict(list)
    for i,y in v:r[i].append(y)
    return [sum(x)/len(x) for x in r.values()]
def lower(v):return float(S.mean(v))-2*S.stdev(v)/math.sqrt(len(v)) if len(v)>1 else -1e9
def q2_fit(v):
    pool=C.defaultdict(list);glob=C.defaultdict(list);cells=C.defaultdict(list)
    for i,a,b,p,h,y1,y2 in v:pool[a,b].append((i,y2));glob[b].append((i,y2));cells[a,h,b].append((i,y2))
    q={};l={};n={};t={k:[[0]*8 for _ in range(N)] for k in ('continue','stop')}
    for a in range(N):
        for h in range(8):
            for b in range(N):
                rr=root_means(cells[a,h,b]);pp=root_means(pool[a,b] or glob[b])
                q[a,h,b]=(sum(rr)+8*S.mean(pp))/(len(rr)+8);l[a,h,b]=lower(rr);n[a,h,b]=len(rr)
            good=[b for b in range(N) if n[a,h,b]>=6 or b==a]
            t['continue'][a][h]=max(good,key=lambda b:q[a,h,b])
            good=[b for b in range(N) if n[a,h,b]>=6 and l[a,h,b]>0 and q[a,h,b]>0]
            t['stop'][a][h]=max(good,key=lambda b:q[a,h,b]) if good else STOP
    return q,l,n,t
def js(a,b):
    x={m:n/a['visits'] for m,n in zip(a['moves'],a['n'])};y={m:n/b['visits'] for m,n in zip(b['moves'],b['n'])}
    return sum((p*math.log(2*p/(p+q)) if p else 0.)/2+(q*math.log(2*q/(p+q)) if q else 0.)/2 for m in x.keys()|y.keys() for p,q in [(x.get(m,0),y.get(m,0))])
def geom(a,b):
    x=[.5+.5*w/(n+4) for w,n in zip(a['w'],a['n'])];y=[.5+.5*w/(n+4) for w,n in zip(b['w'],b['n'])]
    def f(x):
        t=.25;z=max(x);w=[math.exp((v-z)/t) for v in x];return z+t*math.log(sum(w)/len(w)),[v/sum(w) for v in w]
    f0,p=f(x);f1,q=f(y);dr=sum(v*(b-a) for v,a,b in zip(p,x,y));g=.25*sum(v*math.log(v/w) for v,w in zip(p,q));near(f1-f0,dr+g);return g,dr
def stats(aa,z):
    near(S.mean(a['u'] for a in aa),z['u']);near(S.mean(a['opt']-a['u'] for a in aa),z['regret']);near(S.mean(a['net'] for a in aa),z['net'])
    near(S.mean(a['nn'] for a in aa),z['NN']);assert sum(a['exact'] for a in aa)==z['matched'];assert sum(a['stopped'] for a in aa)==z['stopped']
    cc=C.Counter()
    for a in aa:cc.update(a['counts'])
    for k,v in cc.items():assert v==z['counts'][k]
def pair(a,b,z):
    d=[x['u']-y['u'] for x,y in zip(a,b)];near(S.mean(d),z['gain']);assert z['n']==len(d) and z['better']==sum(v>0 for v in d) and z['worse']==sum(v<0 for v in d)
    near(S.mean(x['net']-y['net'] for x,y in zip(a,b)),z['net'])
def compare(a,b,z):
    pair(a,b,z['all']);m=[(x,y) for x,y in zip(a,b) if x['exact'] and y['exact']]
    if m:pair([x for x,y in m],[y for x,y in m],z['matched'])
    else:assert z['matched']['n']==0
def budget_audit(B):
    m=model['budgets'][str(B)];fr=frozen['budgets'][str(B)];su=summary['budgets'][str(B)]
    fit=[];train_ids=[];cal=C.defaultdict(list);cal_ids=[];testd={};testl={};pred=[];tot=C.Counter();by=C.defaultdict(C.Counter);pc={}
    for phase in ('TRAIN_CAL_ATLAS','CAL_LIVE','TEST_ATLAS','TEST_LIVE'):
        nn=vis=requests=0;seen=set();lk=read(phase+'_LOCK.json')
        if phase.startswith('TEST'):assert lk['freeze']==sha(D/'FROZEN.json')
        if phase=='CAL_LIVE':assert lk['freeze']==sha(D/'MODEL.json')
        for t in rows(phase,B):
            r=t['root'];i=r['id'];assert i not in seen and roots[i]==r and t['budget']==B;seen.add(i);u=lab.get(i)
            if phase.endswith('ATLAS'):
                rng=random.Random(SEED+900000+i*100+B);x=list(range(N));y=x.copy();rng.shuffle(x);rng.shuffle(y)
                paths=[(k,k) for k in range(N)]+list(zip(x[:10],y[:10]))
                oo=list(range(30));random.Random(SEED+800000+i*100+B).shuffle(oo)
                assert t['path_order']==[list(p) for p in paths] and t['order']==oo
                for j,a in enumerate(t['paths']):
                    c=native_verify(a,r,paths[j],B,tot,by);requests+=1;nn+=a['NN_calls'];vis+=a['visits'];assert a['prefix']==t['paths'][0]['prefix']
                    if u and r['group']=='TRAIN':
                        a0,b0=paths[j];u0=u[a['prefix']['best'] if c is None else c];u1=u[a['mid_best']]
                        fit.append((i,a0,b0,a['prefix_bin'],a['mid_bin'],u1-u0-F(a['stages'][0]['budget']['used_nn'],500),u[a['best']]-u1-F(a['stages'][1]['budget']['used_nn'],500)))
                    if u and phase=='TEST_ATLAS':
                        a0,b0=paths[j];g0,d=geom(a['stages'][0]['state'],a['result'])
                        pred.append((float(u[a['best']]-u[a['mid_best']]-F(a['stages'][1]['budget']['used_nn'],500)),fr['q2'][a0][a['mid_bin']][b0],g0,d))
                if u and r['group']=='TRAIN':train_ids.append(i)
                if u and phase=='TEST_ATLAS':testd[i]=[small(a,u) for a in t['paths'][:N]]
            else:
                oo=[(k,j) for k in KINDS for j in range(2 if phase=='TEST_LIVE' else 1)];random.Random(SEED+700000+i*100+B).shuffle(oo);assert t['order']==[list(z) for z in oo]
                for k,j in oo:
                    a=t['results'][k+str(j)];native_verify(a,r,choice(a,k,m),B,tot,by);requests+=1;nn+=a['NN_calls'];vis+=a['visits']
                    assert a['prefix']==next(iter(t['results'].values()))['prefix']
                    if u and phase=='CAL_LIVE':cal[k].append(reward(a,u))
                if u and phase=='CAL_LIVE':cal_ids.append(i)
                if u and phase=='TEST_LIVE':testl[i]={k:small(a,u) for k,a in t['results'].items()}
        fs=[z for z in lk['files'] if z['budget']==B]
        assert (requests,nn,vis,len(seen))==tuple(sum(z[k] for z in fs) for k in ('requests','NN','visits','roots'))
        pc[phase]=dict(requests=requests,NN=nn,visits=vis,roots=len(seen))
    # Crossfit reconstruction on two root-disjoint folds.
    assert sorted(train_ids)==m['train_ids'];folds=sorted(train_ids);random.Random(SEED+110000+B).shuffle(folds)
    f0=set(folds[::2]);assert sorted(f0)==m['fold0'];v1={k:C.defaultdict(list) for k in ('continue','stop')}
    for ids in (f0,set(folds)-f0):
        q,l,n,t=q2_fit([z for z in fit if z[0] in ids])
        for i,a,b,p,h,y1,y2 in fit:
            if i in ids:continue
            for k in v1:
                z=t[k][a][h];v1[k][p,a].append((i,y1+(0 if z==STOP else q[a,h,z])))
    q,l,n,t=q2_fit(fit)
    for a in range(N):
        for h in range(8):
            for b in range(N):near(q[a,h,b],m['q2'][a][h][b]);near(l[a,h,b],m['lower'][a][h][b]);assert n[a,h,b]==m['support'][a][h][b]
    for k,v in v1.items():
        assert t[k]==m['tables'][k]['second'];pp=C.defaultdict(list)
        for (p,a),z in v.items():pp[a]+=z
        for p in range(4):
            q0=[]
            for a in range(N):
                x=root_means(v[p,a]);y=(sum(x)+8*S.mean(root_means(pp[a])))/(len(x)+8);near(y,m['q1'][k][p][a]);q0.append(y)
            assert m['tables'][k]['first'][p]==max(range(N),key=q0.__getitem__)
    assert sorted(cal_ids)==fr['cal_ids']
    cv={k:S.mean(cal[k]) for k in KINDS};base=max(KINDS[:2],key=cv.__getitem__)
    assert base==fr['baseline']
    for k,v in cv.items():near(v,fr['calnet'][k]);near(lower([x-y for x,y in zip(cal[k],cal[base])]),fr['cal_lower'][k])
    allowed=[base]+[k for k in KINDS[2:] if fr['cal_lower'][k]>0];assert fr['selected']==max(allowed,key=cv.__getitem__)
    ids=sorted(testd);assert ids==su['ids'] and set(testd)==set(testl)
    diag={k:[testd[i][k] for i in ids] for k in range(N)};live={k:[testl[i][k] for i in ids] for k in next(iter(testl.values()))}
    for k,v in diag.items():stats(v,su['diag'][str(k)])
    for k,v in live.items():stats(v,su['live'][k])
    for k in KINDS:
        aa=live[k+'0'];bb=live[k+'1'];compare(aa,live[base+'0'],su['live_compare'][k]);z=su['repeats'][k]
        near(math.sqrt(S.mean((a['u']-b['u'])**2/2 for a,b in zip(aa,bb))),z['utility_sd']);near(math.sqrt(S.mean((a['nn']-b['nn'])**2/2 for a,b in zip(aa,bb))),z['NN_sd'])
        near(S.mean(js(a['result'],b['result']) for a,b in zip(aa,bb)),z['JSD']);assert sum(a['best']!=b['best'] for a,b in zip(aa,bb))==z['disagree']
    for k,z in su['protected'].items():
        a,b=map(int,k.split(':'));compare(diag[a],diag[b],z)
    yy,pp,gg,dr=zip(*pred);fc=su['forecast']
    assert len(pred)==fc['n'];near(S.mean(yy),fc['actual']);near(S.mean(pp),fc['pred']);near(S.mean(g for g in gg),fc['geom_mean']);near(S.mean(dr),fc['drift_mean'])
    near(math.sqrt(S.mean((x-y)**2 for x,y in zip(yy,pp))),fc['rmse']);near(S.mean(y-x for x,y in zip(yy,pp)),fc['bias'])
    near(S.correlation(gg,yy),fc['geom_corr']);near(S.correlation(pp,yy),fc['prediction_corr'])
    return dict(B=B,phase_counts=pc,totals=dict(tot),by_action={str(k):dict(v) for k,v in by.items()},forecast_count=len(pred),crossfit_cells=N*8*N,roots=len(ids))
if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=4) as ex:budgets=list(ex.map(budget_audit,BS))
    print('BUDGETS_AUDITED',[(z['B'],z['totals']['stages']) for z in budgets],flush=True)
    totals=C.Counter()
    for z in budgets:totals.update(z['totals'])
    main=dict(requests=sum(y['requests'] for z in budgets for y in z['phase_counts'].values()),NN=sum(y['NN'] for z in budgets for y in z['phase_counts'].values()),visits=sum(y['visits'] for z in budgets for y in z['phase_counts'].values()))
    extra={};timing=C.defaultdict(list);trace_rows=[]
    for phase in ('DEEP','TIMING'):
        lk=read(phase+'_LOCK.json');fn=D/(phase+'.jsonl.gz');assert sha(fn)==lk['sha256'];n=nn=vis=0;t0=C.Counter();by=C.defaultdict(C.Counter)
        for t in raw(fn):
            r=t['root'];B=t['budget']
            for k,a in t['results'].items():
                path=[int(k)]*2 if phase=='DEEP' else choice(a,frozen['budgets'][str(B)]['selected'],model['budgets'][str(B)]) if k.startswith('selected:') else [int(k.split(':')[0])]*2
                native_verify(a,r,path,B,t0,by);n+=1;nn+=a['NN_calls'];vis+=a['visits']
                if phase=='DEEP':trace_rows.append((r,a))
                else:timing[str(B)+':'+k.split(':')[0]].append((a['cpu_ns']/1e6,a['wall_ns']/1e6,sum(z['wall_us'] for z in a['stages'])/1000,a['NN_calls']))
        assert n==lk['requests'];extra[phase]=dict(requests=n,NN=nn,visits=vis,totals=dict(t0))
    import struct
    def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
    def exactboard(b):return tuple(tuple(c) for c in b)
    trace_iters=trace_edges=trace_delivered=trace_blocked=0
    for r,a in trace_rows:
        root=g.replay(r['history']);ledger={};payloads={};samples=C.defaultdict(list);count=C.Counter()
        boundaries={a['prefix']['visits']:(a['prefix_lite'],a['prefix_hooks']),a['stages'][0]['state']['visits']:(a['stages'][0]['lite'],a['stages'][0]['hooks']),a['visits']:(a['stages'][1]['lite'],a['stages'][1]['hooks'])}
        trace=a['stages'][1]['hooks']['trace'];assert len(trace)==a['visits']
        def ls(i):
            return dict(iterations=i,edges=count['edges'],children=len(ledger),contexts=sum(len(x) for x in ledger.values()),shared=sum(len(x)>1 for x in ledger.values()),unique_base_leaf=len(payloads),eligible=count['eligible'],novel=count['novel'],blocked_receipt=count['blocked_receipt'],transferred=count['transferred'],receipt_items=sum(len(z[1]) for x in ledger.values() for z in x.values()),applied=count['applied'],changed=count['changed'],repeated_delivery=0)
        for i,t in enumerate(trace):
            trace_iters+=1;b=[c[:] for c in root];side=len(r['history'])%2+1;keys=[]
            assert t['mask']==(0 if i<8 else A[a['chosen'][0]][0] if i<a['stages'][0]['state']['visits'] else A[a['chosen'][1]][0])
            for mv in t['path']:
                assert not g.won(b,1) and not g.won(b,2) and len(b[mv])<6
                parent=exactboard(b);b[mv].append(side);side=3-side;keys.append((exactboard(b),(parent,mv)))
            terminal=g.won(b,1) or g.won(b,2) or sum(map(len,b))==42
            assert t['terminal']==terminal;leaf=(exactboard(b),terminal);raw0=y0=f32(t['raw']);y=f32(t['leaf'])
            if terminal:assert raw0==y==(-1 if g.won(b,3-side) else 0)
            if not t['mask']&4:assert raw0==y
            if leaf in payloads:assert payloads[leaf]==raw0
            else:payloads[leaf]=raw0
            for j in reversed(range(len(keys))):
                trace_edges+=1;count['edges']+=1;child,ctx=keys[j];cc=ledger.setdefault(child,{})
                own,received=cc.setdefault(ctx,({},{}));other={}
                for p,(dd,rr) in cc.items():
                    if p==ctx:continue
                    for k,v in dd.items():
                        if k in other:assert other[k]==v
                        other[k]=v
                other.pop(leaf,None);elig={k:v for k,v in other.items() if k not in own};new={k:v for k,v in elig.items() if k not in received}
                s=t['steps'][j];assert (s['eligible'],s['novel'],s['blocked'])==(len(elig),len(new),len(elig)-len(new))
                count['eligible']+=len(elig);count['novel']+=len(new);count['blocked_receipt']+=len(elig)-len(new)
                mean=S.mean(new.values()) if new else None
                if mean is None:assert s['mean'] is None
                else:near(mean,s['mean'])
                assert f32(s['incoming'])==y
                if leaf in own:assert own[leaf]==raw0
                if leaf in received:assert received[leaf]==raw0
                own[leaf]=raw0
                if not terminal and t['mask']&8 and new:
                    assert not received.keys()&new.keys();received.update(new);count['transferred']+=len(new);count['applied']+=1
                    old=y;alpha=min(.25,len(new)/(len(new)+4));y=f32(max(-1.,min(1.,(1-alpha)*y+alpha*mean)));count['changed']+=abs(y-old)>1e-7
                raw0=-raw0;y=-y;assert f32(s['backup'])==y
            samples[t['path'][0]].append(y)
            if i+1 in boundaries:
                lite,h=boundaries[i+1];assert ls(i+1)==h['ledger']
                for k,mv in enumerate(lite['moves'][:lite['len']]):
                    v=samples[mv];assert len(v)==lite['n'][k];mu=S.mean(v) if v else 0.;near(sum(v),lite['w'][k]);near(sum((x-mu)**2 for x in v),lite['m2'][k])
        trace_delivered+=count['transferred'];trace_blocked+=count['blocked_receipt']
    out=dict(source_attempts=src,source_plies=source_plies,source_NN=source_NN,source_queries=queries,source_locks=len(locks),roots=len(roots),statuses=dict(statuses),exact_actions=actions,oracle_nodes=nodes,main=main,main_totals=dict(totals),budgets=budgets,extra=extra,timing={k:dict(n=len(v),cpu_ms=S.mean(x[0] for x in v),wall_ms=S.mean(x[1] for x in v),tail_ms=S.mean(x[2] for x in v),NN=S.mean(x[3] for x in v)) for k,v in timing.items()},trace_requests=len(trace_rows),trace_iterations=trace_iters,trace_edges=trace_edges,trace_delivered=trace_delivered,trace_blocked=trace_blocked,scope='independent source replay, raw stage accounting, exact crossfit, live selection, repeat and geometry, chronological traces; no CNN/solver rerun')
    (D/'C09_AUDIT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print('AUDIT',json.dumps({k:v for k,v in out.items() if k not in ('budgets','timing')},sort_keys=True),flush=True)
