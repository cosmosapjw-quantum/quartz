"""Independent exact readback of locked K27 data and policy, not another CNN/solver execution."""
import pathlib,hashlib,subprocess as SP,importlib.util,json,gzip,random,math,statistics as S,collections as C
from fractions import Fraction as F
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2];P=D.parent
SEED=2026112632
A={1:(0,1,0),4:(0,1,0),5:(0,1,0),6:(0,1,0),7:(15,8,4),8:(15,8,4),9:(15,8,4)}
N=11;STOP=0
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
plan=read('PLAN.json');co=read('COHORT.json');frozen=read('FROZEN.json');summary=read('SUMMARY.json')
assert sha(D/'PLAN.json')==co['plan'] and sha(D/'SOURCE.jsonl.gz')==co['source']
assert frozen['cohort']==sha(D/'COHORT.json') and frozen['atlas']==sha(D/'TRAIN_CAL_ATLAS_LOCK.json') and frozen['labels']==sha(D/'TRAIN_CAL_LABELS.json') and frozen['code']==sha(D/'trial.py')
assert read('TEST_ATLAS_LOCK.json')['freeze']==sha(D/'FROZEN.json')
locks=plan['history_locks']+list(plan['prior_locks'].values())+plan['assets']['helpers']
for lk in locks:assert hashlib.sha256(git('show',lk['commit']+':'+lk['path'])).hexdigest()==lk['sha256']
prefix=str(D.relative_to(ROOT))+'/'
c=git('log','-1','--format=%H','--',prefix+'FROZEN.json').decode().strip()
assert hashlib.sha256(git('show',c+':'+prefix+'FROZEN.json')).hexdigest()==sha(D/'FROZEN.json')
for z in ('TEST_ATLAS_LOCK.json','TEST_LABELS.json'):assert SP.run(['git','-C',str(ROOT),'cat-file','-e',c+':'+prefix+z],capture_output=True).returncode!=0
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
assert src==3600 and (source_NN,source_plies,queries)==(co['NN'],co['plies'],co['queries'])
lab={};statuses=C.Counter();actions=nodes=0
for part in ('TRAIN_CAL','TEST'):
    x=read(part+'_LABELS.json');assert x['atlas']==sha(D/(part+'_ATLAS_LOCK.json'))
    for z in x['rows']:
        r=z['root'];assert roots[r['id']]==r;statuses[z['status']]+=1
        if z['status']!='EXACT':continue
        assert z['agreement'];u={int(k):v for k,v in z['independent'].items()}
        assert set(u)==set(r['legal']) and all(v==(z['scores'][a]>0)-(z['scores'][a]<0) for a,v in u.items())
        lab[r['id']]={a:F(v+1,2) for a,v in u.items()};actions+=len(u);nodes+=z['nodes']
# Independent grid/response and budget readback, no imported collector/fitter.
def proof_grid(r):
    b=g.replay(r['history']);side=len(r['history'])%2+1;pr=[]
    for mv in r['legal']:
        b[mv].append(side)
        if g.won(b,side):lo=hi=1.
        elif sum(map(len,b))==42:lo=hi=.5
        elif g.canwin(b,3-side):lo=hi=0.
        else:lo,hi=0.,1.
        pr.append([mv,lo,hi]);b[mv].pop()
    return pr
def certificate(pr):
    ub=max(x[2] for x in pr);return next((m for m,l,u in pr if l>=ub),None)
def feature_grid(r,s,pr):
    c=certificate(pr);c=s['best'] if c is None else c
    ix=sorted(range(s['len']),key=lambda i:(s['moves'][i]!=c,-s['n'][i],i))
    heights=list(map(len,g.replay(r['history'])));ps={m:(l,u) for m,l,u in pr};rank=[];allow=[];f=[len(r['history'])/42]
    for i in ix:
        m=s['moves'][i];n=s['n'][i];l,u=ps[m];rank.append(m);allow.append([int(2*l),int(2*u)])
        f.extend([math.sqrt(n/s['visits']),math.sqrt(max(0,s['prior'][i])),(1+s['w'][i]/(n+4))/2,math.sqrt(max(0,s['m2'][i])/(n+1))/2,heights[m]/6,abs(m-3)/3,l,u])
    return rank,f,allow
def native(a,r,arm,tot,by,trace=False):
    assert a['schema']=='quartz_k32_c4_v1' and a['history']==r['history'] and a['legal']==r['legal'] and a['arm']==arm
    pr=proof_grid(r);assert pr==a['proof'];cc=certificate(pr)
    if r['id'] in lab:assert all(lo<=lab[r['id']][m]<=hi for m,lo,hi in pr)
    pre,mid,end=[a[k] for k in ('prefix','mid','end')]
    ph,mh,eh=[a[k] for k in ('prefix_hooks','mid_hooks','end_hooks')]
    assert pre['visits']==8 and pre['nn']==1+ph['hooks']['base_evals'] and ph['ledger']['iterations']==8 and ph['ledger']['transferred']==0
    rank,f,allowed=feature_grid(r,mid,pr);assert rank==a['rank'] and allowed==a['allowed']
    for x,y in zip(f,a['features']):near(x,y)
    assert a['mid_best']==rank[0] and a['native_best']==(end['best'] if cc is None else cc)
    c=C.Counter(requests=1,NN=a['NN_calls'],physical=a['physical_calls'],visits=a['visits'],rules=a['rule_work'])
    def stage(b,prev,s,h,old,mask,w,reserve):
        q=h['ledger'];hh=h['hooks'];assert b['target']==32 and b['cap']==128 and b['reserve']==reserve
        assert b['used_nn']==s['nn']-prev['nn']==hh['base_evals']+hh['extra_evals'] and b['used_nn']+b['unused']==32
        assert b['visits']==s['visits']-prev['visits']==hh['base_evals']+hh['terminal_leaves']
        assert 0<=b['visits']<=128 and 0<=b['unused']<=32
        assert q['iterations']==s['visits'] and q['eligible']==q['novel']+q['blocked_receipt']
        assert q['transferred']==q['receipt_items']>=old['transferred'] and q['repeated_delivery']==0
        assert 0<=h['peak']<=w and (b['visits']==0)==(h['peak']==0)
        assert hh['graph_solved_skips']==0
        if not mask:assert hh['select_nodes']==hh['width_nodes']==hh['refinements']==hh['extra_evals']==0 and q['transferred']==old['transferred']
        if reserve:
            assert b['protected_extra']+b['released']==reserve and b['protected_extra']+b['borrowed']==b['pre_extra']
            assert b['protected_base']+b['borrowed']<=32-reserve and b['pre_extra']<=hh['extra_evals']
        if b['cause']=='CERTIFIED':assert 4<=arm<=8 and a['end_graph']['witness'] is not None
        elif b['cause']=='NN_TARGET':assert b['unused']==0
        else:assert b['cause']=='VISIT_CAP' and b['visits']==128 and b['unused']>0
        c.update(stages=1,target=int(b['cause']=='NN_TARGET'),cap=int(b['cause']=='VISIT_CAP'),cert=int(b['cause']=='CERTIFIED'),unused=b['unused'],transferred=q['transferred']-old['transferred'],blocked=q['blocked_receipt']-old['blocked_receipt'],refinements=hh['refinements'])
    stage(a['first'],pre,mid,mh,ph['ledger'],0,1,0)
    q=a['queue']
    if arm in A:
        mask,w,res=A[arm];stage(a['tail'],mid,end,eh,mh['ledger'],mask,w,res)
        if arm>=4:
            b=a['tail'];used=vis=0
            for i,z in enumerate(b['rounds']):
                used+=z['nn'];vis+=z['visits'];assert 0<z['visits']<=8 and used+z['shared']+z['protected']==32
                if z['witness'] is not None:assert i==len(b['rounds'])-1 and b['cause']=='CERTIFIED'
            assert (used,vis)==(b['used_nn'],b['visits'])
            c['waves']+=len(b['rounds'])
            assert sum(z['forced'] for z in b['rounds'])==len(a['end_graph']['forced'] if a['end_graph'] else [])
            if b['cause']=='CERTIFIED' and b['rounds']:assert b['rounds'][-1]['witness']==a['end_graph']['witness']
    else:assert end==mid and a['tail'] is None and q is None and eh['ledger']==mh['ledger']
    if arm in (8,9):
        assert q['width']==8 and q['dedup'] and q['logical']==end['nn']-mid['nn']
        assert q['logical']==q['physical']+q['coalesced']==q['distinct']+q['repeat_keys']
        assert q['physical']>=q['distinct'] and q['coalesced']<=q['repeat_keys']
        lh,hh=q['logical_hist'],q['physical_hist']
        assert len(lh)==len(hh)==9 and lh[0]==hh[0]==0
        assert sum(lh)==sum(hh)==q['batches'] and sum(j*x for j,x in enumerate(lh))==q['logical'] and sum(j*x for j,x in enumerate(hh))==q['physical']
        assert 0<=q['peak_pending']<=8 and 0<=q['wait_ns']<=q['latency_ns']
        kh=[0]*9
        for j,x in enumerate(hh):
            if j:kh[next(w for w in (1,2,4,8) if w>=j)]+=x
        assert q['specialized'] and kh==q['kernel_hist'] and sum(j*x for j,x in enumerate(kh))==q['kernel_lanes']
        assert q['parity_checks']==(q['physical'] if trace else 0) and (q['trace'] is not None)==trace
        c.update({k:q[k] for k in ('logical','coalesced','repeat_keys','distinct','batches','wait_ns','compute_ns','latency_ns','parity_checks')})
        c['batch_physical']+=q['physical'];c['kernel_lanes']+=q['kernel_lanes'];c['multi_batches']+=sum(lh[2:])
    else:assert q is None
    if 2<=arm<=8 or arm==10:c.update(graph_audit(a,r,trace))
    else:assert a['rule_work']==0 and a['best']==a['native_best'] and a['end_graph'] is None
    assert a['NN_calls']==end['nn'] and a['visits']==end['visits'] and a['physical_calls']==a['NN_calls']-(q['coalesced'] if q else 0)
    assert (eh['trace'] is not None)==trace and eh['ledger']['iterations']==end['visits'] and eh['ledger']['repeated_delivery']==0
    for s in (pre,mid,end):assert sum(s['n'][:s['len']])==s['visits'] and set(s['moves'][:s['len']])==set(rank)
    snap=a['snapshot'];assert snap['root_n']==end['visits'] and snap['evaluator_calls']==end['nn'] and snap['best']==end['best']
    for j,m in enumerate(end['moves'][:end['len']]):
        e=next((e for e in snap['edges'] if e['move']==m),{})
        assert e.get('n',0)==end['n'][j] and e.get('vl',0)==e.get('vvalue',0)==0
        near(e.get('w',0),end['w'][j]);near(e.get('m2',0),end['m2'][j])
    tot.update(c);by[arm].update(c)
def rows(phase):
    lk=read(phase+'_LOCK.json');assert lk['cohort']==sha(D/'COHORT.json')
    for fi in lk['files']:
        p=D/fi['file'];assert sha(p)==fi['sha256'];epoch=0
        for t in raw(p):
            for k,j in t['order']:
                a=t['results'][str(k)+':'+str(j)]
                if a['queue']:
                    epoch+=1;assert a['queue']['epoch']==epoch
            yield t
# Independent exact column tuples, last-move terminal checks, bottom-up graph closure.
def bkey(b):
    side=sum(map(len,b))%2+1;k=1<<84 if side==1 else 0
    for x,c in enumerate(b):
        for y in range(6):k|=(1 if y>=len(c) else 2 if c[y]==1 else 0)<<(2*(7*y+x))
    return format(k,'022x')
def child(b,m):
    assert 0<=m<7 and len(b[m])<6
    side=sum(map(len,b))%2+1;y=len(b[m]);c=b[:m]+(b[m]+(side,),)+b[m+1:]
    for dx,dy in ((1,0),(0,1),(1,1),(1,-1)):
        n=1
        for sg in (-1,1):
            x,z=m+sg*dx,y+sg*dy
            while 0<=x<7 and 0<=z<len(c[x]) and c[x][z]==side:
                n+=1;x+=sg*dx;z+=sg*dy
        if n>=4:return c,-1
    return c,0 if sum(map(len,c))==42 else None
class ExactGraph:
    def __init__(self,b):
        self.root=b;self.nodes={};self.checks=0;self.replay=0;self.registered=0
    def reg(self,b,terminal=None):
        if self.nodes.get(b) is not None:return
        self.registered+=1
        if terminal is not None:self.nodes[b]=terminal;return
        aa=[]
        for m,c in enumerate(b):
            if len(c)==6:continue
            k,t=child(b,m);self.checks+=1;aa.append((m,k))
            if t is not None:
                if self.nodes.get(k) is None:self.registered+=1
                else:assert self.nodes[k]==t
                self.nodes[k]=t
            else:self.nodes.setdefault(k,None)
        assert aa;self.nodes[b]=aa
    def path(self,p,register=False):
        b=self.root;t=None
        for m in p:
            assert t is None
            b,t=child(b,m);self.replay+=1
            if register:self.reg(b,t)
        return b,t
    def closure(self,over=None):
        v={}
        for b,n in sorted(self.nodes.items(),key=lambda x:-sum(map(len,x[0]))):
            if over is not None and b==over[0]:v[b]=(over[1],over[1])
            elif n is None:v[b]=(-1,1)
            elif isinstance(n,int):v[b]=(n,n)
            else:v[b]=(max(-v[c][1] for m,c in n),max(-v[c][0] for m,c in n))
        return v
    def actions(self,b,v):
        return [(m,(-v[c][1],-v[c][0])) for m,c in self.nodes[b]]
    @staticmethod
    def risk(a):
        return (min(max(0,max((u for j,(l,u) in a if j!=m),default=-1)-l) for m,(l,u) in a),sum(u-l for m,(l,u) in a))
    @staticmethod
    def allowed(a):
        return [m for m,(l,u) in a if u>=max(l for j,(l,v) in a)]
    @staticmethod
    def witness(a):
        return next((m for m,(l,u) in a if all(j==m or l>=v for j,(_,v) in a)),None)
    def frontier(self,v,limit=None):
        seen={self.root};todo=C.deque([(self.root,[])])
        while todo:
            b,p=todo.popleft()
            if v[b][0]==v[b][1]:continue
            n=self.nodes[b]
            if n is None:
                yield b,p
                if limit is not None:
                    limit-=1
                    if not limit:return
            elif not isinstance(n,int):
                ok=self.allowed(self.actions(b,v))
                for m,c in n:
                    if m in ok and c not in seen:seen.add(c);todo.append((c,p+[m]))
    def verify(self,a,rep,v,dump=False,midpoint=True,replay=True):
        b=self.root;aa=self.actions(b,v);allow=self.allowed(aa);w=self.witness(aa)
        st=a['mid'] if midpoint else a['end'];anchor=a['mid_best'] if midpoint else a['native_best'];vis=dict(zip(st['moves'][:st['len']],st['n'][:st['len']]))
        best=w if w is not None else anchor if anchor in allow else max(allow,key=lambda m:(vis[m],-m))
        expected=dict(key=bkey(b),actions=[[m,list(q)] for m,q in aa],interval=list(v[b]),allowed=allow,witness=w,best=best,risk=list(self.risk(aa)),nodes=len(self.nodes),registered=self.registered,checks=self.checks,replay=self.replay,rule_work=self.checks+self.replay)
        for k,x in expected.items():assert rep[k]==x,(k,rep[k],x,a['history'])
        if dump:
            z={x['key']:x for x in rep['graph']};assert len(z)==len(self.nodes)
            for b,n in self.nodes.items():
                x=z[bkey(b)];assert x['interval']==list(v[b])
                assert x['terminal']==(n if isinstance(n,int) else None)
                assert x['actions']==([[m,bkey(c)] for m,c in n] if isinstance(n,list) else None)
def graph_audit(a,r,trace):
    arm=a['arm'];replay=arm==10;gr=a['replay_graph'] if replay else None
    m=gr['before'] if replay else a['mid_graph'];e=gr['end'] if replay else a['end_graph'];paths=gr['paths'] if replay else e['paths']
    assert len(paths)==a['visits']
    if trace:assert paths==[x['path'] for x in a['end_hooks']['trace']]
    root=tuple(tuple(c) for c in g.replay(r['history']));ex=ExactGraph(root);ex.reg(root)
    for mv,b in ex.nodes[root]:
        k,t=child(root,mv);assert k==b;ex.replay+=1;ex.reg(k,t)
    for p in paths[:a['mid']['visits']]:ex.path(p,True)
    if not replay:ex.replay=len(ex.nodes[root])
    v=ex.closure();ex.verify(a,m,v)
    steps=(gr['steps'] if replay else a['rule_steps']) or [];prev=m
    assert len(steps)<=8 and (arm in (3,10) or not steps)
    full=trace or r['id']%37==0
    for st in steps:
        aa=ex.actions(root,v);assert ex.witness(aa) is None
        available=list(ex.frontier(v,16));assert available
        p=st['path'];b,t=ex.path(p);assert t is None and ex.nodes[b] is None and st['key']==bkey(b)
        assert any(k==b and pp==p for k,pp in available)
        sc=st.get('hypothetical_score',st.get('score'))
        if st['kind']=='FIFO_FALLBACK':assert p==available[0][1] and sc is None
        if full:
            risk=ex.risk(aa);vals=[]
            for k,pp in available:
                vals.append(max((risk[0]-w[0],risk[1]-w[1]) for u in range(v[k][0],v[k][1]+1) for w in [ex.risk(ex.actions(root,ex.closure((k,u))))]))
            i=max(range(len(vals)),key=lambda j:(vals[j],-len(available[j][1]),-j))
            if vals[i]>(0,0):assert st['kind']=='FRONTIER' and available[i][1]==p and list(vals[i])==sc
            else:assert st['kind']=='FIFO_FALLBACK'
        ex.reg(b);v=ex.closure();z=st['post'];ex.verify(a,z,v)
        assert z['risk'][0]<=prev['risk'][0] and z['risk'][1]<=prev['risk'][1];prev=z
    if not replay and arm in range(4,9):
        r0=ex.replay
        for p in paths[a['mid']['visits']:]:ex.path(p,True)
        ex.replay=r0
    v=ex.closure();ex.verify(a,e,v,trace,midpoint=(arm in (2,3,10)))
    assert a['rule_work']==e['rule_work'] and a['best']==e['best']
    # Every forced event corresponds to a real legal edge traversed in tail.
    ev=set()
    for p in paths[a['mid']['visits']:]:
        b=root
        for d,mv in enumerate(p):
            ev.add((bkey(b),mv,d));b,t=child(b,mv)
    forced=e.get('forced',[])
    for z in forced:
        assert (z['key'],z['move'],z['depth']) in ev and z['rank']<z['after'] and z['hidden']==(z['rank']>=z['before'])
        if z['mode']==2:assert z['hidden']
    if r['id'] in lab:
        u=lab[r['id']]
        for rep in [m,e]+[st['post'] for st in steps]:
            assert all(lo<=2*u[mv]-1<=hi for mv,(lo,hi) in rep['actions'])
            if rep['witness'] is not None:assert u[rep['witness']]==max(u.values())
    return C.Counter(graph_requests=1,graph_steps=len(steps),graph_full_plan_checks=int(full and bool(steps)),graph_witness=int(e['witness'] is not None),forced=len(forced),hidden=sum(z['hidden'] for z in forced),frontier_hits=e.get('frontier_hits',0),graph_shared_notices=e['shared_notices'])
def small(a,u):
    e=a['end'];return dict(u=u[a['best']],native=u[a['native_best']],opt=max(u.values()),net=u[a['best']]-F(a['NN_calls'],500)-F(a['rule_work'],50000),
        NN=a['NN_calls'],physical=a['physical_calls'],tail_NN=a['NN_calls']-a['mid']['nn'],rules=a['rule_work'],visits=a['visits'],best=a['best'],arm=a['arm'],n=e['n'][:e['len']],moves=e['moves'][:e['len']],q=a['queue'])
tot=C.Counter();by=C.defaultdict(C.Counter);pc={};fixed={};cal=C.defaultdict(list);calids=[];eq=0;graph_tot=C.defaultdict(C.Counter)
for phase in ('TRAIN_CAL_ATLAS','TEST_ATLAS'):
    start=tot.copy();seen=set();lk=read(phase+'_LOCK.json')
    for t in rows(phase):
        r=t['root'];i=r['id'];assert roots[i]==r and i not in seen;seen.add(i);u=lab[i];z=t['results']
        oo=[(c,j) for c in range(N) for j in range(2)];random.Random(SEED+900000+i).shuffle(oo);assert t['order']==[list(x) for x in oo]
        a0=z['0:0']
        for j in (0,1):
            a=z['3:'+str(j)];b=z['10:'+str(j)]
            assert a['best']==b['best'] and a['end_graph']['paths']==b['replay_graph']['paths']
            assert [x['path'] for x in a['rule_steps']]==[x['path'] for x in b['replay_graph']['steps']]
            for x,y in [(a['mid_graph'],b['replay_graph']['before']),(a['end_graph'],b['replay_graph']['end'])]:
                for key in ('actions','interval','allowed','witness','risk','nodes','registered','best','checks'):assert x[key]==y[key]
        eq+=1
        for c,j in oo:
            a=z[str(c)+':'+str(j)]
            assert all(a[k]==a0[k] for k in ('prefix','mid','rank','features'))
            native(a,r,c,tot,by)
        if phase=='TEST_ATLAS':fixed[i]={str(c):[small(z[str(c)+':'+str(j)],u) for j in (0,1)] for c in range(N)}
        elif r['group']=='CAL':
            calids.append(i)
            for c in range(N):cal[str(c)].append(S.mean(small(z[str(c)+':'+str(j)],u)['net'] for j in (0,1)))
    assert len(seen)==lk['roots']
    for k in ('requests','NN','physical','visits','rules'):assert tot[k]-start[k]==lk[k]
    pc[phase]={k:lk[k] for k in ('roots','requests','NN','physical','visits','rules')}
    print('AUDITED_PHASE',phase,pc[phase],flush=True)
assert eq==len(roots) and sorted(calids)==frozen['cal_ids']
mu={k:S.mean(v) for k,v in cal.items()};accepted=[]
for k,v in mu.items():near(v,frozen['means'][k])
for k in list(mu)[1:]:
    d=[x-y for x,y in zip(cal[k],cal['0'])];lo=float(S.mean(d))-2*S.stdev(d)/math.sqrt(len(d))
    near(lo,frozen['lower'][k])
    if lo>0:accepted.append(k)
sel=int(max(accepted,key=lambda k:mu[k])) if accepted else 0
assert sel==summary['selected']==frozen['selected']
ids=sorted(fixed);assert len(ids)==summary['n']==summary['replay_equiv_roots']
def stats(v,z):
    aa=[a for w in v for a in w]
    for k in ('u','native','net','NN','physical','visits','tail_NN','rules'):near(S.mean(a[k] for a in aa),z[k])
    near(S.mean(a['opt']-a['u'] for a in aa),z['regret']);assert sum(a['arm']==STOP for a in aa)==z['stopped']
    q=C.Counter()
    for a in aa:
        if a['q']:
            t=a['q'];q.update({k:t[k] for k in ('logical','physical','coalesced','distinct','repeat_keys','batches','wait_ns','compute_ns','latency_ns')})
            q['padded_lanes']+=8*t['batches'];q['kernel_lanes']+=t['kernel_lanes'];q['multi_batches']+=sum(t['logical_hist'][2:])
            for j,x in enumerate(t['logical_hist']):q['LH'+str(j)]+=x
            for j,x in enumerate(t['physical_hist']):q['PH'+str(j)]+=x
            for j,x in enumerate(t['kernel_hist']):q['KH'+str(j)]+=x
    for k,x in q.items():assert z['counts'][k]==x
def pair(a,b,z):
    assert len(a)==len(b)==z['n']
    if not a:return
    d=[S.mean(x['u']-y['u'] for x,y in zip(v,w)) for v,w in zip(a,b)]
    near(S.mean(d),z['gain']);near(S.mean(x['net']-y['net'] for v,w in zip(a,b) for x,y in zip(v,w)),z['net_gain'])
    assert sum(x>0 for x in d)==z['better'] and sum(x<0 for x in d)==z['worse']
    near(S.mean(x['NN']-y['NN'] for v,w in zip(a,b) for x,y in zip(v,w)),z['NN_delta'])
    near(S.mean(x['physical']-y['physical'] for v,w in zip(a,b) for x,y in zip(v,w)),z['physical_delta'])
def compare(a,b,z):
    pair(a,b,z['all'])
    for key,name in [('NN','equal_logical'),('physical','equal_physical')]:
        m=[(v,w) for v,w in zip(a,b) if all(x[key]==y[key] for x,y in zip(v,w))]
        pair([v for v,w in m],[w for v,w in m],z[name])
def js(a,b):
    x={m:n/sum(a['n']) for m,n in zip(a['moves'],a['n'])};y={m:n/sum(b['n']) for m,n in zip(b['moves'],b['n'])}
    return sum((p*math.log(2*p/(p+q)) if p else 0.)/2+(q*math.log(2*q/(p+q)) if q else 0.)/2 for m in x.keys()|y.keys() for p,q in [(x.get(m,0),y.get(m,0))])
fd={str(c):[fixed[i][str(c)] for i in ids] for c in range(N)}
for k,v in fd.items():
    stats(v,summary['fixed'][k]);z=summary['repeats'][k]
    near(math.sqrt(S.mean((a['u']-b['u'])**2/2 for a,b in v)),z['utility_sd'])
    near(math.sqrt(S.mean((a['NN']-b['NN'])**2/2 for a,b in v)),z['logical_sd'])
    near(math.sqrt(S.mean((a['physical']-b['physical'])**2/2 for a,b in v)),z['physical_sd'])
    near(S.mean(js(a,b) for a,b in v),z['JSD']);assert sum(a['best']!=b['best'] for a,b in v)==z['disagree']
for k,z in summary['contrasts'].items():
    a,b=k.split(':');compare(fd[a],fd[b],z)
print('MAIN_AUDITED',dict(tot),flush=True)
# Batch trace actual order and exact keys; independent receipt chronological replay.
import struct
def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
def exactboard(b):return tuple(tuple(c) for c in b)
def statekey(b,side):
    k=1<<84 if side==1 else 0
    for x in range(7):
        for y in range(6):
            digit=1 if y>=len(b[x]) else 2 if b[x][y]==1 else 0
            k|=digit<<(2*(7*y+x))
    return format(k,'022x')
def check_trace(r,a):
    q=a['queue'];records=q['trace'];assert len(records)==q['batches'];basekeys=C.Counter();queued=C.Counter();lh=C.Counter();ph=C.Counter()
    for z in records:
        ks=z['keys'];route=z['route'];assert len(ks)==len(route)==len(z['wait_ns'])
        u=[];m={}
        for key in ks:
            if q['dedup']:
                if key not in m:m[key]=len(m)
                u.append(m[key])
            else:u.append(len(u))
        assert u==route and z['physical']==len(set(route))
        assert z['kernel_width']==next(w for w in (1,2,4,8) if w>=z['physical'])
        queued.update(ks);lh[len(ks)]+=1;ph[z['physical']]+=1
    assert sum(z['kernel_width'] for z in records)==q['kernel_lanes']
    assert sum(queued.values())==q['logical'] and len(queued)==q['distinct']
    assert [lh[i] for i in range(9)]==q['logical_hist'] and [ph[i] for i in range(9)]==q['physical_hist']
    assert sum(sum(z['wait_ns']) for z in records)==q['wait_ns'] and sum(z['compute_ns'] for z in records)==q['compute_ns']
    root=g.replay(r['history']);ledger={};payloads={};samples=C.defaultdict(list);count=C.Counter();trace=a['end_hooks']['trace']
    bounds={a['prefix']['visits']:(a['prefix'],a['prefix_hooks']),a['mid']['visits']:(a['mid'],a['mid_hooks']),a['end']['visits']:(a['end'],a['end_hooks'])}
    assert len(trace)==a['visits']
    def ls(i):
        return dict(iterations=i,edges=count['edges'],children=len(ledger),contexts=sum(len(x) for x in ledger.values()),shared=sum(len(x)>1 for x in ledger.values()),unique_base_leaf=len(payloads),eligible=count['eligible'],novel=count['novel'],blocked_receipt=count['blocked_receipt'],transferred=count['transferred'],receipt_items=sum(len(z[1]) for x in ledger.values() for z in x.values()),applied=count['applied'],changed=count['changed'],repeated_delivery=0)
    for i,t in enumerate(trace):
        b=[c[:] for c in root];side=len(r['history'])%2+1;keys=[];mask=0 if i<a['mid']['visits'] else A[a['arm']][0]
        assert t['mask']==mask
        for mv in t['path']:
            assert not g.won(b,1) and not g.won(b,2) and len(b[mv])<6
            p0=exactboard(b);b[mv].append(side);side=3-side;keys.append((exactboard(b),(p0,mv)))
        terminal=g.won(b,1) or g.won(b,2) or sum(map(len,b))==42
        assert t['terminal']==terminal
        if i>=a['mid']['visits'] and not terminal:basekeys[statekey(b,side)]+=1
        leaf=(exactboard(b),terminal);raw0=f32(t['raw']);y=f32(t['leaf'])
        if terminal:assert raw0==y==(-1 if g.won(b,3-side) else 0)
        if not mask&4:assert raw0==y
        if leaf in payloads:assert payloads[leaf]==raw0
        else:payloads[leaf]=raw0
        for j in reversed(range(len(keys))):
            count['edges']+=1;child,ctx=keys[j];cc=ledger.setdefault(child,{})
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
            if not terminal and mask&8 and new:
                assert not received.keys()&new.keys();received.update(new);count['transferred']+=len(new);count['applied']+=1
                old=y;alpha=min(.25,len(new)/(len(new)+4));y=f32(max(-1.,min(1.,(1-alpha)*y+alpha*mean)));count['changed']+=abs(y-old)>1e-7
            raw0=-raw0;y=-y;assert f32(s['backup'])==y
        samples[t['path'][0]].append(y)
        if i+1 in bounds:
            s,h=bounds[i+1];assert ls(i+1)==h['ledger']
            for j,m in enumerate(s['moves'][:s['len']]):
                v=samples[m];assert len(v)==s['n'][j];mu=S.mean(v) if v else 0.
                near(sum(v),s['w'][j]);near(sum((x-mu)**2 for x in v),s['m2'][j])
    assert all(queued[k]>=n for k,n in basekeys.items())
    assert sum(basekeys.values())==a['end_hooks']['hooks']['base_evals']
    assert sum((queued-basekeys).values())==a['end_hooks']['hooks']['extra_evals']
    return dict(iterations=len(trace),edges=count['edges'],receipt=count['transferred'],blocked=count['blocked_receipt'],queue_base=sum(basekeys.values()),queue_extra=sum((queued-basekeys).values()),parity=q['parity_checks'])
extra={};timing=C.defaultdict(list);tr=C.Counter()
for phase in ('DIAGNOSTIC','TIMING'):
    lk=read(phase+'_LOCK.json');p=D/(phase+'.jsonl.gz');assert sha(p)==lk['sha256'];t0=C.Counter();b0=C.defaultdict(C.Counter);seen=[]
    for t in raw(p):
        r=t['root'];seen.append(r['id'])
        for k,a in t['results'].items():
            c=int(k.split(':')[0]);native(a,r,c,t0,b0,phase=='DIAGNOSTIC')
            if phase=='DIAGNOSTIC' and a['queue']:tr.update(check_trace(r,a))
            if phase=='TIMING':timing[str(c)].append((a['cpu_ns']/1e6,a['wall_ns']/1e6,a['tail_us']/1000,(a['queue_setup_us']+a['queue_join_us'])/1000,a['NN_calls'],a['physical_calls']))
    assert seen==lk['roots'] and t0['requests']==lk['requests'];extra[phase]=dict(t0)
out=dict(source_attempts=src,source_plies=source_plies,source_NN=source_NN,source_queries=queries,source_locks=len(locks),roots=len(roots),statuses=dict(statuses),exact_actions=actions,oracle_nodes=nodes,
    main=dict(tot),by_arm={str(k):dict(v) for k,v in by.items()},phase_counts=pc,equivalent_roots=eq,selected=sel,extra=extra,trace=dict(tr),
    timing={k:dict(n=len(v),**{a:S.mean(x[i] for x in v) for i,a in enumerate(('cpu_ms','wall_ms','tail_ms','setup_join_ms','NN','physical'))}) for k,v in timing.items()},
    scope='Independent source grids, exact stored WDL, full raw accounting, incremental and replay graph independently reconstructed, rule paths and actual forced edge identity, actual policy/repeat readback, exact batch routes and receipt traces; no independent CNN/solver rerun except native batch-scalar trace parity.')
(D/'C09_AUDIT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print('AUDIT',json.dumps({k:v for k,v in out.items() if k not in ('by_arm','timing')},sort_keys=True),flush=True)
