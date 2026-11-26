"""Independent exact readback of locked K27 data and policy, not another CNN/solver execution."""
import pathlib,hashlib,subprocess as SP,importlib.util,json,gzip,random,math,statistics as S,collections as C
from fractions import Fraction as F
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2];P=D.parent
SEED=2026112631
A=[(0,1,0,False,0),(0,4,0,False,0),(0,8,0,False,0),(0,8,8,True,0),(15,8,0,False,4),(15,8,8,True,4)]
N=9;STOP=9;KINDS=('short','serial','tree4','continue','priced')
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
ml=read('MODEL_LOCK.json');cal_scoring=read('CAL_SCORING.json')
assert sha(D/'MODEL.json')==ml['model']==frozen['model'] and sha(D/'MODEL_LOCK.json')==frozen['model_lock']
assert sha(D/'CAL_LIVE_LOCK.json')==frozen['cal_live'] and sha(D/'CAL_SCORING.json')==ml['cal_scoring'] and sha(D/'HISTORICAL.json')==ml['historical'] and sha(D/'CONSTANT.json')==ml['constant']
assert sha(D/'TRAIN_CAL_ATLAS_LOCK.json')==ml['atlas'] and sha(D/'TRAIN_CAL_LABELS.json')==ml['labels']
assert sha(D/'trial.py')==ml['code']==frozen['code']
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
def native(a,r,arm,policy,tot,by,trace=False):
    assert a['schema']=='quartz_k31_c4_v1' and a['history']==r['history'] and a['legal']==r['legal'] and a['arm']==arm and a['policy']==policy
    pr=proof_grid(r);assert pr==a['proof'];cc=certificate(pr)
    if r['id'] in lab:assert all(lo<=lab[r['id']][m]<=hi for m,lo,hi in pr)
    pre,mid,end=[a[k] for k in ('prefix','mid','end')]
    ph,mh,eh=[a[k] for k in ('prefix_hooks','mid_hooks','end_hooks')]
    assert pre['visits']==8 and pre['nn']==1+ph['hooks']['base_evals']
    assert ph['ledger']['iterations']==8 and ph['ledger']['transferred']==0 and ph['hooks']['extra_evals']==0
    rank,f,allowed=feature_grid(r,mid,pr);assert rank==a['rank'] and allowed==a['allowed'] and len(f)==len(a['features'])
    for x,y in zip(f,a['features']):near(x,y)
    assert a['mid_best']==rank[0] and a['native_best']==(end['best'] if cc is None else cc)
    df=direct_features(f,pre,mid)
    assert len(df)==len(a['direct_features'])==16
    for x,y in zip(df,a['direct_features']):near(x,y)
    c=C.Counter(requests=1,NN=a['NN_calls'],physical=a['physical_calls'],visits=a['visits'],rules=a['rule_work'],stopped=int(arm==STOP))
    def stage(b,prev,s,h,old,mask,w,reserve):
        q=h['ledger'];hh=h['hooks'];assert b['target']==32 and b['cap']==128 and b['reserve']==reserve
        assert b['used_nn']==s['nn']-prev['nn']==hh['base_evals']+hh['extra_evals'] and b['used_nn']+b['unused']==32
        assert b['visits']==s['visits']-prev['visits']==hh['base_evals']+hh['terminal_leaves']
        assert 0<b['visits']<=128 and 0<=b['unused']<=32 and (b['unused']==0 or b['visits']==128)
        assert q['iterations']==s['visits'] and q['eligible']==q['novel']+q['blocked_receipt']
        assert q['transferred']==q['receipt_items']>=old['transferred'] and q['repeated_delivery']==0
        assert 1<=h['peak']<=w
        if not mask:assert hh['select_nodes']==hh['width_nodes']==hh['refinements']==hh['extra_evals']==0 and q['transferred']==old['transferred']
        if reserve:
            assert b['protected_extra']+b['released']==reserve and b['protected_extra']+b['borrowed']==b['pre_extra']
            assert b['protected_base']+b['borrowed']<=32-reserve and b['pre_extra']<=hh['extra_evals']
        c.update(stages=1,target=int(b['unused']==0),cap=int(b['unused']>0),unused=b['unused'],transferred=q['transferred']-old['transferred'],blocked=q['blocked_receipt']-old['blocked_receipt'],refinements=hh['refinements'])
    stage(a['first'],pre,mid,mh,ph['ledger'],0,1,0)
    q=a['queue']
    if arm>=6:
        assert end==mid and a['tail'] is None and q is None and eh['ledger']==mh['ledger']
        if arm==STOP:assert a['proof_graph'] is None and a['rule_work']==0 and a['best']==a['native_best']
        else:
            c.update(graph_audit(a,r,trace))

    else:
        assert a['proof_graph'] is None and a['rule_work']==0 and a['best']==a['native_best']
        mask,w,width,dedup,res=A[arm];stage(a['tail'],mid,end,eh,mh['ledger'],mask,w,res)
        if width:
            assert q['width']==width and q['dedup']==dedup and q['logical']==end['nn']-mid['nn']
            assert q['logical']==q['physical']+q['coalesced']==q['distinct']+q['repeat_keys']
            assert q['physical']>=q['distinct'] and q['coalesced']<=q['repeat_keys']
            lh,hh=q['logical_hist'],q['physical_hist']
            assert len(lh)==len(hh)==9 and lh[0]==hh[0]==0
            assert sum(lh)==sum(hh)==q['batches'] and sum(j*x for j,x in enumerate(lh))==q['logical'] and sum(j*x for j,x in enumerate(hh))==q['physical']
            assert not any(lh[width+1:]) and not any(hh[width+1:]) and 0<=q['peak_pending']<=w
            assert 0<=q['wait_ns']<=q['latency_ns'] and q['compute_ns']>=0
            assert q['epoch']>=1 and q['specialized']==True
            kh=[0]*9
            for j,x in enumerate(hh):
                if j:kh[next(w for w in (1,2,4,8) if w>=j)]+=x
            assert kh==q['kernel_hist'] and sum(j*x for j,x in enumerate(kh))==q['kernel_lanes']
            if not dedup:assert q['coalesced']==0 and lh==hh
            assert q['parity_checks']==(q['physical'] if trace else 0) and (q['trace'] is not None)==trace
            c.update({k:q[k] for k in ('logical','coalesced','repeat_keys','distinct','batches','wait_ns','compute_ns','latency_ns','parity_checks')})
            c['batch_physical']+=q['physical'];c['padded_lanes']+=8*q['batches'];c['kernel_lanes']+=q['kernel_lanes'];c['multi_batches']+=sum(lh[2:])
            for j,z in enumerate(lh):c['LH'+str(j)]+=z
            for j,z in enumerate(hh):c['PH'+str(j)]+=z
            for j,z in enumerate(q['kernel_hist']):c['KH'+str(j)]+=z
        else:assert q is None
    assert a['NN_calls']==end['nn'] and a['visits']==end['visits'] and a['physical_calls']==a['NN_calls']-(q['coalesced'] if q else 0)
    assert (eh['trace'] is not None)==trace and eh['ledger']['repeated_delivery']==0
    for s in (pre,mid,end):assert sum(s['n'][:s['len']])==s['visits'] and set(s['moves'][:s['len']])==set(rank)
    snap=a['snapshot'];assert snap['root_n']==end['visits'] and snap['evaluator_calls']==end['nn'] and snap['best']==end['best']
    for j,m in enumerate(end['moves'][:end['len']]):
        e=next((e for e in snap['edges'] if e['move']==m),{})
        assert e.get('n',0)==end['n'][j] and e.get('vl',0)==e.get('vvalue',0)==0
        near(e.get('w',0),end['w'][j]);near(e.get('m2',0),end['m2'][j])
    if policy is None:assert a['prediction'] is None
    else:assert a['prediction'] is not None and a['belief_identity']==ml['model']
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
def direct_features(f,pre,mid):
    v=[f[i:i+8] for i in range(1,len(f),8)];l=len(v)
    r=[max(x[i] for x in v[1:]) for i in range(4)]
    ent=[-math.fsum(x[i]**2*math.log(x[i]**2) for x in v if x[i]>0)/math.log(l) for i in (0,1)]
    return [f[0],l/7,v[0][0]**2,r[0]**2,v[0][1]**2,r[1]**2,v[0][2],r[2],v[0][3],r[3],*ent,mid['nn']/41,mid['visits']/136,sum(x[7]==0 for x in v)/l,float(pre['best']==mid['best'])]
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
    def verify(self,a,rep,v,dump=False):
        b=self.root;aa=self.actions(b,v);allow=self.allowed(aa);w=self.witness(aa)
        mid=a['mid'];anchor=a['mid_best'];vis=dict(zip(mid['moves'][:mid['len']],mid['n'][:mid['len']]))
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
    gr=a['proof_graph'];arm=a['arm'];assert gr['arm']==arm and len(gr['paths'])==a['mid']['visits']
    if trace:assert gr['paths']==[x['path'] for x in a['end_hooks']['trace'][:a['mid']['visits']]]
    root=tuple(tuple(c) for c in g.replay(r['history']));ex=ExactGraph(root);ex.reg(root)
    for m,c in ex.nodes[root]:
        k,t=child(root,m);assert k==c;ex.replay+=1;ex.reg(k,t)
    for p in gr['paths']:ex.path(p,True)
    v=ex.closure();ex.verify(a,gr['before'],v);prev=gr['before'];scored=0
    assert len(gr['steps'])<=8 and (arm!=6 or not gr['steps'])
    full=trace or r['id']%37==0
    for st in gr['steps']:
        aa=ex.actions(root,v);assert ex.witness(aa) is None
        available=list(ex.frontier(v,16))
        assert available
        p=st['path'];b,t=ex.path(p);assert t is None and ex.nodes[b] is None and st['key']==bkey(b)
        assert any(k==b and pp==p for k,pp in available)
        if arm==7 or st['kind']=='FIFO_FALLBACK':
            assert p==available[0][1] and st['hypothetical_score'] is None
        if arm==8:
            scored+=len(available)
            if full:
                risk=ex.risk(aa);vals=[]
                for k,pp in available:
                    gains=[]
                    for u in range(v[k][0],v[k][1]+1):
                        w=ex.risk(ex.actions(root,ex.closure((k,u))))
                        gains.append((risk[0]-w[0],risk[1]-w[1]))
                    vals.append(max(gains))
                i=max(range(len(vals)),key=lambda j:(vals[j],-len(available[j][1]),-j))
                if vals[i]>(0,0):assert st['kind']=='FRONTIER' and available[i][1]==p and list(vals[i])==st['hypothetical_score']
                else:assert st['kind']=='FIFO_FALLBACK'
        ex.reg(b);v=ex.closure();z=st['post'];ex.verify(a,z,v)
        assert z['scored']==scored and z['risk'][0]<=prev['risk'][0] and z['risk'][1]<=prev['risk'][1]
        assert z['registered']>prev['registered'] and z['rule_work']>prev['rule_work']
        assert z['changes']>=prev['changes'] and z['notices']>=prev['notices'] and z['shared_notices']>=prev['shared_notices']
        prev=z
    ex.verify(a,gr['end'],v,trace)
    assert gr['end']['rule_work']==a['rule_work'] and a['best']==gr['end']['best']
    if r['id'] in lab:
        u=lab[r['id']]
        for rep in [gr['before']]+[z['post'] for z in gr['steps']]+[gr['end']]:
            assert all(lo<=2*u[m]-1<=hi for m,(lo,hi) in rep['actions'])
            if rep['witness'] is not None:assert u[rep['witness']]==max(u.values())
    return C.Counter(graph_requests=1,graph_steps=len(gr['steps']),graph_guided=sum(z['kind']=='FRONTIER' for z in gr['steps']),graph_full_plan_checks=int(full),graph_new_witness=int(gr['before']['witness'] is None and gr['end']['witness'] is not None),graph_shared_notices=gr['end']['shared_notices'],graph_extra_shared=gr['end']['shared_notices']-gr['before']['shared_notices'])
def outcomes(a,u):
    return [u[a['best']]-u[a['mid_best']],a['NN_calls']-a['mid']['nn'],a.get('rule_work',0)]
def datum(t,u):
    z=t['results'];a=z['0:0'];f=direct_features(a['features'],a['prefix'],a['mid'])
    return dict(f=f,y=[[S.mean(outcomes(z[str(c)+':'+str(j)],u)[k] for j in range(2)) for k in range(3)] for c in range(N)])
def predict(m,f):
    ar=[]
    for c,a in enumerate(m['arms']):
        z=[(v-m)/s for v,m,s in zip(f,a['mean'],a['scale'])]
        y=[w[0]+math.fsum(x*t for x,t in zip(w[1:],z)) for w in a['coef']]
        g=max(-1,min(1,y[0]));k=max(0,min(32,y[1])) if c<6 else 0.;l=max(0,y[2]) if c>=6 else 0.
        ar.append(dict(g=g,nn=k,rules=l,net=g-.002*k-.00002*l))
    def am(n):
        i=0
        for j in range(1,n):
            if ar[j]['net']>ar[i]['net']+1e-12:i=j
        return i
    i=am(N);return dict(arms=ar,fixed_arm=am(6),priced_arm=i if ar[i]['net']>1e-12 else STOP)
def check_prediction(a,p):
    q=a['prediction'];assert q['fixed_arm']==p['fixed_arm'] and q['priced_arm']==p['priced_arm']
    for x,y in zip(q['arms'],p['arms']):
        for k in ('g','nn','rules','net'):near(x[k],y[k])
    assert a['arm']==(p['fixed_arm'] if a['policy']=='continue' else p['priced_arm'])
def small(a,u):
    e=a['end'];return dict(u=u[a['best']],native=u[a['native_best']],opt=max(u.values()),net=u[a['best']]-F(a['NN_calls'],500)-F(a['rule_work'],50000),
        NN=a['NN_calls'],physical=a['physical_calls'],tail_NN=a['NN_calls']-a['mid']['nn'],rules=a['rule_work'],visits=a['visits'],best=a['best'],arm=a['arm'],n=e['n'][:e['len']],moves=e['moves'][:e['len']],q=a['queue'])
tot=C.Counter();by=C.defaultdict(C.Counter);pc={};dat={};train={};calids=[];testids=[];fixed={};live={};cal=C.defaultdict(list);actual_model_checks=0;fp={};grsum=C.defaultdict(C.Counter);drsum=C.defaultdict(list)
for phase in ('TRAIN_CAL_ATLAS','CAL_LIVE','TEST_ATLAS','TEST_LIVE'):
    start=tot.copy();seen=set();lk=read(phase+'_LOCK.json')
    if phase=='CAL_LIVE':assert lk['freeze']==sha(D/'MODEL_LOCK.json')
    if phase.startswith('TEST'):assert lk['freeze']==sha(D/'FROZEN.json')
    for t in rows(phase):
        r=t['root'];i=r['id'];assert roots[i]==r and i not in seen;seen.add(i);u=lab.get(i)
        z=t['results'];oo=[(c,j) for c in range(N) for j in range(2)] if phase.endswith('ATLAS') else [(k,j) for k in KINDS for j in range(2)]
        random.Random(SEED+(900000 if phase.endswith('ATLAS') else 800000)+i).shuffle(oo);assert t['order']==[list(x) for x in oo]
        a0=next(iter(z.values()))
        if phase.endswith('ATLAS') and u:
            d=datum(t,u);dat[i]=d
            if r['group']=='TRAIN':train[i]=d
            elif r['group']=='CAL':calids.append(i)
            else:
                testids.append(i);fixed[i]={str(c):[small(z[str(c)+':'+str(j)],u) for j in range(2)] for c in range(N)}
                p=predict(model,d['f']);fp[i]=p;q=[v['net'] for v in p['arms']]+[0.]
                for name,a,b in [('fixed',p['fixed_arm'],0),('priced',p['priced_arm'],STOP)]:
                    def y(c,j):
                        if c==STOP:return F(0)
                        v,k,l=outcomes(z[str(c)+':'+str(j)],u);return v-F(k,500)-F(l,50000)
                    dd=S.mean(y(a,j)-y(b,j) for j in (0,1));dr=[]
                    for c,j in oo:
                        v=y(c,j);w=N*((c==a)-(c==b));dr.append((w*v,q[a]-q[b]+w*(float(v)-q[c])))
                    near(S.mean(x[0] for x in dr),dd);near(S.mean(x[1] for x in dr),dd)
                    drsum[name].append(dict(disagree=int(a!=b),full=dd,direct=q[a]-q[b],ips=dr[0][0],dr=dr[0][1],ips_var=S.pvariance(x[0] for x in dr),dr_var=S.pvariance(x[1] for x in dr)))
                for c in (6,7,8):
                    for j in (0,1):
                        a=z[str(c)+':'+str(j)];gr=a['proof_graph'];b,e=gr['before'],gr['end']
                        grsum[str(c)].update(dict(n=1,before_risk=b['risk'][0],end_risk=e['risk'][0],width_before=b['risk'][1],width_end=e['risk'][1],before_witness=int(b['witness'] is not None),end_witness=int(e['witness'] is not None),new_witness=int(b['witness'] is None and e['witness'] is not None),changes=int(e['best']!=a['mid_best']),gain=u[a['best']]-u[a['mid_best']],rule=a['rule_work'],build_rule=b['rule_work'],steps=len(gr['steps']),guided=sum(st['kind']=='FRONTIER' for st in gr['steps']),scored=e['scored'],shared=e['shared_notices'],extra_shared=e['shared_notices']-b['shared_notices']))
        if not phase.endswith('ATLAS') and u:
            fp[i]=predict(model,dat[i]['f'])
            v={k:[small(z[k+':'+str(j)],u) for j in range(2)] for k in KINDS}
            if phase=='CAL_LIVE':
                for k in KINDS:cal[k].append(S.mean(a['net'] for a in v[k]))
            else:live[i]=v
        for k,j in oo:
            a=z[str(k)+':'+str(j)];assert a['prefix']==a0['prefix'] and a['mid']==a0['mid'] and a['rank']==a0['rank'] and a['features']==a0['features']
            pol=None if phase.endswith('ATLAS') or k in KINDS[:3] else k
            arm=k if phase.endswith('ATLAS') else {'short':STOP,'serial':0,'tree4':1}.get(k,a['arm'])
            native(a,r,arm,pol,tot,by)
            if pol and u:check_prediction(a,fp[i]);actual_model_checks+=1
    assert len(seen)==lk['roots']
    assert all(tot[k]-start[k]==lk[k] for k in ('requests','NN','physical','visits','rules'));pc[phase]={k:lk[k] for k in ('roots','requests','NN','physical','visits','rules')}
    print('AUDITED_PHASE',phase,dict(pc[phase]),flush=True)
# Independent numpy linear solve for all TRAIN models and CAL choice; historical rows from locked old raw.
import numpy as np
old=read('HISTORICAL.json');hist={str(i):[] for i in range(N)}
for lk in old['locks']:assert hashlib.sha256(git('show',lk['commit']+':'+lk['path'])).hexdigest()==lk['sha256']
for cy in (29,30):
    k=P/f'quartz_kinetic{cy}_20261126'
    ll={x['root']['id']:{int(m):F(v+1,2) for m,v in x['independent'].items()} for x in json.loads((k/'TRAIN_CAL_LABELS.json').read_text())['rows'] if x['status']=='EXACT' and x['root']['group']=='TRAIN'}
    n=0
    for fi in json.loads((k/'TRAIN_CAL_ATLAS_LOCK.json').read_text())['files']:
        p=k/fi['file'];assert sha(p)==fi['sha256']
        for t in raw(p):
            i=t['root']['id']
            if i not in ll:continue
            z=t['results'];a=z['0:0'];x=direct_features(a['features'],a['prefix'],a['mid']);n+=1
            for new,c in ((0,0),(1,1),(2,2),(4,6)):
                y=[S.mean(outcomes(z[str(c)+':'+str(j)],ll[i])[k] for j in range(2)) for k in range(3)]
                hist[str(new)].append(dict(source=cy,id=i,f=x,y=y))
    assert n==len(ll)
for c,ls in hist.items():
    assert len(ls)==len(old['donors'][c])
    for a,b in zip(ls,old['donors'][c]):
        assert (a['source'],a['id'])==(b['source'],b['id'])
        for x,y in zip(a['f']+a['y'],b['f']+b['y']):near(x,y)
tr=sorted(train);random.Random(SEED+11001).shuffle(tr);ids=sorted(calids);random.Random(SEED+10001).shuffle(ids);ca=ids[::2];cb=ids[1::2]
assert tr==ml['train_ids'] and ids==ml['cal_ids'] and ca==ml['cal_a'] and cb==ml['cal_b']
def fit_ind(rr,lam,augment):
    ar=[]
    for c in range(N):
        x=[r['f'] for r in rr];y=[r['y'][c] for r in rr]
        if augment:x.extend(z['f'] for z in hist[str(c)]);y.extend(z['y'] for z in hist[str(c)])
        x=np.array(x,dtype=float);y=np.array(y,dtype=float);mu=x.mean(axis=0);sc=x.std(axis=0);sc[sc<1e-6]=1.
        X=np.column_stack((np.ones(len(x)),(x-mu)/sc))
        if lam is None:co=np.column_stack((y.mean(axis=0),np.zeros((3,16))))
        else:
            R=np.diag([0]+[len(x)*lam]*16);co=np.linalg.solve(X.T@X+R,X.T@y).T
        ar.append(dict(mean=mu.tolist(),scale=sc.tolist(),coef=co.tolist()))
    return dict(arms=ar)
def mse_ind(m,rr):return S.mean(S.mean((x['g']-r['y'][i][0])**2 for i,x in enumerate(predict(m,r['f'])['arms'])) for r in rr)
def model_eq(a,b):
    for x,y in zip(a['arms'],b['arms']):
        for k in ('mean','scale'):
            for u,v in zip(x[k],y[k]):near(u,v)
        for u,v in zip(x['coef'],y['coef']):
            for m,n in zip(u,v):near(m,n)
models=[];scores=[]
for aug in (False,True):
    for lam in (None,.03,.3,3.):
        m=fit_ind(list(train.values()),lam,aug);models.append(m);a=mse_ind(m,[dat[i] for i in ca]);o=[]
        for j in range(2):
            fit=tr[j::2];val=tr[1-j::2];f=fit_ind([train[i] for i in fit],lam,aug);o.append((len(val),mse_ind(f,[train[i] for i in val])))
        b=sum(n*x for n,x in o)/len(tr);z=cal_scoring['candidates'][len(scores)]
        assert (z['augment'],z['lambda_'])==(aug,lam);near(a,z['cal']);near(b,z['oof']);scores.append(a)
sel=min(range(len(scores)),key=scores.__getitem__);assert sel==cal_scoring['selected'];model_eq(models[sel],model);model_eq(models[0],read('CONSTANT.json'))
cv={k:S.mean(cal[k]) for k in KINDS}
for k,v in cv.items():near(v,frozen['calnet'][k])
for name,bs,c in [('fixed',('serial','tree4'),'continue'),('priced',('short','serial','tree4'),'priced')]:
    base=max(bs,key=cv.__getitem__);d=[x-y for x,y in zip(cal[c],cal[base])];lo=float(S.mean(d))-2*S.stdev(d)/math.sqrt(len(d))
    near(lo,frozen['cal_lower'][c]);assert frozen['base'+name]==base and frozen[name]==(c if lo>0 else base)
assert actual_model_checks==(len(cb)+len(testids))*4
ids=sorted(testids);assert ids==sorted(lab.keys() & set(testids)) and len(ids)==summary['n'] and set(ids)==set(fixed)==set(live)
for c,z in grsum.items():
    for k,v in z.items():near(v,summary['graph'][c][k])
for k,z in drsum.items():
    s0=summary['dr'][k];assert len(z)==s0['n'] and sum(x['disagree'] for x in z)==s0['disagree']
    for key in ('full','direct','ips','dr'):near(S.mean(x[key] for x in z),s0[key])
    for key in ('ips','dr'):
        near(math.sqrt(S.mean(x[key+'_var'] for x in z)),s0[key+'_cond_sd']);near(S.stdev(x[key] for x in z),s0[key+'_empirical_sd'])
# Actual means, repeated deviations, root-paired contrasts.
ids=sorted(testids);assert len(ids)==summary['n'] and set(ids)==set(fixed)==set(live)
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
fd={str(c):[fixed[i][str(c)] for i in ids] for c in range(N)};lv={k:[live[i][k] for i in ids] for k in KINDS}
for d,s0 in [(fd,'fixed'),(lv,'live')]:
    for k,v in d.items():
        stats(v,summary[s0][k]);z=summary[s0+'_repeats'][k]
        near(math.sqrt(S.mean((a['u']-b['u'])**2/2 for a,b in v)),z['utility_sd']);near(math.sqrt(S.mean((a['NN']-b['NN'])**2/2 for a,b in v)),z['logical_sd'])
        near(math.sqrt(S.mean((a['physical']-b['physical'])**2/2 for a,b in v)),z['physical_sd']);near(S.mean(js(a,b) for a,b in v),z['JSD'])
        assert sum(a['best']!=b['best'] for a,b in v)==z['disagree']
    for k,z in summary[s0+'_contrasts'].items():
        a,b=k.split(':');compare(d[a],d[b],z)
for k,z in summary['graph_vs_short'].items():compare(fd[k],lv['short'],z)
cm=read('CONSTANT.json')
for c,z in enumerate(summary['forecast']):
    xx=[fp[i]['arms'][c]['g'] for i in ids];yy=[dat[i]['y'][c][0] for i in ids];qq=[predict(cm,dat[i]['f'])['arms'][c]['g'] for i in ids]
    near(S.mean(xx),z['pred']);near(S.mean(yy),z['actual']);near(S.mean(x-y for x,y in zip(xx,yy)),z['bias'])
    near(math.sqrt(S.mean((x-y)**2 for x,y in zip(xx,yy))),z['rmse']);near(math.sqrt(S.mean((x-y)**2 for x,y in zip(qq,yy))),z['constant_rmse'])
    if z['corr'] is not None:near(S.correlation(xx,yy),z['corr'])
    for k,j,p in [('nn',1,'pred_nn'),('rules',2,'pred_rules')]:
        near(S.mean(dat[i]['y'][c][j] for i in ids),z[k]);near(S.mean(fp[i]['arms'][c][k] for i in ids),z[p])
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
            c=int(k.split(':')[0]);native(a,r,c,None,t0,b0,phase=='DIAGNOSTIC')
            if phase=='DIAGNOSTIC' and a['queue']:tr.update(check_trace(r,a))
            else:timing[str(c)].append((a['cpu_ns']/1e6,a['wall_ns']/1e6,a['tail_us']/1000,(a['queue_setup_us']+a['queue_join_us'])/1000,a['NN_calls'],a['physical_calls']))
    assert seen==lk['roots'] and t0['requests']==lk['requests'];extra[phase]=dict(t0)
out=dict(source_attempts=src,source_plies=source_plies,source_NN=source_NN,source_queries=queries,source_locks=len(locks),roots=len(roots),statuses=dict(statuses),exact_actions=actions,oracle_nodes=nodes,
    main=dict(tot),by_arm={str(k):dict(v) for k,v in by.items()},phase_counts=pc,model_native_checks=actual_model_checks,extra=extra,trace=dict(tr),
    timing={k:dict(n=len(v),**{a:S.mean(x[i] for x in v) for i,a in enumerate(('cpu_ms','wall_ms','tail_ms','setup_join_ms','NN','physical'))}) for k,v in timing.items()},
    scope='Independent source grids, exact stored WDL, full raw accounting, direct ridge independently refit and native forecasts; bottom-up exact DAG and rule paths, actual policy/repeat readback, exact batch routes and receipt traces; no independent CNN/solver rerun except native batch-scalar trace parity.')
(D/'C09_AUDIT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print('AUDIT',json.dumps({k:v for k,v in out.items() if k not in ('by_arm','timing')},sort_keys=True),flush=True)
