"""K35 same frozen Rapfi evaluator, all games MCTS vs MCTS."""
import pathlib,sys,json,gzip,hashlib,random,math,statistics as S,time,os,select,subprocess as SP,signal,ctypes
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2];R=D.parent
BIN=ROOT.parent/'quartz_kinetic35_target_20261126/mcts_demo'
OLD=ROOT.parent/'quartz_kinetic34_target_20261126/mcts_demo'
BRIDGE=ROOT.parent/'rapfi_250615_quartz_bridge_build_20261125/pbrain-rapfi'
LOCK=json.loads((D/'RAPFI_LOCK.json').read_text());AS={x['name']:x for x in LOCK['files']};W=pathlib.Path(AS['Rapfi_NNUE_freestyle']['path'])
sys.path.insert(0,str(R/'quartz_tactical08_20261125'))
from rules import replay,orbit_board,win_at,fixtures
SEED=2026112635;N=192;ARENA=64;PRICE=.002;WORKERS=6
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(n):return json.loads((D/n).read_text())
def save(n,x):
    p=D/n;assert not p.exists(),p;p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def raw(p):
    with gzip.open(p,'rt') as f:
        for l in f:yield json.loads(l)
def emit(f,x):f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n')
def identity():
    for a in LOCK['files']:assert sha(pathlib.Path(a['path']))==a['sha256']
    assert sha(BRIDGE)=='bacbcfc90f6e4bfd0d9d035455afa3d6667ec32e7fd80cbc1d65a4001b30f64e'
    b=read('RUST_BUILD_IDENTITY.json');assert sha(BIN)==b['binary_sha256']
    return dict(binary=b['binary_sha256'],source_head=b['source_head'],bridge=sha(BRIDGE),weight=sha(W),config=AS['Rapfi_config']['sha256'],upstream=LOCK['source_commit'])
def family(h):
    b,p,w=replay(h[:6],'freestyle');assert w is None
    return hashlib.sha256(orbit_board(b,p,'freestyle')).hexdigest()
def historical():
    files=[]
    for d in sorted(R.glob('quartz_rapfi*'))+sorted(R.glob('quartz_kinetic*'))+[R/'quartz_preflight07_20261125',R/'quartz_tactical08_20261125']:
        if d==D:continue
        for pattern in ('*PLAN.json','*COHORT.json'):
            files.extend(p for p in sorted(d.glob(pattern)) if p.stat().st_size<=10_000_000)
    fam=set();locks=[]
    def visit(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k in ('opening','history','final_history') and isinstance(v,list) and len(v)>=6 and all(type(a)==int and 0<=a<225 for a in v[:6]) and len(set(v[:6]))==6:fam.add(family(v[:6]))
                elif k not in ('board','policy','features'):visit(v)
        elif isinstance(x,list):
            for v in x:
                if isinstance(v,(list,dict)):visit(v)
    for p in files:
        before=len(fam);visit(json.loads(p.read_text()))
        if len(fam)>before:locks.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),added=len(fam)-before))
    return fam,locks
def plan():
    fam,locks=historical();n=len(fam);rng=random.Random(SEED);jobs=[]
    for i in range(N+ARENA):
        cells=[15*r+c for r in range(4,11) for c in range(4,11)] if i%2==0 else [15*r+c for r in range(2,13) for c in range(2,13)]
        while True:
            h=rng.sample(cells,6);k=family(h)
            if k not in fam:break
        fam.add(k);jobs.append(dict(id=i,opening=h,family=k,group='TRAIN' if i<96 else 'CAL' if i<144 else 'TEST' if i<192 else 'ARENA',
            target=10 if i%2==0 else 14,seed=rng.randrange(2**63),color_order=rng.sample([1,-1],2)))
    save('PLAN.json',dict(identity=identity(),seed=SEED,prior_families=n,prior_locks=locks,jobs=jobs,price=PRICE,margin=.05,NN_reduction=.30,workers=WORKERS,
        scope='same frozen Rapfi evaluator and shared proof throughout; no external alpha-beta opponent'))
    print('PLAN',n,len(locks),len(jobs),flush=True)
def clock(pid):
    f=ctypes.CDLL(None).clock_getcpuclockid;f.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_int)];f.restype=ctypes.c_int
    c=ctypes.c_int();rc=f(pid,ctypes.byref(c))
    if rc:raise OSError(rc,os.strerror(rc))
    return c.value
class Server:
    def __init__(self,bin=BIN):
        env=dict(os.environ,QUARTZ_RAPFI_EVAL_EXE=str(BRIDGE),QUARTZ_RAPFI_EVAL_WEIGHT=str(W),RAYON_NUM_THREADS='4',QUARTZ_DISABLE_QIPC_SHM='1')
        self.err=open(BIN.parent/'arena_stderr.log','a')
        self.p=SP.Popen(['nice','-n','5',str(bin),'--server'],stdin=SP.PIPE,stdout=SP.PIPE,stderr=self.err,env=env,start_new_session=True,bufsize=0)
        self.buf=b'';self.clocks=[]
        try:
            a=self.request(dict(cmd='research_rapfi_shared_open'));assert a['status']=='OPEN' and a['ready']=='250615 mix9svq freestyle15'
            kids=pathlib.Path(f'/proc/{self.p.pid}/task/{self.p.pid}/children').read_text().split();assert len(kids)==1,kids
            self.clocks=[clock(self.p.pid),clock(int(kids[0]))]
        except BaseException:self.abort();raise
    def request(self,q):
        cpu=[time.clock_gettime_ns(c) for c in self.clocks];t=time.perf_counter_ns()
        self.p.stdin.write((json.dumps(q,separators=(',',':'))+'\n').encode());self.p.stdin.flush()
        deadline=time.monotonic()+40
        while b'\n' not in self.buf:
            dt=deadline-time.monotonic()
            if dt<=0 or not select.select([self.p.stdout],[],[],dt)[0]:raise RuntimeError('native timeout')
            x=os.read(self.p.stdout.fileno(),65536)
            if not x:raise RuntimeError('native EOF '+str(self.p.poll()))
            self.buf+=x
            assert len(self.buf)<8*1024*1024
        line,self.buf=self.buf.split(b'\n',1);a=json.loads(line);a['transport_ns']=time.perf_counter_ns()-t
        if cpu:a['cpu_ns']=[time.clock_gettime_ns(c)-v for c,v in zip(self.clocks,cpu)]
        return a
    def search(self,h,seed,mask=0,cap=128,threads=1,lock=False,table=None,trace=False):
        if table is not None:mask=None
        q=dict(cmd='research_rapfi_k35',history=h,seed=seed,mask=mask,table=table,cap=cap,threads=threads,lock=lock,trace=trace)
        a=self.request(q);check(a,q);return a
    def abort(self):
        if self.p.poll() is None:
            os.killpg(self.p.pid,signal.SIGTERM)
            try:self.p.wait(timeout=3)
            except SP.TimeoutExpired:os.killpg(self.p.pid,signal.SIGKILL);self.p.wait()
        self.err.close()
    def close(self):
        try:
            self.clocks=[]
            a=self.request(dict(cmd='research_rapfi_shared_close'));assert a['status']=='CLOSED'
            self.p.stdin.close();assert self.p.wait(timeout=10)==0
        except BaseException:self.abort();raise
        self.err.close()
    def __enter__(self):return self
    def __exit__(self,t,*args):
        if t is None:self.close()
        else:self.abort()
def reach(r):
    ca=r['candidates'];es=r['edges'];assert [e['move'] for e in es]==[m for m,v in ca[:len(es)]]
    n=[e['n'] for e in es]+[0]*(len(ca)-len(es));L=max(range(len(n)),key=lambda i:(n[i],i))
    assert ca[L][0]==r['best'] and sum(n)==r['root_n']>0
    return min((n[L]-n[i]+int(i<L) for i in range(len(n)) if i!=L),default=None)
def getbin(a):
    d=reach(a['prefix']['result']);v=a['root_eval']['value']
    return int(abs(v)>=.5)+2*int(d is None or d>=4)
def check(a,q):
    assert a.get('schema')=='quartz_rapfi_k35_v1',a
    h=q['history'];b,p,w=replay(h,'freestyle');assert w is None and a['history']==h and a['player']==p and a['seed']==q['seed']
    cap=q['cap'];assert a['cap']==cap and a['threads']==q['threads'] and a['locked']==q['lock'] and a['request_mask']==q['mask'] and a['table']==q['table']
    pr=a['proof'];legal=[i for i,v in enumerate(b) if not v];assert pr['windows']==572 and [x[0] for x in pr['actions']]==legal and a['best'] in legal
    assert len(a['cpu_ns'])==2 and min(a['cpu_ns'])>=0
    if pr['gate'] is not None:
        assert a['best']==pr['gate'] and a['cause']==pr['kind'] and a['NN_calls']==a['visits']==a['bridge_wait_ns']==0 and a['root_eval'] is None and a['hooks'] is None and a['waves']==[] and a['selected_mask'] is None
        return
    r=a['result'];mid=a['prefix']['result'];ini=a['initial'];pb=a['prefix']['budget'];ph=a['prefix']['hooks'];eh=a['hooks']
    assert ini['evaluator_calls']==1 and ini['root_n']==0 and ini['candidates']==mid['candidates']==r['candidates']
    assert pb['used_nn']+pb['unused']==15 and pb['visits']==mid['root_n']<=16 and pb['used_nn']==mid['evaluator_calls']-1
    assert ph['ledger']['iterations']==mid['root_n'] and ph['ledger']['transferred']==0
    assert a['bin']==getbin(a);mask=q['mask'] if q['table'] is None else q['table'][a['bin']];assert a['selected_mask']==mask
    n=mid['root_n'];nn=mid['evaluator_calls'];d=reach(mid)
    for z in a['waves']:
        assert (z['before_n'],z['before_nn'],z['before_d'])==(n,nn,d)
        if q['lock']:assert d is not None and d<=cap-n
        t=z['budget'];assert t['target']==cap-nn and t['cap']==min(8,cap-n) and t['used_nn']+t['unused']==t['target']
        n+=t['visits'];nn+=t['used_nn'];d=z['after_d'];assert (n,nn)==(z['after_n'],z['after_nn'])
    assert (a['visits'],a['NN_calls'])==(n,nn)==(r['root_n'],r['evaluator_calls']) and n<=cap and nn<=cap and a['best']==r['best'] and reach(r)==d
    if a['cause']=='VOTE_LOCK':assert q['lock'] and n<cap and nn<cap and (d is None or d>cap-n)
    elif a['cause']=='NN_TARGET':assert nn==cap
    else:assert a['cause']=='VISIT_CAP' and n==cap
    l=eh['ledger'];h0=eh['hooks'];assert l['iterations']==n and l['transferred']==l['receipt_items'] and l['repeated_delivery']==0 and l['eligible']==l['novel']+l['blocked_receipt']
    assert h0['base_evals']+h0['extra_evals']==nn-mid['evaluator_calls'] and h0['base_evals']+h0['terminal_leaves']==n-mid['root_n']
    if not mask&1:assert h0['select_nodes']==0
    if not mask&2:assert h0['width_nodes']==0
    if not mask&4:assert h0['extra_evals']==0
    if not mask&8:assert l['transferred']==0
    assert 0<=eh['peak']<=q['threads'] and (eh['trace'] is not None)==q['trace']
    ev=a['root_eval'];assert ev['board']==''.join(map(str,b)) and ev['player']==p and len(ev['policy'])==225 and abs(sum(ev['policy'])-1)<1e-5
    assert all(ev['policy'][i]==0 for i,v in enumerate(b) if v)
    assert all(x['vl']==0 and x['vvalue']==0 for x in r['edges']) and a['bridge_wait_ns']>=0
# Independent proof computation reused only for pre-treatment admission; no search/NN at target.
sys.path.insert(0,str(R/'quartz_kinetic34_20261126'))
from audit_k34 import proof as independent_proof
def seed(j,ply):return (j['seed']+ply*0x9e3779b97f4a7c15)&((1<<64)-1)
def small(a):
    return {k:a[k] for k in ('history','best','player','cap','threads','locked','request_mask','table','selected_mask','bin','NN_calls','visits','cause','proof_us','bridge_wait_ns','eval_ipc_ns','transport_ns','cpu_ns','seed')}
def source_one(j,s):
    h=list(j['opening']);rows=[]
    while len(h)<j['target']:
        b,p,w=replay(h,'freestyle')
        if w is not None:break
        a=s.search(h,seed(j,len(h)));rows.append(small(a));h.append(a['best'])
    b,p,w=replay(h,'freestyle')
    reason='TERMINAL' if w is not None else 'GATED' if independent_proof(tuple(b),p)[0]['gate'] is not None else 'ADMITTED'
    return dict(job=j,history=h,rows=rows,reason=reason,side=p)
def count_rows(a,c):
    c['requests']+=1;c['NN']+=a['NN_calls'];c['visits']+=a['visits']
def source_worker(arg):
    idx,rr=arg;p=D/f'SOURCE_{idx}.jsonl.gz';assert not p.exists();c=Counter()
    with Server() as s,gzip.open(p,'wt',compresslevel=3) as f:
        for j in rr:
            z=source_one(j,s);emit(f,z);c[z['reason']]+=1
            for a in z['rows']:count_rows(a,c)
    return dict(file=p.name,sha256=sha(p),**c)
def source():
    j=[j for j in read('PLAN.json')['jobs'] if j['group']!='ARENA']
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:fs=list(ex.map(source_worker,[(i,j[i::WORKERS]) for i in range(WORKERS)]))
    co=[];c=Counter()
    for fi in fs:
        for z in raw(D/fi['file']):
            if z['reason']=='ADMITTED':co.append(dict(job=z['job'],history=z['history'],side=z['side']))
        c.update({k:v for k,v in fi.items() if k not in ('file','sha256')})
    co.sort(key=lambda r:r['job']['id'])
    save('COHORT.json',dict(roots=co,groups=dict(Counter(r['job']['group'] for r in co)),files=fs,counts=dict(c),plan=sha(D/'PLAN.json')))
    print('SOURCE',dict(c),read('COHORT.json')['groups'],flush=True)
def label(s,r,m):
    j=r['job'];h=r['history']+[m];rr=[]
    while True:
        b,p,w=replay(h,'freestyle')
        if w is not None:break
        a=s.search(h,seed(j,len(h)));rr.append(small(a));h.append(a['best'])
    return dict(action=m,history=h,winner=w,utility=(1+r['side']*w)/2,rows=rr)
def atlas_worker(arg):
    mode,i,roots=arg;p=D/f'{mode}_{i}.jsonl.gz';assert not p.exists();c=Counter()
    with Server() as s,gzip.open(p,'wt',compresslevel=3) as f:
        for r in roots:
            j=r['job'];h=r['history'];order=[(m,w,t) for m in range(16) for w in (1,4) for t in (0,1)]
            random.Random(SEED+j['id']).shuffle(order);z={}
            for m,w,t in order:
                a=s.search(h,seed(j,len(h)),m,80,w,False);assert a['proof']['gate'] is None
                z[f'{m}:{w}:{t}']=a;count_rows(a,c)
            common=z['0:1:0']
            assert all(a['prefix']['result']==common['prefix']['result'] and a['root_eval']==common['root_eval'] and a['proof']==common['proof'] for a in z.values())
            # Hold full independent actual search records; one exact observed continuation per unique chosen action.
            actions=sorted({a['best'] for a in z.values()})
            ll={str(m):label(s,r,m) for m in actions}
            for a in ll.values():
                for k in a['rows']:c['label_requests']+=1;c['label_NN']+=k['NN_calls'];c['label_visits']+=k['visits']
            c['labels']+=len(ll);c['roots']+=1
            emit(f,dict(root=r,order=order,results=z,labels=ll));f.flush()
            print('ATLAS',mode,i,j['id'],'labels',len(ll),'NN',c['NN'],flush=True)
    return dict(file=p.name,sha256=sha(p),**c)
def collect_atlas(mode):
    co=read('COHORT.json');rr=[r for r in co['roots'] if r['job']['group'] in (('TRAIN','CAL') if mode=='TRAIN_CAL' else ('TEST',))]
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:fs=list(ex.map(atlas_worker,[(mode,i,rr[i::WORKERS]) for i in range(WORKERS)]))
    c=Counter()
    for x in fs:c.update({k:v for k,v in x.items() if k not in ('file','sha256')})
    save(mode+'_LOCK.json',dict(cohort=sha(D/'COHORT.json'),freeze=sha(D/'FROZEN.json') if mode=='TEST' else None,files=fs,counts=dict(c),code=sha(D/'experiment.py')))
    print('COLLECT',mode,dict(c),flush=True)
def atlas_rows(mode):
    for x in read(mode+'_LOCK.json')['files']:
        p=D/x['file'];assert sha(p)==x['sha256'];yield from raw(p)
def util(z,a):return z['labels'][str(a['best'])]['utility']
def net(z,a):return util(z,a)-PRICE*a['NN_calls']
def means(z,w):
    return [S.mean(net(z,z['results'][f'{m}:{w}:{t}']) for t in (0,1)) for m in range(16)]
def fit():
    train=[];bs=defaultdict(list)
    for z in atlas_rows('TRAIN_CAL'):
        if z['root']['job']['group']=='TRAIN':
            y=means(z,1);d=[x-y[0] for x in y];train.append(d);bs[z['results']['0:1:0']['bin']].append(d)
    pooled=[S.mean(v[m] for v in train) for m in range(16)]
    static=max(range(16),key=lambda m:(pooled[m],-m))
    table=[max(range(16),key=lambda m:((sum(v[m] for v in bs[b])+8*pooled[m])/(len(bs[b])+8),-m)) for b in range(4)]
    save('MODEL.json',dict(atlas=sha(D/'TRAIN_CAL_LOCK.json'),train=len(train),bin_n=[len(bs[b]) for b in range(4)],pooled=pooled,static=static,table=table,price=PRICE,shrink=8))
    print('FIT',read('MODEL.json'),flush=True)
def live(mode):
    m=read('MODEL.json');out=[];count=Counter();frozen=read('FROZEN.json') if mode=='TEST' else None
    with Server() as s:
        for z in atlas_rows('TRAIN_CAL' if mode=='CAL' else 'TEST'):
            r=z['root'];j=r['job'];h=r['history']
            if j['group']!=mode:continue
            policies={'base':[0]*4,'static':[m['static']]*4,'table':m['table']}
            if frozen is not None:policies['selected']=frozen['table']
            order=list(policies);random.Random(SEED+81+j['id']).shuffle(order);res={}
            for k in order:
                a=s.search(h,seed(j,len(h)),cap=80,table=policies[k]);ref=z['results'][f"{a['selected_mask']}:1:0"]
                assert a['result']==ref['result'] and a['NN_calls']==ref['NN_calls'] and a['prefix']['result']==ref['prefix']['result']
                res[k]=dict(response=a,u=util(z,a),net=net(z,a));count_rows(a,count)
            out.append(dict(root=r,order=order,policies=res))
    p=D/f'{mode}_LIVE.jsonl.gz';assert not p.exists()
    with gzip.open(p,'wt',compresslevel=3) as f:
        for z in out:emit(f,z)
    stats={}
    for k in out[0]['policies']:
        vv=[z['policies'][k]['net']-z['policies']['base']['net'] for z in out]
        stats[k]=dict(net=S.mean(z['policies'][k]['net'] for z in out),u=S.mean(z['policies'][k]['u'] for z in out),diff=S.mean(vv),lower=S.mean(vv)-2*S.stdev(vv)/math.sqrt(len(vv)) if len(vv)>1 else -1.)
    save(mode+'_LIVE_LOCK.json',dict(sha256=sha(p),model=sha(D/'MODEL.json'),n=len(out),counts=dict(count),stats=stats))
    if mode=='CAL':
        selected=max(['base']+[k for k in ('static','table') if stats[k]['lower']>0],key=lambda k:stats[k]['net'])
        tab={'base':[0]*4,'static':[m['static']]*4,'table':m['table']}[selected]
        save('FROZEN.json',dict(model=sha(D/'MODEL.json'),cal=sha(D/'CAL_LIVE_LOCK.json'),selected=selected,table=tab))
    print('LIVE',mode,stats,'frozen',read('FROZEN.json'),flush=True)
def score_atlas():
    z=list(atlas_rows('TEST'));n=len(z);out={}
    for w in (1,4):
        by={}
        for m in range(16):
            aa=[(x,x['results'][f'{m}:{w}:{t}']) for x in z for t in (0,1)]
            def avg(fn):return S.mean(fn(x,a) for x,a in aa)
            by[str(m)]=dict(u=avg(util),net=avg(net),NN=avg(lambda x,a:a['NN_calls']),visits=avg(lambda x,a:a['visits']),cpu_ms=avg(lambda x,a:sum(a['cpu_ns'])/1e6),wall_ms=avg(lambda x,a:a['transport_ns']/1e6),wait_ms=avg(lambda x,a:a['bridge_wait_ns']/1e6),ipc_ms=avg(lambda x,a:a['eval_ipc_ns']/1e6),
                S=sum(a['hooks']['hooks']['select_changed'] for x,a in aa),X=sum(a['hooks']['width_realized'] for x,a in aa),V=sum(a['hooks']['hooks']['eval_changed'] for x,a in aa),B=sum(a['hooks']['ledger']['transferred'] for x,a in aa),blocked=sum(a['hooks']['ledger']['blocked_receipt'] for x,a in aa),shared=sum(a['hooks']['ledger']['shared'] for x,a in aa),
                extra=sum(a['hooks']['hooks']['extra_evals'] for x,a in aa),peak4=sum(a['hooks']['peak']==4 for x,a in aa),
                disagree=sum(x['results'][f'{m}:{w}:0']['best']!=x['results'][f'{m}:{w}:1']['best'] for x in z),
                utility_sd=math.sqrt(S.mean((util(x,x['results'][f'{m}:{w}:0'])-util(x,x['results'][f'{m}:{w}:1']))**2/2 for x in z)))
        effects={}
        for k in ('u','NN','net','wall_ms','cpu_ms','wait_ms'):
            vv=[by[str(m)][k] for m in range(16)]
            b=[sum(v*(-1)**((i&~m).bit_count()) for m,v in enumerate(vv))/16 for i in range(16)]
            assert all(abs(vv[m]-sum(b[i]*(-1)**((i&~m).bit_count()) for i in range(16)))<1e-8 for m in range(16))
            effects[k]=dict(beta=b,main=[2*b[i] for i in (1,2,4,8)],pair=[4*b[i] for i in (3,5,9,6,10,12)],full_base=vv[15]-vv[0])
        out[str(w)]=dict(arms=by,effects=effects)
    save('ATLAS_SUMMARY.json',dict(n=n,workers=out,scope='finite B128 continuation utility, actual paired atlas; not minimax'))
    print('ATLAS_SCORE',n,[(w,out[str(w)]['effects']['net']['main']) for w in (1,4)],flush=True)
def arena_policy(arm):
    return {'B128':dict(mask=0,cap=128,lock=False),'B80':dict(mask=0,cap=80,lock=False),'L80':dict(mask=0,cap=80,lock=True),'F80L':dict(mask=15,cap=80,lock=True),'SELECT80L':dict(table=read('FROZEN.json')['table'],cap=80,lock=True)}[arm]
def game(s,j,arm,color):
    h=list(j['opening']);rr=[]
    while True:
        b,p,w=replay(h,'freestyle')
        if w is not None:break
        who=arm if p==color else 'B128';a=s.search(h,seed(j,len(h)),**arena_policy(who));z=small(a);z['arm']=who
        rr.append(z);h.append(a['best'])
    return dict(job=j,arm=arm,color=color,history=h,winner=w,score=(1+color*w)/2,rows=rr)
def arena_worker(arg):
    arm,i,rr=arg;p=D/f'ARENA_{arm}_{i}.jsonl.gz';assert not p.exists();c=Counter()
    with Server() as s,gzip.open(p,'wt',compresslevel=3) as f:
        for j in rr:
            for color in j['color_order']:
                z=game(s,j,arm,color);emit(f,z);f.flush();c['games']+=1
                for a in z['rows']:count_rows(a,c)
            print('ARENA',arm,i,j['id'],dict(c),flush=True)
    return dict(file=p.name,sha256=sha(p),arm=arm,**c)
def arena():
    pl=read('PLAN.json');rr=[j for j in pl['jobs'] if j['group']=='ARENA'];tab=read('FROZEN.json')['table']
    arms=['B80','L80','F80L'];alias=None
    if tab==[0]*4:alias='L80'
    elif tab==[15]*4:alias='F80L'
    else:arms.append('SELECT80L')
    tasks=[(a,i,rr[i::3]) for a in arms for i in range(3)];random.Random(SEED+99).shuffle(tasks)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:fs=list(ex.map(arena_worker,tasks))
    c=Counter()
    for x in fs:c.update({k:v for k,v in x.items() if k not in ('file','sha256','arm')})
    save('ARENA_LOCK.json',dict(plan=sha(D/'PLAN.json'),freeze=sha(D/'FROZEN.json'),files=fs,arms=arms,alias=alias,counts=dict(c)))
    print('ARENA_COMPLETE',dict(c),'alias',alias,flush=True)
def score_arena():
    lk=read('ARENA_LOCK.json');g=defaultdict(list)
    for x in lk['files']:
        p=D/x['file'];assert sha(p)==x['sha256'];g[x['arm']].extend(raw(p))
    out={}
    for arm,gs in g.items():
        groups=defaultdict(list)
        for z in gs:groups[z['job']['id']].append(z)
        assert len(groups)==ARENA and all(len(v)==2 and {z['color'] for z in v}=={1,-1} for v in groups.values())
        def count(v):
            c=Counter()
            for z in v:
                for a in z['rows']:
                    k='candidate' if a['player']==z['color'] else 'base';c[k+'_moves']+=1
                    for key,val in [('NN',a['NN_calls']),('visits',a['visits']),('wall_ns',a['transport_ns']),('cpu_ns',sum(a['cpu_ns'])),('wait_ns',a['bridge_wait_ns'])]:c[k+'_'+key]+=val
                    c[k+'_'+a['cause']]+=1
            return c
        pc={i:count(v) for i,v in groups.items()};ps={i:S.mean(z['score'] for z in v) for i,v in groups.items()};ct=count(gs)
        def red(c):return 1-(c['candidate_NN']/c['candidate_moves'])/(c['base_NN']/c['base_moves'])
        rng=random.Random(SEED+42);ids=sorted(groups);bs=[];bn=[]
        for _ in range(4000):
            ss=rng.choices(ids,k=len(ids));bs.append(S.mean(ps[i] for i in ss));cc=Counter()
            for i in ss:cc.update(pc[i])
            bn.append(red(cc))
        bs.sort();bn.sort();lower=bs[199]
        out[arm]=dict(games=len(gs),pairs=len(groups),wins=sum(z['score']==1 for z in gs),draws=sum(z['score']==.5 for z in gs),losses=sum(z['score']==0 for z in gs),score=S.mean(ps.values()),score_ci=[bs[99],bs[3899]],NI_lower=lower,
            NN_reduction=red(ct),NN_ci=[bn[99],bn[3899]],joint=lower>=.45 and red(ct)>=.3,counts=dict(ct))
    save('ARENA_SUMMARY.json',dict(arms=out,alias=lk['alias'],scope='paired opening families, identical pretrained evaluator, no external alpha-beta games'))
    print('ARENA_SCORE',json.dumps(out,sort_keys=True),flush=True)
def diagnostic():
    co=[r for r in read('COHORT.json')['roots'] if r['job']['group']=='TRAIN'][:3];p=D/'DIAGNOSTIC.jsonl.gz';assert not p.exists();c=Counter()
    with Server() as s,gzip.open(p,'wt',compresslevel=3) as f:
        for r in co:
            j=r['job'];h=r['history'];z={}
            for m in (0,1,2,4,8,15):
                for w in (1,4):
                    a=s.search(h,seed(j,len(h)),m,80,w,False,trace=True);z[f'{m}:{w}']=a;count_rows(a,c)
            emit(f,dict(root=r,results=z))
    save('DIAGNOSTIC_LOCK.json',dict(sha256=sha(p),counts=dict(c),roots=len(co)))
    print('DIAG',dict(c),flush=True)
def smoke():
    h=fixtures()['trans_a'];out=[]
    with Server() as s:
        for m in range(16):
            a=s.search(h,17,m,80,1,False);out.append(a)
        assert all(a['prefix']['result']==out[0]['prefix']['result'] for a in out)
        a=s.search(h,17,cap=80,table=[15]*4);assert a['result']==out[15]['result']
        b=s.search(h,17,15,80,4,True,trace=True);assert b['selected_mask']==15
        c=s.search([110,0,111,2,113,4],17,15,80,1,True);assert c['cause']=='WIN3'
        for q in [dict(cmd='research_rapfi_k35',history=[],mask=16,cap=80,threads=1,lock=False,seed=17),dict(cmd='research_rapfi_k35',history=[],mask=0,table=[0]*4,cap=80,threads=1,lock=False,seed=17)]:assert 'error' in s.request(q)
    save('SMOKE.json',dict(rows=len(out)+3,mask0=small(out[0]),mask15=small(out[15]),parallel=small(b),cert=small(c),rejected=2))
    print('SMOKE',len(out)+3,flush=True)
if __name__=='__main__':
    if sys.argv[1] in ('collect_atlas','live'):globals()[sys.argv[1]](sys.argv[2])
    else:globals()[sys.argv[1]]()
