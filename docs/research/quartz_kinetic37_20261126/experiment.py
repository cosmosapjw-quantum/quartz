"""K37 same frozen Rapfi evaluator, all games MCTS vs MCTS."""
import pathlib,sys,json,gzip,hashlib,random,math,statistics as S,time,os,select,subprocess as SP,signal,ctypes
from fractions import Fraction as F
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2];R=D.parent
BIN=ROOT.parent/'quartz_kinetic37_target_20261126/mcts_demo'
OLD=ROOT.parent/'quartz_kinetic35_target_20261126/mcts_demo'
BRIDGE=ROOT.parent/'rapfi_250615_quartz_bridge_build_20261125/pbrain-rapfi'
LOCK=json.loads((D/'RAPFI_LOCK.json').read_text());AS={x['name']:x for x in LOCK['files']};W=pathlib.Path(AS['Rapfi_NNUE_freestyle']['path'])
sys.path.insert(0,str(R/'quartz_tactical08_20261125'))
from rules import replay,orbit_board,win_at,fixtures
SEED=2026112637;N=1088;ARENA=192;PRICE=.002;WORKERS=8;H=(80,96,112,128,256)
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
        cells=[15*r+c for r in range(4,11) for c in range(4,11)] if (i//4)%2==0 else [15*r+c for r in range(2,13) for c in range(2,13)]
        while True:
            h=rng.sample(cells,6);k=family(h)
            if k not in fam:break
        fam.add(k);jobs.append(dict(id=i,opening=h,family=k,group='TRAIN' if i<480 else 'CAL' if i<720 else 'TEST' if i<960 else 'NATURAL' if i<1088 else 'ARENA',
            target=(9,12,15,18)[i%4],seed=rng.randrange(2**63),color_order=rng.sample([1,-1],2)))
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
    def search(self,h,seed,cap=128,lock=False,horizons=None,trace=False):
        if horizons is not None:cap=None
        q=dict(cmd='research_rapfi_k36',history=h,seed=seed,cap=cap,horizons=horizons,lock=lock,trace=trace)
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
    r=a['probe'];d=reach(r)
    return int(r['candidates'][0][0]!=r['best'])+2*int(a['prefix']['result']['best']!=r['best'])+4*int(d is None or d>=8)
def check(a,q):
    assert a.get('schema')=='quartz_rapfi_k36_v1',a
    h=q['history'];b,p,w=replay(h,'freestyle');assert w is None and a['history']==h and a['player']==p and a['seed']==q['seed']
    assert a['threads']==1 and a['locked']==q['lock'] and a['requested_cap']==q['cap'] and a['horizons']==q['horizons'] and a['request_mask']==0
    pr=a['proof'];legal=[i for i,v in enumerate(b) if not v];assert pr['windows']==572 and [x[0] for x in pr['actions']]==legal and a['best'] in legal
    assert len(a['cpu_ns'])==2 and min(a['cpu_ns'])>=0
    if pr['gate'] is not None:
        assert a['best']==pr['gate'] and a['cause']==pr['kind'] and a['NN_calls']==a['visits']==a['bridge_wait_ns']==0 and a['root_eval'] is None and a['hooks'] is None and a['waves']==[] and a['probe'] is None and a['bin'] is None and a['selected_mask'] is None
        assert a['cap']==(q['cap'] if q['cap'] is not None else 128);return
    r=a['result'];mid=a['prefix']['result'];ini=a['initial'];pb=a['prefix']['budget'];ph=a['prefix']['hooks'];eh=a['hooks'];probe=a['probe']
    assert ini['evaluator_calls']==1 and ini['root_n']==0 and ini['candidates']==mid['candidates']==probe['candidates']==r['candidates']
    assert pb['used_nn']+pb['unused']==15 and pb['visits']==mid['root_n']<=16 and pb['used_nn']==mid['evaluator_calls']-1
    assert ph['ledger']['iterations']==mid['root_n'] and ph['ledger']['transferred']==0
    assert a['bin']==getbin(a) and a['selected_mask']==0
    cap=a['cap'];assert cap==(q['cap'] if q['cap'] is not None else q['horizons'][a['bin']])
    n=mid['root_n'];nn=mid['evaluator_calls'];d=reach(mid)
    k=a['probe_wave'];assert 0<k<=len(a['waves'])
    for i,z in enumerate(a['waves']):
        assert (z['before_n'],z['before_nn'],z['before_d'])==(n,nn,d)
        pre=i<k;hcap=(q['cap'] or 256) if pre else cap
        if pre:assert n<32 and nn<32
        if q['lock']:assert d is not None and d<=hcap-n
        t=z['budget'];assert t['target']==hcap-nn and t['cap']==min(8,hcap-n) and t['used_nn']+t['unused']==t['target']
        n+=t['visits'];nn+=t['used_nn'];d=z['after_d'];assert (n,nn)==(z['after_n'],z['after_nn'])
        if i+1==k:assert (n,nn)==(probe['root_n'],probe['evaluator_calls']) and (n>=32 or nn>=32) and n<=39 and reach(probe)==d and probe['best']==z['after_best']
    assert (a['visits'],a['NN_calls'])==(n,nn)==(r['root_n'],r['evaluator_calls']) and n<=cap and nn<=cap and a['best']==r['best'] and reach(r)==d
    if a['cause']=='VOTE_LOCK':assert q['lock'] and n<cap and nn<cap and (d is None or d>cap-n)
    elif a['cause']=='NN_TARGET':assert nn==cap
    else:assert a['cause']=='VISIT_CAP' and n==cap
    l=eh['ledger'];h0=eh['hooks'];assert l['iterations']==n and l['transferred']==l['receipt_items']==l['repeated_delivery']==0 and l['eligible']==l['novel']+l['blocked_receipt']
    assert h0['base_evals']+h0['extra_evals']==nn-mid['evaluator_calls'] and h0['base_evals']+h0['terminal_leaves']==n-mid['root_n']
    assert h0['select_nodes']==h0['width_nodes']==h0['extra_evals']==0
    assert 0<=eh['peak']<=1 and (eh['trace'] is not None)==q['trace']
    ev=a['root_eval'];assert ev['board']==''.join(map(str,b)) and ev['player']==p and len(ev['policy'])==225 and abs(sum(ev['policy'])-1)<1e-5
    assert all(ev['policy'][i]==0 for i,v in enumerate(b) if v)
    assert all(x['vl']==0 and x['vvalue']==0 for x in r['edges']) and a['bridge_wait_ns']>=0
# Independent proof computation reused only for pre-treatment admission; no search/NN at target.
sys.path.insert(0,str(R/'quartz_kinetic34_20261126'))
from audit_k34 import proof as independent_proof
def seed(j,ply):return (j['seed']+ply*0x9e3779b97f4a7c15)&((1<<64)-1)
def votes(r):
    if r is None:return None
    es={e['move']:e['n'] for e in r['edges']}
    return [[m,es.get(m,0)] for m,v in r['candidates']]
def small(a):
    z={k:a[k] for k in ('history','best','player','cap','threads','locked','requested_cap','horizons','selected_mask','bin','NN_calls','visits','cause','proof_us','bridge_wait_ns','eval_ipc_ns','transport_ns','cpu_ns','seed','probe_wave')}
    z['votes']=votes(a['result']);z['probe_votes']=votes(a['probe'])
    z['prefix_votes']=votes(a['prefix']['result']) if a['prefix'] else None
    z['prefix_nn']=a['prefix']['result']['evaluator_calls'] if a['prefix'] else None
    z['probe_nn']=a['probe']['evaluator_calls'] if a['probe'] else None
    z['waves']=[[w['before_n'],w['before_nn'],w['before_d'],w['after_n'],w['after_nn'],w['after_d'],w['budget']['target'],w['budget']['cap']] for w in a['waves']]
    return z
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
from estimator import sample_stable,estimate_absolute,delta,fit_table
GROUPS=('TRAIN','CAL','TEST','NATURAL')
REFS=((128,128),(256,256),(128,256),(256,128))
def screen_worker(arg):
    i,roots=arg;p=D/f'SCREEN_{i}.jsonl.gz';assert not p.exists();c=Counter()
    with Server() as s,gzip.open(p,'wt',compresslevel=3) as f:
        for r in roots:
            j=r['job'];h=r['history'];order=[(cap,lock) for cap in H for lock in (False,True)]
            random.Random(SEED+j['id']).shuffle(order);z={}
            for cap,lock in order:
                a=s.search(h,seed(j,len(h)),cap,lock);assert a['proof']['gate'] is None
                z[('L' if lock else 'B')+str(cap)]=a;count_rows(a,c)
            common=z['B128']
            assert all(a['prefix']['result']==common['prefix']['result'] and a['root_eval']==common['root_eval'] and a['probe']==common['probe'] and a['proof']==common['proof'] for a in z.values())
            for cap in H:
                a,b=z['B'+str(cap)],z['L'+str(cap)]
                assert a['best']==b['best'] and b['NN_calls']<=a['NN_calls'] and b['visits']<=a['visits']
            actions=sorted({a['best'] for a in z.values()});c['roots']+=1;c['disagree']+=len(actions)>1
            emit(f,dict(root=r,order=order,results=z,actions=actions));f.flush()
            if c['roots']%16==0:print('SCREEN',i,c['roots'],dict(c),flush=True)
    return dict(file=p.name,sha256=sha(p),**c)
def screen():
    rr=read('COHORT.json')['roots']
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:fs=list(ex.map(screen_worker,[(i,rr[i::WORKERS]) for i in range(WORKERS)]))
    c=Counter();frame=[]
    for f in fs:
        c.update({k:v for k,v in f.items() if k not in ('file','sha256')})
        for z in raw(D/f['file']):
            frame.append(dict(root=z['root'],actions=z['actions'],bin=z['results']['L128']['bin'],arms={k:{x:a[x] for x in ('best','NN_calls','visits','cause')} for k,a in z['results'].items()}))
    frame.sort(key=lambda r:r['root']['job']['id'])
    save('FRAME.json',frame);save('SCREEN_LOCK.json',dict(cohort=sha(D/'COHORT.json'),frame=sha(D/'FRAME.json'),files=fs,counts=dict(c),code=sha(D/'experiment.py')))
    print('SCREEN COMPLETE',dict(c),flush=True)
def sample():
    fr=read('FRAME.json');groups={}
    for i,g in enumerate(GROUPS):
        rr=[r for r in fr if r['root']['job']['group']==g]
        stable=[r['root']['job']['id'] for r in rr if len(r['actions'])==1]
        hard=[r['root']['job']['id'] for r in rr if len(r['actions'])>1]
        ss=stable if g=='NATURAL' else sample_stable(stable,2026112737+i)
        groups[g]=dict(N=len(rr),stable_N=len(stable),stable_n=len(ss),hard_N=len(hard),stable_ids=ss,hard_ids=hard,
            stable_pi=[len(ss),len(stable)] if stable else None)
    save('SAMPLE.json',dict(frame=sha(D/'FRAME.json'),groups=groups,seed=2026112737))
    print('SAMPLE',[(g,{k:v for k,v in x.items() if not k.endswith('_ids')}) for g,x in groups.items()],flush=True)
def label(s,r,m,ref):
    j=r['job'];h=r['history']+[m];rr=[]
    while True:
        b,p,w=replay(h,'freestyle')
        if w is not None:break
        cap=ref[0] if p==r['side'] else ref[1]
        a=s.search(h,seed(j,len(h)),cap=cap);rr.append(small(a));h.append(a['best'])
    return dict(action=m,ref=list(ref),history=h,winner=w,utility=(1+r['side']*w)/2,rows=rr)
def label_worker(arg):
    mode,i,roots=arg;p=D/f'{mode}_{i}.jsonl.gz';assert not p.exists();c=Counter()
    with Server() as s,gzip.open(p,'wt',compresslevel=3) as f:
        for z in roots:
            r=z['root'];ll={str(m):[label(s,r,m,ref) for ref in REFS] for m in z['actions']}
            for vs in ll.values():
                for a in vs:
                    for v in a['rows']:count_rows(v,c)
                    c['labels']+=1
                c['actions']+=1
            c['roots']+=1;emit(f,dict(root=r,labels=ll));f.flush()
            if c['roots']%8==0:print('LABEL',mode,i,c['roots'],dict(c),flush=True)
    return dict(file=p.name,sha256=sha(p),**c)
def labels(mode):
    groups=('TRAIN','CAL') if mode=='TRAIN_CAL' else ('TEST','NATURAL')
    sp=read('SAMPLE.json')['groups'];ids={i for g in groups for i in sp[g]['stable_ids']+sp[g]['hard_ids']}
    rr=[r for r in read('FRAME.json') if r['root']['job']['id'] in ids]
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:fs=list(ex.map(label_worker,[(mode,i,rr[i::WORKERS]) for i in range(WORKERS)]))
    c=Counter()
    for f in fs:c.update({k:v for k,v in f.items() if k not in ('file','sha256')})
    save(mode+'_LOCK.json',dict(sample=sha(D/'SAMPLE.json'),frame=sha(D/'FRAME.json'),freeze=sha(D/'FROZEN.json') if mode!='TRAIN_CAL' else None,files=fs,counts=dict(c)))
    print('LABEL COMPLETE',mode,dict(c),flush=True)
def load_labels(mode):
    out={}
    for f in read(mode+'_LOCK.json')['files']:
        p=D/f['file'];assert sha(p)==f['sha256']
        for z in raw(p):
            i=z['root']['job']['id'];assert i not in out
            out[i]={k:[F(x['utility']) for x in v] for k,v in z['labels'].items()}
    return out
def U(l,ref=None):
    if l is None:return None
    return {k:sum(v,F(0))/len(v) if ref is None else v[ref] for k,v in l.items()}
def gain(z,l,h,ref=None):
    return delta(z['arms']['L'+str(h)]['best'],z['arms']['L128']['best'],U(l,ref))
def fit():
    lab=load_labels('TRAIN_CAL');rr=[z for z in read('FRAME.json') if z['root']['job']['group']=='TRAIN']
    a=[dict(bin=z['bin'],cost={h:z['arms']['L'+str(h)]['NN_calls'] for h in (80,128)},gain={h:gain(z,lab.get(z['root']['job']['id']),h) for h in (80,128)}) for z in rr]
    m=fit_table(a);m['gain']=float(m['gain']);m.update(frame=sha(D/'FRAME.json'),labels=sha(D/'TRAIN_CAL_LOCK.json'))
    save('MODEL.json',m);print('FIT',m,flush=True)
def live(mode):
    m=read('MODEL.json');frozen=read('FROZEN.json') if mode=='HELDOUT' else None
    lab=load_labels('TRAIN_CAL' if mode=='CAL' else 'HELDOUT')
    rr=[z for z in read('FRAME.json') if z['root']['job']['group'] in (('CAL',) if mode=='CAL' else ('TEST','NATURAL'))]
    tabs={'L128':[128]*8,'L80':[80]*8,'table':m['table'] if mode=='CAL' else frozen['table']}
    p=D/f'{mode}_LIVE.jsonl.gz';assert not p.exists();c=Counter();values=defaultdict(list)
    with Server() as s,gzip.open(p,'wt',compresslevel=3) as f:
        for z in rr:
            r=z['root'];j=r['job'];h=r['history'];i=j['id'];out={}
            for k,tab in tabs.items():
                a=s.search(h,seed(j,len(h)),lock=True,horizons=tab);ref=z['arms']['L'+str(a['cap'])]
                assert all(a[x]==ref[x] for x in ('best','NN_calls','visits','cause')) and a['bin']==z['bin']
                d=gain(z,lab.get(i),a['cap']);out[k]=dict(response=small(a),gain=float(d),net=float(d-F(a['NN_calls'],500)))
                values[k].append((d,F(a['NN_calls'],500),a['NN_calls']));count_rows(a,c)
            emit(f,dict(root=r,policies=out))
    def low(v):return float(S.mean(v))-2*S.stdev(v)/math.sqrt(len(v)) if len(v)>1 else -1.
    stats={}
    for k,v in values.items():
        u=[x[0] for x in v];dn=[(x[0]-x[1])-(b[0]-b[1]) for x,b in zip(v,values['L80'])]
        stats[k]=dict(gain=float(S.mean(u)),u_lower=low(u),NN=S.mean(x[2] for x in v),net_gain=float(S.mean(dn)),net_lower=low(dn))
    save(mode+'_LIVE_LOCK.json',dict(sha256=sha(p),n=len(rr),counts=dict(c),stats=stats))
    if mode=='CAL':
        admitted=m['table'] not in ([80]*8,[128]*8) and stats['table']['net_lower']>0 and stats['table']['u_lower']>=-.05
        save('FROZEN.json',dict(model=sha(D/'MODEL.json'),cal=sha(D/'CAL_LIVE_LOCK.json'),admitted=admitted,table=m['table'] if admitted else [80]*8))
    print('LIVE',mode,stats,read('FROZEN.json'),flush=True)
def summarize():
    fr=read('FRAME.json');sp=read('SAMPLE.json')['groups'];allout={}
    for group in GROUPS:
        rr=[z for z in fr if z['root']['job']['group']==group];N=len(rr);g=sp[group]
        lab=load_labels('TRAIN_CAL' if group in ('TRAIN','CAL') else 'HELDOUT');out={}
        for h in H:
            d=[gain(z,lab.get(z['root']['job']['id']),h) for z in rr]
            refs=[sum(gain(z,lab.get(z['root']['job']['id']),h,k) for z in rr)/N for k in range(4)]
            strata=[[],[]];hard=[]
            for z in rr:
                i=z['root']['job']['id'];a=z['arms']['L'+str(h)]['best'];l=lab.get(i)
                if l is not None:strata[int(len(z['actions'])>1)].append(U(l)[str(a)])
                if len(z['actions'])>1:hard.append(gain(z,l,h))
            e=estimate_absolute(N,[(g['stable_N'],strata[0]),(g['hard_N'],strata[1])])
            out[str(h)]=dict(gain=float(sum(d)/N),ref_gains=[float(x) for x in refs],NN=S.mean(z['arms']['L'+str(h)]['NN_calls'] for z in rr),
                B_NN=S.mean(z['arms']['B'+str(h)]['NN_calls'] for z in rr),absolute=float(e['mean']),design_se=math.sqrt(float(e['variance'])),naive=float(S.mean(strata[0]+strata[1])),
                ESS=float(e['ess']),hard_gain=float(S.mean(hard)) if hard else None,nonzero=sum(x!=0 for x in d))
        ls=[lab[i] for i in g['stable_ids']+g['hard_ids']]
        allout[group]=dict(N=N,sampled=len(ls),hard=g['hard_N'],arms=out,actions=sum(len(v) for v in ls),reference_disagreement=sum(len(set(v))>1 for x in ls for v in x.values()),
             nonconstant=sum(len(set(U(l).values()))>1 for l in ls),positive_hard=sum(any(gain(z,lab.get(z['root']['job']['id']),h)>0 for h in H) for z in rr if len(z['actions'])>1))
    save('ATLAS_SUMMARY.json',dict(groups=allout,scope='finite-frame exact decision differences, stratified absolute utility; four policy-conditioned continuation labels'))
    print('ATLAS',[(g,x['N'],x['sampled'],x['hard'],x['nonconstant'],x['arms']['80']) for g,x in allout.items()],flush=True)
def arena_policy(arm):
    return {'B128':dict(cap=128,lock=False),'L128':dict(cap=128,lock=True),'B80':dict(cap=80,lock=False),'L80':dict(cap=80,lock=True),'ADAPT':dict(horizons=read('FROZEN.json')['table'],lock=True)}[arm]
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
    arms=['B80','L80','L128'];alias='L80'
    if read('FROZEN.json')['admitted']:arms.append('ADAPT');alias=None
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
    co=[r for r in read('COHORT.json')['roots'] if r['job']['group']=='TRAIN'][:12];p=D/'DIAGNOSTIC.jsonl.gz';assert not p.exists();c=Counter()
    with Server() as s,gzip.open(p,'wt',compresslevel=3) as f:
        for i,r in enumerate(co):
            j=r['job'];h=r['history'];z={};order=[(cap,lock) for cap in H for lock in (False,True)]
            random.Random(SEED+991+j['id']).shuffle(order)
            for cap,lock in order:
                a=s.search(h,seed(j,len(h)),cap,lock,trace=i<3);z[('L' if lock else 'B')+str(cap)]=a;count_rows(a,c)
            for cap in H:assert z['B'+str(cap)]['best']==z['L'+str(cap)]['best']
            emit(f,dict(root=r,results=z,order=order))
    save('DIAGNOSTIC_LOCK.json',dict(sha256=sha(p),counts=dict(c),roots=len(co)))
    print('DIAG',dict(c),flush=True)
def smoke():
    h=fixtures()['trans_a'];out={}
    with Server() as s:
        for cap in H:
            for lock in (False,True):
                out[('L' if lock else 'B')+str(cap)]=s.search(h,17,cap,lock)
        assert all(a['probe']==out['B128']['probe'] for a in out.values())
        for cap in H:assert out['B'+str(cap)]['best']==out['L'+str(cap)]['best']
        a=s.search(h,17,lock=True,horizons=[96]*8);assert a['result']==out['L96']['result']
        b=s.search([110,0,111,2,113,4],17,lock=True,horizons=[96]*8);assert b['cause']=='WIN3'
        for q in [dict(cmd='research_rapfi_k36',history=[],cap=81,horizons=None,lock=False,seed=17),dict(cmd='research_rapfi_k36',history=[],cap=80,horizons=[80]*8,lock=True,seed=17),dict(cmd='research_rapfi_k36',history=[],cap=None,horizons=[80]*8,lock=False,seed=17)]:assert 'error' in s.request(q)
    save('SMOKE.json',dict(rows={k:small(a) for k,a in out.items()},adaptive=small(a),cert=small(b),rejected=3))
    print('SMOKE',len(out)+2,flush=True)
if __name__=='__main__':
    if sys.argv[1] in ('labels','live'):globals()[sys.argv[1]](sys.argv[2])
    else:globals()[sys.argv[1]]()
