"""K34 real Rapfi frozen board-only evaluator and separate official engine arena."""
import pathlib,sys,json,gzip,hashlib,random,math,statistics as S,time,os,select,subprocess as SP,signal,importlib.util
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2];R=D.parent;BIN=ROOT.parent/'quartz_kinetic34_target_20261126/mcts_demo'
OLD=ROOT.parent/'quartz_kinetic32_target_20261126/mcts_demo';BRIDGE=ROOT.parent/'rapfi_250615_quartz_bridge_build_20261125/pbrain-rapfi'
LOCK=json.loads((D/'RAPFI_LOCK.json').read_text());AS={x['name']:x for x in LOCK['files']};W=pathlib.Path(AS['Rapfi_NNUE_freestyle']['path']);EXE=pathlib.Path(AS['Rapfi_executable']['path'])
sys.path.insert(0,str(R/'quartz_tactical08_20261125'))
from rules import replay,orbit_board,win_at,proof_reference,fixtures
SEED=2026112634;ARMS=('B128','B80','F128','F80','F80L','F80L4');CAND=ARMS[1:];PAIRS=36;EXT_PAIRS=6
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
    return dict(binary=b['binary_sha256'],source_head=b['source_head'],bridge=sha(BRIDGE),external=sha(EXE),weight=sha(W),config=AS['Rapfi_config']['sha256'],upstream=LOCK['source_commit'])
def family(h):
    b,p,w=replay(h[:6],'freestyle');assert w is None
    return hashlib.sha256(orbit_board(b,p,'freestyle')).hexdigest()
def historical():
    files=[]
    # Explicit prior plan inventory, not a claim every historic trajectory is excluded.
    for d in sorted(R.glob('quartz_rapfi*'))+sorted(R.glob('quartz_kinetic*'))+[R/'quartz_preflight07_20261125',R/'quartz_tactical08_20261125']:
        if d==D:continue
        for pattern in ('*PLAN.json','*COHORT.json'):
            for p in sorted(d.glob(pattern)):
                if p.stat().st_size<=10_000_000:files.append(p)
    fam=set();locks=[]
    def visit(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k in ('opening','history','final_history') and isinstance(v,list) and len(v)>=6 and all(type(a)==int and 0<=a<225 for a in v[:6]) and len(set(v[:6]))==6:
                    fam.add(family(v[:6]))
                elif k not in ('board','policy','features'):visit(v)
        elif isinstance(x,list):
            for v in x:
                if isinstance(v,(list,dict)):visit(v)
    for p in files:
        before=len(fam);visit(json.loads(p.read_text()))
        if len(fam)>before:locks.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),added=len(fam)-before))
    return fam,locks
def plan():
    fam,locks=historical();prev=len(fam);rng=random.Random(SEED);jobs=[]
    for i in range(PAIRS+EXT_PAIRS):
        cells=[15*r+c for r in range(4,11) for c in range(4,11)] if i%2==0 else [15*r+c for r in range(2,13) for c in range(2,13)]
        while True:
            h=rng.sample(cells,6);k=family(h)
            if k not in fam:break
        fam.add(k);jobs.append(dict(id=i,opening=h,family=k,kind='MAIN' if i<PAIRS else 'EXTERNAL',stratum='central' if i%2==0 else 'broad',
            seed=rng.randrange(2**63),color_order=rng.sample([1,-1],2)))
    save('PLAN.json',dict(identity=identity(),seed=SEED,main_pairs=PAIRS,external_pairs=EXT_PAIRS,prior_families=prev,prior_locks=locks,jobs=jobs,
        arms=ARMS,NI_margin=.05,NN_reduction=.30,external_turn_ms=100,scope='fixed common proof main, separate actual external Rapfi finite time; no refit'))
    print('PLAN',prev,len(locks),len(jobs),flush=True)
class Server:
    def __init__(self,bin=BIN):
        env=dict(os.environ,QUARTZ_RAPFI_EVAL_EXE=str(BRIDGE),QUARTZ_RAPFI_EVAL_WEIGHT=str(W),RAYON_NUM_THREADS='4',QUARTZ_DISABLE_QIPC_SHM='1')
        self.err=open(BIN.parent/'arena_stderr.log','a')
        self.p=SP.Popen(['nice','-n','5',str(bin),'--server'],stdin=SP.PIPE,stdout=SP.PIPE,stderr=self.err,env=env,start_new_session=True,bufsize=0)
        self.buf=b''
        try:
            a=self.request(dict(cmd='research_rapfi_shared_open'))
            assert a['status']=='OPEN' and a['ready']=='250615 mix9svq freestyle15'
        except BaseException:self.abort();raise
    def request(self,q):
        t=time.perf_counter_ns();self.p.stdin.write((json.dumps(q,separators=(',',':'))+'\n').encode());self.p.stdin.flush()
        deadline=time.monotonic()+40
        while b'\n' not in self.buf:
            dt=deadline-time.monotonic()
            if dt<=0 or not select.select([self.p.stdout],[],[],dt)[0]:raise RuntimeError('native timeout')
            a=os.read(self.p.stdout.fileno(),65536)
            if not a:raise RuntimeError('native EOF '+str(self.p.poll()))
            self.buf+=a
            assert len(self.buf)<8*1024*1024
        line,self.buf=self.buf.split(b'\n',1);a=json.loads(line)
        a['transport_ns']=time.perf_counter_ns()-t
        return a
    def search(self,h,arm,seed,trace=False):
        a=self.request(dict(cmd='research_rapfi_k34',history=h,arm=arm,seed=seed,trace=trace))
        check(a,h,arm,seed,trace);return a
    def abort(self):
        if self.p.poll() is None:
            os.killpg(self.p.pid,signal.SIGTERM)
            try:self.p.wait(timeout=3)
            except SP.TimeoutExpired:os.killpg(self.p.pid,signal.SIGKILL);self.p.wait()
        self.err.close()
    def close(self):
        try:
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
def check(a,h,arm,seed,trace=False):
    assert a.get('schema')=='quartz_rapfi_k34_v1',a
    b,p,w=replay(h,'freestyle');assert w is None and a['history']==h and a['player']==p and a['arm']==arm and a['seed']==seed
    cap=128 if arm.endswith('128') else 80;assert a['cap']==cap and a['threads']==(4 if arm=='F80L4' else 1)
    pr=a['proof'];legal=[i for i,v in enumerate(b) if not v]
    assert pr['windows']==572 and [x[0] for x in pr['actions']]==legal and a['best'] in legal
    if pr['gate'] is not None:
        assert a['best']==pr['gate'] and a['cause']==pr['kind'] and a['NN_calls']==a['visits']==0 and a['root_eval'] is None and a['hooks'] is None and a['waves']==[]
        return
    r=a['result'];mid=a['prefix']['result'];ini=a['initial'];pb=a['prefix']['budget'];ph=a['prefix']['hooks'];eh=a['hooks']
    assert ini['evaluator_calls']==1 and ini['root_n']==0 and ini['candidates']==mid['candidates']==r['candidates']
    assert pb['used_nn']+pb['unused']==15 and pb['visits']==mid['root_n']<=16 and pb['used_nn']==mid['evaluator_calls']-1
    assert ph['ledger']['iterations']==mid['root_n'] and ph['ledger']['transferred']==0
    n=mid['root_n'];nn=mid['evaluator_calls'];d=reach(mid)
    for z in a['waves']:
        assert (z['before_n'],z['before_nn'],z['before_d'])==(n,nn,d)
        if arm.endswith('L') or arm.endswith('L4'):assert d is not None and d<=cap-n
        t=z['budget'];assert t['target']==cap-nn and t['cap']==min(8,cap-n) and t['used_nn']+t['unused']==t['target']
        n+=t['visits'];nn+=t['used_nn'];d=z['after_d'];assert (n,nn)==(z['after_n'],z['after_nn'])
    assert (a['visits'],a['NN_calls'])==(n,nn)==(r['root_n'],r['evaluator_calls']) and n<=cap and nn<=cap and a['best']==r['best'] and reach(r)==d
    if a['cause']=='VOTE_LOCK':assert arm in ('F80L','F80L4') and n<cap and nn<cap and (d is None or d>cap-n)
    elif a['cause']=='NN_TARGET':assert nn==cap
    else:assert a['cause']=='VISIT_CAP' and n==cap
    assert eh['ledger']['iterations']==n and eh['ledger']['transferred']==eh['ledger']['receipt_items'] and eh['ledger']['repeated_delivery']==0
    assert eh['ledger']['eligible']==eh['ledger']['novel']+eh['ledger']['blocked_receipt']
    h0=eh['hooks'];assert h0['base_evals']+h0['extra_evals']==nn-mid['evaluator_calls'] and h0['base_evals']+h0['terminal_leaves']==n-mid['root_n']
    if arm.startswith('B'):assert eh['ledger']['transferred']==h0['extra_evals']==h0['select_nodes']==h0['width_nodes']==0
    assert 0<=eh['peak']<=a['threads'] and (eh['trace'] is not None)==trace
    ev=a['root_eval'];assert ev['board']==''.join(map(str,b)) and ev['player']==p and len(ev['policy'])==225 and abs(sum(ev['policy'])-1)<1e-5
    assert all(ev['policy'][i]==0 for i,v in enumerate(b) if v)
    assert all(x['vl']==0 and x['vvalue']==0 for x in r['edges'])
def smoke():
    f=fixtures();hs=[f['trans_a'],f['trans_b'],[110,0,111,2,113,4],f['immediate'],f['double_loss'],[112,97,113,127,98,126]]
    rows=[]
    with Server() as s:
        for h in hs:
            z={k:s.search(h,k,17,True) for k in ARMS}
            assert all(x['proof']==z['B128']['proof'] for x in z.values())
            if z['B128']['root_eval'] is not None:
                assert all(x['prefix']['result']==z['B128']['prefix']['result'] and x['root_eval']==z['B128']['root_eval'] for x in z.values())
            assert z['F80']['best']==z['F80L']['best']
            rows.append(z);print('SMOKE',h,[(k,x['cause'],x['NN_calls'],x['visits'],x['best']) for k,x in z.items()],flush=True)
        for q in [dict(cmd='research_rapfi_k34',history=[],arm='BAD',seed=17),dict(cmd='research_rapfi_k34',history=[0,0],arm='B128',seed=17),dict(cmd='research_rapfi_k34',history=[],arm='B128',seed=17,rule='renju')]:
            assert 'error' in s.request(q)
    save('SMOKE.json',dict(rows=rows,rejected=3,identity=identity()))
def game(s,j,arm,color,external=False):
    h=list(j['opening']);rows=[]
    while True:
        b,p,w=replay(h,'freestyle')
        if w is not None:break
        who=arm if p==color else ('RAPFI100' if external else 'B128')
        seed=(j['seed']+len(h)*0x9e3779b97f4a7c15)&((1<<64)-1)
        if who=='RAPFI100':
            a=external_move(h);a['arm']=who;a['history']=h.copy();a['player']=p;a['seed']=seed;a['NN_calls']=None
        else:a=s.search(h,who,seed)
        m=a['best'];assert type(m)==int and 0<=m<225 and b[m]==0
        rows.append(a);h.append(m)
    return dict(job=j,arm=arm,color=color,external=external,history=h,winner=w,score=(1+color*w)/2,rows=rows)
# Reuse previously verified official Piskvork protocol (fresh process per move).
def external_move(h):
    p=R/'cognitive_metareasoning_20261120/highbranch_transfer_14/coding/rapfi_teacher14.py'
    spec=importlib.util.spec_from_file_location('rapfi_original34',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    a=m.query_one(EXE,EXE.parent,h,0,100,15)
    assert a['status']=='OK' and a['compatible_weight_reported'] and a['saw_loaded_nnue'],a
    a['best']=a['flat_action'];a['transport_ns']=int(a['elapsed_wall_ms']*1e6)
    return a
def worker(arg):
    arm,shard,rr,external=arg;identity()
    name=('EXT' if external else 'MAIN')+'_'+arm+'_'+str(shard)+'.jsonl.gz';p=D/name;assert not p.exists();tmp=p.with_suffix('.tmp')
    c=Counter()
    with Server() as s,gzip.open(tmp,'wt',compresslevel=3) as f:
        for j in rr:
            for color in j['color_order']:
                z=game(s,j,arm,color,external);emit(f,z);f.flush()
                c.update(games=1,moves=len(z['rows']),NN=sum(x['NN_calls'] or 0 for x in z['rows']),external_moves=sum(x['NN_calls'] is None for x in z['rows']))
            print('PAIR',name,j['id'],dict(c),flush=True)
    tmp.replace(p);return dict(file=name,sha256=sha(p),**c)
def collect(mode):
    p=read('PLAN.json');assert p['identity']==identity();ext=mode=='EXT'
    rr=[j for j in p['jobs'] if j['kind']==('EXTERNAL' if ext else 'MAIN')]
    arms=('B128','F80L') if ext else CAND;shards=2 if ext else 3;tasks=[(a,k,rr[k::shards],ext) for a in arms for k in range(shards)]
    random.Random(SEED+3+ext).shuffle(tasks)
    with ProcessPoolExecutor(max_workers=4 if ext else 6) as ex:fs=list(ex.map(worker,tasks))
    save(mode+'_LOCK.json',dict(plan=sha(D/'PLAN.json'),code=sha(D/'arena.py'),files=fs,**{k:sum(z[k] for z in fs) for k in ('games','moves','NN','external_moves')}))
    print('COLLECT',mode,read(mode+'_LOCK.json')|{'files':len(fs)},flush=True)
def summarize(mode):
    lk=read(mode+'_LOCK.json');assert lk['plan']==sha(D/'PLAN.json');arms=defaultdict(list)
    for fi in lk['files']:
        p=D/fi['file'];assert sha(p)==fi['sha256']
        for g in raw(p):arms[g['arm']].append(g)
    out={}
    for arm,gs in arms.items():
        groups=defaultdict(list)
        for g in gs:groups[g['job']['id']].append(g)
        assert len(groups)==(EXT_PAIRS if mode=='EXT' else PAIRS) and all(len(g)==2 and {x['color'] for x in g}=={1,-1} for g in groups.values())
        def count(gg):
            c=Counter()
            for g in gg:
                for x in g['rows']:
                    tag='candidate' if x['player']==g['color'] else 'base'
                    c[tag+'_moves']+=1;c[tag+'_NN']+=x['NN_calls'] or 0;c[tag+'_wall_ns']+=x['transport_ns']
                    if x['NN_calls'] is not None:
                        c[tag+'_visits']+=x['visits'];c[tag+'_'+x['cause']]+=1;c[tag+'_proof_us']+=x['proof_us']
                        if x['hooks']:
                            z=x['hooks'];h=z['hooks'];l=z['ledger']
                            for k in ('select_changed','refinements','eval_changed','extra_evals'):c[tag+'_'+k]+=h[k]
                            c[tag+'_width_realized']+=z['width_realized'];c[tag+'_transferred']+=l['transferred'];c[tag+'_blocked']+=l['blocked_receipt']
                            c[tag+'_all4']+=int(h['select_changed']>0 and z['width_realized']>0 and h['eval_changed']>0 and l['transferred']>0)
                            c[tag+'_peak4']+=z['peak']==4
            return c
        c=count(gs);pair={i:S.mean(g['score'] for g in v) for i,v in groups.items()};pc={i:count(v) for i,v in groups.items()}
        def saving(z):
            return 1-(z['candidate_NN']/z['candidate_moves'])/(z['base_NN']/z['base_moves'])
        rng=random.Random(SEED+41);bs=[];bn=[];ids=list(sorted(groups))
        for _ in range(4000):
            pp=rng.choices(ids,k=len(ids));bs.append(S.mean(pair[i] for i in pp))
            if mode=='MAIN':
                cc=Counter()
                for i in pp:cc.update(pc[i])
                bn.append(saving(cc))
        lo=sorted(bs)[199];out[arm]=dict(games=len(gs),pairs=len(groups),wins=sum(g['score']==1 for g in gs),draws=sum(g['score']==.5 for g in gs),losses=sum(g['score']==0 for g in gs),score=S.mean(pair.values()),score_ci=[sorted(bs)[99],sorted(bs)[3899]],NI_lower=lo,counts=dict(c),side_scores={str(p):S.mean(g['score'] for g in gs if g['color']==p) for p in (1,-1)})
        if mode=='MAIN':out[arm].update(NN_reduction=saving(c),NN_ci=[sorted(bn)[99],sorted(bn)[3899]],point_gate=lo>=.45 and saving(c)>=.30)
    save(mode+'_SUMMARY.json',dict(arms=out,scope='opening-pair bootstrap, fixed panel, not general noninferiority or matched wall'))
    print('SUMMARY',mode,json.dumps(out,sort_keys=True),flush=True)
def diagnostic():
    plan=read('PLAN.json');rr=[j for j in plan['jobs'] if j['kind']=='MAIN'][:12];rows=[]
    with Server() as s:
        for j in rr:
            h=j['opening'];z={a:s.search(h,a,j['seed'],True) for a in ARMS};assert z['F80']['best']==z['F80L']['best'];rows.append(dict(job=j,results=z))
    p=D/'DIAGNOSTIC.jsonl.gz';assert not p.exists()
    with gzip.open(p,'wt',compresslevel=3) as f:
        for z in rows:emit(f,z)
    save('DIAGNOSTIC_LOCK.json',dict(n=len(rows),requests=len(rows)*len(ARMS),sha256=sha(p),scope='fixed first12 opening roots, raw receipts and serial lock parity'))
    print('DIAG',len(rows),flush=True)
if __name__=='__main__':
    if sys.argv[1] in ('collect','summarize'):globals()[sys.argv[1]](sys.argv[2])
    else:globals()[sys.argv[1]]()
