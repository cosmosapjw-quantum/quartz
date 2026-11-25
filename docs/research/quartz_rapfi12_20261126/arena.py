"""Same frozen Rapfi NN on both sides; bounded research matches, never a teacher."""
import collections as C,json,math,os,random,select,signal,statistics as S,subprocess as SP,sys,time,hashlib,gzip
from pathlib import Path
from publish import D,ROOT,sha
from probe_rapfi import EXE,W,LOCK,call
sys.path.insert(0,str(ROOT/'docs/research/quartz_tactical08_20261125'))
from rules import replay,proof_reference,orbit_board,fixtures
BIN=ROOT.parent/'quartz_rapfi12_target_20261126/mcts_demo'
ARMS=('STANDARD_PW','STANDARD_FULL','QUARTZ')
def write(n,x):
    p=D/n;assert not p.exists();p.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n')
def read(n):return json.loads((D/n).read_text())
def identity():
    a=read('RUST_BUILD_IDENTITY.json');b=read('RAPFI_EVAL_SMOKE.json')
    assert sha(BIN)==a['binary_sha256'] and sha(EXE)==b['bridge_sha256']
    for r in LOCK['files']:
        p=Path(r['path']);assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
    return dict(rust_binary=sha(BIN),rust_source=a['source_head'],rapfi_bridge=sha(EXE),rapfi_source=LOCK['source_commit'],rapfi_weight=sha(W),rapfi_config=sha(W.parent/'config.toml'))
class Server:
    def __init__(self):
        self.ident=identity()
        env=dict(os.environ,QUARTZ_RAPFI_EVAL_EXE=str(EXE),QUARTZ_RAPFI_EVAL_WEIGHT=str(W),RAYON_NUM_THREADS='1',QUARTZ_DISABLE_QIPC_SHM='1',QUARTZ_DISABLE_NN_CACHE='1',QUARTZ_DISABLE_ASYNC_PIPELINE='1')
        self.err=open(BIN.parent/'arena_stderr.log','a+')
        self.p=SP.Popen(['nice','-n','10',str(BIN),'--server'],stdin=SP.PIPE,stdout=SP.PIPE,stderr=self.err,bufsize=0,env=env,start_new_session=True)
        self.buf=b''
        try:
            self.open=self.request(dict(cmd='research_rapfi_shared_open'))
            assert self.open['status']=='OPEN' and self.open['ready']=='250615 mix9svq freestyle15'
        except BaseException:
            self.abort();raise
    def request(self,q):
        start=time.perf_counter_ns()
        self.p.stdin.write((json.dumps(q,separators=(',',':'))+'\n').encode());self.p.stdin.flush()
        deadline=time.monotonic()+30
        while b'\n' not in self.buf:
            t=deadline-time.monotonic()
            if t<=0 or not select.select([self.p.stdout],[],[],t)[0]:raise RuntimeError('native request timeout')
            x=os.read(self.p.stdout.fileno(),65536)
            if not x:raise RuntimeError('native EOF: '+str(self.p.poll()))
            self.buf+=x
            if len(self.buf)>4*1024*1024:raise RuntimeError('oversized native reply')
        line,self.buf=self.buf.split(b'\n',1)
        a=json.loads(line)
        a['transport_ns']=time.perf_counter_ns()-start
        return a
    def search(self,h,arm,cap=256,seed=17,prefix=None):
        q=dict(cmd='research_rapfi_shared_search',rule='freestyle',history=h,arm=arm,visits=cap,seed=seed)
        if prefix is not None:q['prefix']=prefix
        a=self.request(q)
        assert 'error' not in a,a
        check(a,h,arm,cap,seed)
        return a
    def abort(self):
        if self.p.poll() is None:
            os.killpg(self.p.pid,signal.SIGTERM)
            try:self.p.wait(timeout=3)
            except SP.TimeoutExpired:
                os.killpg(self.p.pid,signal.SIGKILL);self.p.wait()
        self.err.close()
    def close(self):
        try:
            a=self.request(dict(cmd='research_rapfi_shared_close'));assert a['status']=='CLOSED'
            self.p.stdin.close();assert self.p.wait(timeout=10)==0
        except BaseException:
            self.abort();raise
        self.err.close()
    def __enter__(self):return self
    def __exit__(self,t,*args):
        if t is None:self.close()
        else:self.abort()
def check(a,h,arm,cap,seed):
    b,p,w=replay(h,'freestyle');assert w is None
    legal=[i for i,v in enumerate(b) if not v];pr=proof_reference(b,p,'freestyle')
    assert a['schema']=='quartz_rapfi_shared_v1' and a['status']=='OK' and a['evaluator']=='rapfi_250615_mix9svq'
    assert a['history']==h and a['board']==b and a['player']==p and a['legal']==legal and a['proof']==pr and a['arm']==arm and a['cap']==cap and a['seed']==seed and a['threads']==1
    assert a['best'] in legal
    guard=pr['own'][0] if pr['own'] else pr['opponent'][0] if len(pr['opponent'])==1 else legal[0] if len(legal)==1 else None
    assert a['whole_us']*1000<=a['transport_ns']+1000000
    if guard is not None:
        assert a['cause']=='GUARDED' and a['best']==guard and a['NN_calls']==a['visits']==0 and a['root_eval'] is None and a['result'] is None and a['quartz'] is None
        return
    n=a['visits'];assert 1<=n<=(2*cap if arm=='P5ALLOC' else cap) and (arm in ('QUARTZ','DSSTOP','P5PW','P5FULL','P5Q','P5ALLOC') or n==cap)
    r=a['result'];init=a['initial'];ev=a['root_eval']
    assert ev['board']==''.join(map(str,b)) and ev['player']==p
    wdl=[ev[x] for x in ('win','draw','loss')];pi=ev['policy']
    assert len(pi)==225 and all(math.isfinite(x) and x>=0 for x in wdl+pi) and abs(sum(wdl)-1)<1e-6 and abs(sum(pi)-1)<1e-5
    assert abs(ev['value']-(ev['win']-ev['loss']))<1e-7
    assert all(pi[i]==0 for i,v in enumerate(b) if v)
    assert init['root_n']==0 and init['evaluator_calls']==1 and init['candidates']==r['candidates']
    assert len(r['candidates'])==len(legal) and {m for m,v in r['candidates']}==set(legal)
    assert all(abs(v-pi[m])<1e-8 for m,v in r['candidates'])
    assert r['root_n']==n and r['best']==a['best'] and r['evaluator_calls']==a['NN_calls'] and a['NN_calls']>=1
    es=r['edges'];assert [e['move'] for e in es]==[m for m,v in r['candidates'][:len(es)]]
    assert sum(e['n'] for e in es)==n and abs(sum(e['w'] for e in es)-r['root_w'])<1e-6
    for e in es:
        z=e['n'];W=e['w'];M=e['m2']
        assert e['vl']==0 and abs(e['vvalue'])<1e-6 and abs(W)<=z+1e-6 and -1e-7<=M<=(z-W*W/z if z else 0)+1e-5
    if arm in ('QUARTZ','P5Q'):assert a['quartz'] is not None and (arm=='P5Q' or a['quartz']['checks'])
    else:assert a['quartz'] is None
    if arm in ('STANDARD_FULL','P5FULL'):assert len(es)==len(legal)
def smoke():
    f=fixtures();f['block']=[0,100,2,101,4,102,99,103]
    f['odd']=f['trans_a']+[0]
    roots=['trans_a','odd','immediate','block','double_loss']
    out=[];rejected=[]
    with Server() as s:
        for name in roots:
            h=f[name]
            a=[s.search(h,m,256 if m!='STANDARD_FULL' else 64) for m in ARMS]
            assert a[0]['root_eval']==a[1]['root_eval']==a[2]['root_eval']
            assert a[0]['initial'] is None or a[0]['initial']['candidates']==a[1]['initial']['candidates']==a[2]['initial']['candidates']
            if a[0]['cause']=='GUARDED':assert len({x['best'] for x in a})==1
            for x in a:
                x['fixture']=name;out.append(x)
                print('SMOKE',name,x['arm'],x['best'],x['visits'],x['NN_calls'],x['cause'],x['transport_ns']/1e6,flush=True)
        for q in [dict(cmd='research_rapfi_shared_search',rule='renju',history=[],arm='QUARTZ',visits=8,seed=17),dict(cmd='research_rapfi_shared_search',rule='freestyle',history=[0,0],arm='QUARTZ',visits=8,seed=17),dict(cmd='research_rapfi_shared_open',extra=1)]:
            r=s.request(q);assert 'error' in r;rejected.append(dict(request=q,response=r))
        opened=s.open
    independent=call([f['trans_a'],f['odd'],f['double_loss']])
    for a in out:
        if a['root_eval'] is None:continue
        v=next(z for z in independent if z['history']==a['history']);ev=a['root_eval']
        assert all(abs(x-y)<1e-7 for x,y in zip(ev['policy'],v['policy']))
        assert all(abs(ev[k]-p)<1e-7 for k,p in zip(('win','draw','loss'),v['wdl']))
    write('SHARED_SMOKE.json',dict(identity=identity(),open=opened,rows=out,rejected=rejected,independent_evaluator_comparison=True))
    print('PASS shared root priors+WDL across all three arms, both sides and guard',flush=True)
def plan():
    # Separate new D4 opening family; do not use the published calibration roots.
    old=set();p=ROOT/'docs/research/quartz_tactical08_20261125/C07_COHORT.json'
    for r in json.loads(p.read_text())['CAL']+json.loads(p.read_text())['TEST']:
        b,pl,w=replay(r['history'][:6],'freestyle');old.add(orbit_board(b,pl,'freestyle'))
    root=ROOT/'docs/research/quartz_preflight07_20261125/P2_CORPUS'
    for f in root.glob('shard_*/*.json.gz'):
        with gzip.open(f,'rt') as z:g=json.load(z)
        b,pl,w=replay(g['job']['opening'],'freestyle');old.add(orbit_board(b,pl,'freestyle'))
    rng=random.Random(2026112509);cells=[15*r+c for r in range(4,11) for c in range(4,11)]
    while True:
        h=rng.sample(cells,6);b,pl,w=replay(h,'freestyle')
        if w is None and orbit_board(b,pl,'freestyle') not in old and not proof_reference(b,pl,'freestyle')['own'] and not proof_reference(b,pl,'freestyle')['opponent']:break
    p=dict(identity=identity(),opening=h,family=hashlib.sha256(orbit_board(b,pl,'freestyle')).hexdigest(),games=[dict(id='PAIR0_Q_BLACK',black='QUARTZ',white='STANDARD_PW'),dict(id='PAIR0_Q_WHITE',black='STANDARD_PW',white='QUARTZ')],cap=256,seed=17,max_ply=225,scope='one natural color-swapped pair, fixed visit caps not matched time; engineering pilot only; no training')
    write('PAIR_PLAN.json',p);print('PLAN',h,p['family'],flush=True)
def game(i):
    p=read('PAIR_PLAN.json');assert p['identity']==identity()
    job=p['games'][i];h=list(p['opening']);rows=[];start=time.monotonic()
    name='PAIR_GAME_'+str(i)+'.json.gz';path=D/name;assert not path.exists()
    with Server() as s:
        while True:
            b,side,w=replay(h,'freestyle')
            if w is not None:break
            arm=job['black'] if side==1 else job['white']
            a=s.search(h,arm,p['cap'],p['seed'])
            rows.append(dict(ply=len(h),player=side,played=a['best'],response=a))
            h.append(a['best'])
            if len(h)%10==0:print(job['id'],'ply',len(h),'elapsed_s',round(time.monotonic()-start,2),flush=True)
        assert len(h)<=p['max_ply']
    out=dict(identity=identity(),job=job,opening=p['opening'],family=p['family'],final_history=h,winner_black=w,rows=rows,status='COMPLETED',seconds=time.monotonic()-start,plan_sha256=sha(D/'PAIR_PLAN.json'))
    for row in rows:row['z']=w*row['player']
    data=gzip.compress((json.dumps(out,separators=(',',':'),allow_nan=False)+'\n').encode(),mtime=0)
    path.write_bytes(data)
    print('COMPLETED',job['id'],len(h),w,'moves',len(rows),'NN',sum(x['response']['NN_calls'] for x in rows),'seconds',out['seconds'],flush=True)
if __name__=='__main__':
    if sys.argv[1]=='smoke':smoke()
    elif sys.argv[1]=='plan':plan()
    elif sys.argv[1]=='game':game(int(sys.argv[2]))
