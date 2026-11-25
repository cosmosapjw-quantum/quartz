"""Frozen real MCTS closed loop with independent response and yoked controls."""
from arena import Server,identity,check,write,read,D,ROOT,BIN,sha,replay,proof_reference,orbit_board,fixtures
from monitor_ref import verify
import collections as C,gzip,json,hashlib,random,statistics as S,math,time,sys,os,importlib.util,ctypes
from pathlib import Path
ARMS=('META','OPEN','MBASE','YOKE')
def rows(name):
    with gzip.open(D/name,'rt') as f:
        for line in f:yield json.loads(line)
def emit(f,x):
    f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n');f.flush()
def host():
    z={}
    for line in open('/proc/vmstat'):
        k,v=line.split()
        if k in ('pswpin','pswpout'):z[k]=int(v)
    return dict(mono=time.monotonic(),load=os.getloadavg(),swap=z)
def host_end(start,peak):
    e=host();dt=e['mono']-start['mono']
    return dict(start=start,end=e,seconds=dt,swap_pages_s=sum(e['swap'][k]-start['swap'][k] for k in e['swap'])/dt,max_load=peak)
def process_clock(pid):
    # POSIX/Linux clock_getcpuclockid, returns an error number, not errno.
    lib=ctypes.CDLL(None)
    f=lib.clock_getcpuclockid;f.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_int)];f.restype=ctypes.c_int
    clock=ctypes.c_int()
    rc=f(pid,ctypes.byref(clock))
    if rc:raise OSError(rc,os.strerror(rc))
    return clock.value
class Meter:
    def __init__(self,s):
        self.s=s;self.start=host();self.peak=self.start['load'][0]
        ch=Path(f'/proc/{s.p.pid}/task/{s.p.pid}/children').read_text().split();assert len(ch)==1,ch
        self.ids=[process_clock(s.p.pid),process_clock(int(ch[0]))]
        self.res=[time.clock_getres(x) for x in self.ids]
        assert all(v<=1e-6 for v in self.res),self.res
    def search(self,h,arm,cap=256,seed=17,schedule=None):
        t=[time.clock_gettime_ns(i) for i in self.ids];p0=time.process_time_ns()
        a=self.s.search(h,arm,cap,seed,schedule=schedule)
        py=time.process_time_ns()-p0
        delta=[time.clock_gettime_ns(i)-v for i,v in zip(self.ids,t)]
        assert all(x>=0 for x in delta)
        a['measure']=dict(rust_cpu_ns=delta[0],rapfi_cpu_ns=delta[1],python_cpu_ns=py,clock_res_s=self.res)
        self.peak=max(self.peak,os.getloadavg()[0])
        if arm in ARMS:verify(a,seed,schedule)
        return a
    def end(self):return host_end(self.start,self.peak)
def family(h):
    b,p,w=replay(h[:6],'freestyle');assert w is None
    return hashlib.sha256(orbit_board(b,p,'freestyle')).hexdigest()
def historical():
    p=ROOT/'docs/research/quartz_rapfi12_20261126/experiment.py'
    sp=importlib.util.spec_from_file_location('prior_sources',p);old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
    fam,boards=old.historical()
    for n in ('C06_SOURCE_GAMES.jsonl.gz','C08_ARENA_RAW.jsonl.gz'):
        with gzip.open(p.parent/n,'rt') as f:
            for line in f:
                g=json.loads(line)
                fam.add(family(g['job']['opening']))
                for h in [r['response']['history'] for r in g['rows']]+[g['final_history']]:
                    b,p0,w=replay(h,'freestyle');boards.add(orbit_board(b,p0,'freestyle'))
    return fam,boards
def smoke():
    f=fixtures();out=[];reject=[]
    with Server() as s:
        m=Meter(s)
        for nm in ('trans_a','long','immediate'):
            h=f[nm]
            a=[m.search(h,arm,256,17) for arm in ARMS[:3]]
            sch=[t['channel'] for t in a[0]['meta']['steps']] if a[0]['meta'] else [0]*15
            y=m.search(h,'YOKE',256,17,sch);a.append(y)
            if a[0]['meta']:
                assert a[0]['result']==y['result'] and a[0]['meta']['b_trace']==y['meta']['b_trace']
                assert all(z['root_eval']==a[0]['root_eval'] and z['meta']['prefix']['result']==a[0]['meta']['prefix']['result'] for z in a)
                std=m.search(h,'STANDARD_PW',256,17);assert std['result']==a[2]['result']
            else:
                assert all(z['cause']=='GUARDED' and z['best']==a[0]['best'] for z in a);std=None
            out.append(dict(fixture=nm,rows=a,standard=std,schedule=sch))
            print('SMOKE',nm,[(z['arm'],z['visits'],z['NN_calls'],z['best']) for z in a],flush=True)
        for q in [dict(cmd='research_rapfi_shared_search',rule='freestyle',history=[],arm='YOKE',visits=256,seed=17,schedule=[0]*14),dict(cmd='research_rapfi_shared_search',rule='freestyle',history=[],arm='META',visits=256,seed=17,schedule=[0]*15),dict(cmd='research_rapfi_shared_search',rule='freestyle',history=[],arm='META',visits=128,seed=17)]:
            z=s.request(q);assert 'error' in z;reject.append(z)
        ho=m.end()
    write('C06_SMOKE.json',dict(identity=identity(),rows=out,rejected=reject,host=ho))
def plan():
    old,boards=historical();n0=len(old);rng=random.Random(2026112613);cells=[15*i+j for i in range(4,11) for j in range(4,11)]
    jobs=[]
    for i in range(16):
        for _ in range(1000):
            h=rng.sample(cells,6);k=family(h)
            if k not in old:break
        else:raise RuntimeError('opening exhaustion')
        old.add(k);jobs.append(dict(id=f'G{i:02d}',group='CAL' if i<4 else 'TEST',family=k,opening=h,control_seed=17+i))
    write('C06_SOURCE_PLAN.json',dict(identity=identity(),jobs=jobs,excluded_families=n0,excluded_boards=len(boards),source_cap=128,continuation_caps=[128,512]))
def play(m,h,cap):
    h=list(h);rr=[]
    while True:
        b,p,w=replay(h,'freestyle')
        if w is not None:return h,w,rr
        a=m.search(h,'STANDARD_PW',cap,17)
        rr.append(dict(ply=len(h),player=p,move=a['best'],response=a));h.append(a['best'])
def source():
    p=read('C06_SOURCE_PLAN.json');assert p['identity']==identity();_,boards=historical();co=[]
    n=D/'C06_SOURCE_GAMES.jsonl.gz';assert not n.exists()
    with Server() as s,gzip.open(n,'wt',compresslevel=3) as f:
        m=Meter(s)
        for j in p['jobs']:
            h,w,rr=play(m,j['opening'],128);elig=[]
            for row in rr:
                a=row['response'];b,p0,_=replay(a['history'],'freestyle');k=orbit_board(b,p0,'freestyle')
                if not a['proof']['own'] and not a['proof']['opponent'] and k not in boards:elig.append((a,k))
            z=None
            if elig:
                a,k=min(elig,key=lambda x:(abs(len(x[0]['history'])-18),len(x[0]['history'])))
                z=dict(id=j['id'],family=j['family'],group=j['group'],history=a['history'],player=a['player'],control_seed=j['control_seed']);boards.add(k);co.append(z)
            emit(f,dict(job=j,final_history=h,winner_black=w,rows=rr,selected=z))
            print('SOURCE',j['id'],len(h),w,'root',None if z is None else len(z['history']),flush=True)
        ho=m.end()
    write('C06_COHORT.json',dict(identity=identity(),roots=co,missing=[j['id'] for j in p['jobs'] if j['id'] not in {z['id'] for z in co}],source_sha256=sha(n),host=ho))
def primary():
    co=read('C06_COHORT.json')['roots'];n=D/'C07_PRIMARY_RAW.jsonl.gz';assert not n.exists()
    with Server() as s,gzip.open(n,'wt',compresslevel=3) as f:
        m=Meter(s)
        for i,r in enumerate(co):
            aa={};arms=ARMS[:3];off=i%3
            for arm in arms[off:]+arms[:off]:
                a=m.search(r['history'],arm,256,r['control_seed']);assert a['meta'] is not None
                emit(f,dict(root=r,response=a));aa[arm]=a
            z=aa['META'];assert all(a['root_eval']==z['root_eval'] and a['meta']['prefix']['result']==z['meta']['prefix']['result'] for a in aa.values())
            print('PRIMARY',r['id'],[(k,a['best']) for k,a in aa.items()],flush=True)
        ho=m.end()
    write('C07_PRIMARY_LOCK.json',dict(identity=identity(),requests=len(co)*3,raw_sha256=sha(n),host=ho))
def yoke():
    co=read('C06_COHORT.json')['roots'];assert sha(D/'C07_PRIMARY_RAW.jsonl.gz')==read('C07_PRIMARY_LOCK.json')['raw_sha256']
    z={x['root']['id']:x['response'] for x in rows('C07_PRIMARY_RAW.jsonl.gz') if x['response']['arm']=='META'}
    mapping={}
    for group in ('CAL','TEST'):
        rr=sorted((r for r in co if r['group']==group),key=lambda r:r['id']);assert len(rr)>1
        for i,r in enumerate(rr):
            d=rr[(i+1)%len(rr)]['id'];mapping[r['id']]=dict(donor=d,schedule=[t['channel'] for t in z[d]['meta']['steps']])
    write('C07_YOKE_PLAN.json',dict(identity=identity(),primary_sha256=sha(D/'C07_PRIMARY_RAW.jsonl.gz'),mapping=mapping))
def yoked():
    plan=read('C07_YOKE_PLAN.json');co=read('C06_COHORT.json')['roots']
    assert plan['primary_sha256']==sha(D/'C07_PRIMARY_RAW.jsonl.gz')
    prev={x['root']['id']:x['response'] for x in rows('C07_PRIMARY_RAW.jsonl.gz') if x['response']['arm']=='META'}
    n=D/'C07_YOKE_RAW.jsonl.gz';assert not n.exists()
    with Server() as s,gzip.open(n,'wt',compresslevel=3) as f:
        m=Meter(s)
        for r in co:
            z=plan['mapping'][r['id']]
            a=m.search(r['history'],'YOKE',256,r['control_seed'],z['schedule'])
            b=prev[r['id']];assert a['root_eval']==b['root_eval'] and a['meta']['prefix']['result']==b['meta']['prefix']['result']
            emit(f,dict(root=r,donor=z['donor'],response=a))
            print('YOKED',r['id'],z['donor'],a['best'],flush=True)
        ho=m.end()
    write('C07_YOKE_LOCK.json',dict(identity=identity(),requests=len(co),raw_sha256=sha(n),plan_sha256=sha(D/'C07_YOKE_PLAN.json'),host=ho))
def all_searches():
    for name,lock in [('C07_PRIMARY_RAW.jsonl.gz','C07_PRIMARY_LOCK.json'),('C07_YOKE_RAW.jsonl.gz','C07_YOKE_LOCK.json')]:
        assert sha(D/name)==read(lock)['raw_sha256']
        yield from rows(name)
def continuation():
    req={}
    for x in all_searches():
        a=x['response'];r=x['root']
        for k in (a['meta']['prefix']['result']['best'],a['best']):req[r['id'],k]=r
    assert len(req)<=80
    n=D/'C08_CONT_RAW.jsonl.gz';assert not n.exists()
    with Server() as s,gzip.open(n,'wt',compresslevel=3) as f:
        m=Meter(s)
        for (i,a),r in sorted(req.items()):
            for cap in (128,512):
                h,w,rr=play(m,r['history']+[a],cap)
                emit(f,dict(root=r,action=a,cap=cap,final_history=h,winner_black=w,utility=(1+r['player']*w)/2,rows=rr))
            print('CONT',i,a,flush=True)
        ho=m.end()
    write('C08_CONT_LOCK.json',dict(identity=identity(),unique_actions=len(req),count=2*len(req),raw_sha256=sha(n),host=ho))
if __name__=='__main__':{'smoke':smoke,'plan':plan,'source':source,'primary':primary,'yoke':yoke,'yoked':yoked,'continuation':continuation}[sys.argv[1]]()
