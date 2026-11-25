"""Prospective bounded real Rapfi decision exposure study."""
from arena import Server,identity,check,write,read,D,ROOT,BIN,sha,replay,proof_reference,orbit_board,fixtures
from response_ref import ARMS,CAPS,verify,panel
import collections as C,gzip,json,hashlib,random,statistics as S,math,time,sys,os,importlib.util,ctypes
from pathlib import Path
from itertools import product
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
def process_clock(pid):
    lib=ctypes.CDLL(None);f=lib.clock_getcpuclockid
    f.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_int)];f.restype=ctypes.c_int
    c=ctypes.c_int();rc=f(pid,ctypes.byref(c))
    if rc:raise OSError(rc,os.strerror(rc))
    return c.value
class Meter:
    def __init__(self,s):
        self.s=s;self.start=host();self.peak=self.start['load'][0]
        ch=Path(f'/proc/{s.p.pid}/task/{s.p.pid}/children').read_text().split();assert len(ch)==1
        self.ids=[process_clock(s.p.pid),process_clock(int(ch[0]))];self.res=[time.clock_getres(x) for x in self.ids]
        assert all(0<v<=1e-6 for v in self.res)
    def search(self,h,arm,cap,seed=17):
        t=[time.clock_gettime_ns(i) for i in self.ids];p=time.process_time_ns()
        a=self.s.search(h,arm,cap,seed)
        py=time.process_time_ns()-p;d=[time.clock_gettime_ns(i)-v for i,v in zip(self.ids,t)]
        assert min(d)>=0
        a['measure']=dict(rust_cpu_ns=d[0],rapfi_cpu_ns=d[1],python_cpu_ns=py,clock_res_s=self.res)
        self.peak=max(self.peak,os.getloadavg()[0])
        if arm in ARMS:verify(a)
        return a
    def end(self):
        e=host();dt=e['mono']-self.start['mono']
        return dict(start=self.start,end=e,seconds=dt,swap_pages_s=sum(e['swap'][k]-self.start['swap'][k] for k in e['swap'])/dt,max_load=self.peak)
def family(h):
    b,p,w=replay(h[:6],'freestyle');assert w is None
    return hashlib.sha256(orbit_board(b,p,'freestyle')).hexdigest()
def historical():
    p=ROOT/'docs/research/quartz_kinetic14_20261126/kinetic_run.py'
    sp=importlib.util.spec_from_file_location('prior_kinetic_sources',p);old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
    fam,boards=old.historical()
    for n in ('C06_SOURCE_GAMES.jsonl.gz','C08_TRAIN_LABELS.jsonl.gz','C08_HELDOUT_LABELS.jsonl.gz'):
        with gzip.open(p.parent/n,'rt') as f:
            for line in f:
                g=json.loads(line)
                if 'job' in g:fam.add(family(g['job']['opening']))
                for h in [r['response']['history'] for r in g['rows']]+[g['final_history']]+([g['root']['history']] if 'root' in g else []):
                    b,p0,w=replay(h,'freestyle');boards.add(orbit_board(b,p0,'freestyle'))
    return fam,boards
def smoke():
    f=fixtures();out=[];reject=[]
    with Server() as s:
        m=Meter(s)
        for nm in ('trans_a','long','immediate'):
            h=f[nm]
            for cap in CAPS:
                a=[m.search(h,arm,cap) for arm in ARMS];std=m.search(h,'STANDARD_PW',cap)
                if a[0]['exposure']:
                    pre=a[0]['exposure']['prefix']['result']
                    assert all(z['exposure']['prefix']['result']==pre for z in a)
                    assert a[0]['result']==std['result'] and a[-1]['best']==std['best']
                else:assert all(z['cause']=='GUARDED' and z['best']==std['best'] for z in a)
                out.append(dict(fixture=nm,cap=cap,rows=a,standard=std))
                print('SMOKE',nm,cap,[(z['arm'],z['best'],z['visits']) for z in a],flush=True)
        for q in (dict(arm='K15N',visits=64),dict(arm='K15N',visits=32,schedule=[0,0]),dict(arm='K15FH',visits=32,prefix=8)):
            a=s.request(dict(cmd='research_rapfi_shared_search',rule='freestyle',history=[],seed=17,**q));assert 'error' in a;reject.append(a)
        ho=m.end()
    write('C06_SMOKE.json',dict(identity=identity(),rows=out,rejected=reject,host=ho))
def plan():
    old,boards=historical();n=len(old);rng=random.Random(2026112615);cells=[15*i+j for i in range(4,11) for j in range(4,11)];jobs=[]
    for i in range(48):
        for _ in range(1000):
            h=rng.sample(cells,6);k=family(h)
            if k not in old:break
        else:raise RuntimeError('family exhaustion')
        old.add(k);jobs.append(dict(id=f'G{i:02d}',group='TRAIN' if i<24 else 'CAL' if i<32 else 'TEST',stratum='UNFILTERED' if i%2==0 else 'SCREEN',family=k,opening=h))
    write('C06_SOURCE_PLAN.json',dict(identity=identity(),jobs=jobs,excluded_families=n,excluded_boards=len(boards),source_cap=128,screen_cap=32,continuations=[['STANDARD_PW',128],['STANDARD_FULL',256]],arms=list(ARMS),caps=list(CAPS)))
def play(m,h,arm,cap):
    h=list(h);rr=[]
    while True:
        b,p,w=replay(h,'freestyle')
        if w is not None:return h,w,rr
        a=m.search(h,arm,cap)
        rr.append(dict(ply=len(h),player=p,move=a['best'],response=a));h.append(a['best'])
def panel_moves(r):
    ca=[x[0] for x in r['candidates']]
    return list(dict.fromkeys(ca[i] for i in panel(r) if i is not None))
def source():
    p=read('C06_SOURCE_PLAN.json');assert p['identity']==identity();_,boards=historical();co=[]
    n=D/'C06_SOURCE_RAW.jsonl.gz';assert not n.exists()
    with Server() as s,gzip.open(n,'wt',compresslevel=3) as f:
        m=Meter(s)
        for j in p['jobs']:
            h,w,rr=play(m,j['opening'],'STANDARD_PW',128);elig=[]
            for row in rr:
                a=row['response'];b,p0,_=replay(a['history'],'freestyle');k=orbit_board(b,p0,'freestyle')
                if not a['proof']['own'] and not a['proof']['opponent'] and k not in boards:elig.append((a,k))
            elig=sorted(elig,key=lambda x:(abs(len(x[0]['history'])-14),len(x[0]['history'])))[:2 if j['stratum']=='SCREEN' else 1]
            attempts=[];z=None
            for a,k in elig:
                hist=a['history'];pre=m.search(hist,'STANDARD_PW',8);moves=panel_moves(pre['result'])
                trials=[]
                if j['stratum']=='SCREEN':
                    for move in moves:
                        h1,w1,rr1=play(m,hist+[move],'STANDARD_PW',32)
                        trials.append(dict(move=move,final_history=h1,winner_black=w1,utility=(1+a['player']*w1)/2,rows=rr1))
                contrast=len({x['utility'] for x in trials})>1
                accept=j['stratum']=='UNFILTERED' or contrast
                attempts.append(dict(history=hist,prefix=pre,panel=moves,screen=trials,contrast=contrast,accepted=accept))
                if accept:
                    z=dict(id=j['id'],group=j['group'],stratum=j['stratum'],family=j['family'],history=hist,player=a['player'],panel=moves)
                    boards.add(k);co.append(z);break
            emit(f,dict(job=j,final_history=h,winner_black=w,rows=rr,attempts=attempts,selected=z))
            print('SOURCE',j['id'],j['group'],j['stratum'],len(h),'selected',None if z is None else len(z['history']),flush=True)
        ho=m.end()
    write('C06_COHORT.json',dict(identity=identity(),roots=co,missing=[j['id'] for j in p['jobs'] if j['id'] not in {z['id'] for z in co}],source_sha256=sha(n),host=ho))
def atlas():
    co=read('C06_COHORT.json')['roots'];n=D/'C07_ATLAS_RAW.jsonl.gz';assert not n.exists()
    with Server() as s,gzip.open(n,'wt',compresslevel=3) as f:
        m=Meter(s);pairs=list(product(CAPS,ARMS))
        for i,r in enumerate(co):
            aa={};off=i%len(pairs)
            for cap,arm in pairs[off:]+pairs[:off]:
                a=m.search(r['history'],arm,cap);assert a['exposure']
                emit(f,dict(root=r,cap=cap,arm=arm,response=a));aa[cap,arm]=a
            pre=aa[32,'K15N']['exposure']['prefix']['result'];assert panel_moves(pre)==r['panel']
            for (cap,arm),a in aa.items():
                assert a['exposure']['prefix']['result']==pre and a['root_eval']==aa[32,'K15N']['root_eval']
                if arm=='K15LOCK':assert a['best']==aa[cap,'K15N']['best']
            print('ATLAS',r['id'],'raw',len({a['best'] for a in aa.values()}),'sub',len({a['exposure']['sub_best'] for a in aa.values()}),flush=True)
        ho=m.end()
    write('C07_ATLAS_LOCK.json',dict(identity=identity(),requests=len(co)*len(pairs),raw_sha256=sha(n),host=ho))
def atlas_rows():
    assert sha(D/'C07_ATLAS_RAW.jsonl.gz')==read('C07_ATLAS_LOCK.json')['raw_sha256']
    return list(rows('C07_ATLAS_RAW.jsonl.gz'))
def panels():
    d={}
    for x in atlas_rows():
        r=x['root'];a=x['response'];i=r['id']
        if i not in d:d[i]=dict(root=r,moves=set(r['panel']))
        d[i]['moves'].update([a['best'],a['exposure']['sub_best']])
    return {i:dict(root=z['root'],moves=sorted(z['moves'])) for i,z in d.items()}
def labels(group):
    panel0=read('C07_PANEL.json');assert panel0['atlas_sha256']==sha(D/'C07_ATLAS_RAW.jsonl.gz')
    n=D/f'C08_{group}_LABELS.jsonl.gz';assert not n.exists()
    with Server() as s,gzip.open(n,'wt',compresslevel=3) as f:
        m=Meter(s);count=0
        for i,z in sorted(panel0['panels'].items()):
            r=z['root']
            if (r['group']=='TRAIN')!=(group=='TRAIN'):continue
            for a in z['moves']:
                for arm,cap in [('STANDARD_PW',128),('STANDARD_FULL',256)]:
                    h,w,rr=play(m,r['history']+[a],arm,cap)
                    emit(f,dict(root=r,action=a,arm=arm,cap=cap,final_history=h,winner_black=w,utility=(1+r['player']*w)/2,rows=rr));count+=1
            print('LABEL',group,i,len(z['moves']),flush=True)
        ho=m.end()
    write(f'C08_{group}_LOCK.json',dict(identity=identity(),count=count,raw_sha256=sha(n),host=ho))
def cpu(a):return (a['measure']['rust_cpu_ns']+a['measure']['rapfi_cpu_ns'])/1e6
def label_index(part):
    name='C08_'+part+'_LABELS.jsonl.gz'
    assert sha(D/name)==read('C08_'+part+'_LOCK.json')['raw_sha256']
    z=list(rows(name));u={(g['root']['id'],g['action'],g['arm']):g['utility'] for g in z};assert len(z)==len(u)
    return u
def U(u,i,a):return S.mean(u[i,a,k] for k in ('STANDARD_PW','STANDARD_FULL'))
def expo(aa,u,ids):
    x={}
    for i in ids:
        raw=[];unforced=[]
        for cap in CAPS:
            b=U(u,i,aa[i,cap,'K15N']['best'])
            for arm in ARMS[1:]:
                a=aa[i,cap,arm];raw.append(U(u,i,a['best'])-b)
                # STOP and LOCK included in unforced set, as fixed available options.
                unforced.append(U(u,i,a['exposure']['sub_best'])-b)
        x[i]=dict(exposed=any(v!=0 for v in raw),positive=any(v>0 for v in raw),negative=any(v<0 for v in raw),sub_exposed=any(v!=0 for v in unforced))
    counts={k:sum(v[k] for v in x.values()) for k in ('exposed','positive','negative','sub_exposed')}
    return dict(n=len(ids),counts=counts,details=x,pass_gate=counts['exposed']>=6 and counts['positive']>=3 and counts['negative']>=3 and counts['sub_exposed']>=3)
def fit():
    import numpy as np
    aa={(x['root']['id'],x['cap'],x['arm']):x['response'] for x in atlas_rows() if x['root']['group']=='TRAIN'}
    u=label_index('TRAIN');ids=sorted({i for i,c,a in aa})
    gate=expo(aa,u,ids);models={}
    if gate['pass_gate']:
        for cap in CAPS:
            g={}
            for kind,name in enumerate(('RAW','GIBBS','OCCUPATION')):
                x=np.array([aa[i,cap,'K15N']['exposure']['prefix']['features'][kind] for i in ids]);p=x.T@x+4*np.eye(x.shape[1])
                m={}
                for arm in ARMS:
                    dg=np.array([U(u,i,aa[i,cap,arm]['best'])-U(u,i,aa[i,cap,'K15N']['best']) for i in ids])
                    dc=np.array([cpu(aa[i,cap,arm])-cpu(aa[i,cap,'K15N']) for i in ids])
                    m[arm]=dict(gain=np.linalg.solve(p,x.T@dg).tolist(),cost=np.linalg.solve(p,x.T@dc).tolist(),mean_gain=float(np.mean(dg)),mean_cost=float(np.mean(dc)))
                g[name]=m
            models[str(cap)]=g
    write('C08_FROZEN.json',dict(identity=identity(),gate=gate,models=models,ridge=4,price=.002,train_families=ids,train_labels_sha256=sha(D/'C08_TRAIN_LABELS.jsonl.gz'),atlas_sha256=sha(D/'C07_ATLAS_RAW.jsonl.gz'),model_status='FITTED_OFFLINE_ONLY' if models else 'NOT_FIT_EXPOSURE_GATE'))
    print('GATE',gate['counts'],gate['pass_gate'],flush=True)
if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='panel':write('C07_PANEL.json',dict(atlas_sha256=sha(D/'C07_ATLAS_RAW.jsonl.gz'),panels=panels()))
    elif cmd=='labels':labels(sys.argv[2])
    else:{'smoke':smoke,'plan':plan,'source':source,'atlas':atlas,'fit':fit}[cmd]()
