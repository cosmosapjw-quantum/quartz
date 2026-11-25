"""Prospective complete factorial response atlas, training and live two-pulse control."""
from arena import Server,identity,check,write,read,D,ROOT,BIN,sha,replay,proof_reference,orbit_board,fixtures
import kinetic_ref as kr
import collections as C,gzip,json,hashlib,random,statistics as S,math,time,sys,os,importlib.util,ctypes
from pathlib import Path
from itertools import product
ARMS=('KRAW','KGIBBS','KFLUX')
PAIRS=list(product(range(5),repeat=2))
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
    def search(self,h,arm,cap=96,schedule=None,model=None):
        t=[time.clock_gettime_ns(i) for i in self.ids];p=time.process_time_ns()
        a=self.s.search(h,arm,cap,17,schedule=schedule,model=model)
        py=time.process_time_ns()-p;d=[time.clock_gettime_ns(i)-v for i,v in zip(self.ids,t)]
        assert min(d)>=0
        a['measure']=dict(rust_cpu_ns=d[0],rapfi_cpu_ns=d[1],python_cpu_ns=py,clock_res_s=self.res)
        self.peak=max(self.peak,os.getloadavg()[0])
        if arm=='KINETIC' or arm in ARMS:kr.verify(a,schedule,model)
        return a
    def end(self):
        e=host();dt=e['mono']-self.start['mono']
        return dict(start=self.start,end=e,seconds=dt,swap_pages_s=sum(e['swap'][k]-self.start['swap'][k] for k in e['swap'])/dt,max_load=self.peak)
def family(h):
    b,p,w=replay(h[:6],'freestyle');assert w is None
    return hashlib.sha256(orbit_board(b,p,'freestyle')).hexdigest()
def historical():
    p=ROOT/'docs/research/quartz_metacog13_20261126/experiment.py'
    sp=importlib.util.spec_from_file_location('prior_meta_sources',p);old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
    fam,boards=old.historical()
    for n in ('C06_SOURCE_GAMES.jsonl.gz','C08_CONT_RAW.jsonl.gz'):
        with gzip.open(p.parent/n,'rt') as f:
            for line in f:
                g=json.loads(line)
                if 'job' in g:fam.add(family(g['job']['opening']))
                for h in [r['response']['history'] for r in g['rows']]+[g['final_history']]+([g['root']['history']] if 'root' in g else []):
                    b,p0,w=replay(h,'freestyle');boards.add(orbit_board(b,p0,'freestyle'))
    return fam,boards
def zero_model(kind):
    return dict(kind=kind,**{k:[[0.]*21 for i in range(5)] for k in ('first_gain','first_cost','second_gain','second_cost')})
def smoke():
    f=fixtures();out=[];reject=[]
    with Server() as s:
        m=Meter(s)
        for nm in ('trans_a','long','immediate'):
            h=f[nm];aa=[m.search(h,'KINETIC',schedule=[0,0]),m.search(h,'KINETIC',schedule=[1,3]),m.search(h,'KINETIC',schedule=[3,1])]
            mod=zero_model('FLUX');mod['first_gain'][1][0]=1;mod['second_gain'][3][0]=1
            z=m.search(h,'KFLUX',model=mod);aa.append(z)
            if aa[0]['kinetic']:
                assert all(a['kinetic']['prefix']['result']==aa[0]['kinetic']['prefix']['result'] and a['root_eval']==aa[0]['root_eval'] for a in aa)
                assert aa[1]['result']==z['result'] and aa[1]['kinetic']['b_trace']==z['kinetic']['b_trace']
                std=m.search(h,'STANDARD_PW',96);assert std['result']==aa[0]['result']
            else:assert all(a['cause']=='GUARDED' and a['best']==aa[0]['best'] for a in aa);std=None
            out.append(dict(fixture=nm,rows=aa,standard=std,model=mod))
            print('SMOKE',nm,[(a['arm'],a['best'],a['visits']) for a in aa],flush=True)
        for q in [dict(arm='KINETIC',visits=96,schedule=[0,5]),dict(arm='KRAW',visits=96,model=zero_model('FLUX')),dict(arm='KINETIC',visits=128,schedule=[0,0]),dict(arm='KRAW',visits=96,model=dict(zero_model('RAW'),first_gain=[[0.]]))]:
            z=s.request(dict(cmd='research_rapfi_shared_search',rule='freestyle',history=[],seed=17,**q));assert 'error' in z;reject.append(z)
        ho=m.end()
    write('C06_SMOKE.json',dict(identity=identity(),rows=out,rejected=reject,host=ho))
def plan():
    old,boards=historical();n=len(old);rng=random.Random(2026112614);cells=[15*i+j for i in range(4,11) for j in range(4,11)];jobs=[]
    for i in range(24):
        for _ in range(1000):
            h=rng.sample(cells,6);k=family(h)
            if k not in old:break
        else:raise RuntimeError('family exhaustion')
        old.add(k);jobs.append(dict(id=f'G{i:02d}',group='TRAIN' if i<12 else 'CAL' if i<16 else 'TEST',stratum='UNFILTERED' if i%2==0 else 'SENSITIVE',family=k,opening=h))
    write('C06_SOURCE_PLAN.json',dict(identity=identity(),jobs=jobs,excluded_families=n,excluded_boards=len(boards),source_cap=128,continuation_caps=[128,512]))
def play(m,h,cap):
    h=list(h);rr=[]
    while True:
        b,p,w=replay(h,'freestyle')
        if w is not None:return h,w,rr
        a=m.search(h,'STANDARD_PW',cap)
        rr.append(dict(ply=len(h),player=p,move=a['best'],response=a));h.append(a['best'])
def distance(r):
    n=kr.counts(r);L=max(range(len(n)),key=lambda i:(n[i],i));assert r['best']==r['candidates'][L][0]
    return min((n[L]-v+int(i<L) for i,v in enumerate(n) if i!=L),default=10**9)
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
            elig=sorted(elig,key=lambda x:(abs(len(x[0]['history'])-18),len(x[0]['history'])))[:3]
            probes=[];z=None
            for a,k in elig:
                p0=m.search(a['history'],'STANDARD_PW',16);d=distance(p0['result'])
                probes.append(dict(history=a['history'],distance=d,response=p0))
                if j['stratum']=='UNFILTERED' or d<=4:
                    z=dict(id=j['id'],family=j['family'],group=j['group'],stratum=j['stratum'],history=a['history'],player=a['player'],distance16=d);boards.add(k);co.append(z);break
            emit(f,dict(job=j,final_history=h,winner_black=w,rows=rr,selected=z,probes=probes))
            print('SOURCE',j['id'],j['group'],j['stratum'],len(h),'selected',None if z is None else z['distance16'],flush=True)
        ho=m.end()
    write('C06_COHORT.json',dict(identity=identity(),roots=co,missing=[j['id'] for j in p['jobs'] if j['id'] not in {z['id'] for z in co}],source_sha256=sha(n),host=ho))
def atlas():
    co=read('C06_COHORT.json')['roots'];n=D/'C07_ATLAS_RAW.jsonl.gz';assert not n.exists()
    with Server() as s,gzip.open(n,'wt',compresslevel=3) as f:
        m=Meter(s)
        for i,r in enumerate(co):
            aa={};off=i%25
            for c,d in PAIRS[off:]+PAIRS[:off]:
                a=m.search(r['history'],'KINETIC',schedule=[c,d]);assert a['kinetic']
                emit(f,dict(root=r,schedule=[c,d],response=a));aa[c,d]=a
            z=aa[0,0];pre=z['kinetic']['prefix']['result']
            assert distance(pre)==r['distance16']
            for (c,d),a in aa.items():
                assert a['kinetic']['prefix']['result']==pre and a['root_eval']==z['root_eval']
                assert a['kinetic']['steps'][0]['result']==aa[c,0]['kinetic']['steps'][0]['result']
            print('ATLAS',r['id'],len({a['best'] for a in aa.values()}),'final moves',flush=True)
        ho=m.end()
    write('C07_ATLAS_LOCK.json',dict(identity=identity(),requests=len(co)*25,raw_sha256=sha(n),host=ho))
def atlas_rows():
    assert sha(D/'C07_ATLAS_RAW.jsonl.gz')==read('C07_ATLAS_LOCK.json')['raw_sha256']
    return list(rows('C07_ATLAS_RAW.jsonl.gz'))
def panels():
    d={}
    for x in atlas_rows():
        r=x['root'];a=x['response'];i=r['id'];pre=a['kinetic']['prefix']['result']
        if i not in d:
            ca=[z[0] for z in pre['candidates']];n=kr.counts(pre)
            L=pre['best'];ch=max((k for k,v in enumerate(n) if ca[k]!=L and v>0),key=lambda k:(n[k],k),default=None)
            alt=ca[ch] if ch is not None else next((m for m in ca if m!=L),None)
            hidden=ca[len(pre['edges'])] if len(pre['edges'])<len(ca) else None
            d[i]=dict(root=r,independent=[v for v in (alt,hidden) if v is not None],moves={L,*[v for v in (alt,hidden) if v is not None]})
        d[i]['moves'].update([a['kinetic']['steps'][0]['result']['best'],a['best']])
    return {i:dict(root=x['root'],independent=x['independent'],moves=sorted(x['moves'])) for i,x in d.items()}
def labels(group):
    panel=read('C07_PANEL.json');assert panel['atlas_sha256']==sha(D/'C07_ATLAS_RAW.jsonl.gz')
    n=D/f'C08_{group}_LABELS.jsonl.gz';assert not n.exists()
    with Server() as s,gzip.open(n,'wt',compresslevel=3) as f:
        m=Meter(s);count=0
        for i,z in sorted(panel['panels'].items()):
            r=z['root']
            if (r['group']=='TRAIN')!=(group=='TRAIN'):continue
            for a in z['moves']:
                for cap in (128,512):
                    h,w,rr=play(m,r['history']+[a],cap)
                    emit(f,dict(root=r,action=a,cap=cap,final_history=h,winner_black=w,utility=(1+r['player']*w)/2,rows=rr));count+=1
            print('LABEL',group,i,len(z['moves']),flush=True)
        ho=m.end()
    write(f'C08_{group}_LOCK.json',dict(identity=identity(),count=count,raw_sha256=sha(n),host=ho))
def cpu(a):return (a['measure']['rust_cpu_ns']+a['measure']['rapfi_cpu_ns'])/1e6
def fit():
    import numpy as np
    aa={(x['root']['id'],*x['schedule']):x['response'] for x in atlas_rows() if x['root']['group']=='TRAIN'}
    labels0=list(rows('C08_TRAIN_LABELS.jsonl.gz'));u={(x['root']['id'],x['action'],x['cap']):x['utility'] for x in labels0}
    ids=sorted({k[0] for k in aa});assert len(ids)>=2
    def U(i,c,d):return S.mean(u[i,aa[i,c,d]['best'],k] for k in (128,512))
    def C0(i,c,d):return cpu(aa[i,c,d])
    out={};counts0={}
    for kind,name in enumerate(('RAW','GIBBS','FLUX')):
        model=zero_model(name);diagnostics={}
        for stage in ('first','second'):
            for c in range(1,5):
                xx=[];yg=[];yc=[];families=[]
                for i in ids:
                    if stage=='first':
                        xx.append(aa[i,0,0]['kinetic']['prefix']['features'][kind])
                        yg.append(S.mean(U(i,c,d)-U(i,0,d) for d in range(5)))
                        yc.append(S.mean(C0(i,c,d)-C0(i,0,d) for d in range(5)));families.append(i)
                    else:
                        for d in range(5):
                            xx.append(aa[i,d,0]['kinetic']['steps'][0]['features'][kind])
                            yg.append(U(i,d,c)-U(i,d,0));yc.append(C0(i,d,c)-C0(i,d,0));families.append(i)
                X=np.array(xx);g=np.array(yg);C1=np.array(yc);P=X.T@X+4*np.eye(21)
                wg=np.linalg.solve(P,X.T@g);wc=np.linalg.solve(P,X.T@C1)
                model[stage+'_gain'][c]=wg.tolist();model[stage+'_cost'][c]=wc.tolist()
                diagnostics[stage+str(c)]=dict(rows=len(g),families=len(set(families)),nonzero_families=len({f for f,y in zip(families,yg) if y!=0}),mean_gain=float(np.mean(g)),mean_cost_ms=float(np.mean(C1)),fit_MSE=float(np.mean((X@wg-g)**2)))
        out['K'+name]=model;counts0[name]=diagnostics
    write('C08_MODELS.json',dict(identity=identity(),models=out,train_families=ids,train_labels_sha256=sha(D/'C08_TRAIN_LABELS.jsonl.gz'),atlas_sha256=sha(D/'C07_ATLAS_RAW.jsonl.gz'),ridge=4,price=.002,diagnostics=counts0))
    print('FIT',len(ids),'families',len(labels0),'terminal labels',flush=True)
def live():
    mod=read('C08_MODELS.json');assert mod['atlas_sha256']==sha(D/'C07_ATLAS_RAW.jsonl.gz')
    assert mod['train_labels_sha256']==sha(D/'C08_TRAIN_LABELS.jsonl.gz')
    aa={(x['root']['id'],*x['schedule']):x['response'] for x in atlas_rows()}
    co=[r for r in read('C06_COHORT.json')['roots'] if r['group']!='TRAIN']
    n=D/'C08_LIVE_RAW.jsonl.gz';assert not n.exists()
    with Server() as s,gzip.open(n,'wt',compresslevel=3) as f:
        m=Meter(s)
        for i,r in enumerate(co):
            for arm in ARMS[i%3:]+ARMS[:i%3]:
                a=m.search(r['history'],arm,model=mod['models'][arm]);ch=[x['channel'] for x in a['kinetic']['steps'][:2]]
                b=aa[r['id'],*ch];assert a['result']==b['result'] and a['kinetic']['b_trace']==b['kinetic']['b_trace']
                assert a['kinetic']['prefix']['result']==b['kinetic']['prefix']['result']
                for x,y in zip(a['kinetic']['steps'],b['kinetic']['steps'],strict=True):assert x['result']==y['result'] and x['features']==y['features']
                emit(f,dict(root=r,schedule=ch,response=a))
                print('LIVE',r['id'],arm,ch,a['best'],flush=True)
        ho=m.end()
    write('C08_LIVE_LOCK.json',dict(identity=identity(),requests=len(co)*3,raw_sha256=sha(n),models_sha256=sha(D/'C08_MODELS.json'),host=ho))
if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='panel':write('C07_PANEL.json',dict(atlas_sha256=sha(D/'C07_ATLAS_RAW.jsonl.gz'),panels=panels()))
    elif cmd=='labels':labels(sys.argv[2])
    else:{'smoke':smoke,'plan':plan,'source':source,'atlas':atlas,'fit':fit,'live':live}[cmd]()
