"""Prospective two-domain paired evidence experiment; exact C4 oracle offline."""
from arena import Server,identity,check,write,read,D,ROOT,BIN,sha,replay,proof_reference,orbit_board,fixtures
from pair_ref import ARMS,CAPS,verify,pair,moments
import collections as C,gzip,json,hashlib,random,statistics as S,math,time,sys,os,importlib.util,ctypes,subprocess as SP
from pathlib import Path
from itertools import product
MODEL=ROOT/'docs/research/cognitive_metareasoning_20261120/connect4_generalization_07/evidence/mlp_conv_train_20261121/conv_model.json'
MID='161f9e6e26840754384cb47335d1082d837fa014735cbfaa2517209422e29f24'
EXT=ROOT.parent/'quartz_external_engines_20261120/pascalpons_connect4'
ORACLE=EXT/'c4solver';BOOK=EXT/'7x6_small.book'
def ident():
    a=identity();assert sha(MODEL)==MID
    assert sha(ORACLE)=='fca2b665b1b3dbd88f2f50a7e0ba86fd8ae253f2861da1ad51cf110c69f741ed'
    assert sha(BOOK)=='38f9834317c37d9516e45a21da598569a5d1556595686593d14c2e63f59c1f38'
    return dict(a,c4_model=MID,c4_oracle=sha(ORACLE),c4_book=sha(BOOK))
def rows(name):
    with gzip.open(D/name,'rt') as f:
        for line in f:yield json.loads(line)
def emit(f,x):
    f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n');f.flush()
def host():
    z={}
    for l in open('/proc/vmstat'):
        k,v=l.split()
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
    def search(self,h,arm,cap,domain='GOMOKU'):
        t=[time.clock_gettime_ns(i) for i in self.ids];p=time.process_time_ns()
        if domain=='GOMOKU':a=self.s.search(h,arm,cap,17)
        else:
            a=self.s.request(dict(cmd='research_connect4_evidence_cpu',history=h,arm=arm,cap=cap,seed=17,model_path=str(MODEL),model_identity=MID))
            assert 'error' not in a,a
            assert a['schema']=='quartz_k16_c4_v1' and a['history']==h and a['arm']==arm and a['cap']==cap and a['seed']==17 and a['model_identity']==MID
            b,z=c4replay(h);assert a['legal']==[i for i in range(7) if z[i]<6] and not any(won(x) for x in b)
        py=time.process_time_ns()-p;d=[time.clock_gettime_ns(i)-v for i,v in zip(self.ids,t)]
        assert min(d)>=0
        a['measure']=dict(rust_cpu_ns=d[0],rapfi_cpu_ns=d[1],python_cpu_ns=py,clock_res_s=self.res)
        self.peak=max(self.peak,os.getloadavg()[0])
        if arm in ARMS:verify(a)
        return a
    def end(self):
        e=host();dt=e['mono']-self.start['mono']
        return dict(start=self.start,end=e,seconds=dt,swap_pages_s=sum(e['swap'][k]-self.start['swap'][k] for k in e['swap'])/dt,max_load=self.peak)
def won(x):return any((m:=x&(x>>d))&(m>>(2*d)) for d in (1,7,6,8))
def c4replay(h):
    b=[0,0];z=[0]*7
    for i,c in enumerate(h):
        k=int(c)-1;assert 0<=k<7 and z[k]<6 and not any(won(x) for x in b)
        b[i%2]|=1<<(7*k+z[k]);z[k]+=1
    return b,z
def c4flat(b):return [1 if b[0]>>(7*x+y)&1 else -1 if b[1]>>(7*x+y)&1 else 0 for y in range(6) for x in range(7)]
def mirror(h):return min(h,h.translate(str.maketrans('1234567','7654321')))
def boardkey(b):return min(tuple(b),tuple(b[y*7+x] for y in range(6) for x in range(6,-1,-1)))
def canwin(b,z,p):return any(won(b[p]|(1<<(7*c+z[c]))) for c in range(7) if z[c]<6)
def gitbytes(ref,path):return SP.check_output(['git','show',ref+':'+path],cwd=ROOT)
def c4historical():
    old='9b8a4718f7f92d83db2fb00e5806c6c4bd8cbfc2';d0='docs/research/quartz_fouraxis05_20261125/'
    manifest=json.loads(gitbytes(old,d0+'C07_COHORT.json'));src=manifest['sources']
    refs={'FA01':('1f3034594dfda138a4dc3fbd0e0a1e5f28838f45','quartz_fouraxis_20261125'),'FA02':('c19ad67ea5799f0ce89aee10e6e3af0a8f311201','quartz_fouraxis02_20261125'),'FA03':('a900c4e23a1d6b0f0fa2dac5f0c885b932683f15','quartz_fouraxis03_20261125'),'FA04':('dfefd8bc0bb3cccf783263356c2ebbd5300d292d','quartz_fouraxis04_20261125'),'NN02':('543e12dd3a759ebef1e7163a80a1f57c45c653d5','quartz_nn_intervention_20261124'),'NN03':('3d22dadcade29f8cb96b44e7ef40571c4d5c1db3','quartz_nn_multiquanta_20261124'),'NN04':('d302bd0008c21a7f7bfd848a0d964d3f94662d78','quartz_nn_costcal_20261125')}
    p=set();boards=set();locks={}
    for name,info in src.items():
        lane,file=name.split('/',1)
        if lane in refs:
            h,dir=refs[lane];path='docs/research/'+dir+'/'+file
        else:h=old;path='docs/research/cognitive_metareasoning_20261120/'+name
        raw=gitbytes(h,path);assert hashlib.sha256(raw).hexdigest()==info['sha256'],name
        locks[name]=dict(commit=h,path=path,sha256=info['sha256'])
        if name.endswith('.gz'):
            for l in gzip.decompress(raw).splitlines():
                a=json.loads(l);p.add(mirror(a['opening']))
        else:
            a=json.loads(raw);rr=a['queries'] if name.endswith('C07_B_QUERIES.json') else (a.get('rows') or a.get('connect4'))
            for r in rr:
                hist=r.get('history',r.get('sequence'));b=r.get('board')
                if b is None:b=c4flat(c4replay(hist)[0])
                boards.add(boardkey(b))
                if lane not in refs:p.add(mirror(hist[:5]))
    raw=gitbytes(old,d0+'C07_ATTEMPTS.jsonl.gz');assert hashlib.sha256(raw).hexdigest()==manifest['attempt_sha']
    for l in gzip.decompress(raw).splitlines():p.add(mirror(json.loads(l)['opening']))
    locks['FA05/attempts']=dict(commit=old,path=d0+'C07_ATTEMPTS.jsonl.gz',sha256=manifest['attempt_sha'])
    for r in manifest['rows']:boards.add(boardkey(r['board']))
    return p,boards,locks
def family(h):
    b,p,w=replay(h[:6],'freestyle');assert w is None
    return hashlib.sha256(orbit_board(b,p,'freestyle')).hexdigest()
def ghistorical():
    p=ROOT/'docs/research/quartz_kinetic15_20261126/response_run.py'
    # This old module depends on its own response_ref; import under old path only for historical.
    sys.path.insert(0,str(p.parent))
    try:
        sp=importlib.util.spec_from_file_location('prior_k15_sources',p);old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
        fam,boards=old.historical()
    finally:sys.path.remove(str(p.parent))
    for name in ('C06_SOURCE_RAW.jsonl.gz','C08_TRAIN_LABELS.jsonl.gz','C08_HELDOUT_LABELS.jsonl.gz'):
        with gzip.open(p.parent/name,'rt') as f:
            for line in f:
                g=json.loads(line)
                if 'job' in g:fam.add(family(g['job']['opening']))
                hh=[r['response']['history'] for r in g['rows']]+[g['final_history']]+([g['root']['history']] if 'root' in g else [])
                for a in g.get('attempts',[]):
                    hh.append(a['history'])
                    for t in a['screen']:hh.extend([r['response']['history'] for r in t['rows']]+[t['final_history']])
                for h in hh:
                    b,p0,w=replay(h,'freestyle');boards.add(orbit_board(b,p0,'freestyle'))
    return fam,boards
def smoke():
    f=fixtures();out=[];reject=[]
    with Server() as s:
        m=Meter(s)
        for dom,h in [('C4','4433226223'),('GOMOKU',f['trans_a']),('GOMOKU',f['long']),('GOMOKU',f['immediate'])]:
            for cap in CAPS:
                aa=[m.search(h,a,cap,dom) for a in ARMS]
                if dom=='C4':
                    std=s.request(dict(cmd='research_connect4_cpu',history=h,eval_mode='conv3x3',model_path=str(MODEL),model_identity=MID,cache_model=True,seed=17,force_stop_visits=cap))
                    assert 'error' not in std and std['steps'][-1]['best_action']==aa[0]['best'],std
                else:
                    std=m.search(h,'STANDARD_PW',cap,dom)
                    assert std['result']==aa[0]['result']
                if aa[0]['evidence']:assert all(a['evidence']['prefix']==aa[0]['evidence']['prefix'] for a in aa)
                out.append(dict(domain=dom,history=h,cap=cap,rows=aa,standard=std))
                print('SMOKE',dom,cap,[(a['arm'],a['best'],None if not a['evidence'] else a['evidence']['pulse_best']) for a in aa],flush=True)
        for q in [dict(cmd='research_connect4_evidence_cpu',history='4433226223',arm='K16N',cap=48,seed=17,model_path=str(MODEL),model_identity=MID),dict(cmd='research_connect4_evidence_cpu',history='4433226223',arm='K16N',cap=32,seed=17,model_path=str(MODEL),model_identity=MID,oracle=[1]),dict(cmd='research_rapfi_shared_search',rule='freestyle',history=[],arm='K16BAL',visits=48,seed=17)]:
            a=s.request(q);assert 'error' in a;reject.append(a)
    write('C06_SMOKE.json',dict(identity=ident(),rows=out,rejected=reject))
def plan():
    p,b,locks=c4historical();gp,gb=ghistorical();rng=random.Random(2026112616);jobs=[];n0=len(p)
    # Freeze attempts without labels; each opening chosen from unused families.
    for grp,count in [('TRAIN',216),('CAL',72),('TEST',144)]:
        for i in range(count):
            for _ in range(10000):
                h=''.join(str(rng.randrange(1,8)) for _ in range(5));k=mirror(h)
                if k not in p:break
            else:raise RuntimeError('family exhaustion')
            p.add(k);jobs.append(dict(id=f'C{len(jobs):03d}',group=grp,domain='C4',family=k,opening=h,seed=rng.randrange(2**63)))
    gcells=[15*i+j for i in range(4,11) for j in range(4,11)];gj=[]
    for i in range(24):
        for _ in range(1000):
            h=rng.sample(gcells,6);k=family(h)
            if k not in gp:break
        else:raise RuntimeError('gomoku family exhaustion')
        gp.add(k);gj.append(dict(id=f'G{i:02d}',group='TRAIN' if i<12 else 'CAL' if i<16 else 'TEST',domain='GOMOKU',family=k,opening=h))
    write('C06_PLAN.json',dict(identity=ident(),c4=jobs,gomoku=gj,c4_old_families=n0,c4_old_boards=len(b),c4_sources=locks,g_old_boards=len(gb)))
def play(m,h,arm,cap):
    h=list(h);rr=[]
    while True:
        b,p,w=replay(h,'freestyle')
        if w is not None:return h,w,rr
        a=m.search(h,arm,cap);rr.append(dict(ply=len(h),player=p,move=a['best'],response=a));h.append(a['best'])
def source():
    plan0=read('C06_PLAN.json');assert plan0['identity']==ident()
    _,boards,locks=c4historical();assert locks==plan0['c4_sources']
    co=[];cnt=C.Counter();targets={'TRAIN':18,'CAL':6,'TEST':12}
    attempts=[]
    for j in plan0['c4']:
        if cnt[j['group']]>=targets[j['group']]:continue
        rng=random.Random(j['seed']);h=j['opening'];b,z=c4replay(h)
        while len(h)<28:
            p=len(h)%2;opts=[c for c in range(7) if z[c]<6 and not won(b[p]|(1<<(7*c+z[c])))]
            if not opts:break
            c=rng.choice(opts);b[p]|=1<<(7*c+z[c]);z[c]+=1;h+=str(c+1)
        k=boardkey(c4flat(b));leg=[i for i in range(7) if z[i]<6]
        ok=len(h)==28 and len(leg)>=2 and not any(canwin(b,z,p) for p in (0,1)) and k not in boards
        attempts.append(dict(job=j,history=h,accepted=ok))
        if ok:
            boards.add(k);cnt[j['group']]+=1;co.append(dict(id=j['id'],domain='C4',group=j['group'],family=j['family'],history=h,legal=leg,player=1))
    print('C4_SOURCE',dict(cnt),'attempts',len(attempts),flush=True)
    _,gb=ghistorical();n=D/'C06_GOMOKU_SOURCE.jsonl.gz';assert not n.exists()
    with Server() as s,gzip.open(n,'wt',compresslevel=3) as f:
        m=Meter(s)
        for j in plan0['gomoku']:
            h,w,rr=play(m,j['opening'],'STANDARD_PW',128);elig=[]
            for row in rr:
                a=row['response'];b,p0,_=replay(a['history'],'freestyle');k=orbit_board(b,p0,'freestyle')
                if not a['proof']['own'] and not a['proof']['opponent'] and len(a['legal'])>=2 and k not in gb:elig.append((a,k))
            sel=None
            if elig:
                a,k=min(elig,key=lambda x:(abs(len(x[0]['history'])-14),len(x[0]['history'])))
                gb.add(k);sel=dict(id=j['id'],domain='GOMOKU',group=j['group'],family=j['family'],history=a['history'],legal=a['legal'],player=a['player']);co.append(sel)
            emit(f,dict(job=j,final_history=h,winner_black=w,rows=rr,selected=sel))
            print('GOMOKU_SOURCE',j['id'],len(h),None if sel is None else len(sel['history']),flush=True)
        ho=m.end()
    write('C06_COHORT.json',dict(identity=ident(),roots=co,c4_attempts=attempts,c4_targets=targets,missing_gomoku=[j['id'] for j in plan0['gomoku'] if j['id'] not in {r['id'] for r in co}],source_sha256=sha(n),host=ho))
def atlas():
    co=read('C06_COHORT.json')['roots'];n=D/'C07_ATLAS.jsonl.gz';assert not n.exists()
    with Server() as s,gzip.open(n,'wt',compresslevel=3) as f:
        m=Meter(s);opts=list(product(CAPS,ARMS))
        for i,r in enumerate(co):
            aa={}
            for cap,arm in opts[i%6:]+opts[:i%6]:
                a=m.search(r['history'],arm,cap,r['domain']);assert a['evidence']
                aa[cap,arm]=a;emit(f,dict(root=r,cap=cap,arm=arm,response=a))
            pre=aa[32,'K16N']['evidence']['prefix']
            assert all(a['evidence']['prefix']==pre for a in aa.values())
            print('ATLAS',r['id'],aa[32,'K16N']['evidence']['pair'],len({a['evidence']['pulse_best'] for a in aa.values()}),flush=True)
        ho=m.end()
    write('C07_ATLAS_LOCK.json',dict(identity=ident(),requests=len(co)*6,sha256=sha(n),host=ho))
def atlas_rows():
    assert sha(D/'C07_ATLAS.jsonl.gz')==read('C07_ATLAS_LOCK.json')['sha256']
    return list(rows('C07_ATLAS.jsonl.gz'))
def panels():
    out={}
    for x in atlas_rows():
        r=x['root'];a=x['response'];e=a['evidence'];i=r['id']
        if i not in out:out[i]=dict(root=r,pair=e['pair'],moves=set(e['pair']))
        assert out[i]['pair']==e['pair'];out[i]['moves'].update([a['best'],e['prefix_best'],e['pulse_best'],e['final_best']])
    return {i:dict(root=z['root'],pair=z['pair'],moves=sorted(z['moves'])) for i,z in out.items()}
def oracle(r):
    h=r['history'];t=time.perf_counter()
    try:
        p=SP.run([str(ORACLE),'-a','-b',str(BOOK)],input=h+'\n',capture_output=True,text=True,timeout=8,cwd=EXT)
        vals=p.stdout.strip().split()
        if p.returncode or len(vals)!=8 or vals[0]!=h:return dict(root=r,status='ERROR',returncode=p.returncode,stdout=p.stdout[-500:])
        v=list(map(int,vals[1:]));assert all((x!=-1000)==(i in r['legal']) and (x==-1000 or -21<=x<=21) for i,x in enumerate(v))
        return dict(root=r,status='EXACT',scores=v,wall_s=time.perf_counter()-t)
    except SP.TimeoutExpired:return dict(root=r,status='TIMEOUT')
def labels(part):
    pan=read('C07_PANEL.json');assert pan['atlas_sha256']==sha(D/'C07_ATLAS.jsonl.gz')
    n=D/f'C08_{part}_LABELS.jsonl.gz';assert not n.exists()
    with Server() as s,gzip.open(n,'wt',compresslevel=3) as f:
        m=Meter(s);count=C.Counter()
        for i,z in sorted(pan['panels'].items()):
            r=z['root']
            if (r['group']=='TRAIN')!=(part=='TRAIN'):continue
            if r['domain']=='C4':
                a=oracle(r);emit(f,a);count[a['status']]+=1
            else:
                for a in z['moves']:
                    for arm,cap in [('STANDARD_PW',128),('STANDARD_FULL',256)]:
                        h,w,rr=play(m,r['history']+[a],arm,cap)
                        emit(f,dict(root=r,action=a,arm=arm,cap=cap,final_history=h,winner_black=w,utility=(1+r['player']*w)/2,rows=rr));count['GOMOKU']+=1
            print('LABEL',part,i,flush=True)
        ho=m.end()
    write(f'C08_{part}_LOCK.json',dict(identity=ident(),counts=dict(count),sha256=sha(n),host=ho))
def label_index(part):
    name=f'C08_{part}_LABELS.jsonl.gz';assert sha(D/name)==read(f'C08_{part}_LOCK.json')['sha256']
    u={};gg=C.defaultdict(dict);missing=[]
    for x in rows(name):
        r=x['root'];i=r['id']
        if r['domain']=='C4':
            if x['status']!='EXACT':missing.append(i);continue
            u[i]={a:(1+(x['scores'][a]>0)-(x['scores'][a]<0))/2 for a in r['legal']}
        else:gg[i][x['action'],x['arm']]=x['utility']
    for i,z in gg.items():u[i]={a:(v+z[a,'STANDARD_FULL'])/2 for (a,k),v in z.items() if k=='STANDARD_PW'}
    return u,missing
def cpu(a):return (a['measure']['rust_cpu_ns']+a['measure']['rapfi_cpu_ns'])/1e6
GRID=list(product(range(5),repeat=2))
def bin0(x):return int(x>0.1)-int(x< -0.1)
def obs(a):
    e=a['evidence'];p=e['prefix_score'];q=e['pulse_score'];return bin0(p[1]-p[0]),bin0(q[1]-q[0])
def norm(x):
    s=math.fsum(x);return [v/s for v in x]
def mix(c,base,k=4):
    return norm([a+k*b for a,b in zip(c,base,strict=True)])
def diff_mean(p):return math.fsum(v*(b-a)/4 for v,(a,b) in zip(p,GRID,strict=True))
def fit():
    u,missing=label_index('TRAIN');raw=atlas_rows();aa={(x['root']['id'],x['cap'],x['arm']):x['response'] for x in raw if x['root']['group']=='TRAIN'}
    co=read('C06_COHORT.json')['roots'];models={};gates={}
    for dom in ('C4','GOMOKU'):
        ids=sorted(r['id'] for r in co if r['domain']==dom and r['group']=='TRAIN' and r['id'] in u)
        pairs={i:aa[i,32,'K16N']['evidence']['pair'] for i in ids}
        ds=[u[i][b]-u[i][a] for i,(a,b) in pairs.items()]
        gate=dict(n=len(ids),nonzero=sum(d!=0 for d in ds),positive=sum(d>0 for d in ds),negative=sum(d<0 for d in ds))
        gate['pass']=gate['nonzero']>=4 and gate['positive']>=1 and gate['negative']>=1;gates[dom]=gate
        if not gate['pass']:continue
        p=norm([1/25+sum((int(4*u[i][a]),int(4*u[i][b]))==c for i,(a,b) in pairs.items()) for c in GRID])
        mm={}
        for cap in CAPS:
            mm[str(cap)]={}
            for arm in ARMS:
                hists=C.defaultdict(lambda:[0]*25);joint=C.defaultdict(lambda:[0]*25)
                for i in ids:
                    a,b=pairs[i];h,k=obs(aa[i,cap,arm]);ix=GRID.index((int(4*u[i][a]),int(4*u[i][b])))
                    hists[h][ix]+=1;joint[h,k][ix]+=1
                ph={str(h):mix(hists[h],p) for h in (-1,0,1)}
                pq={f'{h},{k}':mix(joint[h,k],ph[str(h)]) for h,k in product((-1,0,1),repeat=2)}
                gains=[];costs=[]
                for i in ids:
                    a=aa[i,cap,arm];b=aa[i,cap,'K16N'];e=a['evidence'];pair0=e['pair'];h,k=obs(a)
                    sel=pair0[diff_mean(pq[f'{h},{k}'])>0]
                    gains.append(u[i][sel]-u[i][b['best']]);costs.append(cpu(a)-cpu(b))
                mm[str(cap)][arm]=dict(prefix=ph,response=pq,global_hist=p,train_families=ids,train_gain=S.mean(gains),train_incremental_cpu_ms=S.mean(costs),train_net=S.mean(gains)-.002*S.mean(costs))
        models[dom]=mm
    write('C08_FROZEN.json',dict(identity=ident(),gates=gates,models=models,missing=missing,atlas_sha256=sha(D/'C07_ATLAS.jsonl.gz'),train_sha256=sha(D/'C08_TRAIN_LABELS.jsonl.gz'),price=.002))
    print('GATES',gates,flush=True)
if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='panel':write('C07_PANEL.json',dict(atlas_sha256=sha(D/'C07_ATLAS.jsonl.gz'),panels=panels()))
    elif cmd=='labels':labels(sys.argv[2])
    else:{'smoke':smoke,'plan':plan,'source':source,'atlas':atlas,'fit':fit}[cmd]()
