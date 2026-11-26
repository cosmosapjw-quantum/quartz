"""Path-independent frozen source, exact solvers and native JSON transport."""
import ast, pathlib,sys,json,gzip,hashlib,subprocess as SP,random,statistics as S,collections as C,math,time,os,select,ctypes
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import product
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2];R=ROOT/'docs/research'
BASE='0f1ebd33da6048f8ccea7ccdb3f421c9aad3317a';SEED=2026112628;PRICE=.002
BIN=ROOT.parent/'quartz_kinetic28_target_20261126/mcts_demo'
SOURCE_BIN=ROOT.parent/'quartz_kinetic17_target_20261126/mcts_demo'
MODEL=R/'cognitive_metareasoning_20261120/connect4_generalization_07/evidence/mlp_conv_train_20261121/conv_model.json'
MID='161f9e6e26840754384cb47335d1082d837fa014735cbfaa2517209422e29f24'
EXT=ROOT.parent/'quartz_external_engines_20261120/pascalpons_connect4';ORACLE=EXT/'c4solver';BOOK=EXT/'7x6_small.book'
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def read(n):return json.loads((D/n).read_text())
def save(n,x):
    f=D/n;assert not f.exists(),n;tmp=f.with_name(f.name+'.tmp');assert not tmp.exists()
    tmp.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n');tmp.replace(f)
def raw(p):
    with gzip.open(p,'rt') as f:
        for l in f:yield json.loads(l)
def emit(f,x):f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n')
def lock(p):
    rel=str(p.relative_to(ROOT));b=SP.check_output(['git','-C',str(ROOT),'show',BASE+':'+rel])
    assert hashlib.sha256(b).hexdigest()==sha(p);return dict(commit=BASE,path=rel,sha256=sha(p))
def extract(rel,names):
    p=ROOT/rel;lk=lock(p);t=ast.parse(p.read_text());nodes=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert set(n.name for n in nodes)==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),globals())
    return lk
HELPERS=[extract('docs/research/quartz_kinetic16_20261126/pair_run.py',['won','c4replay','c4flat','mirror','boardkey','canwin','gitbytes','c4historical','oracle','process_clock'])]
# The frozen independent row-major solver uses these exact constants.
lines=[sum(1<<(7*(y+k*dy)+x+k*dx) for k in range(4)) for y in range(6) for x in range(7) for dx,dy in ((1,0),(0,1),(1,1),(-1,1)) if 0<=x+3*dx<7 and 0<=y+3*dy<6]
assert len(lines)==69 and len(set(lines))==69
through=[[m for m in lines if m>>i&1] for i in range(42)]
HELPERS.append(extract('docs/research/quartz_kinetic16_20261126/audit.py',['independent_c4','exact_wdl']))
def assets():
    a=dict(source_binary=sha(SOURCE_BIN),model=sha(MODEL),oracle=sha(ORACLE),book=sha(BOOK),helpers=HELPERS)
    assert a['source_binary']=='c238b84ca99695e8415974d122b17d576f6b8a30d135d8a5ff92fa88a18bf64b'
    assert a['model']==MID and a['oracle']=='fca2b665b1b3dbd88f2f50a7e0ba86fd8ae253f2861da1ad51cf110c69f741ed'
    assert a['book']=='38f9834317c37d9516e45a21da598569a5d1556595686593d14c2e63f59c1f38'
    return a
def identity():
    b=read('RUST_BUILD_IDENTITY.json');assert sha(BIN)==b['binary_sha256']
    return dict(binary=sha(BIN),source=b['source_head'],assets=assets())
def historical():
    _,boards,prior=c4historical();families=set();locks=[]
    for rel in ('quartz_kinetic16_20261126/C06_COHORT.json','quartz_kinetic17_20261126/COHORT.json','quartz_kinetic17_20261126/quiet/COHORT.json','quartz_kinetic18_20261126/COHORT.json','quartz_kinetic19_20261126/COHORT.json'):
        p=R/rel;locks.append(lock(p));co=json.loads(p.read_text())
        for r in co['roots']:
            if r.get('domain','C4')!='C4':continue
            h=r['history'];boards.add(boardkey(c4flat(c4replay(h)[0])))
            if len(h)>=8:families.add(mirror(h[:8]))
        for j in co.get('jobs',co.get('c4_attempts',[])):
            h=j['history']
            if len(h)>=8:families.add(mirror(h[:8]))
    for i in range(20,28):
        k=R/f'quartz_kinetic{i}_20261126'
        for n in ('PLAN.json','COHORT.json','SOURCE.jsonl.gz'):locks.append(lock(k/n))
        co=json.loads((k/'COHORT.json').read_text())
        for r in co['roots']:boards.add(boardkey(c4flat(c4replay(r['history'])[0])))
        for j in json.loads((k/'PLAN.json').read_text())['jobs']:families.add(j['family'])
    return families,boards,locks,prior
def proof(h):
    b,z=c4replay(h);s=len(h)%2;out=[]
    for c in range(7):
        if z[c]==6:continue
        bit=1<<(7*c+z[c]);bb=b.copy();zz=z.copy();bb[s]|=bit;zz[c]+=1
        if won(bb[s]):l=u=1.
        elif sum(zz)==42:l=u=.5
        elif canwin(bb,zz,1-s):l=u=0.
        else:l,u=0.,1.
        out.append([c,l,u])
    return out
def cert(p):
    u=max(x[2] for x in p);return next((x[0] for x in p if x[1]>=u),None)
class Server:
    def __init__(self,bin=BIN,threads=8):
        self.p=SP.Popen(['nice','-n','10',str(bin),'--server'],stdin=SP.PIPE,stdout=SP.PIPE,stderr=SP.DEVNULL,bufsize=0,
            env=dict(os.environ,RAYON_NUM_THREADS=str(threads),QUARTZ_DISABLE_NN_CACHE='1',QUARTZ_DISABLE_ASYNC_PIPELINE='1',QUARTZ_DISABLE_QIPC_SHM='1'))
        self.buf=b'';self.cpu=process_clock(self.p.pid)
    def request(self,h,cmd='research_connect4_k28_cpu',**opts):
        q=dict(cmd=cmd,history=h,seed=17,model_path=str(MODEL),model_identity=MID,**opts)
        t=time.perf_counter_ns();cpu=time.clock_gettime_ns(self.cpu)
        self.p.stdin.write((json.dumps(q,separators=(',',':'))+'\n').encode());self.p.stdin.flush()
        deadline=time.monotonic()+60
        while b'\n' not in self.buf:
            left=deadline-time.monotonic()
            if left<=0 or not select.select([self.p.stdout],[],[],left)[0]:raise RuntimeError('native timeout')
            x=os.read(self.p.stdout.fileno(),65536)
            if not x:raise RuntimeError(('native EOF',self.p.poll(),q))
            self.buf+=x
            if len(self.buf)>8*1024*1024:raise RuntimeError('oversized reply')
        line,self.buf=self.buf.split(b'\n',1);a=json.loads(line)
        a['cpu_ns']=time.clock_gettime_ns(self.cpu)-cpu;a['wall_ns']=time.perf_counter_ns()-t
        return a
    def __enter__(self):return self
    def __exit__(self,*a):
        self.p.stdin.close()
        try:self.p.wait(timeout=5)
        except SP.TimeoutExpired:self.p.terminate();self.p.wait(timeout=5)
        if self.p.returncode!=0 and a[0] is None:raise RuntimeError(('server exit',self.p.returncode))
def plan():
    fam,boards,locks,prior=historical();nf=len(fam);rng=random.Random(SEED);jobs=[]
    for i in range(1800):
        while True:
            seq='';b=[0,0];z=[0]*7
            for j in range(8):
                opts=[c for c in range(7) if z[c]<6 and not won(b[j%2]|(1<<(7*c+z[c])))]
                c=rng.choice(opts);b[j%2]|=1<<(7*c+z[c]);z[c]+=1;seq+=str(c+1)
            k=mirror(seq)
            if k not in fam:break
        fam.add(k);jobs.append(dict(id=i,group='TRAIN' if i<900 else 'CAL' if i<1350 else 'TEST',opening=seq,family=k,seed=rng.randrange(2**63)))
    save('PLAN.json',dict(assets=assets(),seed=SEED,historical_families=nf,historical_boards=len(boards),history_locks=locks,prior_locks=prior,jobs=jobs))
    print('PLAN',len(jobs),nf,len(boards),flush=True)
def source_shard(jobs):
    rows=[]
    with Server(SOURCE_BIN,1) as srv:
        for j in jobs:
            h=j['opening'];rng=random.Random(j['seed']);eligible=[];steps=[]
            while True:
                b,z=c4replay(h);side=len(h)%2
                if any(won(x) for x in b) or len(h)==42:break
                legal=[c for c in range(7) if z[c]<6]
                own=[c for c in legal if won(b[side]|(1<<(7*c+z[c])))]
                opp=[c for c in legal if won(b[1-side]|(1<<(7*c+z[c])))]
                if len(h) in (24,26,28,30) and len(legal)>=3 and not own and not opp:eligible.append(dict(history=h,legal=legal))
                if len(h)>=30:break
                a=None
                if own:mv=own[0];kind='WIN'
                elif len(opp)==1:mv=opp[0];kind='BLOCK'
                elif len(legal)==1:mv=legal[0];kind='ONLY'
                elif rng.random()<.75:
                    a=srv.request(h,'research_connect4_evidence_cpu',arm='K16N',cap=32)
                    assert 'error' not in a and a['best'] in legal,a;mv=a['best'];kind='N32'
                else:mv=rng.choice(legal);kind='UNIFORM'
                steps.append(dict(move=mv,kind=kind,NN=0 if a is None else a['NN_calls']));h+=str(mv+1)
            rows.append(dict(job=j,history=h,steps=steps,eligible=eligible))
    return rows
def source():
    pl=read('PLAN.json');assert pl['assets']==assets();fam,boards,locks,prior=historical()
    assert locks==pl['history_locks'] and prior==pl['prior_locks']
    jobs=pl['jobs'];path=D/'SOURCE.jsonl.gz';assert not path.exists()
    with ProcessPoolExecutor(max_workers=6) as ex:
        rows=sorted((r for sh in ex.map(source_shard,[jobs[i::6] for i in range(6)]) for r in sh),key=lambda r:r['job']['id'])
    roots=[];nn=plies=queries=0
    with gzip.open(path,'wt',compresslevel=3) as fd:
        for j,t in zip(jobs,rows):
            assert j==t['job'] and j['family'] not in fam;fam.add(j['family']);selected=None
            for a in t['eligible']:
                key=boardkey(c4flat(c4replay(a['history'])[0]))
                if key not in boards:
                    boards.add(key);selected=dict(id=j['id'],group=j['group'],family=j['family'],history=a['history'],legal=a['legal'],depth=len(a['history']),tactical=False);roots.append(selected);break
            t['selected']=selected;emit(fd,t)
            nn+=sum(s['NN'] for s in t['steps']);queries+=sum(s['kind']=='N32' for s in t['steps']);plies+=len(t['steps'])
    save('COHORT.json',dict(plan=sha(D/'PLAN.json'),source=sha(path),roots=roots,NN=nn,queries=queries,plies=plies,counts=dict(C.Counter(r['group'] for r in roots))))
    print('COHORT',len(roots),dict(C.Counter(r['group'] for r in roots)),nn,queries,plies,flush=True)
def one_label(r):
    q=oracle(r);item=dict(root=r,status=q['status'])
    if q['status']=='EXACT':
        try:
            v,n=exact_wdl(r['history']);truth={a:(q['scores'][a]>0)-(q['scores'][a]<0) for a in r['legal']}
            item.update(scores=q['scores'],independent=v,nodes=n,agreement=v==truth)
            if v!=truth:item['status']='MISMATCH'
        except RuntimeError as e:item.update(status='INDEPENDENT_CAP',reason=str(e),scores=q['scores'])
    else:item['oracle_result']=q
    return item
if __name__=='__main__':globals()[sys.argv[1]]()
