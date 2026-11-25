"""Fresh native sequential stopping; historical data used only for exclusions/diagnostics."""
import pathlib,sys,json,gzip,hashlib,subprocess as SP,random,statistics as S,collections as C,math
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2];R=ROOT/'docs/research'
P20=ROOT.parent/'quartz_kinetic20_20261126/docs/research/quartz_kinetic20_20261126'
sys.path.insert(0,str(P20));import acquire as k20
sys.path.insert(0,str(D))

p=k20.p;old=k20.old;BIN=ROOT.parent/'quartz_kinetic26_target_20261126/mcts_demo'
BASE='3c538ae0e7cc71f0c4065d6bf6e729070f733dde';SEED=2026112626;PRICE=.002
def sha(path):return hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
def read(n):return json.loads((D/n).read_text())
def save(n,x):
    f=D/n;assert not f.exists(),n;f.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n')
def raw(path):
    with gzip.open(path,'rt') as f:
        for l in f:yield json.loads(l)
def emit(f,x):f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n');f.flush()
def lock(path):
    rel=str(path.relative_to(ROOT));b=SP.check_output(['git','-C',str(ROOT),'show',BASE+':'+rel])
    assert hashlib.sha256(b).hexdigest()==sha(path);return dict(commit=BASE,path=rel,sha256=sha(path))
def ident():
    b=read('RUST_BUILD_IDENTITY.json');assert sha(BIN)==b['binary_sha256']
    return dict(binary=sha(BIN),source=b['source_head'],parent=k20.identity())
def historical():
    f,b,l,h=k20.historical();locks=[]
    for n in ('PLAN.json','COHORT.json','SOURCE.jsonl.gz'):
        locks.append(lock(R/'quartz_kinetic20_20261126'/n))
    co=json.loads((R/'quartz_kinetic20_20261126/COHORT.json').read_text())
    for r in co['roots']:
        b.add(old.boardkey(old.c4flat(old.c4replay(r['history'])[0])))
    for j in json.loads((R/'quartz_kinetic20_20261126/PLAN.json').read_text())['jobs']:f.add(j['family'])
    k21=R/'quartz_kinetic21_20261126'
    for name in ('PLAN.json','COHORT.json','SOURCE.jsonl.gz','sequential.py'):
        locks.append(lock(k21/name))
    co=json.loads((k21/'COHORT.json').read_text())
    for rr in co['roots']:b.add(old.boardkey(old.c4flat(old.c4replay(rr['history'])[0])))
    for j in json.loads((k21/'PLAN.json').read_text())['jobs']:f.add(j['family'])
    k22=R/'quartz_kinetic22_20261126'
    for name in ('PLAN.json','COHORT.json','SOURCE.jsonl.gz','source_gen.py'):
        locks.append(lock(k22/name))
    co=json.loads((k22/'COHORT.json').read_text())
    for rr in co['roots']:b.add(old.boardkey(old.c4flat(old.c4replay(rr['history'])[0])))
    for j in json.loads((k22/'PLAN.json').read_text())['jobs']:f.add(j['family'])
    k23=R/'quartz_kinetic23_20261126'
    for name in ('PLAN.json','COHORT.json','SOURCE.jsonl.gz','source_gen.py'):
        locks.append(lock(k23/name))
    co=json.loads((k23/'COHORT.json').read_text())
    for rr in co['roots']:b.add(old.boardkey(old.c4flat(old.c4replay(rr['history'])[0])))
    for j in json.loads((k23/'PLAN.json').read_text())['jobs']:f.add(j['family'])
    k24=R/'quartz_kinetic24_20261126'
    for name in ('PLAN.json','COHORT.json','SOURCE.jsonl.gz','source_gen.py'):
        locks.append(lock(k24/name))
    co=json.loads((k24/'COHORT.json').read_text())
    for rr in co['roots']:b.add(old.boardkey(old.c4flat(old.c4replay(rr['history'])[0])))
    for j in json.loads((k24/'PLAN.json').read_text())['jobs']:f.add(j['family'])
    k25=R/'quartz_kinetic25_20261126'
    for name in ('PLAN.json','COHORT.json','SOURCE.jsonl.gz','source_gen.py'):
        locks.append(lock(k25/name))
    co=json.loads((k25/'COHORT.json').read_text())
    for rr in co['roots']:b.add(old.boardkey(old.c4flat(old.c4replay(rr['history'])[0])))
    for j in json.loads((k25/'PLAN.json').read_text())['jobs']:f.add(j['family'])
    return f,b,locks,l,h
def plan():
    f,b,l,oldl,h=historical();n=len(f);rng=random.Random(SEED);jobs=[]
    for i in range(300):
        while True:
            seq='';board=[0,0];ht=[0]*7
            for j in range(8):
                opts=[c for c in range(7) if ht[c]<6 and not old.won(board[j%2]|(1<<(7*c+ht[c])))]
                c=rng.choice(opts);board[j%2]|=1<<(7*c+ht[c]);ht[c]+=1;seq+=str(c+1)
            k=old.mirror(seq)
            if k not in f:break
        f.add(k);jobs.append(dict(id=i,group='TRAIN' if i<150 else 'CAL' if i<225 else 'TEST',opening=seq,family=k,seed=rng.randrange(2**63)))
    save('PLAN.json',dict(identity=ident(),seed=SEED,historical_families=n,historical_boards=len(b),K20_K21_K22_K23_K24_K25_locks=l,prior_locks=oldl,history_locks=h,jobs=jobs))
    print('PLAN',len(jobs),n,len(b),flush=True)
def source():
    pl=read('PLAN.json');assert pl['identity']==ident();f,b,l,oldl,h=historical();assert l==pl['K20_K21_K22_K23_K24_K25_locks'] and oldl==pl['prior_locks'] and h==pl['history_locks']
    path=D/'SOURCE.jsonl.gz';assert not path.exists();roots=[];queries=nn=plies=0
    with p.Server(p.BIN) as srv,gzip.open(path,'wt',compresslevel=3) as fd:
        for j in pl['jobs']:
            assert j['family'] not in f;f.add(j['family'])
            hh=j['opening'];rng=random.Random(j['seed']);eligible=[];steps=[]
            while True:
                board,z=old.c4replay(hh);side=len(hh)%2
                if any(old.won(x) for x in board) or len(hh)==42:break
                legal=[c for c in range(7) if z[c]<6]
                own=[c for c in legal if old.won(board[side]|(1<<(7*c+z[c])))]
                opp=[c for c in legal if old.won(board[1-side]|(1<<(7*c+z[c])))]
                key=old.boardkey(old.c4flat(board))
                if len(hh) in (24,26,28,30) and len(legal)>=3 and not own and not opp and key not in b:eligible.append((hh,legal,key))
                if len(hh)>=30:break
                a=None
                if own:mv=own[0];kind='WIN'
                elif len(opp)==1:mv=opp[0];kind='BLOCK'
                elif len(legal)==1:mv=legal[0];kind='ONLY'
                elif rng.random()<.75:
                    a=srv.request(hh,'K16N',32);assert 'error' not in a and a['best'] in legal
                    mv=a['best'];kind='N32';queries+=1;nn+=a['NN_calls']
                else:mv=rng.choice(legal);kind='UNIFORM'
                steps.append(dict(move=mv,kind=kind,NN=0 if a is None else a['NN_calls']));plies+=1;hh+=str(mv+1)
            root=None
            if eligible:
                h0,legal,key=eligible[0];b.add(key);root=dict(id=j['id'],group=j['group'],family=j['family'],history=h0,legal=legal,depth=len(h0),tactical=False);roots.append(root)
            emit(fd,dict(job=j,history=hh,steps=steps,selected=root))
            if j['id']%60==0:print('SOURCE',j['id'],len(roots),flush=True)
    save('COHORT.json',dict(plan=sha(D/'PLAN.json'),source=sha(path),roots=roots,queries=queries,NN=nn,plies=plies,counts=dict(C.Counter(x['group'] for x in roots))))
    print('COHORT',len(roots),C.Counter(x['group'] for x in roots),'NN',nn,'queries',queries,flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
