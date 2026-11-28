from pathlib import Path
import json,gzip,hashlib,random,collections as C,shutil
D=Path(__file__).resolve().parent; ROOT=D.parents[2]; T=ROOT.parent/'quartz_origin65_allocation_run_20261128'
P=D.parent/'quartz_kinetic64_marginal_20261128'; P58=D.parent/'quartz_kinetic58_stop_20261128'
B=Path.home()/'Dropbox/personal_projects/quartz_origin65_allocation_backups'
SEEDS=(101,102,103,105,41,42,43); EXTRA=(211,222,233)
NOVEL=(311,322,333); ALL=SEEDS+EXTRA+NOVEL
ARMS=('H4','FREE','L4','A4','M4','MA4','X4'); DEV=('L4','A4','M4','MA4','X4')
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def family(h):
    z=[]
    for t in range(8):
        out=[]
        for a in h:
            r,c=divmod(a,7)
            if t>=4:c=6-c
            for _ in range(t%4):r,c=c,6-r
            out.append(r*7+c)
        z.append(tuple(sorted(out[::2])+sorted(out[1::2])))
    return min(z)
def feature(x,bounds):
    es={e[0]:e[1:] for e in x['edges']};a,b=x['move'],x['rival'];n,w,_=es[a];m,v,_=es[b]
    return 0 if n==0 else 1 if m==0 and w<0 else 2 if m==0 else 3 if v*n>w*m else 4 if w<0 else 5
def make():
    parent=read(P/'PLAN.json');rng=random.Random(2026112865);roots=parent['roots'];pop=parent['population'];locks0={m:sha(P/(m+'_LOCK.json')) for m in ('DEV','MAIN','ARCHIVE','NOVEL')}
    assert len(roots)==554 and len({r['id'] for r in roots})==554
    prior=set();locks=[];model_hashes=set()
    for p in sorted(D.parent.glob('quartz_*/PLAN.json')):
        if p.parent==D:continue
        x=read(p);js=x.get('jobs',[])
        for v in x.get('models',{}).values() if isinstance(x.get('models'),dict) else []:
            if isinstance(v,dict) and 'sha256' in v:model_hashes.add(v['sha256'])
        if not isinstance(js,list):continue
        valid=[j['opening'] for j in js if isinstance(j,dict) and isinstance(j.get('opening'),list) and len(j['opening'])==4 and all(type(a)==int and 0<=a<49 for a in j['opening'])]
        if valid:prior.update(family(h) for h in valid);locks.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),count=len(valid)))
    n=len(prior);jobs=[]
    for panel,N in [('MAIN',12),('ARCHIVE',6),('NOVEL',6)]:
        for g in ('central','whole'):
            cells=[7*r+c for r in range(1,6) for c in range(1,6)] if g=='central' else list(range(49))
            for _ in range(N):
                while True:
                    op=rng.sample(cells,4);f=family(op)
                    if f not in prior:break
                prior.add(f);jobs.append(dict(id=len(jobs),opening=op,family=f,geometry=g,panel=panel,color_order=rng.sample([1,-1],2),arm_order=rng.sample(list(ARMS),7)))
    assert len(jobs)==48 and len(prior)==n+48
    models=parent['models']
    for ss in ALL:
        m=models[str(ss)];assert sha(m['path'])==m['sha256']
        if ss in EXTRA+NOVEL:assert sha(m['backup'])==m['sha256']
    binlock=read(P/'BINARY_LOCK.json');binary=ROOT.parent/'quartz_kinetic64_marginal_run_20261128/mcts_demo';assert sha(binary)==binlock['binary_sha256']
    for p,digest in binlock['source_sha256'].items():assert sha(ROOT/p)==digest
    assert read(P/'TABLE.json')['table']==[0]*6
    T.mkdir(exist_ok=True)
    pl=dict(jobs=jobs,roots=roots,models=models,parent_lock_sha256=locks0,parent_plan_sha256=sha(P/'PLAN.json'),population=pop,prior_families=n,prior_locks=locks,old_binary=str(binary),old_binary_sha256=binlock['binary_sha256'],old_source_sha256=binlock['source_sha256'],workers=6,train_workers=4,seed=2026112865,bootstrap_seed=202611286500,bootstrap_reps=8000,lambda_num=1,lambda_den=500)
    (D/'PLAN.json').write_text(json.dumps(pl,indent=2,sort_keys=True)+'\n')
    print('population',pop,'roots',len(roots),'heldout families',len(jobs),'prior',n,'locks',len(locks),'novel weights',len(NOVEL))
if __name__=='__main__':make()
