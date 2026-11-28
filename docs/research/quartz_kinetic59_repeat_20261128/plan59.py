from pathlib import Path
import json,hashlib,random
D=Path(__file__).resolve().parent; ROOT=D.parents[2]; T=ROOT.parent/'quartz_kinetic59_repeat_target_20261128'
P=D.parent/'quartz_kinetic58_stop_20261128'
SEEDS=(101,102,103,105,41,42,43)
ARMS=('CONTROL','FIRST','REPEAT')
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
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
def make():
    prior=set();locks=[]
    for p in sorted(D.parent.glob('quartz_*/PLAN.json')):
        if p.parent==D:continue
        x=json.loads(p.read_text());js=x.get('jobs',[])
        if not isinstance(js,list):continue
        valid=[j['opening'] for j in js if isinstance(j,dict) and isinstance(j.get('opening'),list) and len(j['opening'])==4 and all(type(a)==int and 0<=a<49 for a in j['opening'])]
        if valid:
            prior.update(family(h) for h in valid);locks.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'families':len(valid)})
    old=json.loads((P/'PLAN.json').read_text());models=old['models']
    evalmodels=json.loads((D.parent/'quartz_kinetic46_confirm_20261127/PLAN.json').read_text())['models']
    for s in SEEDS:
        a,b=models[str(s)],evalmodels[str(s)]
        assert sha(Path(a['path']))==a['sha256']==b['sha256']==sha(Path(b['path']))
    assert len({models[str(s)]['sha256'] for s in SEEDS})==7
    n=len(prior);rng=random.Random(2026112859);jobs=[]
    for h in (4,8,12,16):
        for g in ('central','whole'):
            cells=[7*r+c for r in range(1,6) for c in range(1,6)] if g=='central' else list(range(49))
            for i in range(8):
                while True:
                    op=rng.sample(cells,4);f=family(op)
                    if f not in prior:break
                prior.add(f)
                jobs.append(dict(id=len(jobs),opening=op,family=f,horizon=h,geometry=g,color_order=rng.sample([1,-1],2),arm_order=rng.sample(list(ARMS),3)))
    assert len(jobs)==64 and len(prior)==n+64
    lock=json.loads((P/'BINARY_LOCK.json').read_text())
    binary=ROOT.parent/'quartz_kinetic58_stop_target_20261128/mcts_demo'
    assert sha(binary)==lock['binary_sha256']
    for p,digest in lock['source_sha256'].items():assert sha(ROOT/p)==digest
    assert sha(ROOT/'Cargo.lock')==lock['cargo_lock_sha256']
    T.mkdir(exist_ok=True)
    (D/'PLAN.json').write_text(json.dumps(dict(jobs=jobs,models=models,prior_families=n,prior_locks=locks,binary=str(binary),binary_sha256=sha(binary),source_commit=lock['source_commit'],source_sha256=lock['source_sha256'],workers=6,seed=2026112859,bootstrap_seed=202611285900,bootstrap_reps=8000),indent=2,sort_keys=True)+'\n')
    print('fresh families',len(jobs),'prior',n,'locks',len(locks),'models',len(models))
if __name__=='__main__':make()
