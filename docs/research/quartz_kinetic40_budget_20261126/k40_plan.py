import pathlib,json,hashlib,random
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2]
assert D.name=='quartz_kinetic40_budget_20261126'
SEED=2026112640;SEEDS=(101,102,103,105);FIT=(101,103,105);WORKERS=8
SPLITS={'TRAIN':120,'CAL':60,'TEST':60,'ARENA':64}
H=(20,26,32)
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
def save(name,x):
    p=D/name;assert not p.exists(),p;p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def load(name):return json.loads((D/name).read_text())
def plan():
    paths=[D.parent/f/'PLAN.json' for f in ('quartz_kinetic33_20261126','quartz_kinetic38_nnlock_20261126','quartz_kinetic39_voc_20261126')]
    old=[json.loads(p.read_text()) for p in paths]
    seen={family(j['opening']) for x in old for j in x['jobs']}
    prior=len(seen);rng=random.Random(SEED);jobs=[]
    models=old[-1]['models']
    for s,m in models.items():assert sha(pathlib.Path(m['path']))==m['sha256']
    for split,N in SPLITS.items():
        for i in range(N):
            cells=[7*r+c for r in range(1,6) for c in range(1,6)] if (i//6)%2==0 else list(range(49))
            while True:
                h=rng.sample(cells,4);f=family(h)
                if f not in seen:break
            seen.add(f);jobs.append(dict(id=len(jobs),split=split,opening=h,family=f,target=4+i%6,source=(101 if split in ('TRAIN','CAL') and SEEDS[(i//6)%4]==102 else SEEDS[(i//6)%4]),color_order=rng.sample([1,-1],2)))
    save('PLAN.json',dict(seed=SEED,models=models,jobs=jobs,fit=list(FIT),heldout=102,splits=SPLITS,workers=WORKERS,horizons=H,budget_fraction='7/10',margin=.05,prior_families=prior,prior_locks=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in paths],scope='matched common board families across fixed evaluators, no cross-model games; source assigned before data; heldout102 from all fitting and admission; frozen random hash policy; pre-data source correction excludes 102 from TRAIN/CAL source as well as fit and admission'))
    print('PLAN',len(jobs),'prior',prior,'models',list(models))
if __name__=='__main__':plan()
