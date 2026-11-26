import pathlib,json,hashlib,random
D=pathlib.Path(__file__).resolve().parent; ROOT=D.parents[2]
SEEDS=(101,102,103); PAIRS=32; WORKERS=8
SEED=2026112638; ARMS=('L32','L20')
MODEL=pathlib.Path.home()/'Dropbox/personal_projects/quartz/results/phase15_stage7/gomoku7/training'
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
    p=D/name;assert not p.exists();p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def load(name):return json.loads((D/name).read_text())
def plan():
    old=D.parent/'quartz_kinetic33_20261126/PLAN.json'
    seen={family(j['opening']) for j in json.loads(old.read_text())['jobs']}
    prior=len(seen);rng=random.Random(SEED);jobs=[]
    for s in SEEDS:
        for i in range(PAIRS):
            cells=[7*r+c for r in range(1,6) for c in range(1,6)] if i%2==0 else list(range(49))
            while True:
                h=rng.sample(cells,4);f=family(h)
                if f not in seen:break
            seen.add(f);jobs.append(dict(seed=s,id=i,opening=h,family=f,color_order=rng.sample([1,-1],2)))
    models={}
    oldmodels=json.loads(old.read_text())['models']
    for s in SEEDS:
        p=MODEL/f'seed_{s}/best.pt';digest=sha(p)
        assert digest==oldmodels[str(s)]['sha256']
        models[str(s)]=dict(path=str(p),sha256=digest,bytes=p.stat().st_size,champion=json.loads((p.parent/'champion.json').read_text()))
    save('PLAN.json',dict(seed=SEED,models=models,jobs=jobs,arms=ARMS,pairs=PAIRS,workers=WORKERS,c_puct=2.0,margin=.05,savings=.30,prior_family_count=prior,prior_sha256=sha(old),scope='actual same-checkpoint games, fixed common exact gate, serial PUCT no TT/no noise, hard visits, deterministic tie smallest index; no post-test fit'))
    print('LOCKED',len(jobs),len(jobs)*len(ARMS)*2,models.keys())
if __name__=='__main__':plan()
