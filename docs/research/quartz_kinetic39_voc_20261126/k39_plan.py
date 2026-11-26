import pathlib,json,hashlib,random
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2]
assert D.name=='quartz_kinetic39_voc_20261126'
SEED=2026112639;SEEDS=(101,102,103,105);WORKERS=8
SPLITS={'TRAIN':64,'CAL':32,'TEST':64,'ARENA':48}
ARMS=('L20_B20','L20_B32','L32_B20','L32_B32','A_B32')
PRICE=.002
MODEL=pathlib.Path.home()/'Dropbox/personal_projects/quartz/results/phase15_stage7/gomoku7/training'
BACKUP=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic39_voc_backups'
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
    oldpaths=[D.parent/f/'PLAN.json' for f in ('quartz_kinetic33_20261126','quartz_kinetic38_nnlock_20261126')]
    old=[json.loads(p.read_text()) for p in oldpaths]
    seen={family(j['opening']) for x in old for j in x['jobs']}
    prior=len(seen);rng=random.Random(SEED);jobs=[];models={}
    for s in SEEDS:
        p=MODEL/f'seed_{s if s!=105 else 103}'/('best.pt' if s!=105 else 'gen_5.pt')
        digest=sha(p)
        if s!=105:assert digest==old[-1]['models'][str(s)]['sha256']
        m=json.loads((p.parent/'champion.json').read_text())
        models[str(s)]=dict(path=str(p),sha256=digest,bytes=p.stat().st_size,champion=m,
              training_log_sha256=sha(p.parent/'train_log.jsonl'),role='103 unpromoted trained gen5 (related checkpoint)' if s==105 else 'K38 frozen best checkpoint')
        for split,N in SPLITS.items():
            for i in range(N):
                cells=[7*r+c for r in range(1,6) for c in range(1,6)] if (i//8)%2==0 else list(range(49))
                while True:
                    h=rng.sample(cells,4);f=family(h)
                    if f not in seen:break
                seen.add(f);jobs.append(dict(id=len(jobs),seed=s,split=split,opening=h,family=f,target=4+i%8,color_order=rng.sample([1,-1],2)))
    save('PLAN.json',dict(seed=SEED,models=models,jobs=jobs,arms=ARMS,splits=SPLITS,workers=WORKERS,c_puct=2.,price=PRICE,margin=.05,nn_reduction=.3,prior_families=prior,prior_locks=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in oldpaths],scope='fixed model strata; 105 related to 103; same exact weights on both sides; D4 opening families excluded across K33/K38; root targets4..11 balanced parity correction pre-data; four factorial Quartz-vs-baseline and one adaptive arm; no post-test fitting'))
    print('PLAN',len(jobs),prior,[(s,sum(j['seed']==s for j in jobs)) for s in SEEDS])
if __name__=='__main__':plan()
