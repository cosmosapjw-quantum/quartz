"""Prospective actual legacy A1 vs exact root certificate and lower-cap controls."""
import os,sys,json,time,hashlib,pathlib,random,statistics as S,gzip,math,subprocess as SP
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2];sys.path.insert(0,str(ROOT))
os.environ.update(QUARTZ_NO_COMPILE='1',QUARTZ_DISABLE_NN_CACHE='1',QUARTZ_DISABLE_ASYNC_PIPELINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')

SEED=2026112633
ARMS=('proof32','blind20','proof20')
SEEDS=(101,102,103);PAIRS=12;MARGIN=.05;WORKERS=4
BIN=ROOT.parent/'quartz_kinetic32_target_20261126/mcts_demo'
MODEL=pathlib.Path.home()/'Dropbox/personal_projects/quartz/results/phase15_stage7/gomoku7/training'
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(n):return json.loads((D/n).read_text())
def save(n,x):(D/n).write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
def lines(p):
    with gzip.open(p,'rt') as f:
        for s in f:yield json.loads(s)
def transform(h,t):
    out=[]
    for m in h:
        r,c=divmod(m,7)
        if t>=4:c=6-c
        for _ in range(t%4):r,c=c,6-r
        out.append(r*7+c)
    return out
def family(h):return min(tuple(transform(h,t)) for t in range(8))
def plan():
    rng=random.Random(SEED);jobs=[];families=set();cells=[7*r+c for r in range(1,6) for c in range(1,6)]
    for seed in SEEDS:
        for i in range(PAIRS):
            while True:
                h=rng.sample(cells,4);f=family(h)
                if f not in families:break
            families.add(f);assert len(set(h))==4
            jobs.append(dict(seed=seed,id=i,opening=h,family=f,game_seed=rng.randrange(2**63),candidate_order=rng.sample([1,-1],2)))
    models={}
    for seed in SEEDS:
        p=MODEL/f'seed_{seed}/best.pt';models[str(seed)]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,
            champion=json.loads((p.parent/'champion.json').read_text()),train_log_sha256=sha(p.parent/'train_log.jsonl'))
    save('PLAN.json',dict(seed=SEED,models=models,binary=dict(path=str(BIN),sha256=sha(BIN)),arms=ARMS,margin=MARGIN,pairs=PAIRS,workers=WORKERS,
        env={k:os.environ[k] for k in ('QUARTZ_NO_COMPILE','QUARTZ_DISABLE_NN_CACHE','QUARTZ_DISABLE_ASYNC_PIPELINE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},jobs=jobs,scope='fixed checkpoint panel; original Gomoku7 4-in-row, A1 settings; seeded four-ply openings; actual paired color swap; no policy fitting'))
    print('PLAN',len(jobs),'opening pairs per arm',len(jobs)*len(ARMS)*2,'games',flush=True)

if __name__=='__main__':plan()
