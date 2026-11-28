from pathlib import Path
import json,gzip,hashlib,random,collections as C
D=Path(__file__).resolve().parent; ROOT=D.parents[2]; T=ROOT.parent/'quartz_kinetic60_voc_target_20261128'
P=D.parent/'quartz_kinetic59_repeat_20261128'
P58=D.parent/'quartz_kinetic58_stop_20261128'
SEEDS=(101,102,103,105,41,42,43)
ARMS=('CONTROL','REPEAT','ADAPT')
HS=(4,8,12,16)
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
def make():
    parent=read(P/'PLAN.json');lock=read(P/'NN_LOCK.json');assert sha(P/'PLAN.json')==lock['plan']
    candidates=C.defaultdict(dict)
    for m in lock['files']:
        f=P/m['file'];assert sha(f)==m['sha256']
        for gs in raw(f):
            g=gs['REPEAT'];j=g['job']
            for i,r in enumerate(g['rows']):
                if 'early' in r and r['arm']=='Q':
                    key=(g['seed'],tuple(r['history']),j['horizon'])
                    candidates[j['id']][key]=dict(family_id=j['id'],seed=g['seed'],history=r['history'],h0=j['horizon'],color=g['color'],source_file=m['file'],source_index=i,source_opening=j['opening'])
    rng=random.Random(2026112860);roots=[]
    for k in sorted(candidates):
        c=[v for kk,v in sorted(candidates[k].items())]
        for r in rng.sample(c,min(4,len(c))):
            roots.append(dict(id=len(roots),**r,arm_order=rng.sample([0,*HS],5)))
    assert len({(r['seed'],tuple(r['history']),r['h0'],r['family_id']) for r in roots})==len(roots)
    prior=set();locks=[]
    for p in sorted(D.parent.glob('quartz_*/PLAN.json')):
        if p.parent==D:continue
        x=read(p);js=x.get('jobs',[])
        if not isinstance(js,list):continue
        valid=[j['opening'] for j in js if isinstance(j,dict) and isinstance(j.get('opening'),list) and len(j['opening'])==4 and all(type(a)==int and 0<=a<49 for a in j['opening'])]
        if valid:prior.update(family(h) for h in valid);locks.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),count=len(valid)))
    n=len(prior);jobs=[]
    for h in HS:
        for g in ('central','whole'):
            cells=[7*r+c for r in range(1,6) for c in range(1,6)] if g=='central' else list(range(49))
            for _ in range(8):
                while True:
                    op=rng.sample(cells,4);f=family(op)
                    if f not in prior:break
                prior.add(f);jobs.append(dict(id=len(jobs),opening=op,family=f,horizon=h,geometry=g,color_order=rng.sample([1,-1],2),arm_order=rng.sample(list(ARMS),3)))
    assert len(jobs)==64 and len(prior)==n+64
    for s in SEEDS:
        m=parent['models'][str(s)];assert sha(m['path'])==m['sha256']
    assert sha(parent['binary'])==parent['binary_sha256']
    for p,digest in parent['source_sha256'].items():assert sha(ROOT/p)==digest
    T.mkdir(exist_ok=True)
    pl=dict(jobs=jobs,roots=roots,models=parent['models'],parent_lock_sha256=sha(P/'NN_LOCK.json'),parent_plan_sha256=sha(P/'PLAN.json'),candidate_families={str(k):len(v) for k,v in candidates.items()},prior_families=n,prior_locks=locks,old_binary=parent['binary'],old_binary_sha256=parent['binary_sha256'],old_source_sha256=parent['source_sha256'],workers=6,train_workers=4,seed=2026112860,bootstrap_seed=202611286000,bootstrap_reps=8000,lambda_num=1,lambda_den=500,min_family=8)
    (D/'PLAN.json').write_text(json.dumps(pl,indent=2,sort_keys=True)+'\n')
    print('root count',len(roots),'families',len(candidates),'candidate count',sum(map(len,candidates.values())),'fresh games',len(jobs)*len(SEEDS)*6,'prior',n,'locks',len(locks))
if __name__=='__main__':make()
