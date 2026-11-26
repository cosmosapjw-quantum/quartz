import pathlib,json,hashlib,random,importlib.util,collections
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2]
assert D.name=='quartz_kinetic41_exact_20261126'
SEED=2026112641;SEEDS=(101,102,103,105);FIT=(101,103,105);WORKERS=8
H=(20,26,32,64);BUDGET=25000
spec=importlib.util.spec_from_file_location('_k33_rules',D.parent/'quartz_kinetic33_20261126/proof.py')
rule=importlib.util.module_from_spec(spec);spec.loader.exec_module(rule)
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
def safe_history(rng,n):
    a=b=0;h=[]
    for i in range(n):
        x=a if i%2==0 else b
        safe=[m for m in rule.cells(rule.FULL^(a|b)) if not any((x|1<<m)&l==l for l in rule.BYCELL[m])]
        if not safe:return None
        m=rng.choice(safe);h.append(m)
        if i%2==0:a|=1<<m
        else:b|=1<<m
    return h
def plan():
    paths=[D.parent/f/'PLAN.json' for f in ('quartz_kinetic33_20261126','quartz_kinetic38_nnlock_20261126','quartz_kinetic39_voc_20261126','quartz_kinetic40_budget_20261126')]
    old=[json.loads(p.read_text()) for p in paths];seen={family(j['opening']) for x in old for j in x['jobs']}
    prior=len(seen);rng=random.Random(SEED);jobs=[];models=old[-1]['models'];ct=collections.Counter()
    for m in models.values():assert sha(pathlib.Path(m['path']))==m['sha256']
    for split,N in [('TRAIN',120),('CAL',60),('TEST',60),('LATE',64)]:
        for i in range(N):
            e=(8,10,12)[i%3]
            while True:
                ct['attempts']+=1
                if ct['attempts']>6000:raise RuntimeError(f'frame admission ceiling {dict(ct)}')
                h=safe_history(rng,49-e)
                if h is None:ct['placement_dead_end']+=1;continue
                a,b=rule.from_history(h);f=family(h)
                if f in seen:ct['duplicate']+=1;continue
                p=rule.root_certificate(a,b) if len(h)%2==0 else rule.root_certificate(b,a)
                if p['move'] is not None:ct['gated']+=1;continue
                break
            seen.add(f);ct['admitted']+=1
            jobs.append(dict(id=len(jobs),split=split,opening=h,family=f,empties=e,color_order=rng.sample([1,-1],2)))
    for i in range(64):
        cells=[7*r+c for r in range(1,6) for c in range(1,6)] if i%2==0 else list(range(49))
        while True:
            h=rng.sample(cells,4);f=family(h)
            if f not in seen:break
        seen.add(f);jobs.append(dict(id=len(jobs),split='EARLY',opening=h,family=f,empties=45,color_order=rng.sample([1,-1],2)))
    save('PLAN.json',dict(seed=SEED,models=models,jobs=jobs,fit=list(FIT),heldout=102,workers=WORKERS,horizons=H,proof_node_budget=BUDGET,budget_fraction='7/10',margin=.05,prior_families=prior,prior_locks=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in paths],generation_counts=dict(ct),scope='safe random legal nonterminal late frame, K33 unresolved filter independent of exact outcomes; 64 late and 64 early actual game starts; fixed checkpoints'))
    print('PLAN',len(jobs),dict(ct), 'prior',prior)
if __name__=='__main__':plan()
