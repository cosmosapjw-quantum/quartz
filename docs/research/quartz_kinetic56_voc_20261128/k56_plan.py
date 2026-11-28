import pathlib,json,hashlib,random,sys,importlib.util
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2];T=ROOT.parent/'quartz_kinetic56_voc_target_20261128'
SEED=2026112856;SEEDS=(101,102,103,105,41,42,43);WORKERS=8;TIERS=((3,256),(5,1024),(7,4096))
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):
    p=D/n;assert not p.exists(),p;p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def load(n):return json.loads((D/n).read_text())
def family(h,n=7):
    z=[]
    for t in range(8):
        out=[]
        for a in h:
            r,c=divmod(a,n)
            if t>=4:c=n-1-c
            for _ in range(t%4):r,c=c,n-1-r
            out.append(r*n+c)
        z.append(tuple(sorted(out[::2])+sorted(out[1::2])))
    return min(z)
def complete_arms(a,rng):
    missing=set(("CONTROL","COLD","REUSE","MULTI"))-set(a);assert len(missing)==1
    a.insert(rng.randrange(4),missing.pop());return a
def insert_mix(a,rng):
    a.insert(rng.randrange(5),"MIX");return a
def plan():
    paths=[D.parent/f/'PLAN.json' for f in ('quartz_kinetic33_20261126','quartz_kinetic38_nnlock_20261126','quartz_kinetic39_voc_20261126','quartz_kinetic40_budget_20261126','quartz_kinetic41_exact_20261126','quartz_kinetic42_threshold_20261127','quartz_kinetic43_partial_20261127','quartz_kinetic44_incremental_20261127','quartz_kinetic45_sequential_20261127','quartz_kinetic46_confirm_20261127','quartz_kinetic47_native_20261127','quartz_kinetic48_voc_20261127','quartz_kinetic49_stability_20261127','quartz_kinetic50_intervention_20261127','quartz_kinetic51_targeted_20261127','quartz_kinetic52_confirm_20261127','quartz_kinetic53_reuse_20261127','quartz_kinetic54_cache_20261127','quartz_kinetic55_policy_20261128')]
    old=[json.loads(p.read_text()) for p in paths];seen={family(j['opening']) for x in old for j in x['jobs']}
    prior=len(seen);rng=random.Random(SEED);arm_rng=random.Random(SEED+5301);mix_rng=random.Random(SEED+5501);jobs=[];models=old[-1]['models']
    for m in models.values():assert sha(pathlib.Path(m['path']))==m['sha256']
    assert len({x['sha256'] for x in models.values()})==7
    for split,N in [('ARENA',48)]:
        for i in range(N):
            cells=[7*r+c for r in range(1,6) for c in range(1,6)] if i%2==0 else list(range(49))
            while True:
                h=rng.sample(cells,4);f=family(h)
                if f not in seen:break
            seen.add(f);jobs.append(dict(id=len(jobs),split=split,opening=h,family=f,color_order=rng.sample([1,-1],2),arm_order=arm_rng.sample(['CONTROL','COLD','REUSE','VOC'],4)))
    # Existing Rapfi historical discovery excludes its own K37 plan; include it explicitly.
    sys.path.insert(0,str(D.parent/'quartz_kinetic37_20261126'))
    import experiment as e
    fam,locks=e.historical();old37=e.read('PLAN.json')
    fam.update(j['family'] for j in old37['jobs'])
    fam.update(j['family'] for x in old[-13:] for j in x['rapfi_jobs'])
    n0=len(fam);rjobs=[]
    while len(rjobs)<512:
        dr,dc=rng.choice([(0,1),(1,0),(1,1),(1,-1)])
        r,c=rng.randrange(4,11),rng.randrange(4,11)
        ks=rng.choice([(0,1,3),(0,2,3)])
        black=[(r+(k-1)*dr,c+(k-1)*dc) for k in ks]
        if not all(1<=x<14 and 1<=y<14 for x,y in black):continue
        white=rng.sample([(x,y) for x in range(r-3,r+4) for y in range(c-3,c+4) if (x,y) not in black],3)
        rng.shuffle(black);rng.shuffle(white);h=[15*x+y for pair in zip(black,white) for x,y in pair]
        f=e.family(h)
        if f in fam:continue
        fam.add(f);rjobs.append(dict(id=len(rjobs),split='ARENA',stratum='unresolved_tactical',opening=h,family=f,seed=rng.randrange(2**63),color_order=rng.sample([1,-1],2)))
    ni=json.loads((D.parent/'quartz_kinetic40_budget_20261126/NATIVE_IDENTITY.json').read_text())
    native=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic40_budget_backups/mcts_demo'
    assert sha(native)==ni['binary_sha256']
    identity=e.identity();identity.update(actual_binary=str(native),actual_binary_sha256=sha(native))
    assert all(set(j['arm_order'])=={'CONTROL','COLD','REUSE','VOC'} for j in jobs)
    assert len(jobs)==48 and len(rjobs)==512
    save('PLAN.json',dict(seed=SEED,models=models,jobs=jobs,prior_families=prior,prior_locks=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in paths],rapfi_candidates=rjobs,rapfi_jobs=[],historical_target_lock_sha256=sha(paths[-1].parent/'NN_LOCK.json'),rapfi_prior=n0,rapfi_prior_locks=locks,rapfi_k37_plan_sha256=sha(e.D/'PLAN.json'),rapfi_identity=identity,tiers=TIERS,workers=WORKERS,rapfi_workers=6,bootstrap_replicates=8000,bootstrap_seed=202611285600,NI_margin=.05,NN_target=.30,scope='48 fresh NN families four cached arms;512 Rapfi screen candidates, selected plan separately locked'))
    print('PLAN',len(jobs),'Rapfi',len(rjobs),'prior',prior,n0)
if __name__=='__main__':plan()
