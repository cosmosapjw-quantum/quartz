import pathlib,json,hashlib,random,sys,importlib.util
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2];T=ROOT.parent/'quartz_kinetic46_confirm_target_20261127'
SEED=2026112746;SEEDS=(101,102,103,105,41,42,43);WORKERS=8;TIERS=((3,256),(5,1024),(7,4096))
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
def plan():
    paths=[D.parent/f/'PLAN.json' for f in ('quartz_kinetic33_20261126','quartz_kinetic38_nnlock_20261126','quartz_kinetic39_voc_20261126','quartz_kinetic40_budget_20261126','quartz_kinetic41_exact_20261126','quartz_kinetic42_threshold_20261127','quartz_kinetic43_partial_20261127','quartz_kinetic44_incremental_20261127','quartz_kinetic45_sequential_20261127')]
    old=[json.loads(p.read_text()) for p in paths];seen={family(j['opening']) for x in old for j in x['jobs']}
    prior=len(seen);rng=random.Random(SEED);jobs=[];models=old[-1]['models']
    for m in models.values():assert sha(pathlib.Path(m['path']))==m['sha256']
    import shutil
    src=pathlib.Path.home()/'Dropbox/personal_projects/quartz/results/phase15_ablation/gomoku7/bootstrap'
    dst=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic46_confirm_backups/models'
    hashes={41:'d058771b43e948a9dec121be72bc0aa4228b2bdd9d1b52e55ccdc36626ca471b',42:'75d85209040f4e9047b6157ea5aece756bbe78cbef76bc92d515661189bd334f',43:'4a1fd820002fdfa62e2693febded77a4d4d5884a85a0b1a18f88a5f7c248374b'}
    for seed,digest in hashes.items():
        q=src/f'seed_{seed}';pt=q/'gen_2.pt';assert sha(pt)==digest
        b=dst/f'seed_{seed}_gen2.pt';assert not b.exists();shutil.copy2(pt,b);assert sha(b)==digest
        train=json.loads((q/'bootstrap.json').read_text());status=json.loads((q/'checkpoint_status.json').read_text());log=json.loads((q/'train_log.jsonl').read_text().splitlines()[-1])
        assert train['seed']==seed and train['returncode']==0 and log['iter']==2 and log['train_executed']
        models[str(seed)]=dict(path=str(b),sha256=digest,original_path=str(pt),original_seed=seed,generation=2,architecture='64f/4b/vh64',promotion=status['saw_promotion'],training=train,train_log_last=log)
    assert len({x['sha256'] for x in models.values()})==7
    for split,N in [('ARENA',96)]:
        for i in range(N):
            cells=[7*r+c for r in range(1,6) for c in range(1,6)] if i%2==0 else list(range(49))
            while True:
                h=rng.sample(cells,4);f=family(h)
                if f not in seen:break
            seen.add(f);jobs.append(dict(id=len(jobs),split=split,opening=h,family=f,color_order=rng.sample([1,-1],2)))
    # Existing Rapfi historical discovery excludes its own K37 plan; include it explicitly.
    sys.path.insert(0,str(D.parent/'quartz_kinetic37_20261126'))
    import experiment as e
    fam,locks=e.historical();old37=e.read('PLAN.json')
    fam.update(j['family'] for j in old37['jobs'])
    fam.update(j['family'] for x in old[-3:] for j in x['rapfi_jobs'])
    n0=len(fam);rjobs=[]
    for split,N in [('ARENA',64)]:
        for i in range(N):
            cells=[15*r+c for r in range(4,11) for c in range(4,11)] if i%2==0 else [15*r+c for r in range(2,13) for c in range(2,13)]
            while True:
                h=rng.sample(cells,6);f=e.family(h)
                if f not in fam:break
            fam.add(f);rjobs.append(dict(id=len(rjobs),split=split,opening=h,family=f,seed=rng.randrange(2**63),color_order=rng.sample([1,-1],2)))
    ni=json.loads((D.parent/'quartz_kinetic40_budget_20261126/NATIVE_IDENTITY.json').read_text())
    native=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic40_budget_backups/mcts_demo'
    assert sha(native)==ni['binary_sha256']
    identity=e.identity();identity.update(actual_binary=str(native),actual_binary_sha256=sha(native))
    save('PLAN.json',dict(seed=SEED,models=models,jobs=jobs,prior_families=prior,prior_locks=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in paths],rapfi_jobs=rjobs,rapfi_prior=n0,rapfi_prior_locks=locks,rapfi_k37_plan_sha256=sha(e.D/'PLAN.json'),rapfi_identity=identity,tiers=TIERS,workers=WORKERS,rapfi_workers=4,bootstrap_replicates=8000,bootstrap_seed=202611274600,NI_margin=.05,NN_target=.30,scope='96 fresh same-evaluator NN families, both Q20/Q32, old and new panels separately; 64 Rapfi pairs; no adaptive fitting'))
    print('PLAN',len(jobs),'Rapfi',len(rjobs),'prior',prior,n0)
if __name__=='__main__':plan()
