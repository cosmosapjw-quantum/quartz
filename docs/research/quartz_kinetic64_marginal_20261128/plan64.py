from pathlib import Path
import json,gzip,hashlib,random,collections as C,shutil
D=Path(__file__).resolve().parent; ROOT=D.parents[2]; T=ROOT.parent/'quartz_kinetic64_marginal_run_20261128'
P=D.parent/'quartz_kinetic63_select_20261128'; P58=D.parent/'quartz_kinetic58_stop_20261128'
B=Path.home()/'Dropbox/personal_projects/quartz_kinetic64_marginal_backups'
SEEDS=(101,102,103,105,41,42,43); EXTRA=(211,222,233)
NOVEL=(311,322,333); ALL=SEEDS+EXTRA+NOVEL
ARMS=('H4','FREE','T2','T4','K63','ROBUST'); DEV=('FREE','T2','T4','OTHER')
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
    parent=read(P/'PLAN.json');roots=[];pop={};locks0={};rng=random.Random(2026112864)
    for mode in ('MAIN','ARCHIVE'):
        lock=read(P/(mode+'_LOCK.json'));assert sha(P/'PLAN.json')==lock['plan'];locks0[mode]=sha(P/(mode+'_LOCK.json'));candidates=C.defaultdict(list)
        for meta in lock['files']:
            f=P/meta['file'];assert sha(f)==meta['sha256']
            for gs in raw(f):
                g=gs['FREE'];j=g['job']
                for i,r in enumerate(g['rows']):
                    if not r.get('early',{}).get('reached'):continue
                    x=r['early']['at'];assert x['visits']==4 and r['arm']=='Q' and x['rival']!=x['move'] and r['select63']['op']==1
                    c=feature(x,r['partial']['bounds'])
                    candidates[c].append(dict(domain=mode,family_id=j['id'],seed=g['seed'],history=r['history'],color=g['color'],source_file=meta['file'],source_index=i,source_opening=j['opening'],cell=c,snapshot=x,bounds=r['partial']['bounds'],root_nn=r['actual_nn'],free=r['move']))
        pop[mode]=[len(candidates[i]) for i in range(6)]
        assert pop[mode]==([145,28,190,334,135,116] if mode=='MAIN' else [24,2,6,68,20,6])
        for k in range(6):
            c=sorted(candidates[k],key=lambda r:(r['family_id'],r['seed'],r['color'],r['source_index']))
            n=min(80,len(c)) if mode=='MAIN' else len(c)
            for r in rng.sample(c,n):roots.append(dict(id=len(roots),**r,arm_order=rng.sample(list(DEV),4),population=len(c),sample=n))
    assert len(roots)==554 and len({(r['domain'],r['family_id'],r['seed'],r['color'],r['source_index']) for r in roots})==554
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
    for panel,N in [('MAIN',16),('ARCHIVE',16),('NOVEL',12)]:
        for g in ('central','whole'):
            cells=[7*r+c for r in range(1,6) for c in range(1,6)] if g=='central' else list(range(49))
            for _ in range(N):
                while True:
                    op=rng.sample(cells,4);f=family(op)
                    if f not in prior:break
                prior.add(f);jobs.append(dict(id=len(jobs),opening=op,family=f,geometry=g,panel=panel,color_order=rng.sample([1,-1],2),arm_order=rng.sample(list(ARMS),6)))
    assert len(jobs)==88 and len(prior)==n+88
    models=parent['models']
    for s in SEEDS:assert sha(models[str(s)]['path'])==models[str(s)]['sha256']
    for ss in EXTRA:
        m=models[str(ss)];assert sha(m['path'])==m['sha256']==sha(m['backup'])
    base=Path.home()/'Dropbox/personal_projects/quartz/results/phase8c_v5_none/gomoku7/models/T4_S_VL'
    (B/'models').mkdir(exist_ok=True)
    for ss in (11,22,33):
        f=base/f'seed_{ss}'/'latest.pt';log=f.parent/'train_log.jsonl';status=f.parent/'checkpoint_status.json';tr=[json.loads(x) for x in log.read_text().splitlines()]
        assert len(tr)>=2 and tr[-1]['train_executed'] and read(status)['preferred_posttrain_checkpoint']=='latest.pt'
        digest=sha(f);assert digest not in model_hashes and digest not in {m['sha256'] for m in models.values()}
        back=B/'models'/f'seed_{ss+300}.pt';assert not back.exists();shutil.copy2(f,back);assert sha(back)==digest
        models[str(ss+300)]=dict(path=str(f),backup=str(back),sha256=digest,original_seed=ss,role='unseen phase8c_v5_none T4_S_VL trained checkpoint, not independent seed',training_log_sha256=sha(log),checkpoint_status_sha256=sha(status),train_last_iter=tr[-1]['iter'],train_executed=True)
    binlock=read(P/'BINARY_LOCK.json');binary=ROOT.parent/'quartz_kinetic63_select_run_20261128/mcts_demo';assert sha(binary)==binlock['binary_sha256']
    for p,digest in binlock['source_sha256'].items():assert sha(ROOT/p)==digest
    assert read(P/'TABLE.json')['table']==[2,0,0,1,2,0]
    T.mkdir(exist_ok=True)
    pl=dict(jobs=jobs,roots=roots,models=models,parent_lock_sha256=locks0,parent_plan_sha256=sha(P/'PLAN.json'),population=pop,prior_families=n,prior_locks=locks,old_binary=str(binary),old_binary_sha256=binlock['binary_sha256'],old_source_sha256=binlock['source_sha256'],workers=6,train_workers=4,seed=2026112864,bootstrap_seed=202611286400,bootstrap_reps=8000,lambda_num=1,lambda_den=500)
    (D/'PLAN.json').write_text(json.dumps(pl,indent=2,sort_keys=True)+'\n')
    print('population',pop,'roots',len(roots),'heldout families',len(jobs),'prior',n,'locks',len(locks),'distinct archived weights',len(NOVEL))
if __name__=='__main__':make()
