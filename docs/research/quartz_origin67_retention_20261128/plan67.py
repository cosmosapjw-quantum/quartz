from pathlib import Path
import json,gzip,hashlib,random,collections as C,shutil
D=Path(__file__).resolve().parent; ROOT=D.parents[2]; T=ROOT.parent/'quartz_origin67_retention_run_20261128'
P=D.parent/'quartz_origin66_omission_20261128'; P58=D.parent/'quartz_kinetic58_stop_20261128'
B=Path.home()/'Dropbox/personal_projects/quartz_origin67_retention_backups'
SEEDS=(101,102,103,105,41,42,43); EXTRA=(211,222,233)
NOVEL=(311,322,333); TRANSFER=(411,422,433); ALL=SEEDS+EXTRA+NOVEL+TRANSFER
ARMS=('H4','FREE','MA4','VOTE','FOCUS','EXPAND','CALV','CALF','CALX'); DEV=('MA4','VOTE','FOCUS','EXPAND')
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
    parent=read(P/'PLAN.json');rng=random.Random(2026112867);roots=[];pop={};locks0={};splits={}
    for panel in ('MAIN','ARCHIVE','NOVEL'):
        lock=read(P/(panel+'_LOCK.json'));assert lock['plan']==sha(P/'PLAN.json');locks0[panel]=sha(P/(panel+'_LOCK.json'))
        js=[j for j in parent['jobs'] if j['panel']==panel]
        cal=set()
        for geom in ('central','whole'):
            ids=sorted(j['id'] for j in js if j['geometry']==geom);cal.update(rng.sample(ids,len(ids)//3))
        splits[panel]=sorted(cal);c=C.Counter()
        for meta in lock['files']:
            f=P/meta['file'];assert sha(f)==meta['sha256']
            for gs in raw(f):
                g=gs['MA4'];j=g['job']
                for i,r in enumerate(g['rows']):
                    if 'allocation65' not in r:continue
                    x=r['allocation65'];bd=r['partial']['bounds'];assert r['arm']=='Q' and r['visits']==4
                    split='CAL' if j['id'] in cal else 'TRAIN';c[split]+=1;c['discord']+=x['fallback']!=x['chosen']
                    roots.append(dict(id=len(roots),domain=panel,split=split,family_id=j['id'],seed=g['seed'],color=g['color'],history=r['history'],source_file=meta['file'],source_index=i,source_opening=j['opening'],snapshot=x,bounds=bd,root_nn=r['actual_nn'],incumbent=r['move'],arm_order=rng.sample(list(DEV),4)))
        pop[panel]=dict(c)
    assert len(roots)==749
    assert {k:v['TRAIN']+v['CAL'] for k,v in pop.items()}=={'MAIN':518,'ARCHIVE':109,'NOVEL':122}
    assert {k:v['discord'] for k,v in pop.items()}=={'MAIN':257,'ARCHIVE':73,'NOVEL':89}
    # Preserve old atlas source, all TRAIN; board leakage exclusions applied before fit.
    atlas=read(P/'DEV_LOCK.json');assert atlas['plan']==sha(P/'PLAN.json')
    for m in atlas['files']:assert sha(P/m['file'])==m['sha256']
    locks0['DEV']=sha(P/'DEV_LOCK.json')
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
    for panel,N in [('MAIN',12),('ARCHIVE',6),('NOVEL',6),('TRANSFER',6)]:
        for g in ('central','whole'):
            cells=[7*r+c for r in range(1,6) for c in range(1,6)] if g=='central' else list(range(49))
            for _ in range(N):
                while True:
                    op=rng.sample(cells,4);f=family(op)
                    if f not in prior:break
                prior.add(f);jobs.append(dict(id=len(jobs),opening=op,family=f,geometry=g,panel=panel,color_order=rng.sample([1,-1],2),arm_order=rng.sample(list(ARMS),9)))
    assert len(jobs)==60 and len(prior)==n+60
    models=parent['models']
    for ss in SEEDS+EXTRA+NOVEL:
        m=models[str(ss)];assert sha(m['path'])==m['sha256']
        if ss in EXTRA+NOVEL:assert sha(m['backup'])==m['sha256']
    base=Path.home()/'Dropbox/personal_projects/quartz/results/phase8c_v4_legacy_az/gomoku7/models/T1_noS_noVL'
    (B/'models').mkdir(exist_ok=True)
    for ss in (11,22,33):
        f=base/f'seed_{ss}'/'latest.pt';log=f.parent/'train_log.jsonl';status=f.parent/'checkpoint_status.json';tr=[json.loads(x) for x in log.read_text().splitlines()]
        assert len(tr)>=2 and tr[-1]['train_executed'] and read(status)['preferred_posttrain_checkpoint']=='latest.pt'
        digest=sha(f);assert digest not in model_hashes and digest not in {m['sha256'] for m in models.values()}
        back=B/'models'/f'seed_{ss+400}.pt';assert not back.exists();shutil.copy2(f,back);assert sha(back)==digest
        models[str(ss+400)]=dict(path=str(f),backup=str(back),sha256=digest,original_seed=ss,role='legacy AZ trained checkpoint, transfer model not independent training seed',training_log_sha256=sha(log),checkpoint_status_sha256=sha(status),train_last_iter=tr[-1]['iter'],train_executed=True)
    binlock=read(P/'BINARY_LOCK.json');binary=ROOT.parent/'quartz_origin66_omission_run_20261128/mcts_demo';assert sha(binary)==binlock['binary_sha256']
    for p,digest in binlock['source_sha256'].items():assert sha(ROOT/p)==digest
    T.mkdir(exist_ok=True)
    pl=dict(jobs=jobs,roots=roots,models=models,parent_lock_sha256=locks0,parent_plan_sha256=sha(P/'PLAN.json'),population=pop,cal_family_ids=splits,prior_families=n,prior_locks=locks,old_binary=str(binary),old_binary_sha256=binlock['binary_sha256'],old_source_sha256=binlock['source_sha256'],workers=6,train_workers=6,seed=2026112867,bootstrap_seed=202611286700,bootstrap_reps=8000,lambda_num=1,lambda_den=500)
    (D/'PLAN.json').write_text(json.dumps(pl,indent=2,sort_keys=True)+'\n')
    print('population',pop,'roots',len(roots),'heldout families',len(jobs),'prior',n,'locks',len(locks),'transfer weights',len(NOVEL))
if __name__=='__main__':make()
