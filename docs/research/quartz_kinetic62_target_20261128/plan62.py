from pathlib import Path
import json,gzip,hashlib,random,collections as C,shutil
D=Path(__file__).resolve().parent; ROOT=D.parents[2]; T=ROOT.parent/'quartz_kinetic62_target_run_20261128'
P=D.parent/'quartz_kinetic61_harm_20261128'; P58=D.parent/'quartz_kinetic58_stop_20261128'
B=Path.home()/'Dropbox/personal_projects/quartz_kinetic62_target_backups'
SEEDS=(101,102,103,105,41,42,43); EXTRA=(211,222,233)
ARMS=('H4','H8','TARGET'); DEV=(*ARMS,'FORCE_B')
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
    es={e[0]:e[1:] for e in x['edges']};a=x['move'];bd={m:(l,u) for m,l,u in bounds};n,w,_=es[a]
    if n==0:return 0
    superior=any(m!=a and nn>0 and bd[m][1]!=-1 and bd[m][1]>bd[a][0] and ww*n>w*nn for m,(nn,ww,_) in es.items())
    return (1 if w<0 else 3)+int(superior)
def make():
    parent=read(P/'PLAN.json');lock=read(P/'ARENA_LOCK.json');assert sha(P/'PLAN.json')==lock['plan']
    candidates=C.defaultdict(list)
    for meta in lock['files']:
        f=P/meta['file'];assert sha(f)==meta['sha256']
        for gs in raw(f):
            g=gs['H4'];j=g['job']
            for i,r in enumerate(g['rows']):
                if not r.get('early',{}).get('reached'):continue
                x=r['early']['at'];assert x['visits']==4 and r['arm']=='Q' and x['move']==r['move'] and x['rival']!=x['move']
                c=feature(x,r['partial']['bounds'])
                candidates[c].append(dict(family_id=j['id'],seed=g['seed'],history=r['history'],h0=4,color=g['color'],source_file=meta['file'],source_index=i,source_opening=j['opening'],cell=c,snapshot=x,bounds=r['partial']['bounds'],root_nn=r['actual_nn']))
    assert sum(map(len,candidates.values()))==1073 and set(candidates)==set(range(5))
    rng=random.Random(2026112862);roots=[];pop=[]
    for k in range(5):
        c=sorted(candidates[k],key=lambda r:(r['family_id'],r['seed'],r['color'],r['source_index']))
        pop.append(len(c));n=80;assert len(c)>=n
        for r in rng.sample(c,n):roots.append(dict(id=len(roots),**r,arm_order=rng.sample(list(DEV),4),population=len(c),sample=n))
    assert len(roots)==400 and len({(r['family_id'],r['seed'],r['color'],r['source_index']) for r in roots})==400
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
    for panel,N in [('MAIN',32),('ARCHIVE',8)]:
        for g in ('central','whole'):
            cells=[7*r+c for r in range(1,6) for c in range(1,6)] if g=='central' else list(range(49))
            for _ in range(N):
                while True:
                    op=rng.sample(cells,4);f=family(op)
                    if f not in prior:break
                prior.add(f);jobs.append(dict(id=len(jobs),opening=op,family=f,geometry=g,panel=panel,color_order=rng.sample([1,-1],2),arm_order=rng.sample(list(ARMS),3)))
    assert len(jobs)==80 and len(prior)==n+80
    models=parent['models']
    for s in SEEDS:assert sha(models[str(s)]['path'])==models[str(s)]['sha256']
    base=Path.home()/'Dropbox/personal_projects/quartz/results/phase8c_v4_kl_lucb_stop/gomoku7/models/T1_noS_noVL'
    for s in (11,22,33):
        f=base/f'seed_{s}'/'latest.pt';logs=f.parent/'train_log.jsonl';status=f.parent/'checkpoint_status.json';z=read(status)
        tr=[json.loads(x) for x in logs.read_text().splitlines()]
        assert len(tr)>=2 and tr[-1]['train_executed'] and tr[-1]['iter']==2 and z['preferred_posttrain_checkpoint']=='latest.pt'
        digest=sha(f);assert digest not in model_hashes and digest not in {v['sha256'] for v in models.values()}
        back=B/'models'/f'seed_{s+200}.pt';assert not back.exists();shutil.copy2(f,back);assert sha(back)==digest
        models[str(s+200)]=dict(path=str(f),backup=str(back),sha256=digest,original_seed=s,role='archived trained phase8c T1 latest; separate validation panel',training_log_sha256=sha(logs),checkpoint_status_sha256=sha(status),train_last_iter=tr[-1]['iter'],train_executed=True)
    binlock=read(P/'BINARY_LOCK.json');binary=ROOT.parent/'quartz_kinetic61_harm_target_20261128/mcts_demo';assert sha(binary)==binlock['binary_sha256']
    for p,digest in binlock['source_sha256'].items():assert sha(ROOT/p)==digest
    T.mkdir(exist_ok=True)
    pl=dict(jobs=jobs,roots=roots,models=models,parent_lock_sha256=sha(P/'ARENA_LOCK.json'),parent_plan_sha256=sha(P/'PLAN.json'),population=pop,prior_families=n,prior_locks=locks,old_binary=str(binary),old_binary_sha256=binlock['binary_sha256'],old_source_sha256=binlock['source_sha256'],workers=6,train_workers=4,seed=2026112862,bootstrap_seed=202611286200,bootstrap_reps=8000,lambda_num=1,lambda_den=500)
    (D/'PLAN.json').write_text(json.dumps(pl,indent=2,sort_keys=True)+'\n')
    print('population',pop,'roots',len(roots),'heldout families',len(jobs),'prior',n,'locks',len(locks),'distinct archived weights',len(EXTRA))
if __name__=='__main__':make()
