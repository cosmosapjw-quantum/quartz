from pathlib import Path
import json,gzip,hashlib,random,collections as C,shutil
D=Path(__file__).resolve().parent; ROOT=D.parents[2]; T=ROOT.parent/'quartz_origin66_omission_run_20261128'
P=D.parent/'quartz_origin65_allocation_20261128'; P58=D.parent/'quartz_kinetic58_stop_20261128'
B=Path.home()/'Dropbox/personal_projects/quartz_origin66_omission_backups'
SEEDS=(101,102,103,105,41,42,43); EXTRA=(211,222,233)
NOVEL=(311,322,333); ALL=SEEDS+EXTRA+NOVEL
ARMS=('H4','FREE','MA4','FOCUS','EXPAND'); DEV=('MA4','FOCUS','EXPAND')
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
    parent=read(P/'PLAN.json');rng=random.Random(2026112866);roots=[];pop={};locks0={}
    for panel,limit in [('MAIN',32),('ARCHIVE',16),('NOVEL',12)]:
        lock=read(P/(panel+'_LOCK.json'));assert lock['plan']==sha(P/'PLAN.json')
        locks0[panel]=sha(P/(panel+'_LOCK.json'));cs=C.defaultdict(list)
        for meta in lock['files']:
            f=P/meta['file'];assert sha(f)==meta['sha256']
            for gs in raw(f):
                g=gs['MA4'];j=g['job']
                for i,r in enumerate(g['rows']):
                    if 'allocation65' not in r:continue
                    x=r['allocation65'];bd=r['partial']['bounds'];safe=set(x['safe'])
                    assert r['arm']=='Q' and r['visits']==4 and r['move']==x['chosen']
                    cell=str(int(any(u==-1 for m,l,u in bd)))+str(int(any(m in safe and n==0 for m,n,w,pi in x['edges'])))
                    cs[cell].append(dict(domain=panel,cell=cell,family_id=j['id'],seed=g['seed'],color=g['color'],history=r['history'],source_file=meta['file'],source_index=i,source_opening=j['opening'],snapshot=x,bounds=bd,root_nn=r['actual_nn'],incumbent=r['move']))
        pop[panel]={k:len(v) for k,v in sorted(cs.items())}
        for k,rs in sorted(cs.items()):
            rs.sort(key=lambda z:(z['family_id'],z['seed'],z['color'],z['source_index']))
            for r in rng.sample(rs,min(limit,len(rs))):
                roots.append(dict(id=len(roots),**r,population=len(rs),sample=min(limit,len(rs)),action_order=rng.sample([x[0] for x in r['bounds']],len(r['bounds'])),operation_order=rng.sample(['FOCUS','EXPAND'],2)))
    assert pop=={'MAIN':{'01':438,'10':47,'11':73},'ARCHIVE':{'01':91,'10':11,'11':12},'NOVEL':{'01':59,'10':15,'11':13}}
    assert len(roots)==171 and len({(r['domain'],r['family_id'],r['seed'],r['color'],r['source_index']) for r in roots})==171
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
    for panel,N in [('MAIN',12),('ARCHIVE',6),('NOVEL',6)]:
        for g in ('central','whole'):
            cells=[7*r+c for r in range(1,6) for c in range(1,6)] if g=='central' else list(range(49))
            for _ in range(N):
                while True:
                    op=rng.sample(cells,4);f=family(op)
                    if f not in prior:break
                prior.add(f);jobs.append(dict(id=len(jobs),opening=op,family=f,geometry=g,panel=panel,color_order=rng.sample([1,-1],2),arm_order=rng.sample(list(ARMS),5)))
    assert len(jobs)==48 and len(prior)==n+48
    models=parent['models']
    for ss in ALL:
        m=models[str(ss)];assert sha(m['path'])==m['sha256']
        if ss in EXTRA+NOVEL:assert sha(m['backup'])==m['sha256']
    binlock=read(P/'BINARY_LOCK.json');binary=ROOT.parent/'quartz_origin65_allocation_run_20261128/mcts_demo';assert sha(binary)==binlock['binary_sha256']
    for p,digest in binlock['source_sha256'].items():assert sha(ROOT/p)==digest
    T.mkdir(exist_ok=True)
    pl=dict(jobs=jobs,roots=roots,models=models,parent_lock_sha256=locks0,parent_plan_sha256=sha(P/'PLAN.json'),population=pop,prior_families=n,prior_locks=locks,old_binary=str(binary),old_binary_sha256=binlock['binary_sha256'],old_source_sha256=binlock['source_sha256'],workers=6,train_workers=6,seed=2026112866,bootstrap_seed=202611286600,bootstrap_reps=8000,lambda_num=1,lambda_den=500)
    (D/'PLAN.json').write_text(json.dumps(pl,indent=2,sort_keys=True)+'\n')
    print('population',pop,'roots',len(roots),'heldout families',len(jobs),'prior',n,'locks',len(locks),'novel weights',len(NOVEL))
if __name__=='__main__':make()
