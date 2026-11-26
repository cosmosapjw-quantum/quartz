"""Prospective actual legacy A1 vs exact root certificate and lower-cap controls."""
import os,sys,json,time,hashlib,pathlib,random,statistics as S,gzip,math,subprocess as SP
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2];sys.path.insert(0,str(ROOT))
os.environ.update(QUARTZ_NO_COMPILE='1',QUARTZ_DISABLE_NN_CACHE='1',QUARTZ_DISABLE_ASYNC_PIPELINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from proof import from_history,won,root_certificate,FULL
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
            families.add(f);a,b=from_history(h);assert not won(a) and not won(b)
            jobs.append(dict(seed=seed,id=i,opening=h,family=f,game_seed=rng.randrange(2**63),candidate_order=rng.sample([1,-1],2)))
    models={}
    for seed in SEEDS:
        p=MODEL/f'seed_{seed}/best.pt';models[str(seed)]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,
            champion=json.loads((p.parent/'champion.json').read_text()),train_log_sha256=sha(p.parent/'train_log.jsonl'))
    save('PLAN.json',dict(seed=SEED,models=models,binary=dict(path=str(BIN),sha256=sha(BIN)),arms=ARMS,margin=MARGIN,pairs=PAIRS,workers=WORKERS,
        env={k:os.environ[k] for k in ('QUARTZ_NO_COMPILE','QUARTZ_DISABLE_NN_CACHE','QUARTZ_DISABLE_ASYNC_PIPELINE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},jobs=jobs,scope='fixed checkpoint panel; original Gomoku7 4-in-row, A1 settings; seeded four-ply openings; actual paired color swap; no policy fitting'))
    print('PLAN',len(jobs),'opening pairs per arm',len(jobs)*len(ARMS)*2,'games',flush=True)
def client(seed):
    import torch
    from scripts.controller_sweep import build_base_cfg,build_anchor_candidates,apply_runtime_overrides,_new_search_client
    from quartz.eval_runtime_profile import load_eval_runtime_overrides_from_model
    from quartz.runtime_support import build_search_manifest,search_manifest_hash
    p=MODEL/f'seed_{seed}/best.pt';lk=load('PLAN.json');assert sha(p)==lk['models'][str(seed)]['sha256'] and sha(BIN)==lk['binary']['sha256']
    b,device=build_base_cfg('gomoku7','cpu',32);anchor=build_anchor_candidates(b)[0];assert anchor['id']=='A1_legacy_base'
    cfg=apply_runtime_overrides(b,anchor['overrides']);over=load_eval_runtime_overrides_from_model(str(p),str(device));cfg.update(over or {})
    cfg['iters']=32
    assert (cfg['board'],cfg['win'],cfg['actions'])==(7,4,49) and cfg['penalty_mode']=='GatedRefreshLegacy'
    c=_new_search_client(str(p),cfg,device,str(BIN));ct=[]
    h=c.model.register_forward_pre_hook(lambda m,x:ct.append(int(x[0].shape[0])))
    meta=dict(seed=seed,checkpoint=lk['models'][str(seed)],anchor=anchor,runtime_overrides=over,manifest=build_search_manifest(cfg),manifest_hash=search_manifest_hash(cfg),torch=torch.__version__,model_parameters=sum(x.numel() for x in c.model.parameters()))
    return c,h,ct,meta
def native(c,ct,h,cap,seed):
    import numpy as np
    board=np.zeros(49,dtype=np.int8)
    for i,m in enumerate(h):board[m]=1 if i%2==0 else -1
    side=1 if len(h)%2==0 else -1;ct.clear();c.cfg['iters']=cap;c.cfg['seed']=seed
    t=time.monotonic_ns();r=c.search_move(board,side,penalty_mode=c.cfg['penalty_mode']);wall=time.monotonic_ns()-t
    assert isinstance(r,dict) and 'error' not in r and 'best_move' in r,r
    m=int(r['best_move']);assert 0<=m<49 and board[m]==0,(h,r)
    b=r['realized_budget'];man=r['search_manifest'];nn=b['evaluator_calls'];assert isinstance(nn,int) and nn>=0
    assert man['benchmark_safe'] and man['profile']=='quartz' and man['requested_iteration_limit']==cap
    return dict(move=m,kind='NATIVE',NN=nn,physical=sum(ct),batches=list(ct),wall_ns=wall,budget=b,manifest=man,policy=r['policy'])
def decide(c,ct,h,arm,seed):
    proof=None;t=time.monotonic_ns()
    if arm.startswith('proof'):
        a,b=from_history(h)
        if len(h)%2:a,b=b,a
        proof=root_certificate(a,b)
    rule_ns=time.monotonic_ns()-t
    if proof is not None and proof['move'] is not None:
        return dict(move=proof['move'],kind='CERTIFICATE',NN=0,physical=0,batches=[],wall_ns=rule_ns,rule_ns=rule_ns,proof=proof,budget=None,manifest=None)
    cap=20 if arm.endswith('20') else 32
    x=native(c,ct,h,cap,seed);x.update(rule_ns=rule_ns,proof=proof,wall_ns=x['wall_ns']+rule_ns);return x
def game(c,ct,j,arm,side):
    h=list(j['opening']);rows=[]
    while True:
        a,b=from_history(h)
        if won(a) or won(b) or a|b==FULL:break
        p=1 if len(h)%2==0 else -1;who=arm if p==side else 'base32'
        seed=(j['game_seed']+len(h)*0x9e3779b97f4a7c15)&((1<<64)-1)
        x=decide(c,ct,h,who,seed);x.update(player=p,arm=who,ply=len(h),seed=seed)
        assert x['move'] not in h;h.append(x['move']);rows.append(x)
    a,b=from_history(h);winner=1 if won(a) else -1 if won(b) else 0
    assert not (won(a) and won(b)) and (winner or a|b==FULL)
    return dict(job=j,arm=arm,candidate_side=side,history=h,winner=winner,score=(1+side*winner)/2,rows=rows)
_INIT=False
def worker(arg):
    global _INIT
    import torch
    if not _INIT:torch.set_num_threads(1);torch.set_num_interop_threads(1);_INIT=True
    seed,arm,shard,rr=arg
    name=f'GAMES_{seed}_{arm}_{shard}.jsonl.gz';p=D/name;assert not p.exists()
    tmp=p.with_suffix('.tmp');n=0;ct0=Counter()
    c,h,ct,meta=client(seed)
    try:
        with gzip.open(tmp,'wt',compresslevel=3) as fd:
            for j in rr:
                for side in j['candidate_order']:
                    row=game(c,ct,j,arm,side);fd.write(json.dumps(row,separators=(',',':'))+'\n');fd.flush();n+=1
                    ct0.update(games=1,moves=len(row['rows']),NN=sum(x['NN'] for x in row['rows']),physical=sum(x['physical'] for x in row['rows']))
                print('PAIR',seed,arm,shard,j['id'],n,dict(ct0),flush=True)
    finally:h.remove();c.stop()
    tmp.replace(p);return dict(file=name,sha256=sha(p),meta=meta,**ct0)
def collect():
    lk=load('PLAN.json');jobs=lk['jobs'];tasks=[]
    for s in SEEDS:
        rr=[j for j in jobs if j['seed']==s]
        for arm in ARMS:
            for q in range(2):tasks.append((s,arm,q,rr[q::2]))
    random.Random(SEED+1).shuffle(tasks)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:fs=list(ex.map(worker,tasks))
    save('GAMES_LOCK.json',dict(plan=sha(D/'PLAN.json'),code=sha(D/'arena.py'),proof=sha(D/'proof.py'),files=fs,**{k:sum(z[k] for z in fs) for k in ('games','moves','NN','physical')}))
    print('COLLECT',load('GAMES_LOCK.json')['games'],load('GAMES_LOCK.json')['NN'],flush=True)
def summary():
    lk=load('GAMES_LOCK.json');assert lk['plan']==sha(D/'PLAN.json')
    games=defaultdict(list)
    for fi in lk['files']:
        p=D/fi['file'];assert sha(p)==fi['sha256']
        for g in lines(p):games[g['arm']].append(g)
    out={}
    for arm,gg in games.items():
        groups=defaultdict(list);ct=Counter();natives=Counter();side=Counter()
        for g in gg:
            groups[(g['job']['seed'],g['job']['id'])].append(g)
            side[(g['candidate_side'],g['score'])]+=1
            for z in g['rows']:
                tag='candidate' if z['player']==g['candidate_side'] else 'base'
                for k,v in [('moves',1),('NN',z['NN']),('physical',z['physical']),('wall_ns',z['wall_ns']),('rule_ns',z['rule_ns']),('cert',int(z['kind']=='CERTIFICATE'))]:ct[tag+'_'+k]+=v
                if z['proof']:natives[z['proof']['kind']]+=1
        assert len(groups)==len(SEEDS)*PAIRS and all(len(v)==2 and {g['candidate_side'] for g in v}=={1,-1} for v in groups.values())
        pair={k:S.mean(g['score'] for g in v) for k,v in groups.items()}
        def costs(gg):
            c=Counter()
            for g in gg:
                for x in g['rows']:
                    tag='candidate' if x['player']==g['candidate_side'] else 'base'
                    c[tag+'_moves']+=1;c[tag+'_NN']+=x['NN'];c[tag+'_physical']+=x['physical']
            return c
        def saving(c,k):
            return 1-(c['candidate_'+k]/c['candidate_moves'])/(c['base_'+k]/c['base_moves'])
        by={s:[k for k in groups if k[0]==s] for s in SEEDS}
        rng=random.Random(SEED+321);bt=[];bc=[];bp=[]
        pc={k:costs(v) for k,v in groups.items()}
        for _ in range(4000):
            ids=[k for s in SEEDS for k in rng.choices(by[s],k=len(by[s]))]
            bt.append(S.mean(pair[k] for k in ids));b=Counter()
            for k in ids:b.update(pc[k])
            bc.append(saving(b,'NN'));bp.append(saving(b,'physical'))
        ci=lambda a:[sorted(a)[99],sorted(a)[3899]]
        low=sorted(bt)[199];save0=saving(ct,'NN');save1=saving(ct,'physical')
        out[arm]=dict(games=len(gg),pairs=len(groups),wins=sum(g['score']==1 for g in gg),draws=sum(g['score']==.5 for g in gg),losses=sum(g['score']==0 for g in gg),
            score=S.mean(pair.values()),score_ci=ci(bt),NI_lower=low,NI_margin=MARGIN,NI_point_gate=low>=.5-MARGIN,NN_reduction=save0,NN_reduction_ci=ci(bc),physical_reduction=save1,physical_reduction_ci=ci(bp),
            count=dict(ct),certificate_kinds=dict(natives),seed_scores={str(s):S.mean(pair[k] for k in by[s]) for s in SEEDS},side_scores={str(p):S.mean(g['score'] for g in gg if g['candidate_side']==p) for p in (1,-1)})
    save('SUMMARY.json',dict(arms=out,scope='stratified paired opening bootstrap, fixed checkpoint panel, descriptive pilot; not independently established mature strength or full research-grade P2'))
    print(json.dumps(out,sort_keys=True),flush=True)
def smoke():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    c,h,ct,meta=client(101)
    try:
        history=[23,0,25,48];p=root_certificate(*from_history(history));assert p['kind']=='WIN3'
        old=native(c,ct,history,32,SEED);new=decide(c,ct,history,'proof32',SEED)
        assert old['NN']>0 and new['NN']==new['physical']==0 and new['kind']=='CERTIFICATE'
        assert new['move']==p['move']
        print('GREEN',json.dumps(dict(old=old,new=new,meta=meta)),flush=True)
        h0=[24,17,25,31];r=decide(c,ct,h0,'blind20',SEED);assert r['NN']>0 and r['physical']>0
        print('NATIVE',r['NN'],r['physical'],r['budget'],r['manifest'],flush=True)
    finally:h.remove();c.stop()
if __name__=='__main__':globals()[sys.argv[1]]()
