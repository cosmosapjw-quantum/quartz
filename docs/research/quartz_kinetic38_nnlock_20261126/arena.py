"""Prospective same-checkpoint actual full games; independent workers."""
import os,sys,pathlib,json,gzip,random,statistics as S,time,math
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from plan import D,ROOT,SEEDS,PAIRS,WORKERS,ARMS,load,save,sha
from engine import Puct,load_eval,from_history,won,FULL
assert D.name=="quartz_kinetic38_nnlock_20261126" and ARMS==("L32","L20"),(D,ARMS)
def game(nn,j,arm,side):
    assert arm in ARMS and side in (-1,1)
    h=list(j['opening']);rows=[]
    while True:
        a,b=from_history(h)
        if won(a) or won(b) or a|b==FULL:break
        p=1-2*(len(h)%2);who=arm if p==side else 'B32'
        x=Puct(nn).search(h,20 if who=='L20' else 32,who!='B32')
        x.update(player=p,arm=who,ply=len(h));assert x['move'] not in h
        h.append(x['move']);rows.append(x)
    winner=1 if won(a) else -1 if won(b) else 0
    assert not(won(a) and won(b)) and (winner or a|b==FULL)
    return dict(job=j,arm=arm,candidate_side=side,history=h,winner=winner,score=(1+side*winner)/2,rows=rows)
def worker(arg):
    seed,shard,jobs=arg;assert all(j in load("PLAN.json")["jobs"] and j["seed"]==seed for j in jobs);nn=load_eval(seed)
    name=f'GAMES_{seed}_{shard}.jsonl.gz';p=D/name;assert not p.exists();tmp=p.with_suffix('.tmp')
    counts=Counter();t=time.perf_counter();c=time.process_time()
    with gzip.open(tmp,'wt',compresslevel=3) as f:
        for j in jobs:
            for arm in ARMS:
                for side in j['color_order']:
                    g=game(nn,j,arm,side)
                    f.write(json.dumps(g,separators=(',',':'),allow_nan=False)+'\n')
                    counts.update(games=1,moves=len(g['rows']),nn=sum(x['nn'] for x in g['rows']))
            f.flush()
            print('PAIR',seed,shard,j['id'],dict(counts),flush=True)
    tmp.replace(p);assert counts['nn']==nn.rows and counts['games']==4*len(jobs)
    return dict(file=name,sha256=sha(p),seed=seed,shard=shard,wall_s=time.perf_counter()-t,cpu_s=time.process_time()-c,**counts)
def collect():
    lk=load('PLAN.json');jobs=lk['jobs'];tasks=[]
    for s in SEEDS:
        rr=[j for j in jobs if j['seed']==s]
        for k in range(4):tasks.append((s,k,rr[k::4]))
    random.Random(lk['seed']+1).shuffle(tasks);t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn')) as ex:
        fs=list(ex.map(worker,tasks))
    save('GAMES_LOCK.json',dict(plan=sha(D/'PLAN.json'),engine=sha(D/'engine.py'),arena=sha(D/'arena.py'),files=fs,elapsed_s=time.perf_counter()-t,**{k:sum(f[k] for f in fs) for k in ('games','moves','nn')}))
    print('COLLECT',load('GAMES_LOCK.json'),flush=True)
def probe(j):
    nn=load_eval(j['seed'])
    gs=[game(nn,j,a,s) for a in ARMS for s in (-1,1)]
    assert len(gs)==4 and gs[0]['history']==gs[1]['history']
    assert nn.rows==sum(r['nn'] for g in gs for r in g['rows'])
    return dict(path=str(D),arms=list(ARMS),games=len(gs),nn=nn.rows)
def smoke():
    with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn')) as ex:
        result=ex.submit(probe,load('PLAN.json')['jobs'][0]).result()
    assert result['path']==str(D) and result['arms']==list(ARMS)
    print('SPAWN_SMOKE',result,flush=True)
def raw(p):
    with gzip.open(p,'rt') as f:
        for l in f:yield json.loads(l)
def summary():
    lock=load('GAMES_LOCK.json');groups=defaultdict(list)
    for fi in lock['files']:
        assert sha(D/fi['file'])==fi['sha256']
        for g in raw(D/fi['file']):groups[g['arm']].append(g)
    out={}
    for arm,gg in groups.items():
        pairs=defaultdict(list)
        for g in gg:pairs[g['job']['seed'],g['job']['id']].append(g)
        assert len(pairs)==len(SEEDS)*PAIRS and all(len(v)==2 and {g['candidate_side'] for g in v}=={-1,1} for v in pairs.values())
        ps={k:S.mean(g['score'] for g in v) for k,v in pairs.items()}
        def count(games):
            c=Counter()
            for g in games:
                for r in g['rows']:
                    x='L' if r['player']==g['candidate_side'] else 'B'
                    for k,v in [('moves',1),('nn',r['nn']),('visits',r['visits']),('cpu_ns',r['cpu_ns']),('wall_ns',r['wall_ns']),('rule_ns',r['rule_ns']),('cert',int(r['kind']=='CERTIFICATE')),('lock',int(r['kind']=='VOTE_LOCK'))]:c[x+'_'+k]+=v
            return c
        def saving(c,k):return 1-(c['L_'+k]/c['L_moves'])/(c['B_'+k]/c['B_moves'])
        pc={k:count(v) for k,v in pairs.items()};ct=count(gg)
        by={s:[k for k in pairs if k[0]==s] for s in SEEDS};rng=random.Random(2026112638+99)
        bs=[];bn=[];bc=[];bw=[]
        for _ in range(4000):
            ks=[k for s in SEEDS for k in rng.choices(by[s],k=len(by[s]))]
            bs.append(S.mean(ps[k] for k in ks));c=Counter()
            for k in ks:c.update(pc[k])
            bn.append(saving(c,'nn'));bc.append(saving(c,'cpu_ns'));bw.append(saving(c,'wall_ns'))
        ci=lambda x:[sorted(x)[99],sorted(x)[3899]]
        low=sorted(bs)[199]
        out[arm]=dict(games=len(gg),pairs=len(pairs),wins=sum(g['score']==1 for g in gg),draws=sum(g['score']==.5 for g in gg),losses=sum(g['score']==0 for g in gg),score=S.mean(ps.values()),score_ci=ci(bs),NI_lower=low,NN_saving=saving(ct,'nn'),NN_ci=ci(bn),CPU_saving=saving(ct,'cpu_ns'),CPU_ci=ci(bc),wall_saving=saving(ct,'wall_ns'),wall_ci=ci(bw),joint=low>=.45 and saving(ct,'nn')>=.30,count=dict(ct),same_color_swap_histories=sum(v[0]['history']==v[1]['history'] for v in pairs.values()),seed_scores={str(s):S.mean(ps[k] for k in by[s]) for s in SEEDS})
    save('SUMMARY.json',dict(arms=out,scope='fixed checkpoint panel, stratified pair bootstrap, descriptive CPU/wall under host contention; no A1 claim'))
    print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
