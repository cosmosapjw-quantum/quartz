"""Exact late-game regret, frozen diagnostic LP, real same-checkpoint games."""
import collections as C, gzip,json,multiprocessing as mp,random,sys,time,statistics as S,importlib.util
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as F
from k41_plan import D,ROOT,SEEDS,FIT,H,WORKERS,SEED,BUDGET,rule,family,load,save,sha
from k41_engine import Puct,load_eval,from_history,won,FULL
from exact41 import solve
sp=importlib.util.spec_from_file_location('_k40_budget',D.parent/'quartz_kinetic40_budget_20261126/budget.py')
budget=importlib.util.module_from_spec(sp);sp.loader.exec_module(budget)
assert D.name=='quartz_kinetic41_exact_20261126'
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def emit(f,x):f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n')
def end(h):
    a,b=from_history(h)
    return 1 if won(a) else -1 if won(b) else 0 if a|b==FULL else None
def side(h):return 1-2*(len(h)%2)
def exact(j):
    a,b=from_history(j['opening'])
    if side(j['opening'])<0:a,b=b,a
    t=time.perf_counter_ns();c=time.process_time_ns();r=solve(a,b,BUDGET)
    return dict(job=j,exact=r,cpu_ns=time.process_time_ns()-c,wall_ns=time.perf_counter_ns()-t)
def game(e,j,color):
    h=j['opening'].copy();rows=[]
    while end(h) is None:
        p=side(h);r=e.quartz41(h) if p==color else e.search(h,32,False)
        r.update(arm='Q41' if p==color else 'B32',player=p,ply=len(h))
        rows.append(r);h.append(r['move'])
    w=end(h)
    return dict(job=j,color=color,history=h,winner=w,score2=1+color*w,rows=rows,nn=sum(r['nn'] for r in rows))
def worker(arg):
    mode,seed,i,items=arg;t=time.perf_counter();ct=C.Counter()
    out=D/f'{mode}_{seed}_{i}.jsonl.gz';assert not out.exists();tmp=out.with_suffix('.tmp')
    e=Puct(load_eval(seed)) if mode.startswith('ATLAS') or mode=='ARENA' else None
    with gzip.open(tmp,'wt',compresslevel=3) as f:
        for x in items:
            if mode.startswith('EXACT'):
                r=exact(x);ct['complete']+=r['exact']['complete'];ct['censored']+=not r['exact']['complete'];ct['solver_nodes']+=r['exact']['nodes'];ct['solver_cpu_ns']+=r['cpu_ns']
            elif mode.startswith('ATLAS'):
                j=x['job'];v=x['exact'];assert v['complete']
                a=e.atlas41(j['opening']);vv={k:v['values'][str(a[k]['move'])] for k in ('B32',*(f'L{h}' for h in H))}
                r=dict(job=j,seed=seed,atlas=a,values=vv,best_value=v['best_value'],nn=a['actual_nn'])
                ct['nn']+=r['nn'];ct['suboptimal20']+=vv['L20']!=v['best_value'];ct['suboptimal32']+=vv['L32']!=v['best_value'];ct['suboptimal64']+=vv['L64']!=v['best_value']
            else:
                assert mode=='ARENA'
                for color in x['color_order']:
                    g=game(e,x,color);g['seed']=seed;emit(f,g);ct['games']+=1;ct['moves']+=len(g['rows']);ct['nn']+=g['nn']
                    for z in g['rows']:
                        ct['exact_attempts']+=z.get('exact_attempt',False);ct['exact_complete']+=z.get('exact_complete',False);ct['solver_nodes']+=z.get('exact_nodes',0);ct['solver_cpu_ns']+=z.get('exact_cpu_ns',0)
                ct['items']+=1;continue
            emit(f,r);ct['items']+=1
    if e is not None:assert e.ev.rows==ct['nn']
    tmp.replace(out);r=dict(file=out.name,sha256=sha(out),seed=seed,shard=i,counts=dict(ct),elapsed_s=time.perf_counter()-t)
    print('SHARD',mode,seed,i,dict(ct),flush=True);return r
def execute(mode,groups):
    tasks=[];t=time.perf_counter()
    for s,items in groups.items():
        for i in range(8 if mode.startswith('EXACT') else 4):
            n=8 if mode.startswith('EXACT') else 4
            if items[i::n]:tasks.append((mode,s,i,items[i::n]))
    random.Random(SEED+len(mode)).shuffle(tasks)
    with ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(worker,tasks))
    ct=C.Counter()
    for f in fs:ct.update(f['counts'])
    save(mode+'_LOCK.json',dict(plan=sha(D/'PLAN.json'),solver=sha(D/'exact41.py'),engine=sha(D/'k41_engine.py'),experiment=sha(D/'k41_experiment.py'),files=fs,counts=dict(ct),elapsed_s=time.perf_counter()-t))
    print(mode,dict(ct),flush=True)
def rows(mode):
    lk=load(mode+'_LOCK.json')
    for f in lk['files']:
        assert sha(D/f['file'])==f['sha256'];yield from raw(D/f['file'])
def phase(split):
    mode='EXACT_'+split
    execute(mode,{0:[j for j in load('PLAN.json')['jobs'] if j['split']==split]})
    rr=sorted([r for r in rows(mode) if r['exact']['complete']],key=lambda x:x['job']['id'])
    seeds=FIT if split=='TRAIN' else SEEDS
    execute('ATLAS_'+split,{s:rr for s in seeds})
def key(r,cuts):
    return sum((1<<i)*int(x>t) for i,(x,t) in enumerate(zip(r['atlas']['features'][:3],cuts)))
def gain4(r,h):return 2*(r['values'][f'L{h}']-r['values']['L20'])
def extra(r,h):return r['atlas'][f'L{h}']['nn']-r['atlas']['L20']['nn']
def points(rr,cuts):
    p={}
    for r in rr:
        z=key(r,cuts);a=p.setdefault(z,{h:[0,0] for h in (20,26,32)})
        for h in a:a[h][0]+=extra(r,h);a[h][1]+=gain4(r,h)
    return p
def fit():
    rr=list(rows('ATLAS_TRAIN'));assert rr and all(r['seed'] in FIT for r in rr)
    cuts=[S.median(r['atlas']['features'][i] for r in rr) for i in range(3)]
    pts=points(rr,cuts);B=F(7,10)*sum(r['atlas']['B32']['nn'] for r in rr)-sum(r['atlas']['L20']['nn'] for r in rr)
    s=budget.solve(pts,B)
    p={str(z):{str(h):str(q) for h,q in hh.items()} for z,hh in s['prob'].items()}
    out=dict(cuts=cuts,policy=p,budget=str(B),feasible=s['feasible'],extra=str(s['cost']),gain4=str(s['gain']),dual_score_per_row=None if s['dual'] is None else str(s['dual']/4),dual_gain4=None if s['dual_bound'] is None else str(s['dual_bound']),train_points=pts,train_lock=sha(D/'ATLAS_TRAIN_LOCK.json'),train_seeds=list(FIT),note='Frozen before CAL/TEST; diagnostic expected randomized policy only, not deployed to arena; unseen bins L20')
    save('FROZEN.json',out);print('FROZEN',json.dumps(out,separators=(',',':')),flush=True)
def describe(rr,f):
    import numpy as np
    assert rr
    cuts=f['cuts'];pol=f['policy'];by=C.defaultdict(list);ct=C.Counter();cost=C.Counter();reg=C.Counter();bad=C.Counter();gains=C.Counter();flips=C.Counter()
    for r in rr:
        a=r['atlas'];v=r['values'];b=r['best_value'];z=key(r,cuts);ct[z]+=1
        for k in ('B32',*(f'L{h}' for h in H)):
            cost[k]+=a[k]['nn'];reg[k]+=b-v[k];bad[k]+=v[k]!=b
            gains[k]+=v[k]-v['L20'];flips[k]+=a[k]['move']!=a['L20']['move']
        hh=pol.get(str(z),{'20':'1'})
        g=sum(F(q)*gain4(r,int(h))/4 for h,q in hh.items())
        k=sum(F(q)*extra(r,int(h)) for h,q in hh.items())
        by[r['job']['id']].append((float(g),float(k),a['L20']['nn'],a['B32']['nn']))
    n=len(rr);vals=np.asarray([[sum(v[i] for v in rows) for i in range(4)]+[len(rows)] for _,rows in sorted(by.items())])
    rng=np.random.default_rng(SEED+19);boot=vals[rng.integers(0,len(vals),(4000,len(vals)))].sum(axis=1);ss=vals.sum(axis=0)
    gain=np.sort(boot[:,0]/boot[:,4]);red=np.sort(1-(boot[:,1]+boot[:,2])/boot[:,3])
    return dict(rows=n,families=len(by),bin_counts=dict(ct),nn=dict(cost),bad=dict(bad),mean_regret={k:v/(2*n) for k,v in reg.items()},mean_gain={k:v/(2*n) for k,v in gains.items()},flip=dict(flips),expected_gain=float(ss[0]/n),expected_reduction=float(1-(ss[1]+ss[2])/ss[3]),gain_lower=float(gain[199]),gain_CI=[float(gain[99]),float(gain[3899])],reduction_CI=[float(red[99]),float(red[3899])])
def summarize():
    import numpy as np
    f=load('FROZEN.json');out={}
    for split in ('TRAIN','CAL','TEST'):
        rr=list(rows('ATLAS_'+split));out[split]=dict(all=describe(rr,f),fit=describe([r for r in rr if r['seed'] in FIT],f),by_seed={str(s):describe([r for r in rr if r['seed']==s],f) for s in SEEDS if any(r['seed']==s for r in rr)})
    cal=out['CAL']['fit'];admitted=bool(f['feasible'] and cal['gain_lower']>0 and cal['expected_reduction']>=.3)
    exact={}
    for split in ('TRAIN','CAL','TEST'):
        rr=list(rows('EXACT_'+split));c=C.Counter();vals=C.Counter();n=0;cpu=0;mx=0
        for r in rr:
            z=r['exact'];c['complete']+=z['complete'];c['censored']+=not z['complete'];n+=z['nodes'];cpu+=r['cpu_ns'];mx=max(mx,z['nodes'])
            if z['complete']:
                vals[z['best_value']]+=1;c['mixed_action_values']+=len(set(z['values'].values()))>1;c['actions']+=len(z['values'])
        exact[split]=dict(counts=dict(c),value_counts=dict(vals),nodes=n,max_nodes=mx,cpu_ns=cpu)
    gs=list(rows('ARENA'));arms={}
    for split in ('EARLY','LATE'):
        q=[g for g in gs if g['job']['split']==split];pairs=C.defaultdict(list)
        for g in q:pairs[g['job']['id']].append(g)
        assert len(pairs)==64 and all(len(x)==8 for x in pairs.values())
        def stat(gg):
            v=[0]*16
            for g in gg:
                v[0]+=g['score2'];v[1]+=1;v[2]+=g['score2']==2;v[3]+=g['score2']==1;v[4]+=g['score2']==0
                for r in g['rows']:
                    i=5 if r['player']==g['color'] else 9
                    v[i]+=1;v[i+1]+=r['nn'];v[i+2]+=r['cpu_ns'];v[i+3]+=r['wall_ns']
                    v[13]+=r.get('exact_attempt',False);v[14]+=r.get('exact_complete',False);v[15]+=r.get('exact_nodes',0)
            return v
        def rec(v):
            red=lambda i:1-(v[5+i]/v[5])/(v[9+i]/v[9]) if v[5] and v[9] and v[9+i] else None
            return dict(games=int(v[1]),wins=int(v[2]),draws=int(v[3]),losses=int(v[4]),score=float(v[0]/(2*v[1])),candidate_moves=int(v[5]),candidate_nn=int(v[6]),baseline_moves=int(v[9]),baseline_nn=int(v[10]),nn_reduction=red(1),cpu_reduction=red(2),wall_reduction=red(3),exact_attempts=int(v[13]),exact_complete=int(v[14]),exact_nodes=int(v[15]))
        a=np.asarray([stat(v) for _,v in sorted(pairs.items())],dtype=float);rng=np.random.default_rng(SEED+43);b=a[rng.integers(0,len(a),(4000,len(a)))].sum(axis=1)
        scores=np.sort(b[:,0]/(2*b[:,1]));valid=(b[:,5]>0)&(b[:,9]>0)&(b[:,10]>0);red=np.sort(1-(b[valid,6]/b[valid,5])/(b[valid,10]/b[valid,9]))
        r=rec(stat(q));r.update(NI_lower=float(scores[199]),score_CI=[float(scores[99]),float(scores[3899])],NN_CI=[float(np.quantile(red,.025)),float(np.quantile(red,.975))] if len(red) else None,by_seed={str(s):rec(stat([g for g in q if g['seed']==s])) for s in SEEDS})
        r['exact_pairs']=sum(v[0]['history']==v[1]['history'] for _,vv in pairs.items() for s in SEEDS for v in [[g for g in vv if g['seed']==s]])
        arms[split]=r
    x=dict(admitted=admitted,exact=exact,atlas=out,arena=arms,counts={m:load(m+'_LOCK.json')['counts'] for m in ('EXACT_TRAIN','ATLAS_TRAIN','EXACT_CAL','ATLAS_CAL','EXACT_TEST','ATLAS_TEST','ARENA')},scope='selected quiet late-state frame; complete-case exact corpus with explicit censoring; diagnostic policy not deployed; same-evaluator Q41/B32 arena')
    save('SUMMARY.json',x);print('SUMMARY',json.dumps(x,separators=(',',':')),flush=True)
if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd in ('TRAIN','CAL','TEST'):phase(cmd)
    elif cmd=='arena':execute('ARENA',{s:[j for j in load('PLAN.json')['jobs'] if j['split'] in ('EARLY','LATE')] for s in SEEDS})
    else:globals()[cmd]()
