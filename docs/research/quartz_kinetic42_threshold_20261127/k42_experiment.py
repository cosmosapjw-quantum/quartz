"""Cold same-state benchmark and actual same-checkpoint paired games."""
import collections as C,gzip,json,multiprocessing as mp,random,sys,time,importlib.util,pathlib,subprocess
from concurrent.futures import ProcessPoolExecutor
from k42_plan import D,ROOT,SEEDS,WORKERS,SEED,BUDGET,rule,load,save,sha
from threshold42 import solve,T
from k42_engine import Puct,load_eval,from_history,won,FULL
OLD=D.parent/'quartz_kinetic41_exact_20261126'
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def emit(f,x):f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n')
def orient(h):
    a,b=from_history(h);return (a,b) if len(h)%2==0 else (b,a)
def timed(fn):
    t=time.perf_counter_ns();c=time.process_time_ns();z=fn()
    return dict(result=z,cpu_ns=time.process_time_ns()-c,wall_ns=time.perf_counter_ns()-t)
def old_solver():
    for name in ('k41_plan','exact41'):
        s=importlib.util.spec_from_file_location(name,OLD/(name+'.py'))
        m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m)
    return sys.modules['exact41'].solve
def benchmark():
    old=old_solver();pl=load('PLAN.json');jobs=[dict(j,frame='FRESH') for j in pl['jobs'] if j['split']=='ATLAS']
    prev=json.loads((OLD/'PLAN.json').read_text());jobs += [dict(j,frame='K41') for j in prev['jobs'] if j['split'] in ('TRAIN','CAL','TEST')]
    oracle={(r['a'],r['b']):r['values'] for r in raw(OLD/'INDEPENDENT.jsonl.gz') if 'corpus_complete' in r['tags'] or 'corpus_censored' in r['tags']}
    assert len(jobs)==384 and len(oracle)==240
    random.Random(SEED).shuffle(jobs)
    p=D/'BENCHMARK.jsonl.gz';assert not p.exists()
    ct=C.Counter()
    with gzip.open(p,'wt',compresslevel=3) as f:
        for i,j in enumerate(jobs):
            a,b=orient(j['opening']);ops={'action':lambda:solve(a,b,BUDGET),'census':lambda:solve(a,b,BUDGET,True)}
            if j['frame']=='FRESH' and j['empties']<=12:ops['python']=lambda:old(a,b,25000)
            order=list(ops);order=order if i%2==0 else order[::-1]
            rs={k:timed(ops[k]) for k in order};q=rs['action']['result'];z=rs['census']['result']
            if q['complete'] and z['complete']:assert q['best']==z['best'] and q['best_value']==z['best_value']
            truth=oracle[(a,b)] if j['frame']=='K41' else None
            if truth is not None:
                if q['complete']:assert q['best']==min(int(m) for m,v in truth.items() if v==max(truth.values())) and all(x[0]<=truth[m]<=x[1] for m,x in q['bounds'].items())
                if z['complete']:assert z['values']==truth
            emit(f,dict(job=j,a=a,b=b,order=order,runs=rs,truth=truth))
            ct[j['frame']]+=1
            for k,r in rs.items():ct[k+'_complete']+=r['result']['complete'];ct[k+'_nodes']+=r['result']['nodes']
            if (i+1)%64==0:print('BENCH',i+1,dict(ct),flush=True)
    save('BENCHMARK_LOCK.json',dict(file=p.name,sha256=sha(p),plan=sha(D/'PLAN.json'),source=sha(D/'threshold42.cpp'),binary=sha(T/'threshold42.so'),counts=dict(ct),scope='single process serial cold calls; actual process CPU and elapsed wall; Python 80 fresh 8..12 only, separate node counters'))
    print('BENCH_DONE',dict(ct),flush=True)
def end(h):
    a,b=from_history(h)
    return 1 if won(a) else -1 if won(b) else 0 if a|b==FULL else None
def game(e,j,color):
    h=j['opening'].copy();rr=[]
    while end(h) is None:
        p=1-2*(len(h)%2);r=e.quartz42(h) if p==color else e.search(h,32,False)
        r.update(arm='Q42' if p==color else 'B32',player=p,ply=len(h));rr.append(r);h.append(r['move'])
    w=end(h);return dict(job=j,color=color,history=h,winner=w,score2=1+color*w,rows=rr,nn=sum(r['nn'] for r in rr))
def worker(arg):
    mode,s,i,items=arg;p=D/f'{mode}_{s}_{i}.jsonl.gz';assert not p.exists();tmp=p.with_suffix('.tmp')
    ct=C.Counter();t=time.monotonic()
    ev=Puct(load_eval(s)) if mode=='ARENA' else None
    with gzip.open(tmp,'wt',compresslevel=3) as f:
        for j in items:
            if mode=='ORACLE':
                a,b=orient(j['opening']);r=timed(lambda:solve(a,b,500000,True));z=r['result'];emit(f,dict(job=j,**r));ct['complete']+=z['complete'];ct['nodes']+=z['nodes'];ct['items']+=1
            else:
                for c in j['color_order']:
                    g=game(ev,j,c);g['seed']=s;emit(f,g);ct['games']+=1;ct['moves']+=len(g['rows']);ct['nn']+=g['nn']
                    for r in g['rows']:
                        ct['attempts']+=r.get('exact_attempt',False);ct['complete']+=r.get('exact_complete',False);ct['nodes']+=r.get('exact_nodes',0);ct['proof_cpu_ns']+=r.get('exact_cpu_ns',0)
                ct['items']+=1
    if ev is not None:assert ev.ev.rows==ct['nn']
    tmp.replace(p);x=dict(file=p.name,sha256=sha(p),counts=dict(ct),elapsed_s=time.monotonic()-t,seed=s,shard=i)
    print('SHARD',mode,s,i,dict(ct),flush=True);return x
def execute(mode):
    jobs=load('PLAN.json')['jobs'];groups={0:[j for j in jobs if j['split']=='ATLAS']} if mode=='ORACLE' else {s:[j for j in jobs if j['split']!='ATLAS'] for s in SEEDS}
    tasks=[];n=8 if mode=='ORACLE' else 4
    for s,js in groups.items():
        for i in range(n):tasks.append((mode,s,i,js[i::n]))
    random.Random(SEED+len(mode)).shuffle(tasks);t=time.monotonic()
    with ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(worker,tasks))
    ct=C.Counter()
    for f in fs:ct.update(f['counts'])
    save(mode+'_LOCK.json',dict(plan=sha(D/'PLAN.json'),source=sha(D/'threshold42.cpp'),binary=sha(T/'threshold42.so'),engine=sha(D/'k42_engine.py'),collector=sha(D/'k42_experiment.py'),files=fs,counts=dict(ct),elapsed_s=time.monotonic()-t))
    print(mode,dict(ct),flush=True)
def rows(mode):
    for fi in load(mode+'_LOCK.json')['files']:
        p=D/fi['file'];assert sha(p)==fi['sha256'];yield from raw(p)
def summarize():
    import numpy as np
    bk=load('BENCHMARK_LOCK.json');p=D/bk['file'];assert sha(p)==bk['sha256']
    bench=list(raw(p));oref={r['job']['id']:r['result'] for r in rows('ORACLE')};stats={}
    for r in bench:
        truth=r['truth'];j=r['job']
        if j['frame']=='FRESH':
            z=oref[j['id']];truth=z.get('values')
        for key,q in r['runs'].items():
            z=q['result']
            if z['complete'] and truth is not None:
                assert z['best_value']==max(truth.values()) and z['best']==min(int(m) for m,v in truth.items() if v==z['best_value'])
                if 'bounds' in z:assert all(x[0]<=truth[m]<=x[1] for m,x in z['bounds'].items())
                if key!='action':assert z['values']==truth
    for key,rs in [('K41',[r for r in bench if r['job']['frame']=='K41']),('FRESH',[r for r in bench if r['job']['frame']=='FRESH'])]+[(f'fresh_e{e}',[r for r in bench if r['job']['frame']=='FRESH' and r['job']['empties']==e]) for e in range(8,17)]:
        x={}
        for k in ('action','census','python'):
            v=[r['runs'][k] for r in rs if k in r['runs']]
            if v:x[k]=dict(n=len(v),complete=sum(r['result']['complete'] for r in v),nodes=sum(r['result']['nodes'] for r in v),cpu_ns=sum(r['cpu_ns'] for r in v),wall_ns=sum(r['wall_ns'] for r in v))
        both=[r for r in rs if all(r['runs'][k]['result']['complete'] for k in ('action','census'))]
        if both:
            x['both_complete']=dict(n=len(both),nodes_reduction=1-sum(r['runs']['action']['result']['nodes'] for r in both)/sum(r['runs']['census']['result']['nodes'] for r in both),cpu_reduction=1-sum(r['runs']['action']['cpu_ns'] for r in both)/sum(r['runs']['census']['cpu_ns'] for r in both),wall_reduction=1-sum(r['runs']['action']['wall_ns'] for r in both)/sum(r['runs']['census']['wall_ns'] for r in both))
        bp=[r for r in rs if 'python' in r['runs'] and r['runs']['python']['result']['complete'] and r['runs']['action']['result']['complete']]
        if bp:x['python_both']=dict(n=len(bp),cpu_speedup=sum(r['runs']['python']['cpu_ns'] for r in bp)/sum(r['runs']['action']['cpu_ns'] for r in bp),wall_speedup=sum(r['runs']['python']['wall_ns'] for r in bp)/sum(r['runs']['action']['wall_ns'] for r in bp))
        stats[key]=x
    arena={};gs=list(rows('ARENA'))
    for split,N in [('EARLY',64),('LATE',48),('MID',48)]:
        q=[g for g in gs if g['job']['split']==split];pairs=C.defaultdict(list);first=C.Counter()
        for g in q:
            pairs[g['job']['id']].append(g);r=g['rows'][0];tag='Q' if r['arm']=='Q42' else 'B';first[tag+'_nn']+=r['nn'];first[tag+'_cpu_ns']+=r['cpu_ns'];first[tag+'_wall_ns']+=r['wall_ns']
        assert len(pairs)==N and all(len(v)==8 for v in pairs.values())
        def stat(gg):
            v=[0]*16
            for g in gg:
                v[0]+=g['score2'];v[1]+=1;v[2]+=g['score2']==2;v[3]+=g['score2']==1;v[4]+=g['score2']==0
                for r in g['rows']:
                    i=5 if r['player']==g['color'] else 9;v[i]+=1;v[i+1]+=r['nn'];v[i+2]+=r['cpu_ns'];v[i+3]+=r['wall_ns']
                    v[13]+=r.get('exact_attempt',False);v[14]+=r.get('exact_complete',False);v[15]+=r.get('exact_nodes',0)
            return v
        def rec(v):
            red=lambda k:1-(v[5+k]/v[5])/(v[9+k]/v[9]) if v[5] and v[9] and v[9+k] else None
            return dict(games=int(v[1]),wins=int(v[2]),draws=int(v[3]),losses=int(v[4]),score=float(v[0]/(2*v[1])),candidate_moves=int(v[5]),candidate_nn=int(v[6]),baseline_moves=int(v[9]),baseline_nn=int(v[10]),nn_reduction=red(1),cpu_reduction=red(2),wall_reduction=red(3),attempts=int(v[13]),complete=int(v[14]),nodes=int(v[15]))
        a=np.asarray([stat(v) for _,v in sorted(pairs.items())],dtype=float);rng=np.random.default_rng(SEED+43);b=a[rng.integers(0,len(a),(4000,len(a)))].sum(axis=1)
        sc=np.sort(b[:,0]/(2*b[:,1]));valid=(b[:,5]>0)&(b[:,9]>0)&(b[:,10]>0);red=np.sort(1-(b[valid,6]/b[valid,5])/(b[valid,10]/b[valid,9]))
        r=rec(stat(q));r.update(NI_lower=float(sc[199]),score_CI=[float(sc[99]),float(sc[3899])],NN_CI=[float(np.quantile(red,.025)),float(np.quantile(red,.975))] if len(red) else None,first=dict(first),by_seed={str(s):rec(stat([g for g in q if g['seed']==s])) for s in SEEDS})
        r['identical_pairs']=sum(v[0]['history']==v[1]['history'] for vv in pairs.values() for s in SEEDS for v in [[g for g in vv if g['seed']==s]])
        arena[split]=r
    save('SUMMARY.json',dict(benchmark=stats,oracle=load('ORACLE_LOCK.json')['counts'],arena=arena,counts=load('ARENA_LOCK.json')['counts'],scope='fixed sampled quiet late/mid and early opening families; same model per game; cold solver benchmark; native full census is shared-implementation reference, independent C++ audit separate'))
    print('SUMMARY',json.dumps(dict(benchmark={k:v for k,v in stats.items() if k in ('K41','FRESH')},oracle=load('ORACLE_LOCK.json')['counts'],arena=arena),separators=(',',':')),flush=True)
if __name__=='__main__':
    if sys.argv[1] in ('ORACLE','ARENA'):execute(sys.argv[1])
    else:globals()[sys.argv[1]]()
