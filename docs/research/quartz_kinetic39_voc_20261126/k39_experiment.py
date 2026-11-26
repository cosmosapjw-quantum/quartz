"""Same-checkpoint finite-cap experiments; frozen split, single-tree extension, exact zero contrast."""
import collections as C, gzip, json, multiprocessing as mp, pathlib, random, sys, time
from concurrent.futures import ProcessPoolExecutor
from k39_plan import D,ROOT,SEEDS,SPLITS,ARMS,WORKERS,SEED,PRICE,family,load,save,sha
from k39_engine import Puct,load_eval,from_history,won,FULL,proof,bin_of
assert D.name=='quartz_kinetic39_voc_20261126'
def raw(path):
    with gzip.open(path,'rt') as f:
        for l in f:yield json.loads(l)
def emit(f,row):f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
def end(h):
    a,b=from_history(h)
    return 1 if won(a) else -1 if won(b) else 0 if (a|b)==FULL else None
def side(h):return 1-2*(len(h)%2)
def decide(eng,h,arm,mask=0):
    if arm=='A':return eng.adaptive(h,mask)
    return eng.search(h,int(arm[1:]),arm[0]=='L')
def played(eng,h,arm,base,player,mask=0):
    h=list(h);start=h.copy();rows=[]
    while end(h) is None:
        p=side(h);who=arm if p==player else base
        x=decide(eng,h,who,mask);x.update(arm=who,player=p,ply=len(h));h.append(x['move']);rows.append(x)
    w=end(h);return dict(start=start,history=h,winner=w,score2=1+player*w,player=player,rows=rows,nn=sum(r['nn'] for r in rows))
def screen_one(eng,j):
    h=j['opening'].copy();source=[];n=eng.ev.rows
    while len(h)<j['target'] and end(h) is None:
        who='L32' if side(h)==1 else 'B32';x=decide(eng,h,who)
        x.update(arm=who,player=side(h),ply=len(h));h.append(x['move']);source.append(x)
    atlas=None;status='TERMINAL' if end(h) is not None else 'GATED'
    if status!='TERMINAL':
        a,b=from_history(h);pr=proof(a,b) if side(h)==1 else proof(b,a)
        if pr['move'] is None:status='ROOT';atlas=eng.atlas(h)
    actual=sum(x['nn'] for x in source)+(atlas['actual_nn'] if atlas else 0)
    assert actual==eng.ev.rows-n
    return dict(job=j,history=h,status=status,source=source,atlas=atlas,actual_nn=actual)
def label_one(eng,r):
    j=r['job'];h=r['history'];a=r['atlas'];low=a['L20']['move'];hi=a['L32']['move'];n=eng.ev.rows
    if low==hi:return dict(id=j['id'],seed=j['seed'],split=j['split'],bin=a['bin'],flip=False,delta4=0,labels=[],cost_delta=a['L32']['nn']-a['L20']['nn'],actual_nn=0)
    labels=[]
    for m in (low,hi):
        for cap in (32,64):
            g=played(eng,h+[m],f'L{cap}',f'B{cap}',side(h));g.update(action=m,refcap=cap)
            labels.append(g)
    v=[sum(g['score2'] for g in labels if g['action']==m) for m in (low,hi)]
    actual=sum(g['nn'] for g in labels);assert actual==eng.ev.rows-n
    return dict(id=j['id'],seed=j['seed'],split=j['split'],bin=a['bin'],flip=True,delta4=v[1]-v[0],labels=labels,cost_delta=a['L32']['nn']-a['L20']['nn'],actual_nn=actual)
def worker(arg):
    mode,seed,shard,items,mask=arg
    assert mode in ('SCREEN','TRAIN_CAL','TEST','ARENA') and all((x.get('job',x))['seed']==seed for x in items)
    ev=load_eval(seed);eng=Puct(ev);t=time.perf_counter()
    out=D/f'{mode}_{seed}_{shard}.jsonl.gz';assert not out.exists();tmp=out.with_suffix('.tmp')
    count=C.Counter()
    with gzip.open(tmp,'wt',compresslevel=3) as f:
        for item in items:
            if mode=='SCREEN':
                r=screen_one(eng,item);emit(f,r);count[r['status']]+=1;count['nn']+=r['actual_nn']
            elif mode in ('TRAIN_CAL','TEST'):
                r=label_one(eng,item);emit(f,r);count['roots']+=1;count['flips']+=r['flip'];count['reference_games']+=len(r['labels']);count['nn']+=r['actual_nn']
            else:
                for arm in ARMS:
                    q,b=arm.split('_')
                    for p in item['color_order']:
                        g=played(eng,item['opening'],q,b,p,mask)
                        g.update(job=item,match=arm)
                        emit(f,g);count['games']+=1;count['moves']+=len(g['rows']);count['nn']+=g['nn']
            count['items']+=1
        f.flush()
    assert count['nn']==ev.rows
    if mode=='ARENA':assert count['games']==len(items)*len(ARMS)*2
    tmp.replace(out)
    result=dict(file=out.name,sha256=sha(out),seed=seed,shard=shard,elapsed_s=time.perf_counter()-t,counts=dict(count))
    print('SHARD',mode,seed,shard,dict(count),flush=True)
    return result
def execute(mode,items,mask=0):
    t=time.perf_counter();tasks=[]
    for s in SEEDS:
        group=[x for x in items if (x.get('job',x))['seed']==s]
        for k in range(4):
            if group[k::4]:tasks.append((mode,s,k,group[k::4],mask))
    random.Random(SEED+len(mode)).shuffle(tasks)
    with ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(worker,tasks))
    c=C.Counter()
    for f in fs:c.update(f['counts'])
    save(mode+'_LOCK.json',dict(plan_sha256=sha(D/'PLAN.json'),engine_sha256=sha(D/'k39_engine.py'),experiment_sha256=sha(D/'k39_experiment.py'),files=fs,counts=dict(c),elapsed_s=time.perf_counter()-t,mask=mask))
    print(mode,dict(c),flush=True)
def rows(mode):
    lk=load(mode+'_LOCK.json')
    for fi in lk['files']:
        assert sha(D/fi['file'])==fi['sha256']
        yield from raw(D/fi['file'])
def screen():
    execute('SCREEN',[j for j in load('PLAN.json')['jobs'] if j['split']!='ARENA'])
    rs=sorted(rows('SCREEN'),key=lambda r:r['job']['id'])
    seen=set();ids=[];duplicates=[];c=C.Counter()
    for r in rs:
        j=r['job'];c[j['split']+':'+r['status']]+=1
        if r['status']!='ROOT':continue
        f=family(r['history'])
        if f in seen:duplicates.append(j['id']);continue
        seen.add(f);ids.append(j['id']);c[j['split']+':ADMITTED']+=1
    save('FRAME.json',dict(ids=ids,duplicates=duplicates,screen=sha(D/'SCREEN_LOCK.json'),counts=dict(c)))
    print('FRAME',dict(c),'duplicate',len(duplicates),flush=True)
def label(mode):
    ids=set(load('FRAME.json')['ids']);sp={'TRAIN','CAL'} if mode=='TRAIN_CAL' else {'TEST'}
    rr=[r for r in rows('SCREEN') if r['job']['id'] in ids and r['job']['split'] in sp]
    execute(mode,rr)
def net_units(r):return 125*r['delta4']-r['cost_delta'] # units 1/500 score
def aggregates(rr,mask):
    n=len(rr);assert n>0
    z=C.Counter()
    for r in rr:
        z['roots']+=1;z['flips']+=r['flip'];z['positive']+=r['delta4']>0;z['negative']+=r['delta4']<0;z['delta4']+=r['delta4'];z['cost_delta']+=r['cost_delta']
        if mask>>r['bin']&1:
            z['extended']+=1;z['policy_delta4']+=r['delta4'];z['policy_extra_nn']+=r['cost_delta'];z['net_units']+=net_units(r)
    return dict(z,mean_delta=z['delta4']/(4*n),policy_mean_delta=z['policy_delta4']/(4*n),mean_net=z['net_units']/(500*n))
def low_ci(rr,mask):
    import numpy as np
    rng=np.random.default_rng(SEED+913)
    vals=[]
    for s in SEEDS:
        a=np.asarray([net_units(r)*(bool(mask>>r['bin']&1))/500 for r in rr if r['seed']==s],dtype=float)
        if len(a):vals.append(a[rng.integers(0,len(a),size=(4000,len(a)))].sum(axis=1))
    b=np.sort(sum(vals)/len(rr));return [float(b[99]),float(b[199]),float(b[3899])]
def tables(rr):
    return [dict(mask=m,net_units=sum(net_units(r) for r in rr if m>>r['bin']&1),extra_nn=sum(r['cost_delta'] for r in rr if m>>r['bin']&1)) for m in range(16)]
def describe(rr,m):
    return dict(all=aggregates(rr,m),net_ci=low_ci(rr,m),by_seed={str(s):aggregates([r for r in rr if r['seed']==s],m) for s in SEEDS if any(r['seed']==s for r in rr)},by_bin={str(z):aggregates([r for r in rr if r['bin']==z],m) for z in range(4) if any(r['bin']==z for r in rr)})
def fit():
    rr=list(rows('TRAIN_CAL'));tr=[r for r in rr if r['split']=='TRAIN'];cal=[r for r in rr if r['split']=='CAL']
    tab=tables(tr);best=max(tab,key=lambda x:(x['net_units'],-x['extra_nn'],-x['mask']))
    m=best['mask'];cald=describe(cal,m)
    out=dict(mask=m,admitted=cald['net_ci'][1]>0,price=PRICE,train_table=tab,train=describe(tr,m),cal=cald,train_cal_lock=sha(D/'TRAIN_CAL_LOCK.json'),frame_lock=sha(D/'FRAME.json'),note='Candidate frozen before TEST labels/arena; admission if CAL one-sided 95% net lower >0; rejected candidate evaluated descriptively, no post-test fit')
    save('FROZEN.json',out)
    print('FROZEN',json.dumps(out,separators=(',',':')),flush=True)
def arena():
    m=load('FROZEN.json')['mask'];execute('ARENA',[j for j in load('PLAN.json')['jobs'] if j['split']=='ARENA'],m)
def summarise():
    import numpy as np
    m=load('FROZEN.json')['mask'];test=list(rows('TEST'))
    rootmap={r['job']['id']:r for r in rows('SCREEN') if r['atlas']}
    rr=list(rows('TRAIN_CAL'))+test
    data={}
    for group in ('TRAIN','CAL','TEST'):
        q=[r for r in rr if r['split']==group];n=len(q)
        a=[rootmap[r['id']]['atlas'] for r in q]
        cost={k:sum(x[k]['nn'] for x in a) for k in ('B20','L20','B32','L32')}
        cost['A']=sum(x['L32' if m>>x['bin']&1 else 'L20']['nn'] for x in a)
        data[group]=dict(rows=n,cost=cost,detail=describe(q,m),uniform32=describe(q,15),fixed_low=describe(q,0),flip_proxy=describe(q,10))
    gg=list(rows('ARENA'));by=C.defaultdict(list)
    for g in gg:by[g['match']].append(g)
    arms={}
    for arm in ARMS:
        pairs=C.defaultdict(list)
        for g in by[arm]:pairs[g['job']['seed'],g['job']['id']].append(g)
        assert len(pairs)==len(SEEDS)*SPLITS['ARENA'] and all(len(v)==2 and {x['player'] for x in v}=={-1,1} for v in pairs.values())
        def stats(gs):
            # score2, wins, draws, losses, candidate move/nn/cpu/wall, baseline move/nn/cpu/wall
            v=[0]*14
            for g in gs:
                v[0]+=g['score2'];v[1]+=g['score2']==2;v[2]+=g['score2']==1;v[3]+=g['score2']==0
                for r in g['rows']:
                    i=4 if r['player']==g['player'] else 9
                    v[i]+=1;v[i+1]+=r['nn'];v[i+2]+=r['cpu_ns'];v[i+3]+=r['wall_ns'];v[i+4]+=r['visits']
            return v
        def rec(x):
            n=x[1]+x[2]+x[3];score=x[0]/(2*n)
            ratio=lambda i:1-(x[4+i]/x[4])/(x[9+i]/x[9]) if x[9+i] and x[4] and x[9] else None
            return dict(games=int(n),wins=int(x[1]),draws=int(x[2]),losses=int(x[3]),score=float(score),NN_reduction=ratio(1),CPU_reduction=ratio(2),wall_reduction=ratio(3),visits_reduction=ratio(4),candidate_moves=int(x[4]),candidate_NN=int(x[5]),baseline_moves=int(x[9]),baseline_NN=int(x[10]))
        v=stats(by[arm]);r=rec(v)
        rng=np.random.default_rng(SEED+197);arr=np.zeros((4000,14),dtype=np.float64)
        for s in SEEDS:
            a=np.asarray([stats(vv) for (ss,j),vv in sorted(pairs.items()) if ss==s],dtype=np.float64)
            arr+=a[rng.integers(0,len(a),size=(4000,len(a)))].sum(axis=1)
        scores=np.sort(arr[:,0]/(2*arr[:,1:4].sum(axis=1)))
        ratio=1-(arr[:,5]/arr[:,4])/(arr[:,10]/arr[:,9]);ratio=np.sort(ratio)
        r.update(NI_lower=float(scores[199]),score_CI=[float(scores[99]),float(scores[3899])],NN_CI=[float(ratio[99]),float(ratio[3899])],joint=bool(scores[199]>=.45 and r['NN_reduction']>=.30) if arm.endswith('B32') else None,
                 by_seed={str(s):rec(stats([g for g in by[arm] if g['job']['seed']==s])) for s in SEEDS},same_color_swap_histories=sum(vv[0]['history']==vv[1]['history'] for vv in pairs.values()))
        arms[arm]=r
    save('SUMMARY.json',dict(mask=m,admitted=load('FROZEN.json')['admitted'],atlas=data,arena=arms,counts={mode:load(mode+'_LOCK.json')['counts'] for mode in ('SCREEN','TRAIN_CAL','TEST','ARENA')},scope='paired opening bootstrap within fixed checkpoint strata; 103/105 related; descriptive conditional reference utilities and actual games; no A1 claim'))
    print('SUMMARY',json.dumps(dict(mask=m,atlas={k:dict(rows=v['rows'],cost=v['cost'],all=v['detail']['all'],CI=v['detail']['net_ci']) for k,v in data.items()},arena=arms),separators=(',',':')),flush=True)
if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd in ('TRAIN_CAL','TEST'):label(cmd)
    else:globals()[cmd]()
