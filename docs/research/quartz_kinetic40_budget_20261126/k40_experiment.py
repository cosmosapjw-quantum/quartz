"""Matched common-state corpus and frozen budget allocation; same weights in every game."""
import collections as C, gzip,json,multiprocessing as mp,random,sys,time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as F
from k40_plan import D,ROOT,SEEDS,FIT,SPLITS,H,WORKERS,SEED,family,load,save,sha
from k40_engine import Puct,load_eval,from_history,won,FULL,proof,draw_h
from budget import solve
assert D.name=='quartz_kinetic40_budget_20261126'
ARMS=('L20','A')
def raw(p):
    with gzip.open(p,'rt') as f:
        yield from map(json.loads,f)
def emit(f,x):f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n')
def end(h):
    a,b=from_history(h)
    return 1 if won(a) else -1 if won(b) else 0 if a|b==FULL else None
def side(h):return 1-2*(len(h)%2)
def decide(e,h,arm,policy):
    return e.adaptive40(h,policy) if arm=='A' else e.search(h,int(arm[1:]),arm[0]=='L')
def played(e,h,arm,base,p,policy):
    h=list(h);start=h.copy();rows=[]
    while end(h) is None:
        who=arm if side(h)==p else base
        r=decide(e,h,who,policy);r.update(arm=who,player=side(h),ply=len(h));h.append(r['move']);rows.append(r)
    w=end(h);return dict(start=start,history=h,winner=w,score2=1+p*w,player=p,rows=rows,nn=sum(r['nn'] for r in rows))
def source(e,j):
    h=j['opening'].copy();rr=[];start=e.ev.rows
    while len(h)<j['target'] and end(h) is None:
        r=decide(e,h,'L32' if side(h)==1 else 'B32',{});r.update(player=side(h),ply=len(h));h.append(r['move']);rr.append(r)
    status='TERMINAL' if end(h) is not None else 'GATED'
    if status!='TERMINAL':
        a,b=from_history(h);pr=proof(a,b) if side(h)==1 else proof(b,a)
        if pr['move'] is None:status='ROOT'
    nn=sum(r['nn'] for r in rr);assert nn==e.ev.rows-start
    return dict(job=j,history=h,status=status,source=rr,nn=nn)
def atlas(e,r,seed):
    start=e.ev.rows;a=e.atlas40(r['history']);assert a['bin'] is not None
    assert a['actual_nn']==e.ev.rows-start
    return dict(id=r['job']['id'],seed=seed,split=r['job']['split'],history=r['history'],atlas=a,nn=a['actual_nn'])
def label(e,r):
    a=r['atlas'];h=r['history'];acts={cap:a[f'L{cap}']['move'] for cap in H};before=e.ev.rows
    labels=[];u={}
    if len(set(acts.values()))>1:
        for m in sorted(set(acts.values())):
            for cap in (32,64):
                g=played(e,h+[m],f'L{cap}',f'B{cap}',side(h),{});g.update(action=m,refcap=cap);labels.append(g)
            u[m]=sum(g['score2'] for g in labels if g['action']==m)
    # Each gain is in quarter-score units; stable action contrasts are exactly zero.
    d={str(cap):u.get(acts[cap],0)-u.get(acts[20],0) for cap in H}
    c={str(cap):a[f'L{cap}']['nn']-a['L20']['nn'] for cap in H}
    nn=sum(g['nn'] for g in labels);assert nn==e.ev.rows-before
    return dict(id=r['id'],seed=r['seed'],split=r['split'],history=h,bin=a['bin'],entropy=a['entropy'],root_value=a['root_value'],prior_mismatch=a['L20']['move']!=a['L20']['prior_best'],actions=acts,delta4=d,cost_delta=c,low_nn=a['L20']['nn'],base_nn=a['B32']['nn'],labels=labels,nn=nn)
def worker(arg):
    mode,seed,shard,items,policy=arg
    e=Puct(load_eval(seed));t=time.perf_counter();counts=C.Counter()
    out=D/f'{mode}_{seed}_{shard}.jsonl.gz';assert not out.exists();tmp=out.with_suffix('.tmp')
    with gzip.open(tmp,'wt',compresslevel=3) as f:
        for item in items:
            if mode=='SOURCE':
                assert item['source']==seed;r=source(e,item);emit(f,r);counts[r['status']]+=1;counts['nn']+=r['nn']
            elif mode=='ATLAS':
                r=atlas(e,item,seed);emit(f,r);counts['nn']+=r['nn']
            elif mode in ('FIT_LABEL','TEST_LABEL'):
                assert item['seed']==seed;r=label(e,item);emit(f,r);counts['nn']+=r['nn'];counts['reference_games']+=len(r['labels']);counts['disagreement']+=bool(r['labels'])
            elif mode=='ARENA':
                for arm in ARMS:
                    for p in item['color_order']:
                        g=played(e,item['opening'],arm,'B32',p,policy);g.update(job=item,seed=seed,match=arm+'_B32');emit(f,g)
                        counts['nn']+=g['nn'];counts['games']+=1;counts['moves']+=len(g['rows'])
            counts['items']+=1
        f.flush()
    assert counts['nn']==e.ev.rows
    tmp.replace(out);r=dict(file=out.name,sha256=sha(out),seed=seed,shard=shard,counts=dict(counts),elapsed_s=time.perf_counter()-t)
    print('SHARD',mode,seed,shard,dict(counts),flush=True);return r
def execute(mode,groups,policy={}):
    tasks=[];t=time.perf_counter()
    for s,items in groups.items():
        for k in range(4):
            if items[k::4]:tasks.append((mode,s,k,items[k::4],policy))
    random.Random(SEED+len(mode)).shuffle(tasks)
    with ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(worker,tasks))
    c=C.Counter()
    for f in fs:c.update(f['counts'])
    save(mode+'_LOCK.json',dict(plan=sha(D/'PLAN.json'),engine=sha(D/'k40_engine.py'),experiment=sha(D/'k40_experiment.py'),files=fs,counts=dict(c),elapsed_s=time.perf_counter()-t,policy=policy))
    print(mode,dict(c),flush=True)
def rows(mode):
    lk=load(mode+'_LOCK.json')
    for fi in lk['files']:
        assert sha(D/fi['file'])==fi['sha256']
        yield from raw(D/fi['file'])
def screen():
    jobs=[j for j in load('PLAN.json')['jobs'] if j['split']!='ARENA']
    execute('SOURCE',{s:[j for j in jobs if j['source']==s] for s in SEEDS})
    rr=sorted(rows('SOURCE'),key=lambda r:r['job']['id']);seen=set();ad=[];dups=[];ct=C.Counter()
    for r in rr:
        j=r['job'];ct[j['split']+':'+r['status']]+=1
        if r['status']!='ROOT':continue
        z=family(r['history'])
        if z in seen:dups.append(j['id']);continue
        seen.add(z);ad.append(r);ct[j['split']+':ADMITTED']+=1
    save('FRAME.json',dict(ids=[r['job']['id'] for r in ad],duplicates=dups,counts=dict(ct),source_lock=sha(D/'SOURCE_LOCK.json')))
    # Atlas is computed for all four models, but 102 TRAIN/CAL never used in fit/admission.
    execute('ATLAS',{s:ad for s in SEEDS})
    print('FRAME',dict(ct),flush=True)
def labels(mode):
    rr=list(rows('ATLAS'))
    if mode=='FIT_LABEL':rr=[r for r in rr if r['split'] in ('TRAIN','CAL') and r['seed'] in FIT]
    else:rr=[r for r in rr if r['split']=='TEST' or r['seed']==102]
    execute(mode,{s:[r for r in rr if r['seed']==s] for s in SEEDS})
def points(rr):
    out={}
    for r in rr:
        p=out.setdefault(r['bin'],{cap:[0,0] for cap in H})
        for cap in H:p[cap][0]+=r['cost_delta'][str(cap)];p[cap][1]+=r['delta4'][str(cap)]
    return out
def fracpolicy(p):return {str(z):{str(h):str(q) for h,q in hh.items()} for z,hh in p.items()}
def expected(r,p):
    hh=p.get(str(r['bin']),{'20':'1'})
    return sum(F(q)*r['delta4'][h]/4 for h,q in hh.items()),sum(F(q)*r['cost_delta'][h] for h,q in hh.items())
def describe(rr,p,bootstrap=True):
    assert rr
    c=C.Counter();gain=F(0);extra=F(0);by=C.defaultdict(list)
    for r in rr:
        q,k=expected(r,p);hh=draw_h(r['history'],p,r['bin']);g=F(r['delta4'][str(hh)],4);actual=r['cost_delta'][str(hh)]
        gain+=q;extra+=k;c.update(roots=1,disagreements=bool(r['labels']),positive32=r['delta4']['32']>0,negative32=r['delta4']['32']<0,delta4_26=r['delta4']['26'],delta4_32=r['delta4']['32'],base=r['base_nn'],low=r['low_nn'],realized_delta4=4*g,realized_extra=actual)
        by[r['id']].append((q,k,r['low_nn'],r['base_nn']))
        c[f'h{hh}']+=1
    n=len(rr);out=dict(roots=n,families=len(by),counts={k:float(v) if isinstance(v,F) else v for k,v in c.items()},mean_gain=float(gain/n),mean_extra=float(extra/n),expected_cost=float(F(c['low'])+extra),expected_NN_reduction=float(1-(F(c['low'])+extra)/c['base']),realized_gain=float(F(c['realized_delta4'],4*n)),realized_NN_reduction=float(1-F(c['low']+c['realized_extra'],c['base'])),mean32=c['delta4_32']/(4*n),mean26=c['delta4_26']/(4*n),flip32=sum(r['actions']['20']!=r['actions']['32'] for r in rr)/n)
    if bootstrap:
        import numpy as np
        a=np.asarray([[sum(float(v[i]) for v in g) for i in range(4)]+[len(g)] for _,g in sorted(by.items())])
        rng=np.random.default_rng(SEED+76);b=a[rng.integers(0,len(a),(4000,len(a)))].sum(axis=1)
        g=np.sort(b[:,0]/b[:,4]);c=np.sort(1-(b[:,1]+b[:,2])/b[:,3]);out.update(gain_CI=[float(g[99]),float(g[3899])],gain_lower=float(g[199]),NN_CI=[float(c[99]),float(c[3899])])
    return out
def fit():
    rr=list(rows('FIT_LABEL'));tr=[r for r in rr if r['split']=='TRAIN'];cal=[r for r in rr if r['split']=='CAL']
    assert all(r['seed'] in FIT for r in rr)
    p=points(tr);B=F(7,10)*sum(r['base_nn'] for r in tr)-sum(r['low_nn'] for r in tr)
    s=solve(p,B);policy=fracpolicy(s['prob']);t=describe(tr,policy);c=describe(cal,policy)
    admitted=s['feasible'] and c['gain_lower']>0 and c['expected_NN_reduction']>=.30
    out=dict(policy=policy,feasible=s['feasible'],budget=str(B),gain4=str(s['gain']),extra=str(s['cost']),dual_score_per_row=None if s['dual'] is None else str(s['dual']/4),dual_gain4=None if s['dual_bound'] is None else str(s['dual_bound']),train_points=p,train=t,cal=c,admitted=admitted,fit_lock=sha(D/'FIT_LABEL_LOCK.json'),frame_lock=sha(D/'FRAME.json'),note='Frozen before TEST/heldout102 labels and arena; unseen bins fallback L20, no refit.')
    save('FROZEN.json',out);print('FROZEN',json.dumps(out,separators=(',',':')),flush=True)
def arena():
    p=load('FROZEN.json')['policy'];jobs=[j for j in load('PLAN.json')['jobs'] if j['split']=='ARENA']
    execute('ARENA',{s:jobs for s in SEEDS},p)
def summarize():
    import numpy as np
    p=load('FROZEN.json')['policy'];rr=list(rows('FIT_LABEL'))+list(rows('TEST_LABEL'))
    test=[r for r in rr if r['split']=='TEST']
    info={k:describe(q,p) for k,q in [('all',test),('fit_models',[r for r in test if r['seed'] in FIT]),('heldout102',[r for r in test if r['seed']==102])]}
    for s in SEEDS:info[str(s)]=describe([r for r in test if r['seed']==s],p)
    controls={str(h):describe(test,{str(z):{str(h):'1'} for z in range(8)}) for h in H}
    matched={};by=C.defaultdict(dict)
    for r in test:by[r['id']][r['seed']]=r
    for fam,ss in by.items():assert set(ss)==set(SEEDS)
    for a,b in [(105,103),(102,101),(103,101)]:
        d=np.asarray([(ss[a]['delta4']['32']-ss[b]['delta4']['32'])/4 for ss in by.values()])
        rng=np.random.default_rng(SEED+72);v=np.sort(d[rng.integers(0,len(d),(4000,len(d)))].mean(axis=1));matched[f'{a}-{b}']=dict(families=len(d),mean=float(d.mean()),CI=[float(v[99]),float(v[3899])],nonzero=int((d!=0).sum()))
    # The policy-conditioned proxy for root value: label only when root's selected L20 move has actual reference score.
    diag={}
    for s in SEEDS:
        q=[r for r in test if r['seed']==s];v=[]
        for r in q:
            if r['labels']:
                act=r['actions']['20'];u=sum(g['score2'] for g in r['labels'] if g['action']==act)/4
                v.append((r['root_value']-(2*u-1))**2)
        diag[str(s)]=dict(mean_entropy=float(np.mean([r['entropy'] for r in q])),prior_mismatch=float(np.mean([r['prior_mismatch'] for r in q])),flip32=sum(r['actions']['20']!=r['actions']['32'] for r in q)/len(q),selected_reference_n=len(v),selected_value_MSE=float(np.mean(v)) if v else None)
    gs=list(rows('ARENA'));arms={}
    for arm in ARMS:
        match=arm+'_B32';q=[g for g in gs if g['match']==match];pairs=C.defaultdict(list)
        for g in q:pairs[g['job']['id']].append(g)
        assert len(pairs)==64 and all(len(v)==8 for v in pairs.values())
        def stat(gg):
            v=[0]*11
            for g in gg:
                v[0]+=g['score2'];v[1]+=1;v[2]+=g['score2']==2;v[3]+=g['score2']==1;v[4]+=g['score2']==0
                for r in g['rows']:
                    i=5 if r['player']==g['player'] else 8
                    v[i]+=1;v[i+1]+=r['nn'];v[i+2]+=r['cpu_ns']
            return v
        def rec(v):
            return dict(games=int(v[1]),wins=int(v[2]),draws=int(v[3]),losses=int(v[4]),score=float(v[0]/(2*v[1])),NN_reduction=float(1-(v[6]/v[5])/(v[9]/v[8])),CPU_reduction=float(1-(v[7]/v[5])/(v[10]/v[8])),candidate_NN=int(v[6]),baseline_NN=int(v[9]),candidate_moves=int(v[5]),baseline_moves=int(v[8]))
        a=np.asarray([stat(v) for _,v in sorted(pairs.items())],dtype=np.float64);rng=np.random.default_rng(SEED+17);b=a[rng.integers(0,len(a),(4000,len(a)))].sum(axis=1)
        scores=np.sort(b[:,0]/(2*b[:,1]));cost=np.sort(1-(b[:,6]/b[:,5])/(b[:,9]/b[:,8]));r=rec(stat(q))
        r.update(NI_lower=float(scores[199]),score_CI=[float(scores[99]),float(scores[3899])],NN_CI=[float(cost[99]),float(cost[3899])],joint=bool(scores[199]>=.45 and r['NN_reduction']>=.30),by_seed={str(s):rec(stat([g for g in q if g['seed']==s])) for s in SEEDS});arms[match]=r
    out=dict(admitted=load('FROZEN.json')['admitted'],test=info,controls=controls,matched=matched,diagnostics=diag,arena=arms,counts={m:load(m+'_LOCK.json')['counts'] for m in ('SOURCE','ATLAS','FIT_LABEL','TEST_LABEL','ARENA')},scope='family-cluster bootstrap, matched common states, selected conditional value proxy, fixed checkpoints; no universal model or minimax inference')
    save('SUMMARY.json',out);print('SUMMARY',json.dumps(out,separators=(',',':')),flush=True)
if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd in ('FIT_LABEL','TEST_LABEL'):labels(cmd)
    else:globals()[cmd]()
