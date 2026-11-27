"""Fresh paired same-evaluator trajectories; native nested counterfactual root probes."""
import collections as C,gzip,json,random,sys,time,multiprocessing as mp,math
from concurrent.futures import ProcessPoolExecutor
from k48_plan import D,T,SEEDS,SEED,WORKERS,load,save,sha
from native48 import NNServer,rapfi_module,rsearch,BIN
def emit(f,x):f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n')
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def rows(mode):
    lk=load(mode+'_LOCK.json')
    for x in lk['files']:
        p=D/x['file'];assert sha(p)==x['sha256'];yield from raw(p)
def teach(ev,h,r,value):
    l=r['low'];u=r['high'];bd={m:(a,b) for m,a,b in r['partial']['bounds']}
    a,b=l['move'],u['move'];labels=[];vals=[]
    if a==b:labels=[None,None];vals=None;gain=0.
    else:
        for m in (a,b):
            lo,hi=bd[m]
            if lo==hi:t=None;s=float(lo)
            else:t=value(ev,h,m);s=max(lo,min(hi,t['raw']))
            labels.append(t);vals.append(s)
        gain=(vals[1]-vals[0])/2
        assert (bd[b][0]-bd[a][1])/2<=gain<=(bd[b][1]-bd[a][0])/2
    return dict(gain=gain,values=vals,teacher=labels,cost=u['nn']-l['nn'],visits=u['visits']-l['visits'],phase_wall_ns=r['whole_ns']-r['low_wall_ns'],changed=a!=b)
def nn_worker(arg):
    mode,s,i,js,mask=arg
    from k46_engine import load_eval,from_history,won,FULL
    sys.path.insert(0,str(D.parent/'quartz_kinetic47_native_20261127'))
    from risk47 import value
    ev=load_eval(s);ct=C.Counter();path=D/f'{mode}_{s}_{i}.jsonl.gz';assert not path.exists()
    source=mode in ('TRAIN','CAL','TEST')
    def end(h):
        a,b=from_history(h);return 1 if won(a) else -1 if won(b) else 0 if a|b==FULL else None
    with NNServer(ev) as srv,gzip.open(path,'wt',compresslevel=3) as f:
        for j in js:
            for color in j['color_order'][:1] if source else j['color_order']:
                h=j['opening'].copy();rr=[]
                while end(h) is None:
                    side=1-2*(len(h)%2);q=side==color
                    if q and source:r=srv.nested(h,atlas=True)
                    elif q and mode=='NN_ADAPT':r=srv.nested(h,mask=mask)
                    else:r=srv.search(h,q)
                    assert r['history']==h and r['move'] not in h
                    if q and source and r.get('low'):
                        r['utility']=teach(ev,h,r,value)
                        z=r['utility'];ct['atlas_roots']+=1;ct['extra_nn']+=z['cost'];ct['teacher_nn']+=sum(x['nn'] for x in z['teacher'] if x);ct['teacher_calls']+=sum(x is not None for x in z['teacher'])
                    r.update(arm='Q' if q else 'B',player=side,ply=len(h));rr.append(r);h.append(r['move'])
                    ct['nn']+=r['nn'];ct['moves']+=1;ct['common']+=r['kind']=='CERTIFICATE'
                w=end(h);emit(f,dict(job=j,seed=s,color=color,history=h,winner=w,score2=1+color*w,rows=rr,atlas=[],nn=sum(r['nn'] for r in rr)));ct['games']+=1
    assert ev.rows==ct['nn']+ct['extra_nn']+ct['teacher_nn'],(ev.rows,ct)
    print('SHARD',mode,s,i,dict(ct),flush=True)
    return dict(file=path.name,sha256=sha(path),counts=dict(ct))
def rapfi_worker(arg):
    mode,s,i,js,cap=arg;e=rapfi_module()
    ct=C.Counter();path=D/f'{mode}_{s}_{i}.jsonl.gz';assert not path.exists()
    with e.Server(BIN) as server,gzip.open(path,'wt',compresslevel=3) as f:
        for j in js:
            for color in j['color_order']:
                h=j['opening'].copy();rr=[]
                while True:
                    _,side,w=e.replay(h,'freestyle')
                    if w is not None:break
                    q=side==color;r=rsearch(e,server,h,e.seed(j,len(h)),q,cap)
                    r.update(arm='Q' if q else 'B',player=side,ply=len(h));rr.append(r);h.append(r['move'])
                    ct['nn']+=r['nn'];ct['moves']+=1;ct['common']+=r['kind']=='CERTIFICATE'
                emit(f,dict(job=j,seed=0,color=color,history=h,winner=w,score2=1+color*w,rows=rr,atlas=[],nn=sum(r['nn'] for r in rr)));ct['games']+=1
    print('SHARD',mode,s,i,dict(ct),flush=True)
    return dict(file=path.name,sha256=sha(path),counts=dict(ct))
def run(mode):
    pl=load('PLAN.json');nn=not mode.startswith('R');source=mode in ('TRAIN','CAL','TEST')
    js=[j for j in pl['jobs' if nn else 'rapfi_jobs'] if j['split']==(mode if source else 'ARENA')]
    mask=load('FROZEN.json')['mask'] if mode=='NN_ADAPT' else int(mode[1:]) if not nn else None
    seeds=SEEDS if nn else (0,);tasks=[(mode,s,i,js[i::4],mask) for s in seeds for i in range(4)]
    random.Random(SEED+len(mode)).shuffle(tasks)
    assert sha(BIN)==load('PARITY_LOCK.json')['binary']
    with ProcessPoolExecutor(max_workers=WORKERS if nn else 4,mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(nn_worker if nn else rapfi_worker,tasks))
    ct=sum((C.Counter(x['counts']) for x in fs),C.Counter());assert ct['games']==len(js)*len(seeds)*(1 if source else 2)
    save(mode+'_LOCK.json',dict(files=fs,counts=dict(ct),plan=sha(D/'PLAN.json'),source=sha(D/'experiment48.py'),bridge=sha(D/'native48.py'),binary=sha(BIN),mask_or_cap=mask))
    print(mode,dict(ct),flush=True)
def atlas(mode):
    for g in rows(mode):
        for r in g['rows']:
            if 'utility' in r:
                yield dict(job=g['job']['id'],seed=g['seed'],bin=r['bin'],disagree=int(r['low']['original_move']!=r['low']['prior_best']),**r['utility'])
def summary(rs):
    n=len(rs);g=math.fsum(r['gain'] for r in rs);c=sum(r['cost'] for r in rs);p=[r['gain'] for r in rs if r['gain']>0];m=[-r['gain'] for r in rs if r['gain']<0]
    return dict(n=n,gain=g,cost=c,mean=g/n if n else None,positive=len(p),negative=len(m),mean_positive=math.fsum(p)/len(p) if p else None,mean_negative=math.fsum(m)/len(m) if m else None,changed=sum(r['changed'] for r in rs),visits=sum(r['visits'] for r in rs),wall_ns=sum(r['phase_wall_ns'] for r in rs))
def fit():
    rr=list(atlas('TRAIN'));b=[summary([r for r in rr if r['bin']==z]) for z in range(8)];tot=summary(rr);cap=tot['cost']//4
    feasible=[]
    for m in range(256):
        s=[b[i] for i in range(8) if m>>i&1];c=sum(r['cost'] for r in s);g=math.fsum(r['gain'] for r in s)
        if c<=cap:feasible.append((g,-m.bit_count(),-m))
    mask=-max(feasible)[2]
    d=[summary([r for r in rr if r['disagree']==i]) for i in range(2)]
    out=dict(mask=mask,train=tot,bins=b,disagree=d,budget=cap,train_lock=sha(D/'TRAIN_LOCK.json'),scope='TRAIN-only binary allocation; no use of CAL or TEST')
    save('FROZEN.json',out);print('FROZEN',json.dumps(out),flush=True)
def eval_split(mode):
    f=load('FROZEN.json');rs=list(atlas(mode));tot=summary(rs);sel=summary([r for r in rs if f['mask']>>r['bin']&1]);p=sel['cost']/tot['cost'] if tot['cost'] else 0
    const=p*tot['gain'];admit=sel['gain']>0 and sel['gain']>const
    bins=[summary([r for r in rs if r['bin']==i]) for i in range(8)]
    def mse(fn):return math.fsum((r['gain']-fn(r))**2 for r in rs)/len(rs)
    base=f['train']['mean']
    pred=dict(constant=mse(lambda r:base),disagreement=mse(lambda r:f['disagree'][r['disagree']]['mean'] if f['disagree'][r['disagree']]['n'] else base),bin=mse(lambda r:f['bins'][r['bin']]['mean'] if f['bins'][r['bin']]['n'] else base))
    # family-cluster bootstrap, cost-matched mixture recomputed for each resample.
    import numpy as np
    groups=C.defaultdict(list)
    for r in rs:groups[r['job']].append(r)
    ar=[]
    for _,rr in sorted(groups.items()):
        a=summary(rr);b=summary([r for r in rr if f['mask']>>r['bin']&1]);ar.append([a['gain'],a['cost'],a['n'],b['gain'],b['cost']])
    ar=np.array(ar);rng=np.random.default_rng(SEED+len(mode));n=len(ar);v=ar[rng.integers(0,n,(8000,n))].sum(axis=1);delta=(v[:,3]-np.divide(v[:,4]*v[:,0],v[:,1],out=np.zeros(len(v)),where=v[:,1]>0))/v[:,2]
    return dict(total=tot,selected=sel,bins=bins,constant_fraction=p,constant_gain=const,excess_gain=sel['gain']-const,mean_excess_CI=list(map(float,np.quantile(delta,[.025,.975]))),positive_excess=admit,predictive_mse=pred)
def admit():
    x=eval_split('CAL');f=load('FROZEN.json');save('ADMISSION.json',dict(admitted=x['positive_excess'],mask=f['mask'] if x['positive_excess'] else 0,exploratory_mask=f['mask'],cal=x,frozen_sha=sha(D/'FROZEN.json')))
    print('ADMISSION',json.dumps(x),flush=True)
def test():
    x=eval_split('TEST');save('TEST_EVAL.json',x);print('TEST',json.dumps(x),flush=True)
if __name__=='__main__':
    m=sys.argv[1]
    if m in ('fit','admit','test'):globals()[m]()
    else:run(m)
