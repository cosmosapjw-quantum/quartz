"""Fresh paired same-evaluator trajectories; native nested counterfactual root probes."""
import collections as C,gzip,json,random,sys,time,multiprocessing as mp,math
from concurrent.futures import ProcessPoolExecutor
from k49_plan import D,T,SEEDS,SEED,WORKERS,load,save,sha
from native49 import NNServer,rapfi_module,rsearch,BIN
def emit(f,x):f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n')
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def rows(mode):
    lk=load(mode+'_LOCK.json')
    for x in lk['files']:
        p=D/x['file'];assert sha(p)==x['sha256'];yield from raw(p)
from teacher49 import pair,physical,calls
from allocation49 import fit as robust_fit,evaluate,H
def teach(ev,h,r):
    z=pair(ev,h,r)
    z['phase_wall_ns']=r['whole_ns']-r['low_wall_ns']
    return z
def nn_worker(arg):
    mode,s,i,js,mask=arg
    from k46_engine import load_eval,from_history,won,FULL
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
                        r['utility']=teach(ev,h,r)
                        z=r['utility'];ct['atlas_roots']+=1;ct['extra_nn']+=z['cost'];ct['teacher_nn']+=physical(z);ct['teacher_calls']+=calls(z)
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
    mask=load('ADMISSION.json')['mask'] if mode=='NN_ADAPT' else int(mode[1:]) if not nn else None
    seeds=SEEDS if nn else (0,);tasks=[(mode,s,i,js[i::4],mask) for s in seeds for i in range(4)]
    random.Random(SEED+len(mode)).shuffle(tasks)
    assert sha(BIN)==load('BINARY_LOCK.json')['binary_sha256']
    with ProcessPoolExecutor(max_workers=WORKERS if nn else 4,mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(nn_worker if nn else rapfi_worker,tasks))
    ct=sum((C.Counter(x['counts']) for x in fs),C.Counter());assert ct['games']==len(js)*len(seeds)*(1 if source else 2)
    save(mode+'_LOCK.json',dict(files=fs,counts=dict(ct),plan=sha(D/'PLAN.json'),source=sha(D/'experiment49.py'),bridge=sha(D/'native49.py'),binary=sha(BIN),mask_or_cap=mask))
    print(mode,dict(ct),flush=True)
def atlas(mode):
    for g in rows(mode):
        for r in g['rows']:
            if 'utility' in r:
                yield dict(job=g['job']['id'],seed=g['seed'],bin=r['bin'],disagree=int(r['low']['original_move']!=r['low']['prior_best']),**r['utility'])

def fit():
    rs=list(atlas('TRAIN'));x=robust_fit(rs,SEEDS)
    x.update(train=evaluate(rs,x['mask']),train_lock=sha(D/'TRAIN_LOCK.json'),scope='TRAIN-only robust horizon and leave-one-checkpoint-out objective')
    save('FROZEN.json',x);print('FROZEN',json.dumps(x),flush=True)
def evaluate_split(mode):
    import numpy as np
    f=load('FROZEN.json');rs=list(atlas(mode));x=evaluate(rs,f['mask']);tr=list(atlas('TRAIN'))
    groups={j['id']:[] for j in load('PLAN.json')['jobs'] if j['split']==mode}
    for r in rs:groups[r['job']].append(r)
    a=np.array([[sum(r['cost'] for r in rr),sum(r['cost'] for r in rr if f['mask']>>r['bin']&1),len(rr)]+[math.fsum(r['gains'][h] for r in rr) for h in H]+[math.fsum(r['gains'][h] for r in rr if f['mask']>>r['bin']&1) for h in H] for rr in groups.values()],dtype=float)
    rng=np.random.default_rng(SEED+len(mode));n=len(a);v=a[rng.integers(0,n,(8000,n))].sum(axis=1)
    p=np.divide(v[:,1],v[:,0],out=np.zeros(len(v)),where=v[:,0]>0)
    for i,h in enumerate(H):
        d=np.divide(v[:,6+i]-p*v[:,3+i],v[:,2],out=np.zeros(len(v)),where=v[:,2]>0)
        x['horizons'][h]['mean_excess_CI']=list(map(float,np.quantile(d,[.025,.975])))
        base=math.fsum(r['gains'][h] for r in tr)/len(tr)
        def avg(key,z):
            a=[r['gains'][h] for r in tr if r[key]==z];return math.fsum(a)/len(a) if a else base
        pred={k:{z:avg(k,z) for z in range(nz)} for k,nz in [('bin',8),('disagree',2)]}
        x['horizons'][h]['mse']={k:math.fsum((r['gains'][h]-fn(r))**2 for r in rs)/len(rs) for k,fn in [('constant',lambda r:base),('bin',lambda r:pred['bin'][r['bin']]),('disagreement',lambda r:pred['disagree'][r['disagree']])]}
    return x
def admit():
    x=evaluate_split('CAL');f=load('FROZEN.json')
    save('ADMISSION.json',dict(admitted=x['admitted'],mask=f['mask'] if x['admitted'] else 0,selected=f['mask'],cal=x,frozen_sha=sha(D/'FROZEN.json')))
    print('ADMISSION',json.dumps(x),flush=True)
def test():
    x=evaluate_split('TEST');save('TEST_EVAL.json',x);print('TEST',json.dumps(x),flush=True)
if __name__=='__main__':
    m=sys.argv[1]
    if m in ('fit','admit','test'):globals()[m]()
    else:run(m)
