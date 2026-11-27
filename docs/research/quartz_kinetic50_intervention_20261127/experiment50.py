"""Two separate Quartz-vs-baseline games per matched NN episode; Rapfi strata."""
import collections as C,gzip,json,random,sys,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from k50_plan import D,T,SEEDS,SEED,WORKERS,load,save,sha
from native50 import NNServer,rapfi_module,rsearch,BIN
from intervention50 import request_mask,pair
def emit(f,x):f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n')
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def rows(mode):
    for x in load(mode+'_LOCK.json')['files']:
        p=D/x['file'];assert sha(p)==x['sha256'];yield from raw(p)
def nn_worker(arg):
    mode,seed,i,js=arg
    from k46_engine import load_eval,from_history,won,FULL
    ev=load_eval(seed);ct=C.Counter();path=D/f'{mode}_{seed}_{i}.jsonl.gz';assert not path.exists()
    def end(h):
        a,b=from_history(h);return 1 if won(a) else -1 if won(b) else 0 if a|b==FULL else None
    with NNServer(ev) as srv,gzip.open(path,'wt',compresslevel=3) as f:
        for j in js:
            for color in j['color_order']:
                games={}
                for arm in j['arm_order']:
                    h=j['opening'].copy();rr=[];eligible=0;event=None;k=j['ordinal']
                    while end(h) is None:
                        side=1-2*(len(h)%2);q=side==color
                        mask=request_mask(eligible,k,arm) if q else None
                        r=srv.nested(h,mask=mask) if mask is not None else srv.search(h,q)
                        assert r['history']==h and r['move'] not in h
                        if r.get('low'):
                            assert q and mask is not None
                            eligible+=1;r['eligible_index']=eligible
                            if eligible==k:event=len(rr)
                            assert bool(r['extended'])==(arm=='EXTEND' and eligible==k)
                            assert r['physical_nn']==r['nn']
                        r.update(arm='Q' if q else 'B',player=side,ply=len(h));rr.append(r);h.append(r['move'])
                        ct['nn']+=r['nn'];ct['moves']+=1;ct['common']+=r['kind']=='CERTIFICATE'
                    w=end(h);g=dict(job=j,seed=seed,color=color,policy=arm,event=event,history=h,winner=w,score2=1+color*w,rows=rr,atlas=[],nn=sum(r['nn'] for r in rr))
                    games[arm]=g;ct['games']+=1
                z=pair(games['CONTROL'],games['EXTEND']);ct['pairs']+=1;ct['reached']+=z['reached'];ct['changed']+=z['changed'];ct['local_nn']+=z['local_nn'];ct['delta2']+=z['delta2']
                emit(f,dict(control=games['CONTROL'],extend=games['EXTEND'],pair=z))
    assert ev.rows==ct['nn']
    print('SHARD',mode,seed,i,dict(ct),flush=True)
    return dict(file=path.name,sha256=sha(path),counts=dict(ct))
def rapfi_worker(arg):
    mode,s,i,js=arg;e=rapfi_module()
    ct=C.Counter();path=D/f'{mode}_{s}_{i}.jsonl.gz';assert not path.exists()
    with e.Server(BIN) as server,gzip.open(path,'wt',compresslevel=3) as f:
        for j in js:
            for color in j['color_order']:
                h=j['opening'].copy();rr=[]
                while True:
                    _,side,w=e.replay(h,'freestyle')
                    if w is not None:break
                    q=side==color;r=rsearch(e,server,h,e.seed(j,len(h)),q,96)
                    r.update(arm='Q' if q else 'B',player=side,ply=len(h));rr.append(r);h.append(r['move'])
                    ct['nn']+=r['nn'];ct['moves']+=1;ct['common']+=r['kind']=='CERTIFICATE'
                emit(f,dict(job=j,seed=0,color=color,history=h,winner=w,score2=1+color*w,rows=rr,atlas=[],nn=sum(r['nn'] for r in rr)));ct['games']+=1
    print('SHARD',mode,s,i,dict(ct),flush=True)
    return dict(file=path.name,sha256=sha(path),counts=dict(ct))
def run(mode):
    pl=load('PLAN.json');nn=mode!='R96';js=[j for j in pl['jobs' if nn else 'rapfi_jobs'] if j['split']==(mode if nn else 'ARENA')]
    seeds=SEEDS if nn else (0,);tasks=[(mode,s,i,js[i::4]) for s in seeds for i in range(4)]
    random.Random(SEED+len(mode)).shuffle(tasks)
    assert sha(BIN)==load('BINARY_LOCK.json')['binary_sha256']
    with ProcessPoolExecutor(max_workers=WORKERS if nn else 4,mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(nn_worker if nn else rapfi_worker,tasks))
    ct=C.Counter()
    for x in fs:ct.update(x['counts'])
    assert ct['games']==len(js)*len(seeds)*(4 if nn else 2)
    save(mode+'_LOCK.json',dict(files=fs,counts=dict(ct),plan=sha(D/'PLAN.json'),source=sha(D/'experiment50.py'),bridge=sha(D/'native50.py'),binary=sha(BIN)))
    print(mode,dict(ct),flush=True)
if __name__=='__main__':run(sys.argv[1])
