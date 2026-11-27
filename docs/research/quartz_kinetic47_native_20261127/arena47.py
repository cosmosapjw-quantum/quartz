"""Actual fresh games: native Q20 vs B32 on same NN; native Rapfi Q128 vs B128."""
import collections as C,gzip,json,random,sys,time,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from k47_plan import D,T,SEEDS,SEED,WORKERS,load,save,sha
from native47 import NNServer,rapfi_module,rsearch,BIN,REF
def emit(f,x):f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n')
def nn_worker(arg):
    mode,s,i,js=arg
    from k46_engine import load_eval,from_history,won,FULL
    ev=load_eval(s);ct=C.Counter();path=D/f'{mode}_{s}_{i}.jsonl.gz';assert not path.exists()
    def end(h):
        a,b=from_history(h);return 1 if won(a) else -1 if won(b) else 0 if a|b==FULL else None
    with NNServer(ev) as srv,gzip.open(path,'wt',compresslevel=3) as f:
        for j in js:
            for color in j['color_order']:
                h=j['opening'].copy();rr=[]
                while end(h) is None:
                    side=1-2*(len(h)%2);q=side==color
                    r=srv.search(h,q)
                    assert r['history']==h and r['move'] not in h
                    r.update(arm='Q' if q else 'B',player=side,ply=len(h));rr.append(r);h.append(r['move'])
                    ct['nn']+=r['nn'];ct['moves']+=1;ct['common']+=r['kind']=='CERTIFICATE'
                w=end(h);emit(f,dict(job=j,seed=s,color=color,history=h,winner=w,score2=1+color*w,rows=rr,atlas=[],nn=sum(r['nn'] for r in rr)));ct['games']+=1
    assert ev.rows==ct['nn']
    print('SHARD',mode,s,i,dict(ct),flush=True)
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
                    q=side==color;r=rsearch(e,server,h,e.seed(j,len(h)),q)
                    assert r['move'] not in h
                    r.update(arm='Q' if q else 'B',player=side,ply=len(h));rr.append(r);h.append(r['move'])
                    ct['nn']+=r['nn'];ct['moves']+=1;ct['common']+=r['kind']=='CERTIFICATE'
                emit(f,dict(job=j,seed=0,color=color,history=h,winner=w,score2=1+color*w,rows=rr,atlas=[],nn=sum(r['nn'] for r in rr)));ct['games']+=1
    print('SHARD',mode,s,i,dict(ct),flush=True)
    return dict(file=path.name,sha256=sha(path),counts=dict(ct))
def run(mode):
    assert mode in ('NN_LOW','R_ARENA')
    pl=load('PLAN.json');nn=mode=='NN_LOW';js=pl['jobs' if nn else 'rapfi_jobs']
    assert len(js)==(48 if nn else 64) and all(j['split']=='ARENA' for j in js)
    assert sha(BIN)==load('NN_PARITY_LOCK.json')['binary']==load('R_PARITY_LOCK.json')['binary']==load('PROOF_PARITY_LOCK.json')['binary']
    seeds=SEEDS if nn else (0,);tasks=[(mode,s,i,js[i::4]) for s in seeds for i in range(4)]
    random.Random(SEED+len(mode)).shuffle(tasks)
    with ProcessPoolExecutor(max_workers=WORKERS if nn else 4,mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(nn_worker if nn else rapfi_worker,tasks))
    ct=sum((C.Counter(x['counts']) for x in fs),C.Counter());assert ct['games']==(672 if nn else 128)
    save(mode+'_LOCK.json',dict(files=fs,counts=dict(ct),plan=sha(D/'PLAN.json'),source=sha(D/'arena47.py'),bridge=sha(D/'native47.py'),binary=sha(BIN)))
    print(mode,dict(ct),flush=True)
if __name__=='__main__':run(sys.argv[1])
