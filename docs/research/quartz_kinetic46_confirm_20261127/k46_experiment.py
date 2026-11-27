import collections as C,gzip,json,random,sys,time,pathlib,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from k46_plan import D,ROOT,T,SEEDS,SEED,WORKERS,load,save,sha
from hybrid46 import bundle,replace
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def emit(f,x):f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n')
def rows(mode):
    lk=load(mode+'_LOCK.json')
    for x in lk['files']:
        p=D/x['file'];assert sha(p)==x['sha256'];yield from raw(p)
def nn_worker(arg):
    mode,s,i,js=arg
    from k46_engine import Puct,load_eval,from_history,won,FULL
    ev=Puct(load_eval(s));ct=C.Counter();p=D/f'{mode}_{s}_{i}.jsonl.gz';assert not p.exists();source=False
    cap=20 if mode=='NN_LOW' else 32
    def end(h):
        a,b=from_history(h);return 1 if won(a) else -1 if won(b) else 0 if a|b==FULL else None
    with gzip.open(p,'wt',compresslevel=3) as f:
        for j in js:
            for color in (j['color_order'][:1] if source else j['color_order']):
                h=j['opening'].copy();rr=[];aa=[]
                while end(h) is None:
                    side=1-2*(len(h)%2);q=side==color
                    r=ev.quartz46(h,cap) if q else ev.search(h,32,False)
                    r.update(arm='Q' if q else 'B',player=side,ply=len(h));rr.append(r);h.append(r['move'])
                    ct['nn']+=r['nn'];ct['moves']+=1;ct['common']+=r['kind']=='CERTIFICATE'
                w=end(h);emit(f,dict(job=j,seed=s,color=color,history=h,winner=w,score2=1+color*w,rows=rr,atlas=aa,nn=sum(r['nn'] for r in rr)));ct['games']+=1
    assert ev.ev.rows==ct['nn']+ct['atlas_extra_nn']
    print('SHARD',mode,s,i,dict(ct),flush=True)
    return dict(file=p.name,sha256=sha(p),counts=dict(ct))
def rapfi_worker(arg):
    mode,s,i,js=arg
    sys.path.insert(0,str(D.parent/'quartz_kinetic37_20261126'))
    import experiment as e
    e.BIN=T/'mcts_demo';assert sha(e.BIN)==load('PLAN.json')['rapfi_identity']['actual_binary_sha256']
    ct=C.Counter();p=D/f'{mode}_{s}_{i}.jsonl.gz';assert not p.exists()
    with e.Server(e.BIN) as server,gzip.open(p,'wt',compresslevel=3) as f:
        for j in js:
            for color in j['color_order']:
                h=j['opening'].copy();rr=[]
                while True:
                    b,side,w=e.replay(h,'freestyle')
                    if w is not None:break
                    q=side==color;t=time.perf_counter_ns();c=time.process_time_ns()
                    z=bundle(h,15) if q else None;g=z['cold']['receipt']['gate'] if z else None
                    if z and g is None and z['cert'] is not None:
                        r=dict(move=z['cert'],nn=0,visits=0,kind='PARTIAL46',votes=[],original_move=None,replacement=False,gate=None,certificate_checks=0,native=None,partial=z,proof_cpu_ns=z['cpu_ns'])
                        native_cpu=0
                    else:
                        a=server.search(h,e.seed(j,len(h)),128,q);native_cpu=sum(a['cpu_ns']);gate=a['proof']['gate'];v=e.votes(a['result']) or []
                        r=dict(move=a['best'],nn=a['NN_calls'],visits=a['visits'],kind='CERTIFICATE' if gate is not None else a['cause'],votes=v,gate=gate,certificate_checks=a.get('certificate_checks',0),native=e.small(a))
                        if z:
                            assert g==gate
                            o=r['move'];r['move']=replace(o,v,z['bounds'],15) if gate is None else o
                            r.update(original_move=o,replacement=r['move']!=o,partial=z if gate is None else None,proof_cpu_ns=z['cpu_ns'])
                    r.update(cpu_ns=time.process_time_ns()-c+native_cpu,wall_ns=time.perf_counter_ns()-t,arm='Q' if q else 'B',player=side,ply=len(h));rr.append(r);h.append(r['move'])
                    ct['nn']+=r['nn'];ct['moves']+=1;ct['common']+=r['kind']=='CERTIFICATE'
                emit(f,dict(job=j,seed=0,color=color,history=h,winner=w,score2=1+color*w,rows=rr,atlas=[],nn=sum(r['nn'] for r in rr)));ct['games']+=1
    print('SHARD',mode,s,i,dict(ct),flush=True)
    return dict(file=p.name,sha256=sha(p),counts=dict(ct))
def run(mode):
    pl=load('PLAN.json');nn=mode.startswith('NN');split=mode.split('_')[1];split='ARENA' if split in ('FIXED','LOW','ARENA') else split
    js=[j for j in pl['jobs' if nn else 'rapfi_jobs'] if j['split']==split]
    seeds=SEEDS if nn else (0,);tasks=[(mode,s,i,js[i::4]) for s in seeds for i in range(4)]
    random.Random(SEED+len(mode)).shuffle(tasks)
    with ProcessPoolExecutor(max_workers=WORKERS if nn else 4,mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(nn_worker if nn else rapfi_worker,tasks))
    ct=sum((C.Counter(x['counts']) for x in fs),C.Counter())
    save(mode+'_LOCK.json',dict(files=fs,counts=dict(ct),plan=sha(D/'PLAN.json'),source=sha(D/'k46_experiment.py'),engine=sha(D/'k46_engine.py'),hybrid=sha(D/'hybrid46.py'),library=sha(T/'pair46.so')))
    print(mode,dict(ct),flush=True)

if __name__=='__main__':run(sys.argv[1])
