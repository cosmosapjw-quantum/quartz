"""Native frozen search with measured Python readout. All games vs identical evaluator B32."""
from plan67 import D,T,SEEDS,EXTRA,NOVEL,TRANSFER,ARMS,DEV,sha,read
from eval67 import n,load_eval,from_history,won,FULL,select,allocation,operation
from retention67 import apply
import json,gzip,sys,collections as C,random,multiprocessing as mp,time
from concurrent.futures import ProcessPoolExecutor
CAL={'CALV':'VOTE','CALF':'FOCUS','CALX':'EXPAND'}
def req(s,h,policy,model):
    native=CAL.get(policy,policy)
    if native=='H4':r=s.early(h,4,1)
    elif native=='FREE':r=select(s,h)
    elif native in ('MA4','VOTE'):r=allocation(s,h,3)
    else:
        assert native in ('FOCUS','EXPAND');r=operation(s,h,5 if native=='FOCUS' else 6)
    t=time.perf_counter_ns();c=time.process_time_ns()
    if policy in CAL:
        m,z=apply(r,native,model)
        if z is not None:r['retention67']=z
    elif native=='VOTE' and 'allocation65' in r:m=r['allocation65']['fallback']
    else:m=r['move']
    r['cpu_ns']+=time.process_time_ns()-c;r['wall_ns']+=time.perf_counter_ns()-t
    return r,m
def play(s,ev,start,color,policy,model=None,dev=False):
    h=list(start);rr=[];e0=ev.rows
    while True:
        a,b=from_history(h);w=1 if won(a) else -1 if won(b) else 0 if a|b==FULL else None
        if w is not None:break
        side=1-2*(len(h)%2);q=side==color
        if q:r,m=req(s,h,policy if not dev or not rr else 'FREE',model)
        else:r=s.search(h,False);m=r['move']
        assert r['history']==h and r['move'] not in h and m not in h and 0<=m<49
        r.update(arm='Q' if q else 'B',player=side,ply=len(h),played=m);rr.append(r);h.append(m)
    g=dict(policy=policy,history=h,rows=rr,winner=w,score2=1+color*w,actual_nn=sum(r['actual_nn'] for r in rr),nn=sum(r['nn'] for r in rr),targets=0,switches=0)
    assert ev.rows-e0==g['actual_nn'];return g
def sem(x):
    if isinstance(x,dict):return {k:sem(v) for k,v in x.items() if not k.endswith('_ns')}
    if isinstance(x,list):return [sem(v) for v in x]
    return x
def worker(arg):
    mode,seed,i,js=arg;ev=load_eval(seed);path=T/f'{mode}_{seed}_{i}.jsonl.gz';assert not path.exists();ct=C.Counter()
    model=read(D/'MODEL.json') if not mode.startswith('DEV') else None
    def count(g):
        ct['games']+=1;ct['moves']+=len(g['rows']);ct['physical']+=g['actual_nn'];ct['logical']+=g['nn']
    with n.NNServer(ev,bin=T/'mcts_demo') as s,gzip.open(path,'wt',compresslevel=3) as f:
        for j in js:
            if mode.startswith('DEV'):
                gs={arm:play(s,ev,j['history'],j['color'],arm,dev=True) for arm in j['arm_order']}
                for g in gs.values():count(g)
                for x,g in gs.items():
                    for y,v in gs.items():
                        if g['rows'][0]['played']==v['rows'][0]['played']:
                            assert g['score2']==v['score2'] and g['history']==v['history'] and sem(g['rows'][1:])==sem(v['rows'][1:])
                ct['roots']+=1
                f.write(json.dumps(dict(job=j,games=gs),separators=(',',':'),allow_nan=False)+'\n')
            else:
                for color in j['color_order']:
                    gs={}
                    for arm in j['arm_order']:
                        g=play(s,ev,j['opening'],color,arm,model);g.update(job=j,color=color,seed=seed);gs[arm]=g;count(g)
                    f.write(json.dumps(gs,separators=(',',':'),allow_nan=False)+'\n');ct['episodes']+=1
    assert ev.rows==ct['physical']
    print(mode,'SHARD',seed,i,dict(ct),flush=True);return dict(file=path.name,sha256=sha(path),counts=dict(ct))
def main(mode):
    p=read(D/'PLAN.json');lock=read(D/'BINARY_LOCK.json');assert sha(T/'mcts_demo')==lock['binary_sha256'];assert lock['plan_sha256']==sha(D/'PLAN.json')
    if mode=='DEV_SMOKE':
        j=p['roots'][0];tasks=[(mode,j['seed'],99,[j])]
    elif mode=='DEV':
        tasks=[]
        for s in SEEDS+EXTRA+NOVEL:
            js=[j for j in p['roots'] if j['seed']==s]
            tasks.extend((mode,s,i,js[i::3]) for i in range(3) if js[i::3])
    else:
        assert mode in ('MAIN','ARCHIVE','NOVEL','TRANSFER','SMOKE')
        panel='MAIN' if mode=='SMOKE' else mode;js=[j for j in p['jobs'] if j['panel']==panel];seeds={'MAIN':SEEDS,'ARCHIVE':EXTRA,'NOVEL':NOVEL,'TRANSFER':TRANSFER}[panel]
        tasks=[(mode,SEEDS[0],0,js[:1])] if mode=='SMOKE' else [(mode,s,i,js[i::3]) for s in seeds for i in range(3) if js[i::3]]
    random.Random(202611286701).shuffle(tasks)
    if len(tasks)==1:fs=[worker(tasks[0])]
    else:
        with ProcessPoolExecutor(max_workers=p['train_workers'] if mode=='DEV' else p['workers'],mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(worker,tasks))
    ct=C.Counter()
    for m in fs:ct.update(m['counts'])
    expected={'DEV_SMOKE':4,'DEV':2996,'SMOKE':18,'MAIN':3024,'ARCHIVE':648,'NOVEL':648,'TRANSFER':648}[mode];assert ct['games']==expected
    (T/(mode+'_LOCK.json')).write_text(json.dumps(dict(files=fs,counts=dict(ct),plan=sha(D/'PLAN.json'),source=sha(D/'games67.py'),eval_source=sha(D/'eval67.py'),retention_source=sha(D/'retention67.py'),binary=lock['binary_sha256'],model=None if mode.startswith('DEV') else sha(D/'MODEL.json')),indent=2,sort_keys=True)+'\n')
    print(mode,dict(ct),flush=True)
if __name__=='__main__':main(sys.argv[1])
