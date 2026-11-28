"""Actual same-evaluator B32 arena, separate paired historical root probes."""
from plan65 import D,T,SEEDS,EXTRA,NOVEL,ARMS,DEV,sha,read
from eval65 import n,load_eval,from_history,won,FULL,select,allocation
import json,gzip,sys,collections as C,random,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
def req(s,h,policy):
    if policy=='H4':return s.early(h,4,1)
    if policy=='FREE':return select(s,h)
    return allocation(s,h,DEV.index(policy))
def play(s,ev,start,color,policy):
    h=list(start);rr=[];e0=ev.rows
    while True:
        a,b=from_history(h);w=1 if won(a) else -1 if won(b) else 0 if a|b==FULL else None
        if w is not None:break
        side=1-2*(len(h)%2);q=side==color
        r=req(s,h,policy) if q else s.search(h,False)
        assert r['history']==h and r['move'] not in h
        m=r['move'];r.update(arm='Q' if q else 'B',player=side,ply=len(h),played=m);rr.append(r);h.append(m)
    g=dict(policy=policy,history=h,rows=rr,winner=w,score2=1+color*w,actual_nn=sum(r['actual_nn'] for r in rr),nn=sum(r['nn'] for r in rr),targets=0,switches=0)
    assert ev.rows-e0==g['actual_nn'];return g
def worker(arg):
    mode,seed,i,js=arg;ev=load_eval(seed);path=T/f'{mode}_{seed}_{i}.jsonl.gz';assert not path.exists();ct=C.Counter()
    with n.NNServer(ev,bin=T/'mcts_demo') as s,gzip.open(path,'wt',compresslevel=3) as f:
        for j in js:
            if mode.startswith('DEV'):
                order=random.Random(655000+j['id']).sample(list(DEV),len(DEV))
                rs={arm:req(s,j['history'],arm) for arm in order}
                for r in rs.values():ct['requests']+=1;ct['physical']+=r['actual_nn'];ct['logical']+=r['nn']
                ct['roots']+=1
                f.write(json.dumps(dict(job=j,arms=rs,order=order),separators=(',',':'),allow_nan=False)+'\n')
            else:
                for color in j['color_order']:
                    gs={}
                    for arm in j['arm_order']:
                        g=play(s,ev,j['opening'],color,arm);g.update(job=j,color=color,seed=seed);gs[arm]=g
                        ct['games']+=1;ct['moves']+=len(g['rows']);ct['physical']+=g['actual_nn'];ct['logical']+=g['nn']
                    f.write(json.dumps(gs,separators=(',',':'),allow_nan=False)+'\n');ct['episodes']+=1
    assert ev.rows==ct['physical']
    print(mode,'SHARD',seed,i,dict(ct),flush=True);return dict(file=path.name,sha256=sha(path),counts=dict(ct))
def main(mode):
    p=read(D/'PLAN.json');lock=read(D/'BINARY_LOCK.json');assert sha(T/'mcts_demo')==lock['binary_sha256'];assert lock['plan_sha256']==sha(D/'PLAN.json')
    if mode=='DEV_SMOKE':
        j=p['roots'][0];tasks=[(mode,j['seed'],99,[j])]
    elif mode=='DEV':
        tasks=[]
        for s in SEEDS+EXTRA:
            js=[j for j in p['roots'] if j['seed']==s]
            tasks.extend((mode,s,i,js[i::2]) for i in range(2) if js[i::2])
    else:
        assert mode in ('MAIN','ARCHIVE','NOVEL','SMOKE')
        panel='MAIN' if mode=='SMOKE' else mode;js=[j for j in p['jobs'] if j['panel']==panel];seeds=SEEDS if panel=='MAIN' else EXTRA if panel=='ARCHIVE' else NOVEL
        tasks=[(mode,SEEDS[0],0,js[:1])] if mode=='SMOKE' else [(mode,s,i,js[i::3]) for s in seeds for i in range(3) if js[i::3]]
    random.Random(202611286501).shuffle(tasks)
    if len(tasks)==1:fs=[worker(tasks[0])]
    else:
        with ProcessPoolExecutor(max_workers=p['train_workers'] if mode=='DEV' else p['workers'],mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(worker,tasks))
    ct=C.Counter()
    for m in fs:ct.update(m['counts'])
    if mode.startswith('DEV'):assert ct['requests']==(5 if mode=='DEV_SMOKE' else 2770)
    else:assert ct['games']=={'SMOKE':14,'MAIN':2352,'ARCHIVE':504,'NOVEL':504}[mode]
    (T/(mode+'_LOCK.json')).write_text(json.dumps(dict(files=fs,counts=dict(ct),plan=sha(D/'PLAN.json'),source=sha(D/'games65.py'),eval_source=sha(D/'eval65.py'),binary=lock['binary_sha256']),indent=2,sort_keys=True)+'\n')
    print(mode,dict(ct),flush=True)
if __name__=='__main__':main(sys.argv[1])
