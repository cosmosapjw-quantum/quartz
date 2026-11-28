"""Full legal intervention atlas and actual same-evaluator arena."""
from plan66 import D,T,SEEDS,EXTRA,NOVEL,ARMS,DEV,sha,read
from eval66 import n,load_eval,from_history,won,FULL,select,allocation,operation
import json,gzip,sys,collections as C,random,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
def req(s,h,policy):
    if policy=='H4':return s.early(h,4,1)
    if policy=='FREE':return select(s,h)
    if policy=='MA4':return allocation(s,h,3)
    return operation(s,h,5 if policy=='FOCUS' else 6)
def play(s,ev,start,color,policy,force=None,dev=False):
    h=list(start);rr=[];e0=ev.rows
    while True:
        a,b=from_history(h);w=1 if won(a) else -1 if won(b) else 0 if a|b==FULL else None
        if w is not None:break
        side=1-2*(len(h)%2);q=side==color
        r=req(s,h,policy if not dev or not rr else 'FREE') if q else s.search(h,False)
        assert r['history']==h and r['move'] not in h
        m=r['move'] if force is None or rr else force
        assert m not in h and 0<=m<49
        r.update(arm='Q' if q else 'B',player=side,ply=len(h),played=m);rr.append(r);h.append(m)
    g=dict(policy=policy,history=h,rows=rr,winner=w,score2=1+color*w,actual_nn=sum(r['actual_nn'] for r in rr),nn=sum(r['nn'] for r in rr),targets=0,switches=0)
    assert ev.rows-e0==g['actual_nn'];return g
def sem(x):
    if isinstance(x,dict):return {k:sem(v) for k,v in x.items() if not k.endswith('_ns')}
    if isinstance(x,list):return [sem(v) for v in x]
    return x
def worker(arg):
    mode,seed,i,js=arg;ev=load_eval(seed);path=T/f'{mode}_{seed}_{i}.jsonl.gz';assert not path.exists();ct=C.Counter()
    def count(g):
        ct['games']+=1;ct['moves']+=len(g['rows']);ct['physical']+=g['actual_nn'];ct['logical']+=g['nn']
    with n.NNServer(ev,bin=T/'mcts_demo') as s,gzip.open(path,'wt',compresslevel=3) as f:
        for j in js:
            if mode.startswith('DEV'):
                # All legal actions are played separately. Randomize order as precommitted.
                gs={str(m):play(s,ev,j['history'],j['color'],'MA4',m,True) for m in j['action_order']}
                ops={arm:play(s,ev,j['history'],j['color'],arm,dev=True) for arm in j['operation_order']}
                for g in list(gs.values())+list(ops.values()):count(g)
                for arm,g in ops.items():
                    m=g['rows'][0]['played'];v=gs[str(m)]
                    assert g['score2']==v['score2'] and g['history']==v['history'] and sem(g['rows'][1:])==sem(v['rows'][1:])
                ct['roots']+=1;ct['forced_games']+=len(gs);ct['operation_games']+=len(ops)
                f.write(json.dumps(dict(job=j,forced=gs,operations=ops),separators=(',',':'),allow_nan=False)+'\n')
            else:
                for color in j['color_order']:
                    gs={}
                    for arm in j['arm_order']:
                        g=play(s,ev,j['opening'],color,arm);g.update(job=j,color=color,seed=seed);gs[arm]=g;count(g)
                    f.write(json.dumps(gs,separators=(',',':'),allow_nan=False)+'\n');ct['episodes']+=1
    assert ev.rows==ct['physical']
    print(mode,'SHARD',seed,i,dict(ct),flush=True);return dict(file=path.name,sha256=sha(path),counts=dict(ct))
def main(mode):
    p=read(D/'PLAN.json');lock=read(D/'BINARY_LOCK.json');assert sha(T/'mcts_demo')==lock['binary_sha256'];assert lock['plan_sha256']==sha(D/'PLAN.json')
    if mode=='DEV_SMOKE':
        j=min(p['roots'],key=lambda r:len(r['action_order']));tasks=[(mode,j['seed'],99,[j])]
    elif mode=='DEV':
        tasks=[]
        for s in SEEDS+EXTRA+NOVEL:
            js=[j for j in p['roots'] if j['seed']==s]
            tasks.extend((mode,s,i,js[i::2]) for i in range(2) if js[i::2])
    else:
        assert mode in ('MAIN','ARCHIVE','NOVEL','SMOKE')
        panel='MAIN' if mode=='SMOKE' else mode;js=[j for j in p['jobs'] if j['panel']==panel];seeds=SEEDS if panel=='MAIN' else EXTRA if panel=='ARCHIVE' else NOVEL
        tasks=[(mode,SEEDS[0],0,js[:1])] if mode=='SMOKE' else [(mode,s,i,js[i::3]) for s in seeds for i in range(3) if js[i::3]]
    random.Random(202611286601).shuffle(tasks)
    if len(tasks)==1:fs=[worker(tasks[0])]
    else:
        with ProcessPoolExecutor(max_workers=p['train_workers'] if mode=='DEV' else p['workers'],mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(worker,tasks))
    ct=C.Counter()
    for m in fs:ct.update(m['counts'])
    if mode.startswith('DEV'):
        rs=p['roots'] if mode=='DEV' else [min(p['roots'],key=lambda r:len(r['action_order']))]
        assert ct['roots']==len(rs) and ct['forced_games']==sum(len(j['action_order']) for j in rs) and ct['operation_games']==2*len(rs)
    else:assert ct['games']=={'SMOKE':10,'MAIN':1680,'ARCHIVE':360,'NOVEL':360}[mode]
    (T/(mode+'_LOCK.json')).write_text(json.dumps(dict(files=fs,counts=dict(ct),plan=sha(D/'PLAN.json'),source=sha(D/'games66.py'),eval_source=sha(D/'eval66.py'),binary=lock['binary_sha256']),indent=2,sort_keys=True)+'\n')
    print(mode,dict(ct),flush=True)
if __name__=='__main__':main(sys.argv[1])
