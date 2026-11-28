from plan61 import D,T,P58,SEEDS,ARMS,sha,read
import json,gzip,sys,collections as C,random,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(P58))
import native58 as n
n.T=T
def play(s,ev,j,color,arm,tables):
    from k46_engine import from_history,won,FULL
    history=j['opening'].copy();rr=[];e0=ev.rows;attempts=[];events=[]
    while True:
        a,b=from_history(history);w=1 if won(a) else -1 if won(b) else 0 if a|b==FULL else None
        if w is not None:break
        side=1-2*(len(history)%2);q=side==color
        if not q:r=s.search(history,False)
        elif arm=='CONTROL':r=s.search(history,True)
        elif arm=='H4':r=s.early(history,4,1)
        else:r=s.cached_request(dict(cmd='research_gomoku7_k61',history=history,cap=20,quartz=True,trace=False,atlas=False,mask=0,early=dict(horizon=8,op=6),extend61=tables[arm]),True,True)
        assert r['history']==history and r['move'] not in history and 'target' not in r
        if 'early' in r:
            assert q and arm!='CONTROL';attempts.append(len(rr))
            if r['early']['reached']:events.append(len(rr))
        r.update(arm='Q' if q else 'B',player=side,ply=len(history));rr.append(r);history.append(r['move'])
    g=dict(job=j,color=color,policy=arm,attempts=attempts,events=events,history=history,rows=rr,winner=w,score2=1+color*w,actual_nn=sum(r['actual_nn'] for r in rr),nn=sum(r['nn'] for r in rr))
    assert ev.rows-e0==g['actual_nn'];return g
def worker(arg):
    seed,i,js,smoke=arg
    from k46_engine import load_eval
    p=read(D/'PLAN.json');tab=read(D/'TABLE.json');tables={'SELECT':tab['table'],'CONTRAST':tab['contrast']};ev=load_eval(seed);ct=C.Counter();path=T/f'{"SMOKE" if smoke else "ARENA"}_{seed}_{i}.jsonl.gz';assert not path.exists()
    with n.NNServer(ev,bin=T/'mcts_demo') as s,gzip.open(path,'wt',compresslevel=3) as f:
        for j in js:
            for color in j['color_order']:
                gs={}
                for arm in j['arm_order']:
                    g=play(s,ev,j,color,arm,tables);g['seed']=seed;gs[arm]=g
                    ct['games']+=1;ct['moves']+=len(g['rows']);ct['physical']+=g['actual_nn'];ct['logical']+=g['nn']
                    ct[arm+'_attempts']+=len(g['attempts']);ct[arm+'_events']+=len(g['events'])
                f.write(json.dumps(gs,separators=(',',':'),allow_nan=False)+'\n');ct['episodes']+=1
    assert ev.rows==ct['physical']
    print('ARENA SHARD',seed,i,dict(ct),flush=True)
    return dict(file=path.name,sha256=sha(path),counts=dict(ct))
def main(smoke=False):
    p=read(D/'PLAN.json');lock=read(D/'BINARY_LOCK.json');assert sha(T/'mcts_demo')==lock['binary_sha256']
    assert sha(D/'TABLE.json')==lock['table_sha256']
    js=p['jobs'];tasks=[(SEEDS[0],0,js[:1],True)] if smoke else [(s,i,js[i::4],False) for s in SEEDS for i in range(4)]
    random.Random(202611286102).shuffle(tasks)
    if smoke:fs=[worker(tasks[0])]
    else:
        with ProcessPoolExecutor(max_workers=p['workers'],mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(worker,tasks))
    ct=C.Counter()
    for x in fs:ct.update(x['counts'])
    assert ct['games']==(8 if smoke else 2688)
    name='SMOKE_LOCK.json' if smoke else 'ARENA_LOCK.json'
    (T/name).write_text(json.dumps(dict(files=fs,counts=dict(ct),plan=sha(D/'PLAN.json'),source=sha(D/'arena61.py'),binary=lock['binary_sha256'],table=sha(D/'TABLE.json')),indent=2,sort_keys=True)+'\n')
    print(name,dict(ct),flush=True)
if __name__=='__main__':main('--smoke' in sys.argv)
