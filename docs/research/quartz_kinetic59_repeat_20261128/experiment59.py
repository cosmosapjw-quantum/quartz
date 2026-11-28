"""Repeated frozen STOP; every actual game is one Quartz arm vs B32."""
import collections as C,gzip,json,random,sys,multiprocessing as mp,os
from concurrent.futures import ProcessPoolExecutor
from plan59 import D,T,P,ROOT,SEEDS,ARMS,sha
from policy59 import operation,advance
sys.path.insert(0,str(P))
import native58 as native
native.T=T
def load():return json.loads((D/'PLAN.json').read_text())
def emit(f,x):f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n')
def sem(x):
    if isinstance(x,dict):return {k:sem(v) for k,v in x.items() if not k.endswith('_ns')}
    if isinstance(x,list):return [sem(v) for v in x]
    return x
def play(srv,ev,j,color,arm):
    from k46_engine import from_history,won,FULL
    h=j['opening'].copy();rr=[];used=False;attempts=[];events=[];e0=ev.rows
    while True:
        a,b=from_history(h);w=1 if won(a) else -1 if won(b) else 0 if a|b==FULL else None
        if w is not None:break
        side=1-2*(len(h)%2);q=side==color;mode,op=operation(arm,q,used)
        r=srv.early(h,j['horizon'],op) if mode=='early' else srv.search(h,q)
        assert r['history']==h and r['move'] not in h and 0<=r['move']<49 and 'target' not in r
        if 'early' in r:
            assert q and mode=='early' and r['early']['op']==op
            attempts.append(len(rr))
            if r['early']['reached']:events.append(len(rr))
        used=advance(used,r)
        r.update(arm='Q' if q else 'B',player=side,ply=len(h));rr.append(r);h.append(r['move'])
    g=dict(job=j,color=color,policy=arm,attempts=attempts,events=events,history=h,winner=w,score2=1+color*w,rows=rr,nn=sum(r['nn'] for r in rr),actual_nn=sum(r['actual_nn'] for r in rr))
    assert g['actual_nn']==ev.rows-e0
    return g
def same_until_first(f,r):
    if not f['attempts']:
        assert sem(f['rows'])==sem(r['rows']) and f['history']==r['history']
    else:
        i=f['attempts'][0]
        assert r['attempts'][0]==i and sem(f['rows'][:i+1])==sem(r['rows'][:i+1])
    if len(r['attempts'])<=1:
        assert sem(f['rows'])==sem(r['rows']) and f['history']==r['history']
def worker(arg):
    seed,i,js,smoke=arg
    from k46_engine import load_eval
    plan=load();ev=load_eval(seed);ct=C.Counter()
    path=T/f'{"SMOKE" if smoke else "NN"}_{seed}_{i}.jsonl.gz'
    assert not path.exists(),path
    with native.NNServer(ev,bin=plan['binary']) as srv,gzip.open(path,'wt',compresslevel=3) as f:
        for j in js:
            for color in j['color_order']:
                games={}
                for arm in j['arm_order']:
                    g=play(srv,ev,j,color,arm);g['seed']=seed;games[arm]=g
                    ct['physical']+=g['actual_nn'];ct['logical']+=g['nn'];ct['moves']+=len(g['rows']);ct['games']+=1
                    ct[arm+'_attempts']+=len(g['attempts']);ct[arm+'_events']+=len(g['events'])
                same_until_first(games['FIRST'],games['REPEAT'])
                emit(f,games);ct['episodes']+=1
    assert ev.rows==ct['physical']
    print('SHARD',seed,i,dict(ct),flush=True)
    return dict(file=path.name,sha256=sha(path),counts=dict(ct))
def main(smoke=False):
    p=load();assert sha(p['binary'])==p['binary_sha256']
    js=p['jobs'];tasks=[(SEEDS[0],0,js[:2],True)] if smoke else [(s,i,js[i::4],False) for s in SEEDS for i in range(4)]
    random.Random(202611285901).shuffle(tasks)
    if smoke:fs=[worker(tasks[0])]
    else:
        with ProcessPoolExecutor(max_workers=p['workers'],mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(worker,tasks))
    ct=C.Counter()
    for x in fs:ct.update(x['counts'])
    assert ct['games']==(12 if smoke else 2688)
    name='SMOKE_LOCK.json' if smoke else 'NN_LOCK.json'
    (T/name).write_text(json.dumps(dict(files=fs,counts=dict(ct),plan=sha(D/'PLAN.json'),source=sha(D/'experiment59.py'),policy=sha(D/'policy59.py'),binary=p['binary_sha256']),indent=2,sort_keys=True)+'\n')
    print(name,dict(ct),flush=True)
if __name__=='__main__':main('--smoke' in sys.argv)
