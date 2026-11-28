"""Actual independent Quartz-v-identical-B32 games. Forced rival is only a root intervention."""
from plan64 import D,T,SEEDS,EXTRA,NOVEL,ARMS,DEV,sha,read
from eval64 import n,load_eval,from_history,won,FULL,target,select,marginal
import json,gzip,sys,collections as C,random,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
def play(s,ev,start,color,policy,root=None):
    h=list(start);rr=[];e0=ev.rows;ct=C.Counter()
    table=read(D/'TABLE.json')['table'] if policy=='ROBUST' else None
    while True:
        a,b=from_history(h);w=1 if won(a) else -1 if won(b) else 0 if a|b==FULL else None
        if w is not None:break
        side=1-2*(len(h)%2);q=side==color
        op=policy if root is None or not rr else 'FREE'
        if not q:r=s.search(h,False)
        elif op=='T4':r=target(s,h)
        elif op=='T2':r=marginal(s,h)
        elif op=='ROBUST':r=marginal(s,h,table)
        elif op=='K63':r=select(s,h,[2,0,0,1,2,0])
        elif op in ('FREE','OTHER'):r=select(s,h)
        else:
            assert op=='H4';r=s.early(h,4,1)
        assert r['history']==h and r['move'] not in h
        m=r['move']
        if root is not None and not rr and policy=='OTHER':
            assert q and r['early']['reached'] and m==root['free'];x=r['early']['at'];m=x['rival'] if m==x['move'] else x['move'];assert m not in h
        if 'target' in r:
            assert q and op in ('T2','T4','K63','ROBUST') and r['target']['extra_nn']<=4
            ct['targets']+=1;ct['switches']+=r['target']['chosen']!=r['target']['incumbent']
        r.update(arm='Q' if q else 'B',player=side,ply=len(h),played=m);rr.append(r);h.append(m)
    g=dict(policy=policy,history=h,rows=rr,winner=w,score2=1+color*w,actual_nn=sum(r['actual_nn'] for r in rr),nn=sum(r['nn'] for r in rr),targets=ct['targets'],switches=ct['switches'])
    assert ev.rows-e0==g['actual_nn'];return g
def sem(x):
    if isinstance(x,dict):return {k:sem(v) for k,v in x.items() if not k.endswith('_ns')}
    if isinstance(x,list):return [sem(v) for v in x]
    return x
def pairs(j,gs):
    r=gs['FREE']['rows'][0];x=r['early']['at'];a,b=x['move'],x['rival'];f=r['move'];other=b if f==a else a
    assert x==j['snapshot'] and r['partial']['bounds']==j['bounds'] and r['actual_nn']==j['root_nn'] and f==j['free']
    y={k:g['score2'] for k,g in gs.items()};m={k:g['rows'][0]['played'] for k,g in gs.items()}
    assert m['FREE']==f and m['OTHER']==other
    for p in gs:
        z=gs[p]['rows'][0];assert z['early']['at']==x and m[p] in (a,b)
        for q in gs:
            if m[p]==m[q]:assert y[p]==y[q] and gs[p]['history']==gs[q]['history'] and sem(gs[p]['rows'][1:])==sem(gs[q]['rows'][1:])
    assert gs['OTHER']['rows'][0]['actual_nn']==r['actual_nn']
    extra={k:gs[k]['rows'][0]['actual_nn']-r['actual_nn'] for k in ('FREE','T2','T4')}
    assert extra['FREE']==0 and 0<=extra['T2']<=2 and 0<=extra['T4']<=4
    return dict(a=a,b=b,free=f,other=other,moves=m,score2=y,oracle2=max(y['FREE'],y['OTHER']),extra=extra)
def worker(arg):
    mode,seed,i,js=arg;ev=load_eval(seed);path=T/f'{mode}_{seed}_{i}.jsonl.gz';assert not path.exists();ct=C.Counter()
    def count(g):
        ct['games']+=1;ct['moves']+=len(g['rows']);ct['physical']+=g['actual_nn'];ct['logical']+=g['nn'];ct['targets']+=g['targets'];ct['switches']+=g['switches']
    with n.NNServer(ev,bin=T/'mcts_demo') as s,gzip.open(path,'wt',compresslevel=3) as f:
        for j in js:
            if mode.startswith('DEV'):
                gs={arm:play(s,ev,j['history'],j['color'],arm,j) for arm in j['arm_order']}
                z=pairs(j,gs);ct['roots']+=1
                for g in gs.values():count(g)
                f.write(json.dumps(dict(job=j,games=gs,pair=z),separators=(',',':'),allow_nan=False)+'\n')
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
        j=p['roots'][0];tasks=[(mode,j['seed'],99,[j])]
    elif mode=='DEV':
        tasks=[]
        for s in SEEDS+EXTRA:
            js=[j for j in p['roots'] if j['seed']==s]
            tasks.extend((mode,s,i,js[i::3]) for i in range(3) if js[i::3])
    else:
        assert mode in ('MAIN','ARCHIVE','NOVEL','SMOKE')
        panel='MAIN' if mode=='SMOKE' else mode;js=[j for j in p['jobs'] if j['panel']==panel];seeds=SEEDS if panel=='MAIN' else EXTRA if panel=='ARCHIVE' else NOVEL
        tasks=[(mode,SEEDS[0],0,js[:1])] if mode=='SMOKE' else [(mode,s,i,js[i::4]) for s in seeds for i in range(4)]
    random.Random(202611286401).shuffle(tasks)
    if len(tasks)==1:fs=[worker(tasks[0])]
    else:
        with ProcessPoolExecutor(max_workers=p['train_workers'] if mode=='DEV' else p['workers'],mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(worker,tasks))
    ct=C.Counter()
    for m in fs:ct.update(m['counts'])
    expected={'DEV_SMOKE':4,'DEV':2216,'SMOKE':12,'MAIN':2688,'ARCHIVE':1152,'NOVEL':864}[mode];assert ct['games']==expected
    (T/(mode+'_LOCK.json')).write_text(json.dumps(dict(files=fs,counts=dict(ct),plan=sha(D/'PLAN.json'),source=sha(D/'games64.py'),eval_source=sha(D/'eval64.py'),binary=lock['binary_sha256'],table=sha(D/'TABLE.json') if not mode.startswith('DEV') else None),indent=2,sort_keys=True)+'\n')
    print(mode,dict(ct),flush=True)
if __name__=='__main__':main(sys.argv[1])
