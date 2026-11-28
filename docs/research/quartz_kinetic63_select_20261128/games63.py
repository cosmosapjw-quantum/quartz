"""Actual independent Quartz-v-identical-B32 games. Forced rival is only a root intervention."""
from plan63 import D,T,SEEDS,EXTRA,ARMS,DEV,sha,read
from eval63 import n,load_eval,from_history,won,FULL,target,select
import json,gzip,sys,collections as C,random,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
def play(s,ev,start,color,policy,root=None):
    h=list(start);rr=[];e0=ev.rows;ct=C.Counter()
    table=read(D/'TABLE.json')['table'] if policy=='SELECT' else None
    while True:
        a,b=from_history(h);w=1 if won(a) else -1 if won(b) else 0 if a|b==FULL else None
        if w is not None:break
        side=1-2*(len(h)%2);q=side==color
        op=policy if root is None or not rr else 'H4'
        if not q:r=s.search(h,False)
        elif op=='TARGET':r=target(s,h)
        elif op in ('FREE','SELECT'):r=select(s,h,table if op=='SELECT' else None)
        else:r=s.early(h,8 if op=='H8' else 4,1)
        assert r['history']==h and r['move'] not in h
        m=r['move']
        if root is not None and not rr and policy=='FORCE_B':
            assert q and r['early']['reached'];m=r['early']['at']['rival'];assert m==root['snapshot']['rival'] and m not in h
        if 'target' in r:
            assert q and op in ('TARGET','SELECT') and r['target']['extra_nn']<=4
            ct['targets']+=1;ct['switches']+=r['target']['chosen']!=r['target']['incumbent']
        r.update(arm='Q' if q else 'B',player=side,ply=len(h),played=m);rr.append(r);h.append(m)
    g=dict(policy=policy,history=h,rows=rr,winner=w,score2=1+color*w,actual_nn=sum(r['actual_nn'] for r in rr),nn=sum(r['nn'] for r in rr),targets=ct['targets'],switches=ct['switches'])
    assert ev.rows-e0==g['actual_nn'];return g
def sem(x):
    if isinstance(x,dict):return {k:sem(v) for k,v in x.items() if not k.endswith('_ns')}
    if isinstance(x,list):return [sem(v) for v in x]
    return x
def pairs(j,gs):
    r=gs['H4']['rows'][0];x=r['early']['at'];a,b=x['move'],x['rival']
    assert x==j['snapshot'] and r['partial']['bounds']==j['bounds'] and r['actual_nn']==j['root_nn']
    f=gs['FORCE_B']['rows'][0];assert sem({k:v for k,v in f.items() if k!='played'})==sem({k:v for k,v in r.items() if k!='played'}) and f['played']==b
    t=gs['TARGET']['rows'][0];assert t['early']['at']==x and t['target']['incumbent']==a and t['target']['rival']==b
    h=gs['FREE']['rows'][0];assert h['early']['at']==x and h['actual_nn']==r['actual_nn'] and h['select63']['op']==1
    y={k:g['score2'] for k,g in gs.items()};m={k:g['rows'][0]['played'] for k,g in gs.items()}
    for p in gs:
        for q in gs:
            if m[p]==m[q]:assert y[p]==y[q] and gs[p]['history']==gs[q]['history'] and sem(gs[p]['rows'][1:])==sem(gs[q]['rows'][1:])
    assert all(m[k] in (a,b) and y[k]==y['FORCE_B' if m[k]==b else 'H4'] for k in ('TARGET','FREE'))
    return dict(a=a,b=b,moves=m,score2=y,oracle2=max(y['H4'],y['FORCE_B']),regret2=max(y['H4'],y['FORCE_B'])-y['H4'],target_residual2=max(y['H4'],y['FORCE_B'])-y['TARGET'],free_residual2=max(y['H4'],y['FORCE_B'])-y['FREE'],target_extra=t['actual_nn']-r['actual_nn'],free_extra=h['actual_nn']-r['actual_nn'])
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
        for s in SEEDS:
            js=[j for j in p['roots'] if j['seed']==s]
            tasks.extend((mode,s,i,js[i::3]) for i in range(3) if js[i::3])
    else:
        assert mode in ('MAIN','ARCHIVE','SMOKE')
        panel='MAIN' if mode=='SMOKE' else mode;js=[j for j in p['jobs'] if j['panel']==panel];seeds=SEEDS if panel=='MAIN' else EXTRA
        tasks=[(mode,SEEDS[0],0,js[:1])] if mode=='SMOKE' else [(mode,s,i,js[i::4]) for s in seeds for i in range(4)]
    random.Random(202611286301).shuffle(tasks)
    if len(tasks)==1:fs=[worker(tasks[0])]
    else:
        with ProcessPoolExecutor(max_workers=p['train_workers'] if mode=='DEV' else p['workers'],mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(worker,tasks))
    ct=C.Counter()
    for m in fs:ct.update(m['counts'])
    expected={'DEV_SMOKE':4,'DEV':1720,'SMOKE':8,'MAIN':2688,'ARCHIVE':384}[mode];assert ct['games']==expected
    (T/(mode+'_LOCK.json')).write_text(json.dumps(dict(files=fs,counts=dict(ct),plan=sha(D/'PLAN.json'),source=sha(D/'games63.py'),eval_source=sha(D/'eval63.py'),binary=lock['binary_sha256'],table=sha(D/'TABLE.json') if not mode.startswith('DEV') else None),indent=2,sort_keys=True)+'\n')
    print(mode,dict(ct),flush=True)
if __name__=='__main__':main(sys.argv[1])
