"""Cached same-evaluator paired games; actual physical forward counts, logical receipts retained."""
import collections as C,gzip,json,random,sys,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from k54_plan import D,T,SEEDS,SEED,WORKERS,load,sha
from native54 import NNServer,rapfi_module,rsearch,BIN
def emit(f,x):f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n')
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def rows(mode):
    for x in load(mode+'_LOCK.json')['files']:
        p=D/x['file'];assert sha(p)==x['sha256'];yield from raw(p)
def sem(x):
    if isinstance(x,dict):return {k:sem(v) for k,v in x.items() if k not in ('cpu_ns','wall_ns','proof_ns','whole_ns','low_wall_ns','cache_receipt','actual_nn','physical_nn')}
    if isinstance(x,list):return [sem(v) for v in x]
    return x
def local(r,physical=True):
    if 'target' not in r:return 0
    if physical and 'cache_receipt' in r:
        z=r['cache_receipt'];return z['physical']-z['root_physical']
    return r['target']['extra_nn']
def cost(g,tag=None,physical=True):
    return sum(r['actual_nn'] if physical else r['nn'] for r in g['rows'] if tag is None or r['arm']==tag)
def repeat_allowed(policy,rs,events):
    if not events:return True
    if policy!='CONDITIONAL' or len(events)>=2:return False
    t=rs[events[0]]['target'];return t['chosen']==t['incumbent']
def pair(c,t):
    assert c['job']==t['job'] and c['seed']==t['seed'] and c['color']==t['color']
    e=c['event'];assert e==t['event'];change=False
    if e is None:assert c['history']==t['history'] and c['nn']==t['nn'] and c['actual_nn']==t['actual_nn']
    else:
        assert sem(c['rows'][:e])==sem(t['rows'][:e])
        a,b=c['rows'][e],t['rows'][e];assert a['history']==b['history'] and a['low']==b['low'] and a['bin']==b['bin']
        if t['policy']=='CONDITIONAL':
            assert c['policy']=='COLD' and sem(a)==sem(b)
            if len(t['events'])<=1:assert c['history']==t['history'] and c['actual_nn']==t['actual_nn']
            else:
                i=t['events'][1];p=t['rows'][i]['ply'];assert c['history'][:p]==t['history'][:p]
                assert b['target']['chosen']==b['target']['incumbent']
                change=t['rows'][i]['move']!=t['rows'][i]['target']['incumbent']
                if not change:assert c['history']==t['history']
        else:
            change=a['move']!=b['move']
            if not change:assert c['history']==t['history']
    l=sum(local(r) for r in t['rows'])-sum(local(r) for r in c['rows'])
    ll=sum(local(r,False) for r in t['rows'])-sum(local(r,False) for r in c['rows'])
    dq=cost(t,'Q')-cost(c,'Q');db=cost(t,'B')-cost(c,'B')
    assert dq+db==t['actual_nn']-c['actual_nn']
    return dict(job=c['job']['id'],seed=c['seed'],color=c['color'],policy=t['policy'],reached=e is not None,changed=change,same_history=c['history']==t['history'],delta2=t['score2']-c['score2'],local_nn=l,logical_extra=ll,downstream_Q=dq-l,downstream_B=db,delta_Q=dq,delta_B=db,delta_total=dq+db,interventions=len(t['events']),base_interventions=len(c['events']),score2=[c['score2'],t['score2']])
def nn_worker(arg):
    mode,seed,i,js=arg
    from k46_engine import load_eval,from_history,won,FULL
    ev=load_eval(seed);ct=C.Counter();path=T/f'{mode}_{seed}_{i}.jsonl.gz';assert not path.exists()
    def end(h):
        a,b=from_history(h);return 1 if won(a) else -1 if won(b) else 0 if a|b==FULL else None
    with NNServer(ev) as srv,gzip.open(path,'wt',compresslevel=3) as f:
        for j in js:
            for color in j['color_order']:
                games={}
                for arm in j['arm_order'] if mode=='NN' else ['COLD']:
                    h=j['opening'].copy();rr=[];events=[];cache=mode=='NN'
                    while end(h) is None:
                        side=1-2*(len(h)%2);q=side==color
                        if q and repeat_allowed(arm,rr,events):
                            r=srv.nested(h,mask=0,cache=cache) if arm=='CONTROL' else srv.target(h,cache=cache)
                        else:r=srv.search(h,q,cache=cache)
                        assert r['history']==h and r['move'] not in h
                        if r.get('low') and r['bin'] in (3,7):
                            assert q and repeat_allowed(arm,rr,events);events.append(len(rr))
                            if arm!='CONTROL':assert 'target' in r
                        r.update(arm='Q' if q else 'B',player=side,ply=len(h));rr.append(r);h.append(r['move'])
                        ct['physical']+=r['actual_nn'];ct['logical']+=r['nn'];ct['moves']+=1;ct['common']+=r['kind']=='CERTIFICATE'
                    w=end(h);g=dict(job=j,seed=seed,color=color,policy=arm,event=events[0] if events else None,events=events,history=h,winner=w,score2=1+color*w,rows=rr,nn=sum(r['nn'] for r in rr),actual_nn=sum(r['actual_nn'] for r in rr))
                    games[arm.lower()]=g;ct['games']+=1
                    ct[arm.lower()+'_events']+=len(events);ct[arm.lower()+'_extra']+=sum(local(r) for r in rr);ct[arm.lower()+'_extra_logical']+=sum(local(r,False) for r in rr)
                if mode=='NN':
                    ps={a:pair(games[b],games[a]) for a,b in [('cold','control'),('conditional','cold')]}
                    ct['episodes']+=1
                    for a,z in ps.items():ct[a+'_delta2']+=z['delta2']
                    emit(f,dict(**games,pairs=ps))
                else:emit(f,games['cold'])
    assert ev.rows==ct['physical']
    print('SHARD',mode,seed,i,dict(ct),flush=True)
    return dict(file=path.name,sha256=sha(path),counts=dict(ct))
def rapfi_worker(arg):
    mode,_,i,js=arg;e=rapfi_module()
    ct=C.Counter();path=T/f'{mode}_0_{i}.jsonl.gz';assert not path.exists()
    with e.Server(BIN) as server,gzip.open(path,'wt',compresslevel=3) as f:
        for j in js:
            for color in j['color_order']:
                h=j['opening'].copy();rr=[];event=None
                while True:
                    _,side,w=e.replay(h,'freestyle')
                    if w is not None:break
                    q=side==color;r=rsearch(e,server,h,e.seed(j,len(h)),q,'T' if q and mode=='R_T' and event is None else 96)
                    if 'target' in r:
                        assert event is None and q;event=len(rr);ct['events']+=1;ct['extra']+=r['target']['extra_nn'];ct['changed']+=r['move']!=r['target']['incumbent']
                    r.update(arm='Q' if q else 'B',player=side,ply=len(h));rr.append(r);h.append(r['move'])
                    ct['nn']+=r['nn'];ct['moves']+=1;ct['common']+=r['kind']=='CERTIFICATE'
                emit(f,dict(job=j,seed=0,color=color,policy=mode,event=event,history=h,winner=w,score2=1+color*w,rows=rr,atlas=[],nn=sum(r['nn'] for r in rr),qcap=96));ct['games']+=1
    print('SHARD',mode,0,i,dict(ct),flush=True)
    return dict(file=path.name,sha256=sha(path),counts=dict(ct))
def run(mode):
    pl=load('PLAN.json');nn=mode in ('NN','UNCACHED');js=pl['jobs'] if mode=='NN' else pl['jobs'][:8] if mode=='UNCACHED' else pl['rapfi_jobs']
    seeds=SEEDS if nn else (0,);shards=4 if nn else 12;tasks=[(mode,s,i,js[i::shards]) for s in seeds for i in range(shards)]
    random.Random(SEED+len(mode)).shuffle(tasks);assert sha(BIN)==load('BINARY_LOCK.json')['binary_sha256']
    with ProcessPoolExecutor(max_workers=WORKERS if nn else pl['rapfi_workers'],mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(nn_worker if nn else rapfi_worker,tasks))
    ct=C.Counter()
    for x in fs:ct.update(x['counts'])
    assert ct['games']==len(js)*len(seeds)*(6 if mode=='NN' else 2)
    p=T/(mode+'_LOCK.json');assert not p.exists();p.write_text(json.dumps(dict(files=fs,counts=dict(ct),plan=sha(D/'PLAN.json'),source=sha(D/'experiment54.py'),bridge=sha(D/'native54.py'),binary=sha(BIN)),indent=2,sort_keys=True)+'\n')
    print(mode,dict(ct),flush=True)
if __name__=='__main__':run(sys.argv[1])
