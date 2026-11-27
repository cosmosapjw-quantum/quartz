"""Actual same-evaluator games, first-event frozen integer and expected-budget tables."""
import collections as C,gzip,json,random,sys,multiprocessing as mp,hashlib
from concurrent.futures import ProcessPoolExecutor
from k55_plan import D,T,SEEDS,SEED,WORKERS,load,sha
from native55 import NNServer,rapfi_module,rsearch,BIN
ARMS=('control','cold','reuse','select','mix')
def emit(f,x):f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n')
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def rows(mode):
    for x in load(mode+'_LOCK.json')['files']:
        p=D/x['file'];assert sha(p)==x['sha256'];yield from raw(p)
def sem(x):
    if isinstance(x,dict):return {k:sem(v) for k,v in x.items() if k not in ('cpu_ns','wall_ns','proof_ns','whole_ns','low_wall_ns','selection')}
    if isinstance(x,list):return [sem(v) for v in x]
    return x
def cell(r):return 2*int(r['bin']==7)+int(r['ply']>8)
def mix_table(job,seed,color):
    f=load('MIX_FROZEN.json')
    coin=int.from_bytes(hashlib.sha256(f'quartz-k55-mix:{SEED}:{job}:{seed}:{color}'.encode()).digest()[:8],'big')
    high=coin*f['prob_den']<(1<<64)*f['prob_num']
    return f['high'] if high else f['low'],coin,high
def local(r,physical=True):
    if 'target' not in r:return 0
    if physical:
        z=r['cache_receipt'];return z['physical']-z['root_physical']
    return r['target']['extra_nn']
def cost(g,tag=None,physical=True):
    return sum(r['actual_nn'] if physical else r['nn'] for r in g['rows'] if tag is None or r['arm']==tag)
def pair(c,t):
    assert c['job']==t['job'] and c['seed']==t['seed'] and c['color']==t['color']
    e=c['event'];assert e==t['event'];changed=False;z=None;op=0
    if e is None:assert c['history']==t['history'] and c['actual_nn']==t['actual_nn'] and c['nn']==t['nn']
    else:
        assert sem(c['rows'][:e])==sem(t['rows'][:e])
        a,b=c['rows'][e],t['rows'][e];assert a['history']==b['history'] and a['low']==b['low'] and a['bin']==b['bin']
        z=cell(a);changed=a['move']!=b['move']
        op=1 if t['policy']=='COLD' else 2 if t['policy']=='REUSE' else b['selection']['op']
        if not changed:assert c['history']==t['history']
    l=sum(local(r) for r in t['rows']);ll=sum(local(r,False) for r in t['rows']);dq=cost(t,'Q')-cost(c,'Q');db=cost(t,'B')-cost(c,'B')
    assert dq+db==t['actual_nn']-c['actual_nn']
    if not changed:assert dq==l and db==0
    return dict(job=c['job']['id'],seed=c['seed'],color=c['color'],policy=t['policy'],cell=z,op=op,reached=e is not None,changed=changed,same_history=c['history']==t['history'],delta2=t['score2']-c['score2'],local_nn=l,logical_extra=ll,downstream_Q=dq-l,downstream_B=db,delta_Q=dq,delta_B=db,delta_total=dq+db,interventions=int(op>0),base_interventions=0,score2=[c['score2'],t['score2']])
def verify_selection(gs,policy):
    g=gs[policy];e=g['event'];op=0 if e is None else g['rows'][e]['selection']['op'];b=gs[('control','cold','reuse')[op]]
    assert g['history']==b['history'] and g['score2']==b['score2'] and g['actual_nn']==b['actual_nn'] and g['nn']==b['nn']
    assert sem(g['rows'])==sem(b['rows'])
    return op
def nn_worker(arg):
    mode,seed,i,js=arg
    from k46_engine import load_eval,from_history,won,FULL
    ev=load_eval(seed);ct=C.Counter();path=T/f'{mode}_{seed}_{i}.jsonl.gz';assert not path.exists()
    table=load('FROZEN.json')['table']
    def end(h):
        a,b=from_history(h);return 1 if won(a) else -1 if won(b) else 0 if a|b==FULL else None
    with NNServer(ev) as srv,gzip.open(path,'wt',compresslevel=3) as f:
        for j in js:
            for color in j['color_order']:
                games={};mt,coin,high=mix_table(j['id'],seed,color)
                for arm in j['arm_order']:
                    h=j['opening'].copy();rr=[];event=None
                    while end(h) is None:
                        side=1-2*(len(h)%2);q=side==color
                        if q and event is None:
                            if arm=='CONTROL':r=srv.nested(h,mask=0)
                            elif arm in ('COLD','REUSE'):r=srv.target(h,reuse=arm=='REUSE')
                            else:r=srv.select(h,table if arm=='SELECT' else mt)
                        else:r=srv.search(h,q)
                        assert r['history']==h and r['move'] not in h
                        if r.get('low') and r['bin'] in (3,7):
                            assert q and event is None;event=len(rr)
                            if arm in ('SELECT','MIX'):
                                z=2*int(r['bin']==7)+int(len(h)>8)
                                assert r['selection']==dict(cell=z,op=(table if arm=='SELECT' else mt)[z],table=table if arm=='SELECT' else mt)
                        r.update(arm='Q' if q else 'B',player=side,ply=len(h));rr.append(r);h.append(r['move'])
                        ct['physical']+=r['actual_nn'];ct['logical']+=r['nn'];ct['moves']+=1;ct['common']+=r['kind']=='CERTIFICATE'
                    w=end(h);g=dict(job=j,seed=seed,color=color,policy=arm,event=event,events=[] if event is None else [event],history=h,winner=w,score2=1+color*w,rows=rr,nn=sum(r['nn'] for r in rr),actual_nn=sum(r['actual_nn'] for r in rr))
                    games[arm.lower()]=g;ct['games']+=1
                    ct[arm.lower()+'_events']+=int(event is not None);ct[arm.lower()+'_extra']+=sum(local(r) for r in rr);ct[arm.lower()+'_extra_logical']+=sum(local(r,False) for r in rr)
                ps={a:pair(games['control'],games[a]) for a in ARMS[1:]}
                ct['episodes']+=1
                for a,z in ps.items():ct[a+'_delta2']+=z['delta2']
                for a in ('select','mix'):ct[a+'_op'+str(verify_selection(games,a))]+=1
                emit(f,dict(**games,pairs=ps,coin=coin,high=high))
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
    pl=load('PLAN.json');nn=mode=='NN';js=pl['jobs'] if nn else pl['rapfi_jobs']
    seeds=SEEDS if nn else (0,);shards=4 if nn else 12;tasks=[(mode,s,i,js[i::shards]) for s in seeds for i in range(shards)]
    random.Random(SEED+len(mode)).shuffle(tasks);assert sha(BIN)==load('BINARY_LOCK.json')['binary_sha256']
    with ProcessPoolExecutor(max_workers=WORKERS if nn else pl['rapfi_workers'],mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(nn_worker if nn else rapfi_worker,tasks))
    ct=C.Counter()
    for x in fs:ct.update(x['counts'])
    assert ct['games']==len(js)*len(seeds)*(10 if mode=='NN' else 2)
    p=T/(mode+'_LOCK.json');assert not p.exists();p.write_text(json.dumps(dict(files=fs,counts=dict(ct),plan=sha(D/'PLAN.json'),source=sha(D/'experiment55.py'),bridge=sha(D/'native55.py'),binary=sha(BIN),frozen=sha(D/'FROZEN.json'),mix=sha(D/'MIX_FROZEN.json')),indent=2,sort_keys=True)+'\n')
    print(mode,dict(ct),flush=True)
if __name__=='__main__':run(sys.argv[1])
