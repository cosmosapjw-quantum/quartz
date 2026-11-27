"""Actual separate same-evaluator Q-v-B games; one preobservable targeted intervention."""
import collections as C,gzip,json,random,sys,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from k52_plan import D,T,SEEDS,SEED,WORKERS,load,save,sha
from native52 import NNServer,rapfi_module,rsearch,BIN
def emit(f,x):f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n')
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def rows(mode):
    for x in load(mode+'_LOCK.json')['files']:
        p=D/x['file'];assert sha(p)==x['sha256'];yield from raw(p)
def semantic(x):
    if isinstance(x,dict):return {k:semantic(v) for k,v in x.items() if k not in ('cpu_ns','wall_ns','proof_ns','whole_ns','low_wall_ns','mask')}
    if isinstance(x,list):return [semantic(v) for v in x]
    return x
def cost(g,tag=None,after=-1):return sum(r['nn'] for i,r in enumerate(g['rows']) if i>after and (tag is None or r['arm']==tag))
def pair(c,t):
    assert c['job']==t['job'] and c['seed']==t['seed'] and c['color']==t['color'] and c['policy']=='CONTROL'
    assert t['policy'] in ('GLOBAL','TARGET')
    e=c['event'];assert e==t['event'];local=0;changed=False;bn=None;act=None;ply=None;postQ=postB=0
    if e is None:
        assert c['history']==t['history'] and c['score2']==t['score2'] and c['nn']==t['nn']
        assert semantic(c['rows'])==semantic(t['rows'])
    else:
        assert semantic(c['rows'][:e])==semantic(t['rows'][:e])
        r,s=c['rows'][e],t['rows'][e]
        assert r['arm']==s['arm']=='Q' and r['history']==s['history'] and r['bin']==s['bin'] and r['low']==s['low']
        assert r['bin'] in (3,7) and semantic(r['partial'])==semantic(s['partial'])
        assert not r['extended'] and r['nn']==r['low']['nn'] and r['move']==r['low']['move']
        if t['policy']=='GLOBAL':
            assert s['extended'] and s['mask']==136 and s['nn']==s['physical_nn']
            assert all(m==mm and n<=nn for (m,n),(mm,nn) in zip(r['votes'],s['votes'],strict=True))
        else:
            assert not s['extended'] and s['mask']==0 and s['physical_nn']==r['nn']
            assert s['target']['incumbent']==r['move'] and s['nn']==r['nn']+s['target']['extra_nn'] and s['move']==s['target']['chosen']
        local=s['nn']-r['nn'];assert local>=0;changed=s['move']!=r['move'];bn=r['bin'];act=[r['move'],s['move']];ply=r['ply']
        postQ=cost(t,'Q',e)-cost(c,'Q',e);postB=cost(t,'B',e)-cost(c,'B',e)
        if not changed:
            assert c['history']==t['history'] and c['score2']==t['score2'] and postQ==postB==0
            assert semantic(c['rows'][e+1:])==semantic(t['rows'][e+1:])
    dq=cost(t,'Q')-cost(c,'Q');db=cost(t,'B')-cost(c,'B')
    assert dq==local+postQ and db==postB and dq+db==t['nn']-c['nn']
    return dict(job=c['job']['id'],seed=c['seed'],color=c['color'],policy=t['policy'],reached=e is not None,event=e,ply=ply,bin=bn,actions=act,changed=changed,score2=[c['score2'],t['score2']],delta2=t['score2']-c['score2'],local_nn=local,downstream_Q=postQ,downstream_B=postB,delta_Q=dq,delta_B=db,delta_total=dq+db,same_history=c['history']==t['history'])
def nn_worker(arg):
    mode,seed,i,js=arg
    from k46_engine import load_eval,from_history,won,FULL
    ev=load_eval(seed);ct=C.Counter();path=D/f'{mode}_{seed}_{i}.jsonl.gz';assert not path.exists()
    def end(h):
        a,b=from_history(h);return 1 if won(a) else -1 if won(b) else 0 if a|b==FULL else None
    with NNServer(ev) as srv,gzip.open(path,'wt',compresslevel=3) as f:
        for j in js:
            for color in j['color_order']:
                games={}
                for arm in j['arm_order']:
                    h=j['opening'].copy();rr=[];event=None
                    while end(h) is None:
                        side=1-2*(len(h)%2);q=side==color
                        if q and event is None:r=srv.target(h) if arm=='TARGET' else srv.nested(h,mask=136 if arm=='GLOBAL' else 0)
                        else:r=srv.search(h,q)
                        assert r['history']==h and r['move'] not in h
                        if r.get('low') and r['bin'] in (3,7):
                            assert q and event is None;event=len(rr)
                            if arm=='TARGET':
                                z=r['target'];assert r['nn']==r['native_physical_nn']==r['physical_nn']+z['extra_nn']
                            elif arm=='GLOBAL':assert r['extended']
                        r.update(arm='Q' if q else 'B',player=side,ply=len(h));rr.append(r);h.append(r['move'])
                        ct['nn']+=r['nn'];ct['moves']+=1;ct['common']+=r['kind']=='CERTIFICATE'
                    w=end(h);g=dict(job=j,seed=seed,color=color,policy=arm,event=event,history=h,winner=w,score2=1+color*w,rows=rr,atlas=[],nn=sum(r['nn'] for r in rr))
                    games[arm.lower()]=g;ct['games']+=1
                ps={a:pair(games['control'],games[a]) for a in ('global','target')}
                ct['episodes']+=1
                for a,z in ps.items():
                    for k in ('reached','changed','local_nn','delta2'):ct[a+'_'+k]+=z[k]
                emit(f,dict(**games,pairs=ps))
    assert ev.rows==ct['nn'],(ev.rows,ct)
    print('SHARD',mode,seed,i,dict(ct),flush=True)
    return dict(file=path.name,sha256=sha(path),counts=dict(ct))
def rapfi_worker(arg):
    mode,s,i,js=arg;e=rapfi_module();cap='AD' if mode.endswith('_AD') else int(mode.split('_')[-1])
    ct=C.Counter();path=D/f'{mode}_{s}_{i}.jsonl.gz';assert not path.exists()
    with e.Server(BIN) as server,gzip.open(path,'wt',compresslevel=3) as f:
        for j in js:
            for color in j['color_order']:
                h=j['opening'].copy();rr=[]
                while True:
                    _,side,w=e.replay(h,'freestyle')
                    if w is not None:break
                    q=side==color;r=rsearch(e,server,h,e.seed(j,len(h)),q,cap)
                    r.update(arm='Q' if q else 'B',player=side,ply=len(h));rr.append(r);h.append(r['move'])
                    ct['nn']+=r['nn'];ct['moves']+=1;ct['common']+=r['kind']=='CERTIFICATE'
                emit(f,dict(job=j,seed=0,color=color,history=h,winner=w,score2=1+color*w,rows=rr,atlas=[],nn=sum(r['nn'] for r in rr),qcap=cap));ct['games']+=1
    print('SHARD',mode,s,i,dict(ct),flush=True)
    return dict(file=path.name,sha256=sha(path),counts=dict(ct))
def run(mode):
    pl=load('PLAN.json');nn=mode=='NN'
    js=pl['jobs'] if nn else [j for j in pl['rapfi_jobs'] if j['split']==mode.split('_')[1]]
    seeds=SEEDS if nn else (0,);tasks=[(mode,s,i,js[i::4]) for s in seeds for i in range(4)]
    random.Random(SEED+len(mode)).shuffle(tasks)
    assert sha(BIN)==load('BINARY_LOCK.json')['binary_sha256']
    with ProcessPoolExecutor(max_workers=WORKERS if nn else 4,mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(nn_worker if nn else rapfi_worker,tasks))
    ct=C.Counter()
    for x in fs:ct.update(x['counts'])
    assert ct['games']==len(js)*len(seeds)*(6 if nn else 2)
    save(mode+'_LOCK.json',dict(files=fs,counts=dict(ct),plan=sha(D/'PLAN.json'),source=sha(D/'experiment52.py'),bridge=sha(D/'native52.py'),target_source=sha(D/'../../..'/'src/mcts_server/k47_nn.rs'),binary=sha(BIN)))
    print(mode,dict(ct),flush=True)
if __name__=='__main__':run(sys.argv[1])
