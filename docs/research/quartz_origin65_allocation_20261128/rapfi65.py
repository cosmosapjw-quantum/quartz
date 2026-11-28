"""Secondary diagnostic: Q96 and Q104 each independently play B128 with identical Rapfi."""
from pathlib import Path
import sys,json,random,gzip,collections as C,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from plan65 import D,T,P58 as P,ROOT,sha
sys.path.insert(0,str(P))
import native58 as native
R=D.parent/'quartz_kinetic37_20261126'
def load(p):return json.loads(p.read_text())
def make():
    sys.path.insert(0,str(R));import experiment as e
    prior,locks=e.historical();old=e.read('PLAN.json')
    prior.update(j['family'] for j in old['jobs'])
    n=len(prior);rng=random.Random(202611286504);jobs=[]
    for i in range(16):
        while True:
            r,c=rng.randrange(4,11),rng.randrange(4,11)
            op=rng.sample([15*x+y for x in range(r-2,r+3) for y in range(c-2,c+3)],6);f=e.family(op)
            if f not in prior:break
        prior.add(f);jobs.append(dict(id=i,opening=op,family=f,seed=rng.randrange(2**63),color_order=rng.sample([1,-1],2),arm_order=rng.sample(['Q96','Q104'],2)))
    assets=load(R/'RAPFI_LOCK.json')['files']
    for a in assets:assert sha(a['path'])==a['sha256']
    assert sha(e.BRIDGE)=='bacbcfc90f6e4bfd0d9d035455afa3d6667ec32e7fd80cbc1d65a4001b30f64e'
    p=load(D/'PLAN.json');assert sha(p['old_binary'])==p['old_binary_sha256']
    (D/'RAPFI_PLAN.json').write_text(json.dumps(dict(jobs=jobs,prior_families=n,prior_locks=locks,assets=assets,bridge=str(e.BRIDGE),bridge_sha256=sha(e.BRIDGE),binary=p['old_binary'],binary_sha256=p['old_binary_sha256'],scope='exploratory 16 compact families, Q96 vs B128 and Q104 vs B128 separately, two colors; fixed-cap response, not adaptive transfer',workers=2),indent=2,sort_keys=True)+'\n')
    print('Rapfi plan',len(jobs),n,flush=True)
def worker(arg):
    i,js=arg;e=native.rapfi_module();e.BIN=T/'mcts_demo';pl=load(D/'RAPFI_PLAN.json');p=T/f'RAPFI_{i}.jsonl.gz';assert not p.exists()
    ct=C.Counter()
    with e.Server(pl['binary']) as s,gzip.open(p,'wt',compresslevel=3) as f:
        for j in js:
            for color in j['color_order']:
                gs={}
                for arm in j['arm_order']:
                    h=j['opening'].copy();rr=[];events=[]
                    while True:
                        _,side,w=e.replay(h,'freestyle')
                        if w is not None:break
                        q=side==color;r=native.rsearch(e,s,h,e.seed(j,len(h)),q,int(arm[1:]) if q else 96)
                        if 'target' in r:
                            raise AssertionError('unexpected TARGET')
                        r.update(arm='Q' if q else 'B',player=side,ply=len(h));rr.append(r);h.append(r['move'])
                    gs[arm]=dict(job=j,color=color,policy=arm,events=events,history=h,winner=w,score2=1+color*w,rows=rr,nn=sum(r['nn'] for r in rr))
                    ct['games']+=1;ct['moves']+=len(rr);ct['nn']+=gs[arm]['nn'];ct[arm+'_events']+=len(events)
                f.write(json.dumps(gs,separators=(',',':'),allow_nan=False)+'\n');ct['episodes']+=1
    print('RAPFI SHARD',i,dict(ct),flush=True)
    return dict(file=p.name,sha256=sha(p),counts=dict(ct))
def run():
    pl=load(D/'RAPFI_PLAN.json');assert sha(pl['binary'])==pl['binary_sha256']
    tasks=[(i,pl['jobs'][i::4]) for i in range(4)]
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(worker,tasks))
    ct=C.Counter()
    for x in fs:ct.update(x['counts'])
    assert ct['games']==64 and ct['episodes']==32
    (T/'RAPFI_LOCK.json').write_text(json.dumps(dict(files=fs,counts=dict(ct),plan=sha(D/'RAPFI_PLAN.json'),source=sha(Path(__file__))),indent=2,sort_keys=True)+'\n')
    print('RAPFI',dict(ct),flush=True)
if __name__=='__main__':make() if sys.argv[1]=='plan' else run()
