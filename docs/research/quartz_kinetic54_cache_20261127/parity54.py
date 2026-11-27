"""Native exact cache, old binary and uncached new binary, full semantic traces."""
import collections as C,gzip,json,pathlib
from k54_plan import D,T,SEEDS,load,save,sha
from native54 import NNServer,BIN
from k46_engine import load_eval
def sem(x):
    if isinstance(x,dict):return {k:sem(v) for k,v in x.items() if k not in ('cpu_ns','wall_ns','proof_ns','whole_ns','low_wall_ns','cache_receipt','actual_nn','physical_nn')}
    if isinstance(x,list):return [sem(v) for v in x]
    return x
def check(a):
    z=a['cache_receipt'];keys=[tuple(x) for x in z['keys']];seen=set();hi=[]
    for i,k in enumerate(keys):
        if k in seen:hi.append(i)
        seen.add(k)
    assert len(keys)==z['logical'] and len(seen)==z['physical'] and hi==z['hit_indices']
    assert z['hits']==len(hi) and z['root_physical']==z['root_logical']-sum(i<z['root_logical'] for i in hi)
    assert [tuple([r['a'],r['b'],r['p']]) for r in a['eval_trace']]==keys
    return z
def main():
    d=D.parent/'quartz_kinetic53_reuse_20261127';lk=json.loads((d/'CACHE_LOCK.json').read_text());assert sha(d/lk['file'])==lk['sha256']
    ob=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic53_reuse_backups/mcts_demo';assert sha(ob)==load('BINARY_LOCK.json')['old_binary_sha256']
    with gzip.open(d/lk['file'],'rt') as f:old=list(map(json.loads,f))
    ct=C.Counter();p=D/'PARITY.jsonl.gz'
    with gzip.open(p,'wt',compresslevel=3) as f:
        for seed in SEEDS:
            ev=load_eval(seed)
            with NNServer(ev,ob) as osrv,NNServer(ev) as nsrv:
                for x in old:
                    if x['seed']!=seed:continue
                    h=x['history'];n=ev.rows;a=osrv.target(h,trace=True,cache=False);b=nsrv.target(h,trace=True,cache=False);c=nsrv.target(h,trace=True,cache=True)
                    assert sem(a)==sem(b)==sem(c)==sem(x['plain'])
                    z=check(c);assert ev.rows-n==a['nn']+b['nn']+z['physical']
                    ct.update(target_roots=1,old_nn=a['nn'],new_uncached=b['nn'],logical=z['logical'],physical=z['physical'],hits=z['hits'])
                    f.write(json.dumps(dict(seed=seed,policy='TARGET',old=a,plain=b,cached=c),separators=(',',':'))+'\n')
                for j in load('PLAN.json')['jobs'][:4]:
                    h=j['opening'];n=ev.rows;a=osrv.search(h,False,trace=True,cache=False);b=nsrv.search(h,False,trace=True,cache=False);c=nsrv.search(h,False,trace=True,cache=True)
                    assert sem(a)==sem(b)==sem(c)
                    z=check(c);assert ev.rows-n==a['nn']+b['nn']+z['physical']
                    ct.update(baseline_roots=1,old_nn=a['nn'],new_uncached=b['nn'],logical=z['logical'],physical=z['physical'],hits=z['hits'])
                    f.write(json.dumps(dict(seed=seed,policy='B32',old=a,plain=b,cached=c),separators=(',',':'))+'\n')
            print('PARITY',seed,dict(ct),flush=True)
    assert ct['target_roots']==56 and ct['baseline_roots']==28
    save('PARITY_LOCK.json',dict(counts=dict(ct),file=p.name,sha256=sha(p),source=sha(D/'parity54.py'),binary=sha(BIN),old_binary=sha(ob),historical_lock=sha(d/'CACHE_LOCK.json')))
if __name__=='__main__':main()
