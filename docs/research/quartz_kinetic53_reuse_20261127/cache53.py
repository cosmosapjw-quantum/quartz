"""Exact per-request evaluator memoization, actual old/new/cached traces."""
import collections as C,gzip,json,pathlib
from k53_plan import D,T,SEEDS,load,save,sha
from native53 import NNServer,BIN
from k46_engine import load_eval
class Memo:
    def __init__(self,ev):self.ev=ev;self.rows=0;self.cache={};self.hits=0
    def clear(self):self.cache.clear()
    def __call__(self,a,b,p,legal):
        k=(a,b,p,tuple(legal));self.rows+=1
        if k in self.cache:self.hits+=1
        else:
            q,v=self.ev(a,b,p,legal);self.cache[k]=(q.copy(),v)
        q,v=self.cache[k];return q.copy(),v
def sem(x):
    if isinstance(x,dict):return {k:sem(v) for k,v in x.items() if k not in ('cpu_ns','wall_ns','proof_ns','whole_ns','low_wall_ns')}
    if isinstance(x,list):return [sem(v) for v in x]
    return x
def main():
    old=D.parent/'quartz_kinetic52_confirm_20261127';p=old/'NN_LOCK.json';assert sha(p)==load('PLAN.json')['historical_target_lock_sha256']
    ob=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic52_confirm_backups/mcts_demo';assert sha(ob)==load('BINARY_LOCK.json')['old_binary_sha256']
    ss=C.defaultdict(list)
    for fi in json.loads(p.read_text())['files']:
        pp=old/fi['file'];assert sha(pp)==fi['sha256']
        with gzip.open(pp,'rt') as f:
            for x in map(json.loads,f):
                g=x['target'];s=g['seed']
                if g['event'] is not None and len(ss[s])<8:ss[s].append(g['rows'][g['event']]['history'])
    assert all(len(ss[s])==8 for s in SEEDS)
    ct=C.Counter();path=D/'CACHE_PARITY.jsonl.gz'
    with gzip.open(path,'wt',compresslevel=3) as f:
        for s in SEEDS:
            ev=load_eval(s);memo=Memo(ev)
            with NNServer(ev,ob) as oldsrv,NNServer(ev) as plain,NNServer(memo) as cached:
                for h in ss[s]:
                    n=ev.rows;o=oldsrv.target(h,trace=True);a=plain.target(h,trace=True);assert sem(a)==sem(o)
                    assert ev.rows-n==o['nn']+a['nn'];memo.clear();n=ev.rows;hits=memo.hits;b=cached.target(h,trace=True)
                    assert sem(a)==sem(b)
                    phys=ev.rows-n;keys={(z['a'],z['b'],z['p'],tuple(z['legal'])) for z in b['eval_trace']}
                    assert phys==len(keys) and phys+memo.hits-hits==b['nn']
                    ct['roots']+=1;ct['old_nn']+=o['nn'];ct['plain_nn']+=a['nn'];ct['cache_logical']+=b['nn'];ct['cache_physical']+=phys;ct['hits']+=memo.hits-hits
                    f.write(json.dumps(dict(seed=s,history=h,plain=a,cached=b,old=o,physical=phys,hits=memo.hits-hits),separators=(',',':'))+'\n')
            print('CACHE',s,dict(ct),flush=True)
    assert ct['roots']==56
    save('CACHE_LOCK.json',dict(counts=dict(ct),file=path.name,sha256=sha(path),source=sha(D/'cache53.py'),binary=sha(BIN),old_binary=sha(ob),historical=sha(p)))
if __name__=='__main__':main()
