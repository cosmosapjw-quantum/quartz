"""Old/new actual cached roots, and all three native selector operations."""
import collections as C,gzip,json,pathlib
from k55_plan import D,T,SEEDS,load,sha
from native55 import NNServer,BIN
from k46_engine import load_eval
from experiment55 import sem
def main():
    d=D.parent/'quartz_kinetic54_cache_20261127';lk=json.loads((d/'PARITY_LOCK.json').read_text());assert sha(d/lk['file'])==lk['sha256']
    old=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic54_cache_backups/mcts_demo';assert sha(old)==load('BINARY_LOCK.json')['old_binary_sha256']
    roots={}
    with gzip.open(d/lk['file'],'rt') as f:
        for x in map(json.loads,f):
            if x['policy']=='TARGET':roots.setdefault(x['seed'],[]).append(x['cached']['history'])
    out=[];ct=C.Counter()
    for seed in SEEDS:
        ev=load_eval(seed)
        with NNServer(ev,old) as a,NNServer(ev) as b:
            for h in roots[seed][:2]:
                for op in range(3):
                    def fixed(s):return s.nested(h,mask=0,trace=True) if op==0 else s.target(h,reuse=op==2,trace=True)
                    x=fixed(a);y=fixed(b);z=b.select(h,[op]*4,trace=True)
                    assert sem(x)==sem(y)==sem(z)
                    assert z['selection']['op']==op and x['actual_nn']==y['actual_nn']==z['actual_nn']
                    ct['requests']+=1;ct['old_nn']+=x['actual_nn'];ct['new_nn']+=y['actual_nn'];ct['selector_nn']+=z['actual_nn'];ct['op'+str(op)]+=1
                    out.append(dict(seed=seed,history=h,op=op,old=x,new=y,selector=z))
    assert ct['requests']==42 and all(ct['op'+str(a)]==14 for a in range(3))
    (T/'PARITY.json').write_text(json.dumps(dict(counts=dict(ct),rows=out,old_binary=sha(old),new_binary=sha(BIN),source=sha(d/'PARITY_LOCK.json')),indent=2,allow_nan=False)+'\n')
    print('PARITY',dict(ct),flush=True)
if __name__=='__main__':main()
