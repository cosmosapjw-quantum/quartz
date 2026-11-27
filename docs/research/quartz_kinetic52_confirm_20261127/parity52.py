"""Exact historical K51 receipt comparison, frozen old binary and actual NN counters."""
import collections as C,gzip,json,sys,pathlib,math
from k52_plan import D,T,SEEDS,load,save,sha
from native52 import NNServer,BIN,REF
from k46_engine import load_eval
OLD=D.parent/'quartz_kinetic51_targeted_20261127'
OB=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic51_targeted_backups/mcts_demo'
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def sem(x):
    if isinstance(x,dict):return {k:sem(v) for k,v in x.items() if k not in ('cpu_ns','wall_ns','proof_ns','whole_ns','low_wall_ns','native_physical_nn','eval_trace')}
    if isinstance(x,list):return [sem(v) for v in x]
    return x
def eq(a,b):
    if isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b),(set(a)-set(b),set(b)-set(a))
        for k in a:eq(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):eq(x,y)
    elif isinstance(a,float):
        assert isinstance(b,(float,int)) and abs(a-b)<=2e-15,(a,b)
    else:assert a==b,(a,b)
def root_record(g):
    r=g['rows'][g['event']]
    return {k:v for k,v in r.items() if k not in ('arm','player','ply')}
def main():
    p=OLD/'NN_LOCK.json';assert sha(p)==load('PLAN.json')['historical_target_lock_sha256']
    assert sha(OB)=='686784dd17f6ad8fe74fd56b97ec2c499110f59c3dbe09731e7d25d601e1bd4f'
    xs=[]
    for fi in json.loads(p.read_text())['files']:
        q=OLD/fi['file'];assert sha(q)==fi['sha256'];xs+=list(raw(q))
    ct=C.Counter();path=D/'PARITY.jsonl.gz'
    with gzip.open(path,'wt',compresslevel=3) as f:
        for seed in SEEDS:
            ev=load_eval(seed);start=ev.rows
            with NNServer(ev,OB) as old,NNServer(ev) as new:
                for x in xs:
                    g=x['target']
                    if g['seed']!=seed or g['event'] is None:continue
                    h=g['rows'][g['event']]['history'];ref=root_record(g)
                    t=new.target(h);eq(sem(t),sem(ref))
                    a=new.nested(h,mask=0);b=new.nested(h,mask=136)
                    c=old.nested(h,mask=0);d=old.nested(h,mask=136)
                    for j,k,r in ((a,c,x['control']),(b,d,x['global'])):
                        eq(sem(j),sem(k));eq(sem(j),sem(root_record(r)))
                    ct['events']+=1;ct['target_extra_nn']+=t['target']['extra_nn'];ct['child_receipts']+=sum(z is not None for z in t['target']['children'])
                    ct['native_nn']+=t['nn']+a['nn']+b['nn'];ct['old_nn']+=c['nn']+d['nn']
                    f.write(json.dumps(dict(seed=seed,history=h,target=t,control=a,global_=b,old_control=c,old_global=d),separators=(',',':'),allow_nan=False)+'\n')
                # fixed baseline comparison on 4 predetermined source openings per seed
                for j in load('PLAN.json')['jobs'][:4]:
                    h=j['opening'];a=new.search(h,False);b=old.search(h,False);eq(sem(a),sem(b));ct['baseline_checks']+=1;ct['native_nn']+=a['nn'];ct['old_nn']+=b['nn']
            print('PARITY_SEED',seed,ev.rows-start,flush=True)
    assert ct['events']==200 and ct['child_receipts']==400 and ct['target_extra_nn']==2398 and ct['baseline_checks']==28
    save('PARITY_LOCK.json',dict(file=path.name,sha256=sha(path),counts=dict(ct),source=sha(D/'parity52.py'),old_binary=sha(OB),binary=sha(BIN),old_lock=sha(p)))
    print('PARITY',dict(ct),flush=True)
if __name__=='__main__':main()
