"""Actual old/new fixed K57 parity and earliest stable stop on same root prefix."""
import json,pathlib,collections as C
from k58_plan import D,T,SEEDS,load,sha
from native58 import NNServer,BIN
from k46_engine import load_eval
def stable(a,b):
    if a['locked20'] or b['locked20'] or b['visits']!=a['visits']+4 or b['visits'] not in (8,12,16):return False
    if a['move']!=b['move'] or a['move']!=a['original_move'] or b['move']!=b['original_move'] or a['rival']!=b['rival'] or b['rival'] is None or b['distance']<a['distance']:return False
    for x in (a,b):
        e={m:(n,w) for m,n,w,p in x['edges']};n,w=e[x['move']];k,v=e[x['rival']]
        if not(n>0 and k>0 and w*k>=v*n):return False
    return True
def sem(x):
    if isinstance(x,dict):return {k:sem(v) for k,v in x.items() if k not in ('cpu_ns','wall_ns','proof_ns','whole_ns','low_wall_ns')}
    if isinstance(x,list):return [sem(v) for v in x]
    return x
def main():
    old=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic57_early_backups/mcts_demo';assert sha(old)==load('BINARY_LOCK.json')['old_binary_sha256']
    p=load('PLAN.json');ct=C.Counter();out=[]
    for seed in SEEDS:
        ev=load_eval(seed)
        with NNServer(ev,old) as a,NNServer(ev) as b:
            for j in p['jobs'][:4]:
                h=j['opening'];H=j['horizon'];r=[]
                for op in (0,1):
                    x=a.early(h,H,op,trace=True);y=b.early(h,H,op,trace=True);assert sem(x)==sem(y)
                    ct['old_nn']+=x['actual_nn'];ct['new_nn']+=y['actual_nn'];ct['fixed_requests']+=1;r.append(y)
                x=r[0];y=b.stable(h,trace=True);ct['stable_nn']+=y['actual_nn'];ct['roots']+=1
                if 'early' in x:
                    ss=x['early']['snapshots'];q=next((v for u,v in zip(ss,ss[1:]) if stable(u,v)),None)
                    assert y['early']['at']==q and y['early']['reached']==(q is not None)
                    assert y['early']['snapshots']==ss[:len(y['early']['snapshots'])]
                    assert y['eval_trace']==x['eval_trace'][:len(y['eval_trace'])]
                    ct['stable_reached']+=q is not None
                    if q is not None:assert y['move']==q['move']
                else:assert sem(x)==sem(y)
                out.append(dict(seed=seed,history=h,horizon=H,control=x,stop=r[1],stable=y))
    assert ct['roots']==28 and ct['fixed_requests']==56
    (T/'PARITY.json').write_text(json.dumps(dict(counts=dict(ct),rows=out,old_binary=sha(old),new_binary=sha(BIN)),separators=(',',':'),allow_nan=False)+'\n')
    print('PARITY',dict(ct),flush=True)
if __name__=='__main__':main()
