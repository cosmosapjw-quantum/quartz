"""Reference Q20 old/new semantic parity and early target/cached trace prefix."""
import json,gzip,pathlib,collections as C
from k57_plan import D,T,SEEDS,load,sha
from native57 import NNServer,BIN
from k46_engine import load_eval
def main():
    d=D.parent/'quartz_kinetic56_voc_20261128'
    old=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic56_voc_backups/mcts_demo'
    assert sha(old)==load('BINARY_LOCK.json')['old_binary_sha256']
    p=load('PLAN.json');ct=C.Counter();out=[]
    # Actual old and new Q20 at fixed fresh plan roots, no full outcome observed.
    keys=('move','kind','nn','visits','terminal_leaves','votes','distance','original_move','replacement','cap','prior_best','cache_receipt','eval_trace')
    for seed in SEEDS:
        ev=load_eval(seed)
        with NNServer(ev,old) as a,NNServer(ev) as b:
            for j in p['jobs'][:2]:
                h=j['opening']
                for H in (4,8,12,16):
                    x=a.search(h,True,trace=True);y=b.early(h,H,0,trace=True)
                    assert all(x[k]==y[k] for k in keys),(seed,h,H)
                    q=[b.early(h,H,o,trace=True) for o in (1,2,3)]
                    if 'early' not in y:assert all(all(x[k]==z[k] for k in keys) for z in q)
                    else:
                        for z in q:
                            assert z['early']['at']==y['early']['at']
                            assert z['early']['snapshots']==y['early']['snapshots'][:len(z['early']['snapshots'])]
                            m=z['cache_receipt']['root_logical']
                            assert z['eval_trace'][:m]==y['eval_trace'][:m]
                            if not y['early']['reached']:assert all(x[k]==z[k] for k in keys)
                        if y['early']['reached']:
                            ct['reached']+=1
                            assert q[0]['move']==y['early']['at']['move']
                    ct['roots']+=1;ct['old_nn']+=x['actual_nn'];ct['control_nn']+=y['actual_nn'];ct['early_nn']+=sum(z['actual_nn'] for z in q)
                    out.append(dict(seed=seed,history=h,horizon=H,old=x,control=y,stop=q[0],cold=q[1],reuse=q[2]))
    assert ct['roots']==56
    (T/'PARITY.json').write_text(json.dumps(dict(counts=dict(ct),rows=out,old_binary=sha(old),new_binary=sha(BIN)),separators=(',',':'),allow_nan=False)+'\n')
    print('PARITY',dict(ct),flush=True)
if __name__=='__main__':main()
