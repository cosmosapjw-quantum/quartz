"""New exploratory 15x15 resident NN trajectories, integration-only, no WDL transfer."""
import hashlib, json, random, sys
from pathlib import Path
D=Path(__file__).resolve().parent; ROOT=D.parents[2]
PREV=ROOT.parent/'quartz_nn_decision_20261124'/'docs/research/quartz_nn_decision_20261124'
sys.path.insert(0,str(PREV))
from engine import Engine, MODELS, BIN, sha
assert sha(MODELS['gomoku15'])=='57b93ba867ff8f9b74f91adef6b38f02f94de0bd7dc177785d668d32e1dcc3ae'
print('GOMOKU15 checkpoint',sha(MODELS['gomoku15']),'binary',sha(BIN),flush=True)
def winning(b,a,p):
    x,y=a%15,a//15
    for dx,dy in ((1,0),(0,1),(1,1),(1,-1)):
        k=1
        for s in (-1,1):
            xx,yy=x+s*dx,y+s*dy
            while 0<=xx<15 and 0<=yy<15 and b[yy*15+xx]==p:
                k+=1;xx+=s*dx;yy+=s*dy
        if k>=5:return True
    return False
def main():
    rng=random.Random(2026112405);rr=[]
    eng=Engine('gomoku15')
    try:
        for target in (7,8,9,10):
            b=[0]*225;h=[];ev=[]
            for i in range(target):
                p=1-2*(i%2);legal=[a for a,v in enumerate(b) if not v]
                if i<5: a=rng.choice(legal)
                else:
                    sid,q=eng.open(dict(board=b,player=p))
                    try:
                        explore=rng.random()<.45;a=rng.choice(legal) if explore else q['best_move']
                        ev.append(dict(ply=i,action=a,explore=explore,response=q))
                    finally:eng.close(sid)
                assert a in legal;b[a]=p;h.append(a);assert not winning(b,a,p)
            s=dict(board=b,player=1-2*(target%2))
            sid,q=eng.open(s)
            try:f=eng.step(s,sid,q)
            finally:eng.close(sid)
            assert q['actual_nn_rows']>0 and f['actual_nn_rows']>0
            assert q['evaluator_calls']==q['actual_nn_rows']
            assert f['additional_evaluator_calls']==f['actual_nn_rows']
            rr.append(dict(ply=target,history=h,board=b,player=s['player'],generation=ev,prefix=q,final=f))
            print('GOMOKU15',target,q['actual_nn_rows'],f['actual_nn_rows'],flush=True)
    finally:eng.stop()
    p=D/'C09_GOMOKU15_REUSE.json';assert not p.exists()
    p.write_text(json.dumps(dict(scope='integration smoke only; no exact WDL or controller transfer',binary_sha256=sha(BIN),checkpoint_sha256=sha(MODELS['gomoku15']),engine_sha256=sha(PREV/'engine.py'),rows=rr),indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
