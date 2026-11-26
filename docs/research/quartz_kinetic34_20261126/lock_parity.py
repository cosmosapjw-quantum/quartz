"""Actual same-trajectory and previous native C4 regression, no new policy fitting."""
import pathlib,json,gzip,sys,statistics as S,math
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2];R=D.parent
def read(n):return json.loads((D/n).read_text())
def rows(p):
    with gzip.open(p,'rt') as f:
        for x in f:yield json.loads(x)
def f80(arm):
    return {(g['job']['id'],g['color']):g for p in sorted(D.glob('MAIN_'+arm+'_*.jsonl.gz')) for g in rows(p)}
a=f80('F80');b=f80('F80L');assert set(a)==set(b) and len(a)==72
same=0;nn0=nn1=stops=vis0=vis1=0;different=[];all_prefix=0;candidate_moves=0
for k,g in a.items():
    h=b[k]
    if g['history']!=h['history']:different.append(k);continue
    same+=1;assert g['score']==h['score'] and len(g['rows'])==len(h['rows'])
    for x,y in zip(g['rows'],h['rows']):
        assert x['history']==y['history'] and x['best']==y['best'] and x['proof']==y['proof']
        if x['player']==g['color']:
            candidate_moves+=1;nn0+=x['NN_calls'];nn1+=y['NN_calls'];vis0+=x['visits'];vis1+=y['visits'];stops+=y['cause']=='VOTE_LOCK'
            assert y['NN_calls']<=x['NN_calls'] and y['visits']<=x['visits']
            if x['prefix'] is not None:
                assert x['prefix']['result']==y['prefix']['result'] and x['root_eval']==y['root_eval']
                # Every actually executed wave has identical logical sequence and result, excluding time.
                for w,z in zip(x['waves'],y['waves']):
                    assert {k:v for k,v in w.items() if k!='step_us'}=={k:v for k,v in z.items() if k!='step_us'}
                all_prefix+=1
        else:assert x['NN_calls']==y['NN_calls'] and x['visits']==y['visits']
assert same==72 and not different
sys.path.insert(0,str(R/'quartz_kinetic32_20261126'))
from core import Server
co=json.loads((R/'quartz_kinetic32_20261126/COHORT.json').read_text())['roots'][:8]
n=0
with Server(ROOT.parent/'quartz_kinetic32_target_20261126/mcts_demo',8) as old,Server(ROOT.parent/'quartz_kinetic34_target_20261126/mcts_demo',8) as new:
    for r in co:
        for arm in (0,1,6):
            x=old.request(r['history'],arm=arm);y=new.request(r['history'],arm=arm)
            for k in ('schema','prefix','mid','end','snapshot','best','NN_calls','proof','rank','features'):assert x[k]==y[k],(r['id'],arm,k)
            for k in ('prefix_hooks','mid_hooks','end_hooks'):assert x[k]['ledger']==y[k]['ledger']
            n+=1
out=dict(same_histories=same,candidate_moves=candidate_moves,serial_common_prefix=all_prefix,NN_F80=nn0,NN_F80L=nn1,NN_saved=nn0-nn1,visits_F80=vis0,visits_F80L=vis1,lock_stops=stops,relative_reduction=1-nn1/nn0,C4_parity=n)
(D/'LOCK_PARITY.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,sort_keys=True))
