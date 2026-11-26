from core import *
from trial import req
rr=json.loads((R/'quartz_kinetic30_20261126/COHORT.json').read_text())['roots'][:12]
n=0
with Server(ROOT.parent/'quartz_kinetic30_target_20261126/mcts_demo',8) as p,Server(BIN,8) as s:
    for r in rr:
        for old,arm in ((0,0),(8,9)):
            a=p.request(r['history'],'research_connect4_k30_cpu',arm=old);b=req(s,r,arm=arm)
            assert a['schema']=='quartz_k30_c4_v1'
            for k in ('prefix','mid','end','snapshot','best','NN_calls','proof','rank','features'):
                assert a[k]==b[k],(r['id'],arm,k)
            for k in ('mid_hooks','end_hooks'):assert a[k]['ledger']==b[k]['ledger']
            n+=1
print('K30 vs K31 scalar resident parity',n,'paired requests',flush=True)
