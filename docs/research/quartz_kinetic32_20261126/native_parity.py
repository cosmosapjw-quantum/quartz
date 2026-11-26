from core import *
rr=json.loads((R/'quartz_kinetic31_20261126/COHORT.json').read_text())['roots'][:12]
n=0
with Server(ROOT.parent/'quartz_kinetic31_target_20261126/mcts_demo',8) as p,Server(BIN,8) as s:
    for r in rr:
        for old,new in ((0,1),(9,0)):
            a=p.request(r['history'],'research_connect4_k31_cpu',arm=old);b=s.request(r['history'],arm=new)
            assert a['schema']=='quartz_k31_c4_v1' and b['schema']=='quartz_k32_c4_v1'
            for k in ('prefix','mid','end','snapshot','best','NN_calls','proof','rank','features'):
                assert a[k]==b[k],(r['id'],old,k)
            for k in ('mid_hooks','end_hooks'):assert a[k]['ledger']==b[k]['ledger']
            n+=1
print('K31 vs K32 scalar resident parity',n,'paired requests',flush=True)
