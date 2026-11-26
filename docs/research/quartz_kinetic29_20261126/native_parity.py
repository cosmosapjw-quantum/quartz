from core import *
from trial import req
rr=json.loads((R/'quartz_kinetic28_20261126/COHORT.json').read_text())['roots'][:12]
n=0
with Server(ROOT.parent/'quartz_kinetic28_target_20261126/mcts_demo',8) as p,Server(BIN,8) as s:
    for r in rr:
        for arm in (0,8):
            a=p.request(r['history'],'research_connect4_k28_cpu',budget=32,path=[0,0 if arm==0 else 20]);b=req(s,r,arm=arm)
            assert a['schema']=='quartz_k28_c4_v1'
            for x,y in [(a['prefix_lite'],b['prefix']),(a['stages'][0]['lite'],b['mid']),(a['stages'][-1]['lite'],b['end']),(a['snapshot'],b['snapshot']),(a['best'],b['best']),(a['NN_calls'],b['NN_calls']),(a['proof'],b['proof'])]:assert x==y,(r['id'],arm)
            assert a['stages'][0]['hooks']['ledger']==b['mid_hooks']['ledger']
            assert a['stages'][-1]['hooks']['ledger']==b['end_hooks']['ledger']
            n+=1
print('K28 vs K29 scalar resident parity',n,'paired requests',flush=True)
