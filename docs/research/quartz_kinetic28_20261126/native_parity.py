from core import *
from trial import req
rr=json.loads((R/'quartz_kinetic27_20261126/COHORT.json').read_text())['roots'][:12]
n=0
with Server(ROOT.parent/'quartz_kinetic27_target_20261126/mcts_demo',8) as p,Server(BIN,8) as s:
    for r in rr:
        for B in (8,32):
            for x,y in ((0,0),(3,4)):
                a=p.request(r['history'],'research_connect4_k27_cpu',budget=B,path=[x,x]);b=req(s,r,B,path=[y,y])
                assert a['schema']=='quartz_k27_c4_v1'
                for k in ('prefix','result','snapshot','NN_calls','visits','best','proof'):assert a[k]==b[k],(r['id'],B,x,k)
                for a0,b0 in zip(a['stages'],b['stages']):
                    assert a0['state']==b0['state']
                    assert a0['hooks']['ledger']==b0['hooks']['ledger']
                n+=1
print('K27 vs K28 serial common parity',n,'paired requests',flush=True)
