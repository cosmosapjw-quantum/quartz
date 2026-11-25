"""Actual parent-engine comparison and edge materialization diagnostic."""
from source_gen import D,ROOT,read,raw,save,k20,BIN
from collections import Counter
par=ROOT.parent/'quartz_kinetic21_target_20261126/mcts_demo'
def pre(x):return {k:v for k,v in x.items() if k!='net'}
n=0;edge=Counter();by=Counter();changes=[]
with k20.Server(par) as a,k20.Server(BIN) as b:
    for t in raw(D/'TEST_ATLAS.jsonl.gz'):
        h=t['root']['history'];x=t['arms'][0];s=x['states'][0];r=a.request(h,'K21LIVE',seq_table=[[0.]*8 for _ in range(7)])
        assert 'error' not in r and r['visits']==8 and pre(r['states'][0])==pre(s)
        e={q['move'] for q in r['result']['edges']};m=x['panel'][2]
        edge['candidate_list']+=m in s['moves'][:s['len']];edge['materialized']+=m in e;edge['visited']+=s['n'][s['moves'][:s['len']].index(m)]>0
        y=a.request(h,'K21MATCH',nn_budget=x['goal_nn']);assert y['result']==x['result'] and y['best']==x['best'] and y['NN_calls']==x['NN_calls'] and y['visits']==x['visits']
        # Actual forced reservoir response, check if it changes materialized edges or recommendation.
        z=t['arms'][2];ix=s['moves'][:s['len']].index(m);label=next(v for v in read('TEST_LABELS.json')['rows'] if v['root']['id']==t['root']['id'])
        u={int(k):(v+1)/2 for k,v in label['independent'].items()};key='materialized' if m in e else 'unmaterialized'
        by[(key,'roots')]+=1;by[(key,'better')]+=int(u[z['best']]>u[x['best']]);by[(key,'worse')]+=int(u[z['best']]<u[x['best']])
        if u[z['best']]!=u[x['best']]:changes.append(dict(id=t['root']['id'],prefix=s['best'],normal=x['best'],reservoir=z['best'],target=m,materialized=m in e,prior_visits=s['n'][ix],u_normal=u[x['best']],u_reservoir=u[z['best']]))
        n+=1
    for arm,opts in [('K22LIVE',{}),('K22LIVE',{'causal_table':[3]*8}),('K22N',{'causal_table':[0]*8}),('K22F',{'oracle':1})]:
        z=b.request(h,arm,**opts);assert 'error' in z
save('C09_NATIVE_PARITY.json',dict(parent_prefix_parity=n,parent_normal_target_parity=n,malformed_rejected=4,coverage=dict(edge),strata={k+':'+j:v for (k,j),v in by.items()},changes=changes))
print(read('C09_NATIVE_PARITY.json'))
