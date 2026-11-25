from source_gen import D,ROOT,k20,BIN,read,raw,save
import math
p=ROOT.parent/'quartz_kinetic23_target_20261126/mcts_demo'
n=0
with k20.Server(p) as old,k20.Server(BIN) as new:
    for t in raw(D/'TEST_ATLAS.jsonl.gz'):
        h=t['root']['history'];pre=t['arms'][0]['prefix']
        a=old.request(h,'K21TRACE');assert 'error' not in a
        s=next(x for x in a['states'] if x['visits']==32)
        assert s['nn']==pre['evaluator_calls'] and s['best']==pre['best']
        es={e['move']:e for e in pre['edges']}
        for i,m in enumerate(s['moves'][:s['len']]):
            e=es.get(m,{})
            assert s['n'][i]==e.get('n',0)
            assert math.isclose(s['w'][i],e.get('w',0),rel_tol=1e-10,abs_tol=1e-10)
            assert math.isclose(s['m2'][i],e.get('m2',0),rel_tol=1e-10,abs_tol=1e-10)
        n+=1
    bad=0
    for arm,opts in [('K24LIVE',{}),('K24LIVE',{'merge_table':[3]*4}),('K24N',{'merge_table':[0]*4}),('K24D',{'oracle':1})]:
        x=new.request(h,arm,**opts);assert 'error' in x;bad+=1
save('C09_NATIVE_PARITY.json',dict(parent_prefix32=n,invalid_rejected=bad))
print(read('C09_NATIVE_PARITY.json'))
