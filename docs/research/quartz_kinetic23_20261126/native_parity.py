from source_gen import D,ROOT,k20,BIN,read,raw,save
from collections import Counter
par=ROOT.parent/'quartz_kinetic22_target_20261126/mcts_demo'
n=Counter();targets=Counter()
with k20.Server(par) as old,k20.Server(BIN) as new:
    for t in raw(D/'TEST_ATLAS.jsonl.gz'):
        h=t['root']['history']
        for i,arm in [(0,'K22N'),(6,'K22X')]:
            x=t['arms'][i];y=old.request(h,arm);assert 'error' not in y
            assert x['states'][0]==y['states'][0];n['prefix']+=1
            if x['incremental_nn']==[6,6]:
                assert x['result']==y['result'] and x['best']==y['best'] and x['NN_calls']==y['NN_calls'] and x['visits']==y['visits']
                n['NN' if i==0 else 'XN']+=1
            else:n['cap_excluded']+=1
        for x in t['arms']:
            for j,c in enumerate(x['chosen']):
                if c==0:continue
                s=x['states'][j];m=x['forced'][j];n0=s['n'][s['moves'].index(m)]
                targets[(j,c,'listed')]+=1
                targets[(j,c,'visited')]+=int(n0>0)
    for arm,opts in [('K23PATH',{}),('K23PATH',{'h2_path':[0,3]}),('K23LIVE',{'h2_table':{'first':[0]*8,'second':[[0]*8]*2}}),('K23LIVE',{'h2_table':{'first':[3]*8,'second':[[0]*8]*3}}),('K22N',{'h2_path':[0,0]})]:
        y=new.request(h,arm,**opts);assert 'error' in y;n['invalid_rejected']+=1
save('C09_NATIVE_PARITY.json',dict(parity=dict(n),targets={f'{j}:{c}:{k}':v for (j,c,k),v in targets.items()},target_scope='listed in typed legal array; visited implies materialized; zero visits do not certify unmaterialized'))
print(read('C09_NATIVE_PARITY.json'))
