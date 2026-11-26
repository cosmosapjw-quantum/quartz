from experiment import *
roots=read('COHORT.json')['roots'][:8]
n=0
with Server(OLD) as old,Server() as new:
    for r in roots:
        h=r['history'];j=r['job'];sd=seed(j,len(h))
        for arm,mask,cap,lk in [('B128',0,128,False),('B80',0,80,False),('F80',15,80,False),('F80L',15,80,True)]:
            a=old.request(dict(cmd='research_rapfi_k34',history=h,arm=arm,seed=sd,trace=True))
            b=new.search(h,sd,mask,cap,1,lk,trace=True)
            assert a['schema']=='quartz_rapfi_k34_v1'
            for k in ('proof','root_eval','initial','result','best','NN_calls','visits','cause'):
                assert a[k]==b[k],(j['id'],arm,k)
            assert a['prefix']['result']==b['prefix']['result']
            for k in ('prefix','hooks'):
                x=a[k]['hooks'] if k=='prefix' else a[k];y=b[k]['hooks'] if k=='prefix' else b[k]
                assert x['ledger']==y['ledger'] and x['trace']==y['trace']
            n+=1
print('Actual K34/K35 same-evaluator old/new parity',n,'paired requests',flush=True)
