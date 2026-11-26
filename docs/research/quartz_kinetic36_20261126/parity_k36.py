from experiment import *
roots=read('COHORT.json')['roots'][:8]
n=0
with Server(OLD) as old,Server() as new:
    for r in roots:
        j=r['job'];h=r['history'];sd=seed(j,len(h))
        for cap in (80,128):
            for lk in (False,True):
                a=old.request(dict(cmd='research_rapfi_k35',history=h,seed=sd,cap=cap,threads=1,mask=0,table=None,lock=lk,trace=True))
                b=new.search(h,sd,cap,lk,trace=True)
                assert a['schema']=='quartz_rapfi_k35_v1'
                for k in ('proof','root_eval','initial','result','best','NN_calls','visits','cause'):
                    assert a[k]==b[k],(j['id'],cap,lk,k)
                assert a['prefix']['result']==b['prefix']['result']
                for k in ('prefix','hooks'):
                    x=a[k]['hooks'] if k=='prefix' else a[k];y=b[k]['hooks'] if k=='prefix' else b[k]
                    assert x['ledger']==y['ledger'] and x['trace']==y['trace']
                n+=1
print('Actual K35/K36 same-evaluator parity',n,'paired requests')
