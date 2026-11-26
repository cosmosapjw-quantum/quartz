import sys,pathlib,json,hashlib
D=pathlib.Path(__file__).resolve().parent
W=pathlib.Path.home()/'quantum/research/quartz_kinetic40_budget_20261126'
sys.path.insert(0,str(W/'docs/research/quartz_kinetic37_20261126'))
import experiment as e
old=pathlib.Path.home()/'quantum/research/quartz_kinetic37_target_20261126/mcts_demo'
new=D/'mcts_demo'
co=e.read('COHORT.json')['roots']
h=co[0]['history'];seed=40
if sys.argv[1]=='red':
    with e.Server(old) as s:
        a=s.search(h,seed,128,True)
    assert a.get('certificate_checks',0)>0,'missing explicit quiescent certificate telemetry'
else:
    keys=('best','cause','visits','NN_calls','proof','root_eval','cap','probe_wave','bin','selected_mask')
    def clean(a):
        return {k:a[k] for k in keys}|{'votes':e.votes(a['result']), 'waves':[[z[k] for k in ('before_n','before_nn','before_d','after_n','after_nn','after_d','after_best')] for z in a['waves']]}
    rows=[];checks=nn=0
    with e.Server(old) as a,e.Server(new) as b:
        for j in co[:48]:
            h=j['history']
            for cap in (80,128):
                pair=[]
                for lock in (False,True):
                    x=a.search(h,seed,cap,lock);y=b.search(h,seed,cap,lock)
                    assert clean(x)==clean(y),(h,cap,lock)
                    assert y.get('certificate_checks',0)>0
                    checks+=y['certificate_checks'];nn+=x['NN_calls']+y['NN_calls']
                    pair.append(y)
                    rows.append(dict(history=h,cap=cap,lock=lock,old=clean(x),new=clean(y),certificate_checks=y['certificate_checks']))
                assert pair[0]['best']==pair[1]['best'] and pair[1]['NN_calls']<=pair[0]['NN_calls']
    out=dict(requests=len(rows)*2,roots=48,quiescent_checks=checks,actual_NN=nn,new_binary_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),old_binary_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),rows=rows)
    (D/'NATIVE_PARITY.json').write_text(json.dumps(out,indent=2)+'\n')
    print({k:v for k,v in out.items() if k!='rows'},flush=True)
