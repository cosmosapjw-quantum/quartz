import importlib.util,json,math
from engine import Engine,D,ROOT,identity
from rules import fixtures
from native_smoke import check
old=ROOT/'docs/research/quartz_fouraxis06_20261125/engine.py'
sp=importlib.util.spec_from_file_location('old_engine',old);oldmod=importlib.util.module_from_spec(sp);sp.loader.exec_module(oldmod)
f=fixtures()
cases=[('trans_a','freestyle'),('overline','standard'),('long','freestyle')]
a=[]
with oldmod.Engine(1) as e:
    for name,rule in cases:a.append(e.search(f[name],rule,'BASE',32,1,17))
par=[]
with Engine(1) as e:
    for (name,rule),old in zip(cases,a):
        new=e.search(f[name],rule,'BASE',32,1,17);check(new,f[name],rule)
        for k in ('board','rule_key','tt_hash','proof','legal','best','visits','evaluator_calls','nn_rows','root_nn_feature_sha256'):assert new[k]==old[k],(name,k)
        x=new['result'];y=old['result']
        for k in ('root_hash','root_n','root_w','best','evaluator_calls','tt_size','candidates'):assert x[k]==y[k],(name,k)
        assert len(x['edges'])==len(y['edges'])
        max_m2=0
        for ex,ey in zip(x['edges'],y['edges']):
            for k in ex:
                if k=='m2':max_m2=max(max_m2,abs(ex[k]-ey[k]));assert abs(ex[k]-ey[k])<1e-10
                else:assert ex[k]==ey[k],(name,k,ex[k],ey[k])
        par.append(dict(fixture=name,rule=rule,max_m2_delta=max_m2,old=old,new=new))
        print('SERIAL_PARITY',name,rule,new['nn_rows'],max_m2,flush=True)
parallel=[]
with Engine(2) as e:
    for name,rule in cases:
        for seed in (17,43):
            r=e.search(f[name],rule,'BASE',64,2,seed);check(r,f[name],rule)
            assert r['visits']==64 and r['nn_rows']==r['evaluator_calls']
            parallel.append(dict(fixture=name,rule=rule,seed=seed,result=r))
            print('PARALLEL',name,rule,seed,r['visits'],r['nn_rows'],flush=True)
(D/'C08_NATIVE_REGRESSION.json').write_text(json.dumps(dict(identity=identity(),serial=par,parallel=parallel),separators=(',',':'),allow_nan=False)+'\n')
