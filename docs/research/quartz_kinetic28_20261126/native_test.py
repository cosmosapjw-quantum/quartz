from core import *
r=json.loads((R/'quartz_kinetic27_20261126/COHORT.json').read_text())['roots'][0]
with Server(pathlib.Path(sys.argv[1]),8) as srv:
    a=srv.request(r['history'],budget=32,path=[15,20])
    print({k:a.get(k) for k in ('schema','chosen','visits','NN_calls','best','stopped')},flush=True)
    assert a.get('schema')=='quartz_k28_c4_v1',a
    assert a['chosen']==[15,20] and a['stopped'] and len(a['stages'])==1
    z=a['stages'][0];b=z['budget'];assert b['reserve']==4 and b['protected_extra']+b['released']==4
    assert b['protected_extra']+b['borrowed']==b['pre_extra'] and b['used_nn']+b['unused']==32
    assert a['best']==a['mid_best'] and a['result']==z['state'] and a['NN_calls']-a['prefix']['nn']==b['used_nn']
