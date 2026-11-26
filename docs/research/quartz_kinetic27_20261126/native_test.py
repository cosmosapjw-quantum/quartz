from core import *
r=json.loads((R/'quartz_kinetic26_20261126/COHORT.json').read_text())['roots'][0]
with Server(pathlib.Path(sys.argv[1]),8) as srv:
    a=srv.request(r['history'],budget=8,path=[7,8],trace=True)
    print({k:a.get(k) for k in ('schema','chosen','visits','NN_calls','best')},flush=True)
    assert a.get('schema')=='quartz_k27_c4_v1',a
    assert a['chosen']==[7,8] and a['prefix']['visits']==8
    for j,s in enumerate(a['stages']):
        b=s['budget'];assert b['reserve']==[2,4][j] and b['released']+b['protected_extra']==b['reserve']
        assert b['used_nn']+b['unused']==8 and b['protected_base']<=8-b['reserve']
    assert a['stages'][1]['hooks']['ledger']['repeated_delivery']==0
