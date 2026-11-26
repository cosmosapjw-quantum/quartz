from core import *
r=json.loads((R/'quartz_kinetic28_20261126/COHORT.json').read_text())['roots'][0]
with Server(pathlib.Path(sys.argv[1]),8) as s:
    a=s.request(r['history'],arm=7,trace=True)
    print({k:a.get(k) for k in ('schema','arm','NN_calls','physical_calls','best')},flush=True)
    assert a.get('schema')=='quartz_k29_c4_v1',a
    q=a['queue'];assert q['logical']==a['NN_calls']-a['mid']['nn']
    assert q['logical']==q['physical']+q['coalesced']
    assert q['parity_checks']==q['physical'] and q['peak_pending']<=8
    assert a['physical_calls']==a['NN_calls']-q['coalesced']
    assert a['tail']['used_nn']<=32 and a['end_hooks']['ledger']['repeated_delivery']==0
