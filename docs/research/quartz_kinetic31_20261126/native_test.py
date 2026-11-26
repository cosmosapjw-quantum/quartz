from core import *
r=json.loads((R/'quartz_kinetic30_20261126/COHORT.json').read_text())['roots'][0]
with Server(pathlib.Path(sys.argv[1]),8) as s:
    a=s.request(r['history'],arm=8,trace=True)
    assert a.get('schema')=='quartz_k31_c4_v1',a
    g=a['proof_graph'];assert g['arm']==8 and len(g['paths'])==a['mid']['visits']
    assert a['mid']==a['end'] and a['tail'] is None and a['queue'] is None
    assert g['end']['rule_work']==a['rule_work']>0 and a['best']==g['end']['best']
    assert len(g['steps'])<=8 and len(a['direct_features'])==16
    print('graph',g['before']['risk'],g['end']['risk'],len(g['steps']),a['rule_work'],flush=True)
