from core import *
r=json.loads((R/'quartz_kinetic31_20261126/COHORT.json').read_text())['roots'][0]
with Server(pathlib.Path(sys.argv[1]),8) as s:
    a=s.request(r['history'],arm=6,trace=True)
    assert a.get('schema')=='quartz_k32_c4_v1',a
    assert a['mid_graph'] and a['end_graph'] and len(a['end_graph']['paths'])==a['end']['visits']
    assert a['tail']['used_nn']+a['tail']['unused']==32 and a['tail']['visits']==a['end']['visits']-a['mid']['visits']
    assert a['rule_work']>0 and a['best']==a['end_graph']['best']
    for e in a['end_graph']['forced']:assert e['rank']<e['after'] and e['hidden']==(e['rank']>=e['before'])
    print('native proof',a['mid_graph']['risk'],a['end_graph']['risk'],a['tail']['cause'],len(a['end_graph']['forced']),sum(e['hidden'] for e in a['end_graph']['forced']),flush=True)
