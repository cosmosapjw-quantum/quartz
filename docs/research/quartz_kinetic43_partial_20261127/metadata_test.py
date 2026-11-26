import sys,pathlib
D=pathlib.Path.home()/'quantum/research/quartz_kinetic43_partial_20261127/docs/research/quartz_kinetic43_partial_20261127'
sys.path.insert(0,str(D))
import importlib
e=importlib.import_module(sys.argv[1])
g=next(g for g in e.rows('NN_SOURCE') if g['atlas'])
r=next(r for r in g['rows'] if r['kind']!='CERTIFICATE')
a=g['atlas'][0];h=g['history'][:r['ply']]
x=e.atlas(h,7,a['B'],a['L']);expected=list(h)
h.append(r['move'])
assert x['history']==expected,'atlas retains mutable history alias'
assert all(x['tiers'][i]['receipt']==a['tiers'][i]['receipt'] for i in range(3))
print('correct immutable atlas prefix and same actual native proof receipts',len(expected))
