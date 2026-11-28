"""Independent terminal-only coordinate minimax on natural exact attempts."""
import json,subprocess,sys,collections as C
from pathlib import Path
from plan59 import D,T,sha,P
sys.path.insert(0,str(P))
import audit58 as a
ex=C.defaultdict(list)
lock=json.loads((D/'NN_LOCK.json').read_text())
for meta in lock['files']:
    p=D/meta['file'];assert sha(p)==meta['sha256']
    for gs in a.raw(p):
        for arm,g in gs.items():
            for r in g['rows']:
                z=r.get('partial')
                if z and z['exact'] is not None:
                    k=a.v.orient(r['history'],7)
                    ex[k].append(z)
assert sum(len(v) for v in ex.values())==6
binary=Path.home()/'Dropbox/personal_projects/quartz_kinetic51_targeted_backups/independent51'
expected=json.loads((D.parent/'quartz_kinetic51_targeted_20261127/EXACT_EXTENDED_LOCK.json').read_text())['binary_sha256']
assert sha(binary)==expected
keys=list(ex)
rs=list(map(json.loads,subprocess.run([str(binary)],input=''.join(f'{i} {a} {b}\n' for i,(a,b) in enumerate(keys)),text=True,capture_output=True,check=True,timeout=120).stdout.splitlines()))
assert len(rs)==len(keys) and {r['id'] for r in rs}==set(range(len(keys)))
rows=[];complete_receipts=0
for r in rs:
    k=keys[r['id']]
    if r['complete']:
        for z in ex[k]:
            vv=r['values'];assert all(l<=vv[str(m)]<=u for m,l,u in z['bounds'])
            if z['cert'] is not None:assert vv[str(z['cert'])]==max(vv.values())
            if z['exact']['complete']:
                assert vv[str(z['exact']['best'])]==z['exact']['value']==max(vv.values())
                complete_receipts+=1
    rows.append(dict(a=k[0],b=k[1],receipts=len(ex[k]),native_complete=sum(z['exact']['complete'] for z in ex[k]),**r))
out=dict(roots=len(keys),independent_complete=sum(r['complete'] for r in rs),native_complete_receipts_checked=complete_receipts,binary_sha256=expected,rows=rows)
(D/'NATURAL_EXACT_AUDIT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out),flush=True)
