"""Deduplicate natural exact roots, independently solve terminal-only coordinate minimax."""
from plan62 import D,T,sha,read
from pathlib import Path
import json,gzip,collections as C,subprocess,sys
def main(modes):
    ex=C.defaultdict(list)
    for mode in modes:
        with gzip.open(T/(mode+'_AUDIT.json.gz'),'rt') as f:a=json.load(f)
        for k,vs in a['exact']:ex[tuple(k)].extend((mode,z) for z in vs)
        assert a['proof'].get('exact_attempt',0)==sum(m==mode for v in ex.values() for m,z in v)
    binary=Path.home()/'Dropbox/personal_projects/quartz_kinetic51_targeted_backups/independent51'
    expected=read(D.parent/'quartz_kinetic51_targeted_20261127/EXACT_EXTENDED_LOCK.json')['binary_sha256'];assert sha(binary)==expected
    keys=list(ex);rows=[];receipt=C.Counter()
    if keys:
        rs=list(map(json.loads,subprocess.run([str(binary)],input=''.join(f'{i} {a} {b}\n' for i,(a,b) in enumerate(keys)),text=True,capture_output=True,check=True,timeout=180).stdout.splitlines()))
        assert len(rs)==len(keys) and {r['id'] for r in rs}==set(range(len(keys)))
        for r in rs:
            k=keys[r['id']]
            for mode,z in ex[k]:
                receipt[mode+'_native_attempt']+=1
                receipt[mode+'_native_complete']+=z['exact']['complete']
                if r['complete']:
                    vv=r['values'];assert all(l<=vv[str(m)]<=u for m,l,u in z['bounds'])
                    if z['cert'] is not None:assert vv[str(z['cert'])]==max(vv.values())
                    if z['exact']['complete']:assert vv[str(z['exact']['best'])]==z['exact']['value']==max(vv.values());receipt[mode+'_native_complete_checked']+=1
                    receipt[mode+'_independent_complete_receipts']+=1
            rows.append(dict(a=k[0],b=k[1],receipts=len(ex[k]),**r))
    out=dict(modes=modes,roots=len(keys),independent_complete=sum(r['complete'] for r in rows),receipt_counts=dict(receipt),binary_sha256=expected,rows=rows)
    (T/('EXACT_'+'_'.join(modes)+'.json')).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print({k:v for k,v in out.items() if k!='rows'},flush=True)
if __name__=='__main__':main(sys.argv[1:])
