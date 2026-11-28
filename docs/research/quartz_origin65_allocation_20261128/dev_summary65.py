from plan65 import D,T,DEV,sha,read
from collections import Counter
import gzip,json,math
with gzip.open(T/'DEV_AUDIT.json.gz','rt') as f:a=json.load(f)
assert a['lock_sha256']==sha(T/'DEV_LOCK.json')
o={}
for d in ('MAIN','ARCHIVE'):
 rows=[r for r in a['rows'] if r['domain']==d];w=sum(r['weight'] for r in rows)
 out={}
 for arm in DEV:
    ms=[r['arms'][arm] for r in rows]
    out[arm]={k:math.fsum(r['weight']*x[k] for r,x in zip(rows,ms))/w for k in ('visited','loss_visits','unvisited_chosen','mass','feasible','physical','logical')}
    out[arm]['raw_loss_visits']=sum(x['loss_visits'] for x in ms);out[arm]['raw_unvisited_chosen']=sum(x['unvisited_chosen'] for x in ms)
    out[arm]['raw_all_loss']=sum(x['loss_visits']==4 for x in ms)
 contrasts={}
 for p,q in [('A4','L4'),('M4','L4'),('MA4','M4'),('MA4','A4'),('X4','MA4')]:
    contrasts[p+'-'+q]=dict(changed=sum(r['arms'][p]['chosen']!=r['arms'][q]['chosen'] for r in rows),weighted_change=math.fsum(r['weight'] for r in rows if r['arms'][p]['chosen']!=r['arms'][q]['chosen'])/w)
 o[d]=dict(n=len(rows),population=w,arms=out,contrasts=contrasts)
(T/'DEV_SUMMARY.json').write_text(json.dumps(o,indent=2,sort_keys=True)+'\n');print(json.dumps(o,indent=2))
