from plan65 import D,T,read,raw,sha,family,ROOT,DEV
from fractions import Fraction as F
from collections import defaultdict,Counter
import math,json
p=read(D/'PLAN.json');lock=read(T/'DEV_LOCK.json');summary=read(T/'DEV_SUMMARY.json')
new={family(j['opening']) for j in p['jobs']};assert len(new)==48
for l in p['prior_locks']:
 path=ROOT/l['path'];assert sha(path)==l['sha256']
 old={family(j['opening']) for j in read(path).get('jobs',[]) if isinstance(j.get('opening'),list) and len(j['opening'])==4 and all(type(x)==int and 0<=x<49 for x in j['opening'])}
 assert not (old & new)
acc=defaultdict(Counter);den=Counter();ids=set();raw_counts=Counter()
for m in lock['files']:
 f=T/m['file'];assert sha(f)==m['sha256']
 for z in raw(f):
  j=z['job'];assert j==p['roots'][j['id']] and j['id'] not in ids;ids.add(j['id'])
  d=j['domain'];weight=F(j['population'],j['sample']);den[d]+=weight
  for arm in DEV:
   r=z['arms'][arm];es=r['allocation65']['edges'];bd={m:(l,u) for m,l,u in r['partial']['bounds']};F0={i for i,(l,u) in bd.items() if u>-1} or set(bd)
   visits=sum(n for i,n,v,pr in es if i in F0);covered=sum(n>0 for i,n,v,pr in es if i in F0)
   bad=sum(n for i,n,v,pr in es if bd[i][1]==-1);unknown=dict(r['votes'])[r['move']]==0
   assert visits+bad==4 and (arm not in ('M4','MA4','X4') or bad==0)
   for k,v in [('visited',covered),('loss_visits',bad),('unvisited_chosen',unknown),('feasible',len(F0)),('physical',r['actual_nn']),('logical',r['nn'])]:
    acc[d,arm][k]+=weight*v
   raw_counts[d,arm,'all_loss']+=int(bad==4);raw_counts[d,arm,'loss_visits']+=bad
assert len(ids)==554
out={}
for d in den:
 assert den[d]==sum(p['population'][d])
 out[d]={}
 for arm in DEV:
  q={k:str(v/den[d]) for k,v in acc[d,arm].items()}
  for k,v in q.items():assert math.isclose(float(F(v)),summary[d]['arms'][arm][k],rel_tol=1e-12,abs_tol=1e-12)
  for k in ('all_loss','loss_visits'):assert raw_counts[d,arm,k]==summary[d]['arms'][arm]['raw_'+k]
  out[d][arm]=q
r=dict(roots=len(ids),families=len(new),prior_locks=len(p['prior_locks']),population={k:str(v) for k,v in den.items()},exact_weighted=out,lock_sha256=sha(T/'DEV_LOCK.json'))
(T/'INDEPENDENT.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(r)
