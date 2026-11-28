"""Exploratory post-atlas proposal-only ceilings; no policy fit or arena selection."""
from plan66 import D,T,read,sha,raw
from fractions import Fraction as F
from collections import Counter,defaultdict
from math import comb
import json
p=read(D/'PLAN.json');lock=read(T/'DEV_LOCK.json');acc=defaultdict(Counter);den=Counter();n=Counter()
for z in lock['files']:
 f=T/z['file'];assert sha(f)==z['sha256']
 for g in raw(f):
  j=g['job'];d=j['domain'];w=F(j['population'],j['sample']);den[d]+=w;n[d]+=1
  y={int(m):x['score2'] for m,x in g['forced'].items()};es={m:(v,q,pr) for m,v,q,pr in j['snapshot']['edges']};bd={m:(l,u) for m,l,u in j['bounds']}
  safe=set(j['snapshot']['safe']);live={m for m in safe if es[m][0] or bd[m][0]==bd[m][1]} or {j['incumbent']}
  ml=max(y[m] for m in live);mf=max(y[m] for m in safe);unseen=sorted(safe-live,key=lambda m:(-es[m][2],m));N=len(unseen)
  acc[d]['safe_omission']+=w*F(mf-ml,2);acc[d]['root_hidden_present']+=w*bool(N)
  for k0 in (1,2,4):
   k=min(k0,N)
   if not k:continue
   z1=F(max([ml]+[y[m] for m in unseen[:k]])-ml,2)
   # E[(max of k without replacement - ml)_+]/2 via survival sum on score2 thresholds.
   q=F()
   for threshold in range(ml+1,3):
    below=sum(y[m]<threshold for m in unseen)
    q+=F(1,2)*(1-F(comb(below,k) if below>=k else 0,comb(N,k)))
   acc[d][f'prior_top{k0}_oracle_gain']+=w*z1;acc[d][f'uniform{k0}_oracle_gain']+=w*q
  if N:
   acc[d]['hidden_improvement_fraction']+=w*F(sum(y[m]>ml for m in unseen),N)
   acc[d]['hidden_better_inc_fraction']+=w*F(sum(y[m]>y[j['incumbent']] for m in unseen),N)
  if mf>ml:
   acc[d]['strict_omission_mass']+=w
   acc[d]['omitted_best_mass']+=w*sum(es[m][2] for m in unseen if y[m]==mf)/sum(es[m][2] for m in safe)
   acc[d]['omitted_best_count']+=w*sum(y[m]==mf for m in unseen)
out={}
for d,w in den.items():
 assert w==sum(p['population'][d].values())
 v={k:float(x/w) for k,x in acc[d].items()};v['sample']=n[d];v['population']=int(w)
 # Exact boundedness: every reveal ceiling <= total omission.
 for k in (1,2,4):
  assert acc[d][f'prior_top{k}_oracle_gain']<=acc[d]['safe_omission'] and acc[d][f'uniform{k}_oracle_gain']<=acc[d]['safe_omission']
 v['exact']={k:str(x/w) for k,x in acc[d].items() if isinstance(x,(int,F))}
 out[d]=v
(T/'FRONTIER.json').write_text(json.dumps(dict(domains=out,scope='oracle reveal ceiling; full finite continuation labels, exploratory, no operational guarantee; absent frontier contributes zero',lock_sha256=sha(T/'DEV_LOCK.json')),indent=2,sort_keys=True)+'\n')
for d,v in out.items():print(d,{k:round(x,8) for k,x in v.items() if isinstance(x,(int,float))})
