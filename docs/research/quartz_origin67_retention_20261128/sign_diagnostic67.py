"""Exploratory diagnosis of sign discrimination, not a new fitted policy."""
from plan67 import D,T,read,sha
from retention67 import expected,predict
import json,gzip,math
from collections import Counter
m=read(D/'MODEL.json')
with gzip.open(T/'DATA.json.gz','rt') as f:rows=json.load(f)
out={}
for sp in ('TRAIN','CAL'):
 for op in ('VOTE','FOCUS','EXPAND'):
  rr=[r for r in rows if r['split']==sp and r['op']==op]
  z=[(expected(predict(m,r['x'])),r['y']) for r in rr]
  pos=[g for g,y in z if y>0];neg=[g for g,y in z if y<0]
  # Conditional AUC for distinguishing positive/negative among nonzero labels; raw pair statistic, not iid.
  n=len(pos)*len(neg);c2=sum(2*int(a>b)+int(a==b) for a in pos for b in neg)
  out[sp+':'+op]=dict(n=len(z),positive=len(pos),negative=len(neg),ties=sum(y==0 for g,y in z),gain_min=min(g for g,y in z),gain_max=max(g for g,y in z),positive_prediction=sum(g>0 for g,y in z),conditional_sign_auc=c2/(2*n) if n else None,auc_pairs=n)
r=dict(scope='post-fit diagnostic, no policy selection, raw conditional AUC among nonzero outcomes',model_sha256=sha(D/'MODEL.json'),data_sha256=sha(T/'DATA.json.gz'),groups=out)
(T/'SIGN_DIAGNOSTIC.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
print(json.dumps(r,indent=2))
