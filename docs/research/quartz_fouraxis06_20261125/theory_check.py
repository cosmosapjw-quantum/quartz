from fractions import Fraction as F
from itertools import product
from collections import defaultdict
import gzip,json,statistics as S,random
from pathlib import Path
from publish import D,ROOT,sha
# Exact two-ply local line lemma for all ternary lines of length8 and all
# empty pairs; four directions follow separately, excluding occupied a.
n=0
def wins(row,b,p,exact):
    k=1
    for d in (-1,1):
        j=b+d
        while 0<=j<len(row) and row[j]==p:k+=1;j+=d
    return k==5 if exact else k>=5
for row in product(range(3),repeat=8):
    for p in (1,2):
        for ex in (False,True):
            old={i for i,x in enumerate(row) if x==0 and wins(row,i,p,ex)}
            for a,v in enumerate(row):
                if v:continue
                b=list(row);b[a]=3-p
                new={i for i,x in enumerate(b) if x==0 and wins(b,i,p,ex)}
                assert new==old-{a};n+=1
def rw(iv):
    return min(max(0,max((h for j,(_,h) in enumerate(iv) if i!=j),default=-1)-l) for i,(l,u) in enumerate(iv))
assert rw([(-1,1)]*3)==2 and rw([(-1,-1),(-1,1),(-1,1)])==2 and rw([(-1,-1),(-1,-1),(-1,1)])==0
for g in (-1,0,1):
    for p in (-1,1):
        assert -g*p==g*(-p);n+=1
# Earlier disjoint FA05 TRAIN/CAL diagnostic first-block observations. Verify
# exact Git blob identity, no FA05 TEST accessed. Matched prefix only.
p=ROOT.parent/'quartz_fouraxis05_20261125/docs/research/quartz_fouraxis05_20261125'
for name in ('C07_DIAG_RAW.jsonl.gz','C07_DIAG_LABELS.json'):
    import subprocess as sp
    assert sp.check_output(['git','hash-object',str(p/name)],cwd=ROOT,text=True).strip()==sp.check_output(['git','rev-parse','9b8a4718f7f92d83db2fb00e5806c6c4bd8cbfc2:docs/research/quartz_fouraxis05_20261125/'+name],cwd=ROOT,text=True).strip()
z={x['id']:x['scores'] for x in json.loads((p/'C07_DIAG_LABELS.json').read_text()) if x['status']=='EXACT'}
utility=lambda x:F(1+(x>0)-(x<0),2)
by={}
with gzip.open(p/'C07_DIAG_RAW.jsonl.gz','rt') as f:
    for line in f:
        a=json.loads(line)
        if a['arm']=='BASE' or not a['steps']:continue
        assert a['group'] in ('TRAIN','CAL') and a['mode']=='DIAG'
        pre=a['prefix'];post=a['steps'][0]['post'];iv=lambda q:[x[1] for x in q['proof']['actions']]
        d=F(rw(iv(pre))-rw(iv(post)),2);assert d>=0
        g=utility(z[a['id']][post['best']])-utility(z[a['id']][pre['best']])
        by[a['group'],a['id'],a['arm']]=(pre['result'],pre['proof'],d,g,a['steps'][0]['plan_us']+a['steps'][0]['search_us'])
out={}
for group in ('TRAIN','CAL'):
    out[group]={}
    for arm in ('SAFE','ACTIVE','FOCUS','EXPAND','FRONTIER','EVAL','WRITER','MERGE'):
        rows=[(v,by[g,i,'SAFE']) for (g,i,a),v in by.items() if g==group and a==arm]
        assert rows
        for v,r in rows:assert v[:2]==r[:2]
        d=[v[2] for v,r in rows];x=[v[3] for v,r in rows];delta=[v[3]-r[3] for v,r in rows]
        out[group][arm]=dict(families=len(rows),risk_drop_count=sum(y>0 for y in d),mean_risk_drop=float(S.mean(d)),p_risk=float(F(sum(y>0 for y in d),len(d))),conditional_risk_drop=float(S.mean([y for y in d if y>0])) if any(d) else None,signed_gain=float(S.mean(x)),paired_vs_safe=float(S.mean(delta)),positive=sum(y>0 for y in x),negative=sum(y<0 for y in x),mean_plan_search_us=S.mean(v[4] for v,r in rows))
(D/'R07_LOCAL_EVIDENCE.json').write_text(json.dumps(dict(assertions=n,sources={name:sha(p/name) for name in ('C07_DIAG_RAW.jsonl.gz','C07_DIAG_LABELS.json')},groups=out,scope='previous exact TRAIN/CAL first block, unresolved common prefix only; descriptive, no new label or generation'),indent=2,sort_keys=True)+'\n')
print('Exact line and terminal-label checks',n)
print(json.dumps(out,sort_keys=True))
