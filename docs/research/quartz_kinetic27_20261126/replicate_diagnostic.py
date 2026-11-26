"""Post-lock descriptive repeat aggregation and worker-count comparison, no refit."""
from core import D,read,raw,sha,save,S,random
from fractions import Fraction as F
f=read('FROZEN.json');labels=read('TEST_LABELS.json')
u={z['root']['id']:{int(a):F(v+1,2) for a,v in z['independent'].items()} for z in labels['rows'] if z['status']=='EXACT'}
rows={}
for fi in read('TEST_LIVE_LOCK.json')['files']:
    p=D/fi['file'];assert sha(p)==fi['sha256']
    for t in raw(p):
        i=t['root']['id']
        if i not in u:continue
        rows[i]=t['results']
assert set(rows)==set(u)
ids=sorted(rows)
def delta(a,b):
    aa=[(u[i][rows[i][a+'0']['best']]+u[i][rows[i][a+'1']['best']])/2-(u[i][rows[i][b+'0']['best']]+u[i][rows[i][b+'1']['best']])/2 for i in ids]
    rng=random.Random(2026112627)
    bs=sorted(sum(rng.choices(list(map(float,aa)),k=len(aa)))/len(aa) for _ in range(4000))
    return dict(n=len(aa),gain=float(S.mean(aa)),ci=[bs[99],bs[3899]],better=sum(x>0 for x in aa),worse=sum(x<0 for x in aa))
def pair_diag(a,b):
    gains=[]
    for fi in read('TEST_ATLAS_LOCK.json')['files']:
        p=D/fi['file'];assert sha(p)==fi['sha256']
        for t in raw(p):
            i=t['root']['id']
            if i not in u:continue
            x,y=t['paths'][a],t['paths'][b];g=u[i][x['best']]-u[i][y['best']]
            gains.append((i,g,all(z['budget']['used_nn']==8 for z in x['stages']+y['stages'])))
    gains.sort()
    def agg(g):
        rng=random.Random(2026112627)
        vals=[float(x[1]) for x in g];bs=sorted(sum(rng.choices(vals,k=len(vals)))/len(vals) for _ in range(4000))
        return dict(n=len(g),gain=S.mean(vals),ci=[bs[99],bs[3899]],better=sum(x>0 for x in vals),worse=sum(x<0 for x in vals))
    return dict(all=agg(gains),matched=agg([z for z in gains if z[2]]))
out=dict(scope='post-lock descriptive diagnostic, two actual repeats averaged per root; no policy refit or selection from TEST',repeat_average={'dynamic_vs_base':delta('dynamic','base'),'dynamic_vs_fast':delta('dynamic','fast'),'fast_vs_base':delta('fast','base')},worker_fixed={'tree4_vs_serial_base':pair_diag(1,0),'tree8_vs_tree4_base':pair_diag(2,1),'tree8_vs_tree4_full':pair_diag(5,4)})
save('REPLICATE_DIAGNOSTIC.json',out);print(out)
