"""FIT-only medians and independent full-pool rank envelope, no utility refit."""
import math,statistics
def point_cost(rows):
    assert rows and all(len(r)==3 and 0<r[0]<=r[1]<=r[2] and all(math.isfinite(x) for x in r) for r in rows)
    return [max(1,math.ceil(statistics.median(r[j] for r in rows))) for j in range(3)]
def envelope(scores):
    assert all(math.isfinite(x) and x>=0 for x in scores)
    m=len(scores);k=(9*(m+1)+9)//10
    return k,None if k>m else max(1.,sorted(scores)[k-1])
def tables(gain,cost,a):
    assert len(gain)==16 and len(cost)==3 and a is not None and math.isfinite(a) and a>=1.
    assert all(0<x<=1e9 for x in cost) and all(len(r)==3 and all(math.isfinite(g) and abs(g)<=1 for g in r) for r in gain)
    out=[[[g,float(c*a)] for g,c in zip(r,cost)] for r in gain]
    assert all(v[1]<=1e9 for r in out for v in r)
    return out
def costs(row):
    s=row['steps'];assert [x['visits'] for x in s]==[4,8,16,32]
    return [sum(x['wall_us'] for x in s[1:j+1]) for j in (1,2,3)]
def score(run,d):
    return max(c/x for r in run['rows'] for c,x in zip(costs(r),d))
def available(run,table,budget):
    return sorted({j for r in run['rows'] for j,(g,c) in zip((8,16,32),table[r['cell']]) if g>0 and math.ceil(c/50)<=budget//50})
