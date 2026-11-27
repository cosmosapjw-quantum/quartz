"""Finite robust mask selection, fixed model-panel leave-one-out, no outcome tuning."""
import math
H=('32','64','128')
def fit(rs,seeds):
    # Empty policy valid, group denominator ALL roots, not selected roots.
    groups=[rs]+[[r for r in rs if r['seed']!=s] for s in seeds]
    assert all(groups)
    costs=[sum(r['cost'] for r in rs if r['bin']==z) for z in range(8)]
    cap=sum(costs)//4
    sums=[[[math.fsum(r['gains'][h] for r in g if r['bin']==z)/len(g) for z in range(8)] for h in H] for g in groups]
    valid=[]
    for m in range(256):
        if sum(costs[z] for z in range(8) if m>>z&1)>cap:continue
        v=min(math.fsum(row[z] for z in range(8) if m>>z&1) for group in sums for row in group)
        valid.append((v,-m.bit_count(),-m))
    val,_,neg=max(valid);mask=-neg
    return dict(mask=mask,worst_mean=val,budget=cap,selected_cost=sum(costs[z] for z in range(8) if mask>>z&1),total_cost=sum(costs),group_count=len(groups),feasible_masks=len(valid))
def evaluate(rs,mask):
    c=sum(r['cost'] for r in rs);sel=[r for r in rs if mask>>r['bin']&1];sc=sum(r['cost'] for r in sel);p=sc/c if c else 0.
    out={}
    for h in H:
        a=math.fsum(r['gains'][h] for r in rs);b=math.fsum(r['gains'][h] for r in sel)
        out[h]=dict(total_gain=a,selected_gain=b,constant_gain=p*a,excess=b-p*a)
    return dict(n=len(rs),selected=len(sel),cost=c,selected_cost=sc,p=p,horizons=out,admitted=bool(mask) and all(x['selected_gain']>0 and x['excess']>0 for x in out.values()))
