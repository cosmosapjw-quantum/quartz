"""Finite signed gain distribution, fixed eight pseudo-observations."""
import math
def shrink(rows,alpha=8):
    assert rows and alpha==8
    counts=[0]*5; bins=[[0]*5 for _ in range(16)]; cost=[0.]*16; n=[0]*16
    for c,d,t in rows:
        assert 0<=c<16 and d in range(-2,3) and math.isfinite(t) and t>0
        counts[d+2]+=1; bins[c][d+2]+=1; n[c]+=1; cost[c]+=t
    glob=[x/len(rows) for x in counts]; tc=sum(cost)/len(rows)
    probs=[[(bins[c][k]+alpha*glob[k])/(n[c]+alpha) for k in range(5)] for c in range(16)]
    gc=[[sum(p*(k-2)/2 for k,p in enumerate(probs[c])),(cost[c]+alpha*tc)/(n[c]+alpha)] for c in range(16)]
    return dict(global_=glob,**{'global':glob},global_cost=tc,cells=probs,gain_cost=gc,counts=n)
