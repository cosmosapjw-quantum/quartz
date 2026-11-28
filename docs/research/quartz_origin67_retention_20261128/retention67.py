"""Readout of the actual frozen native search. No new NN calls."""
import math
DVAL=(-1.,-.5,0.,.5,1.)
def q(e,bd):
    n,w,pi=e;l,u=bd
    return float(l) if l==u else max(l,min(u,w/n)) if n else None
def features(r,op):
    s=r['allocation65'];bd={m:(l,u) for m,l,u in r['partial']['bounds']};safe=s['safe']
    post={m:(n,w,p) for m,n,w,p in s['edges']}
    if op=='VOTE':
        pre=post;a=s['chosen'];b=s['fallback']
    else:
        z=r['operation66'];pre={m:(n,w,p) for m,n,w,p in z['prefix_edges']};a=z['prefix_move'];b=s['chosen']
    assert a in safe and b in safe and set(post)==set(pre)==set(bd)
    qa,qb=q(post[a],bd[a]),q(post[b],bd[b]);pa,pb=q(pre[a],bd[a]),q(pre[b],bd[b])
    za,zb=qa if qa is not None else 0.,qb if qb is not None else 0.
    x=[1.,float(op=='VOTE'),float(op=='FOCUS'),(zb-za)/2,za,(post[b][0]-post[a][0])/5,post[b][2]-post[a][2],
       (qb-pb)/2 if qb is not None and pb is not None else 0.,(qa-pa)/2 if qa is not None and pa is not None else 0.,
       int(pb is None),int(pa is None),(len(bd)-len(safe))/len(bd),len(r['history'])/49,int(qb is None),int(qa is None)]
    assert len(x)==15 and all(math.isfinite(v) and -1.000000000001<=v<=1.000000000001 for v in x)
    return a,b,x
def expected(p):
    # Symmetric differences avoid floating tie ambiguity for exactly symmetric probabilities.
    return (p[4]-p[0])+.5*(p[3]-p[1])
def choose(a,b,p):
    return b if a!=b and expected(p)>0 else a
def predict(model,x):
    z=[(model['offset'][j]+math.fsum(w*v for w,v in zip(model['coef'][j],x)))/model['temperature'] for j in range(5)]
    m=max(z);p=[math.exp(v-m) for v in z];t=math.fsum(p);return [v/t for v in p]
def apply(r,op,model):
    if 'allocation65' not in r:return r['move'],None
    a,b,x=features(r,op)
    if a==b:return a,dict(a=a,b=b,chosen=a,op=op,prob=None,gain=None)
    p=predict(model,x);m=choose(a,b,p)
    return m,dict(a=a,b=b,chosen=m,op=op,prob=p,gain=expected(p))
