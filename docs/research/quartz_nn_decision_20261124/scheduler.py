"""One optional continuation per resident root; fit only on train signed WDL gain."""
import math
def signed_order(scores, ids):
    assert len(scores)==len(ids) and all(math.isfinite(float(s)) for s in scores)
    return sorted(range(len(scores)),key=lambda i:(-scores[i],ids[i]))
def features(row,legal):
    entries=row['policy']
    pairs=[tuple(v.split(':',1)) if isinstance(v,str) else tuple(v) for v in entries]
    actions=[int(a) for a,p in pairs];v=[float(p) for a,p in pairs]
    assert len(set(actions))==len(actions) and set(actions)<=set(legal)
    assert v and all(math.isfinite(x) and x>=0 for x in v) and sum(v)>0
    p=sorted((x/sum(v) for x in v),reverse=True)
    entropy=-sum(x*math.log(x) for x in p if x>0)/math.log(len(legal)) if len(legal)>1 else 0
    return [entropy,p[0],p[0]-(p[1] if len(p)>1 else 0),len(legal)/49.]
def fit(xs,ys):
    import numpy as np
    x=np.array(xs,dtype=float);y=np.array(ys,dtype=float)
    mean=x.mean(axis=0); std=x.std(axis=0);std[std==0]=1
    z=(x-mean)/std
    beta=np.linalg.solve(z.T@z+np.eye(z.shape[1]),z.T@(y-y.mean()))
    return dict(mean=mean.tolist(),std=std.tolist(),beta=beta.tolist(),intercept=float(y.mean()))
def score(model,x):
    return sum(b*(v-m)/s for b,v,m,s in zip(model['beta'],x,model['mean'],model['std']))+model['intercept']
