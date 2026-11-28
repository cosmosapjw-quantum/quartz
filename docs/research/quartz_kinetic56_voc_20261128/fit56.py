"""Fit physical cost and paired score contrast; freeze once before any fresh outcome."""
import collections as C,gzip,json,math
import numpy as np
from k56_plan import D,load,save,sha
PEN=16.
def features(r):
    l=r['low'];v=dict(l['votes']);n=l['visits'];bd=r['partial']['bounds']
    assert r['bin'] in (3,7) and n==sum(v.values()) and len(v)==len(bd)
    return [1.,r['ply']/49,l['distance']/3,n/20,sum(k*k for k in v.values())/(n*n) if n else 0.,v[l['prior_best']]/n if n else 0.,sum(u==-1 for m,lo,u in bd)/len(bd),float(l['replacement'])]
def records(cycle):
    d=D.parent/('quartz_kinetic55_policy_20261128' if cycle==55 else 'quartz_kinetic54_cache_20261127')
    lk=json.loads((d/'NN_LOCK.json').read_text())
    for fi in lk['files']:
        p=d/fi['file'];assert sha(p)==fi['sha256']
        with gzip.open(p,'rt') as f:
            for z in map(json.loads,f):
                g=z['control'];e=g['event']
                if e is None:continue
                arms=['cold','reuse'] if cycle==55 else ['cold']
                r=dict(job=g['job']['id'],seed=g['seed'],color=g['color'],x=features(g['rows'][e]),g=[0.],k=[0.])
                for a in arms:
                    t=z[a];assert t['event']==e and t['rows'][e]['low']==g['rows'][e]['low']
                    r['g'].append((t['score2']-g['score2'])/2)
                    c=t['rows'][e]['cache_receipt'];r['k'].append(c['physical']-c['root_physical'])
                yield r
def predict(m,x):
    g=[0.]+[max(-1.,min(1.,sum(a*b for a,b in zip(row,x)))) for row in m['gain']]
    k=[0.]+[max(0.,min(12.,sum(a*b for a,b in zip(row,x)))) for row in m['cost']]
    return g,k
def choose(m,x):
    g,k=predict(m,x)
    if m['stop']:return 0,g,k
    u=[g[i]-m['lambda']*k[i] for i in range(3)]
    return max(range(3),key=lambda i:(u[i],-i)),g,k
def summary(rs,m,means):
    ct=C.Counter();err=C.Counter()
    for r in rs:
        o,g,k=choose(m,r['x']);ct['n']+=1;ct['op'+str(o)]+=1
        if o<len(r['g']):
            ct['gain2']+=round(2*r['g'][o]);ct['cost']+=r['k'][o]
        else:ct['unobserved_choice']+=1
        ct['cold_cost']+=r['k'][1];ct['cold_gain2']+=round(2*r['g'][1])
        for i in range(1,len(r['g'])):
            for v,est,base in [('gain',g,means['gain']),('cost',k,means['cost'])]:
                y=r['g' if v=='gain' else 'k'][i]
                err[v+str(i)+'_sq']+=(y-est[i])**2
                err[v+str(i)+'_const']+=(y-base[i-1])**2
    return dict(counts=dict(ct),MSE={key:v/len(rs) for key,v in err.items()})
def main():
    a=list(records(55));tr=[r for r in a if r['job']<48];ca=[r for r in a if r['job']>=48];ex=list(records(54))
    X=np.array([r['x'] for r in tr]);G=np.array([r['g'][1:] for r in tr]);K=np.array([r['k'][1:] for r in tr])
    A=X.T@X+PEN*np.diag([0]+[1]*7);Bg=np.linalg.solve(A,X.T@G);Bk=np.linalg.solve(A,X.T@K)
    assert np.max(np.abs(A@Bg-X.T@G))<1e-10 and np.max(np.abs(A@Bk-X.T@K))<1e-9
    m=dict(gain=Bg.T.tolist(),cost=Bk.T.tolist(),stop=False,**{'lambda':0.})
    pred=[predict(m,r['x']) for r in tr];cuts={0.}
    for g,k in pred:
        for i in range(3):
            for j in range(i):
                if k[i]!=k[j]:
                    z=(g[i]-g[j])/(k[i]-k[j])
                    if math.isfinite(z) and z>=0:cuts.add(z)
    cuts=sorted(cuts);ls=cuts+[(a+b)/2 for a,b in zip(cuts,cuts[1:])]+[cuts[-1]+1.]
    budget=.5*sum(k[1] for g,k in pred);feas=[dict(stop=True,**{'lambda':0.},gain=0.,cost=0.,ops=[len(tr),0,0])]
    for l in ls:
        o=[max(range(3),key=lambda i:(g[i]-l*k[i],-i)) for g,k in pred]
        g=sum(g[i] for i,(g,k) in zip(o,pred));k=sum(k[i] for i,(g,k) in zip(o,pred))
        if k<=budget+1e-12:feas.append(dict(stop=False,**{'lambda':l},gain=g,cost=k,ops=[o.count(i) for i in range(3)]))
    best=min(feas,key=lambda z:(-z['gain'],z['cost'],-z['lambda'],not z['stop']))
    m.update(stop=best['stop'],**{'lambda':best['lambda']})
    means=dict(gain=G.mean(0).tolist(),cost=K.mean(0).tolist());tr0=summary(tr,m,means);ca0=summary(ca,m,means);ex0=summary(ex,m,means)
    c=ca0['counts'];calpass=c.get('gain2',0)>0 and 2*c.get('cost',0)<=c['cold_cost']
    out=dict(model=m,fit=best,budget=budget,breakpoints=len(cuts),feasible=len(feas),train=tr0,cal=ca0,external_cold=ex0,cal_admitted=calpass,means=means,ridge=PEN,condition_number=float(np.linalg.cond(A)),train_rows=len(tr),cal_rows=len(ca),external_rows=len(ex),history_locks={str(i):sha(D.parent/('quartz_kinetic55_policy_20261128' if i==55 else 'quartz_kinetic54_cache_20261127')/'NN_LOCK.json') for i in (54,55)},scope='fit only K55 family0..47, historical CAL48..63, K54 cold diagnostic (REUSE choice unobserved); fresh diagnostic even if CAL fails')
    save('FROZEN.json',out);print('FIT',json.dumps(out),flush=True)
if __name__=='__main__':main()
