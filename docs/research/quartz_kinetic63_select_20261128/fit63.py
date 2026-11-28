from plan63 import D,T,read,raw,sha
from fractions import Fraction as F
from itertools import product
from collections import defaultdict,Counter
import gzip,json,math
def main():
    pl=read(D/'PLAN.json')
    with gzip.open(T/'DEV_AUDIT.json.gz','rt') as f:aud=json.load(f)
    rows=aud['rows'];assert len(rows)==430 and aud['lock_sha256']==sha(T/'DEV_LOCK.json')
    cs=defaultdict(list);families=defaultdict(set)
    for r in rows:
        j=pl['roots'][r['id']];c=j['cell'];p=r['pair'];y=p['score2'];q=r['arms']
        assert c==r['cell'];families[c].add(j['family_id'])
        row=[]
        for arm,k in [('H4',0),('FREE',p['free_extra']),('TARGET',p['target_extra'])]:
            gain=F(y[arm]-y['H4'],2);assert 0<=k<=4 and (arm=='TARGET' or k==0)
            row.append(dict(gain=gain,k=F(k),utility=gain-F(k,500),whole=gain-F(q[arm]['Qnn']-q['H4']['Qnn'],500),switch=F(p['moves'][arm]!=p['a']),harm=F(gain<0),help=F(gain>0),residual=F(p['oracle2']-y[arm],2)))
        assert row[0]['residual']-row[1]['residual']==row[1]['gain'] and row[0]['residual']-row[2]['residual']==row[2]['gain']
        cs[c].append(row)
    N=sum(pl['population']);means=[];cells=[]
    for i,Nc in enumerate(pl['population']):
        n=len(cs[i]);assert n==min(80,Nc)
        m=[{k:sum((r[a][k] for r in cs[i]),F())/n for k in cs[i][0][a]} for a in range(3)]
        means.append(m);cells.append(dict(cell=i,N=Nc,n=n,families=len(families[i]),actions=[{k:str(v) for k,v in row.items()} for row in m]))
    def weighted(t,key):return sum((F(Nc,N)*means[i][a][key] for i,(Nc,a) in enumerate(zip(pl['population'],t))),F())
    scores=[];best=None
    for t in product(range(3),repeat=6):
        u=weighted(t,'utility');k=weighted(t,'k');ok=k<=F(3,2) and all(a==0 or len(families[i])>=8 for i,a in enumerate(t))
        scores.append(dict(table=t,utility=str(u),extra=str(k),feasible=ok))
        if ok:
            key=(u,-k,tuple(-x for x in t))
            if best is None or key>best[0]:best=(key,t)
    assert len(scores)==729 and best
    table=best[1];policies={'H4':(0,)*6,'FREE':(1,)*6,'TARGET':(2,)*6,'SELECT':table}
    diag={}
    for name,t in policies.items():
        d={}
        for key in means[0][0]:
            mu=weighted(t,key);var=F()
            for i,Nc in enumerate(pl['population']):
                a=t[i];vals=[r[a][key] for r in cs[i]];n=len(vals);m=means[i][a][key]
                var+=F(Nc,N)**2*(1-F(n,Nc))*sum((v-m)**2 for v in vals)/(n*(n-1))
            d[key]=dict(exact=str(mu),mean=float(mu),design_se=math.sqrt(float(var)))
        diag[name]=d
    out=dict(table=table,cells=cells,enumeration=scores,feasible=sum(x['feasible'] for x in scores),population=N,policies=diag,fit_lock=sha(T/'DEV_LOCK.json'),fit_audit=sha(T/'DEV_AUDIT.json.gz'),price='1/500',budget='3/2',selection_scope='development H4 occupancy only; post-selection optimism')
    (T/'TABLE.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print('table',table,'feasible',out['feasible'],'local gain/extra/utility',[(x,[diag[x][k]['mean'] for k in ('gain','k','utility')]) for x in diag],flush=True)
if __name__=='__main__':main()
