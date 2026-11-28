"""Design-weighted finite continuation regret. Exact Fraction means and finite-population design SE."""
from plan66 import D,T,read,sha,raw
from fractions import Fraction as F
from collections import Counter,defaultdict
import gzip,json,math
with gzip.open(T/'DEV_AUDIT.json.gz','rt') as f:a=json.load(f)
assert a['lock_sha256']==sha(T/'DEV_LOCK.json')
p=read(D/'PLAN.json');rows=a['rows'];assert len(rows)==171
out={}
for domain in ('MAIN','ARCHIVE','NOVEL'):
    rr=[r for r in rows if r['domain']==domain];N=sum(p['population'][domain].values());cells=defaultdict(list)
    for r in rr:cells[r['cell']].append(r)
    def metric(fn):
        total=F();var=F()
        for c,rs in cells.items():
            Nh=p['population'][domain][c];n=len(rs);x=[F(fn(r)) for r in rs];mean=sum(x,F())/n;w=F(Nh,N)
            total+=w*mean
            if n<Nh:
                assert n>1
                var+=w*w*(1-F(n,Nh))*sum((v-mean)**2 for v in x)/(n*(n-1))
        return dict(mean=float(total),exact=str(total),design_se=math.sqrt(float(var)))
    ms={k:metric(lambda r,k=k:F(r[k],2)) for k in ('ma','mf','ml','base','mask','omission','ranking','regret')}
    for k in ('legal','safe','live','base_cost'):ms[k]=metric(lambda r,k=k:r[k])
    for k in ('mask','omission','ranking','regret'):ms[k+'_positive']=metric(lambda r,k=k:int(r[k]>0))
    assert F(ms['regret']['exact'])==sum(F(ms[k]['exact']) for k in ('mask','omission','ranking'))
    ops={}
    for arm in ('FOCUS','EXPAND'):
        def x(r,k):return r['ops'][arm][k]
        z={k:metric(lambda r,k=k:F(x(r,k),2)) for k in ('gain','omission_reduction','ranking_change')}
        for k in ('extra','whole_extra','changed'):z[k]=metric(lambda r,k=k:int(x(r,k)))
        z['root_utility']=metric(lambda r:F(x(r,'gain'),2)-F(x(r,'extra'),500))
        z['whole_utility']=metric(lambda r:F(x(r,'gain'),2)-F(x(r,'whole_extra'),500))
        z['raw']=dict(help=sum(x(r,'gain')>0 for r in rr),harm=sum(x(r,'gain')<0 for r in rr),changed=sum(x(r,'changed') for r in rr),noop=sum(x(r,'target') is None for r in rr),expansion_opportunity=sum(x(r,'omission_reduction')>0 for r in rr),extra=sum(x(r,'extra') for r in rr))
        assert F(z['gain']['exact'])==F(z['omission_reduction']['exact'])-F(z['ranking_change']['exact'])
        ops[arm]=z
    eq=[r for r in rr if r['ops']['FOCUS']['extra']==r['ops']['EXPAND']['extra']==1]
    out[domain]=dict(sample=len(rr),population=N,cell_samples={c:len(rs) for c,rs in cells.items()},metrics=ms,ops=ops,raw={k:sum(r[k]>0 for r in rr) for k in ('mask','omission','ranking','regret')},equal_paid_descriptive=dict(n=len(eq),delta2=sum(r['ops']['EXPAND']['gain']-r['ops']['FOCUS']['gain'] for r in eq)))
(T/'DEV_SUMMARY.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
for d,v in out.items():
 print(d,'n',v['sample'],'N',v['population'],'regret components',[(k,v['metrics'][k]['mean'],v['metrics'][k]['design_se']) for k in ('base','mask','omission','ranking','regret')], 'raw',v['raw'])
 for arm,z in v['ops'].items():print(arm,{k:round(z[k]['mean'],9) for k in ('gain','omission_reduction','ranking_change','extra','root_utility','whole_utility')},z['raw'])
 print('equal paid',v['equal_paid_descriptive'])
