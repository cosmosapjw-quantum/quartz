from plan64 import D,T,read,raw,sha
from fractions import Fraction as F
from itertools import product
from collections import defaultdict,Counter
import gzip,json,math
DOM=('MAIN','ARCHIVE');OPS=('FREE','T2','T4')
def main():
    pl=read(D/'PLAN.json')
    with gzip.open(T/'DEV_AUDIT.json.gz','rt') as f:aud=json.load(f)
    rows=aud['rows'];assert len(rows)==554 and aud['lock_sha256']==sha(T/'DEV_LOCK.json')
    cs=defaultdict(list);fam=defaultdict(set);cert=Counter();rawcnt=Counter()
    for r in rows:
        j=pl['roots'][r['id']];d,c=j['domain'],j['cell'];p=r['pair'];y=p['score2'];q=r['arms'];f=p['free'];a,b=p['a'],p['b']
        assert c==r['cell'] and d==r['domain'];fam[d,c].add(j['family_id'])
        es={i:(n,w) for i,n,w,pi in j['snapshot']['edges']};bd={i:(l,u) for i,l,u in j['bounds']}
        row=[]
        for op,arm in enumerate(OPS):
            k=p['extra'][arm];gain=F(y[arm]-y['FREE'],2);assert 0<=k<=2*op
            row.append(dict(gain=gain,k=F(k),utility=gain-F(k,500),whole=gain-F(q[arm]['Qnn']-q['FREE']['Qnn'],500),switch=F(p['moves'][arm]!=f),harm=F(gain<0),help=F(gain>0),residual=F(p['oracle2']-y[arm],2)))
            assert row[op]['residual']>=0 and row[0]['residual']-row[op]['residual']==gain
            if op:
                def iv(m):
                    n,w=es[m];l,u=bd[m];return (max(l,min(u,(w-op)/(n+op))),max(l,min(u,(w+op)/(n+op)))) if l!=u else (l,u)
                x,z=iv(a),iv(b);same=(f==a and z[1]<=x[0]) or (f==b and z[0]>x[1])
                if same:assert p['moves'][arm]==f
                cert[d,op]+=int(same);rawcnt[d,arm,'help']+=int(gain>0);rawcnt[d,arm,'harm']+=int(gain<0);rawcnt[d,arm,'switch']+=int(p['moves'][arm]!=f);rawcnt[d,arm,'same_score']+=int(gain==0)
        cs[d,c].append(row)
    means={};cells=[]
    for d in DOM:
        for i,Nc in enumerate(pl['population'][d]):
            n=len(cs[d,i]);assert n==(min(80,Nc) if d=='MAIN' else Nc)
            m=[{k:sum((r[a][k] for r in cs[d,i]),F())/n for k in cs[d,i][0][a]} for a in range(3)]
            means[d,i]=m;cells.append(dict(domain=d,cell=i,N=Nc,n=n,families=len(fam[d,i]),actions=[{k:str(v) for k,v in z.items()} for z in m]))
    def weighted(d,t,key):
        pop=pl['population'][d];N=sum(pop)
        return sum((F(Nc,N)*means[d,i][a][key] for i,(Nc,a) in enumerate(zip(pop,t))),F())
    scores=[];best=None;pooled=None
    for t in product(range(3),repeat=6):
        u=[weighted(d,t,'utility') for d in DOM];k=[weighted(d,t,'k') for d in DOM]
        ok=all(x<=F(3,2) for x in k) and all(a==0 or all(len(fam[d,i])>=8 for d in DOM) for i,a in enumerate(t))
        scores.append(dict(table=t,utility=[str(x) for x in u],extra=[str(x) for x in k],feasible=ok))
        if ok:
            key=(min(u),-sum(k),tuple(-x for x in t))
            pk=(sum(u),-sum(k),tuple(-x for x in t))
            if best is None or key>best[0]:best=(key,t)
            if pooled is None or pk>pooled[0]:pooled=(pk,t)
    assert len(scores)==729 and best and best[0][0]>=0
    table=best[1];policies={'FREE':(0,)*6,'T2':(1,)*6,'T4':(2,)*6,'ROBUST':table,'POOLED_DIAGNOSTIC':pooled[1]}
    diag={}
    for name,t in policies.items():
        diag[name]={}
        for d in DOM:
            pop=pl['population'][d];N=sum(pop);z={}
            for key in means[d,0][0]:
                mu=weighted(d,t,key);var=F()
                for i,Nc in enumerate(pop):
                    a=t[i];vals=[r[a][key] for r in cs[d,i]];n=len(vals);m=means[d,i][a][key]
                    if n>1:var+=F(Nc,N)**2*(1-F(n,Nc))*sum((v-m)**2 for v in vals)/(n*(n-1))
                    else:assert n==Nc
                z[key]=dict(exact=str(mu),mean=float(mu),design_se=math.sqrt(float(var)))
            diag[name][d]=z
    out=dict(table=table,pooled_table=pooled[1],cells=cells,enumeration=scores,feasible=sum(x['feasible'] for x in scores),population=pl['population'],policies=diag,certificates={f'{d}:{o}':v for (d,o),v in cert.items()},raw_counts={':'.join(k):v for k,v in rawcnt.items()},fit_lock=sha(T/'DEV_LOCK.json'),fit_audit=sha(T/'DEV_AUDIT.json.gz'),price='1/500',budget='3/2',scope='finite two-domain occurrence populations; post-selection optimism; novel checkpoint panel excluded')
    (T/'TABLE.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print('table',table,'pooled',pooled[1],'feasible',out['feasible'],'local gain/extra/utility',[(x,d,[diag[x][d][k]['mean'] for k in ('gain','k','utility')]) for x in diag for d in DOM],flush=True)
if __name__=='__main__':main()
