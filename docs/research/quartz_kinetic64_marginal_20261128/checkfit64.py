from plan64 import D,T,read,raw,sha
from fractions import Fraction as F
from collections import Counter,defaultdict
import itertools,math,json
DOM=('MAIN','ARCHIVE')
def main():
    pl=read(D/'PLAN.json');lock=read(T/'DEV_LOCK.json');tab=read(T/'TABLE.json');assert tab['fit_lock']==sha(T/'DEV_LOCK.json')
    sums=defaultdict(Counter);families=defaultdict(set);ids=set()
    for m in lock['files']:
        f=T/m['file'];assert sha(f)==m['sha256']
        for z in raw(f):
            j=z['job'];assert j==pl['roots'][j['id']] and j['id'] not in ids;ids.add(j['id']);d,c=j['domain'],j['cell'];families[d,c].add(j['family_id'])
            gs=z['games'];base=gs['FREE'];y=base['score2'];k=base['rows'][0]['actual_nn']
            for a,arm in enumerate(('FREE','T2','T4')):
                g=gs[arm];delta=g['score2']-y;cost=g['rows'][0]['actual_nn']-k
                sums[d,c,a].update(n=1,utility500=250*delta-cost,cost=cost,gain2=delta)
    assert len(ids)==554
    den={}
    for d in DOM:
        pop=pl['population'][d];L=math.lcm(*(min(80,n) if d=='MAIN' else n for n in pop));den[d]=sum(pop)*L
    candidates=[]
    for t in itertools.product(range(3),repeat=6):
        us=[];ks=[];ok=True
        for d in DOM:
            pop=pl['population'][d];L=den[d]//sum(pop);u=k=0
            for i,Nc in enumerate(pop):
                a=t[i];s=sums[d,i,a];n=s['n'];assert n==(min(80,Nc) if d=='MAIN' else Nc)
                w=Nc*(L//n);u+=w*s['utility500'];k+=w*s['cost'];ok=ok and (a==0 or len(families[d,i])>=8)
            ok=ok and 2*k<=3*den[d];us.append(F(u,den[d]*500));ks.append(F(k,den[d]))
        candidates.append((us,ks,t,ok))
    for x,z in zip(candidates,tab['enumeration']):
        u,k,t,ok=x;assert list(t)==z['table'] and u==list(map(F,z['utility'])) and k==list(map(F,z['extra'])) and ok==z['feasible']
    u,k,t,ok=min((x for x in candidates if x[3]),key=lambda x:(-min(x[0]),sum(x[1]),x[2]))
    u2,k2,t2,_=min((x for x in candidates if x[3]),key=lambda x:(-sum(x[0]),sum(x[1]),x[2]))
    assert list(t)==tab['table'] and list(t2)==tab['pooled_table'] and sum(x[3] for x in candidates)==tab['feasible']
    out=dict(candidates=729,feasible=tab['feasible'],table=list(t),utility=list(map(str,u)),extra=list(map(str,k)),pooled_table=list(t2),table_sha256=sha(T/'TABLE.json'))
    (T/'FIT_CHECK.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(out,flush=True)
if __name__=='__main__':main()
