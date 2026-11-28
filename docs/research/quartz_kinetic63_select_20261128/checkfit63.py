from plan63 import D,T,read,raw,sha
from fractions import Fraction as F
from collections import Counter,defaultdict
import itertools,math,json
def main():
    pl=read(D/'PLAN.json');lock=read(T/'DEV_LOCK.json');tab=read(T/'TABLE.json');assert tab['fit_lock']==sha(T/'DEV_LOCK.json')
    sums=defaultdict(Counter);families=defaultdict(set);ids=set()
    for m in lock['files']:
        f=T/m['file'];assert sha(f)==m['sha256']
        for z in raw(f):
            j=z['job'];assert j==pl['roots'][j['id']] and j['id'] not in ids;ids.add(j['id']);c=j['cell'];families[c].add(j['family_id'])
            gs=z['games'];base=gs['H4'];y=base['score2'];k=base['rows'][0]['actual_nn']
            for a,arm in enumerate(('H4','FREE','TARGET')):
                g=gs[arm];delta=g['score2']-y;cost=g['rows'][0]['actual_nn']-k
                sums[c,a].update(n=1,utility500=250*delta-cost,cost=cost,gain2=delta)
    assert len(ids)==430;N=sum(pl['population']);L=math.lcm(*(min(80,n) for n in pl['population']));den=N*L
    candidates=[]
    for t in itertools.product(range(3),repeat=6):
        u=k=0;ok=True
        for i,Nc in enumerate(pl['population']):
            a=t[i];s=sums[i,a];n=s['n'];assert n==min(80,Nc)
            w=Nc*(L//n);u+=w*s['utility500'];k+=w*s['cost'];ok=ok and (a==0 or len(families[i])>=8)
        ok=ok and 2*k<=3*den;candidates.append((u,k,t,ok))
    for x,z in zip(candidates,tab['enumeration']):
        u,k,t,ok=x;assert list(t)==z['table'] and F(u,den*500)==F(z['utility']) and F(k,den)==F(z['extra']) and ok==z['feasible']
    u,k,t,ok=min((x for x in candidates if x[3]),key=lambda x:(-x[0],x[1],x[2]))
    assert list(t)==tab['table'] and sum(x[3] for x in candidates)==tab['feasible']
    out=dict(candidates=729,feasible=tab['feasible'],table=list(t),utility=str(F(u,den*500)),extra=str(F(k,den)),table_sha256=sha(T/'TABLE.json'))
    (T/'FIT_CHECK.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(out,flush=True)
if __name__=='__main__':main()
