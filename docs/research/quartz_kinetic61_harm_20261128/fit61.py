"""Occurrence-weighted stratified estimation with exact arithmetic."""
from plan61 import D,T,sha,read,raw,feature
from collections import Counter,defaultdict
from fractions import Fraction as F
import json
def main():
    p=read(D/'PLAN.json');lock=read(T/'TRAIN_LOCK.json')
    assert lock['plan']==sha(D/'PLAN.json') and lock['source']==sha(D/'continuation61.py')
    cells=defaultdict(Counter);families=defaultdict(set);seen=set()
    for m in lock['files']:
        path=T/m['file'];assert sha(path)==m['sha256']
        for z in raw(path):
            j=z['job'];assert j==p['roots'][j['id']] and j['id'] not in seen;seen.add(j['id']);i=j['cell']
            assert feature(z['games']['4']['rows'][0]['early']['at'],j['bounds'])==i
            families[i].add(j['family_id'])
            for h in (8,0):
                a=z['games']['4'];b=z['games'][str(h)];r=z['pairs'][str(h)]
                g=b['score2']-a['score2'];k=b['rows'][0]['actual_nn']-a['rows'][0]['actual_nn']
                assert (g,k)==(r['gain2'],r['root_cost']) and k>=0
                cells[i,h].update(n=1,g2=g,gplus2=max(g,0),gminus2=max(-g,0),k=k,harm=int(g>0),benefit_stop=int(g<0),flip=int(r['changed']),qcost=r['qcost'],bcost=r['bcost'],reached=int(r['reached']))
    assert len(seen)==500
    rows=[];table=[];population={}
    for h in (8,0):
        weighted=defaultdict(F)
        for i,N in enumerate(p['population']):
            c=cells[i,h];n=c['n'];assert n==100
            g=F(c['g2'],2*n);gp=F(c['gplus2'],2*n);gm=F(c['gminus2'],2*n);k=F(c['k'],n)
            v=gp-gm/2-k/500;assert g==gp-gm
            extend=len(families[i])>=p['min_family'] and v>0
            row=dict(cell=i,h=h,N=N,n=n,families=len(families[i]),counts=dict(c),gain=float(g),positive=float(gp),negative=float(gm),root_cost=float(k),asym_net=float(v),asym_net_exact=str(v),signed_net=float(g-k/500),extend=extend)
            rows.append(row)
            if h==8:table.append(extend)
            for key,val in c.items():weighted[key]+=F(N,n)*val
        N=sum(p['population']);assert weighted['n']==N
        population[str(h)]={key:float(val/N/(2 if key in ('g2','gplus2','gminus2') else 1)) for key,val in weighted.items() if key!='n'}
    out=dict(table=table,contrast=[True,False,True,False,True],rows=rows,population=population,population_size=sum(p['population']),train_lock_sha256=sha(T/'TRAIN_LOCK.json'),price='1/500',loss_weight='1/2',min_family=p['min_family'])
    (T/'TABLE.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,indent=2,sort_keys=True),flush=True)
if __name__=='__main__':main()
