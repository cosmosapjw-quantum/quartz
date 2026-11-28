"""Independent integer family-weighted table recheck, using audited raw games."""
from plan60 import D,T,HS,read,raw,sha
from collections import Counter,defaultdict
from fractions import Fraction
import json
def main():
    p=read(D/'PLAN.json');lock=read(D/'TRAIN_LOCK.json');tab=read(D/'TABLE.json');acc=defaultdict(Counter);rows=Counter()
    assert tab['train_lock_sha256']==sha(D/'TRAIN_LOCK.json')
    for meta in lock['files']:
        path=D/meta['file'];assert sha(path)==meta['sha256']
        for z in raw(path):
            j=z['job'];f=j['family_id'];full=z['games']['0']
            for h in HS:
                g=z['games'][str(h)];s=g['rows'][0]['early']
                if not s['reached']:continue
                x=s['at'];d=x['distance'];i=4*HS.index(h)
                # integer gap comparison, proof-safe prior agreement
                if 2*d>=20-h:i+=2
                if x['move']==x['prior_best']:i+=1
                root_cost=full['rows'][0]['actual_nn']-g['rows'][0]['actual_nn']
                g2=full['score2']-g['score2']
                acc[i,f].update(n=1,u500=g2*250-root_cost,g2=g2,k=root_cost,down2=max(g2,0),harm=int(g2>0),flip=int(full['rows'][0]['move']!=g['rows'][0]['move']))
                rows[i]+=1
    out=[]
    for i in range(16):
        fs=[v for (j,_),v in acc.items() if j==i];n=len(fs)
        net=sum((Fraction(v['u500'],500*v['n']) for v in fs),Fraction())/n if n else None
        stop=bool(n>=8 and net is not None and net<=0)
        t=tab['cells'][i];assert t['n']==rows[i] and t['families']==n and t['net_exact']==(str(net) if net is not None else None) and t['stop']==tab['table'][i]==stop
        out.append(dict(cell=i,families=n,n=rows[i],net=str(net) if net is not None else None,stop=stop))
    # Exact first-checkpoint reachability bound for h=4 and >=2 legal actions:
    assert 2*(4+1)<20-4 and tab['table'][0:2]==[True,True] and rows[2]==rows[3]==0
    r=dict(cells=out,source_table=sha(D/'TABLE.json'),checked_cells=16,h4_possible_cells=[0,1],h4_all_stop=True)
    (T/'FIT_CHECK.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print(json.dumps(r))
if __name__=='__main__':main()
