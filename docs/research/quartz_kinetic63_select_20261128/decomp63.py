from plan63 import D,T,read,raw,sha
from collections import Counter,defaultdict
from fractions import Fraction as F
import json
def main():
    p=read(D/'PLAN.json');lock=read(T/'DEV_LOCK.json');tab=read(D/'TABLE.json');N=sum(p['population'])
    sums=defaultdict(Counter);weights=Counter();count=Counter();seen=set()
    for meta in lock['files']:
        f=T/meta['file'];assert sha(f)==meta['sha256']
        for z in raw(f):
            j=z['job'];assert j['id'] not in seen;seen.add(j['id']);r=z['pair'];m=r['moves'];y=r['score2'];a,b=r['a'],r['b'];c=j['cell'];w=F(j['population'],N*j['sample'])
            assert m['FREE'] in (a,b) and m['TARGET'] in (a,b)
            k=r['target_extra'];g=F(y['TARGET']-y['FREE'],2);same=m['TARGET']==m['FREE']
            assert not same or g==0
            row=dict(n=1,agree=int(same),disagree=int(not same),target_improves_free=int(g>0),target_harms_free=int(g<0),target_ties_free=int(g==0),free_help=int(y['FREE']>y['H4']),free_harm=int(y['FREE']<y['H4']),target_help=int(y['TARGET']>y['H4']),target_harm=int(y['TARGET']<y['H4']),free_target_delta=g,extra=k,incremental_net=g-F(k,500),better_rival=int(y['FORCE_B']>y['H4']),worse_rival=int(y['FORCE_B']<y['H4']))
            assert row['target_improves_free']+row['target_harms_free']+row['target_ties_free']==1
            for key,v in row.items():sums[c][key]+=v;count[key]+=v;weights[key]+=w*v
    assert len(seen)==430 and weights['n']==1
    assert weights['free_target_delta']==F(tab['policies']['TARGET']['gain']['exact'])-F(tab['policies']['FREE']['gain']['exact'])
    out=dict(raw={k:float(v) for k,v in count.items()},weighted={k:dict(mean=float(v),exact=str(v)) for k,v in weights.items()},cells={c:{k:float(v) for k,v in x.items()} for c,x in sums.items()},lock_sha256=sha(T/'DEV_LOCK.json'),scope='fixed H4 occurrence population; estimates not future occupancy')
    (T/'DECOMP.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
