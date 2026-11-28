from plan61 import D,T,sha,read,raw
from fractions import Fraction as F
from collections import Counter,defaultdict
import json
def main():
    lock=read(D/'TRAIN_LOCK.json');table=read(D/'TABLE.json');p=read(D/'PLAN.json')
    acc=defaultdict(Counter);fs=defaultdict(set);seen=set()
    for m in lock['files']:
        f=D/m['file'];assert sha(f)==m['sha256']
        for z in raw(f):
            j=z['job'];i=j['cell'];assert j['id'] not in seen;seen.add(j['id']);fs[i].add(j['family_id'])
            for h in (8,0):
                base=z['games']['4'];g=z['games'][str(h)]
                g2=g['score2']-base['score2'];k=g['rows'][0]['actual_nn']-base['rows'][0]['actual_nn']
                # Integer score numerator on 1/2000 lattice: (g+ - .5*g- - k/500).
                acc[i,h].update(n=1,u2000=1000*max(g2,0)-500*max(-g2,0)-4*k,g2=g2,k=k,harm=int(g2>0),benefit_stop=int(g2<0))
    assert len(seen)==500
    out={}
    for h in (8,0):
        pop=Counter();total=sum(p['population'])
        for i,N in enumerate(p['population']):
            c=acc[i,h];n=c['n'];assert n==100
            v=F(c['u2000'],2000*n);row=next(r for r in table['rows'] if r['cell']==i and r['h']==h)
            assert row['asym_net_exact']==str(v) and row['extend']==(len(fs[i])>=8 and v>0)
            assert row['counts']['g2']==c['g2'] and row['counts']['harm']==c['harm']
            for k in ('g2','harm','benefit_stop','k'):pop[k]+=F(N,n)*c[k]
        for k in pop:
            expected=pop[k]/total/(2 if k=='g2' else 1);assert abs(float(expected)-table['population'][str(h)][k])<1e-12
        out[str(h)]={k:str(v) for k,v in pop.items()}
    print('independent exact cell and HT population check: 10 cells and 8 weighted quantities',out)
    (T/'FIT_CHECK.json').write_text(json.dumps(dict(checked_cells=10,weighted=out,table_sha256=sha(D/'TABLE.json')),indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
