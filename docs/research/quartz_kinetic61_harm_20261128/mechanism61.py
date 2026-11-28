"""Cost/trajectory accounting on common histories, plus design-based training diagnostics."""
from plan61 import D,T,read,raw,sha
from collections import Counter,defaultdict
from fractions import Fraction as F
import json,math
def norm(r):
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if not k.endswith('_ns')}
        if isinstance(x,list):return [clean(v) for v in x]
        return x
    return clean({k:v for k,v in r.items() if k not in ('early','root_nn','low','atlas','mask','physical_nn','native_physical_nn')})
def main():
    pl=read(D/'PLAN.json');tab=read(D/'TABLE.json')
    train=read(D/'TRAIN_LOCK.json');samples=defaultdict(list)
    for meta in train['files']:
        p=D/meta['file'];assert sha(p)==meta['sha256']
        for z in raw(p):
            i=z['job']['cell']
            for h in (8,0):
                a=z['pairs'][str(h)];g=F(a['gain2'],2);k=F(a['root_cost'])
                samples[i,h].append((g,max(g,0)-F(1,2)*max(-g,0)-k/500,g-k/500,F(g>0)))
    diag={}
    N=sum(pl['population'])
    for h in (8,0):
        ds=[]
        for j,name in enumerate(('gain','asym_utility','signed_utility','harm_probability')):
            mean=F();var=F();cells=[]
            for i,nc in enumerate(pl['population']):
                a=[r[j] for r in samples[i,h]];n=len(a);m=sum(a,F())/n
                s2=sum((z-m)**2 for z in a)/(n-1)
                vc=(1-F(n,nc))*s2/n
                mean+=F(nc,N)*m;var+=F(nc,N)**2*vc
                cells.append(dict(cell=i,mean=float(m),design_se=math.sqrt(float(vc))))
            ds.append(dict(quantity=name,weighted_mean=float(mean),design_se=math.sqrt(float(var)),cells=cells))
        diag[str(h)]=ds
    lock=read(T/'ARENA_LOCK.json');assert lock['table']==sha(D/'TABLE.json')
    out={}
    for arm in ('SELECT','CONTRAST'):
        ct=Counter();stops=Counter();cell_counts=Counter()
        for meta in lock['files']:
            p=T/meta['file'];assert sha(p)==meta['sha256']
            for gs in raw(p):
                a,b=gs[arm],gs['H4'];ct['episodes']+=1
                ct['equal_history']+=a['history']==b['history'];ct['delta_score2']+=a['score2']-b['score2']
                ct['delta_total_nn']+=a['actual_nn']-b['actual_nn']
                for r in a['rows']:
                    if 'early' not in r:continue
                    e=r['early']
                    for s in e['snapshots']:
                        if s['visits']==4 and s['cell61'] is not None:
                            ext=s['extend61'];cell_counts[f'{s["cell61"]}:{"extend" if ext else "stop"}']+=1
                            ct['eligible4']+=1;ct['extension']+=ext
                            if ext:ct['extended_root_physical']+=r['actual_nn']-s['physical']
                    if e['reached']:stops[str(e['at']['visits'])]+=1
                    else:stops['fallback']+=1
                div=False
                for x,y in zip(a['rows'],b['rows']):
                    assert x['history']==y['history'];ct['common_roots']+=1
                    ext=any(s.get('extend61',False) for s in x.get('early',{}).get('snapshots',[]))
                    if ext:
                        assert x['arm']=='Q' and x['actual_nn']>=y['actual_nn']
                        ct['common_extensions']+=1;ct['local_nn_extra']+=x['actual_nn']-y['actual_nn']
                    else:assert norm(x)==norm(y)
                    if x['move']!=y['move']:
                        assert ext;ct['first_different_move']+=1;div=True;break
                if not div:assert a['history']==b['history']
        assert ct['episodes']==672 and ct['equal_history']+ct['first_different_move']==672
        out[arm]=dict(counts=dict(ct),stop_hist=dict(stops),cell_counts=dict(cell_counts))
    x=dict(arms=out,finite_population_diagnostics=diag,lock_sha256=sha(T/'ARENA_LOCK.json'),scope='finite frozen occurrence SRS design SE, no generalization or new gate; common prefix cost not causal downstream attribution')
    (T/'MECHANISM.json').write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');print(json.dumps(x),flush=True)
if __name__=='__main__':main()
