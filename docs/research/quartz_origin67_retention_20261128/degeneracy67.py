"""Exact sign bounds on observed feature boxes and actual policy trajectory/cost equivalence."""
from plan67 import D,T,read,sha,raw
from audit67 import independent_features
from fractions import Fraction as F
from collections import Counter,defaultdict
import json,gzip,math
def exp_bounds(x):
    if x<0:
        l,u=exp_bounds(-x);return 1/u,1/l
    assert x<8
    s=t=F(1)
    for n in range(1,41):t*=x/n;s+=t
    return s,s+(t*x/41)/(1-x/42)
def main():
    m=read(D/'MODEL.json');coef=[[F(x) for x in q] for q in m['coef']];off=[F(x) for x in m['offset']];temp=F(m['temperature'])
    with gzip.open(T/'DATA.json.gz','rt') as f:rows=json.load(f)
    xs=defaultdict(list)
    for r in rows:xs[r['op']].append(r['x'])
    panels={}
    for panel in ('MAIN','ARCHIVE','NOVEL','TRANSFER'):
        lk=read(T/(panel+'_LOCK.json'));ct={op:Counter() for op in ('VOTE','FOCUS','EXPAND')}
        for z in lk['files']:
            f=T/z['file'];assert sha(f)==z['sha256']
            for gs in raw(f):
                for arm,op,ref in [('CALV','VOTE','MA4'),('CALF','FOCUS','MA4'),('CALX','EXPAND','EXPAND')]:
                    g,b=gs[arm],gs[ref];c=ct[op];c['episodes']+=1;c['same_history']+=g['history']==b['history'];c['same_score']+=g['score2']==b['score2']
                    cost=0
                    for r in g['rows']:
                        if 'operation66' in r:cost+=r['operation66']['extra_physical']
                        if 'retention67' not in r:continue
                        a0,b0,x=independent_features(r,op);v=r['retention67'];assert (a0,b0)==(v['a'],v['b'])
                        if a0==b0:continue
                        xs[op].append(x);c['discord']+=1;c['switched']+=v['chosen']!=a0
                    c['operation_cost']+=cost;c['both_physical_delta']+=g['actual_nn']-b['actual_nn']
                    if g['history']==b['history']:
                        assert g['score2']==b['score2'];expected=cost if op=='FOCUS' else 0
                        assert g['actual_nn']-b['actual_nn']==expected
                        for x,y in zip(g['rows'],b['rows']):
                            assert x['history']==y['history'] and x['played']==y['played']
                            assert x['actual_nn']-y['actual_nn']==(x.get('operation66',{}).get('extra_physical',0) if op=='FOCUS' else 0)
        panels[panel]={k:dict(v) for k,v in ct.items()}
    cert={}
    for op,xx in xs.items():
        box=[(min(F(x[i]) for x in xx),max(F(x[i]) for x in xx)) for i in range(15)]
        def interval(j,k):
            lo=hi=(off[j]-off[k])/temp
            for a,b,(l,u) in zip(coef[j],coef[k],box):
                t=(a-b)/temp;lo+=min(t*l,t*u);hi+=max(t*l,t*u)
            return lo,hi
        e={}
        for j in (1,3,4):
            lo,hi=interval(j,0);e[j]=(exp_bounds(lo)[0],exp_bounds(hi)[1])
        lo=e[4][0]-1+(e[3][0]-e[1][1])/2;hi=e[4][1]-1+(e[3][1]-e[1][0])/2
        cert[op]=dict(n=len(xx),feature_box=[[float(a),float(b)] for a,b in box],normalized_numerator_interval=[float(lo),float(hi)],sign='positive' if lo>0 else 'negative' if hi<0 else 'unresolved',exact_rational_exp_tail='40 terms, geometric upper bound, exp(-x) reciprocal')
    out=dict(scope='post-fit observed hyperrectangle, not all legal future roots; real-arithmetic decoded model coefficients, runtime replay separately',model_sha256=sha(D/'MODEL.json'),panels=panels,observed_box=cert)
    (T/'DEGENERACY.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
