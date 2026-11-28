"""Independent scalar fit/temperature/label and all-root CAL utility audit."""
from plan67 import D,T,P,ROOT,read,raw,sha,family
from audit67 import independent_features
from collections import Counter,defaultdict
from fractions import Fraction as F
import math,json,gzip
def main():
    p=read(D/'PLAN.json');model=read(D/'MODEL.json');fit=read(T/'FIT.json');lock=read(T/'DEV_LOCK.json')
    assert fit['model_sha256']==sha(D/'MODEL.json') and model['dev_lock']==sha(T/'DEV_LOCK.json') and model['feature_sha256']==sha(D/'retention67.py')
    with gzip.open(T/'DATA.json.gz','rt') as f:rows=json.load(f)
    lookup={(r['source'],r['root'],r['op']):r for r in rows};assert len(lookup)==len(rows)
    exclusions={(r['source'],r['id']) for r in fit['excluded']}
    checked=set();cal=defaultdict(Counter)
    def pred(x,temp):
        z=[(o+sum(u*v for u,v in zip(w,x)))/temp for o,w in zip(model['offset'],model['coef'])]
        e=[math.exp(a-max(z)) for a in z];return [a/sum(e) for a in e]
    def inspect(j,source,op,r,ys):
        a,b,x=independent_features(r,op);assert a==j['incumbent']
        k=(source,j['id'],op)
        if a==b or (source,j['id']) in exclusions:
            assert k not in lookup;return
        z=lookup[k];assert (a,b)==(z['a'],z['b']) and ys[b]-ys[a]==z['y'] and max(abs(v-u) for v,u in zip(x,z['x']))<1e-12
        assert z['board']==list(family(j['history'])) and z['family']==list(family(j['source_opening']))
        checked.add(k)
    for m in read(P/'DEV_LOCK.json')['files']:
        f=P/m['file'];assert sha(f)==m['sha256']
        for z in raw(f):
            j=z['job'];ys={int(m):g['score2'] for m,g in z['forced'].items()}
            for op,r in {'VOTE':z['forced'][str(j['incumbent'])]['rows'][0],**{op:g['rows'][0] for op,g in z['operations'].items()}}.items():inspect(j,'ATLAS',op,r,ys)
    for m in lock['files']:
        f=T/m['file'];assert sha(f)==m['sha256']
        for z in raw(f):
            j=z['job'];gs=z['games'];ys={g['rows'][0]['played']:g['score2'] for g in gs.values()};a=j['incumbent'];base=gs['MA4']
            for op in ('VOTE','FOCUS','EXPAND'):
                g=gs[op];r=g['rows'][0];inspect(j,'CENSUS',op,r,ys)
                if j['split']!='CAL':continue
                a0,b,x=independent_features(r,op);assert a0==a
                pp=pred(x,model['temperature']);est=(pp[4]-pp[0])+.5*(pp[3]-pp[1]);sw=a!=b and est>0
                y=ys[b]-ys[a];extra=r['actual_nn']-base['rows'][0]['actual_nn']
                cb=sum(q['actual_nn'] for q in base['rows'] if q['arm']=='Q')
                co=sum(q['actual_nn'] for q in g['rows'] if q['arm']=='Q')
                # Retaining the anchor after a paid computation means the same base suffix, plus the actual root cost.
                c=cal[j['domain'],op];c['roots']+=1;c['discord']+=a!=b;c['switch']+=sw;c['help']+=sw and y>0;c['harm']+=sw and y<0;c['always_help']+=y>0;c['always_harm']+=y<0
                c['gain2']+=y*sw;c['always_gain2']+=y;c['oracle_gain2']+=max(y,0);c['root_extra']+=extra
                c['whole_extra']+=(co-cb) if sw else extra;c['always_whole_extra']+=co-cb
    assert checked==set(lookup)
    def weighted(rr):
        sums=Counter()
        for r in rr:sums[r['source'],r['domain']]+=r['w']
        return [(r,r['w']/sums[r['source'],r['domain']]/len(sums)) for r in rr]
    tr=[r for r in rows if r['split']=='TRAIN'];cr=[r for r in rows if r['split']=='CAL'];assert len(tr)==model['train_pairs'] and len(cr)==model['cal_pairs']
    assert not ({tuple(r['board']) for r in tr}&{tuple(r['board']) for r in cr})
    assert not ({tuple(r['family']) for r in tr}&{tuple(r['family']) for r in cr})
    prior=[.5]*5
    for r,w in weighted(tr):prior[r['y']+2]+=w*len(tr)
    prior=[v/(len(tr)+2.5) for v in prior];assert max(abs(x-y) for x,y in zip(prior,model['prior']))<1e-12
    grad=[[.02*v for v in w] for w in model['coef']];loss=.01*sum(v*v for w in model['coef'] for v in w)
    for r,w in weighted(tr):
        pr=pred(r['x'],1);y=r['y']+2;loss-=w*math.log(pr[y])
        for k in range(5):
            for j,x in enumerate(r['x']):grad[k][j]+=w*(pr[k]-int(k==y))*x
    norm=math.sqrt(sum(v*v for q in grad for v in q));assert norm<1.1e-9 and abs(loss-fit['objective'])<1e-12
    temperatures=[]
    for t,z0 in fit['temperature_nll']:
        z=-sum(w*math.log(pred(r['x'],t)[r['y']+2]) for r,w in weighted(cr))
        assert abs(z-z0)<1e-12;temperatures.append((t,z))
    assert min(temperatures,key=lambda z:(z[1],abs(z[0]-1),z[0]))[0]==model['temperature']
    for key,m in fit['metrics'].items():
        sp,dom,op=key.split(':');rr=[r for r in rows if r['split']==sp and (dom=='ALL' or r['domain']==dom) and (op=='ALL' or r['op']==op)]
        ns=Counter();scores=Counter()
        for r,w in weighted(rr):
            pr=pred(r['x'],model['temperature']);y=r['y'];g=(pr[4]-pr[0])+.5*(pr[3]-pr[1]);sw=g>0
            scores['nll']-=w*math.log(pr[y+2]);scores['brier']+=w*sum((pr[k]-int(k==y+2))**2 for k in range(5));scores['signed_switch_gain']+=w*y/2*sw;scores['switch_fraction']+=w*sw
            ns['switch_count']+=sw
        for k,v in scores.items():assert abs(v-m[k])<1e-12,(key,k,v,m[k])
        assert ns['switch_count']==m['switch_count']
    out={}
    for (d,op),c in cal.items():
        N=c['roots'];out[d+':'+op]=dict(counts=dict(c),gain=c['gain2']/(2*N),always_gain=c['always_gain2']/(2*N),oracle_gain=c['oracle_gain2']/(2*N),
          root_utility=(c['gain2']/2-c['root_extra']/500)/N,whole_utility=(c['gain2']/2-c['whole_extra']/500)/N,always_whole_utility=(c['always_gain2']/2-c['always_whole_extra']/500)/N)
    new={family(j['opening']) for j in p['jobs']};assert len(new)==60
    for l in p['prior_locks']:
        f=ROOT/l['path'];assert sha(f)==l['sha256']
        old={family(j['opening']) for j in read(f).get('jobs',[]) if isinstance(j.get('opening'),list) and len(j['opening'])==4 and all(type(a)==int and 0<=a<49 for a in j['opening'])}
        assert not old&new
    result=dict(checked_pairs=len(checked),train_gradient_norm=norm,objective=loss,temperature_nll=temperatures,cal_all_roots=out,opening_families=len(new),prior_locks=len(p['prior_locks']),model_sha256=sha(D/'MODEL.json'))
    (T/'INDEPENDENT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print('INDEPENDENT',len(checked),norm,loss,flush=True)
    for k,v in out.items():print(k,v)
if __name__=='__main__':main()
