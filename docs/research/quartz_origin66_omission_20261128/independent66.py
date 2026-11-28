"""Independent integer-label aggregation and oracle acquisition diagnosis from actual raw games."""
from plan66 import D,T,ROOT,read,raw,sha,family
from fractions import Fraction as F
from collections import Counter,defaultdict
import json,math
p=read(D/'PLAN.json');lock=read(T/'DEV_LOCK.json');summary=read(T/'DEV_SUMMARY.json')
new={family(j['opening']) for j in p['jobs']};assert len(new)==48
for z in p['prior_locks']:
    path=ROOT/z['path'];assert sha(path)==z['sha256']
    old={family(j['opening']) for j in read(path).get('jobs',[]) if isinstance(j.get('opening'),list) and len(j['opening'])==4 and all(type(x)==int and 0<=x<49 for x in j['opening'])}
    assert not new&old
acc=defaultdict(Counter);den=Counter();ids=set();rawct=defaultdict(Counter)
def add(d,key,v,w):acc[d][key]+=w*v
for m in lock['files']:
    f=T/m['file'];assert sha(f)==m['sha256']
    for z in raw(f):
        j=z['job'];assert j==p['roots'][j['id']] and j['id'] not in ids;ids.add(j['id'])
        d=j['domain'];weight=F(j['population'],j['sample']);den[d]+=weight
        games={int(m):g for m,g in z['forced'].items()};y={m:g['score2'] for m,g in games.items()};bd={m:(l,u) for m,l,u in j['bounds']};assert set(y)==set(bd)
        es={m:(n,w,pr) for m,n,w,pr in j['snapshot']['edges']};safe={m for m in bd if bd[m][1]!=-1} or set(bd)
        live={m for m in safe if es[m][0] or bd[m][0]==bd[m][1]} or {j['incumbent']}
        a=j['incumbent'];assert a in live
        ma=max(y.values());mf=max(y[m] for m in safe);ml=max(y[m] for m in live)
        for k,v in [('base',y[a]),('ma',ma),('mf',mf),('ml',ml),('mask',ma-mf),('omission',mf-ml),('ranking',ml-y[a]),('regret',ma-y[a])]:
            add(d,k,F(v,2),weight)
        # Exact diagnostics at this fixed continuation, not a future-policy oracle.
        hidden=safe-live;loss=set(y)-safe
        add(d,'outside_best_ties',int(bool(hidden) and max(y[m] for m in hidden)==mf),weight)
        add(d,'outside_strictly_better_than_live',int(mf>ml),weight)
        add(d,'proof_loss_win_exists',int(any(y[m]==2 for m in loss)),weight)
        add(d,'proof_loss_strictly_better',int(ma>mf),weight)
        add(d,'best_all_is_win',int(ma==2),weight)
        if hidden:
            top=max(hidden,key=lambda m:(es[m][2],-m))
            add(d,'prior_top_outside_capture',F(max(ml,y[top])-ml,2),weight)
            add(d,'uniform_outside_capture',sum((F(max(ml,y[m])-ml,2) for m in hidden),F())/len(hidden),weight)
            add(d,'outside_mass',F.from_float(sum(es[m][2] for m in hidden)/sum(es[m][2] for m in safe)),weight)
        for arm,g in z['operations'].items():
            r=g['rows'][0];op=r['operation66'];t=op['target'];b=r['played'];live2=live|({t} if t is not None else set());m2=max(y[m] for m in live2)
            assert b in live2 and g['score2']==y[b]
            cost=r['actual_nn']-games[a]['rows'][0]['actual_nn'];assert cost==op['extra_physical']
            gain=F(y[b]-y[a],2);om=F(m2-ml,2);rank=F((m2-y[b])-(ml-y[a]),2)
            add(d,arm+':gain',gain,weight);add(d,arm+':omission_reduction',om,weight);add(d,arm+':ranking_change',rank,weight);add(d,arm+':extra',cost,weight)
            add(d,arm+':root_utility',gain-F(cost,500),weight)
            rawct[d][arm+':help']+=gain>0;rawct[d][arm+':harm']+=gain<0;rawct[d][arm+':changed']+=b!=a
            if t is not None:
                add(d,arm+':target_true_gain',F(y[t]-y[a],2),weight)
                add(d,arm+':target_better',int(y[t]>y[a]),weight)
                add(d,arm+':target_worse',int(y[t]<y[a]),weight)
                if arm=='EXPAND':
                    add(d,'exp_better_not_selected',int(y[t]>y[a] and b!=t),weight)
                    add(d,'exp_worse_selected',int(y[t]<y[a] and b==t),weight)
assert len(ids)==171
out={}
for d,w in den.items():
    assert w==summary[d]['population']
    v={k:x/w for k,x in acc[d].items()}
    for k in ('base','ma','mf','ml','mask','omission','ranking','regret'):
        assert v[k]==F(summary[d]['metrics'][k]['exact'])
    for arm in ('FOCUS','EXPAND'):
        for k in ('gain','omission_reduction','ranking_change','extra','root_utility'):
            assert v[arm+':'+k]==F(summary[d]['ops'][arm][k]['exact'])
        for k in ('help','harm','changed'):assert rawct[d][arm+':'+k]==summary[d]['ops'][arm]['raw'][k]
    out[d]=dict(means={k:float(x) for k,x in v.items()},exact={k:str(x) for k,x in v.items() if k!='outside_mass'},raw=dict(rawct[d]))
r=dict(roots=len(ids),families=len(new),prior_locks=len(p['prior_locks']),populations={k:str(v) for k,v in den.items()},domains=out,lock_sha256=sha(T/'DEV_LOCK.json'),scope='fixed suffix, actual all legal, oracle-only; uniform outside capture averages all hidden labels')
(T/'INDEPENDENT.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
for d,z in out.items():print(d,z['means'],z['raw'])
