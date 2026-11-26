"""Post-lock frontier vs long native: no model fit or policy selection."""
from trial import *
ll=labelmap('TEST');rootrows={}
for t in rows('TEST_ATLAS'):
    i=t['root']['id']
    if i in ll:rootrows[i]=t['results']
assert set(rootrows)==set(ll)
ids=sorted(ll)
def ci(v):
    rng=random.Random(SEED+310);b=sorted(S.mean(rng.choices(v,k=len(v))) for _ in range(4000))
    return [b[99],b[3899]]
out={}
for a,b in [(7,0),(8,0),(8,7)]:
    dy=[];dn=[];dr=[];dc=[];dt=[];net0=[];net1=[];cert0=cert1=0
    for i in ids:
        u=ll[i];z=rootrows[i];x=[z[f'{a}:{j}'] for j in (0,1)];y=[z[f'{b}:{j}'] for j in (0,1)]
        dy.append(S.mean((u[p['best']]-u[q['best']])/2 for p,q in zip(x,y)))
        dn.append(S.mean(p['NN_calls']-q['NN_calls'] for p,q in zip(x,y)))
        dr.append(S.mean(p['rule_work']-q['rule_work'] for p,q in zip(x,y)))
        dc.append(S.mean((p['cpu_ns']-q['cpu_ns'])/1e6 for p,q in zip(x,y)))
        dt.append(S.mean((p['wall_ns']-q['wall_ns'])/1e6 for p,q in zip(x,y)))
    result=dict(n=len(ids),utility=S.mean(dy),utility_ci=ci(dy),better=sum(v>0 for v in dy),worse=sum(v<0 for v in dy),NN_delta=S.mean(dn),rules_delta=S.mean(dr),cpu_delta_ms=S.mean(dc),wall_delta_ms=S.mean(dt))
    result['price_sensitivity']={str(l):dict(mean=S.mean(v-.002*k-l*r for v,k,r in zip(dy,dn,dr)),ci=ci([v-.002*k-l*r for v,k,r in zip(dy,dn,dr)])) for l in (0,.00002,.00005,.0001,.0002)}
    if S.mean(dr)>0:result['break_even_rule_price']=(S.mean(dy)-.002*S.mean(dn))/S.mean(dr)
    out[f'{a}:{b}']=result
# Pre-treatment opportunities defined from common graph at midpoint.
opp=[]
for i in ids:
    z=rootrows[i];u=ll[i];a=z['8:0'];b=z['7:0'];c=z['6:0'];pr=a['proof_graph']['before'];assert pr==b['proof_graph']['before']==c['proof_graph']['before']
    for j in (6,7,8):
        x,y=z[f'{j}:0'],z[f'{j}:1'];assert x['best']==y['best']
    opp.append(dict(id=i,initial_cert=pr['witness'] is not None,initial_risk=pr['risk'][0],
        fifo_cert=b['proof_graph']['end']['witness'] is not None,directed_cert=a['proof_graph']['end']['witness'] is not None,
        fifo_gain=(u[b['best']]-u[c['best']])/2,directed_gain=(u[a['best']]-u[c['best']])/2))
out['mechanism']=dict(n=len(opp),initial=sum(x['initial_cert'] for x in opp),fifo=sum(x['fifo_cert'] for x in opp),directed=sum(x['directed_cert'] for x in opp),
    both=sum(x['fifo_cert'] and x['directed_cert'] for x in opp),fifo_only=sum(x['fifo_cert'] and not x['directed_cert'] for x in opp),directed_only=sum(x['directed_cert'] and not x['fifo_cert'] for x in opp),
    fifo_gain=sum(x['fifo_gain'] for x in opp),directed_gain=sum(x['directed_gain'] for x in opp),rows=opp)
out['scope']='Post-lock paired actual graph and native continuations, two executions averaged per root, price grid descriptive, no refit, no P2 noninferiority.'
p=BIN.parent/'FRONTIER_DIAGNOSTIC.json';assert not p.exists();p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:({a:b for a,b in v.items() if a!='rows'} if isinstance(v,dict) else v) for k,v in out.items()},sort_keys=True),flush=True)
