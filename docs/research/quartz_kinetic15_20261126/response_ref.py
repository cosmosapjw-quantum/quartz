"""Independent exact forced/free counts, root allocation and native observation check."""
from fractions import Fraction as F
import math,collections as C
import kinetic_ref as kr
import monitor_ref as mr
ARMS=('K15N','K15S','K15X','K15V','K15B','K15F4','K15FH','K15XH','K15STOP','K15LOCK')
CAPS=(32,96)
def choose(n):return max(range(len(n)),key=lambda i:(n[i],i))
def panel(r):
    n=kr.counts(r);ca=[x[0] for x in r['candidates']];l=choose(n)
    z=[i for i,v in enumerate(n) if i!=l and v>0]
    rival=max(z,key=lambda i:(n[i],i)) if z else next((i for i in range(len(n)) if i!=l),None)
    hidden=len(r['edges']) if len(r['edges'])<len(n) else None
    return l,rival,hidden
def distance(n):
    l=choose(n)
    return min((n[l]-x+int(i<l) for i,x in enumerate(n) if i!=l),default=None)
def verify(a):
    arm=a['arm'];cap=a['cap'];assert arm in ARMS and cap in CAPS
    if a['cause']=='GUARDED':assert a['exposure'] is None;return
    e=a['exposure'];p=e['prefix'];r=p['result'];v=a['root_eval']['value']
    assert r['root_n']==8 and r['candidates']==a['initial']['candidates'] and e['raw_best']==a['best']
    n=kr.counts(r);ca=[x[0] for x in r['candidates']]
    l,rv,h=panel(r);mv=lambda j:None if j is None else ca[j]
    assert (e['leader'],e['rival'],e['hidden'])==(mv(l),mv(rv),mv(h))
    target=rv if arm in ('K15F4','K15FH') else h if arm=='K15XH' else None
    high=16 if cap==32 else 56;dose=4 if arm=='K15F4' else high
    assert e['target_rank']==target and e['target']==mv(target) and e['dose']==(dose if target is not None else 0)
    m=mr.observe(r,p['counters'],v)
    for k,x in p['monitor'].items():mr.eq(x,m[k])
    for x,y in zip(kr.feature(r,m,v,None,[0]*6,0,0),p['features'],strict=True):mr.vec(x,y)
    trace=e['b_trace'];assert len(trace)==a['visits']
    expected=C.Counter(ca[t] for t in []);forced=C.Counter()
    # Every visited root action is counted once, including the imposed slot.
    for i,t in enumerate(trace):
        q=t['path'][0];assert q in ca;expected[q]+=1
        if target is not None and 8<=i<8+dose:
            assert q==ca[target];forced[q]+=1
        assert t['mode']==(3 if arm=='K15B' and 8<=i<8+high else 5)
    assert [forced[x] for x in ca]==e['forced']
    cur=8;cc=p['counters']
    for z in e['phases']:
        rr=z['result'];c=z['counters'];nn=kr.counts(rr)
        assert rr['candidates']==r['candidates'] and rr['root_hash']==r['root_hash'] and z['n']==rr['root_n']>cur
        assert rr['evaluator_calls']==1+c['base_evals']+c['extra_evals']
        assert all(x>=y for x,y in zip(nn,n,strict=True))
        mr.vec(z['flux'],kr.flow(n,nn,[x[1] for x in r['candidates']]))
        m=mr.observe(rr,c,v)
        for k,x in z['monitor'].items():mr.eq(x,m[k])
        for k,active in [('select_nodes','K15S'),('width_mass_added','K15X'),('extra_evals','K15V'),('proof_scans','K15V'),('merge_blends','K15B')]:
            assert c[k]>=cc[k]
            if arm!=active or cur>8:
                # Forced root materialization is not X's recorded mass widening.
                assert c[k]==cc[k],(arm,k,cur,c[k],cc[k])
        assert abs(sum(x['w'] for x in rr['edges'])-rr['root_w'])<1e-6
        if arm=='K15LOCK':assert z['n']-cur==min(8,cap-cur) and (distance(n) is not None and distance(n)<=cap-cur)
        n,r,cc,cur=nn,rr,c,z['n']
    assert r==a['result'] and cur==a['visits'] and n==[expected[x] for x in ca]
    if arm=='K15STOP':assert cur==8 and not e['phases'] and not e['locked']
    elif arm=='K15LOCK':
        assert (cur<cap)==e['locked']
        if cur<cap:assert distance(n) is None or distance(n)>cap-cur
    else:
        assert cur==cap and len(e['phases'])==2 and e['phases'][0]['n']==8+dose and not e['locked']
    sub=[x-y for x,y in zip(n,e['forced'],strict=True)];assert min(sub)>=0 and sum(sub)>0
    w=F(sum(e['forced']),cur);tv=sum(abs(F(x,cur)-F(y,sum(sub))) for x,y in zip(n,sub))/2
    assert tv<=w and (e['raw_best'],e['sub_best'])==(ca[choose(n)],ca[choose(sub)])
    mr.eq(e['forced_fraction'],float(w));mr.eq(e['injection_tv'],float(tv))
    assert e['min_d']==distance(n)
    assert sum(t['blends'] for t in trace)==cc['merge_blends'] and sum(len(t['path']) for t in trace)==cc['ledger_observations']
