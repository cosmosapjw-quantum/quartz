"""Independent occupation identity, feature vector and native decision check."""
import math
import monitor_ref as mr
def H(p):return -math.fsum(x*math.log(x) for x in p if x)
def kl(p,q):return math.fsum(x*math.log(x/y) for x,y in zip(p,q,strict=True) if x)
def tv(p,q):return math.fsum(abs(x-y) for x,y in zip(p,q,strict=True))/2
def probs(n):
    z=sum(n);assert z>0;return [v/z for v in n]
def counts(r):
    idx={m:i for i,(m,p) in enumerate(r['candidates'])};n=[0]*len(idx)
    for e in r['edges']:n[idx[e['move']]]=e['n']
    assert sum(n)==r['root_n'];return n
def flow(n,m,prior):
    inc=[b-a for a,b in zip(n,m,strict=True)];assert min(inc)>=0
    a,b,j=map(probs,(n,m,inc));g=sum(inc)/sum(m)
    pi=mr.norm([v+1e-8 for v in prior])
    mix=H(b)-(1-g)*H(a)-g*H(j)
    delta=kl(b,pi)-kl(a,pi)
    assert abs(delta-g*(kl(j,pi)-kl(a,pi))+mix)<1e-10
    assert abs(tv(a,b)-g*tv(a,j))<1e-10
    assert all(abs(x-(1-g)*y-g*z)<1e-10 for x,y,z in zip(b,a,j))
    assert -1e-10<=mix<=H([g,1-g])+1e-10
    return [g,tv(a,j),tv(a,b),mix,delta,H(b)-H(a)]
def feature(r,mon,v,first,f,dp,da):
    n=counts(r);p=mr.norm([x[1]+1e-8 for x in r['candidates']]);o=probs(n)
    L=max(range(len(n)),key=lambda i:(n[i],i));assert r['best']==r['candidates'][L][0]
    d=min((n[L]-n[i]+int(i<L) for i in range(len(n)) if i!=L),default=n[L])
    logm=max(1.,math.log(len(n)));scale=max(1.,-math.log(min(p)));m=mon['shared']
    common=[1,v,o[L],d/sum(n),sum(p[len(r['edges']):]),len(r['edges'])/len(n),m/(1+m),H(p)/logm]+[int(first==i) for i in range(5)]
    g=[mon['phi'],4*mon['uncertainty'],mon['js']/math.log(2),mon['hidden'],mon['kl_prior']/scale,mon['entropy']/logm,dp,4*da]
    k=[kl(o,p)/scale,H(o)/logm,tv(o,p),f[1],f[3]/math.log(2),f[4]/scale,f[5]/logm,f[2]]
    return [common+[0]*8,common+g,common+k]
def choose(model,stage,x):
    g=[math.fsum(a*b for a,b in zip(x,v,strict=True)) for v in model[stage+'_gain']]
    c=[math.fsum(a*b for a,b in zip(x,v,strict=True)) for v in model[stage+'_cost']]
    s=[a-.002*b for a,b in zip(g,c)];best=0
    for i in range(1,5):
        if s[i]>s[best]+1e-12:best=i
    return best,g,c,s
def verify(a,schedule=None,model=None):
    if a['cause']=='GUARDED':assert a['kinetic'] is None;return
    x=a['kinetic'];assert a['cause']=='KINETIC_FIXED' and a['visits']==96 and x['price']==.002
    pre=x['prefix'];r=pre['result'];cc=pre['counters'];v=a['root_eval']['value'];m=mr.observe(r,cc,v)
    assert r['root_n']==16 and r['candidates']==a['initial']['candidates']
    for k,y in pre['monitor'].items():mr.eq(y,m[k])
    prior=[t[1] for t in r['candidates']]
    f=feature(r,m,v,None,[0]*6,0,0)
    for z,y in zip(f,pre['features'],strict=True):mr.vec(z,y)
    tr=x['b_trace'];assert len(tr)==96;first=None;expected=[5]*16
    for i,(t,quantum,N) in enumerate(zip(x['steps'],(32,32,16),(48,80,96),strict=True)):
        if i==2:ch=0
        elif a['arm']=='KINETIC':ch=schedule[i]
        else:
            kind=('RAW','GIBBS','FLUX').index(model['kind'])
            ch,g,c,s=choose(model,'first' if i==0 else 'second',f[kind]);d=t['decision']
            assert d['kind']==kind;mr.vec(g,d['gain']);mr.vec(c,d['cost_ms']);mr.vec(s,d['net'])
        assert t['channel']==ch and t['quantum']==quantum
        if i==0:first=ch
        if i==2 or a['arm']=='KINETIC':assert t['decision'] is None
        r1=t['result'];c=t['counters'];m1=mr.observe(r1,c,v)
        assert r1['root_n']==N and r1['candidates']==r['candidates'] and r1['root_hash']==r['root_hash']
        assert r1['evaluator_calls']==1+c['base_evals']+c['extra_evals']
        for k,y in t['monitor'].items():mr.eq(y,m1[k])
        z=flow(counts(r),counts(r1),prior);mr.vec(z,t['flux'])
        nn=r1['evaluator_calls']-r['evaluator_calls'];res=mr.response(m,m1,nn)
        assert nn==t['nn'];mr.eq(res['delta_phi'],t['delta_phi']);mr.eq(res['delta_a'],t['delta_a'])
        f=feature(r1,m1,v,first,z,res['delta_phi'],res['delta_a'])
        for b,c0 in zip(f,t['features'],strict=True):mr.vec(b,c0)
        for k,active in [('select_nodes',1),('width_nodes',2),('extra_evals',3),('proof_scans',3),('merge_blends',4)]:
            assert c[k]>=cc[k]
            if ch!=active:assert c[k]==cc[k]
        expected += [3 if ch==4 else 5]*quantum
        mr.eq(sum(e['w'] for e in r1['edges']),r1['root_w'])
        r,m,cc=r1,m1,c
    assert r==a['result'] and [t['mode'] for t in tr]==expected
    assert sum(t['blends'] for t in tr)==cc['merge_blends'] and sum(len(t['path']) for t in tr)==cc['ledger_observations']
