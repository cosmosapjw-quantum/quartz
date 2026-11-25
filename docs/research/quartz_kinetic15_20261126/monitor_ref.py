"""Independent double precision reference for the computational observables."""
import math
EPS=1e-8
def norm(x):
    z=math.fsum(x);return [v/z for v in x]
def kl(x,y):return math.fsum(a*math.log(a/b) for a,b in zip(x,y,strict=True))
def observe(r,c,v):
    ca=r['candidates'];es=r['edges'];assert [e['move'] for e in es]==[x[0] for x in ca[:len(es)]]
    ns=[e['n'] for e in es]+[0]*(len(ca)-len(es))
    ws=[e['w'] for e in es]+[0.]*(len(ca)-len(es))
    pi=norm([x[1]+EPS for x in ca]);u=[(w+4*v)/(n+4) for n,w in zip(ns,ws)]
    z=[p*math.exp(4*x) for p,x in zip(pi,u)];q=norm(z);rv=norm([n+EPS for n in ns]);mid=[(a+b)/2 for a,b in zip(q,rv)]
    return dict(n=ns,u=u,q=q,phi=math.log(math.fsum(z))/4,uncertainty=math.fsum(p/(n+4) for p,n in zip(q,ns)),js=(kl(q,mid)+kl(rv,mid))/2,hidden=math.fsum(q[len(es):]),shared=c['ledger_shared_children'],kl_prior=kl(q,pi),entropy=-math.fsum(p*math.log(p) for p in q))
def proposal(m,mem):
    g=m['shared']/(1+m['shared'])
    omega=norm([1,.25+2*m['js']/math.log(2),.25+2*m['hidden'],.25+4*m['uncertainty'],.25+2*g])
    return omega,norm([o*math.exp(8*v) for o,v in zip(omega,mem)])
def response(a,b,nn):
    assert all(x<=y for x,y in zip(a['n'],b['n'],strict=True))
    work=math.fsum(p*(v-u) for p,u,v in zip(a['q'],a['u'],b['u']))
    reorg=kl(a['q'],b['q'])/4
    deplete=math.fsum(q*(1/(n+4)-1/(m+4)) for q,n,m in zip(a['q'],a['n'],b['n']))
    transport=math.fsum((p-q)/(n+4) for q,p,n in zip(a['q'],b['q'],b['n']))
    d=b['uncertainty']-a['uncertainty'];dp=b['phi']-a['phi']
    assert abs(dp-work-reorg)<1e-10 and abs(d+deplete-transport)<1e-10 and deplete>=-1e-12
    return dict(work=work,reorganization=reorg,delta_phi=dp,depletion=deplete,transport=transport,delta_a=d,y=-4*d/(1+nn/16),nn=nn)
def uniform(seed,i):
    mask=2**64-1
    z=(seed+(i+1)*0x9e3779b97f4a7c15)&mask
    z=((z^(z>>30))*0xbf58476d1ce4e5b9)&mask
    z=((z^(z>>27))*0x94d049bb133111eb)&mask
    z^=z>>31
    return (z>>11)/2**53
def select(z,u):
    c=0.
    for i,p in enumerate(z):
        c+=p
        if u<c:return i
    return 4
def eq(a,b):
    assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-10),(a,b)
def vec(a,b):
    for x,y in zip(a,b,strict=True):eq(x,y)
def verify(a,seed,schedule=None):
    if a['cause']=='GUARDED':assert a['meta'] is None;return
    arm=a['arm'];x=a['meta'];v=a['root_eval']['value']
    assert x['mode']==arm and x['beta']==x['kappa']==4 and x['eta']==8
    assert a['cause']=='META_FIXED' and a['visits']==256 and len(x['steps'])==15
    pre=x['prefix'];r=pre['result'];cc=pre['counters']
    assert r['root_n']==16 and r['candidates']==a['initial']['candidates']
    m=observe(r,cc,v)
    for k,y in pre['monitor'].items():eq(y,m[k])
    mem=[0.]*5;seen=0
    names=('select_nodes','width_nodes','extra_evals','proof_scans','merge_blends')
    assert all(cc[k]==0 for k in names)
    for i,t in enumerate(x['steps']):
        r1=t['result'];c=t['counters'];z=observe(r1,c,v)
        assert r1['root_n']==32+i*16 and r1['candidates']==r['candidates']
        assert r1['root_hash']==r['root_hash'] and sum(e['n'] for e in r1['edges'])==r1['root_n']
        eq(sum(e['w'] for e in r1['edges']),r1['root_w'])
        for e in r1['edges']:
            n=e['n'];w=e['w'];M=e['m2']
            assert type(n) is int and n>=0 and abs(w)<=n+1e-6 and -1e-7<=M<=(n-w*w/n if n else 0)+1e-5
            assert e['vl']==0 and abs(e['vvalue'])<1e-6
        for k,y in t['monitor'].items():eq(y,z[k])
        o,p=proposal(m,mem if arm=='META' else [0.]*5);vec(o,t['omega']);vec(p,t['proposal'])
        u=uniform(seed,i);assert u==t['uniform']
        ch=0 if arm=='MBASE' else schedule[i] if arm=='YOKE' else select(p,u)
        assert ch==t['channel']
        vec(mem,t['memory_before'])
        nn=r1['evaluator_calls']-r['evaluator_calls'];y=response(m,z,nn)
        for k,a0 in y.items():eq(a0,t['response'][k])
        eq(t['prediction_error'],y['y']-mem[ch]);mem[ch]=(mem[ch]+y['y'])/2;vec(mem,t['memory_after'])
        for k in names:
            assert c[k]>=cc[k]
            if k in ('select_nodes',) and ch!=1 or k in ('width_nodes',) and ch!=2 or k in ('extra_evals','proof_scans') and ch!=3 or k=='merge_blends' and ch!=4:assert c[k]==cc[k],(arm,i,ch,k)
        assert c['ledger_observations']>=cc['ledger_observations']
        assert r1['evaluator_calls']==1+c['base_evals']+c['extra_evals']
        m,r,cc=z,r1,c;seen+=1
    assert r==a['result'] and r['evaluator_calls']==a['NN_calls']
    tr=x['b_trace'];assert len(tr)==256
    modes=[5]*16
    for t in x['steps']:modes += [3 if t['channel']==4 else 5]*16
    assert [t['mode'] for t in tr]==modes and sum(t['blends'] for t in tr)==cc['merge_blends']
    assert all(t['mode']==3 or t['blends']==0 for t in tr)
    assert sum(len(t['path']) for t in tr)==cc['ledger_observations']
