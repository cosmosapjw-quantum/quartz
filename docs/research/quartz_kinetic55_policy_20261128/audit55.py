"""Independent integer/coordinate proof and chronological action/cost audit."""
import collections as C,gzip,json,pathlib,sys,importlib.util,subprocess,math
from fractions import Fraction
from k55_plan import D,ROOT,T,SEEDS,load,sha,family
V=D.parent/'quartz_kinetic43_partial_20261127'
sys.path.insert(0,str(V));sp=importlib.util.spec_from_file_location('_independent43',V/'k43_audit.py')
v=importlib.util.module_from_spec(sp);sp.loader.exec_module(v)
CT=C.Counter();EX=C.defaultdict(list)
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def save(n,x):
    p=T/n;assert not p.exists();p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def replay_proof(h,n,z):
    a,b=v.orient(h,n);r=z['cold']['receipt'];assert r['mode']=='F' and r['depth']==7 and r['budget']==4096
    k,full,ls=v.G[n];e=full^(a|b)
    bn=int(any(not l&b and (l&e).bit_count()==2 for l in ls))+2*int(any(not l&a and (l&e).bit_count()==2 for l in ls))
    assert r['bin']==bn and 0<=r['skipped']<=len(r['bounds'])
    bd=v.inspect(h,n,r);cert=r['cert'];ex=z['exact']
    assert (ex is not None)==(n==7 and 33<=len(h)<49 and cert is None)
    if ex:
        CT['exact_attempt']+=1;assert ex['nodes']<=25000
        if ex['complete']:
            x={m:(l,u) for m,l,u in ex['bounds']};val=ex['value'];cert=ex['best'];assert set(x)==set(bd) and x[cert]==(val,val)
            assert all(u<=val and (m>=cert or u<val) for m,(l,u) in x.items())
            for m,(l,u) in x.items():
                p,q=bd[m];bd[m]=(max(p,l),min(q,u));assert bd[m][0]<=bd[m][1]
            CT['exact_complete']+=1
        else:assert ex['nodes']==25000 and 'best' not in ex and 'bounds' not in ex
        EX[a,b].append(z)
    assert z['bounds']==[[m,l,u] for m,(l,u) in bd.items()] and z['cert']==cert
    if cert is not None:assert all(bd[cert][0]>=u for m,(l,u) in bd.items() if m!=cert)
    assert z['wall_ns']>=z['cold']['wall_ns']+(ex['wall_ns'] if ex else 0)
    CT['proof_queries']+=r['queries'];CT['proof_receipts']+=1
    return bd
def vote(h,n,r,lock,cap,pre=None):
    a,b=v.orient(h,n);bd,g=v.base(a,b,n);E=list(bd)
    assert r['nn']>=0 and r['visits']>=0
    if n==7:assert r['cap']==cap
    if g is not None:
        assert r['kind']=='CERTIFICATE' and r['move']==g and r['nn']==r['visits']==0 and not r['votes']
        if n==15:assert r['gate']==g and r['native']['history']==h
        CT['common']+=1;return
    if r['kind']=='PARTIAL47':assert lock and r['nn']==r['visits']==0 and not r['votes'];return
    vs=r['votes'];assert len(vs)==len(E) and {m for m,x in vs}==set(E) and all(type(x) is int and x>=0 for m,x in vs)
    i=max(range(len(vs)),key=lambda i:(vs[i][1],i if n==15 else -vs[i][0]));lead,nv=vs[i]
    assert (r['move'] if pre is None else pre)==lead and sum(x for m,x in vs)==r['visits']<=cap
    d=min((nv-x+int(j<i if n==15 else m>lead) for j,(m,x) in enumerate(vs) if j!=i),default=10**9)
    if n==7:
        assert [m for m,x in vs]==E and r['nn']==1+r['visits']-r['terminal_leaves'] and r['distance']==d
        if r['kind']=='VOTE_LOCK':assert lock and d>cap-r['visits'] and r['visits']<cap
        else:assert r['kind']=='CAP' and r['visits']==cap
    else:
        x=r['native'];assert x['history']==h and x['best']==lead and x['NN_calls']==r['nn'] and x['visits']==r['visits'] and x['cap']==cap and x['locked']==lock
        assert r['nn']<=cap and r['certificate_checks']>0
        if r['kind']=='VOTE_LOCK':assert lock and d>cap-r['visits'] and r['visits']<cap and r['nn']<cap
        elif r['kind']=='VISIT_CAP':assert r['visits']==cap
        else:assert r['kind']=='NN_TARGET' and r['nn']==cap
    CT['votes']+=1;CT['edges']+=len(vs)
def check_snapshot(h,r,bd,cap):
    vote(h,7,r,True,cap,r['original_move']);mv=v.changed(r['original_move'],r['votes'],bd,7);assert mv==r['move'] and r['replacement']==(mv!=r['original_move'])
def check_target(h,r,bd):
    t=r['target'];a=r['low']['move'];assert t['incumbent']==a
    votes=dict(r['low']['votes']);alts=[m for m in votes if m!=a and bd[m][1]!=-1]
    b=sorted(alts,key=lambda m:(-votes[m],m))[0] if alts else None
    if b is not None:
        assert bd[a][1]!=-1
        if bd[a][0]>=bd[b][1]:b=None
    assert t['rival']==b
    if b is None:
        assert t['reason']=='no_nondominated_rival' and t['chosen']==a and not t['children'] and not t['values'] and t['extra_nn']==0
    else:
        assert t['reason']=='compared' and len(t['children'])==2
        vals=[];cost=0
        for m,x in zip((a,b),t['children']):
            l,u=bd[m]
            if l==u:
                assert x is None;vals.append(float(l));continue
            assert x is not None;aa,bb,w=v.replay(h+[m],7);term=bool(w) or aa|bb==v.G[7][1]
            assert term==x['terminal']
            if term:
                assert x['nn']==x['visits']==x['terminal_leaves']==0 and not x['samples']
                assert x['raw']==(1. if w else 0.)
            else:
                assert x['visits']==5 and 0<=x['terminal_leaves']<=5 and x['nn']==6-x['terminal_leaves']
                assert len(x['samples'])==5 and all(-1<=q<=1 for q in x['samples'])
                assert [e[0] for e in x['edges']]==[m0 for m0 in range(49) if not (aa|bb)>>m0&1]
                assert sum(e[1] for e in x['edges'])==5 and all(e[1]>=0 and abs(e[2])<=e[1]+1e-12 for e in x['edges'])
                assert abs(math.fsum(e[2] for e in x['edges'])-math.fsum(x['samples']))<1e-12
                assert abs(x['raw']+math.fsum(x['samples'])/5)<1e-15
            vals.append(max(l,min(u,x['raw'])));cost+=x['nn']
            CT['child_receipts']+=1
        assert vals==t['values'] and t['extra_nn']==cost<=12
        assert t['chosen']==(b if vals[1]>vals[0] else a)
        CT['compared']+=1
    assert t['wall_ns']>=0 and r['move']==t['chosen'] and r['nn']==r['native_physical_nn']==r['physical_nn']+t['extra_nn']
    CT['target_events']+=1;CT['target_local_nn']+=t['extra_nn'];CT['target_switch']+=t['chosen']!=a
    return dict(r,move=a,nn=r['physical_nn'])
def wave_audit(r,qcap,q):
    if r['kind'] in ('CERTIFICATE','PARTIAL47'):return
    x=r['native'];cap=r['selected_cap'];assert x['cap']==cap and x['locked']==q
    def dist(vs):
        i=max(range(len(vs)),key=lambda j:(vs[j][1],j))
        return min((vs[i][1]-n+int(j<i) for j,(m,n) in enumerate(vs) if j!=i),default=None)
    if q and qcap=='AD':
        assert x['requested_cap'] is None and x['horizons']==[128]*4+[96]*4
        initial=256
    else:
        assert x['requested_cap']==cap and x['horizons'] is None;initial=cap
    p=x['prefix_votes'];v0=x['probe_votes'];assert [m for m,n in p]==[m for m,n in v0]==[m for m,n in x['votes']]
    n=sum(k for m,k in p);nn=x['prefix_nn'];d=dist(p);probe=x['probe_wave'];assert 0<probe<=len(x['waves'])
    for i,w in enumerate(x['waves']):
        bn,bnn,bd,an,ann,ad,target,vcap=w;assert (bn,bnn,bd)==(n,nn,d)
        h=initial if i<probe else cap
        if i<probe:assert n<32 and nn<32
        assert target==h-nn and vcap==min(8,h-n) and 0<=ann-nn<=target and 0<=an-n<=vcap
        if q:assert d is not None and d>0 and d<=h-n
        n,nn,d=an,ann,ad
        if i+1==probe:assert n==sum(k for m,k in v0) and nn==x['probe_nn'] and d==dist(v0) and (n>=32 or nn>=32) and n<=39
        CT['waves']+=1
    assert (n,nn,d)==(r['visits'],r['nn'],dist(x['votes']))
    CT['wave_receipts']+=1
def check_reuse(h,r,bd):
    t=r['target'];a=r['low']['move'];assert t['incumbent']==a
    vs=dict(r['low']['votes']);alts=sorted((m for m in vs if m!=a and bd[m][1]!=-1),key=lambda m:(-vs[m],m))
    b=alts[0] if alts else None
    if b is not None and bd[a][0]>=bd[b][1]:b=None
    assert b==t['rival']
    if b is None:
        assert t['chosen']==a and t['extra_nn']==0 and not t['children'] and not t['values']
    else:
        assert bd[a][1]!=-1 and len(t['children'])==2
        vals=[];nn=0
        for m,z in zip((a,b),t['children']):
            l,u=bd[m]
            if l==u:assert z is None;vals.append(l);continue
            assert z is not None and z['old_root_n']==vs[m]
            aa,bb,w=v.replay(h+[m],7);term=bool(w) or aa|bb==v.G[7][1]
            assert z['terminal']==term
            if term:assert z['nn']==z['visits']==0 and not z['samples'] and z['raw']==(1. if w else 0.)
            else:
                pre=z['pre_edges'];post=z['edges'];legal=[i for i in range(49) if not (aa|bb)>>i&1]
                assert [e[0] for e in pre]==[e[0] for e in post]==legal
                assert sum(e[1] for e in pre)==max(0,vs[m]-1) and z['initial_eval']==(vs[m]==0)
                assert sum(e[1] for e in post)-sum(e[1] for e in pre)==5 and z['visits']==5
                assert all(0<=x[1]<=y[1] and abs(x[2])<=x[1]+1e-12 and abs(y[2])<=y[1]+1e-12 for x,y in zip(pre,post))
                assert len(z['samples'])==5 and all(-1<=x<=1 for x in z['samples'])
                assert abs(math.fsum(y[2]-x[2] for x,y in zip(pre,post))-math.fsum(z['samples']))<1e-12
                assert abs(z['raw']+math.fsum(z['samples'])/5)<2e-15
                assert z['nn']==int(z['initial_eval'])+5-z['terminal_leaves']
                CT['reused_children']+=not z['initial_eval']
            vals.append(max(l,min(u,z['raw'])));nn+=z['nn'];CT['reuse_child_receipts']+=1
        assert t['values']==vals and t['extra_nn']==nn<=12 and t['chosen']==(b if vals[1]>vals[0] else a)
    assert r['move']==t['chosen'] and r['nn']==r['native_physical_nn']==r['physical_nn']+t['extra_nn']
    CT['reuse_events']+=1;CT['reuse_local_nn']+=t['extra_nn'];CT['reuse_switch']+=t['chosen']!=a
    return dict(r,move=a,nn=r['physical_nn'])
def rapfi_target(h,r,bd):
    t=r['target'];vs=r['votes'];a=t['incumbent'];orig=r['original_move'];assert a==v.changed(orig,vs,bd,15)
    i=max(range(len(vs)),key=lambda j:(vs[j][1],j));assert vs[i][0]==orig
    d=min((vs[i][1]-n+int(j<i) for j,(m,n) in enumerate(vs) if j!=i),default=10**9)
    assert t['distance']==d<=8 and t['prior_best']==vs[0][0]!=orig
    rr=[(j,m,n) for j,(m,n) in enumerate(vs) if m!=a and bd[m][1]!=-1]
    b=max(rr,key=lambda x:(x[2],x[0]))[1] if rr else None
    if b is not None and (bd[a][1]==-1 or bd[a][0]>=bd[b][1]):b=None
    assert b==t['rival'];nn=0
    if b is None:assert t['chosen']==a and not t['children'] and not t['values']
    else:
        vals=[];assert len(t['children'])==2
        for m,z in zip((a,b),t['children']):
            l,u=bd[m]
            if l==u:assert z is None;vals.append(l);continue
            aa,bb,w=v.replay(h+[m],15);term=bool(w) or aa|bb==v.G[15][1]
            assert z['terminal']==term
            if term:assert z['nn']==z['visits']==0 and z['raw']==(1. if w else 0.)
            else:
                es=z['edges'];assert sum(x['n'] for x in es)==z['visits']==5
                legal=set(i for i in range(225) if not (aa|bb)>>i&1)
                assert len({x['move'] for x in es})==len(es) and {x['move'] for x in es}<=legal
                assert all(x['n']>=0 and abs(x['w'])<=x['n']+1e-9 and x['vl']==0 for x in es)
                w0=math.fsum(x['w'] for x in es);assert abs(w0-z['root_w'])<1e-9 and abs(z['raw']+w0/5)<1e-12
                assert z['budget']['target']==z['budget']['cap']==z['budget']['visits']==5 and z['nn']==1+z['budget']['used_nn']<=6
            vals.append(max(l,min(u,z['raw'])));nn+=z['nn'];CT['rapfi_children']+=1
        assert vals==t['values'] and t['chosen']==(b if vals[1]>vals[0] else a)
    assert t['extra_nn']==nn<=12 and r['move']==t['chosen'] and r['nn']==r['root_nn']+nn
    CT['rapfi_events']+=1;CT['rapfi_local_nn']+=nn;CT['rapfi_switch']+=a!=r['move']
    return dict(r,move=a,nn=r['root_nn'])
def cache_check(h,r):
    z=r['cache_receipt'];keys=[tuple(x) for x in z['keys']];seen=set();hi=[]
    # Node input legal order is canonical ascending complement. Identity includes absolute stones and side.
    aa,bb,_=v.replay(h,7)
    if keys:assert keys[0]==(aa,bb,1-2*(len(h)%2))
    for i,(a,b,p) in enumerate(keys):
        assert a&b==0 and (a|b)<(1<<49) and a&aa==aa and b&bb==bb
        assert a.bit_count()-b.bit_count() in (0,1) and p==1-2*(a.bit_count()-b.bit_count())
        k=(a,b,p)
        if k in seen:hi.append(i)
        seen.add(k)
    assert len(keys)==z['logical']==r['nn'] and len(seen)==z['physical']==r['actual_nn'] and z['hit_indices']==hi
    assert z['hits']==len(hi) and z['root_physical']==z['root_logical']-sum(i<z['root_logical'] for i in hi)
    if 'target' in r:assert z['root_logical']==r['physical_nn'] and r['target']['extra_nn']==r['nn']-z['root_logical']
    else:assert z['root_logical']==r['nn']
    CT['cache_requests']+=1;CT['cache_logical']+=z['logical'];CT['cache_physical']+=z['physical'];CT['cache_hits']+=z['hits']
    CT['cache_root_hits']+=sum(i<z['root_logical'] for i in hi)
    CT['cache_target_hits']+=sum(i>=z['root_logical'] for i in hi)
def repeat_check(g,events):return not events
def selected_op(g,r):
    import hashlib
    z=2*int(r['bin']==7)+int(r['ply']>8)
    if g['policy']=='SELECT':table=load('FROZEN.json')['table']
    elif g['policy']=='MIX':
        m=load('MIX_FROZEN.json');seed0=load('PLAN.json')['seed']
        coin=int.from_bytes(hashlib.sha256(f"quartz-k55-mix:{seed0}:{g['job']['id']}:{g['seed']}:{g['color']}".encode()).digest()[:8],'big')
        high=coin*m['prob_den']<(1<<64)*m['prob_num']
        table=m['high'] if high else m['low']
    else:return {'CONTROL':0,'COLD':1,'REUSE':2}[g['policy']]
    assert r['selection']==dict(cell=z,op=table[z],table=table)
    return table[z]
def local_cost(r):
    if 'target' not in r:return 0
    if 'cache_receipt' not in r:return r['target']['extra_nn']
    z=r['cache_receipt'];return z['physical']-z['root_physical']
def audit_game(g,j,n,policy,counts,cache=True):
    assert g['job']==j and g['color'] in j['color_order'] and g['seed'] in ((0,) if n==15 else SEEDS)
    h=j['opening'].copy();events=[];nn=0;physical=0;c=C.Counter()
    for i,r in enumerate(g['rows']):
        side=1-2*(len(h)%2);q=side==g['color'];a,b=v.orient(h,n);bd,g0=v.base(a,b,n)
        assert r['move'] in bd and r['ply']==len(h) and r['player']==side and r['arm']==('Q' if q else 'B')
        z=r.get('partial');assert bool(z)==bool(q and g0 is None)
        cap=(96 if q else 128) if n==15 else 20 if q else 32;rr=r
        if z:
            bds=replay_proof(h,n,z);c['queries']+=z['cold']['receipt']['queries'];c['cert']+=z['cert'] is not None
            if z['cert'] is not None:
                assert r['kind']=='PARTIAL47' and r['move']==z['cert'] and r['original_move'] is None and not r['replacement']
                vote(h,n,r,True,cap)
            else:
                if n==7 and repeat_check(g,events):
                    l=r['low'];bn=int(l['distance']<=3)+2*int(l['original_move']!=l['prior_best'])+4*int(any(u==-1 for _,u in bds.values()))
                    assert bn==r['bin'] and not r['extended'] and r['mask']==0;check_snapshot(h,l,bds,20)
                    if bn in (3,7):
                        events.append(i)
                        op=selected_op(g,r)
                        if op==1:rr=check_target(h,r,bds)
                        elif op==2:rr=check_reuse(h,r,bds)
                        else:assert 'target' not in r
                    else:assert 'target' not in r and 'selection' not in r
                    assert rr['physical_nn']==rr['nn'] and all(rr[f]==l[f] for f in ('move','kind','nn','visits','votes','distance','original_move','replacement','cap'))
                elif n==15:
                    vs=r['votes'];j0=max(range(len(vs)),key=lambda k:(vs[k][1],k));di=min((vs[j0][1]-nv+int(k<j0) for k,(m,nv) in enumerate(vs) if k!=j0),default=10**9)
                    eligible=di<=8 and vs[0][0]!=vs[j0][0] and policy=='R_T' and not events
                    assert ('target' in r)==eligible
                    if eligible:events.append(i);rr=rapfi_target(h,r,bds)
                else:assert not r.get('low') and 'target' not in r
                vote(h,n,rr,True,cap,rr['original_move'])
                mv=v.changed(rr['original_move'],rr['votes'],bds,n);assert mv==rr['move'] and rr['replacement']==(mv!=rr['original_move']);c['replacement']+=rr['replacement']
        else:
            assert not r.get('low') and 'target' not in r;vote(h,n,r,q,cap)
        if n==15:wave_audit(rr,96,q)
        assert r['cpu_ns']>=0 and r['wall_ns']>=0
        if n==7:
            assert r['history']==h and r['lock']==q
            if cache:cache_check(h,r)
            else:assert 'cache_receipt' not in r and r['actual_nn']==r['nn']
            physical+=r['actual_nn']
            counts['physical']+=r['actual_nn'];counts['logical']+=r['nn']
            c[('Q' if q else 'B')+'logical']+=r['nn']
        tag='Q' if q else 'B';c[tag+'moves']+=1
        for f,kk in [('actual_nn' if n==7 else 'nn','nn'),('cpu_ns','cpu'),('wall_ns','wall')]:c[tag+kk]+=r[f]
        nn+=r['nn'];counts['moves']+=1;counts['common']+=r['kind']=='CERTIFICATE';h.append(r['move'])
    assert h==g['history'];a,b,w=v.replay(h,n);assert w or (a|b)==v.G[n][1]
    assert w==g['winner'] and g['score2']==1+g['color']*w and nn==g['nn']
    assert g['event']==(events[0] if events else None) and g['policy']==policy
    extra=sum(r['target']['extra_nn'] for r in g['rows'] if 'target' in r)
    if n==7:
        assert g['events']==events and g['actual_nn']==physical
        counts[policy.lower()+'_events']+=len(events);counts[policy.lower()+'_extra']+=sum(local_cost(r) for r in g['rows']);counts[policy.lower()+'_extra_logical']+=extra
    elif events:
        counts['events']+=1;counts['extra']+=extra;counts['changed']+=g['rows'][events[0]]['target']['incumbent']!=g['rows'][events[0]]['move']
    if n==15:counts['nn']+=nn
    counts['games']+=1;c['games']+=1;c['score2']+=g['score2'];c['wins']+=g['score2']==2;c['draws']+=g['score2']==1;c['losses']+=g['score2']==0
    return c
def pair_ind(c,t,z,cc,tt):
    e=c['event'];assert e==t['event']
    local=sum(local_cost(r) for r in t['rows'])
    logical=sum(r['target']['extra_nn'] for r in t['rows'] if 'target' in r)
    changed=False;cell=None;op=0
    if e is None:assert c['history']==t['history'] and c['actual_nn']==t['actual_nn']
    else:
        a,b=c['rows'][e],t['rows'][e];assert a['low']==b['low'] and a['history']==b['history']
        for r,s in zip(c['rows'][:e],t['rows'][:e]):assert (r['move'],r['nn'],r['visits'],r['votes'],r['actual_nn'])==(s['move'],s['nn'],s['visits'],s['votes'],s['actual_nn'])
        cell=2*int(a['bin']==7)+int(a['ply']>8);op=selected_op(t,b);changed=a['move']!=b['move']
        if not changed:assert c['history']==t['history'] and tt['Qnn']-cc['Qnn']==local and tt['Bnn']==cc['Bnn']
    dq=tt['Qnn']-cc['Qnn'];db=tt['Bnn']-cc['Bnn']
    r=dict(job=c['job']['id'],seed=c['seed'],color=c['color'],policy=t['policy'],cell=cell,op=op,reached=e is not None,changed=changed,same_history=c['history']==t['history'],delta2=t['score2']-c['score2'],local_nn=local,logical_extra=logical,downstream_Q=dq-local,downstream_B=db,delta_Q=dq,delta_B=db,delta_total=dq+db,interventions=int(op>0),base_interventions=0,score2=[c['score2'],t['score2']])
    assert r==z
    return r
def audit_shards(arg):
    mode,files=arg;CT.clear();EX.clear();v.CT.clear();n=7 if mode=='NN' else 15;p=load('PLAN.json');jobs={j['id']:j for j in p['jobs' if n==7 else 'rapfi_jobs']}
    ct=C.Counter();keys=[];out=C.defaultdict(C.Counter);ps=C.defaultdict(C.Counter)
    for f in files:
        path=D/f['file'];assert sha(path)==f['sha256'];counts=C.Counter()
        for x in raw(path):
            if n==15:
                j=jobs[x['job']['id']];keys.append((j['id'],0,x['color']));cc=audit_game(x,j,n,mode,counts);out['all'].update(cc);out[j['stratum']].update(cc)
            else:
                c=x['control'];j=jobs[c['job']['id']];keys.append((j['id'],c['seed'],c['color']));counts['episodes']+=1
                pan='old' if c['seed'] in (101,102,103,105) else 'new';gs={}
                for arm in ('control','cold','reuse','select','mix'):
                    gs[arm]=audit_game(x[arm],j,n,arm.upper(),counts);out[arm+'_all'].update(gs[arm]);out[arm+'_'+pan].update(gs[arm])
                for arm,base in [(a,'control') for a in ('cold','reuse','select','mix')]:
                    r=pair_ind(x[base],x[arm],x['pairs'][arm],gs[base],gs[arm]);counts[arm+'_delta2']+=r['delta2']
                    for k in ('delta2','local_nn','delta_Q','delta_B','delta_total','downstream_Q','downstream_B','same_history','reached','changed','interventions','base_interventions'):ps[arm][k]+=r[k]
                import hashlib
                m=load('MIX_FROZEN.json');seed0=p['seed'];coin=int.from_bytes(hashlib.sha256(f"quartz-k55-mix:{seed0}:{j['id']}:{c['seed']}:{c['color']}".encode()).digest()[:8],'big')
                assert x['coin']==coin and x['high']==(coin*m['prob_den']<(1<<64)*m['prob_num'])
                for arm in ('select','mix'):
                    g=x[arm];e=g['event'];op=0 if e is None else selected_op(g,g['rows'][e]);b=x[('control','cold','reuse')[op]]
                    assert (g['history'],g['score2'],g['nn'],g['actual_nn'])==(b['history'],b['score2'],b['nn'],b['actual_nn'])
                    assert all((a['move'],a['nn'],a['actual_nn'],a['votes'],a['cache_receipt'])==(z['move'],z['nn'],z['actual_nn'],z['votes'],z['cache_receipt']) for a,z in zip(g['rows'],b['rows'],strict=True))
                    counts[arm+'_op'+str(op)]+=1
        assert all(counts[k]==x for k,x in f['counts'].items()) and all(f['counts'].get(k,0)==x for k,x in counts.items()),(mode,counts,f['counts'])
        ct.update(counts)
    return dict(ct),{k:dict(v0) for k,v0 in out.items()},keys,dict(CT),dict(v.CT),{k:dict(v0) for k,v0 in ps.items()},[(list(k),r) for k,r in EX.items()]
def audit_mode(mode):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    lk=load(mode+'_LOCK.json');assert lk['plan']==sha(D/'PLAN.json') and lk['source']==sha(D/'experiment55.py') and lk['bridge']==sha(D/'native55.py') and lk['binary']==sha(T/'mcts_demo')
    fs=lk['files'];ct=C.Counter();tot=C.defaultdict(C.Counter);keys=[];ac=C.Counter();ic=C.Counter();pc=C.defaultdict(C.Counter);ex=C.defaultdict(list)
    workers=2 if mode=='NN' else 4
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        for c,o,k,x,y,z,e in pool.map(audit_shards,[(mode,fs[i::workers]) for i in range(workers)]):
            ct.update(c);keys.extend(k);ac.update(x);ic.update(y)
            for s,t in z.items():pc[s].update(t)
            for s,t in o.items():tot[s].update(t)
            for k,r in e:ex[tuple(k)].extend(r)
    assert dict(ct)==lk['counts'] and len(set(keys))==len(keys)==(ct['episodes'] if mode=='NN' else ct['games'])
    p=load('PLAN.json');assert set(keys)=={(j['id'],s,c) for j in (p['jobs'] if mode=='NN' else p['rapfi_jobs']) for s in (SEEDS if mode=='NN' else (0,)) for c in j['color_order']}
    save('AUDIT_'+mode+'.json',dict(counts=dict(ct),panels={s:dict(t) for s,t in tot.items()},audit=dict(ac),independent=dict(ic),pair={s:dict(t) for s,t in pc.items()},natural_exact=[dict(a=k[0],b=k[1],proofs=r) for k,r in ex.items()]))
    print('AUDITED',mode,dict(ct),dict(ac),dict(ic),'exact_unique',len(ex),flush=True)
def final():
    p=load('PLAN.json');fam={family(j['opening']) for j in p['jobs']};assert len(fam)==64
    for x in p['prior_locks']:
        q=ROOT/x['path'];assert sha(q)==x['sha256'];prev=json.loads(q.read_text());assert not fam&{family(j['opening']) for j in prev['jobs']}
    sys.path.insert(0,str(D.parent/'quartz_kinetic37_20261126'));import experiment as e
    rf={e.family(j['opening']) for j in p['rapfi_jobs']};assert len(rf)==128
    for x in p['rapfi_prior_locks']:assert sha(ROOT/x['path'])==x['sha256']
    for x in p['prior_locks']:assert not rf&{e.family(j['opening']) for j in json.loads((ROOT/x['path']).read_text()).get('rapfi_jobs',[])}
    assert not rf&{e.family(j['opening']) for j in e.read('PLAN.json')['jobs']}
    s=load('SUMMARY.json');total=C.Counter();counts=C.Counter();ex=C.defaultdict(list);modes=('NN','R_96','R_T')
    ar={m:json.loads((T/('AUDIT_'+m+'.json')).read_text()) for m in modes}
    def check_metric(c,z):
        assert all(c.get(k,0)==x for k,x in z['counts'].items())
        assert abs(float(Fraction(c['score2'],2*c['games']))-z['metrics']['score'])<1e-15
        for k,l in [('nn','NN'),('cpu','CPU'),('wall','wall')]:
            val=1-Fraction(c['Q'+k]*c['Bmoves'],c['B'+k]*c['Qmoves']);assert abs(float(val)-z['metrics'][l+'_reduction'])<1e-14
    for m,r in ar.items():
        total.update(r['counts']);counts.update(r['audit']);counts.update({'independent_'+k:v0 for k,v0 in r['independent'].items()})
        for x in r['natural_exact']:ex[x['a'],x['b']].extend(x['proofs'])
    for pan in ('all','old','new'):
        for arm,z in s['nn'][pan]['arms'].items():check_metric({k:v for k,v in ar['NN']['panels'][arm+'_'+pan].items() if not k.endswith('logical')},z['physical'])
    for arm,p0 in ar['NN']['pair'].items():
        for k,v0 in p0.items():assert s['nn']['all']['pairs'][arm]['counts'][k]==v0
    for m in ('R_96','R_T'):
        for pan in ('all','compact','tactical'):check_metric(ar[m]['panels'][pan],s['rapfi'][pan]['arms'][m[2:]])
    assert total['games']==4992 and total['episodes']==896
    # Fresh opening generator constraints, checked independently.
    for j in p['rapfi_jobs']:
        rc=[divmod(m,15) for m in j['opening']]
        if j['stratum']=='compact':assert max(r for r,c in rc)-min(r for r,c in rc)<=4 and max(c for r,c in rc)-min(c for r,c in rc)<=4
        else:
            for side in (rc[::2],rc[1::2]):
                a,b,c=sorted(side);assert (b[0]-a[0],b[1]-a[1])==(c[0]-b[0],c[1]-b[1]) and max(abs(b[0]-a[0]),abs(b[1]-a[1]))==1
            assert all(1<=r<14 and 1<=c<14 for r,c in rc)
    co={}
    xx=[]
    if ex:
        binary=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic51_targeted_backups/independent51'
        assert sha(binary)==json.loads((D.parent/'quartz_kinetic51_targeted_20261127/EXACT_EXTENDED_LOCK.json').read_text())['binary_sha256']
        keys=list(ex);inp=''.join(f'{i} {a} {b}\n' for i,(a,b) in enumerate(keys));rs=list(map(json.loads,subprocess.run([str(binary)],input=inp,text=True,capture_output=True,check=True,timeout=120).stdout.splitlines()));assert len(rs)==len(keys)
        for r in rs:
            k=keys[r['id']]
            if r['complete']:
                for z in ex[k]:
                    vv=r['values'];assert all(l<=vv[str(m)]<=u for m,l,u in z['bounds'])
                    if z['cert'] is not None:assert vv[str(z['cert'])]==max(vv.values())
            xx.append(dict(a=k[0],b=k[1],**r))
    save('NATURAL_EXACT_AUDIT.json',dict(roots=len(xx),complete=sum(r['complete'] for r in xx),rows=xx))
    out=dict(total=dict(total),counts=dict(counts),pair=ar['NN']['pair'],cache=dict(co),natural_exact=dict(roots=len(xx),complete=sum(r['complete'] for r in xx)),modes={m:{k:v0 for k,v0 in x.items() if k!='natural_exact'} for m,x in ar.items()},scope='independent coordinate proof, chronological selector, exact cache and cold/reused child receipts; actual live arm replay; evaluator outputs not all independently rerun')
    save('AUDIT.json',out);print('FINAL_AUDIT',json.dumps({k:v0 for k,v0 in out.items() if k!='modes'}),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='final':final()
    else:audit_mode(sys.argv[1])
