"""Independent integer/coordinate proof and chronological action/cost audit."""
import collections as C,gzip,json,pathlib,sys,importlib.util,subprocess,math
from fractions import Fraction
from k52_plan import D,ROOT,T,SEEDS,load,sha,family
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
def audit_game(g,j,n,policy,counts,qcap):
    assert g['job']==j and g['color'] in j['color_order'] and g['seed'] in ((0,) if n==15 else SEEDS)
    h=j['opening'].copy();event=None;nn=0;c=C.Counter()
    for i,r in enumerate(g['rows']):
        side=1-2*(len(h)%2);q=side==g['color'];a,b=v.orient(h,n);bd,g0=v.base(a,b,n)
        assert r['move'] in bd and r['ply']==len(h) and r['player']==side and r['arm']==('Q' if q else 'B')
        z=r.get('partial');assert bool(z)==bool(q and g0 is None)
        cap=r['selected_cap'] if n==15 else 20 if q else 32
        if n==15:
            if not q:assert cap==128
            elif qcap!='AD':assert cap==qcap
            elif r['kind']=='CERTIFICATE':assert cap==128
            elif r['kind']=='PARTIAL47':assert cap==256
            else:
                x=r['native'];vs=x['probe_votes'];lead=max(range(len(vs)),key=lambda j:(vs[j][1],j));distance=min((vs[lead][1]-n0+int(j<lead) for j,(m,n0) in enumerate(vs) if j!=lead),default=10**9)
                assert cap==(128 if distance<8 else 96) and x['horizons']==[128]*4+[96]*4
                CT['cap_'+str(cap)]+=1
        rr=r
        if z:
            bds=replay_proof(h,n,z);c['queries']+=z['cold']['receipt']['queries'];c['cert']+=z['cert'] is not None
            if z['cert'] is not None:
                assert r['kind']=='PARTIAL47' and r['move']==z['cert'] and r['original_move'] is None and not r['replacement']
                vote(h,n,r,True,cap)
            else:
                if n==7 and event is None:
                    l=r['low'];bn=int(l['distance']<=3)+2*int(l['original_move']!=l['prior_best'])+4*int(any(u==-1 for _,u in bds.values()))
                    assert bn==r['bin'];eligible=bn in (3,7);ext=policy=='GLOBAL' and eligible
                    assert r['extended']==ext and r['mask']==(136 if policy=='GLOBAL' else 0)
                    check_snapshot(h,l,bds,20)
                    if eligible:
                        event=i
                        if policy=='TARGET':rr=check_target(h,r,bds)
                        else:assert 'target' not in r
                        if ext:
                            CT['global_events']+=1;CT['global_local_nn']+=r['nn']-l['nn'];cap=32
                    else:assert 'target' not in r
                    assert rr['physical_nn']==rr['nn']
                    assert all(l['votes'][ii][0]==m and l['votes'][ii][1]<=x for ii,(m,x) in enumerate(rr['votes']))
                    if not ext:
                        assert all(rr[f]==l[f] for f in ('move','kind','nn','visits','votes','distance','original_move','replacement','cap'))
                else:assert not r.get('low') and 'target' not in r
                vote(h,n,rr,True,cap,rr['original_move'])
                mv=v.changed(rr['original_move'],rr['votes'],bds,n);assert mv==rr['move'] and rr['replacement']==(mv!=rr['original_move']);c['replacement']+=rr['replacement']
        else:
            assert not r.get('low') and 'target' not in r
            vote(h,n,r,q,cap)
        assert r['cpu_ns']>=0 and r['wall_ns']>=0
        if n==15:wave_audit(r,qcap,q)
        if n==7:assert r['history']==h and r['lock']==q
        tag='Q' if q else 'B';c[tag+'moves']+=1
        for f,kk in [('nn','nn'),('cpu_ns','cpu'),('wall_ns','wall')]:c[tag+kk]+=r[f]
        nn+=r['nn'];counts['moves']+=1;counts['common']+=r['kind']=='CERTIFICATE';h.append(r['move'])
    assert h==g['history'];a,b,w=v.replay(h,n);assert w or (a|b)==v.G[n][1]
    assert w==g['winner'] and g['score2']==1+g['color']*w and nn==g['nn']
    if n==7:assert g['event']==event and g['policy']==policy
    counts['nn']+=nn;counts['games']+=1
    c['games']+=1;c['score2']+=g['score2'];c['wins']+=g['score2']==2;c['draws']+=g['score2']==1;c['losses']+=g['score2']==0
    return c,event
def compare(c,t,z,ec,et,cc,tt):
    assert ec==et and c['seed']==t['seed'] and c['color']==t['color']
    reached=ec is not None;local=0;postQ=postB=0;bn=None;changed=False;actions=None;ply=None
    if reached:
        r,s=c['rows'][ec],t['rows'][et]
        assert c['history'][:r['ply']]==t['history'][:s['ply']]==r['history']==s['history']
        for x,y in zip(c['rows'][:ec],t['rows'][:et],strict=True):assert (x['move'],x['nn'],x['visits'],x['votes'])==(y['move'],y['nn'],y['visits'],y['votes'])
        assert r['low']==s['low'] and r['bin']==s['bin']
        local=s['nn']-r['nn'];assert local>=0;changed=r['move']!=s['move'];actions=[r['move'],s['move']];bn=r['bin'];ply=r['ply']
        postQ=sum(x['nn'] for x in t['rows'][et+1:] if x['arm']=='Q')-sum(x['nn'] for x in c['rows'][ec+1:] if x['arm']=='Q')
        postB=sum(x['nn'] for x in t['rows'][et+1:] if x['arm']=='B')-sum(x['nn'] for x in c['rows'][ec+1:] if x['arm']=='B')
    dq=tt['Qnn']-cc['Qnn'];db=tt['Bnn']-cc['Bnn'];delta=t['score2']-c['score2'];assert dq==local+postQ and db==postB
    if not changed:assert c['history']==t['history'] and delta==postQ==postB==0
    independent=dict(job=c['job']['id'],seed=c['seed'],color=c['color'],policy=t['policy'],reached=reached,event=ec,ply=ply,bin=bn,actions=actions,changed=changed,score2=[c['score2'],t['score2']],delta2=delta,local_nn=local,downstream_Q=postQ,downstream_B=postB,delta_Q=dq,delta_B=db,delta_total=dq+db,same_history=c['history']==t['history'])
    assert independent==z
    return independent
def jobs(mode):
    p=load('PLAN.json')
    return p['jobs'] if mode=='NN' else [j for j in p['rapfi_jobs'] if j['split']==mode.split('_')[1]]
def audit_shards(arg):
    mode,files=arg;CT.clear();EX.clear();v.CT.clear();n=7 if mode=='NN' else 15;js={j['id']:j for j in jobs(mode)}
    ct=C.Counter();keys=[];out=C.defaultdict(C.Counter);ps=C.defaultdict(C.Counter)
    for f in files:
        path=D/f['file'];assert sha(path)==f['sha256'];counts=C.Counter()
        for x in raw(path):
            if n==15:
                j=js[x['job']['id']];keys.append((j['id'],0,x['color']))
                cap='AD' if mode.endswith('_AD') else int(mode.split('_')[-1]);assert x['qcap']==cap
                c,e=audit_game(x,j,n,None,counts,cap);out['all'].update(c);out[j['stratum']].update(c)
            else:
                c=x['control'];j=js[c['job']['id']];keys.append((j['id'],c['seed'],c['color']))
                cc,ec=audit_game(c,j,n,'CONTROL',counts,20);counts['episodes']+=1
                pan='old' if c['seed'] in (101,102,103,105) else 'new'
                out['control_all'].update(cc);out['control_'+pan].update(cc)
                for arm in ('global','target'):
                    t=x[arm];tt,et=audit_game(t,j,n,arm.upper(),counts,20);r=compare(c,t,x['pairs'][arm],ec,et,cc,tt)
                    for k in ('reached','changed','local_nn','delta2'):counts[arm+'_'+k]+=r[k]
                    for k in ('delta_Q','delta_B','delta_total','downstream_Q','downstream_B','same_history'):ps[arm][k]+=r[k]
                    out[arm+'_all'].update(tt);out[arm+'_'+pan].update(tt)
        assert all(counts[k]==x for k,x in f['counts'].items()) and all(f['counts'].get(k,0)==x for k,x in counts.items()),(mode,counts,f['counts'])
        ct.update(counts)
    return dict(ct),{k:dict(v0) for k,v0 in out.items()},keys,dict(CT),dict(v.CT),{k:dict(v0) for k,v0 in ps.items()},[(list(k),r) for k,r in EX.items()]
def audit_mode(mode):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    lk=load(mode+'_LOCK.json')
    assert lk['plan']==sha(D/'PLAN.json') and lk['source']==sha(D/'experiment52.py') and lk['bridge']==sha(D/'native52.py') and lk['target_source']==sha(ROOT/'src/mcts_server/k47_nn.rs') and lk['binary']==sha(T/'mcts_demo')
    fs=lk['files'];ct=C.Counter();tot=C.defaultdict(C.Counter);keys=[];ac=C.Counter();ic=C.Counter();pc=C.defaultdict(C.Counter);ex=C.defaultdict(list)
    workers=2 if mode=='NN' else 4
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        for c,o,k,x,y,z,e in pool.map(audit_shards,[(mode,fs[i::workers]) for i in range(workers)]):
            ct.update(c);keys.extend(k);ac.update(x);ic.update(y)
            for s,t in z.items():pc[s].update(t)
            for s,t in o.items():tot[s].update(t)
            for k,r in e:ex[tuple(k)].extend(r)
    assert dict(ct)==lk['counts'] and len(set(keys))==len(keys)==(ct['episodes'] if mode=='NN' else ct['games'])
    assert set(keys)=={(j['id'],s,c) for j in jobs(mode) for s in (SEEDS if mode=='NN' else (0,)) for c in j['color_order']}
    if mode=='NN':
        assert ac['global_events']==ct['global_reached'] and ac['target_events']==ct['target_reached']
        assert ac['global_local_nn']==ct['global_local_nn'] and ac['target_local_nn']==ct['target_local_nn'] and ac['target_switch']==ct['target_changed']
    save('AUDIT_'+mode+'.json',dict(counts=dict(ct),panels={s:dict(t) for s,t in tot.items()},audit=dict(ac),independent=dict(ic),pair={s:dict(t) for s,t in pc.items()},natural_exact=[dict(a=k[0],b=k[1],proofs=r) for k,r in ex.items()]))
    print('AUDITED',mode,dict(ct),dict(ac),dict(ic),'exact_unique',len(ex),flush=True)
def final():
    p=load('PLAN.json');fam={family(j['opening']) for j in p['jobs']};assert len(fam)==96
    for x in p['prior_locks']:
        q=ROOT/x['path'];assert sha(q)==x['sha256'];prev=json.loads(q.read_text());assert not fam&{family(j['opening']) for j in prev['jobs']}
    sys.path.insert(0,str(D.parent/'quartz_kinetic37_20261126'));import experiment as e
    rf={e.family(j['opening']) for j in p['rapfi_jobs']};assert len(rf)==80
    for x in p['rapfi_prior_locks']:assert sha(ROOT/x['path'])==x['sha256']
    for x in p['prior_locks']:
        prev=json.loads((ROOT/x['path']).read_text());assert not rf&{e.family(j['opening']) for j in prev.get('rapfi_jobs',[])}
    assert not rf&{e.family(j['opening']) for j in e.read('PLAN.json')['jobs']}
    for j in p['rapfi_jobs']:assert all(not(3<=r<=11 and 3<=c<=11) for r,c in map(lambda x:divmod(x,15),j['opening']))
    s=load('SUMMARY.json');f=load('SELECTION.json');chosen=f['selected']
    total=C.Counter();counts=C.Counter();ex=C.defaultdict(list);modes=('NN','R_CAL_104','R_CAL_112','R_CAL_AD','R_TEST_96','R_TEST_'+chosen)
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
        for arm,z in s['nn'][pan]['arms'].items():check_metric(ar['NN']['panels'][arm+'_'+pan],z)
    for arm,p0 in ar['NN']['pair'].items():
        for k,v0 in p0.items():assert s['nn']['all']['effects'][arm]['counts'][k]==v0
    for c in ('104','112','AD'):check_metric(ar['R_CAL_'+c]['panels']['all'],f['cal'][c])
    check_metric(ar['R_TEST_96']['panels']['all'],s['rapfi']['test']['a'])
    check_metric(ar['R_TEST_'+chosen]['panels']['all'],s['rapfi']['test']['b'])
    cs=('104','112','AD');ok=[c for c in cs if all(f['cal'][c]['gate'].values())]
    expected=max(ok,key=lambda c:(f['cal'][c]['lower95']['NN_reduction'],f['cal'][c]['lower95']['score'],-cs.index(c))) if ok else '112'
    assert f['selected']==expected and f['eligible']==ok and f['admitted']==bool(ok)
    xx=[]
    if ex:
        binary=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic41_exact_backups/independent';assert sha(binary)=='a1e943f250eae145e10a12482fc575d11af08960dfd7ddf804c1f0c0ff14c7af'
        keys=list(ex);inp=''.join(f'{i} {a} {b}\n' for i,(a,b) in enumerate(keys));rs=list(map(json.loads,subprocess.run([str(binary)],input=inp,text=True,capture_output=True,check=True,timeout=120).stdout.splitlines()));assert len(rs)==len(keys)
        for r in rs:
            k=keys[r['id']]
            if r['complete']:
                for z in ex[k]:
                    vv=r['values'];assert all(l<=vv[str(m)]<=u for m,l,u in z['bounds'])
                    if z['cert'] is not None:assert vv[str(z['cert'])]==max(vv.values())
            xx.append(dict(a=k[0],b=k[1],**r))
    save('NATURAL_EXACT_AUDIT.json',dict(roots=len(xx),complete=sum(r['complete'] for r in xx),rows=xx))
    assert total['games']==4416 and total['episodes']==1344
    out=dict(total=dict(total),counts=dict(counts),pair=ar['NN']['pair'],natural_exact=dict(roots=len(xx),complete=sum(r['complete'] for r in xx)),modes={m:{k:v0 for k,v0 in x.items() if k!='natural_exact'} for m,x in ar.items()},scope='independent coordinate proof, chronological root/child receipts, candidate selection, cost and paired outcome arithmetic; native target and root-dependent cap selection')
    save('AUDIT.json',out);print('FINAL_AUDIT',json.dumps({k:v0 for k,v0 in out.items() if k!='modes'}),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='final':final()
    else:audit_mode(sys.argv[1])
