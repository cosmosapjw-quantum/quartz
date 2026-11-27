"""Independent rule/forcing replay, root/cost receipts, prospective family and exact oracle audits."""
import collections as C,gzip,json,pathlib,sys,importlib.util,subprocess,math
from fractions import Fraction
from k46_plan import D,ROOT,T,SEEDS,load,sha,family
V=D.parent/'quartz_kinetic43_partial_20261127';K45=D.parent/'quartz_kinetic45_sequential_20261127'
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
    assert z['cpu_ns']==z['cold']['cpu_ns']+(ex['cpu_ns'] if ex else 0)
    CT['proof_queries']+=r['queries'];CT['proof_receipts']+=1
    return bd
def vote(h,n,r,lock,pre=None,cap=32):
    if n==15:
        v.vote(h,n,dict(r,kind='PARTIAL43') if r['kind']=='PARTIAL46' else r,lock,pre);return
    a,b=v.orient(h,n);bd,g=v.base(a,b,n);E=list(bd)
    assert r['cap']==cap and r['lock']==lock and r['nn']>=0 and r['cpu_ns']>=0 and r['wall_ns']>=0
    if g is not None:
        assert r['kind']=='CERTIFICATE' and r['move']==g and r['nn']==r['visits']==0 and not r['votes'];CT['common']+=1;return
    if r['kind']=='PARTIAL46':
        assert lock and r['nn']==r['visits']==0 and not r['votes'];return
    vs=r['votes'];assert [m for m,x in vs]==E and all(type(x) is int and x>=0 for m,x in vs)
    nv=dict(vs);lead=min(E,key=lambda m:(-nv[m],m));assert (r['move'] if pre is None else pre)==lead
    d=min((nv[lead]-x+int(m>lead) for m,x in vs if m!=lead),default=10**9)
    assert sum(nv.values())==r['visits']<=cap and r['nn']==1+r['visits']-r['terminal_leaves'] and r['distance']==d
    if r['kind']=='VOTE_LOCK':assert lock and d>cap-r['visits'] and r['visits']<cap
    else:assert r['kind']=='CAP' and r['visits']==cap
    CT['votes']+=1;CT['edges']+=len(vs)
def audit_shards(arg):
    mode,files=arg
    CT.clear();EX.clear();v.CT.clear()
    p=load('PLAN.json');n=15 if mode=='R_ARENA' else 7;cap=20 if mode=='NN_LOW' else 32
    jobs={j['id']:j for j in p['rapfi_jobs' if n==15 else 'jobs']}
    out=C.defaultdict(C.Counter);ct=C.Counter();keys=[]
    for f in files:
        path=D/f['file'];assert sha(path)==f['sha256']
        counts=C.Counter()
        for g in raw(path):
            j=g['job'];assert j==jobs[j['id']] and j['split']=='ARENA'
            assert g['seed'] in (SEEDS if n==7 else (0,)) and g['color'] in j['color_order']
            keys.append((j['id'],g['seed'],g['color']));h=j['opening'].copy();nn=0
            assert not g['atlas']
            s=out[str(g['seed'])]
            for r in g['rows']:
                side=1-2*(len(h)%2);q=side==g['color'];a,b=v.orient(h,n);bd,g0=v.base(a,b,n)
                assert r['move'] in bd and r['ply']==len(h) and r['player']==side and r['arm']==('Q' if q else 'B')
                z=r.get('partial');assert bool(z)==bool(q and g0 is None)
                if z:
                    bds=replay_proof(h,n,z)
                    s['attempts']+=1;s['queries']+=z['cold']['receipt']['queries'];s['proof_cpu']+=z['cpu_ns']
                    if z['exact']:s['exact_attempt']+=1;s['exact_complete']+=z['exact']['complete'];s['exact_nodes']+=z['exact']['nodes']
                    if z['cert'] is not None:
                        assert r['kind']=='PARTIAL46' and r['move']==z['cert'] and r['original_move'] is None and not r['replacement'];s['cert']+=1;vote(h,n,r,True,cap=cap)
                    else:
                        vote(h,n,r,True,r['original_move'],cap=cap)
                        mv=v.changed(r['original_move'],r['votes'],bds,n);assert mv==r['move'] and r['replacement']==(mv!=r['original_move'])
                        s['replacement']+=r['replacement']
                else:vote(h,n,r,q,cap=cap if q else 32)
                tag='Q' if q else 'B';s[tag+'moves']+=1
                for k,kk in [('nn','nn'),('cpu_ns','cpu'),('wall_ns','wall')]:s[tag+kk]+=r[k]
                nn+=r['nn'];counts['moves']+=1;counts['common']+=r['kind']=='CERTIFICATE';h.append(r['move'])
            assert h==g['history'];a,b,w=v.replay(h,n);assert w or (a|b)==v.G[n][1]
            assert w==g['winner'] and g['score2']==1+g['color']*w and nn==g['nn']
            s['games']+=1;s['score2']+=g['score2'];s['wins']+=g['score2']==2;s['draws']+=g['score2']==1;s['losses']+=g['score2']==0
            counts['nn']+=nn;counts['games']+=1
        assert dict(counts)==f['counts'],(mode,f['file'],dict(counts),f['counts'])
        ct.update(counts)
    return dict(ct),{k:dict(s) for k,s in out.items()},keys,dict(CT),dict(v.CT),[(list(k),r) for k,r in EX.items()]
def audit_mode(mode):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    lk=load(mode+'_LOCK.json')
    assert lk['plan']==sha(D/'PLAN.json') and lk['source']==sha(D/'k46_experiment.py') and lk['engine']==sha(D/'k46_engine.py') and lk['hybrid']==sha(D/'hybrid46.py') and lk['library']==sha(T/'pair46.so')
    fs=lk['files'];ct=C.Counter();tot=C.defaultdict(C.Counter);keys=[];ac=C.Counter();ic=C.Counter();ex=C.defaultdict(list)
    # For low while another arena runs, limited to two workers; others four.
    workers=2 if mode=='NN_LOW' else 4
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        for c,o,k,x,y,z in pool.map(audit_shards,[(mode,fs[i::workers]) for i in range(workers)]):
            ct.update(c);keys.extend(k);ac.update(x);ic.update(y)
            for s,t in o.items():tot[s].update(t)
            for k,r in z:ex[tuple(k)].extend(r)
    assert dict(ct)==lk['counts'] and len(set(keys))==len(keys)==(128 if mode=='R_ARENA' else 1344)
    p=load('PLAN.json');n=15 if mode=='R_ARENA' else 7
    assert set(keys)=={(j['id'],s,c) for j in p['rapfi_jobs' if n==15 else 'jobs'] for s in ((0,) if n==15 else SEEDS) for c in j['color_order']}
    save('AUDIT_'+mode+'.json',dict(counts=dict(ct),per_seed={s:dict(t) for s,t in tot.items()},audit=dict(ac),independent=dict(ic),natural_exact=[dict(a=k[0],b=k[1],proofs=r) for k,r in ex.items()]))
    print('AUDITED',mode,dict(ct),dict(ac),dict(ic),'exact_unique',len(ex),flush=True)
def audit_pairs():
    pl=load('PAIR_PLAN.json');lk=load('PAIR_LOCK.json');p=D/lk['file'];assert sha(p)==lk['sha256']
    assert lk['plan']==sha(D/'PAIR_PLAN.json') and lk['source']==sha(D/'pair46.cpp') and lk['library']==sha(T/'pair46.so')
    for path,digest in pl['source'].items():assert sha(ROOT/path)==digest
    old=load('PLAN.json');pp=pl['jobs'];ct=C.Counter();tasks={}
    for r,j in zip(raw(p),pp,strict=True):
        assert r['job']==j
        aa,bb=v.orient(j['history'],7);assert aa==j['a'] and bb==j['b']
        l,u=[list(x) for x in zip(*j['input'])];z=r['result']
        for i,ask in enumerate(z['ask']):
            m,t=divmod(i,2);nodes=z['nodes'][i];assert 0<=nodes<=100000
            resolved=l[m]>=t or u[m]<t
            if resolved:assert ask==-2 and nodes==0
            elif ask==-1:assert nodes==100000
            else:
                assert ask in (0,1)
                if ask:l[m]=max(l[m],t)
                else:u[m]=min(u[m],t-1)
                assert l[m]<=u[m]
                if j['lane']=='EARLY':tasks[j['id'],i]=ask
        assert z['bounds']==[[l[i],u[i]] for i in range(2)]
        def gain(bd):
            vv=[x-y for x in range(bd[0][0],bd[0][1]+1) for y in range(bd[1][0],bd[1][1]+1) if (not j['optimal'][0] or x>=y) and (not j['optimal'][1] or x<=y) and (j['actions'][0]!=j['actions'][1] or x==y)]
            return [min(vv),max(vv)]
        assert gain(j['input'])==j['gain2'] and gain(z['bounds'])==z['gain2']
        assert j['gain2'][0]<=z['gain2'][0]<=z['gain2'][1]<=j['gain2'][1]
        if r.get('truth') is not None:
            v0=[r['truth'][str(m)] for m in j['actions']];assert all(x<=v1<=y for (x,y),v1 in zip(z['bounds'],v0));ct['late_truth']+=1
        ct[j['lane']]+=1
    assert ct['EARLY']==379 and ct['LATE']==384 and ct['late_truth']==377 and len(tasks)==10
    i=load('PAIR_INDEPENDENT_LOCK.json');assert sha(D/i['file'])==i['sha256'] and i['pair']==sha(D/'PAIR_LOCK.json')
    for r in raw(D/i['file']):assert r['expected']==tasks[r['id'],r['k']] and r['value'] in (-1,r['expected'])
    t=json.loads((T/'PAIR_TACTICAL.json').read_text());assert t['primary_lock_sha256']==sha(D/'PAIR_LOCK.json')
    assert t['source_sha256']==sha(T/'independent46_tactical.cpp') and t['binary_sha256']==sha(T/'independent46_tactical.so')
    assert {(r['id'],r['k']) for r in t['rows']}==set(tasks)
    assert all(r['expected']==tasks[r['id'],r['k']] and r['value'] in (-1,r['expected']) for r in t['rows'])
    ct['early_independent_confirmed']=t['counts']['confirmed'];ct['early_independent_censored']=t['counts']['censored']
    return dict(ct)
def audit_final():
    p=load('PLAN.json');fam={family(j['opening']) for j in p['jobs']};assert len(fam)==96
    for x in p['prior_locks']:
        q=ROOT/x['path'];assert sha(q)==x['sha256'];prev=json.loads(q.read_text());assert not fam&{family(j['opening']) for j in prev['jobs']}
    sys.path.insert(0,str(D.parent/'quartz_kinetic37_20261126'));import experiment as e
    rf={e.family(j['opening']) for j in p['rapfi_jobs']};assert len(rf)==64
    for x in p['rapfi_prior_locks']:assert sha(ROOT/x['path'])==x['sha256']
    for x in p['prior_locks']:
        prev=json.loads((ROOT/x['path']).read_text());assert not rf&{e.family(j['opening']) for j in prev.get('rapfi_jobs',[])}
    assert not rf&{e.family(j['opening']) for j in e.read('PLAN.json')['jobs']}
    modes=('NN_LOW','NN_FIXED','R_ARENA');ar={m:json.loads((T/('AUDIT_'+m+'.json')).read_text()) for m in modes}
    ex=C.defaultdict(list);s=load('SUMMARY.json');counts=C.Counter();total=C.Counter()
    for m,r in ar.items():
        for k in ('games','moves','nn'):total[k]+=r['counts'][k]
        counts.update(r['audit']);counts.update({'independent_'+k:v for k,v in r['independent'].items()})
        for x in r['natural_exact']:ex[x['a'],x['b']].extend(x['proofs'])
        for name,z in s['arena'][m].items():
            seeds=list(r['per_seed']) if name in ('all','Rapfi') else [str(k) for k in (101,102,103,105) if name=='old'] if name=='old' else [str(k) for k in (41,42,43)] if name=='new' else [name]
            c=sum((C.Counter(r['per_seed'][i]) for i in seeds),C.Counter())
            assert all(c[k]==vv for k,vv in z['counts'].items()),(m,name)
            score=Fraction(c['score2'],2*c['games']);assert abs(float(score)-z['metrics']['score'])<1e-15
            for k,l in [('nn','NN'),('cpu','CPU'),('wall','wall')]:
                v0=1-Fraction(c['Q'+k]*c['Bmoves'],c['B'+k]*c['Qmoves'])
                assert abs(float(v0)-z['metrics'][l+'_reduction'])<1e-14
                v0=1-Fraction(c['Q'+k],c['B'+k]);assert abs(float(v0)-z['metrics'][l+'_game_reduction'])<1e-14
    # Separate archived terminal-only exact vector solver on each natural exact fallback root.
    xx=[]
    if ex:
        binary=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic41_exact_backups/independent'
        assert sha(binary)=='a1e943f250eae145e10a12482fc575d11af08960dfd7ddf804c1f0c0ff14c7af'
        keys=list(ex);inp=''.join(f'{i} {a} {b}\n' for i,(a,b) in enumerate(keys))
        proc=subprocess.run([str(binary)],input=inp,text=True,capture_output=True,check=True,timeout=120)
        rs=list(map(json.loads,proc.stdout.splitlines()));assert len(rs)==len(keys)
        for r in rs:
            k=keys[r['id']]
            if r['complete']:
                for z in ex[k]:
                    vv=r['values'];assert all(l<=vv[str(m)]<=u for m,l,u in z['bounds'])
                    if z['cert'] is not None:assert vv[str(z['cert'])]==max(vv.values())
            xx.append(dict(a=k[0],b=k[1],**r))
    save('NATURAL_EXACT_AUDIT.json',dict(roots=len(xx),complete=sum(r['complete'] for r in xx),rows=xx))
    out=dict(modes={m:{k:v for k,v in x.items() if k!='natural_exact'} for m,x in ar.items()},total=dict(total),counts=dict(counts),pair=audit_pairs(),natural_exact=dict(roots=len(xx),complete=sum(r['complete'] for r in xx)),scope='independent coordinate/integer forcing, chronological votes, all raw cost and game aggregates; primary pair facts independently checked only within respective caps; bootstrap separately')
    save('AUDIT.json',out);print('FINAL_AUDIT',json.dumps({k:v for k,v in out.items() if k!='modes'}),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='final':audit_final()
    else:audit_mode(sys.argv[1])
