"""Independent rule/forcing replay, root/cost receipts, prospective family and exact oracle audits."""
import collections as C,gzip,json,pathlib,sys,importlib.util,subprocess,math
from fractions import Fraction
from k47_plan import D,ROOT,T,SEEDS,load,sha,family
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
    assert z['wall_ns']>=z['cold']['wall_ns']+(ex['wall_ns'] if ex else 0)
    CT['proof_queries']+=r['queries'];CT['proof_receipts']+=1
    return bd
def vote(h,n,r,lock,pre=None,cap=32):
    if n==15:
        v.vote(h,n,dict(r,kind='PARTIAL43') if r['kind']=='PARTIAL47' else r,lock,pre);return
    a,b=v.orient(h,n);bd,g=v.base(a,b,n);E=list(bd)
    assert r['cap']==cap and r['lock']==lock and r['nn']>=0 and r['cpu_ns']>=0 and r['wall_ns']>=0
    if g is not None:
        assert r['kind']=='CERTIFICATE' and r['move']==g and r['nn']==r['visits']==0 and not r['votes'];CT['common']+=1;return
    if r['kind']=='PARTIAL47':
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
                    s['attempts']+=1;s['queries']+=z['cold']['receipt']['queries'];s['proof_wall']+=z['wall_ns']
                    if z['exact']:s['exact_attempt']+=1;s['exact_complete']+=z['exact']['complete'];s['exact_nodes']+=z['exact']['nodes']
                    if z['cert'] is not None:
                        assert r['kind']=='PARTIAL47' and r['move']==z['cert'] and r['original_move'] is None and not r['replacement'];s['cert']+=1;vote(h,n,r,True,cap=cap)
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
    assert lk['plan']==sha(D/'PLAN.json') and lk['source']==sha(D/'arena47.py') and lk['bridge']==sha(D/'native47.py') and lk['binary']==sha(T/'mcts_demo')
    fs=lk['files'];ct=C.Counter();tot=C.defaultdict(C.Counter);keys=[];ac=C.Counter();ic=C.Counter();ex=C.defaultdict(list)
    # For low while another arena runs, limited to two workers; others four.
    workers=2 if mode=='NN_LOW' else 4
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        for c,o,k,x,y,z in pool.map(audit_shards,[(mode,fs[i::workers]) for i in range(workers)]):
            ct.update(c);keys.extend(k);ac.update(x);ic.update(y)
            for s,t in o.items():tot[s].update(t)
            for k,r in z:ex[tuple(k)].extend(r)
    assert dict(ct)==lk['counts'] and len(set(keys))==len(keys)==(128 if mode=='R_ARENA' else 672)
    p=load('PLAN.json');n=15 if mode=='R_ARENA' else 7
    assert set(keys)=={(j['id'],s,c) for j in p['rapfi_jobs' if n==15 else 'jobs'] for s in ((0,) if n==15 else SEEDS) for c in j['color_order']}
    save('AUDIT_'+mode+'.json',dict(counts=dict(ct),per_seed={s:dict(t) for s,t in tot.items()},audit=dict(ac),independent=dict(ic),natural_exact=[dict(a=k[0],b=k[1],proofs=r) for k,r in ex.items()]))
    print('AUDITED',mode,dict(ct),dict(ac),dict(ic),'exact_unique',len(ex),flush=True)
def audit_risk():
    r=load('RISK_SUMMARY.json');p=D/'RISK.jsonl.gz';assert sha(p)==r['raw_sha256']
    for mode,digest in r['source_locks'].items():assert sha(K45/(mode+'_LOCK.json'))==digest
    source=C.defaultdict(list);ct=C.Counter()
    for mode in ('TRAIN','NN_CAL','NN_TEST'):
        lock=json.loads((K45/(mode+'_LOCK.json')).read_text())
        for f in lock['files']:
            path=K45/f['file'];assert sha(path)==f['sha256']
            for row in raw(path):
                if mode=='TRAIN':xs=[row];split='TRAIN';seed=row['seed'];job=row['job']
                else:xs=row['atlas'];split=mode[3:];seed=row['seed'];job=row['job']['id']
                for a in xs:
                    ct[split]+=1
                    if a['proof']['cert'] is not None:continue
                    k=(split,seed,job,tuple(a['history']));source[k].append(a)
    assert dict(ct)==r['source_counts']
    records=list(raw(p));assert len(records)==r['counts']['roots']
    stats=C.Counter();bins=C.defaultdict(list)
    for x in records:
        k=(x['split'],x['seed'],x['job'],tuple(x['history']));assert source[k]
        a=source[k].pop()
        bd={m:(l,u) for m,l,u in a['proof']['bounds']};low=a['L']['20'];m,n=[a['options'][str(k)]['move'] for k in (20,32)]
        assert x['actions']==[m,n] and x['bounds']==[list(bd[m]),list(bd[n])]
        assert x['z']==int(low['distance']<=3)+2*int(low['move']!=low['prior_best'])+4*int(any(u==-1 for l,u in bd.values()))
        if m==n:
            assert x['teacher']==[None,None] and x['values'] is None and x['gain']==0
        else:
            y=[]
            for z,t in zip(x['bounds'],x['teacher']):
                l,u=z
                if l==u:assert t is None;val=l
                else:
                    assert t is not None
                    if t['terminal']:assert t['nn']==0 and t['visits']==0
                    else:
                        assert t['nn']==33-t['terminal_leaves'] and t['visits']==32 and 1<=t['leader_n']<=32
                        assert abs(t['raw']+t['leader_w']/t['leader_n'])<1e-15
                    val=max(l,min(u,t['raw']))
                    stats['teacher_calls']+=1;stats['nn']+=t['nn'];stats['cpu_ns']+=t['cpu_ns']
                y.append(val)
            assert y==x['values'] and x['gain']==(y[0]-y[1])/2
        assert x['label']==int(x['gain']<0);stats['negative']+=x['label'];stats['roots']+=1
        bins[x['split']].append(x)
    assert not any(source.values()) and dict(stats)==r['counts']
    train=bins['TRAIN'];prob={str(z):float(Fraction(sum(x['label'] for x in train if x['z']==z)+1,sum(x['z']==z for x in train)+2)) for z in range(8)}
    base=float(Fraction(sum(x['label'] for x in train)+1,len(train)+2));assert prob==r['probabilities'] and base==r['constant']
    for split,rr in bins.items():
        z=r['results'][split];assert len(rr)==z['n']
        for field,pred in [('brier',lambda x:prob[str(x['z'])]),('constant_brier',lambda x:base)]:
            assert abs(math.fsum((pred(x)-x['label'])**2 for x in rr)/len(rr)-z[field])<1e-13
        assert sum(x['label'] for x in rr)==z['counts']['negative']
    return dict(stats)
def audit_parity():
    n=load('NN_PARITY_LOCK.json');p=D/n['file'];assert sha(p)==n['sha256'];ct=C.Counter()
    for x in raw(p):
        a=x['native'];b=x['reference']
        for k in ('move','proof','cap','lock','nn','visits','terminal_leaves','votes','distance','prior_best'):assert a[k]==b[k]
        ct['NN_requests']+=1;ct['NN_rows']+=a['nn'];assert x['max_eval_error']<=2e-15
    r=load('R_PARITY_LOCK.json');p=D/r['file'];assert sha(p)==r['sha256']
    for x in raw(p):
        a=x['native'];b=x['reference']
        for k in ('move','nn','visits','kind','votes','original_move','replacement'):assert a[k]==b[k]
        ct['Rapfi_requests']+=1;ct['Rapfi_rows']+=a['nn']
    q=load('PROOF_PARITY_LOCK.json');p=D/q['file'];assert sha(p)==q['sha256']
    for x in raw(p):
        ct['proof_requests']+=1
        if x['lane']=='late':
            a=x['exact'];ct['late_complete']+=a['complete']
            if x['truth'] is not None and a['complete']:
                vv=x['truth'];assert all(l<=vv[str(m)]<=u for m,l,u in a['bounds'])
                assert vv[str(a['best'])]==max(vv.values());ct['late_truth']+=1
    assert ct['NN_requests']==112 and ct['Rapfi_requests']==96 and ct['proof_requests']==2873 and ct['late_complete']==377
    return dict(ct)
def audit_final():
    p=load('PLAN.json');fam={family(j['opening']) for j in p['jobs']};assert len(fam)==48
    for x in p['prior_locks']:
        q=ROOT/x['path'];assert sha(q)==x['sha256'];prev=json.loads(q.read_text());assert not fam&{family(j['opening']) for j in prev['jobs']}
    sys.path.insert(0,str(D.parent/'quartz_kinetic37_20261126'));import experiment as e
    rf={e.family(j['opening']) for j in p['rapfi_jobs']};assert len(rf)==64
    for x in p['rapfi_prior_locks']:assert sha(ROOT/x['path'])==x['sha256']
    for x in p['prior_locks']:
        prev=json.loads((ROOT/x['path']).read_text());assert not rf&{e.family(j['opening']) for j in prev.get('rapfi_jobs',[])}
    assert not rf&{e.family(j['opening']) for j in e.read('PLAN.json')['jobs']}
    modes=('NN_LOW','R_ARENA');ar={m:json.loads((T/('AUDIT_'+m+'.json')).read_text()) for m in modes}
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
    out=dict(modes={m:{k:v for k,v in x.items() if k!='natural_exact'} for m,x in ar.items()},total=dict(total),counts=dict(counts),risk=audit_risk(),parity=audit_parity(),natural_exact=dict(roots=len(xx),complete=sum(r['complete'] for r in xx)),scope='independent coordinate/integer forcing, chronological votes, all raw cost and game aggregates; independent risk receipt and proof parity archive audit; bootstrap separately')
    save('AUDIT.json',out);print('FINAL_AUDIT',json.dumps({k:v for k,v in out.items() if k!='modes'}),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='final':audit_final()
    else:audit_mode(sys.argv[1])
