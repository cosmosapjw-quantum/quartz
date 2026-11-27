"""Independent integer/coordinate proof and chronological action/cost audit."""
import collections as C,gzip,json,pathlib,sys,importlib.util,subprocess,math
from fractions import Fraction
from k49_plan import D,ROOT,T,SEEDS,load,sha,family
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
H=('32','64','128')
def check_teacher(h,actions,bd,z,c):
    a,b=actions
    assert z['changed']==(a!=b)
    if a==b:
        assert z['teacher']==[None,None] and z['values'] is None and all(z['gains'][k]==0 for k in H)
        return
    vv={k:[] for k in H}
    for m,t in zip(actions,z['teacher']):
        lo,hi=bd[m]
        if lo==hi:
            assert t is None
            for k in H:vv[k].append(lo)
        else:
            assert t is not None and set(t)==set(H)
            prev=0;term=0
            for k in H:
                x=t[k]
                if x['terminal']:assert x['nn']==x['visits']==0
                else:
                    assert x['visits']==int(k) and x['nn']==1+x['visits']-x['terminal_leaves'] and x['nn']>=prev and x['terminal_leaves']>=term
                    assert 1<=x['leader_n']<=int(k) and abs(x['raw']+x['leader_w']/x['leader_n'])<1e-15
                    prev=x['nn'];term=x['terminal_leaves']
                vv[k].append(max(lo,min(hi,x['raw'])))
            c['teacher_calls']+=1;c['teacher_nn']+=t['128']['nn']
    assert vv==z['values']
    for k in H:assert z['gains'][k]==(vv[k][1]-vv[k][0])/2
def utility(h,r,bd,c):
    l=r['low'];u=r['high'];z=r['utility'];a,b=l['move'],u['move']
    check_snapshot(h,l,bd,20);check_snapshot(h,u,bd,32)
    assert all(l['votes'][i][0]==m and l['votes'][i][1]<=n for i,(m,n) in enumerate(u['votes']))
    assert r['physical_nn']==u['nn'] and r['nn']==l['nn'] and r['move']==a
    assert z['cost']==u['nn']-l['nn']>=0 and z['visits']==u['visits']-l['visits']>=0 and z['phase_wall_ns']==r['whole_ns']-r['low_wall_ns']>=0
    check_teacher(h,[a,b],bd,z,c)
    c['atlas_roots']+=1;c['extra_nn']+=z['cost'];CT['utility_rows']+=1
def audit_shards(arg):
    mode,files=arg;CT.clear();EX.clear();v.CT.clear()
    p=load('PLAN.json');n=15 if mode.startswith('R') else 7;source=mode in ('TRAIN','CAL','TEST');mask=load('ADMISSION.json')['mask']
    jobs={j['id']:j for j in p['rapfi_jobs' if n==15 else 'jobs'] if j['split']==(mode if source else 'ARENA')}
    ct=C.Counter();keys=[];out=C.defaultdict(C.Counter)
    for f in files:
        path=D/f['file'];assert sha(path)==f['sha256'];counts=C.Counter()
        for g in raw(path):
            j=g['job'];assert j==jobs[j['id']] and g['seed'] in ((0,) if n==15 else SEEDS)
            assert g['color'] in (j['color_order'][:1] if source else j['color_order']);keys.append((j['id'],g['seed'],g['color']))
            h=j['opening'].copy();nn=0;s=out[str(g['seed'])]
            for r in g['rows']:
                side=1-2*(len(h)%2);q=side==g['color'];a,b=v.orient(h,n);bd,g0=v.base(a,b,n)
                assert r['move'] in bd and r['ply']==len(h) and r['player']==side and r['arm']==('Q' if q else 'B')
                z=r.get('partial');assert bool(z)==bool(q and g0 is None)
                cap=(int(mode[1:]) if q else 128) if n==15 else 20 if q else 32
                if z:
                    bds=replay_proof(h,n,z);s['attempts']+=1;s['queries']+=z['cold']['receipt']['queries'];s['proof_wall']+=z['wall_ns']
                    if z['exact']:s['exact_attempt']+=1;s['exact_complete']+=z['exact']['complete'];s['exact_nodes']+=z['exact']['nodes']
                    if z['cert'] is not None:assert r['kind']=='PARTIAL47' and r['move']==z['cert'] and r['original_move'] is None and not r['replacement'];s['cert']+=1;vote(h,n,r,True,cap)
                    else:
                        if n==7 and (source or mode=='NN_ADAPT'):
                            l=r['low'];bn=int(l['distance']<=3)+2*int(l['original_move']!=l['prior_best'])+4*int(any(u==-1 for _,u in bds.values()))
                            assert bn==r['bin'] and r['extended']==(source or bool(mask>>bn&1))
                            if source:utility(h,r,bds,counts)
                            else:
                                check_snapshot(h,l,bds,20);cap=32 if r['extended'] else 20
                                assert r['physical_nn']==r['nn'] and r['mask']==mask
                                assert all(l['votes'][i][0]==m and l['votes'][i][1]<=x for i,(m,x) in enumerate(r['votes']))
                                if r['extended']:s['extensions']+=1;s['extra_nn']+=r['nn']-l['nn']
                        vote(h,n,r,True,cap,r['original_move'])
                        mv=v.changed(r['original_move'],r['votes'],bds,n);assert mv==r['move'] and r['replacement']==(mv!=r['original_move']);s['replacement']+=r['replacement']
                else:vote(h,n,r,q,cap)
                assert r['cpu_ns']>=0 and r['wall_ns']>=0
                if n==7:assert r['history']==h and r['lock']==q
                tag='Q' if q else 'B';s[tag+'moves']+=1
                for k,kk in [('nn','nn'),('cpu_ns','cpu'),('wall_ns','wall')]:s[tag+kk]+=r[k]
                nn+=r['nn'];counts['moves']+=1;counts['common']+=r['kind']=='CERTIFICATE';h.append(r['move'])
            assert h==g['history'];a,b,w=v.replay(h,n);assert w or (a|b)==v.G[n][1]
            assert w==g['winner'] and g['score2']==1+g['color']*w and nn==g['nn']
            s['games']+=1;s['score2']+=g['score2'];s['wins']+=g['score2']==2;s['draws']+=g['score2']==1;s['losses']+=g['score2']==0
            counts['nn']+=nn;counts['games']+=1
        assert all(counts[k]==x for k,x in f['counts'].items()) and all(f['counts'].get(k,0)==x for k,x in counts.items()),(mode,counts,f['counts'])
        ct.update(counts)
    return dict(ct),{k:dict(s) for k,s in out.items()},keys,dict(CT),dict(v.CT),[(list(k),r) for k,r in EX.items()]
def audit_mode(mode):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    lk=load(mode+'_LOCK.json');assert lk['plan']==sha(D/'PLAN.json') and lk['source']==sha(D/'experiment49.py') and lk['bridge']==sha(D/'native49.py') and lk['binary']==sha(T/'mcts_demo')
    fs=lk['files'];ct=C.Counter();tot=C.defaultdict(C.Counter);keys=[];ac=C.Counter();ic=C.Counter();ex=C.defaultdict(list)
    workers=4 if mode.startswith('R') else 2
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        for c,o,k,x,y,z in pool.map(audit_shards,[(mode,fs[i::workers]) for i in range(workers)]):
            ct.update(c);keys.extend(k);ac.update(x);ic.update(y)
            for s,t in o.items():tot[s].update(t)
            for k,r in z:ex[tuple(k)].extend(r)
    assert dict(ct)==lk['counts'] and len(set(keys))==len(keys)==ct['games']
    p=load('PLAN.json');n=15 if mode.startswith('R') else 7;source=mode in ('TRAIN','CAL','TEST')
    assert set(keys)=={(j['id'],s,c) for j in p['rapfi_jobs' if n==15 else 'jobs'] if j['split']==(mode if source else 'ARENA') for s in ((0,) if n==15 else SEEDS) for c in (j['color_order'][:1] if source else j['color_order'])}
    save('AUDIT_'+mode+'.json',dict(counts=dict(ct),per_seed={s:dict(t) for s,t in tot.items()},audit=dict(ac),independent=dict(ic),natural_exact=[dict(a=k[0],b=k[1],proofs=r) for k,r in ex.items()]))
    print('AUDITED',mode,dict(ct),dict(ac),dict(ic),'exact_unique',len(ex),flush=True)
def audit_model():
    f=load('FROZEN.json');ad=load('ADMISSION.json');assert f['train_lock']==sha(D/'TRAIN_LOCK.json') and ad['frozen_sha']==sha(D/'FROZEN.json')
    rr={}
    for m in ('TRAIN','CAL','TEST'):
        rr[m]=[(g['seed'],r['bin'],r['utility']['gains'],r['utility']['cost']) for fi in load(m+'_LOCK.json')['files'] for g in raw(D/fi['file']) for r in g['rows'] if 'utility' in r]
    x=rr['TRAIN'];cost=[sum(c for seed,z,g,c in x if z==i) for i in range(8)];cap=sum(cost)//4
    groups=[x]+[[r for r in x if r[0]!=s] for s in SEEDS]
    allowed=[]
    for m in range(256):
        if sum(cost[i] for i in range(8) if m>>i&1)>cap:continue
        obj=min(math.fsum(g[h] for seed,z,g,c in group if m>>z&1)/len(group) for group in groups for h in H)
        allowed.append((obj,-m.bit_count(),-m))
    obj,_,neg=max(allowed);mask=-neg
    assert mask==f['mask'] and cap==f['budget'] and abs(obj-f['worst_mean'])<1e-14
    result={}
    for m,y in rr.items():
        c=sum(c for seed,z,g,c in y);sc=sum(c for seed,z,g,c in y if mask>>z&1);p=sc/c if c else 0.
        a={}
        for h in H:
            g=math.fsum(g[h] for seed,z,g,c in y);sg=math.fsum(g[h] for seed,z,g,c in y if mask>>z&1);const=p*g
            a[h]=dict(total_gain=g,selected_gain=sg,constant_gain=const,excess=sg-const)
        admitted=bool(mask) and all(x['selected_gain']>0 and x['excess']>0 for x in a.values())
        result[m]=dict(n=len(y),cost=c,selected_cost=sc,horizons=a,admitted=admitted)
        z=f['train'] if m=='TRAIN' else ad['cal'] if m=='CAL' else load('TEST_EVAL.json')
        assert z['n']==len(y) and z['cost']==c and z['selected_cost']==sc and z['admitted']==admitted
        for h in H:
            for k,v0 in a[h].items():assert abs(z['horizons'][h][k]-v0)<1e-14
        if m=='CAL':assert ad['admitted']==admitted and ad['mask']==(mask if admitted else 0)
    return dict(mask=mask,objective=obj,results=result)
def audit_historical():
    old=D.parent/'quartz_kinetic48_voc_20261127';lk=load('HIST_LOCK.json');lookup=C.defaultdict(list)
    for m,digest in lk['historical_locks'].items():
        p=old/(m+'_LOCK.json');assert sha(p)==digest
        for fi in json.loads(p.read_text())['files']:
            p=old/fi['file'];assert sha(p)==fi['sha256']
            for g in raw(p):
                for r in g['rows']:
                    if 'utility' in r:lookup[m,g['seed'],g['job']['id'],tuple(r['history'])].append(r)
    c=C.Counter()
    for fi in lk['files']:
        p=D/fi['file'];assert sha(p)==fi['sha256']
        for r in raw(p):
            key=r['split'],r['seed'],r['job'],tuple(r['history']);q=lookup[key].pop()
            z=r['utility'];u=q['utility'];bd={m:(l,v) for m,l,v in q['partial']['bounds']}
            assert r['actions']==[q['low']['move'],q['high']['move']] and r['bin']==q['bin'] and z['cost']==u['cost']
            check_teacher(r['history'],r['actions'],bd,z,c)
            assert z['gains']['32']==u['gain'] and z['changed']==u['changed']
            for t,v0 in zip(z['teacher'],u['teacher']):
                if t is None:assert v0 is None
                else:
                    assert {k:v for k,v in t['32'].items() if k not in ('cpu_ns','wall_ns')}=={k:v for k,v in v0.items() if k not in ('cpu_ns','wall_ns')};c['H32_receipts']+=1
            c['rows']+=1;c['changed']+=z['changed']
    assert not any(lookup.values()) and all(c[k]==lk['counts'][k] for k in c)
    return dict(c)
def final():
    p=load('PLAN.json');fam={family(j['opening']) for j in p['jobs']};assert len(fam)==144
    for x in p['prior_locks']:
        q=ROOT/x['path'];assert sha(q)==x['sha256'];prev=json.loads(q.read_text());assert not fam&{family(j['opening']) for j in prev['jobs']}
    sys.path.insert(0,str(D.parent/'quartz_kinetic37_20261126'));import experiment as e
    rf={e.family(j['opening']) for j in p['rapfi_jobs']};assert len(rf)==128
    for x in p['rapfi_prior_locks']:assert sha(ROOT/x['path'])==x['sha256']
    for x in p['prior_locks']:
        prev=json.loads((ROOT/x['path']).read_text());assert not rf&{e.family(j['opening']) for j in prev.get('rapfi_jobs',[])}
    assert not rf&{e.family(j['opening']) for j in e.read('PLAN.json')['jobs']}
    s=load('SUMMARY.json');modes=['TRAIN','CAL','TEST']+list(s['arena']);ar={m:json.loads((T/('AUDIT_'+m+'.json')).read_text()) for m in modes};ex=C.defaultdict(list);counts=C.Counter();total=C.Counter()
    for m,r in ar.items():
        total.update(r['counts']);counts.update(r['audit']);counts.update({'independent_'+k:v for k,v in r['independent'].items()})
        for x in r['natural_exact']:ex[x['a'],x['b']].extend(x['proofs'])
        if m not in s['arena']:continue
        for name,z in s['arena'][m].items():
            seeds=list(r['per_seed']) if name in ('all','Rapfi') else [str(k) for k in (101,102,103,105)] if name=='old' else [str(k) for k in (41,42,43)] if name=='new' else [name]
            c=sum((C.Counter(r['per_seed'][i]) for i in seeds),C.Counter())
            assert all(c[k]==vv for k,vv in z['counts'].items()),(m,name,c,z['counts'])
            assert abs(float(Fraction(c['score2'],2*c['games']))-z['metrics']['score'])<1e-15
            for k,l in [('nn','NN'),('cpu','CPU'),('wall','wall')]:
                val=1-Fraction(c['Q'+k]*c['Bmoves'],c['B'+k]*c['Qmoves']);assert abs(float(val)-z['metrics'][l+'_reduction'])<1e-14
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
    out=dict(modes={m:{k:v for k,v in x.items() if k!='natural_exact'} for m,x in ar.items()},total=dict(total),physical_nn=total['nn']+total['extra_nn']+total['teacher_nn'],counts=dict(counts),model=audit_model(),historical=audit_historical(),natural_exact=dict(roots=len(xx),complete=sum(r['complete'] for r in xx)),scope='independent integer forcing, chronological votes and gain/cost receipt arithmetic; teacher semantic receipts and H32 historical reproduction; H64/128 cold parity separate')
    # total extra_nn only source atlas (live extension already counted in nn)
    save('AUDIT.json',out);print('FINAL_AUDIT',json.dumps({k:v for k,v in out.items() if k!='modes'}),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='final':final()
    else:audit_mode(sys.argv[1])
