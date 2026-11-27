"""Independent integer/coordinate proof and chronological action/cost audit."""
import collections as C,gzip,json,pathlib,sys,importlib.util,subprocess,math
from fractions import Fraction
from k50_plan import D,ROOT,T,SEEDS,load,sha,family
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
def audit_game(g,j,n,policy,counts):
    assert g['job']==j and g['color'] in j['color_order'] and g['seed'] in ((0,) if n==15 else SEEDS)
    h=j['opening'].copy();eligible=0;event=None;nn=0;c=C.Counter();k=j.get('ordinal')
    for i,r in enumerate(g['rows']):
        side=1-2*(len(h)%2);q=side==g['color'];a,b=v.orient(h,n);bd,g0=v.base(a,b,n)
        assert r['move'] in bd and r['ply']==len(h) and r['player']==side and r['arm']==('Q' if q else 'B')
        z=r.get('partial');assert bool(z)==bool(q and g0 is None)
        cap=(96 if q else 128) if n==15 else 20 if q else 32
        if z:
            bds=replay_proof(h,n,z);c['queries']+=z['cold']['receipt']['queries'];c['cert']+=z['cert'] is not None
            if z['cert'] is not None:
                assert r['kind']=='PARTIAL47' and r['move']==z['cert'] and r['original_move'] is None and not r['replacement']
                vote(h,n,r,True,cap)
            else:
                if n==7 and eligible<k:
                    l=r['low'];bn=int(l['distance']<=3)+2*int(l['original_move']!=l['prior_best'])+4*int(any(u==-1 for _,u in bds.values()))
                    assert bn==r['bin'];eligible+=1;assert r['eligible_index']==eligible
                    ext=policy=='EXTEND' and eligible==k
                    assert r['extended']==ext and r['mask']==(255 if ext else 0) and r['physical_nn']==r['nn']
                    check_snapshot(h,l,bds,20);cap=32 if ext else 20
                    assert all(l['votes'][ii][0]==m and l['votes'][ii][1]<=x for ii,(m,x) in enumerate(r['votes']))
                    if eligible==k:event=i
                    if not ext:
                        assert all(r[f]==l[f] for f in ('move','kind','nn','visits','votes','distance','original_move','replacement','cap'))
                    else:CT['extensions']+=1;CT['extra_local_nn']+=r['nn']-l['nn']
                else:assert not r.get('low')
                vote(h,n,r,True,cap,r['original_move'])
                mv=v.changed(r['original_move'],r['votes'],bds,n);assert mv==r['move'] and r['replacement']==(mv!=r['original_move']);c['replacement']+=r['replacement']
        else:
            assert not r.get('low')
            vote(h,n,r,q,cap)
        assert r['cpu_ns']>=0 and r['wall_ns']>=0
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
    # Independent arithmetic and chronological checks; never call primary pair().
    assert ec==et
    k=c['job']['ordinal'];reached=ec is not None;local=0;postQ=postB=0;bin=None;changed=False;actions=None;ply=None
    if reached:
        r,s=c['rows'][ec],t['rows'][et]
        assert c['history'][:r['ply']]==t['history'][:s['ply']]==r['history']==s['history']
        for x,y in zip(c['rows'][:ec],t['rows'][:et],strict=True):
            assert (x['move'],x['nn'],x['visits'],x['votes'])==(y['move'],y['nn'],y['visits'],y['votes'])
        assert r['low']==s['low'] and r['bin']==s['bin'] and r['eligible_index']==s['eligible_index']==k
        local=s['nn']-r['nn'];assert local>=0;changed=r['move']!=s['move'];actions=[r['move'],s['move']];bin=r['bin'];ply=r['ply']
        postQ=sum(x['nn'] for x in t['rows'][et+1:] if x['arm']=='Q')-sum(x['nn'] for x in c['rows'][ec+1:] if x['arm']=='Q')
        postB=sum(x['nn'] for x in t['rows'][et+1:] if x['arm']=='B')-sum(x['nn'] for x in c['rows'][ec+1:] if x['arm']=='B')
    dq=tt['Qnn']-cc['Qnn'];db=tt['Bnn']-cc['Bnn'];delta=t['score2']-c['score2']
    assert dq==local+postQ and db==postB
    if not changed:assert c['history']==t['history'] and delta==postQ==postB==0
    independent=dict(job=c['job']['id'],split=c['job']['split'],seed=c['seed'],color=c['color'],ordinal=k,reached=reached,event=ec,ply=ply,bin=bin,actions=actions,changed=changed,score2=[c['score2'],t['score2']],delta2=delta,local_nn=local,downstream_Q=postQ,downstream_B=postB,delta_Q=dq,delta_B=db,delta_total=dq+db,same_history=c['history']==t['history'])
    assert independent==z,(independent,z)
    return independent
def audit_shards(arg):
    mode,files=arg;CT.clear();EX.clear();v.CT.clear()
    p=load('PLAN.json');n=15 if mode=='R96' else 7
    jobs={j['id']:j for j in p['rapfi_jobs' if n==15 else 'jobs'] if j['split']==('ARENA' if n==15 else mode)}
    ct=C.Counter();keys=[];out=C.defaultdict(C.Counter);pair_tot=C.Counter()
    for f in files:
        path=D/f['file'];assert sha(path)==f['sha256'];counts=C.Counter()
        for x in raw(path):
            if n==15:
                j=jobs[x['job']['id']];keys.append((j['id'],0,x['color']))
                c,e=audit_game(x,j,n,None,counts);out['all'].update(c);out[j['stratum']].update(c)
            else:
                c,t,z=x['control'],x['extend'],x['pair'];j=jobs[c['job']['id']]
                assert c['seed']==t['seed'] and c['color']==t['color'];keys.append((j['id'],c['seed'],c['color']))
                cc,ec=audit_game(c,j,n,'CONTROL',counts);tt,et=audit_game(t,j,n,'EXTEND',counts)
                r=compare(c,t,z,ec,et,cc,tt)
                counts['pairs']+=1;counts['reached']+=r['reached'];counts['changed']+=r['changed'];counts['local_nn']+=r['local_nn'];counts['delta2']+=r['delta2']
                for k in ('delta_Q','delta_B','delta_total','downstream_Q','downstream_B','same_history'):pair_tot[k]+=r[k]
                panel='old' if c['seed'] in (101,102,103,105) else 'new'
                for arm,x0 in [('control',cc),('extend',tt)]:
                    out[arm+'_all'].update(x0);out[arm+'_'+panel].update(x0)
        assert all(counts[k]==x for k,x in f['counts'].items()) and all(f['counts'].get(k,0)==x for k,x in counts.items()),(mode,counts,f['counts'])
        ct.update(counts)
    return dict(ct),{k:dict(v0) for k,v0 in out.items()},keys,dict(CT),dict(v.CT),dict(pair_tot),[(list(k),r) for k,r in EX.items()]
def audit_mode(mode):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    lk=load(mode+'_LOCK.json');assert lk['plan']==sha(D/'PLAN.json') and lk['source']==sha(D/'experiment50.py') and lk['bridge']==sha(D/'native50.py') and lk['binary']==sha(T/'mcts_demo')
    fs=lk['files'];ct=C.Counter();tot=C.defaultdict(C.Counter);keys=[];ac=C.Counter();ic=C.Counter();pc=C.Counter();ex=C.defaultdict(list)
    workers=4 if mode=='R96' else 2
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        for c,o,k,x,y,z,e in pool.map(audit_shards,[(mode,fs[i::workers]) for i in range(workers)]):
            ct.update(c);keys.extend(k);ac.update(x);ic.update(y);pc.update(z)
            for s,t in o.items():tot[s].update(t)
            for k,r in e:ex[tuple(k)].extend(r)
    assert dict(ct)==lk['counts'] and len(set(keys))==len(keys)==(ct['games'] if mode=='R96' else ct['pairs'])
    p=load('PLAN.json');nn=mode!='R96'
    assert set(keys)=={(j['id'],s,c) for j in p['jobs' if nn else 'rapfi_jobs'] if j['split']==(mode if nn else 'ARENA') for s in (SEEDS if nn else (0,)) for c in j['color_order']}
    if nn:assert ac['extensions']==ct['reached'] and ac['extra_local_nn']==ct['local_nn']
    save('AUDIT_'+mode+'.json',dict(counts=dict(ct),panels={s:dict(t) for s,t in tot.items()},audit=dict(ac),independent=dict(ic),pair=dict(pc),natural_exact=[dict(a=k[0],b=k[1],proofs=r) for k,r in ex.items()]))
    print('AUDITED',mode,dict(ct),dict(ac),dict(ic),'exact_unique',len(ex),flush=True)
def final():
    p=load('PLAN.json');fam={family(j['opening']) for j in p['jobs']};assert len(fam)==96
    for x in p['prior_locks']:
        q=ROOT/x['path'];assert sha(q)==x['sha256'];prev=json.loads(q.read_text());assert not fam&{family(j['opening']) for j in prev['jobs']}
    sys.path.insert(0,str(D.parent/'quartz_kinetic37_20261126'));import experiment as e
    rf={e.family(j['opening']) for j in p['rapfi_jobs']};assert len(rf)==128
    for x in p['rapfi_prior_locks']:assert sha(ROOT/x['path'])==x['sha256']
    for x in p['prior_locks']:
        prev=json.loads((ROOT/x['path']).read_text());assert not rf&{e.family(j['opening']) for j in prev.get('rapfi_jobs',[])}
    assert not rf&{e.family(j['opening']) for j in e.read('PLAN.json')['jobs']}
    for m in ('TRAIN','CAL','TEST'):
        jobs=[j for j in p['jobs'] if j['split']==m];assert sum(j['ordinal']==1 for j in jobs)==len(jobs)//2
    for j in p['rapfi_jobs']:
        rc=[divmod(m,15) for m in j['opening']]
        if j['stratum']=='compact':assert max(r for r,c in rc)-min(r for r,c in rc)<=4 and max(c for r,c in rc)-min(c for r,c in rc)<=4
        else:assert all(not (3<=r<=11 and 3<=c<=11) for r,c in rc)
    s=load('SUMMARY.json');total=C.Counter();counts=C.Counter();pairs=C.Counter();ex=C.defaultdict(list);modes=('TRAIN','CAL','TEST','R96');ar={m:json.loads((T/('AUDIT_'+m+'.json')).read_text()) for m in modes}
    def check_metric(c,z):
        assert all(c.get(k,0)==x for k,x in z['counts'].items())
        assert abs(float(Fraction(c['score2'],2*c['games']))-z['metrics']['score'])<1e-15
        for k,l in [('nn','NN'),('cpu','CPU'),('wall','wall')]:
            val=1-Fraction(c['Q'+k]*c['Bmoves'],c['B'+k]*c['Qmoves']);assert abs(float(val)-z['metrics'][l+'_reduction'])<1e-14
    for m,r in ar.items():
        total.update(r['counts']);counts.update(r['audit']);counts.update({'independent_'+k:v0 for k,v0 in r['independent'].items()});pairs.update(r['pair'])
        for x in r['natural_exact']:ex[x['a'],x['b']].extend(x['proofs'])
        if m=='R96':
            for p0,z in s['rapfi'].items():check_metric(r['panels'][p0],z)
        else:
            for p0,arms in s['nn'][m].items():
                for arm,z in arms.items():check_metric(r['panels'][arm+'_'+p0],z)
            for k,x in r['pair'].items():assert s['paired'][m]['all']['counts'][k]==x
    rr={m:[x['pair'] for fi in load(m+'_LOCK.json')['files'] for x in raw(D/fi['file'])] for m in modes if m!='R96'}
    train=rr['TRAIN'];cc=[sum(r['local_nn'] for r in train if r['bin']==z) for z in range(8)];gg=[sum(r['delta2'] for r in train if r['bin']==z) for z in range(8)]
    cap=sum(cc)//4
    v0=[(sum(gg[z] for z in range(8) if m>>z&1),-m.bit_count(),-m) for m in range(256) if sum(cc[z] for z in range(8) if m>>z&1)<=cap]
    gain,_,neg=max(v0);mask=-neg;f=load('FROZEN.json');ad=load('ADMISSION.json')
    assert f['mask']==mask and f['selected_delta2']==gain and f['budget']==cap and f['train_lock']==sha(D/'TRAIN_LOCK.json') and ad['frozen_sha256']==sha(D/'FROZEN.json')
    for m,y in rr.items():
        sel=[r for r in y if r['bin'] is not None and mask>>r['bin']&1]
        sc=sum(r['local_nn'] for r in sel);c=sum(r['local_nn'] for r in y);sg=sum(r['delta2'] for r in sel);g=sum(r['delta2'] for r in y);p0=Fraction(sc,c) if c else 0
        z=s['replay'][m];assert z['selected']['delta2']==sg and z['selected']['local_nn']==sc and z['total']['delta2']==g and z['total']['local_nn']==c
        assert abs(z['constant_gain']-float(p0*g/2))<1e-14 and z['admitted']==(bool(mask) and sg>0 and sg>p0*g)
        if m=='CAL':assert ad['admitted']==z['admitted']
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
    assert total['games']==2944 and total['pairs']==1344
    out=dict(total=dict(total),counts=dict(counts),pair=dict(pairs),mask=mask,admitted=ad['admitted'],natural_exact=dict(roots=len(xx),complete=sum(r['complete'] for r in xx)),modes={m:{k:v0 for k,v0 in x.items() if k!='natural_exact'} for m,x in ar.items()},scope='independent coordinate/bitmask forcing, chronology, pre-event ordinal, low snapshot/votes, exact paired cost decomposition and replay model arithmetic')
    save('AUDIT.json',out);print('FINAL_AUDIT',json.dumps({k:v0 for k,v0 in out.items() if k!='modes'}),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='final':final()
    else:audit_mode(sys.argv[1])
