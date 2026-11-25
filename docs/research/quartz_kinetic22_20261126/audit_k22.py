"""Independent raw-data readback using grid rules and array algebra, not causal.py."""
import pathlib,subprocess as SP,hashlib,json,gzip,random,collections as C,math,itertools,importlib.util,statistics as S
import numpy as np
D=pathlib.Path(__file__).resolve().parent;R=D.parents[2];P=D.parent
def read(n):return json.loads((D/n).read_text())
def lines(n):
    with gzip.open(D/n,'rt') as f:return [json.loads(x) for x in f]
def sha(x):return hashlib.sha256(x).hexdigest()
def fs(n):return sha((D/n).read_bytes())
def git(*a):return SP.check_output(['git','-C',str(R),*a])
def near(a,b):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-11)
def load(p,n):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
g=load(P/'quartz_kinetic17_20261126/audit_k17.py','grid17')
plan=read('PLAN.json');co=read('COHORT.json');f=read('FROZEN.json');su=read('SUMMARY.json')
for l in plan['K20_K21_locks']+plan['history_locks']+list(plan['prior_locks'].values()):assert sha(git('show',l['commit']+':'+l['path']))==l['sha256']
assert fs('PLAN.json')==co['plan'] and fs('SOURCE.jsonl.gz')==co['source']
assert fs('COHORT.json')==f['cohort'] and fs('TRAIN_CAL_ATLAS.jsonl.gz')==f['trace'] and fs('TRAIN_CAL_LABELS.json')==f['labels'] and fs('causal.py')==f['code']
freeze='4cc714c3337c38fd5c1b8d8dedf514eef7917b0e';prefix=str(D.relative_to(R))+'/'
assert sha(git('show',freeze+':'+prefix+'FROZEN.json'))==fs('FROZEN.json')
for n in ('TEST_ATLAS.jsonl.gz','TEST_LABELS.json'):
    assert SP.run(['git','-C',str(R),'cat-file','-e',freeze+':'+prefix+n],capture_output=True).returncode!=0
assert read('TEST_LOCK.json')['freeze']==fs('FROZEN.json')
# Independent selected roots / source trajectory audit (not rerunning each recorded neural choice).
roots={r['id']:r for r in co['roots']};families=set();boards=set()
for l in plan['history_locks']+plan['K20_K21_locks']:
    if not l['path'].endswith('COHORT.json'):continue
    obj=json.loads(git('show',l['commit']+':'+l['path']))
    for r in obj.get('roots',[]):
        h=r.get('history','')
        if isinstance(h,str) and h and all(c in '1234567' for c in h):boards.add(g.boardkey(g.replay(h)));families.add(g.canonical(h[:8]))
    for j in obj.get('jobs',[]):
        if isinstance(j.get('family'),str):families.add(j['family'])
for l in plan['K20_K21_locks']:
    if l['path'].endswith('PLAN.json'):families.update(j['family'] for j in json.loads(git('show',l['commit']+':'+l['path']))['jobs'])
ss=lines('SOURCE.jsonl.gz');assert len(ss)==720;counts=C.Counter();seen=set();nn=plies=0
for j,t in zip(plan['jobs'],ss):
    assert t['job']==j and j['family']==g.canonical(j['opening']) and j['family'] not in seen and j['family'] not in families
    assert j['group']==('TRAIN' if j['id']<360 else 'CAL' if j['id']<540 else 'TEST')
    seen.add(j['family']);h=j['opening'];b=g.replay(h);rng=random.Random(j['seed'])
    for step in t['steps']:
        side=len(h)%2+1;assert not g.won(b,1) and not g.won(b,2)
        legal=[i for i,c in enumerate(b) if len(c)<6]
        def wins(p):
            out=[]
            for i in legal:
                b[i].append(p)
                if g.won(b,p):out.append(i)
                b[i].pop()
            return out
        own,opp=wins(side),wins(3-side);m=step['move'];assert m in legal
        if own:assert (m,step['kind'])==(own[0],'WIN')
        elif len(opp)==1:assert (m,step['kind'])==(opp[0],'BLOCK')
        elif len(legal)==1:assert (m,step['kind'])==(legal[0],'ONLY')
        elif rng.random()<.75:assert step['kind']=='N32' and step['NN']>=0
        else:assert (m,step['kind'])==(rng.choice(legal),'UNIFORM')
        counts[step['kind']]+=1;nn+=step['NN'];plies+=1;h+=str(m+1);b[m].append(side)
    assert h==t['history'] and (len(h)==30 or g.won(b,1) or g.won(b,2))
    rr=t['selected'];assert (rr is not None)==(j['id'] in roots)
    if rr:
        assert rr==roots[j['id']] and h.startswith(rr['history']) and rr['family']==j['family']
        b=g.replay(rr['history']);k=g.boardkey(b)
        assert k not in boards and len(rr['history']) in (24,26,28,30) and not any(g.canwin(b,s) for s in (1,2))
        assert rr['legal']==[i for i,c in enumerate(b) if len(c)<6] and len(rr['legal'])>=3;boards.add(k)
assert (nn,plies,counts['N32'])==(co['NN'],co['plies'],co['queries'])
U=np.array(list(itertools.product([0.,.5,1.],repeat=3)))
labels={};acts=nodes=0
for part in ('TRAIN_CAL','TEST'):
    lab=read(part+'_LABELS.json');lk=read(part+'_LOCK.json')
    assert lab['cohort']==fs('COHORT.json')==lk['cohort'] and lab['atlas']==lk['sha256']==fs(part+'_ATLAS.jsonl.gz')
    for z in lab['rows']:
        r=z['root'];i=r['id'];assert roots[i]==r and z['agreement'] and z['status']=='EXACT'
        w={int(a):v for a,v in z['independent'].items()};assert set(w)==set(r['legal'])
        for a,v in w.items():assert v==np.sign(z['scores'][a])
        labels[i]={a:(v+1)/2 for a,v in w.items()};acts+=len(w);nodes+=z['nodes']
assert labels.keys()==roots.keys()
records=[];req=states=0;caps=C.Counter()
for part in ('TRAIN_CAL','TEST'):
    for t in lines(part+'_ATLAS.jsonl.gz'):
        rr=t['root'];i=rr['id'];u=labels[i];assert rr==roots[i]
        seq=list(range(3));random.Random(2026112622+900000+i).shuffle(seq)
        assert t['order']==seq and t['logged']==seq[0] and t['propensity']==[1/3]*3
        pre=t['arms'][0]['states'][0];panel=t['arms'][0]['panel']
        nn0=pre['n'][:pre['len']];moves=pre['moves'][:pre['len']]
        a0=moves.index(pre['best']);b0=max((k for k in range(len(moves)) if k!=a0),key=lambda k:(nn0[k],-k));c0=next(k for k in range(len(moves)) if k not in (a0,b0))
        assert panel==[moves[k] for k in (a0,b0,c0)]
        def sc(s):return .5+.5*np.array(s['w'][:s['len']])/(np.array(s['n'][:s['len']])+4)
        def probs(x):a=np.exp(4*x);return a/a.sum(),math.log(a.mean())/4
        y=[];zz=[];cc=[];uu=[]
        for j,a in enumerate(t['arms']+([t['live']] if part=='TEST' else [])):
            req+=1;assert a['states'][0]==pre and a['panel']==panel and a['history']==rr['history'] and a['best'] in u
            assert all(lo<=u[m]<=hi for m,lo,hi in a['proof'])
            st,pl,en=a['states'];assert st['visits']==8 and en['visits']==a['visits'] and a['visits']<=128
            assert en['nn']==a['NN_calls'] and a['goal_nn']==st['nn']+12 and en['nn']-st['nn']==a['incremental_nn']
            assert 0<=a['incremental_nn']<=12
            ix=a['chosen']
            if j<3:
                assert ix==j;caps[(part,a['cause'])]+=1
            else:
                assert ix==f['tables'][f['selected']][pre['bin']] and all(a[k]==t['arms'][ix][k] for k in ('states','result','best','visits','panel','incremental_nn'))
            assert pl['visits']==12
            if ix:
                m=panel[ix];k=moves.index(m);assert pl['n'][k]-st['n'][k]==4 and a['forced']==m
            for q in a['states']:
                states+=1;x=sc(q);x0=np.ones(len(x))*.5 if q is st else sc(st)
                n=np.array(q['n'][:q['len']]);assert n.sum()==q['visits'] and np.all(np.abs(q['w'][:q['len']])<=n+1e-6)
                p0,ph0=probs(x0);p1,ph1=probs(x);div=p0@np.log(p0/p1);dr=p0@(x-x0)
                near([div,dr,ph1,ph1-ph0-dr],[q['kl'],q['drift'],q['phi'],q['bregman']])
                n=np.sort(n);assert q['bin']==4*int(n[-1]/n.sum()>=.75)+2*int(n[-1]-n[-2]>=2)+int(div>=.02)
            assert a['best']==en['best']==a['result']['best']
            if j<3:
                x=sc(en);zz.append(2*int(np.argmax([x[moves.index(m)] for m in panel]))+int(en['kl']>=.02))
                cc.append(a['incremental_nn']);uu.append(u[a['best']]);y.append(uu[-1]-.002*cc[-1])
        records.append(dict(id=i,group=rr['group'],b=pre['bin'],u=int(np.where(np.all(U==np.array([u[m] for m in panel]),axis=1))[0][0]),z=zz,y=y,n=cc,us=uu,pre=u[pre['best']],logged=t['logged'],full=max(u.values())))
assert req==802 and states==2406
tr=[r for r in records if r['group']=='TRAIN'];cal=[r for r in records if r['group']=='CAL'];te=[r for r in records if r['group']=='TEST']
assert [r['id'] for r in tr]==f['train_ids'] and [r['id'] for r in cal]==f['cal_ids']
assert len(tr)==128 and len(cal)==58 and len(te)==61
# Independent ndarray channel reconstruction: full 27x6 matrices, moments, MI, hard/coarse gain.
uc=np.zeros(27);lc=np.zeros((3,27,6))
for r in tr:
    uc[r['u']]+=1
    for c,z in enumerate(r['z']):lc[c,r['u'],z]+=1
prior=(uc+.5)/(uc.sum()+13.5);gl=(lc+.5)/(lc+.5).sum(2,keepdims=True)
global_y=np.mean([r['y'] for r in tr],axis=0);global_n=np.mean([r['n'] for r in tr],axis=0)
for b in range(8):
    rr=[r for r in tr if r['b']==b];p=np.zeros(27);L=np.zeros((3,27,6))
    for r in rr:
        p[r['u']]+=1
        for c,z in enumerate(r['z']):L[c,r['u'],z]+=1
    p=(p+8*prior)/(len(rr)+8);L+=8*gl;L/=L.sum(2,keepdims=True)
    m=p@U;my=(np.sum([r['y'] for r in rr],axis=0)+8*global_y)/(len(rr)+8)
    near(my,f['mu'][b]);assert int(np.argmax(my))==f['tables']['direct'][b]
    hs=[];mi=[];cc=(np.sum([r['n'] for r in rr],axis=0)+8*global_n)/(len(rr)+8)
    for c in range(3):
        ch=f['models'][b][c];near(p,ch['pr']);near(L[c],ch['L']);J=p[:,None]*L[c];z=J.sum(0)
        post=(J.T@U)/z[:,None];near(post,ch['post']);near(m,ch['m'])
        hard=np.max(J.T@U,axis=1).sum()-max(m);ig=np.sum(J*np.log(J/p[:,None]/z[None,:]))
        K=J.reshape(27,3,2).sum(2);pc=K.sum(0);coarse=np.max(K.T@U,axis=1).sum()-max(m)
        ic=np.sum(K*np.log(K/p[:,None]/pc[None,:]))
        near([hard,ig,ig-ic,hard-coarse],[ch['hard'],ch['mi'],ch['cmi'],ch['loss']])
        hs.append(hard-.002*cc[c]);mi.append(ig)
    assert int(np.argmax(hs))==f['tables']['evsi'][b] and int(np.argmax(mi))==f['tables']['mi'][b]
for k,t in f['tables'].items():near(np.mean([r['y'][t[r['b']]] for r in cal]),f['calnet'][k])
assert f['selected']==max(('normal','direct','evsi'),key=f['calnet'].__getitem__)
tables=dict(f['tables'],F=[1]*8,X=[2]*8)
# Actual first arm sparse score and finite assignment moments, independent reconstruction.
ope={}
for key,t in tables.items():
    vals=[];ips=[];dr=[];iv=dv=0.
    for r in te:
        b=r['b'];c=t[b];a=r['logged'];d=np.eye(3)[c]-np.eye(3)[0];y=np.array(r['y']);mu=np.array(f['mu'][b])
        target=d@y;ip=3*d*y;dd=d@mu+3*d*(y-mu)
        near([ip.mean(),dd.mean()],[target,target]);vals.append(target);ips.append(ip[a]);dr.append(dd[a]);iv+=ip.var();dv+=dd.var()
    z=su['estimates'][key]['contrast'];near([np.mean(vals),np.mean(ips),np.mean(dr),math.sqrt(iv)/61,math.sqrt(dv)/61],[z['full'],z['IPS'],z['DR'],z['IPS_sd'],z['DR_sd']])
    near(np.mean([r['us'][t[r['b']]] for r in te]),su['full'][key]['utility'])
# Post-freeze diagnostic: prefix utility known for each root OFFLINE, center potential outcomes.
# Fit ONLY TRAIN signed-gain nuisance (no policy changes); exact design moments on fixed TEST.
gy=np.mean([np.array(r['y'])-r['pre'] for r in tr],axis=0)
gain_mu=[]
for b in range(8):
    rr=[np.array(r['y'])-r['pre'] for r in tr if r['b']==b]
    gain_mu.append((np.sum(rr,axis=0)+8*gy)/(len(rr)+8))
cent={}
for key,t in tables.items():
    ip=[];dr=[];iv=dv=0.;truth=[]
    for r in te:
        b=r['b'];d=np.eye(3)[t[b]]-np.eye(3)[0];y=np.array(r['y'])-r['pre'];mu=gain_mu[b];a=r['logged']
        tar=d@y;ips=3*d*y;aug=d@mu+3*d*(y-mu)
        near([ips.mean(),aug.mean()],[tar,tar]);truth.append(tar);ip.append(ips[a]);dr.append(aug[a]);iv+=ips.var();dv+=aug.var()
    cent[key]=dict(full=float(np.mean(truth)),centered_IPS=float(np.mean(ip)),centered_DR=float(np.mean(dr)),centered_IPS_sd=math.sqrt(iv)/61,centered_DR_sd=math.sqrt(dv)/61)
out=dict(roots=len(roots),source_attempts=len(ss),source_plies=plies,source_NN=nn,source_counts=dict(counts),actions=acts,oracle_nodes=nodes,requests=req,states=states,channels=24,all_test_matched=sum(r['n']==[12]*3 for r in te),centered=cent,scope='independent grid/source and ndarray/model/OPE readback; prefix centering is retrospective offline diagnostic only; no test refit')
(D/'C09_AUDIT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2,sort_keys=True))
