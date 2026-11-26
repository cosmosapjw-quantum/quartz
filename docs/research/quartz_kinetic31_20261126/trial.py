"""Actual K29 paired response law, proper scoring and queued CPU experiment."""
from core import *
from collections import Counter,defaultdict
from fractions import Fraction as F
import gc
A=[(0,1,0,False,0),(0,4,0,False,0),(0,8,0,False,0),(0,8,8,True,0),(15,8,0,False,4),(15,8,8,True,4)]
N=9;STOP=9;KINDS=('short','serial','tree4','continue','priced');SHARDS=2
def close(a,b):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9),(a,b)
def features(r,lite,pr):
    anchor=cert(pr);anchor=lite['best'] if anchor is None else anchor
    ix=sorted(range(lite['len']),key=lambda i:(lite['moves'][i]!=anchor,-lite['n'][i],i))
    ht=c4replay(r['history'])[1];p={m:(lo,hi) for m,lo,hi in pr};rank=[];f=[len(r['history'])/42];allowed=[]
    for i in ix:
        m=lite['moves'][i];n=lite['n'][i];lo,hi=p[m];rank.append(m);allowed.append([int(2*lo),int(2*hi)])
        f.extend((math.sqrt(n/lite['visits']),math.sqrt(max(0,lite['prior'][i])),.5+.5*lite['w'][i]/(n+4),math.sqrt(max(0,lite['m2'][i])/(n+1))/2,ht[m]/6,abs(m-3)/3,lo,hi))
    return rank,f,allowed
def check(a,r,arm=None,policy=None,trace=False):
    assert a.get('schema')=='quartz_k31_c4_v1',a
    assert a['history']==r['history'] and a['legal']==r['legal'] and a['proof']==proof(r['history'])
    x=a['arm'];assert x in range(10) and a['policy']==policy
    if arm is not None:assert x==arm and a['prediction'] is None
    pre,mid,end=[a[k] for k in ('prefix','mid','end')]
    ph,mh,eh=[a[k] for k in ('prefix_hooks','mid_hooks','end_hooks')]
    assert pre['visits']==8 and pre['nn']==1+ph['hooks']['base_evals'] and ph['hooks']['extra_evals']==0
    assert ph['ledger']['iterations']==8 and ph['ledger']['transferred']==0
    b=a['first'];assert b['target']==32 and b['cap']==128 and b['reserve']==0 and b['used_nn']+b['unused']==32
    assert b['used_nn']==mid['nn']-pre['nn']==mh['hooks']['base_evals'] and b['visits']==mid['visits']-pre['visits']
    assert b['visits']==mh['hooks']['base_evals']+mh['hooks']['terminal_leaves'] and mh['hooks']['extra_evals']==0 and b['visits']<=128
    if b['unused']:assert b['visits']==128
    assert mh['ledger']['transferred']==0 and mh['ledger']['iterations']==mid['visits']
    rank,f,allow=features(r,mid,a['proof']);assert rank==a['rank'] and allow==a['allowed']
    for v,w in zip(f,a['features']):close(v,w)
    assert a['mid_best']==rank[0];cc=cert(a['proof']);assert a['native_best']==(end['best'] if cc is None else cc)
    assert a['best'] in rank
    df=direct_features(a)
    for v,w in zip(df,a['direct_features']):close(v,w)
    if x>=6:
        assert a['tail'] is None and a['queue'] is None and end==mid
        assert eh['ledger']==mh['ledger']
        if x==STOP:
            assert a['proof_graph'] is None and a['rule_work']==0 and a['best']==a['native_best']
        else:
            g=a['proof_graph'];assert g['arm']==x and len(g['paths'])==mid['visits']
            assert (len(g['steps'])==0 if x==6 else len(g['steps'])<=8)
            prior=g['before']
            for st in g['steps']:
                z=st['post'];assert z['risk'][0]<=prior['risk'][0] and z['risk'][1]<=prior['risk'][1]
                assert z['registered']>prior['registered'] and z['rule_work']>prior['rule_work']
                assert [i[0] for i in z['actions']]==[i[0] for i in prior['actions']]
                for (_,v),(_,w) in zip(z['actions'],prior['actions']):assert w[0]<=v[0]<=v[1]<=w[1]
                prior=z
            assert all(g['end'][k]==v for k,v in prior.items() if k!='scored')
            assert g['end']['rule_work']==a['rule_work']>0 and a['best']==g['end']['best']
    else:
        assert a['proof_graph'] is None and a['rule_work']==0 and a['best']==a['native_best']
        mask,w,width,dedup,res=A[x];h=eh['hooks'];b=a['tail'];q=eh['ledger'];p=mh['ledger']
        assert b['target']==32 and b['cap']==128 and b['reserve']==res and b['used_nn']+b['unused']==32
        assert b['used_nn']==end['nn']-mid['nn']==h['base_evals']+h['extra_evals']
        assert b['visits']==end['visits']-mid['visits']==h['base_evals']+h['terminal_leaves'] and 0<b['visits']<=128
        if b['unused']:assert b['visits']==128
        if res:
            assert b['protected_extra']+b['released']==res and b['protected_extra']+b['borrowed']==b['pre_extra']
            assert b['protected_base']+b['borrowed']<=32-res and b['pre_extra']<=h['extra_evals']
        assert 1<=eh['peak']<=w and q['iterations']==end['visits']
        assert q['eligible']==q['novel']+q['blocked_receipt'] and q['transferred']==q['receipt_items']>=p['transferred'] and q['repeated_delivery']==0
        if not mask:
            assert h['select_nodes']==h['width_nodes']==h['refinements']==h['extra_evals']==0 and q['transferred']==0
        if width:
            q=a['queue'];assert q['width']==width and q['dedup']==dedup and q['logical']==b['used_nn']
            assert q['logical']==q['physical']+q['coalesced'] and q['logical']==q['distinct']+q['repeat_keys']
            assert q['physical']>=q['distinct'] and q['coalesced']<=q['repeat_keys']
            assert len(q['logical_hist'])==len(q['physical_hist'])==9
            assert sum(q['logical_hist'])==sum(q['physical_hist'])==q['batches']
            assert sum(i*v for i,v in enumerate(q['logical_hist']))==q['logical'] and sum(i*v for i,v in enumerate(q['physical_hist']))==q['physical']
            assert not any(q['logical_hist'][width+1:]) and not any(q['physical_hist'][width+1:]) and q['peak_pending']<=w
            assert q['latency_ns']>=q['wait_ns']>=0 and q['compute_ns']>=0
            assert q['epoch']>=1 and q['specialized']==True
            kh=q['kernel_hist'];expected=[0]*9
            for i,n in enumerate(q['physical_hist']):
                if i:expected[1<<(i-1).bit_length()]+=n
            assert kh==expected and q['kernel_lanes']==sum(i*n for i,n in enumerate(kh))
            assert q['physical']<=q['kernel_lanes']<=8*q['batches']
            if not dedup:assert q['coalesced']==0 and q['logical_hist']==q['physical_hist']
            assert (q['trace'] is not None)==trace and q['parity_checks']==(q['physical'] if trace else 0)
        else:assert a['queue'] is None
    assert a['NN_calls']==end['nn'] and a['visits']==end['visits']
    assert a['physical_calls']==a['NN_calls']-(a['queue']['coalesced'] if a['queue'] else 0)
    assert (eh['trace'] is not None)==trace and eh['ledger']['iterations']==end['visits'] and eh['ledger']['repeated_delivery']==0
    for s in (pre,mid,end):assert sum(s['n'][:s['len']])==s['visits'] and set(s['moves'][:s['len']])==set(rank)
    snap=a['snapshot'];assert snap['root_n']==end['visits'] and snap['evaluator_calls']==end['nn'] and snap['best']==end['best']
    es={e['move']:e for e in snap['edges']}
    for j,m in enumerate(end['moves'][:end['len']]):
        z=es.get(m,{});assert end['n'][j]==z.get('n',0) and z.get('vl',0)==z.get('vvalue',0)==0
        close(end['w'][j],z.get('w',0));close(end['m2'][j],z.get('m2',0))
    if policy is None:assert a['prediction'] is None
    else:
        p=a['prediction'];assert p is not None and len(p['arms'])==N
        assert x==(p['fixed_arm'] if policy=='continue' else p['priced_arm'])
        for v in p['arms']:close(v['net'],v['g']-.002*v['nn']-.00002*v['rules'])
    return a
def req(s,r,arm=None,policy=None,model=None,trace=False):
    kwargs=dict(arm=arm) if arm is not None else dict(policy=policy,belief_path=str(D/'MODEL.json'),belief_identity=model)
    return check(s.request(r['history'],trace=trace,**kwargs),r,arm,policy,trace)
def order(r,phase):
    oo=[(a,j) for a in range(N) for j in range(2)] if phase.endswith('ATLAS') else [(k,j) for k in KINDS for j in range(2)]
    random.Random(SEED+(900000 if phase.endswith('ATLAS') else 800000)+r['id']).shuffle(oo);return oo
def shard(arg):
    phase,ix,rr,mi=arg;fn=D/f'{phase}_{ix}.jsonl.gz';assert not fn.exists();tmp=fn.with_suffix('.tmp')
    n=nn=vis=phys=rules=0;t0=time.monotonic();epoch=0
    with Server() as s,gzip.open(tmp,'wt',compresslevel=3) as fd:
        for i,r in enumerate(rr):
            oo=order(r,phase);z={}
            for k,j in oo:
                if phase.endswith('ATLAS'):a=req(s,r,arm=k)
                else:a=req(s,r,arm={'short':STOP,'serial':0,'tree4':1}[k]) if k in KINDS[:3] else req(s,r,policy=k,model=mi)
                if a['queue']:
                    epoch+=1;assert a['queue']['epoch']==epoch
                z[str(k)+':'+str(j)]=a;n+=1;nn+=a['NN_calls'];phys+=a['physical_calls'];vis+=a['visits'];rules+=a['rule_work']
            p=next(iter(z.values()));assert all(a['prefix']==p['prefix'] and a['mid']==p['mid'] and a['rank']==p['rank'] and a['features']==p['features'] for a in z.values())
            emit(fd,dict(root=r,order=oo,results=z))
            if (i+1)%50==0:print('PROGRESS',phase,ix,i+1,len(rr),n,round(time.monotonic()-t0,2),flush=True)
    tmp.replace(fn);return dict(file=fn.name,sha256=sha(fn),roots=len(rr),requests=n,NN=nn,physical=phys,visits=vis,rules=rules,elapsed=time.monotonic()-t0)
def collect(phase):
    co=read('COHORT.json');assert co['source']==sha(D/'SOURCE.jsonl.gz')
    mi=None;freeze=None
    if phase=='TRAIN_CAL_ATLAS':rr=[r for r in co['roots'] if r['group']!='TEST']
    elif phase=='TEST_ATLAS':
        f=read('FROZEN.json');freeze=sha(D/'FROZEN.json');rr=[r for r in co['roots'] if r['group']=='TEST']
    elif phase=='CAL_LIVE':
        f=read('MODEL_LOCK.json');freeze=sha(D/'MODEL_LOCK.json');mi=sha(D/'MODEL.json');assert f['model']==mi
        rr=[r for r in co['roots'] if r['id'] in set(f['cal_b'])]
    elif phase=='TEST_LIVE':
        f=read('FROZEN.json');freeze=sha(D/'FROZEN.json');mi=sha(D/'MODEL.json');assert f['model']==mi
        rr=[r for r in co['roots'] if r['group']=='TEST']
    else:raise ValueError(phase)
    assert rr
    with ProcessPoolExecutor(max_workers=SHARDS) as ex:fs=list(ex.map(shard,[(phase,i,rr[i::SHARDS],mi) for i in range(SHARDS)]))
    save(phase+'_LOCK.json',dict(cohort=sha(D/'COHORT.json'),identity=identity(),files=fs,freeze=freeze,**{k:sum(z[k] for z in fs) for k in ('roots','requests','NN','physical','visits','rules')}))
    print('COLLECT',phase,{k:read(phase+'_LOCK.json')[k] for k in ('roots','requests','NN','physical','visits','rules')},flush=True)
def rows(phase):
    for f in read(phase+'_LOCK.json')['files']:
        p=D/f['file'];assert sha(p)==f['sha256'];yield from raw(p)
def label_worker(r):
    z=one_label(r);gc.collect();return z
def labels(part):
    lk=read(part+'_ATLAS_LOCK.json');co=read('COHORT.json');assert lk['cohort']==sha(D/'COHORT.json')
    rr=[r for r in co['roots'] if (r['group']=='TEST')==(part=='TEST')]
    with ProcessPoolExecutor(max_workers=8) as ex:out=list(ex.map(label_worker,rr,chunksize=1))
    save(part+'_LABELS.json',dict(atlas=sha(D/(part+'_ATLAS_LOCK.json')),cohort=lk['cohort'],counts=dict(Counter(z['status'] for z in out)),rows=out));print('LABELS',part,dict(Counter(z['status'] for z in out)),flush=True)
def labelmap(part):
    l=read(part+'_LABELS.json');assert l['atlas']==sha(D/(part+'_ATLAS_LOCK.json'))
    return {z['root']['id']:{int(a):int(v+1) for a,v in z['independent'].items()} for z in l['rows'] if z['status']=='EXACT'}
def direct_features(a):
    f=a['features'];v=[f[i:i+8] for i in range(1,len(f),8)];l=len(v)
    def maxr(i):return max(x[i] for x in v[1:])
    def ent(i):return -sum(x[i]**2*math.log(x[i]**2) for x in v if x[i]>0)/math.log(l)
    return [f[0],l/7,v[0][0]**2,maxr(0)**2,v[0][1]**2,maxr(1)**2,v[0][2],maxr(2),v[0][3],maxr(3),ent(0),ent(1),a['mid']['nn']/41,a['mid']['visits']/136,sum(x[7]==0 for x in v)/l,float(a['prefix']['best']==a['mid']['best'])]
def outcomes(a,u):
    return [(u[a['best']]-u[a['mid_best']])/2,a['NN_calls']-a['mid']['nn'],a.get('rule_work',0)]
def datum(t,u):
    z=t['results'];a=z['0:0'];return dict(f=direct_features(a),y=[[S.mean(outcomes(z[str(c)+':'+str(j)],u)[k] for j in range(2)) for k in range(3)] for c in range(N)])
def historical_train():
    donors={str(i):[] for i in range(N)};locks=[];sources=[]
    for old in (29,30):
        p=R/f'quartz_kinetic{old}_20261126'
        ll={z['root']['id']:{int(a):int(v+1) for a,v in z['independent'].items()} for z in json.loads((p/'TRAIN_CAL_LABELS.json').read_text())['rows'] if z['status']=='EXACT' and z['root']['group']=='TRAIN'}
        lk=json.loads((p/'TRAIN_CAL_ATLAS_LOCK.json').read_text());locks.extend(lock(p/n) for n in ('TRAIN_CAL_LABELS.json','TRAIN_CAL_ATLAS_LOCK.json'))
        count=0
        for fi in lk['files']:
            f=p/fi['file'];assert sha(f)==fi['sha256'];locks.append(lock(f))
            for t in raw(f):
                i=t['root']['id']
                if i not in ll:continue
                z=t['results'];x=direct_features(z['0:0']);u=ll[i];count+=1
                for new,c in ((0,0),(1,1),(2,2),(4,6)):
                    y=[S.mean(outcomes(z[str(c)+':'+str(j)],u)[k] for j in range(2)) for k in range(3)]
                    donors[str(new)].append(dict(source=old,id=i,f=x,y=y))
        assert count==len(ll);sources.append(dict(cycle=old,roots=count))
    return dict(donors=donors,locks=locks,sources=sources)
def solve(a,b):
    # Partial-pivot elimination for positive ridge normal equations.
    a=[list(row)+[v] for row,v in zip(a,b)];n=len(b)
    for i in range(n):
        p=max(range(i,n),key=lambda j:abs(a[j][i]));a[i],a[p]=a[p],a[i];d=a[i][i];assert abs(d)>1e-12
        for k in range(i,n+1):a[i][k]/=d
        for j in range(i+1,n):
            d=a[j][i]
            for k in range(i,n+1):a[j][k]-=d*a[i][k]
    x=[0.]*n
    for i in reversed(range(n)):x[i]=a[i][-1]-sum(a[i][j]*x[j] for j in range(i+1,n))
    return x
def fit_arm(data,lam):
    n=len(data);assert n>0
    mu=[S.mean(z['f'][i] for z in data) for i in range(16)]
    sc=[S.pstdev(z['f'][i] for z in data) for i in range(16)];sc=[v if v>=1e-6 else 1. for v in sc]
    if lam is None:return dict(mean=mu,scale=sc,coef=[[S.mean(z['y'][j] for z in data)]+[0.]*16 for j in range(3)])
    x=[[1.]+[(v-m)/s for v,m,s in zip(z['f'],mu,sc)] for z in data]
    gram=[[sum(t[i]*t[j] for t in x)+(n*lam if i==j and i>0 else 0.) for j in range(17)] for i in range(17)]
    cs=[solve(gram,[sum(v[i]*z['y'][j] for v,z in zip(x,data)) for i in range(17)]) for j in range(3)]
    return dict(mean=mu,scale=sc,coef=cs)
def fit_model(rows,old,lam,augment):
    return dict(arms=[fit_arm([dict(f=r['f'],y=r['y'][c]) for r in rows]+(old['donors'][str(c)] if augment else []),lam) for c in range(N)])
def predict(m,f):
    rr=[]
    for c,a in enumerate(m['arms']):
        x=[(v-m)/s for v,m,s in zip(f,a['mean'],a['scale'])]
        g,k,l=[w[0]+sum(v*t for v,t in zip(w[1:],x)) for w in a['coef']]
        g=max(-1,min(1,g));k=max(0,min(32,k)) if c<6 else 0.;l=max(0,l) if c>=6 else 0.
        rr.append(dict(g=g,nn=k,rules=l,net=g-.002*k-.00002*l))
    def am(n):
        a=0
        for i in range(1,n):
            if rr[i]['net']>rr[a]['net']+1e-12:a=i
        return a
    a=am(N);return dict(arms=rr,fixed_arm=am(6),priced_arm=a if rr[a]['net']>1e-12 else STOP)
def mse(m,rows):
    return S.mean(S.mean((a['g']-r['y'][i][0])**2 for i,a in enumerate(predict(m,r['f'])['arms'])) for r in rows)
def fit():
    assert not (D/'CAL_LIVE_LOCK.json').exists() and not (D/'TEST_ATLAS_LOCK.json').exists()
    ll=labelmap('TRAIN_CAL');train={};cal={}
    for t in rows('TRAIN_CAL_ATLAS'):
        r=t['root'];i=r['id']
        if i in ll:(train if r['group']=='TRAIN' else cal)[i]=datum(t,ll[i])
    tr=list(sorted(train));random.Random(SEED+11001).shuffle(tr);ids=sorted(cal);random.Random(SEED+10001).shuffle(ids);ca=ids[::2];cb=ids[1::2]
    old=historical_train();save('HISTORICAL.json',old);candidates=[];models=[]
    for augment in (False,True):
        for lam in (None,.03,.3,3.):
            m=fit_model(list(train.values()),old,lam,augment);models.append(m)
            oof=[]
            for j in range(2):
                fitids=tr[j::2];valid=tr[1-j::2]
                o=fit_model([train[i] for i in fitids],old,lam,augment);oof.append((len(valid),mse(o,[train[i] for i in valid])))
            candidates.append(dict(augment=augment,lambda_=lam,cal=mse(m,[cal[i] for i in ca]),oof=sum(n*x for n,x in oof)/len(tr)))
    best=min(range(len(candidates)),key=lambda i:candidates[i]['cal'])
    save('MODEL.json',models[best]);save('CONSTANT.json',models[0])
    save('CAL_SCORING.json',dict(candidates=candidates,selected=best,scope='root-disjoint OOF fresh TRAIN, CAL-A direct gain MSE; CAL-B actual policies; historical TRAIN-only scalar'))
    save('MODEL_LOCK.json',dict(cohort=sha(D/'COHORT.json'),identity=identity(),code=sha(D/'trial.py'),atlas=sha(D/'TRAIN_CAL_ATLAS_LOCK.json'),labels=sha(D/'TRAIN_CAL_LABELS.json'),model=sha(D/'MODEL.json'),constant=sha(D/'CONSTANT.json'),historical=sha(D/'HISTORICAL.json'),cal_scoring=sha(D/'CAL_SCORING.json'),train_ids=tr,cal_ids=ids,cal_a=ca,cal_b=cb))
    print('FIT',len(tr),len(ca),len(cb),candidates,'selected',best,old['sources'],flush=True)
def util(a,u):return u[a['best']]/2
def net(a,u):return util(a,u)-.002*a['NN_calls']-.00002*a['rule_work']
def calibrate():
    ll=labelmap('TRAIN_CAL');v=defaultdict(list);ids=[]
    for t in rows('CAL_LIVE'):
        i=t['root']['id'];ids.append(i);u=ll[i]
        for k in KINDS:v[k].append(S.mean(net(t['results'][k+':'+str(j)],u) for j in range(2)))
    means={k:S.mean(v[k]) for k in KINDS};lower={}
    def choose(base,arm):
        b=max(base,key=means.__getitem__);d=[x-y for x,y in zip(v[arm],v[b])]
        lower[arm]=S.mean(d)-2*S.stdev(d)/math.sqrt(len(d))
        return (arm if lower[arm]>0 else b),b
    fixed,basefixed=choose(('serial','tree4'),'continue');priced,basepriced=choose(('short','serial','tree4'),'priced')
    f=read('MODEL_LOCK.json');save('FROZEN.json',dict(f,model_lock=sha(D/'MODEL_LOCK.json'),cal_live=sha(D/'CAL_LIVE_LOCK.json'),calnet=means,cal_lower=lower,cal_actual_ids=sorted(ids),fixed=fixed,priced=priced,basefixed=basefixed,basepriced=basepriced))
    print('CAL_LIVE',len(ids),means,'lower',lower,'fixed',fixed,'priced',priced,flush=True)
def qstats(a):
    q=a['queue'];d=Counter()
    if q:
        for k in ('logical','physical','coalesced','distinct','repeat_keys','batches','wait_ns','compute_ns','latency_ns'):d[k]=q[k]
        d['padded_lanes']=8*q['batches'];d['kernel_lanes']=q['kernel_lanes'];d['multi_batches']=sum(q['logical_hist'][2:])
        for j,z in enumerate(q['logical_hist']):d['LH'+str(j)]=z
        for j,z in enumerate(q['physical_hist']):d['PH'+str(j)]=z
        for j,z in enumerate(q['kernel_hist']):d['KH'+str(j)]=z
    return d
def compact(a,u):
    d=qstats(a);h=a['end_hooks']['hooks'];l=a['end_hooks']['ledger'];p=a['mid_hooks']['ledger']
    return dict(u=util(a,u),native=u[a['native_best']]/2,opt=max(u.values())/2,net=net(a,u),NN=a['NN_calls'],physical=a['physical_calls'],tail_NN=a['NN_calls']-a['mid']['nn'],rules=a['rule_work'],visits=a['visits'],arm=a['arm'],
      cpu=a['cpu_ns']/1e6,wall=a['wall_ns']/1e6,tail=a['tail_us']/1000,queue_overhead=(a['queue_setup_us']+a['queue_join_us'])/1000,control=a['control_us']/1000,
      n=a['end']['n'][:a['end']['len']],moves=a['end']['moves'][:a['end']['len']],best=a['best'],counts=dict(d),peak=a['queue']['peak_pending'] if a['queue'] else 0,
      sxvb=[h['select_changed'],a['end_hooks']['width_realized'],h['eval_changed'],l['applied']-p['applied']] if a['arm']<6 else [0]*4,
      receipt=l['transferred']-p['transferred'],blocked=l['blocked_receipt']-p['blocked_receipt'])
def aggregate(v):
    aa=[a for z in v for a in z];ct=Counter();arms=Counter();peak=Counter();sx=[0]*4;all4=0
    for a in aa:
        ct.update(a['counts']);arms[str(a['arm'])]+=1;peak[str(a['peak'])]+=1
        for i,x in enumerate(a['sxvb']):sx[i]+=x
        all4+=all(x>0 for x in a['sxvb'])
    out={k:S.mean(a[k] for a in aa) for k in ('u','native','net','NN','physical','visits','tail_NN','rules','cpu','wall','tail','queue_overhead','control')}
    return dict(n=len(v),requests=len(aa),**out,regret=S.mean(a['opt']-a['u'] for a in aa),counts=dict(ct),arms=dict(arms),peaks=dict(peak),sxvb=sx,all4=all4,receipt=sum(a['receipt'] for a in aa),blocked=sum(a['blocked'] for a in aa),
        stopped=arms['9'],batch_mean=ct['logical']/ct['batches'] if ct['batches'] else None,physical_batch_mean=ct['physical']/ct['batches'] if ct['batches'] else None,wait_us=ct['wait_ns']/ct['logical']/1000 if ct['logical'] else None)
def pair(a,b):
    if not a:return dict(n=0)
    d=[S.mean(x['u']-y['u'] for x,y in zip(v,w)) for v,w in zip(a,b)]
    n=[S.mean(x['net']-y['net'] for x,y in zip(v,w)) for v,w in zip(a,b)]
    rng=random.Random(SEED+712);bs=sorted(sum(rng.choices(d,k=len(d)))/len(d) for _ in range(2000))
    rng=random.Random(SEED+712);bn=sorted(sum(rng.choices(n,k=len(n)))/len(n) for _ in range(2000))
    return dict(n=len(a),gain=S.mean(d),net_gain=S.mean(n),ci=[bs[49],bs[1949]],net_ci=[bn[49],bn[1949]],better=sum(x>0 for x in d),worse=sum(x<0 for x in d),
        NN_delta=S.mean(x['NN']-y['NN'] for v,w in zip(a,b) for x,y in zip(v,w)),physical_delta=S.mean(x['physical']-y['physical'] for v,w in zip(a,b) for x,y in zip(v,w)))
def compare(a,b):
    def match(k):
        mm=[(v,w) for v,w in zip(a,b) if all(x[k]==y[k] for x,y in zip(v,w))]
        return pair([v for v,w in mm],[w for v,w in mm])
    return dict(all=pair(a,b),equal_logical=match('NN'),equal_physical=match('physical'))
def js(a,b):
    x={m:n/sum(a['n']) for m,n in zip(a['moves'],a['n'])};y={m:n/sum(b['n']) for m,n in zip(b['moves'],b['n'])}
    return sum((p*math.log(2*p/(p+q)) if p else 0.)/2+(q*math.log(2*q/(p+q)) if q else 0.)/2 for m in x.keys()|y.keys() for p,q in [(x.get(m,0),y.get(m,0))])
def repeat(v):
    return dict(n=len(v),utility_sd=math.sqrt(S.mean((a['u']-b['u'])**2/2 for a,b in v)),logical_sd=math.sqrt(S.mean((a['NN']-b['NN'])**2/2 for a,b in v)),
      physical_sd=math.sqrt(S.mean((a['physical']-b['physical'])**2/2 for a,b in v)),JSD=S.mean(js(a,b) for a,b in v),disagree=sum(a['best']!=b['best'] for a,b in v))
def verify_prediction(a,p):
    z=a['prediction'];assert z['fixed_arm']==p['fixed_arm'] and z['priced_arm']==p['priced_arm']
    for x,y in zip(z['arms'],p['arms']):
        for k in ('g','nn','rules','net'):close(x[k],y[k])
def forecast(v):
    out=[]
    for c in range(N):
        x=[p['arms'][c]['g'] for a,p,q in v];y=[a['y'][c][0] for a,p,q in v];q=[q['arms'][c]['g'] for a,p,q in v]
        costs=[a['y'][c][1] for a,p,q in v];pred=[p['arms'][c]['nn'] for a,p,q in v]
        rules=[a['y'][c][2] for a,p,q in v];pr=[p['arms'][c]['rules'] for a,p,q in v]
        bins=defaultdict(list)
        for a,b in zip(x,y):bins[sum(a>z for z in (-.05,0,.025,.05,.1))].append((a,b))
        out.append(dict(arm=c,pred=S.mean(x),actual=S.mean(y),bias=S.mean(a-b for a,b in zip(x,y)),rmse=math.sqrt(S.mean((a-b)**2 for a,b in zip(x,y))),constant_rmse=math.sqrt(S.mean((a-b)**2 for a,b in zip(q,y))),
            corr=S.correlation(x,y) if S.pstdev(x)>1e-12 and S.pstdev(y)>1e-12 else None,nn=S.mean(costs),pred_nn=S.mean(pred),rules=S.mean(rules),pred_rules=S.mean(pr),
            bins={str(k):dict(n=len(z),pred=S.mean(a for a,b in z),actual=S.mean(b for a,b in z)) for k,z in bins.items()}))
    return out
def graph_stats(a,u):
    g=a['proof_graph'];assert g is not None
    b,e=g['before'],g['end']
    for p in [b]+[z['post'] for z in g['steps']]+[e]:
        for m,(l,h) in p['actions']:assert l<=u[m]-1<=h
        if p['witness'] is not None:assert u[p['witness']]==max(u.values())
    return dict(before_risk=b['risk'][0],end_risk=e['risk'][0],width_before=b['risk'][1],width_end=e['risk'][1],
        before_witness=int(b['witness'] is not None),end_witness=int(e['witness'] is not None),new_witness=int(b['witness'] is None and e['witness'] is not None),
        changes=int(e['best']!=a['mid_best']),gain=util(a,u)-u[a['mid_best']]/2,rule=a['rule_work'],build_rule=b['rule_work'],steps=len(g['steps']),
        guided=sum(z['kind']=='FRONTIER' for z in g['steps']),scored=e['scored'],shared=e['shared_notices'],extra_shared=e['shared_notices']-b['shared_notices'])
def dr_rows(t,u,p,kind):
    a=p['fixed_arm'] if kind=='fixed' else p['priced_arm'];b=0 if kind=='fixed' else STOP
    q=[z['net'] for z in p['arms']]+[0.]
    def gain(c,j):
        if c==STOP:return 0.
        y,k,l=outcomes(t['results'][str(c)+':'+str(j)],u);return y-.002*k-.00002*l
    d=S.mean(gain(a,j)-gain(b,j) for j in range(2));direct=q[a]-q[b]
    scores=[]
    for c,j in t['order']:
        v=gain(c,j);w=N*(int(c==a)-int(c==b));scores.append((w*v,direct+w*(v-q[c])))
    for k in range(2):close(S.mean(z[k] for z in scores),d)
    return dict(a=a,b=b,full=d,direct=direct,ips=scores[0][0],dr=scores[0][1],ips_var=S.pvariance(z[0] for z in scores),dr_var=S.pvariance(z[1] for z in scores))
def score():
    f=read('FROZEN.json');m=read('MODEL.json');constant=read('CONSTANT.json');ll=labelmap('TEST')
    fixed={};live={};v=[];pred={};graph=defaultdict(list);dr=defaultdict(list);verify_n=0;short={}
    for t in rows('TEST_ATLAS'):
        r=t['root'];i=r['id']
        if i not in ll:continue
        u=ll[i];x=datum(t,u);p=predict(m,x['f']);q=predict(constant,x['f']);v.append((x,p,q));pred[i]=p
        fixed[i]=[[compact(t['results'][str(c)+':'+str(j)],u) for j in range(2)] for c in range(N)]
        for c in (6,7,8):
            for j in range(2):graph[str(c)].append(graph_stats(t['results'][str(c)+':'+str(j)],u))
        for k in ('fixed','priced'):dr[k].append(dr_rows(t,u,p,k))
    for t in rows('TEST_LIVE'):
        i=t['root']['id']
        if i not in ll:continue
        u=ll[i];live[i]={k:[compact(t['results'][k+':'+str(j)],u) for j in range(2)] for k in KINDS}
        for k in KINDS[3:]:
            for j in range(2):verify_prediction(t['results'][k+':'+str(j)],pred[i]);verify_n+=1
    ids=sorted(ll);assert set(ids)==set(fixed)==set(live)
    d={str(c):[fixed[i][c] for i in ids] for c in range(N)};a={k:[live[i][k] for i in ids] for k in KINDS}
    def dr_summary(v):
        return dict(n=len(v),disagree=sum(x['a']!=x['b'] for x in v),full=S.mean(x['full'] for x in v),direct=S.mean(x['direct'] for x in v),ips=S.mean(x['ips'] for x in v),dr=S.mean(x['dr'] for x in v),
            ips_cond_sd=math.sqrt(S.mean(x['ips_var'] for x in v)),dr_cond_sd=math.sqrt(S.mean(x['dr_var'] for x in v)),ips_empirical_sd=S.stdev(x['ips'] for x in v),dr_empirical_sd=S.stdev(x['dr'] for x in v))
    out=dict(n=len(ids),fixed_selected=f['fixed'],priced_selected=f['priced'],fixed={k:aggregate(v) for k,v in d.items()},live={k:aggregate(v) for k,v in a.items()},
        fixed_contrasts={x+':'+y:compare(d[x],d[y]) for x,y in [('1','0'),('2','0'),('3','2'),('5','4'),('7','6'),('8','6'),('8','7')]},
        graph_vs_short={k:compare(d[k],a['short']) for k in ('6','7','8')},
        live_contrasts={x+':'+y:compare(a[x],a[y]) for x,y in [('continue','serial'),('continue','tree4'),('priced','short'),('priced','serial'),(f['fixed'],'serial'),(f['priced'],'short')]},
        fixed_repeats={k:repeat(v) for k,v in d.items()},live_repeats={k:repeat(v) for k,v in a.items()},
        graph={k:{'n':len(v),**{x:sum(z[x] for z in v) for x in v[0]}} for k,v in graph.items()},forecast=forecast(v),dr={k:dr_summary(v) for k,v in dr.items()},native_model_parity=verify_n)
    save('SUMMARY.json',out);print('SUMMARY',len(ids),f['fixed'],f['priced'],'forecast',out['forecast'],'graph',out['graph'],'dr',out['dr'],flush=True)
    for k,v in out['fixed'].items():print('FIXED',k,v,flush=True)
    for k,v in out['live'].items():print('LIVE',k,v,flush=True)
    print('CONTRASTS',out['fixed_contrasts'],out['graph_vs_short'],out['live_contrasts'],flush=True)
def diagnostic():
    rr=[r for r in read('COHORT.json')['roots'] if r['group']=='TRAIN'][:3];fn=D/'DIAGNOSTIC.jsonl.gz'
    assert not fn.exists()
    with Server() as s,gzip.open(fn,'wt',compresslevel=3) as fd:
        for r in rr:emit(fd,dict(root=r,results={str(c):req(s,r,arm=c,trace=True) for c in (3,5,6,7,8)}))
    save('DIAGNOSTIC_LOCK.json',dict(sha256=sha(fn),requests=len(rr)*5,roots=[r['id'] for r in rr]))
def timing():
    rr=[r for r in read('COHORT.json')['roots'] if r['group']=='TEST'][:12];fn=D/'TIMING.jsonl.gz';assert not fn.exists()
    with Server() as s,gzip.open(fn,'wt',compresslevel=3) as fd:
        for r in rr:
            oo=order(r,'TEST_ATLAS');res={str(c)+':'+str(j):req(s,r,arm=c) for c,j in oo}
            emit(fd,dict(root=r,order=oo,results=res))
    save('TIMING_LOCK.json',dict(sha256=sha(fn),requests=len(rr)*18,roots=[r['id'] for r in rr],scope='single server, host not isolated'))
def smoke():
    r=json.loads((R/'quartz_kinetic30_20261126/COHORT.json').read_text())['roots'][0]
    with Server() as s:
        for c in range(10):
            a=req(s,r,arm=c,trace=True);print('SMOKE',c,a['NN_calls'],a['physical_calls'],a['visits'],a['queue'] is not None,flush=True)
        for z in (dict(arm=10),dict(arm=0,policy='prior'),dict(policy='prior'),dict(arm=0,oracle=1)):
            assert 'error' in s.request(r['history'],**z)
    print('SMOKE 10 requests, 4 rejected',flush=True)
if __name__=='__main__':
    if sys.argv[1] in ('collect','labels'):globals()[sys.argv[1]](sys.argv[2])
    else:globals()[sys.argv[1]]()
