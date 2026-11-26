"""Actual K29 paired response law, proper scoring and queued CPU experiment."""
from core import *
from collections import Counter,defaultdict
from fractions import Fraction as F
import gc
A=[(0,1,0,False,0),(0,4,0,False,0),(0,8,0,False,0),(0,4,4,False,0),(0,8,8,False,0),(0,8,8,True,0),(15,8,0,False,4),(15,8,8,True,4)]
N=8;STOP=8;KINDS=('short','serial','tree4','prior','continue','priced');SHARDS=2
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
    assert a.get('schema')=='quartz_k29_c4_v1',a
    assert a['history']==r['history'] and a['legal']==r['legal'] and a['proof']==proof(r['history'])
    x=a['arm'];assert x in range(9) and a['policy']==policy
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
    assert a['z']==rank.index(a['native_best']) and a['best'] in rank
    if x==STOP:assert a['tail'] is None and a['queue'] is None and end==mid
    else:
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
    if policy is None:assert a['best']==a['native_best']
    else:
        p=a['prediction'];assert p is not None and len(p['arms'])==8
        assert x==(8 if policy=='prior' else p['continue_arm'] if policy=='continue' else p['priced_arm'])
        assert a['best']==(cc if cc is not None else rank[p['prior_rank'] if x==8 else p['arms'][x]['post_rank'][a['z']]])
        for c in p['arms']:
            close(sum(c['mass']),1)
            for j,m in enumerate(p['prior']):close(sum(v[j] for v in c['joint_moments']),m)
            close(c['G'],max(0,sum(max(v) for v in c['joint_moments'])-max(p['prior'])))
            close(c['D'],sum(v[i] for i,v in enumerate(c['joint_moments']))-p['prior'][0])
            assert c['G']>=0
    return a
def req(s,r,arm=None,policy=None,model=None,trace=False):
    kwargs=dict(arm=arm) if arm is not None else dict(policy=policy,belief_path=str(D/'MODEL.json'),belief_identity=model)
    return check(s.request(r['history'],trace=trace,**kwargs),r,arm,policy,trace)
def order(r,phase):
    oo=[(a,j) for a in range(N) for j in range(2)] if phase.endswith('ATLAS') else [(k,j) for k in KINDS for j in range(2)]
    random.Random(SEED+(900000 if phase.endswith('ATLAS') else 800000)+r['id']).shuffle(oo);return oo
def shard(arg):
    phase,ix,rr,mi=arg;fn=D/f'{phase}_{ix}.jsonl.gz';assert not fn.exists();tmp=fn.with_suffix('.tmp')
    n=nn=vis=phys=0;t0=time.monotonic()
    with Server() as s,gzip.open(tmp,'wt',compresslevel=3) as fd:
        for i,r in enumerate(rr):
            oo=order(r,phase);z={}
            for k,j in oo:
                if phase.endswith('ATLAS'):a=req(s,r,arm=k)
                else:a=req(s,r,arm={'short':8,'serial':0,'tree4':1}[k]) if k in KINDS[:3] else req(s,r,policy=k,model=mi)
                z[str(k)+':'+str(j)]=a;n+=1;nn+=a['NN_calls'];phys+=a['physical_calls'];vis+=a['visits']
            p=next(iter(z.values()));assert all(a['prefix']==p['prefix'] and a['mid']==p['mid'] and a['rank']==p['rank'] and a['features']==p['features'] for a in z.values())
            emit(fd,dict(root=r,order=oo,results=z))
            if (i+1)%50==0:print('PROGRESS',phase,ix,i+1,len(rr),n,round(time.monotonic()-t0,2),flush=True)
    tmp.replace(fn);return dict(file=fn.name,sha256=sha(fn),roots=len(rr),requests=n,NN=nn,physical=phys,visits=vis,elapsed=time.monotonic()-t0)
def collect(phase):
    co=read('COHORT.json');assert co['source']==sha(D/'SOURCE.jsonl.gz')
    mi=None;freeze=None
    if phase=='TRAIN_CAL_ATLAS':rr=[r for r in co['roots'] if r['group']!='TEST']
    elif phase=='TEST_ATLAS':
        f=read('FROZEN.json');freeze=sha(D/'FROZEN.json');rr=[r for r in co['roots'] if r['group']=='TEST']
    elif phase=='CAL_LIVE':
        f=read('MODEL_LOCK.json');freeze=sha(D/'MODEL_LOCK.json');mi=sha(D/'MODEL.json');assert f['model']==mi
        rr=[r for r in co['roots'] if r['group']=='CAL' and r['id'] in labelmap('TRAIN_CAL')]
    elif phase=='TEST_LIVE':
        f=read('FROZEN.json');freeze=sha(D/'FROZEN.json');mi=sha(D/'MODEL.json');assert f['model']==mi
        rr=[r for r in co['roots'] if r['group']=='TEST']
    else:raise ValueError(phase)
    assert rr
    with ProcessPoolExecutor(max_workers=SHARDS) as ex:fs=list(ex.map(shard,[(phase,i,rr[i::SHARDS],mi) for i in range(SHARDS)]))
    save(phase+'_LOCK.json',dict(cohort=sha(D/'COHORT.json'),identity=identity(),files=fs,freeze=freeze,**{k:sum(z[k] for z in fs) for k in ('roots','requests','NN','physical','visits')}))
    print('COLLECT',phase,{k:read(phase+'_LOCK.json')[k] for k in ('roots','requests','NN','physical','visits')},flush=True)
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
def datum(t,u):
    z=t['results'];a=z['0:0'];rank=a['rank'];return dict(f=a['features'],u=[u[m] for m in rank],allowed=a['allowed'],
      z=[[z[str(c)+':'+str(j)]['z'] for j in range(2)] for c in range(8)],cost=[[z[str(c)+':'+str(j)]['tail']['used_nn'] for j in range(2)] for c in range(8)])
def compatible(d,a):return len(d['u'])==len(a['u']) and all(lo<=u<=hi for u,(lo,hi) in zip(d['u'],a['allowed']))
def neighbors(donors,a):
    f=a['f'];return sorted(((i,sum((x-y)**2 for x,y in zip(d['f'],f))/len(f)) for i,d in enumerate(donors) if compatible(d,a)),key=lambda z:(z[1],z[0]))
def weights(ds,k,eps):
    if k:ds=ds[:k]
    if not ds:return [],1.
    raw=[1/(.01+d) if k else 1. for i,d in ds];s=sum(raw)
    return [(i,v/s*(1-eps)) for (i,d),v in zip(ds,raw)],eps
def argmax(v):
    i=0
    for j in range(1,len(v)):
        if v[j]>v[i]+1e-12:i=j
    return i
def infer(donors,a,k,eps,ds=None,prior_only=False):
    if ds is None:ds=neighbors(donors,a)
    ww,ep=weights(ds,k,eps);l=len(a['u']);allow=a['allowed'];u=a['u'];base=[(lo+hi)/4 for lo,hi in allow];q0=ep/math.prod(hi-lo+1 for lo,hi in allow)
    p=q0;cat=[[ep/(hi-lo+1) if lo<=j<=hi else 0. for j in range(3)] for lo,hi in allow]
    m=[ep*x for x in base]
    for i,w in ww:
        v=donors[i]['u'];p+=w*(v==u)
        for j,x in enumerate(v):m[j]+=w*x/2;cat[j][x]+=w
    prior=argmax(m) if ww else 0
    out=dict(prior=m,prior_rank=prior,prior_nll=-math.log(p),brier=sum(sum((v-(x==j))**2 for j,v in enumerate(z)) for x,z in zip(u,cat))/l,
        utility_mse=sum((x/2-y)**2 for x,y in zip(u,m))/l,prior_regret=(max(u)-u[prior])/2,prior_native_gain=(u[prior]-u[0])/2,donors=len(ww),ess=(1-ep)**2/sum(w*w for i,w in ww) if ww else 0.)
    if prior_only:return out
    ar=[]
    for c in range(8):
        mass=[ep/l]*l;ju=[q0/l]*l;j=[[ep*v/l for v in base] for _ in range(l)];cost=ep*32
        for i,w0 in ww:
            d=donors[i];v=d['u'];w=w0/2
            for r in range(2):
                z=d['z'][c][r];mass[z]+=w;ju[z]+=w*(v==u);cost+=w*d['cost'][c][r]
                for b,x in enumerate(v):j[z][b]+=w*x/2
        post=[argmax(z) if ww else 0 for z in j];g=max(0,sum(row[argmax(row)] for row in j)-m[argmax(m)])
        d=sum(row[z] for z,row in enumerate(j))-m[0]
        zz=a['z'][c];ar.append(dict(mass=mass,post_rank=post,joint_moments=j,G=g,D=d,cost=cost,net=g-.002*cost,
          conditional_nll=sum(-math.log(ju[z]/p) for z in zz)/2,marginal_nll=sum(-math.log(mass[z]) for z in zz)/2,
          posterior_nll=sum(-math.log(ju[z]/mass[z]) for z in zz)/2,actual_native=sum((u[z]-u[0])/2 for z in zz)/2,
          actual_post=sum((u[post[z]]-u[prior])/2 for z in zz)/2,
          actual_post_regret=sum((max(u)-u[post[z]])/2 for z in zz)/2))
    out['arms']=ar;out['continue_arm']=argmax([x['G'] for x in ar]) if ww else 0
    x=argmax([v['net'] for v in ar]);out['priced_arm']=x if ww and ar[x]['net']>1e-12 else 8
    out['cond_nll']=S.mean(z['conditional_nll'] for z in ar);out['marg_nll']=S.mean(z['marginal_nll'] for z in ar);out['joint_nll']=out['prior_nll']+out['cond_nll']
    return out
def hist_prior():
    k=R/'quartz_kinetic28_20261126';locks=[lock(k/n) for n in ('COHORT.json','TRAIN_CAL_ATLAS_LOCK.json','TRAIN_CAL_LABELS.json')]
    ll={z['root']['id']:{int(a):int(v+1) for a,v in z['independent'].items()} for z in json.loads((k/'TRAIN_CAL_LABELS.json').read_text())['rows'] if z['status']=='EXACT' and z['root']['group']=='TRAIN'}
    lk=json.loads((k/'TRAIN_CAL_ATLAS_LOCK.json').read_text());out=[]
    for fi in lk['files']:
        if fi['budget']!=32:continue
        p=k/fi['file'];assert sha(p)==fi['sha256'];locks.append(lock(p))
        for t in raw(p):
            r=t['root']
            if r['id'] not in ll:continue
            a=t['paths'][0];rank,f,allow=features(r,a['stages'][0]['lite'],a['proof']);out.append(dict(f=f,u=[ll[r['id']][x] for x in rank]))
    assert len(out)==len(ll)
    return out,locks
def fast_scores(donors,a,ds,k,eps):
    ww,ep=weights(ds,k,eps);l=len(a['u']);p0=ep/math.prod(hi-lo+1 for lo,hi in a['allowed']);p=p0
    # Only exact-vector joint probabilities for hyperparameter selection.
    joint=[[p0/l]*l for _ in range(8)]
    for i,w in ww:
        d=donors[i]
        if d['u']!=a['u']:continue
        p+=w
        for c in range(8):
            for z in d['z'][c]:joint[c][z]+=w/2
    c=sum(-math.log(joint[i][z]/p) for i in range(8) for z in a['z'][i])/16
    return -math.log(p),c
def fit():
    assert not (D/'CAL_LIVE_LOCK.json').exists() and not (D/'TEST_ATLAS_LOCK.json').exists()
    ll=labelmap('TRAIN_CAL');donors=[];tr=[];cal=[];calids=[]
    for t in rows('TRAIN_CAL_ATLAS'):
        r=t['root'];i=r['id']
        if i not in ll:continue
        x=datum(t,ll[i])
        if r['group']=='TRAIN':
            donors.append({k:x[k] for k in ('f','u','z','cost')});tr.append(i)
        else:cal.append(x);calids.append(i)
    assert donors and cal
    params=[(k,e) for k in (0,32,96,256) for e in (.02,.1,.25)];v={x:[] for x in params}
    for a in cal:
        ds=neighbors(donors,a)
        for k,e in params:v[k,e].append(fast_scores(donors,a,ds,k,e))
    res=[dict(k=k,eps=e,prior=S.mean(x for x,y in v[k,e]),cond=S.mean(y for x,y in v[k,e]),joint=S.mean(x+y for x,y in v[k,e])) for k,e in params]
    best=min(res,key=lambda x:x['joint']);pool=min([x for x in res if x['k']==0],key=lambda x:x['joint'])
    save('MODEL.json',dict(k=best['k'],eps=best['eps'],donors=donors))
    old,locks=hist_prior();save('HIST_PRIOR.json',dict(donors=old,locks=locks))
    save('CAL_SCORING.json',dict(n=len(cal),params=res,selected=best,pooled=pool,scope='joint log loss root-mean, no TEST data'))
    save('MODEL_LOCK.json',dict(cohort=sha(D/'COHORT.json'),identity=identity(),code=sha(D/'trial.py'),atlas=sha(D/'TRAIN_CAL_ATLAS_LOCK.json'),labels=sha(D/'TRAIN_CAL_LABELS.json'),model=sha(D/'MODEL.json'),historical=sha(D/'HIST_PRIOR.json'),cal_scoring=sha(D/'CAL_SCORING.json'),train_ids=tr,cal_ids=calids,pooled=pool))
    print('FIT',len(donors),len(cal),'best',best,'pooled',pool,'historical',len(old),flush=True)
def util(a,u):return u[a['best']]/2
def net(a,u):return util(a,u)-.002*a['NN_calls']
def calibrate():
    ll=labelmap('TRAIN_CAL');v=defaultdict(list);ids=[]
    for t in rows('CAL_LIVE'):
        i=t['root']['id'];ids.append(i);u=ll[i]
        for k in KINDS:v[k].append(S.mean(net(t['results'][k+':'+str(j)],u) for j in range(2)))
    means={k:S.mean(v[k]) for k in KINDS};fixed=max(('serial','tree4','continue'),key=means.__getitem__);priced=max(KINDS,key=means.__getitem__)
    f=read('MODEL_LOCK.json');save('FROZEN.json',dict(f,model_lock=sha(D/'MODEL_LOCK.json'),cal_live=sha(D/'CAL_LIVE_LOCK.json'),calnet=means,cal_actual_ids=sorted(ids),fixed=fixed,priced=priced))
    print('CAL_LIVE',len(ids),means,'fixed',fixed,'priced',priced,flush=True)
def qstats(a):
    q=a['queue'];d=Counter()
    if q:
        for k in ('logical','physical','coalesced','distinct','repeat_keys','batches','wait_ns','compute_ns','latency_ns'):d[k]=q[k]
        d['padded_lanes']=8*q['batches'];d['multi_batches']=sum(q['logical_hist'][2:])
        for j,z in enumerate(q['logical_hist']):d['LH'+str(j)]=z
        for j,z in enumerate(q['physical_hist']):d['PH'+str(j)]=z
    return d
def compact(a,u):
    d=qstats(a);h=a['end_hooks']['hooks'];l=a['end_hooks']['ledger'];p=a['mid_hooks']['ledger']
    return dict(u=util(a,u),native=u[a['native_best']]/2,opt=max(u.values())/2,net=net(a,u),NN=a['NN_calls'],physical=a['physical_calls'],tail_NN=a['NN_calls']-a['mid']['nn'],visits=a['visits'],arm=a['arm'],
      cpu=a['cpu_ns']/1e6,wall=a['wall_ns']/1e6,tail=a['tail_us']/1000,queue_overhead=(a['queue_setup_us']+a['queue_join_us'])/1000,control=a['control_us']/1000,
      n=a['end']['n'][:a['end']['len']],moves=a['end']['moves'][:a['end']['len']],best=a['best'],counts=dict(d),peak=a['queue']['peak_pending'] if a['queue'] else 0,
      sxvb=[h['select_changed'],a['end_hooks']['width_realized'],h['eval_changed'],l['applied']-p['applied']] if a['arm']<8 else [0]*4,
      receipt=l['transferred']-p['transferred'],blocked=l['blocked_receipt']-p['blocked_receipt'])
def aggregate(v):
    aa=[a for z in v for a in z];ct=Counter();arms=Counter();peak=Counter();sx=[0]*4;all4=0
    for a in aa:
        ct.update(a['counts']);arms[str(a['arm'])]+=1;peak[str(a['peak'])]+=1
        for i,x in enumerate(a['sxvb']):sx[i]+=x
        all4+=all(x>0 for x in a['sxvb'])
    out={k:S.mean(a[k] for a in aa) for k in ('u','native','net','NN','physical','visits','tail_NN','cpu','wall','tail','queue_overhead','control')}
    return dict(n=len(v),requests=len(aa),**out,regret=S.mean(a['opt']-a['u'] for a in aa),counts=dict(ct),arms=dict(arms),peaks=dict(peak),sxvb=sx,all4=all4,receipt=sum(a['receipt'] for a in aa),blocked=sum(a['blocked'] for a in aa),
        stopped=arms['8'],batch_mean=ct['logical']/ct['batches'] if ct['batches'] else None,physical_batch_mean=ct['physical']/ct['batches'] if ct['batches'] else None,wait_us=ct['wait_ns']/ct['logical']/1000 if ct['logical'] else None)
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
def verify_prediction(a,inf):
    p=a['prediction']
    assert p['prior_rank']==inf['prior_rank'] and p['continue_arm']==inf['continue_arm'] and p['priced_arm']==inf['priced_arm'] and p['donors']==inf['donors']
    for x,y in zip(p['prior'],inf['prior']):close(x,y)
    close(p['ess'],inf['ess'])
    for c,d in zip(p['arms'],inf['arms']):
        assert c['post_rank']==d['post_rank']
        for k in ('G','D','cost','net'):close(c[k],d[k])
        for x,y in zip(c['mass'],d['mass']):close(x,y)
        for x,y in zip(c['joint_moments'],d['joint_moments']):
            for q,v in zip(x,y):close(q,v)
def score():
    f=read('FROZEN.json');m=read('MODEL.json');old=read('HIST_PRIOR.json')['donors'];pool=f['pooled'];ll=labelmap('TEST')
    fixed={};live={};pred=[];comp=[];hist=[];verify_n=0
    for t in rows('TEST_ATLAS'):
        r=t['root'];i=r['id']
        if i not in ll:continue
        u=ll[i];x=datum(t,u);ds=neighbors(m['donors'],x)
        p=infer(m['donors'],x,m['k'],m['eps'],ds);pred.append((i,p))
        comp.append(infer(m['donors'],x,0,pool['eps'],ds))
        hist.append(infer(old,x,m['k'],m['eps'],prior_only=True))
        fixed[i]=[[compact(t['results'][str(c)+':'+str(j)],u) for j in range(2)] for c in range(8)]
    pm=dict(pred)
    for t in rows('TEST_LIVE'):
        i=t['root']['id']
        if i not in ll:continue
        u=ll[i];live[i]={k:[compact(t['results'][k+':'+str(j)],u) for j in range(2)] for k in KINDS}
        for k in KINDS[3:]:
            for j in range(2):verify_prediction(t['results'][k+':'+str(j)],pm[i]);verify_n+=1
    ids=sorted(ll);assert set(ids)==set(fixed)==set(live)
    d={str(c):[fixed[i][c] for i in ids] for c in range(8)}
    a={k:[live[i][k] for i in ids] for k in KINDS}
    def meanp(v):
        return {k:S.mean(z[k] for z in v) for k in ('prior_nll','brier','utility_mse','prior_regret','prior_native_gain','ess')}
    def fullp(v):
        o=meanp(v);o.update({k:S.mean(z[k] for z in v) for k in ('joint_nll','cond_nll','marg_nll')});o['no_donor']=sum(z['donors']==0 for z in v)
        o['arms']=[]
        for c in range(8):
            aa=[z['arms'][c] for z in v]
            x=[q['D'] for q in aa];y=[q['actual_native'] for q in aa]
            o['arms'].append({**{k:S.mean(q[k] for q in aa) for k in ('G','D','cost','conditional_nll','marginal_nll','posterior_nll','actual_native','actual_post','actual_post_regret')},
                'native_D_rmse':math.sqrt(S.mean((x-y)**2 for x,y in zip(x,y))), 'native_D_corr':S.correlation(x,y) if S.pstdev(x)>0 and S.pstdev(y)>0 else None})
        return o
    out=dict(n=len(ids),fixed_selected=f['fixed'],priced_selected=f['priced'],fixed={k:aggregate(v) for k,v in d.items()},live={k:aggregate(v) for k,v in a.items()},
        fixed_contrasts={x+':'+y:compare(d[x],d[y]) for x,y in [('1','0'),('2','0'),('3','1'),('4','2'),('5','4'),('7','6')]},
        live_contrasts={x+':'+y:compare(a[x],a[y]) for x,y in [('prior','short'),('continue','serial'),('continue','tree4'),('priced','short'),('priced','serial'),(f['fixed'],'serial'),(f['priced'],'short')]},
        fixed_repeats={k:repeat(v) for k,v in d.items()},live_repeats={k:repeat(v) for k,v in a.items()},forecast=fullp([p for i,p in pred]),pooled_forecast=fullp(comp),old_prior=meanp(hist),native_model_parity=verify_n)
    save('SUMMARY.json',out);print('SUMMARY',len(ids),f['fixed'],f['priced'],'forecast',out['forecast'],'pooled',out['pooled_forecast'],'old',out['old_prior'],flush=True)
    for k,v in out['fixed'].items():print('FIXED',k,v,flush=True)
    for k,v in out['live'].items():print('LIVE',k,v,flush=True)
    print('CONTRASTS',out['fixed_contrasts'],out['live_contrasts'],flush=True)
def diagnostic():
    rr=[r for r in read('COHORT.json')['roots'] if r['group']=='TRAIN'][:3];fn=D/'DIAGNOSTIC.jsonl.gz'
    assert not fn.exists()
    with Server() as s,gzip.open(fn,'wt',compresslevel=3) as fd:
        for r in rr:emit(fd,dict(root=r,results={str(c):req(s,r,arm=c,trace=True) for c in (3,4,5,7)}))
    save('DIAGNOSTIC_LOCK.json',dict(sha256=sha(fn),requests=len(rr)*4,roots=[r['id'] for r in rr]))
def timing():
    rr=[r for r in read('COHORT.json')['roots'] if r['group']=='TEST'][:12];fn=D/'TIMING.jsonl.gz';assert not fn.exists()
    with Server() as s,gzip.open(fn,'wt',compresslevel=3) as fd:
        for r in rr:
            oo=order(r,'TEST_ATLAS');res={str(c)+':'+str(j):req(s,r,arm=c) for c,j in oo}
            emit(fd,dict(root=r,order=oo,results=res))
    save('TIMING_LOCK.json',dict(sha256=sha(fn),requests=len(rr)*16,roots=[r['id'] for r in rr],scope='single server, host not isolated'))
def smoke():
    r=json.loads((R/'quartz_kinetic28_20261126/COHORT.json').read_text())['roots'][0]
    with Server() as s:
        for c in range(9):
            a=req(s,r,arm=c,trace=True);print('SMOKE',c,a['NN_calls'],a['physical_calls'],a['visits'],a['queue'] is not None,flush=True)
        for z in (dict(arm=9),dict(arm=0,policy='prior'),dict(policy='prior'),dict(arm=0,oracle=1)):
            assert 'error' in s.request(r['history'],**z)
    print('SMOKE 9 requests, 4 rejected',flush=True)
if __name__=='__main__':
    if sys.argv[1] in ('collect','labels'):globals()[sys.argv[1]](sys.argv[2])
    else:globals()[sys.argv[1]]()
