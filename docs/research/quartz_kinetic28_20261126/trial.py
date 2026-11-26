"""Prospective K28 scaling, signed response, cross-fitted actual STOP."""
from core import *
from collections import Counter,defaultdict
from fractions import Fraction as F
import gc
A=[(0,w,0,0) for w in (1,2,4,8)]+[(15,w,0,0) for w in (1,2,4,8)]+[(15,w,1,4) for w in (1,2,4,8)]+[(15,w,2,4) for w in (1,2,4,8)]+[(4,8,2,4),(11,4,0,0),(15,8,2,2),(15,4,2,2)]
N=len(A);STOP=N;BS=(8,16,32,64);SHARDS=3;KINDS=('serial','tree4','continue','stop')
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9),(a,b)
def bin0(s):
    n=sorted(s['n']);return 2*int(4*n[-1]>=3*s['visits'])+int(n[-1]-n[-2]>=4)
def req(srv,r,B,path=None,table=None,trace=False):
    a=srv.request(r['history'],budget=B,trace=trace,**({'path':list(path)} if path is not None else {'table':table}))
    return check(a,r,B,path,table,trace)
def check(a,r,B,path=None,table=None,trace=False):
    assert a.get('schema')=='quartz_k28_c4_v1',a
    assert a['history']==r['history'] and a['legal']==r['legal'] and a['proof']==proof(r['history'])
    pre=a['prefix'];assert pre['visits']==8 and a['prefix_bin']==bin0(pre)
    c=a['chosen'];assert c==(list(path) if path is not None else [table['first'][a['prefix_bin']],table['second'][c[0]][a['mid_bin']]])
    assert a['budget']==B and a['stopped']==(c[1]==STOP) and len(a['stages'])==(1 if a['stopped'] else 2)
    ph=a['prefix_hooks'];assert pre['nn']==1+ph['hooks']['base_evals'] and ph['hooks']['extra_evals']==0 and ph['ledger']['iterations']==8 and ph['ledger']['transferred']==0
    prev=pre;last=ph
    for i,z in enumerate(a['stages']):
        mask,w,mode,rsv=A[c[i]];assert (z['mask'],z['workers'],z['mode'],z['reserve'])==A[c[i]] and z['action']==c[i]
        h=z['hooks'];b=z['budget'];s=z['state'];q=h['ledger'];l=last['ledger'];hh=h['hooks']
        assert b['target']==B and b['cap']==4*B and b['reserve']==rsv
        assert b['used_nn']==s['nn']-prev['nn']==hh['base_evals']+hh['extra_evals']
        assert b['used_nn']+b['unused']==B and 0<=b['unused']<=B
        assert b['visits']==s['visits']-prev['visits']==hh['base_evals']+hh['terminal_leaves'] and 0<b['visits']<=4*B
        assert b['cause']==('NN_TARGET' if not b['unused'] else 'VISIT_CAP')
        if b['unused']:assert b['visits']==b['cap']
        assert b['protected_extra']+b['released']==rsv and b['protected_extra']+b['borrowed']==b['pre_extra']
        assert 0<=b['protected_base']+b['borrowed']<=B-rsv and b['pre_extra']<=hh['extra_evals'] and b['protected_base']<=hh['base_evals']
        assert 0<=b['protected_visits']<=b['visits'] and (rsv>0 or b['protected_visits']==b['protected_base']==0)
        if rsv and b['protected_visits']<b['cap']:assert b['protected_base']+b['borrowed']==B-rsv
        if mode!=2:assert b['borrowed']==0
        assert h['peak']<=w and h['peak']>=1 and q['iterations']==s['visits']
        assert q['eligible']==q['novel']+q['blocked_receipt'] and q['transferred']==q['receipt_items']>=l['transferred'] and q['repeated_delivery']==0
        if not mask&1:assert hh['select_nodes']==0
        if not mask&2:assert hh['width_nodes']==h['width_realized']==0
        if not mask&4:assert hh['extra_evals']==hh['refinements']==b['denied_refinement']==0
        if not mask&8:assert q['transferred']==l['transferred']
        assert (h['trace'] is not None)==(trace and i==1)
        if trace and i==1:assert len(h['trace'])==s['visits']
        assert z['bin']==2*bin0(s)+int(q['shared']>0) and (i or z['bin']==a['mid_bin'])
        assert sum(s['n'])==s['visits'] and s['moves']==prev['moves']
        lite=z['lite'];assert s['n']==lite['n'][:lite['len']] and s['moves']==lite['moves'][:lite['len']]
        assert s['best']==lite['best'] and s['nn']==lite['nn']
        prev=s;last=h
    assert a['result']==prev and a['NN_calls']==prev['nn'] and a['visits']==prev['visits']<=8+8*B
    cc=cert(a['proof']);assert a['best']==(prev['best'] if cc is None else cc)
    assert a['mid_best']==(a['stages'][0]['state']['best'] if cc is None else cc)
    if a['stopped']:assert a['best']==a['mid_best'] and a['NN_calls']==a['stages'][0]['state']['nn']
    snap=a['snapshot'];lite=a['stages'][-1]['lite']
    assert snap['root_n']==prev['visits'] and snap['evaluator_calls']==prev['nn'] and snap['best']==prev['best']
    es={e['move']:e for e in snap['edges']}
    for j,m in enumerate(prev['moves']):
        e=es.get(m,{});assert prev['n'][j]==e.get('n',0) and e.get('vl',0)==e.get('vvalue',0)==0
        close(prev['w'][j],e.get('w',0));close(lite['m2'][j],e.get('m2',0))
    return a
def paths(r,B):
    x=list(range(N));y=x.copy();rng=random.Random(SEED+900000+r['id']*100+B);rng.shuffle(x);rng.shuffle(y)
    return [(i,i) for i in range(N)]+list(zip(x[:10],y[:10]))
def order(r,B,n):
    rr=list(range(n));random.Random(SEED+800000+r['id']*100+B).shuffle(rr);return rr
def opts(f,k):
    if k=='serial':return dict(path=[0,0])
    if k=='tree4':return dict(path=[2,2])
    return dict(table=f['tables'][k])
def shard_task(arg):
    phase,B,idx,roots,model=arg;fn=D/f'{phase}_B{B}_{idx}.jsonl.gz';assert not fn.exists();tmp=fn.with_suffix('.tmp')
    n=nn=vis=0;t0=time.monotonic()
    with Server() as s,gzip.open(tmp,'wt',compresslevel=3) as fd:
        for j,r in enumerate(roots):
            if phase.endswith('ATLAS'):
                pp=paths(r,B);oo=order(r,B,len(pp));aa=[None]*len(pp)
                for i in oo:aa[i]=req(s,r,B,path=pp[i]);n+=1;nn+=aa[i]['NN_calls'];vis+=aa[i]['visits']
                assert all(a['prefix']==aa[0]['prefix'] for a in aa)
                row=dict(root=r,budget=B,path_order=pp,order=oo,paths=aa)
            else:
                oo=[(k,i) for k in KINDS for i in range(2 if phase=='TEST_LIVE' else 1)]
                random.Random(SEED+700000+r['id']*100+B).shuffle(oo);aa={}
                for k,i in oo:
                    a=req(s,r,B,**opts(model,k));aa[k+str(i)]=a;n+=1;nn+=a['NN_calls'];vis+=a['visits']
                assert all(a['prefix']==next(iter(aa.values()))['prefix'] for a in aa.values())
                row=dict(root=r,budget=B,order=oo,results=aa)
            emit(fd,row)
            if (j+1)%25==0:print('PROGRESS',phase,B,idx,j+1,len(roots),n,round(time.monotonic()-t0,2),flush=True)
    tmp.replace(fn)
    return dict(file=fn.name,budget=B,sha256=sha(fn),roots=len(roots),requests=n,NN=nn,visits=vis,elapsed=time.monotonic()-t0)
def collect(phase):
    co=read('COHORT.json');assert co['source']==sha(D/'SOURCE.jsonl.gz');f=None
    if phase=='TRAIN_CAL_ATLAS':rr=[r for r in co['roots'] if r['group']!='TEST']
    elif phase=='TEST_ATLAS':
        f=read('FROZEN.json');assert f['model']==sha(D/'MODEL.json');rr=[r for r in co['roots'] if r['group']=='TEST']
    elif phase=='CAL_LIVE':
        f=read('MODEL.json');rr=[r for r in co['roots'] if r['group']=='CAL' and r['id'] in labelmap('TRAIN_CAL')]
    elif phase=='TEST_LIVE':
        f=read('FROZEN.json');rr=[r for r in co['roots'] if r['group']=='TEST']
    else:raise ValueError(phase)
    assert rr
    if f:assert f['identity']==identity() and f['code']==sha(D/'trial.py')
    res=[]
    for B in BS:
        model=None if f is None else f['budgets'][str(B)]
        with ProcessPoolExecutor(max_workers=SHARDS) as ex:res+=list(ex.map(shard_task,[(phase,B,i,rr[i::SHARDS],model) for i in range(SHARDS)]))
        print('BUDGET_DONE',phase,B,flush=True)
    save(phase+'_LOCK.json',dict(cohort=sha(D/'COHORT.json'),identity=identity(),files=res,freeze=sha(D/'FROZEN.json') if phase.startswith('TEST') else sha(D/'MODEL.json') if phase=='CAL_LIVE' else None,
        **{k:sum(x[k] for x in res) for k in ('roots','requests','NN','visits')}))
    print('COLLECT',phase,{k:read(phase+'_LOCK.json')[k] for k in ('roots','requests','NN','visits')},flush=True)
def rows(phase,B):
    lk=read(phase+'_LOCK.json')
    for f in lk['files']:
        if f['budget']!=B:continue
        p=D/f['file'];assert sha(p)==f['sha256'];yield from raw(p)
def label_worker(r):
    q=one_label(r);gc.collect();return q
def labels(part):
    lk=read(part+'_ATLAS_LOCK.json');co=read('COHORT.json');assert lk['cohort']==sha(D/'COHORT.json')
    rr=[r for r in co['roots'] if (r['group']=='TEST')==(part=='TEST')]
    with ProcessPoolExecutor(max_workers=8) as ex:out=list(ex.map(label_worker,rr,chunksize=1))
    save(part+'_LABELS.json',dict(atlas=sha(D/(part+'_ATLAS_LOCK.json')),cohort=lk['cohort'],counts=dict(Counter(x['status'] for x in out)),rows=out))
    print('LABELS',part,dict(Counter(x['status'] for x in out)),flush=True)
def labelmap(part):
    l=read(part+'_LABELS.json');assert l['atlas']==sha(D/(part+'_ATLAS_LOCK.json'))
    return {x['root']['id']:{int(a):F(v+1,2) for a,v in x['independent'].items()} for x in l['rows'] if x['status']=='EXACT'}
def util(a,u):return u[a['best']]
def net(a,u):return util(a,u)-F(a['NN_calls'],500)
def sc(a,j):return a['stages'][j]['budget']['used_nn'] if j<len(a['stages']) else 0
def mid(a,u):return u[a['mid_best']]
def prefix_u(a,u):
    cc=cert(a['proof']);return u[a['prefix']['best'] if cc is None else cc]
def lower(v):
    n=len(v)
    return float(S.mean(v))-2*S.stdev(v)/math.sqrt(n) if n>=2 else -1e9
def byroot(v):
    d=defaultdict(list)
    for i,x in v:d[i].append(x)
    return [S.mean(x) for x in d.values()]
# v rows: id,first,second,prefixbin,midbin,net stage1,net stage2.
def second(v):
    glob=defaultdict(list);pool=defaultdict(list);cells=defaultdict(list)
    for i,a,b,p,h,y1,y2 in v:glob[b].append((i,y2));pool[a,b].append((i,y2));cells[a,h,b].append((i,y2))
    q={};lo={};support={};tabs={k:[[0]*8 for _ in range(N)] for k in ('continue','stop')}
    for a in range(N):
        for h in range(8):
            for b in range(N):
                rr=byroot(cells[a,h,b]);mu=S.mean(byroot(pool[a,b] or glob[b]))
                q[a,h,b]=(sum(rr)+8*mu)/(len(rr)+8);lo[a,h,b]=lower(rr);support[a,h,b]=len(rr)
            valid=[b for b in range(N) if support[a,h,b]>=6 or b==a]
            tabs['continue'][a][h]=max(valid,key=lambda b:q[a,h,b])
            valid=[b for b in range(N) if support[a,h,b]>=6 and lo[a,h,b]>0 and q[a,h,b]>0]
            tabs['stop'][a][h]=max(valid,key=lambda b:q[a,h,b]) if valid else STOP
    return q,lo,support,tabs
def fit_budget(B,ll):
    v=[];ids=[]
    for t in rows('TRAIN_CAL_ATLAS',B):
        r=t['root'];i=r['id']
        if r['group']!='TRAIN' or i not in ll:continue
        u=ll[i];ids.append(i)
        for a in t['paths']:
            x,y=a['chosen'];v.append((i,x,y,a['prefix_bin'],a['mid_bin'],mid(a,u)-prefix_u(a,u)-F(sc(a,0),500),util(a,u)-mid(a,u)-F(sc(a,1),500)))
    assert ids
    folds=list(sorted(ids));random.Random(SEED+110000+B).shuffle(folds);split={i:j%2 for j,i in enumerate(folds)}
    vv=[defaultdict(list),defaultdict(list)]
    for k in range(2):
        q,lo,sup,tabs=second([x for x in v if split[x[0]]==k])
        for i,a,b,p,h,y1,y2 in v:
            if split[i]==k:continue
            for j,kind in enumerate(('continue','stop')):
                b0=tabs[kind][a][h];vv[j][p,a].append((i,y1+(0 if b0==STOP else q[a,h,b0])))
    q,lo,sup,tabs=second(v);out={};q1={}
    for j,kind in enumerate(('continue','stop')):
        p0=defaultdict(list)
        for (p,a),z in vv[j].items():p0[a]+=z
        s={(p,a):(sum(byroot(vv[j][p,a]))+8*S.mean(byroot(p0[a])))/(len(byroot(vv[j][p,a]))+8) for p in range(4) for a in range(N)}
        first=[max(range(N),key=lambda a:s[p,a]) for p in range(4)]
        out[kind]=dict(first=first,second=tabs[kind]);q1[kind]=[[float(s[p,a]) for a in range(N)] for p in range(4)]
    return dict(train_ids=sorted(ids),fold0=sorted(i for i,k in split.items() if k==0),tables=out,q1=q1,
        q2=[[[float(q[a,h,b]) for b in range(N)] for h in range(8)] for a in range(N)],lower=[[[lo[a,h,b] for b in range(N)] for h in range(8)] for a in range(N)],
        support=[[[sup[a,h,b] for b in range(N)] for h in range(8)] for a in range(N)])
def fit():
    assert not (D/'CAL_LIVE_LOCK.json').exists() and not (D/'TEST_ATLAS_LOCK.json').exists()
    ll=labelmap('TRAIN_CAL');bs={str(B):fit_budget(B,ll) for B in BS}
    save('MODEL.json',dict(cohort=sha(D/'COHORT.json'),identity=identity(),code=sha(D/'trial.py'),atlas=sha(D/'TRAIN_CAL_ATLAS_LOCK.json'),labels=sha(D/'TRAIN_CAL_LABELS.json'),budgets=bs))
    print('MODEL',{B:{k:v['first'] for k,v in z['tables'].items()} for B,z in bs.items()},flush=True)
def calibrate():
    f=read('MODEL.json');ll=labelmap('TRAIN_CAL');out={}
    for B in BS:
        v=defaultdict(list);ids=[]
        for t in rows('CAL_LIVE',B):
            i=t['root']['id'];ids.append(i);u=ll[i]
            for k in KINDS:v[k].append(net(t['results'][k+'0'],u))
        means={k:S.mean(v[k]) for k in KINDS};base=max(KINDS[:2],key=means.__getitem__)
        gains={k:[a-b for a,b in zip(v[k],v[base])] for k in KINDS}
        admiss=[base]+[k for k in KINDS[2:] if lower(gains[k])>0]
        sel=max(admiss,key=means.__getitem__)
        out[str(B)]=dict(f['budgets'][str(B)],cal_ids=sorted(ids),calnet={k:float(x) for k,x in means.items()},cal_lower={k:lower(gains[k]) for k in KINDS},baseline=base,selected=sel)
        print('CAL',B,len(ids),base,sel,{k:float(x) for k,x in means.items()},{k:lower(gains[k]) for k in KINDS},flush=True)
    save('FROZEN.json',dict(f,model=sha(D/'MODEL.json'),cal=sha(D/'CAL_LIVE_LOCK.json'),budgets=out))
def js(a,b):
    x={m:n/a['visits'] for m,n in zip(a['moves'],a['n'])};y={m:n/b['visits'] for m,n in zip(b['moves'],b['n'])}
    return sum((p*math.log(2*p/(p+q)) if p else 0.)/2+(q*math.log(2*q/(p+q)) if q else 0.)/2 for m in x.keys()|y.keys() for p,q in [(x.get(m,0),y.get(m,0))])
def geometry(a,b):
    assert a['moves']==b['moves']
    x=[.5+.5*w/(n+4) for w,n in zip(a['w'],a['n'])];y=[.5+.5*w/(n+4) for w,n in zip(b['w'],b['n'])]
    def calc(x):
        e=[math.exp(v/.25) for v in x];return .25*math.log(sum(e)/len(e)),[v/sum(e) for v in e]
    pa,p=calc(x);pb,q=calc(y);d=sum(v*(b-a) for v,a,b in zip(p,x,y));g=.25*sum(v*math.log(v/w) for v,w in zip(p,q));close(pb-pa,d+g);return d,g,pb-pa
def compact(a,u):
    ct=Counter();pk=Counter();prev=a['prefix_hooks']['ledger'];st=[]
    for z in a['stages']:
        h=z['hooks'];b=z['budget'];l=h['ledger'];hh=h['hooks'];pk[str(h['peak'])]+=1
        for k in ('select_changed','width_nodes','refinements','eval_changed','extra_evals','base_evals','terminal_leaves'):ct[k]+=hh[k]
        ct['width_realized']+=h['width_realized']
        for k in ('reserve','released','protected_base','protected_extra','borrowed','denied_refinement','unused'):ct[k]+=b[k]
        for k in ('transferred','blocked_receipt','applied','repeated_delivery'):ct[k]+=l[k]-prev[k]
        prev=l;st.append(dict(action=z['action'],workers=z['workers'],mask=z['mask'],mode=z['mode'],reserve=z['reserve'],NN=b['used_nn']))
    return dict(u=float(util(a,u)),opt=float(max(u.values())),net=float(net(a,u)),NN=a['NN_calls'],exact=int(len(a['stages'])==2 and sc(a,0)==sc(a,1)==a['budget']),visits=a['visits'],
        cpu=a['cpu_ns']/1e6,wall=a['wall_ns']/1e6,tail=sum(z['wall_us'] for z in a['stages'])/1000,control=sum(z['control_us'] for z in a['stages']),
        result=a['result'],best=a['best'],chosen=a['chosen'],stopped=a['stopped'],counts=dict(ct),peak=dict(pk),stages=st)
def agg(v):
    ct=Counter();pk=Counter();paths=Counter();workers=Counter();reserves=Counter()
    for a in v:
        ct.update(a['counts']);pk.update(a['peak']);paths[str(a['chosen'])]+=1
        for z in a['stages']:workers[str(z['workers'])]+=1;reserves[str(z['reserve'])]+=1
    n=len(v);d={k:S.mean(a[k] for a in v) for k in ('u','net','NN','visits','cpu','wall','tail','control')}
    return dict(n=n,**d,regret=S.mean(a['opt']-a['u'] for a in v),matched=sum(a['exact'] for a in v),nonoptimal=sum(a['u']<a['opt'] for a in v),
        stopped=sum(a['stopped'] for a in v),switch=sum(a['chosen'][0]!=a['chosen'][1] for a in v),worker_switch=sum(len(a['stages'])==2 and a['stages'][0]['workers']!=a['stages'][1]['workers'] for a in v),
        counts=dict(ct),peak=dict(pk),paths=dict(paths),workers=dict(workers),reserves=dict(reserves))
def paired(v,b):
    if not v:return dict(n=0)
    d=[a['u']-z['u'] for a,z in zip(v,b)];rng=random.Random(SEED+123);bs=sorted(S.mean(rng.choices(d,k=len(d))) for _ in range(1500))
    return dict(n=len(d),gain=S.mean(d),net=S.mean(a['net']-z['net'] for a,z in zip(v,b)),better=sum(x>0 for x in d),worse=sum(x<0 for x in d),ci=[bs[37],bs[1462]])
def compare(v,b):
    mm=[(a,z) for a,z in zip(v,b) if a['exact'] and z['exact']]
    return dict(all=paired(v,b),matched=paired([a for a,z in mm],[z for a,z in mm]))
def repeat(v,b):
    return dict(n=len(v),utility_sd=math.sqrt(S.mean((a['u']-z['u'])**2/2 for a,z in zip(v,b))),NN_sd=math.sqrt(S.mean((a['NN']-z['NN'])**2/2 for a,z in zip(v,b))),
        JSD=S.mean(js(a['result'],z['result']) for a,z in zip(v,b)),disagree=sum(a['best']!=z['best'] for a,z in zip(v,b)),wall_sd=math.sqrt(S.mean((a['wall']-z['wall'])**2/2 for a,z in zip(v,b))))
def forecast(v):
    y=[a[0] for a in v];p=[a[1] for a in v];g=[a[2] for a in v];dr=[a[3] for a in v]
    def corr(a,b):return S.correlation(a,b) if S.pstdev(a)>0 and S.pstdev(b)>0 else None
    cuts=(-.1,-.025,0,.025,.1);bb=defaultdict(list)
    for a,b,c,d in v:bb[sum(b>=x for x in cuts)].append((a,b))
    return dict(n=len(v),actual=S.mean(y),pred=S.mean(p),bias=S.mean(b-a for a,b in zip(y,p)),rmse=math.sqrt(S.mean((b-a)**2 for a,b in zip(y,p))),
        positive=sum(a>0 for a in y),negative=sum(a<0 for a in y),geom_mean=S.mean(g),drift_mean=S.mean(dr),geom_corr=corr(g,y),prediction_corr=corr(p,y),
        bins={str(i):dict(n=len(a),actual=S.mean(x for x,y in a),pred=S.mean(y for x,y in a)) for i,a in bb.items()})
def score():
    f=read('FROZEN.json');ll=labelmap('TEST');out={}
    for B in BS:
        z=f['budgets'][str(B)];diag={};live={};cal=[]
        for t in rows('TEST_ATLAS',B):
            r=t['root'];i=r['id']
            if i not in ll:continue
            u=ll[i];diag[i]=[compact(a,u) for a in t['paths'][:N]]
            for a in t['paths']:
                x,y=a['chosen'];h=a['mid_bin'];d,g,p=geometry(a['stages'][0]['state'],a['result'])
                cal.append((float(util(a,u)-mid(a,u)-F(sc(a,1),500)),z['q2'][x][h][y],g,d))
        for t in rows('TEST_LIVE',B):
            r=t['root'];i=r['id']
            if i in ll:live[i]={k:compact(a,ll[i]) for k,a in t['results'].items()}
        ids=sorted(diag);assert set(ids)==set(live)==set(ll)
        d={i:[diag[j][i] for j in ids] for i in range(N)};l={k:[live[j][k] for j in ids] for k in next(iter(live.values()))}
        res=dict(n=len(ids),ids=ids,baseline=z['baseline'],selected=z['selected'],diag={str(i):agg(v) for i,v in d.items()},live={k:agg(v) for k,v in l.items()},
            live_compare={k:compare(l[k+'0'],l[z['baseline']+'0']) for k in KINDS},repeats={k:repeat(l[k+'0'],l[k+'1']) for k in KINDS},
            protected={f'{a}:{b}':compare(d[a],d[b]) for w in range(4) for a,b in [(12+w,8+w),(12+w,4+w)]},forecast=forecast(cal))
        out[str(B)]=res
        print('B',B,'selected',z['selected'],'baseline',z['baseline'],'n',len(ids),'forecast',res['forecast'],flush=True)
        for i in (0,2,3,7,11,15,18):print('DIAG',B,i,{k:res['diag'][str(i)][k] for k in ('regret','NN','matched','u','net','cpu','wall','tail','counts')},flush=True)
        for k in KINDS:print('LIVE',B,k,res['live'][k+'0'],res['live_compare'][k],res['repeats'][k],flush=True)
    save('SUMMARY.json',dict(budgets=out))
def diagnostic():
    rr=[r for r in read('COHORT.json')['roots'] if r['group']=='TRAIN'][:1];fn=D/'DEEP.jsonl.gz';assert not fn.exists()
    with Server() as s,gzip.open(fn,'wt',compresslevel=3) as fd:
        for r in rr:
            z={str(a):req(s,r,64,path=[a,a],trace=True) for a in (7,11,15,17,18)}
            emit(fd,dict(root=r,budget=64,results=z))
    save('DEEP_LOCK.json',dict(sha256=sha(fn),roots=[r['id'] for r in rr],requests=len(rr)*5,scope='predeclared TRAIN trace'))
def timing():
    f=read('FROZEN.json');rr=[r for r in read('COHORT.json')['roots'] if r['group']=='TEST'][:12];fn=D/'TIMING.jsonl.gz';assert not fn.exists()
    with Server() as s,gzip.open(fn,'wt',compresslevel=3) as fd:
        for B in BS:
            z=f['budgets'][str(B)]
            for r in rr:
                oo=[(str(a),i) for a in (0,2,7,11,15) for i in range(2)]+[('selected',i) for i in range(2)]
                random.Random(SEED+500000+r['id']*100+B).shuffle(oo);res={}
                for k,i in oo:res[k+':'+str(i)]=req(s,r,B,**(opts(z,z['selected']) if k=='selected' else dict(path=[int(k)]*2)))
                emit(fd,dict(root=r,budget=B,order=oo,results=res))
    save('TIMING_LOCK.json',dict(sha256=sha(fn),roots=[r['id'] for r in rr],requests=len(rr)*len(BS)*12,scope='single server, no full host isolation'))
def smoke():
    r=json.loads((R/'quartz_kinetic27_20261126/COHORT.json').read_text())['roots'][0]
    with Server() as s:
        for B in BS:
            for i in range(N):
                a=req(s,r,B,path=[i,(i+1)%N]);print('SMOKE',B,i,a['NN_calls'],a['visits'],flush=True)
            a=req(s,r,B,path=[15,STOP]);assert a['stopped']
        for x in (dict(budget=8,path=[STOP,0]),dict(budget=7,path=[0,0]),dict(budget=8,path=[0,0],table={'first':[0]*4,'second':[[0]*8]*N}),dict(budget=8,path=[0,21])):
            assert 'error' in s.request(r['history'],**x)
    print('SMOKE 84 requests+4 rejections',flush=True)
if __name__=='__main__':
    if sys.argv[1] in ('collect','labels'):globals()[sys.argv[1]](sys.argv[2])
    else:globals()[sys.argv[1]]()
