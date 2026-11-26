"""Large prospective K27 cohort: actual fixed and resident adaptive paths."""
from core import *
from collections import Counter,defaultdict
from fractions import Fraction as F
import gc
A=[(0,1,False,0),(0,4,False,0),(0,8,False,0),(15,1,False,0),(15,4,False,0),(15,8,False,0),(15,4,False,2),(15,8,False,2),(15,8,False,4),(2,4,False,0),(8,4,False,0),(1,4,False,0),(4,8,False,2),(15,2,False,2),(15,4,True,2)]
N=len(A);SHARDS=3;KINDS=('base','fast','static','dynamic')
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9),(a,b)
def bin0(s):
    n=sorted(s['n']);return 2*int(4*n[-1]>=3*s['visits'])+int(n[-1]-n[-2]>=4)
def req(srv,r,path=None,table=None,budget=8,trace=False):
    a=srv.request(r['history'],budget=budget,trace=trace,**({'path':list(path)} if path is not None else {'table':table}))
    return check(a,r,path,table,budget,trace)
def check(a,r,path=None,table=None,budget=8,trace=False):
    assert a.get('schema')=='quartz_k27_c4_v1',a
    assert a['history']==r['history'] and a['legal']==r['legal'] and a['proof']==proof(r['history'])
    pre=a['prefix'];assert pre['visits']==8 and a['prefix_bin']==bin0(pre)
    c=a['chosen'];assert c==(list(path) if path is not None else [table['first'][a['prefix_bin']],table['second'][c[0]][a['mid_bin']]])
    assert a['budget']==budget and len(a['stages'])==2
    ph=a['prefix_hooks'];assert pre['nn']==1+ph['hooks']['base_evals'] and ph['hooks']['extra_evals']==0
    assert ph['ledger']['iterations']==8 and ph['ledger']['transferred']==0
    prev=pre;last=ph
    for i,z in enumerate(a['stages']):
        mask,w,wave,rsv=A[c[i]]
        assert (z['mask'],z['workers'],z['wave'],z['reserve'])==A[c[i]] and z['action']==c[i]
        h=z['hooks'];b=z['budget'];s=z['state'];q=h['ledger'];l=last['ledger'];hh=h['hooks']
        assert b['target']==budget and b['cap']==4*budget and b['reserve']==rsv
        assert b['used_nn']==s['nn']-prev['nn']==hh['base_evals']+hh['extra_evals']
        assert b['used_nn']+b['unused']==budget and 0<=b['unused']<=budget
        assert b['visits']==s['visits']-prev['visits']==hh['base_evals']+hh['terminal_leaves'] and 0<b['visits']<=4*budget
        assert b['cause']==('NN_TARGET' if not b['unused'] else 'VISIT_CAP')
        if b['unused']:assert b['visits']==b['cap']
        assert b['protected_extra']+b['released']==rsv
        assert 0<=b['protected_base']<=budget-rsv and b['protected_extra']<=hh['extra_evals'] and b['protected_base']<=hh['base_evals']
        assert 0<=b['protected_visits']<=b['visits'] and (rsv>0 or b['protected_visits']==b['protected_base']==0)
        if rsv and b['protected_visits']<b['cap']:assert b['protected_base']==budget-rsv
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
    assert a['result']==prev and a['NN_calls']==prev['nn'] and a['visits']==prev['visits']<=8+8*budget
    cc=cert(a['proof']);assert a['best']==(prev['best'] if cc is None else cc)
    snap=a['snapshot'];lite=a['stages'][1]['lite']
    assert snap['root_n']==prev['visits'] and snap['evaluator_calls']==prev['nn'] and snap['best']==prev['best']
    es={e['move']:e for e in snap['edges']}
    for j,m in enumerate(prev['moves']):
        e=es.get(m,{});assert prev['n'][j]==e.get('n',0) and e.get('vl',0)==e.get('vvalue',0)==0
        close(prev['w'][j],e.get('w',0));close(lite['m2'][j],e.get('m2',0))
    return a
def perm(r):
    rr=list(range(N));random.Random(SEED+900000+r['id']).shuffle(rr);return rr
def order(r):
    rr=list(range(2*N));random.Random(SEED+800000+r['id']).shuffle(rr);return rr
def paths(r):return [(i,i) for i in range(N)]+[(i,j) for i,j in enumerate(perm(r))]
def opts(f,k):
    if k=='base':return dict(path=[0,0])
    if k=='fast':return dict(path=[2,2])
    if k=='static':return dict(path=[f['static']]*2)
    if k=='dynamic':return dict(table=f['table'])
    raise ValueError(k)
def shard_task(arg):
    phase,idx,roots,model=arg;fn=D/f'{phase}_{idx}.jsonl.gz';assert not fn.exists();tmp=fn.with_suffix('.tmp')
    n=nn=vis=0;t0=time.monotonic()
    with Server() as s,gzip.open(tmp,'wt',compresslevel=3) as fd:
        for j,r in enumerate(roots):
            if phase.endswith('ATLAS'):
                pp=paths(r);oo=order(r);aa=[None]*len(pp)
                for i in oo:aa[i]=req(s,r,path=pp[i]);n+=1;nn+=aa[i]['NN_calls'];vis+=aa[i]['visits']
                assert all(a['prefix']==aa[0]['prefix'] for a in aa)
                row=dict(root=r,permutation=perm(r),order=oo,paths=aa)
            else:
                oo=[(k,i) for k in KINDS for i in range(2 if phase=='TEST_LIVE' else 1)]
                random.Random(SEED+700000+r['id']).shuffle(oo);aa={}
                for k,i in oo:
                    a=req(s,r,**opts(model,k));aa[k+str(i)]=a;n+=1;nn+=a['NN_calls'];vis+=a['visits']
                assert all(a['prefix']==next(iter(aa.values()))['prefix'] for a in aa.values())
                row=dict(root=r,order=oo,results=aa)
            emit(fd,row)
            if (j+1)%25==0:print('PROGRESS',phase,idx,j+1,len(roots),n,round(time.monotonic()-t0,2),flush=True)
    tmp.replace(fn)
    return dict(file=fn.name,sha256=sha(fn),roots=len(roots),requests=n,NN=nn,visits=vis,elapsed=time.monotonic()-t0)
def collect(phase):
    co=read('COHORT.json');assert co['source']==sha(D/'SOURCE.jsonl.gz')
    if phase=='TRAIN_CAL_ATLAS':rr=[r for r in co['roots'] if r['group']!='TEST'];f=None
    elif phase=='TEST_ATLAS':
        f=read('FROZEN.json');assert f['model']==sha(D/'MODEL.json');rr=[r for r in co['roots'] if r['group']=='TEST']
    elif phase=='CAL_LIVE':
        f=read('MODEL.json');rr=[r for r in co['roots'] if r['group']=='CAL' and r['id'] in labelmap('TRAIN_CAL')]
    elif phase=='TEST_LIVE':
        f=read('FROZEN.json');rr=[r for r in co['roots'] if r['group']=='TEST']
    else:raise ValueError(phase)
    assert rr
    if f:assert f['identity']==identity() and f['code']==sha(D/'trial.py')
    with ProcessPoolExecutor(max_workers=SHARDS) as ex:res=list(ex.map(shard_task,[(phase,i,rr[i::SHARDS],f) for i in range(SHARDS)]))
    save(phase+'_LOCK.json',dict(cohort=sha(D/'COHORT.json'),identity=identity(),files=res,freeze=sha(D/'FROZEN.json') if phase.startswith('TEST') else sha(D/'MODEL.json') if phase=='CAL_LIVE' else None,
        **{k:sum(x[k] for x in res) for k in ('roots','requests','NN','visits')}))
    print('COLLECT',phase,read(phase+'_LOCK.json'),flush=True)
def rows(phase):
    lk=read(phase+'_LOCK.json')
    for f in lk['files']:
        p=D/f['file'];assert sha(p)==f['sha256'];yield from raw(p)
def label_worker(r):
    q=one_label(r);gc.collect();return q
def labels(part):
    lk=read(part+'_ATLAS_LOCK.json');co=read('COHORT.json');assert lk['cohort']==sha(D/'COHORT.json')
    rr=[r for r in co['roots'] if (r['group']=='TEST')==(part=='TEST')]
    with ProcessPoolExecutor(max_workers=8) as ex:
        out=list(ex.map(label_worker,rr,chunksize=1))
    save(part+'_LABELS.json',dict(atlas=sha(D/(part+'_ATLAS_LOCK.json')),cohort=lk['cohort'],counts=dict(Counter(x['status'] for x in out)),rows=out))
    print('LABELS',part,dict(Counter(x['status'] for x in out)),flush=True)
def labelmap(part):
    l=read(part+'_LABELS.json');assert l['atlas']==sha(D/(part+'_ATLAS_LOCK.json'))
    return {x['root']['id']:{int(a):F(v+1,2) for a,v in x['independent'].items()} for x in l['rows'] if x['status']=='EXACT'}
def util(a,u):return u[a['best']]
def net(a,u):return util(a,u)-F(a['NN_calls'],500)
def sc(a,j):return a['stages'][j]['budget']['used_nn']
def fit():
    assert not (D/'CAL_LIVE_LOCK.json').exists() and not (D/'TEST_ATLAS_LOCK.json').exists()
    ll=labelmap('TRAIN_CAL');v=[];train_ids=[];static=defaultdict(list)
    for t in rows('TRAIN_CAL_ATLAS'):
        r=t['root'];i=r['id']
        if r['group']!='TRAIN' or i not in ll:continue
        u=ll[i];train_ids.append(i)
        for k,a in enumerate(t['paths']):
            x,y=a['chosen'];v.append((x,y,a['prefix_bin'],a['mid_bin'],util(a,u)-F(sc(a,1),500),sc(a,0)))
            if k<N:static[k].append(net(a,u))
    assert train_ids
    glob=defaultdict(list);pool=defaultdict(list);cells=defaultdict(list)
    for a,b,p,h,y,n in v:glob[b].append(y);pool[a,b].append(y);cells[a,h,b].append(y)
    q2={};second=[[0]*8 for _ in range(N)];support={}
    for a in range(N):
        for h in range(8):
            valid=[]
            for b in range(N):
                rr=cells[a,h,b];mu=S.mean(pool[a,b]) if pool[a,b] else S.mean(glob[b])
                q2[a,h,b]=(sum(rr)+8*mu)/(len(rr)+8);support[a,h,b]=len(rr)
                if len(rr)>=4 or b==a:valid.append(b)
            second[a][h]=max(valid,key=lambda b:q2[a,h,b])
    by=defaultdict(list);g=defaultdict(list)
    for a,b,p,h,y,n in v:
        x=q2[a,h,second[a][h]]-F(n,500);by[p,a].append(x);g[a].append(x)
    q1={(p,a):(sum(by[p,a])+8*S.mean(g[a]))/(len(by[p,a])+8) for p in range(4) for a in range(N)}
    first=[max(range(N),key=lambda a:q1[p,a]) for p in range(4)]
    sm=[S.mean(static[i]) for i in range(N)];best=max(range(N),key=sm.__getitem__)
    f=dict(cohort=sha(D/'COHORT.json'),identity=identity(),code=sha(D/'trial.py'),atlas=sha(D/'TRAIN_CAL_ATLAS_LOCK.json'),labels=sha(D/'TRAIN_CAL_LABELS.json'),
        train_ids=sorted(train_ids),static=best,static_net=list(map(float,sm)),table=dict(first=first,second=second),
        q1=[[float(q1[p,a]) for a in range(N)] for p in range(4)],q2=[[[float(q2[a,h,b]) for b in range(N)] for h in range(8)] for a in range(N)],
        support=[[[support[a,h,b] for b in range(N)] for h in range(8)] for a in range(N)])
    save('MODEL.json',f);print('MODEL',len(train_ids),'static',best,'first',first,'nonzero cells',sum(v>0 for v in support.values()),flush=True)
def calibrate():
    f=read('MODEL.json');ll=labelmap('TRAIN_CAL');v=defaultdict(list);ids=[]
    for t in rows('CAL_LIVE'):
        i=t['root']['id'];ids.append(i);u=ll[i]
        for k in KINDS:v[k].append(net(t['results'][k+'0'],u))
    assert ids
    means={k:S.mean(v[k]) for k in KINDS};sel=max(KINDS,key=means.__getitem__)
    save('FROZEN.json',dict(f,model=sha(D/'MODEL.json'),cal=sha(D/'CAL_LIVE_LOCK.json'),cal_ids=sorted(ids),selected=sel,calnet={k:float(x) for k,x in means.items()}))
    print('CAL',len(ids),sel,means,flush=True)
def js(a,b):
    x={m:n/a['visits'] for m,n in zip(a['moves'],a['n'])};y={m:n/b['visits'] for m,n in zip(b['moves'],b['n'])}
    return sum((p*math.log(2*p/(p+q)) if p else 0.)/2+(q*math.log(2*q/(p+q)) if q else 0.)/2 for m in x.keys()|y.keys() for p,q in [(x.get(m,0),y.get(m,0))])
def compact(a,u):
    ct=Counter();pk=Counter();prev=a['prefix_hooks']['ledger'];st=[]
    for z in a['stages']:
        h=z['hooks'];b=z['budget'];l=h['ledger'];hh=h['hooks'];pk[str(h['peak'])]+=1
        for k in ('select_changed','width_nodes','refinements','eval_changed','extra_evals','base_evals','terminal_leaves'):ct[k]+=hh[k]
        ct['width_realized']+=h['width_realized']
        for k in ('reserve','released','protected_base','protected_extra','denied_refinement','unused'):ct[k]+=b[k]
        for k in ('transferred','blocked_receipt','applied','repeated_delivery'):ct[k]+=l[k]-prev[k]
        prev=l;st.append(dict(action=z['action'],workers=z['workers'],mask=z['mask'],reserve=z['reserve'],base=hh['base_evals'],extra=hh['extra_evals'],denied=b['denied_refinement'],refinements=hh['refinements'],released=b['released'],prot=b['protected_extra'],visits=b['visits'],NN=b['used_nn']))
    return dict(u=float(util(a,u)),opt=float(max(u.values())),net=float(net(a,u)),NN=a['NN_calls'],exact=int(sc(a,0)==sc(a,1)==a['budget']),visits=a['visits'],
        cpu=a['cpu_ns']/1e6,wall=a['wall_ns']/1e6,tail=sum(z['wall_us'] for z in a['stages'])/1000,control=sum(z['control_us'] for z in a['stages']),
        result=a['result'],best=a['best'],prefix=a['prefix']['best'],prefix_u=float(u[a['prefix']['best']]),chosen=a['chosen'],counts=dict(ct),peak=dict(pk),stages=st)
def agg(v):
    if not v:return dict(n=0)
    ct=Counter();pk=Counter();paths=Counter();workers=Counter();reserves=Counter()
    for a in v:
        ct.update(a['counts']);pk.update(a['peak']);paths[str(a['chosen'])]+=1
        for z in a['stages']:workers[str(z['workers'])]+=1;reserves[str(z['reserve'])]+=1
    n=len(v);d={k:S.mean(a[k] for a in v) for k in ('u','net','NN','visits','cpu','wall','tail','control')}
    return dict(n=n,**d,regret=S.mean(a['opt']-a['u'] for a in v),matched=sum(a['exact'] for a in v),nonoptimal=sum(a['u']<a['opt'] for a in v),prefix_nonoptimal=sum(a['prefix_u']<a['opt'] for a in v),
        switch=sum(a['chosen'][0]!=a['chosen'][1] for a in v),worker_switch=sum(a['stages'][0]['workers']!=a['stages'][1]['workers'] for a in v),counts=dict(ct),peak=dict(pk),paths=dict(paths),workers=dict(workers),reserves=dict(reserves))
def paired(v,b):
    if not v:return dict(n=0)
    d=[a['u']-z['u'] for a,z in zip(v,b)];rng=random.Random(SEED+123)
    # Root-family bootstrap; finite experimental cohort, not iid requests.
    bs=sorted(S.mean(rng.choices(d,k=len(d))) for _ in range(2000))
    return dict(n=len(d),gain=S.mean(d),net=S.mean(a['net']-z['net'] for a,z in zip(v,b)),better=sum(x>0 for x in d),worse=sum(x<0 for x in d),ci=[bs[49],bs[1949]])
def compare(v,b):
    mm=[(a,z) for a,z in zip(v,b) if a['exact'] and z['exact']]
    return dict(all=paired(v,b),matched=paired([a for a,z in mm],[z for a,z in mm]))
def repeat(v,b):
    return dict(n=len(v),utility_sd=math.sqrt(S.mean((a['u']-z['u'])**2/2 for a,z in zip(v,b))),NN_sd=math.sqrt(S.mean((a['NN']-z['NN'])**2/2 for a,z in zip(v,b))),
        JSD=S.mean(js(a['result'],z['result']) for a,z in zip(v,b)),disagree=sum(a['best']!=z['best'] for a,z in zip(v,b)),wall_sd=math.sqrt(S.mean((a['wall']-z['wall'])**2/2 for a,z in zip(v,b))))
def score():
    f=read('FROZEN.json');ll=labelmap('TEST');diag={};live={}
    for t in rows('TEST_ATLAS'):
        r=t['root'];i=r['id']
        if i in ll:diag[i]=[compact(a,ll[i]) for a in t['paths'][:N]]
    for t in rows('TEST_LIVE'):
        r=t['root'];i=r['id']
        if i in ll:live[i]={k:compact(a,ll[i]) for k,a in t['results'].items()}
    ids=sorted(diag);assert set(ids)==set(live)==set(ll)
    d={i:[diag[j][i] for j in ids] for i in range(N)}
    l={k:[live[j][k] for j in ids] for k in next(iter(live.values()))}
    res=dict(n=len(ids),ids=ids,selected=f['selected'],diag={str(i):agg(v) for i,v in d.items()},live={k:agg(v) for k,v in l.items()},
        diag_compare={str(i):compare(v,d[0]) for i,v in d.items()},live_compare={k:compare(l[k+'0'],l['base0']) for k in KINDS},
        repeats={k:repeat(l[k+'0'],l[k+'1']) for k in KINDS},protected={f'{a}:{b}':compare(d[a],d[b]) for a,b in [(6,4),(7,5),(8,5),(8,7),(14,6),(13,6)]})
    # compare selected to fast with actual independent CAL-chosen policy run.
    res['selected_vs_fast']=compare(l[f['selected']+'0'],l['fast0'])
    save('SUMMARY.json',res)
    print('SUMMARY',len(ids),'selected',f['selected'],flush=True)
    for i,v in res['diag'].items():print('DIAG',i,{k:v[k] for k in ('regret','NN','matched','u','net','cpu','wall','tail','counts')},flush=True)
    for k in KINDS:print('LIVE',k,res['live'][k+'0'],res['live_compare'][k],res['repeats'][k],flush=True)
def deeper():
    rr=[r for r in read('COHORT.json')['roots'] if r['group']=='TRAIN'][:24];fn=D/'DEEP.jsonl.gz';assert not fn.exists()
    with Server() as s,gzip.open(fn,'wt',compresslevel=3) as fd:
        for i,r in enumerate(rr):
            aa=[4,5,6,7,8,10,14];random.Random(SEED+600000+r['id']).shuffle(aa)
            z={}
            for a in aa:z[str(a)]=req(s,r,path=[a,a],budget=32,trace=i==0)
            emit(fd,dict(root=r,order=aa,results=z))
    save('DEEP_LOCK.json',dict(sha256=sha(fn),roots=[r['id'] for r in rr],requests=len(rr)*7,scope='TRAIN mechanism diagnostic, no model fit'))
    print('DEEP',len(rr)*7,flush=True)
def timing():
    f=read('FROZEN.json');rr=[r for r in read('COHORT.json')['roots'] if r['group']=='TEST'][:24]
    fn=D/'TIMING.jsonl.gz';assert not fn.exists()
    with Server() as s,gzip.open(fn,'wt',compresslevel=3) as fd:
        for r in rr:
            oo=[(str(a),i) for a in (0,2,5,7,8) for i in range(2)]+[('live',i) for i in range(2)]
            random.Random(SEED+500000+r['id']).shuffle(oo);z={}
            for k,i in oo:
                a=req(s,r,**(opts(f,f['selected']) if k=='live' else dict(path=[int(k)]*2)))
                z[k+':'+str(i)]=a
            emit(fd,dict(root=r,order=oo,results=z))
    save('TIMING_LOCK.json',dict(sha256=sha(fn),roots=[r['id'] for r in rr],requests=len(rr)*12,scope='single server sequential requests; concurrent host workloads not controlled'))
    print('TIMING',len(rr)*12,flush=True)
def smoke():
    r=json.loads((R/'quartz_kinetic26_20261126/COHORT.json').read_text())['roots'][0]
    with Server() as s:
        for i in range(N):
            a=req(s,r,path=[i,(i+1)%N]);print('SMOKE',i,a['NN_calls'],a['visits'],[z['budget'] for z in a['stages']],flush=True)
        for x in (dict(budget=8,path=[15,0]),dict(budget=7,path=[0,0]),dict(budget=8,path=[0,0],table={'first':[0]*4,'second':[[0]*8]*15}),dict(budget=8,path=[0,0],oracle=1)):
            assert 'error' in s.request(r['history'],**x)
    print('SMOKE 15+4 rejection',flush=True)
if __name__=='__main__':
    if sys.argv[1] in ('collect','labels'):globals()[sys.argv[1]](sys.argv[2])
    else:globals()[sys.argv[1]]()
