"""Actual K29 paired response law, proper scoring and queued CPU experiment."""
from core import *
from collections import Counter,defaultdict
from fractions import Fraction as F
import gc
A={1:(0,1,0),4:(0,1,0),5:(0,1,0),6:(0,1,0),7:(15,8,4),8:(15,8,4),9:(15,8,4)}
N=11;STOP=0;SHARDS=2
def close(a,b):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9),(a,b)
def features(r,lite,pr):
    anchor=cert(pr);anchor=lite['best'] if anchor is None else anchor
    ix=sorted(range(lite['len']),key=lambda i:(lite['moves'][i]!=anchor,-lite['n'][i],i))
    ht=c4replay(r['history'])[1];p={m:(lo,hi) for m,lo,hi in pr};rank=[];f=[len(r['history'])/42];allowed=[]
    for i in ix:
        m=lite['moves'][i];n=lite['n'][i];lo,hi=p[m];rank.append(m);allowed.append([int(2*lo),int(2*hi)])
        f.extend((math.sqrt(n/lite['visits']),math.sqrt(max(0,lite['prior'][i])),.5+.5*lite['w'][i]/(n+4),math.sqrt(max(0,lite['m2'][i])/(n+1))/2,ht[m]/6,abs(m-3)/3,lo,hi))
    return rank,f,allowed
def graph_rep(a,rep,anchor):
    aa=rep['actions'];L=max(l for m,(l,u) in aa);allowed=[m for m,(l,u) in aa if u>=L]
    w=next((m for m,(l,u) in aa if all(j==m or l>=v for j,(_,v) in aa)),None)
    risk=min(max(0,max((v for j,(_,v) in aa if m!=j),default=-1)-l) for m,(l,u) in aa)
    assert rep['allowed']==allowed and rep['witness']==w and rep['risk']==[risk,sum(u-l for m,(l,u) in aa)]
    assert rep['interval']==[max(l for m,(l,u) in aa),max(u for m,(l,u) in aa)]
    assert rep['rule_work']==rep['checks']+rep['replay']
    assert rep['best']==w if w is not None else rep['best'] in allowed
    if w is None and anchor in allowed:assert rep['best']==anchor
def check(a,r,arm,trace=False):
    assert a.get('schema')=='quartz_k32_c4_v1',a
    assert a['history']==r['history'] and a['legal']==r['legal'] and a['proof']==proof(r['history']) and a['arm']==arm
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
    if arm in A:
        mask,w,res=A[arm];h=eh['hooks'];b=a['tail'];q=eh['ledger'];p=mh['ledger']
        assert b['target']==32 and b['cap']==128 and b['reserve']==res and b['used_nn']+b['unused']==32
        assert b['used_nn']==end['nn']-mid['nn']==h['base_evals']+h['extra_evals']
        assert b['visits']==end['visits']-mid['visits']==h['base_evals']+h['terminal_leaves'] and 0<=b['visits']<=128
        assert b['cause'] in ('CERTIFIED','NN_TARGET','VISIT_CAP')
        if b['cause']=='CERTIFIED':assert 4<=arm<=8 and a['end_graph']['witness'] is not None
        elif b['cause']=='NN_TARGET':assert b['unused']==0
        else:assert b['visits']==128 and b['unused']>0
        if res:
            assert b['protected_extra']+b['released']==res and b['protected_extra']+b['borrowed']==b['pre_extra']
            assert b['protected_base']+b['borrowed']<=32-res and b['pre_extra']<=h['extra_evals']
        assert 0<=eh['peak']<=w and (b['visits']==0)==(eh['peak']==0) and q['iterations']==end['visits']
        assert q['eligible']==q['novel']+q['blocked_receipt'] and q['transferred']==q['receipt_items']>=p['transferred'] and q['repeated_delivery']==0
        assert h['graph_solved_skips']==0
        if not mask:assert h['select_nodes']==h['width_nodes']==h['refinements']==h['extra_evals']==0 and q['transferred']==0
        if arm>=4:
            waves=b['rounds'];assert sum(t['nn'] for t in waves)==b['used_nn'] and sum(t['visits'] for t in waves)==b['visits']
            cum=0
            for i,t in enumerate(waves):
                cum+=t['nn'];assert 0<t['visits']<=8 and cum+t['shared']+t['protected']==32
                if t['witness'] is not None:assert i==len(waves)-1
            if b['cause']=='CERTIFIED' and waves:assert waves[-1]['witness']==a['end_graph']['witness']
            assert sum(t['forced'] for t in waves)==len(a['end_graph']['forced'] if a['end_graph'] else [])
        if arm in (8,9):
            q=a['queue'];assert q['width']==8 and q['dedup'] and q['logical']==b['used_nn']
            assert q['logical']==q['physical']+q['coalesced']==q['distinct']+q['repeat_keys']
            assert q['physical']>=q['distinct'] and q['coalesced']<=q['repeat_keys']
            assert sum(q['logical_hist'])==sum(q['physical_hist'])==q['batches']
            assert sum(i*v for i,v in enumerate(q['logical_hist']))==q['logical'] and sum(i*v for i,v in enumerate(q['physical_hist']))==q['physical']
            assert q['peak_pending']<=w and q['latency_ns']>=q['wait_ns']>=0 and q['compute_ns']>=0
            kh=[0]*9
            for i,n in enumerate(q['physical_hist']):
                if i:kh[1<<(i-1).bit_length()]+=n
            assert q['specialized'] and kh==q['kernel_hist'] and q['kernel_lanes']==sum(i*n for i,n in enumerate(kh))
            assert q['physical']<=q['kernel_lanes']<=8*q['batches']
            assert (q['trace'] is not None)==trace and q['parity_checks']==(q['physical'] if trace else 0)
        else:assert a['queue'] is None
    else:
        assert a['tail'] is None and a['queue'] is None and end==mid and mh['ledger']==eh['ledger']
    if 2<=arm<=8:
        m=a['mid_graph'];e=a['end_graph'];graph_rep(a,m,a['mid_best']);graph_rep(a,e,a['native_best'])
        assert len(e['paths'])==end['visits'] and m['paths'] is None and a['rule_work']==e['rule_work'] and a['best']==e['best']
        assert m['forced']==[] and e['checks']>=m['checks'] and e['replay']>=m['replay']
        assert e['risk'][0]<=m['risk'][0] and e['risk'][1]<=m['risk'][1]
        for (x,(l,u)),(y,(ll,uu)) in zip(m['actions'],e['actions']):assert x==y and l<=ll<=uu<=u
        for z in e['forced']:assert z['mode'] in (2,3) and z['rank']<z['after'] and z['hidden']==(z['rank']>=z['before'])
        if arm==3:
            steps=a['rule_steps'];assert len(steps)<=8
            for t in steps:graph_rep(a,t['post'],a['mid_best'])
        else:assert a['rule_steps'] is None
        assert a['replay_graph'] is None
    elif arm==10:
        p=a['replay_graph'];assert p['arm']==8 and len(p['paths'])==mid['visits'] and len(p['steps'])<=8
        assert a['rule_work']==p['end']['rule_work'] and a['best']==p['end']['best']
        assert a['mid_graph'] is None and a['end_graph'] is None
    else:assert a['mid_graph'] is None and a['end_graph'] is None and a['replay_graph'] is None and a['rule_work']==0 and a['best']==a['native_best']
    assert a['NN_calls']==end['nn'] and a['visits']==end['visits']
    assert a['physical_calls']==a['NN_calls']-(a['queue']['coalesced'] if a['queue'] else 0)
    assert (eh['trace'] is not None)==(trace or False) and eh['ledger']['iterations']==end['visits'] and eh['ledger']['repeated_delivery']==0
    for s in (pre,mid,end):assert sum(s['n'][:s['len']])==s['visits'] and set(s['moves'][:s['len']])==set(rank)
    snap=a['snapshot'];assert snap['root_n']==end['visits'] and snap['evaluator_calls']==end['nn'] and snap['best']==end['best']
    es={e['move']:e for e in snap['edges']}
    for j,m in enumerate(end['moves'][:end['len']]):
        z=es.get(m,{});assert end['n'][j]==z.get('n',0) and z.get('vl',0)==z.get('vvalue',0)==0
        close(end['w'][j],z.get('w',0));close(end['m2'][j],z.get('m2',0))
    return a
def req(s,r,arm,trace=False):
    return check(s.request(r['history'],arm=arm,trace=trace),r,arm,trace)
def order(r,phase):
    oo=[(a,j) for a in range(N) for j in range(2)]
    random.Random(SEED+(900000 if phase.endswith('ATLAS') else 800000)+r['id']).shuffle(oo);return oo
def shard(arg):
    phase,ix,rr,mi=arg;fn=D/f'{phase}_{ix}.jsonl.gz';assert not fn.exists();tmp=fn.with_suffix('.tmp')
    n=nn=vis=phys=rules=0;t0=time.monotonic();epoch=0
    with Server() as s,gzip.open(tmp,'wt',compresslevel=3) as fd:
        for i,r in enumerate(rr):
            oo=order(r,phase);z={}
            for k,j in oo:
                a=req(s,r,arm=k)
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
def util(a,u):return u[a['best']]/2
def net(a,u):return util(a,u)-.002*a['NN_calls']-.00002*a['rule_work']
def graph_equivalence(z):
    for j in range(2):
        a=z['3:'+str(j)];b=z['10:'+str(j)]
        for x,y in [(a['mid_graph'],b['replay_graph']['before']),(a['end_graph'],b['replay_graph']['end'])]:
            for k in ('actions','interval','allowed','witness','risk','nodes','registered','best'):assert x[k]==y[k],(k,x[k],y[k],a['history'])
        assert [t['path'] for t in a['rule_steps']]==[t['path'] for t in b['replay_graph']['steps']]
        assert a['end_graph']['paths']==b['replay_graph']['paths'] and a['best']==b['best']
        assert a['end_graph']['checks']==b['replay_graph']['end']['checks']
        assert a['rule_work']<=b['rule_work']
    return True
def calibrate():
    ll=labelmap('TRAIN_CAL');v=defaultdict(list);ids=[]
    for t in rows('TRAIN_CAL_ATLAS'):
        r=t['root'];i=r['id'];z=t['results'];graph_equivalence(z)
        if r['group']!='CAL' or i not in ll:continue
        ids.append(i);u=ll[i]
        for c in range(N):v[c].append(S.mean(net(z[str(c)+':'+str(j)],u) for j in range(2)))
    means={str(c):S.mean(v[c]) for c in range(N)};lower={}
    for c in range(1,N):
        d=[x-y for x,y in zip(v[c],v[0])];lower[str(c)]=S.mean(d)-2*S.stdev(d)/math.sqrt(len(d))
    accepted=[c for c in range(1,N) if lower[str(c)]>0]
    selected=max(accepted,key=lambda c:means[str(c)]) if accepted else 0
    save('FROZEN.json',dict(cohort=sha(D/'COHORT.json'),atlas=sha(D/'TRAIN_CAL_ATLAS_LOCK.json'),labels=sha(D/'TRAIN_CAL_LABELS.json'),code=sha(D/'trial.py'),cal_ids=sorted(ids),means=means,lower=lower,selected=selected))
    print('CAL',len(ids),means,lower,'selected',selected,flush=True)
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
      cpu=a['cpu_ns']/1e6,wall=a['wall_ns']/1e6,tail=a['tail_us']/1000,queue_overhead=(a['queue_setup_us']+a['queue_join_us'])/1000,control=0.,
      n=a['end']['n'][:a['end']['len']],moves=a['end']['moves'][:a['end']['len']],best=a['best'],counts=dict(d),peak=a['queue']['peak_pending'] if a['queue'] else 0,
      sxvb=[h['select_changed'],a['end_hooks']['width_realized'],h['eval_changed'],l['applied']-p['applied']] if a['arm'] in A else [0]*4,
      receipt=l['transferred']-p['transferred'],blocked=l['blocked_receipt']-p['blocked_receipt'])
def aggregate(v):
    aa=[a for z in v for a in z];ct=Counter();arms=Counter();peak=Counter();sx=[0]*4;all4=0
    for a in aa:
        ct.update(a['counts']);arms[str(a['arm'])]+=1;peak[str(a['peak'])]+=1
        for i,x in enumerate(a['sxvb']):sx[i]+=x
        all4+=all(x>0 for x in a['sxvb'])
    out={k:S.mean(a[k] for a in aa) for k in ('u','native','net','NN','physical','visits','tail_NN','rules','cpu','wall','tail','queue_overhead','control')}
    return dict(n=len(v),requests=len(aa),**out,regret=S.mean(a['opt']-a['u'] for a in aa),counts=dict(ct),arms=dict(arms),peaks=dict(peak),sxvb=sx,all4=all4,receipt=sum(a['receipt'] for a in aa),blocked=sum(a['blocked'] for a in aa),
        stopped=arms['0'],batch_mean=ct['logical']/ct['batches'] if ct['batches'] else None,physical_batch_mean=ct['physical']/ct['batches'] if ct['batches'] else None,wait_us=ct['wait_ns']/ct['logical']/1000 if ct['logical'] else None)
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
def graph_stats(a,u):
    m=a['mid_graph'];e=a['end_graph']
    if a['arm']==10:m=a['replay_graph']['before'];e=a['replay_graph']['end']
    for rep in [m,e]+[z['post'] for z in (a['rule_steps'] or [])]:
        for mv,(l,h) in rep['actions']:assert l<=u[mv]-1<=h
        if rep['witness'] is not None:assert u[rep['witness']]==max(u.values())
    forced=e.get('forced',[])
    plan_ns=sum(t['plan_ns'] for t in a['rule_steps'] or [])
    if a['arm'] in range(4,9):plan_ns+=sum((t['plan'] or {}).get('plan_ns',0) for t in a['tail']['rounds'])
    if a['arm']==10:plan_ns+=1000*sum(t['plan_us'] for t in a['replay_graph']['steps'])
    return dict(n=1,before_witness=int(m['witness'] is not None),end_witness=int(e['witness'] is not None),
      new_witness=int(m['witness'] is None and e['witness'] is not None),risk_before=m['risk'][0],risk_end=e['risk'][0],width_before=m['risk'][1],width_end=e['risk'][1],
      changed=int(a['best']!=a['mid_best']),gain=util(a,u)-u[a['mid_best']]/2,rule=a['rule_work'],build_rule=m['rule_work'],checks=e['checks'],replay=e['replay'],
      forced=len(forced),hidden=sum(t['hidden'] for t in forced),deep_hidden=sum(t['hidden'] and t['depth']>0 for t in forced),
      unique_hidden=len({(t['key'],t['move']) for t in forced if t['hidden']}),hits=e.get('frontier_hits',0),shared=e['shared_notices'],extra_shared=e['shared_notices']-m['shared_notices'],
      rule_steps=len(a['rule_steps'] or (a['replay_graph']['steps'] if a['replay_graph'] else [])),
      certified=int(bool(a['tail']) and a['tail']['cause']=='CERTIFIED'),
      planner_ns=plan_ns)
def score():
    f=read('FROZEN.json');ll=labelmap('TEST');fixed={};graph=defaultdict(list);strata=defaultdict(list);all_eq=0
    for t in rows('TEST_ATLAS'):
        r=t['root'];i=r['id'];z=t['results'];graph_equivalence(z);all_eq+=1
        if i not in ll:continue
        u=ll[i];fixed[i]=[[compact(z[str(c)+':'+str(j)],u) for j in range(2)] for c in range(N)]
        m=z['2:0']['mid_graph']
        strata['certified' if m['witness'] is not None else 'unresolved'].append(i)
        strata['risk'+str(m['risk'][0])].append(i)
        for c in [2,3,4,5,6,7,8,10]:
            for j in range(2):graph[str(c)].append(graph_stats(z[str(c)+':'+str(j)],u))
    ids=sorted(ll);assert set(ids)==set(fixed)
    d={str(c):[fixed[i][c] for i in ids] for c in range(N)}
    pairs=[(c,0) for c in range(1,N)]+[(3,10),(5,4),(6,4),(6,5),(7,6),(8,7),(8,9),(3,1),(4,1),(5,1),(6,1),(8,1)]
    out=dict(n=len(ids),selected=f['selected'],fixed={k:aggregate(v) for k,v in d.items()},contrasts={str(x)+':'+str(y):compare(d[str(x)],d[str(y)]) for x,y in pairs},
      repeats={k:repeat(v) for k,v in d.items()},graph={k:{x:sum(z[x] for z in v) for x in v[0]} for k,v in graph.items()},
      strata={k:dict(n=len(v),contrasts={str(x)+':0':pair([fixed[i][x] for i in v],[fixed[i][0] for i in v]) for x in (1,3,4,5,6,7,8,9,10)}) for k,v in strata.items()},replay_equiv_roots=all_eq)
    save('SUMMARY.json',out);print('SUMMARY',len(ids),'selected',f['selected'],'equivalence',all_eq,'graph',out['graph'],flush=True)
    for k,v in out['fixed'].items():print('FIXED',k,v,flush=True)
    print('CONTRASTS',out['contrasts'],flush=True)
def diagnostic():
    rr=[r for r in read('COHORT.json')['roots'] if r['group']=='TRAIN'][:3];fn=D/'DIAGNOSTIC.jsonl.gz'
    assert not fn.exists()
    with Server() as s,gzip.open(fn,'wt',compresslevel=3) as fd:
        for r in rr:emit(fd,dict(root=r,results={str(c):req(s,r,arm=c,trace=True) for c in (0,1,2,3,4,5,6,7,8,9,10)}))
    save('DIAGNOSTIC_LOCK.json',dict(sha256=sha(fn),requests=len(rr)*11,roots=[r['id'] for r in rr]))
def timing():
    rr=[r for r in read('COHORT.json')['roots'] if r['group']=='TEST'][:12];fn=D/'TIMING.jsonl.gz';assert not fn.exists()
    with Server() as s,gzip.open(fn,'wt',compresslevel=3) as fd:
        for r in rr:
            oo=order(r,'TEST_ATLAS');res={str(c)+':'+str(j):req(s,r,arm=c) for c,j in oo}
            emit(fd,dict(root=r,order=oo,results=res))
    save('TIMING_LOCK.json',dict(sha256=sha(fn),requests=len(rr)*22,roots=[r['id'] for r in rr],scope='single server, host not isolated'))
def smoke():
    r=json.loads((R/'quartz_kinetic31_20261126/COHORT.json').read_text())['roots'][0]
    with Server() as s:
        z={}
        for c in range(11):
            a=req(s,r,arm=c,trace=True);z[str(c)+':0']=z[str(c)+':1']=a;print('SMOKE',c,a['NN_calls'],a['physical_calls'],a['visits'],a['queue'] is not None,flush=True)
        graph_equivalence(z)
        for z in (dict(arm=11),dict(arm=0,policy='prior'),dict(policy='prior'),dict(arm=0,oracle=1)):
            assert 'error' in s.request(r['history'],**z)
    print('SMOKE 11 requests, 4 rejected',flush=True)
if __name__=='__main__':
    if sys.argv[1] in ('collect','labels'):globals()[sys.argv[1]](sys.argv[2])
    else:globals()[sys.argv[1]]()
