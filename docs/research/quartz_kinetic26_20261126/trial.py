"""Prospective complete factorial over actual MCTS phases and CPU schedules."""
from itertools import product
from source_gen import D,ROOT,R,BASE,SEED,PRICE,sha,read,save,raw,emit,ident,k20,p,old,BIN
import os,subprocess as SP,select,time,math,random,statistics as S,collections as C,gzip,sys,json
METHODS=('SERIAL','TREE2','TREE4','TREE8','WAVE4','ROOT2','ROOT2D')
NAMES=('S','X','V','B')
class Server:
    def __init__(self):
        self.p=SP.Popen(['nice','-n','10',str(BIN),'--server'],stdin=SP.PIPE,stdout=SP.PIPE,stderr=SP.DEVNULL,bufsize=0,env=dict(os.environ,RAYON_NUM_THREADS='8',QUARTZ_DISABLE_NN_CACHE='1',QUARTZ_DISABLE_ASYNC_PIPELINE='1',QUARTZ_DISABLE_QIPC_SHM='1'))
        self.buf=b'';self.cpu=old.process_clock(self.p.pid)
    def request(self,h,method,**opts):
        q=dict(cmd='research_connect4_k26_cpu',history=h,method=method,seed=17,model_path=str(p.MODEL),model_identity=p.MID,**opts)
        t=time.perf_counter_ns();cpu=time.clock_gettime_ns(self.cpu)
        self.p.stdin.write((json.dumps(q,separators=(',',':'))+'\n').encode());self.p.stdin.flush()
        deadline=time.monotonic()+60
        while b'\n' not in self.buf:
            left=deadline-time.monotonic()
            if left<=0 or not select.select([self.p.stdout],[],[],left)[0]:raise RuntimeError('native timeout')
            x=os.read(self.p.stdout.fileno(),65536)
            if not x:raise RuntimeError('native EOF')
            self.buf+=x
            if len(self.buf)>4*1024*1024:raise RuntimeError('oversized reply')
        line,self.buf=self.buf.split(b'\n',1);a=json.loads(line)
        a['cpu_ns']=time.clock_gettime_ns(self.cpu)-cpu;a['wall_ns']=time.perf_counter_ns()-t
        return a
    def __enter__(self):return self
    def __exit__(self,*args):
        self.p.stdin.close()
        try:self.p.wait(timeout=5)
        except SP.TimeoutExpired:self.p.terminate();self.p.wait(timeout=5)
        if self.p.returncode!=0 and args[0] is None:raise RuntimeError(('server exit',self.p.returncode))
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
MASKS=(0,1,2,4,8,15);PATHS=list(product(MASKS,repeat=2));ZERO=0;FULL=35
def bin0(s):
    n=sorted(s['n']);return 2*int(4*n[-1]>=3*s['visits'])+int(n[-1]-n[-2]>=4)
def check(a,r,method,path=None,table=None,trace=False):
    assert a.get('schema')=='quartz_k26_c4_v1',a
    assert a['history']==r['history'] and a['method']==method and a['legal']==r['legal'] and a['proof']==k20.proof(r['history'])
    pre=a['prefix'];assert pre['visits']==8 and a['prefix_bin']==bin0(pre)
    chosen=a['chosen'];assert chosen==(list(path) if path is not None else [table['first'][a['prefix_bin']],table['second'][chosen[0]][a['mid_bin']]])
    k=2 if method.startswith('ROOT2') else 1;threads={'TREE2':2,'TREE4':4,'TREE8':8,'WAVE4':4}.get(method,1)
    assert len(a['prefix_lanes'])==k and len(a['prefix_hooks'])==k and len(a['stages'])==2
    last=a['prefix_lanes'];last_h=a['prefix_hooks'];last_state=pre
    for l,h in zip(last,last_h):
        assert l['visits']==8//k and l['nn']==1+h['hooks']['base_evals'] and h['hooks']['extra_evals']==0
        assert h['ledger']['iterations']==l['visits'] and h['ledger']['transferred']==0
    for j,s in enumerate(a['stages']):
        m=chosen[j];assert s['profile']==m and len(s['lanes'])==len(s['hooks'])==len(s['budgets'])==k
        z=s['state'];assert z['nn']>=last_state['nn'] and 0<=z['nn']-last_state['nn']<=8
        assert 0<z['visits']-last_state['visits']<=32
        assert s['bin']==2*bin0(z)+int(any(h['ledger']['shared']>0 for h in s['hooks']))
        if j==0:assert a['mid_bin']==s['bin']
        for i,(l,h,(bd,times)) in enumerate(zip(s['lanes'],s['hooks'],s['budgets'])):
            prev=last[i];pl=last_h[i]['ledger'];q=h['ledger'];hh=h['hooks']
            assert bd['target']==8//k and bd['cap']==32//k and bd['used_nn']+bd['unused']==bd['target']
            assert 0<=bd['used_nn']<=bd['target'] and bd['visits']==l['visits']-prev['visits'] and 0<bd['visits']<=bd['cap']
            assert l['nn']-prev['nn']==bd['used_nn']==hh['base_evals']+hh['extra_evals']
            assert bd['visits']==hh['base_evals']+hh['terminal_leaves']
            assert bd['cause']==('NN_TARGET' if bd['unused']==0 else 'VISIT_CAP')
            if bd['unused']:assert bd['visits']==bd['cap']
            assert h['peak']<=threads and h['peak']>=1
            assert q['iterations']==l['visits'] and q['eligible']==q['novel']+q['blocked_receipt']
            assert q['transferred']==q['receipt_items'] and q['repeated_delivery']==0 and q['transferred']>=pl['transferred']
            for bit,key in ((1,'select_nodes'),(2,'width_nodes'),(4,'refinements')):
                if not m&bit:assert hh[key]==0
            if not m&2:assert h['width_realized']==0
            if not m&4:assert hh['extra_evals']==0 and bd['denied_refinement']==0
            if not m&8:assert q['transferred']==pl['transferred']
            assert (h['trace'] is not None)==(trace and j==1)
            assert sum(l['n'])==l['visits'] and l['moves']==prev['moves']
            assert times[1]>=times[0]
        last=s['lanes'];last_h=s['hooks'];last_state=z
    assert a['NN_calls']==a['result']['nn']==last_state['nn'] and a['visits']==a['result']['visits']<=72
    assert a['result']==last_state
    for z,ls in [(pre,a['prefix_lanes'])]+[(s['state'],s['lanes']) for s in a['stages']]:
        n=[sum(l['n'][i] for l in ls) for i in range(len(z['n']))];assert n==z['n'] and sum(n)==z['visits']
        assert z['best']==z['moves'][max(range(len(n)),key=lambda i:(n[i],i))]
        assert z['nn']==sum(l['nn'] for l in ls)
        for i,w in enumerate(z['w']):close(w,sum(l['w'][i] for l in ls))
    cert=k20.cert(a['proof']);assert a['best']==(a['result']['best'] if cert is None else cert)
    for l,snap in zip(last,a['snapshots']):
        assert l['visits']==snap['root_n'] and l['nn']==snap['evaluator_calls']
        es={e['move']:e for e in snap['edges']}
        for i,m in enumerate(l['moves'][:l['len']]):
            e=es.get(m,{});assert l['n'][i]==e.get('n',0) and e.get('vl',0)==e.get('vvalue',0)==0
            close(l['w'][i],e.get('w',0));close(l['m2'][i],e.get('m2',0))
    return a
def req(s,r,m,path=None,table=None,trace=False):
    a=s.request(r['history'],m,**({'path':list(path)} if path is not None else {'table':table}),trace=trace)
    return check(a,r,m,path,table,trace)
def test_opts(m,f,kind):
    z=f['methods'][m]
    if kind=='base':return dict(path=(0,0))
    if kind=='full':return dict(path=(15,15))
    if kind=='static':return dict(path=z['static'])
    if kind=='dynamic':return dict(table=z['table'])
    if kind=='selected':return test_opts(m,f,z['selected'])
    raise ValueError(kind)
def trial(part):
    co=read('COHORT.json');assert co['source']==sha(D/'SOURCE.jsonl.gz')
    f=None
    if part=='TEST':
        f=read('FROZEN.json');assert f['model']==sha(D/'MODEL.json') and f['code']==sha(D/'trial.py') and f['cohort']==sha(D/'COHORT.json') and f['identity']==ident()
    path=D/(part+'_ATLAS.jsonl.gz');assert not path.exists()
    cnt=vis=nn=roots=0;t0=time.monotonic()
    with Server() as srv,gzip.open(path,'wt',compresslevel=3) as fd:
        for r in co['roots']:
            if (r['group']=='TEST')!=(part=='TEST'):continue
            order=[(m,i) for m in METHODS for i in range(36)];random.Random(SEED+900000+r['id']).shuffle(order)
            arms={m:[None]*36 for m in METHODS}
            for m,i in order:
                a=req(srv,r,m,path=PATHS[i]);arms[m][i]=a;cnt+=1;vis+=a['visits'];nn+=a['NN_calls']
            for m in METHODS:assert all(a['prefix']==arms[m][0]['prefix'] for a in arms[m])
            assert all(arms[m][0]['prefix']==arms['SERIAL'][0]['prefix'] for m in METHODS[:5])
            t=dict(root=r,order=order,arms=arms)
            if f:
                extra={m:{} for m in METHODS};seq=[(m,k,j) for m in METHODS for k,j in [('base',1),('full',1),('selected',0),('selected',1),('dynamic',0),('dynamic',1)]]
                random.Random(SEED+800000+r['id']).shuffle(seq)
                for m,k,j in seq:
                    a=req(srv,r,m,**test_opts(m,f,k));extra[m][k+str(j)]=a;cnt+=1;vis+=a['visits'];nn+=a['NN_calls']
                t.update(extra=extra,extra_order=seq)
            emit(fd,t);roots+=1
            if roots%5==0:print('PROGRESS',part,roots,cnt,round(time.monotonic()-t0,1),flush=True)
    save(part+'_LOCK.json',dict(cohort=sha(D/'COHORT.json'),sha256=sha(path),identity=ident(),requests=cnt,visits=vis,NN=nn,roots=roots,freeze=None if not f else sha(D/'FROZEN.json')))
    print('TRIAL',part,roots,cnt,vis,nn,flush=True)
def labels(part):
    lk=read(part+'_LOCK.json');assert lk['sha256']==sha(D/(part+'_ATLAS.jsonl.gz'));co=read('COHORT.json');assert lk['cohort']==sha(D/'COHORT.json')
    if part=='TEST':assert lk['freeze']==sha(D/'FROZEN.json')
    out=[];cnt=C.Counter()
    for r in co['roots']:
        if (r['group']=='TEST')!=(part=='TEST'):continue
        q=old.oracle(r);item=dict(root=r,status=q['status'])
        if q['status']=='EXACT':
            try:
                v,n=p.ref.exact_wdl(r['history']);truth={a:(q['scores'][a]>0)-(q['scores'][a]<0) for a in r['legal']}
                item.update(scores=q['scores'],independent=v,nodes=n,agreement=v==truth)
                if v!=truth:item['status']='MISMATCH'
            except RuntimeError as e:item.update(status='INDEPENDENT_CAP',reason=str(e),scores=q['scores'])
        else:item['oracle_result']=q
        cnt[item['status']]+=1;out.append(item)
    save(part+'_LABELS.json',dict(atlas=lk['sha256'],cohort=sha(D/'COHORT.json'),counts=dict(cnt),rows=out));print('LABELS',part,dict(cnt),flush=True)
def data(part):
    lab=read(part+'_LABELS.json');lk=read(part+'_LOCK.json');assert lab['atlas']==lk['sha256']==sha(D/(part+'_ATLAS.jsonl.gz'))
    ll={x['root']['id']:x for x in lab['rows'] if x['status']=='EXACT'}
    for t in raw(D/(part+'_ATLAS.jsonl.gz')):
        r=t['root'];i=r['id']
        if i not in ll:continue
        u={int(k):(v+1)/2 for k,v in ll[i]['independent'].items()}
        yield t,u
def util(a,u):return u[a['best']]
def net(a,u):return util(a,u)-PRICE*a['NN_calls']
def stagecost(a,j):return sum(z[0]['used_nn'] for z in a['stages'][j]['budgets'])
def full(a):return all(stagecost(a,j)==8 for j in range(2))
def fit():
    assert not (D/'TEST_ATLAS.jsonl.gz').exists() and not (D/'CAL_LIVE.jsonl.gz').exists()
    tr=[]
    for t,u in data('TRAIN_CAL'):
        if t['root']['group']=='TRAIN':tr.append((t,u))
    assert tr
    methods={}
    for m in METHODS:
        v=[(a,util(a,u),net(a,u)) for t,u in tr for a in t['arms'][m]]
        q2={};second=[[0]*8 for _ in range(16)];first=[]
        for i in MASKS:
            for j in MASKS:
                rr=[y-PRICE*stagecost(a,1) for a,y,_ in v if a['chosen']==[i,j]]
                pool=S.mean(rr)
                for b in range(8):
                    rr=[y-PRICE*stagecost(a,1) for a,y,_ in v if a['chosen']==[i,j] and a['mid_bin']==b]
                    q2[i,b,j]=(sum(rr)+8*pool)/(len(rr)+8)
            for b in range(8):second[i][b]=max(MASKS,key=lambda j:q2[i,b,j])
        q1={}
        for i in MASKS:
            rr=[(a['prefix_bin'],max(q2[i,a['mid_bin'],j] for j in MASKS)-PRICE*stagecost(a,0)) for a,_,_ in v if a['chosen'][0]==i]
            pool=S.mean(y for b,y in rr)
            for b in range(4):
                ys=[y for k,y in rr if k==b];q1[b,i]=(sum(ys)+8*pool)/(len(ys)+8)
        first=[max(MASKS,key=lambda i:q1[b,i]) for b in range(4)]
        static=[S.mean(net(t['arms'][m][i],u) for t,u in tr) for i in range(36)]
        methods[m]=dict(table=dict(first=first,second=second),static=list(PATHS[max(range(36),key=static.__getitem__)]),
            static_net=static,q1=[[q1[b,i] for i in MASKS] for b in range(4)],q2=[[[q2[i,b,j] for j in MASKS] for b in range(8)] for i in MASKS])
    save('MODEL.json',dict(cohort=sha(D/'COHORT.json'),trace=sha(D/'TRAIN_CAL_ATLAS.jsonl.gz'),labels=sha(D/'TRAIN_CAL_LABELS.json'),code=sha(D/'trial.py'),identity=ident(),train_ids=[t['root']['id'] for t,u in tr],methods=methods))
    print('MODEL',len(tr),{m:(z['table']['first'],z['static']) for m,z in methods.items()},flush=True)
def calibrate():
    f=read('MODEL.json');assert f['code']==sha(D/'trial.py') and f['identity']==ident()
    assert not (D/'CAL_LIVE.jsonl.gz').exists() and not (D/'TEST_ATLAS.jsonl.gz').exists()
    rows=[(t['root'],u) for t,u in data('TRAIN_CAL') if t['root']['group']=='CAL'];assert rows
    by={m:{k:[] for k in ('base','static','dynamic')} for m in METHODS};n=nn=vis=0
    with Server() as s,gzip.open(D/'CAL_LIVE.jsonl.gz','wt',compresslevel=3) as fd:
        for r,u in rows:
            order=[(m,k) for m in METHODS for k in ('base','static','dynamic')];random.Random(SEED+100000+r['id']).shuffle(order)
            res={m:{} for m in METHODS}
            for m,k in order:
                a=req(s,r,m,**test_opts(m,f,k));res[m][k]=a;by[m][k].append(net(a,u));n+=1;nn+=a['NN_calls'];vis+=a['visits']
            emit(fd,dict(root=r,order=order,results=res))
    methods={}
    for m,z in f['methods'].items():
        val={k:S.mean(v) for k,v in by[m].items()};methods[m]=dict(z,calnet=val,selected=max(val,key=val.__getitem__))
    save('FROZEN.json',dict(model=sha(D/'MODEL.json'),cal_sha=sha(D/'CAL_LIVE.jsonl.gz'),cohort=f['cohort'],identity=f['identity'],code=f['code'],cal_ids=[r['id'] for r,u in rows],cal_requests=n,cal_NN=nn,cal_visits=vis,methods=methods))
    print('CAL',len(rows),n,nn,vis,{m:(z['selected'],z['calnet']) for m,z in methods.items()},flush=True)
def js(a,b):
    x={m:n/a['visits'] for m,n in zip(a['moves'],a['n'])};y={m:n/b['visits'] for m,n in zip(b['moves'],b['n'])}
    return sum((p*math.log(2*p/(p+q)) if p else 0.)/2+(q*math.log(2*q/(p+q)) if q else 0.)/2 for m in x.keys()|y.keys() for p,q in [(x.get(m,0),y.get(m,0))])
def aggregate(rows):
    u=[util(a,t) for a,t in rows];ct=C.Counter();peaks=C.Counter();replica_same=0;overlap=[]
    for a,t in rows:
        if len(a['stages'][1]['lanes'])==2:
            ls=a['stages'][1]['lanes'];replica_same+=ls[0]==ls[1]
        last=a['prefix_hooks']
        for s in a['stages']:
            for h,(b,times),prev in zip(s['hooks'],s['budgets'],last):
                ct.update({k:h['hooks'][k] for k in ('select_nodes','select_changed','width_nodes','width_mass_added','refinements','eval_changed','extra_evals','base_evals','terminal_leaves')})
                ct['width_realized']+=h['width_realized'];ct['denied_refinement']+=b['denied_refinement'];ct['unused']+=b['unused']
                for k in ('transferred','blocked_receipt','applied','repeated_delivery'):ct[k]+=h['ledger'][k]-prev['ledger'][k]
                peaks[str(h['peak'])]+=1
            if len(s['budgets'])==2:
                a0,b0=s['budgets'][0][1];a1,b1=s['budgets'][1][1];overlap.append(max(0,min(b0,b1)-max(a0,a1)))
            last=s['hooks']
    return dict(n=len(rows),utility=S.mean(u),regret=S.mean(max(t.values())-v for (a,t),v in zip(rows,u)),net=S.mean(net(a,t) for a,t in rows),
        NN=S.mean(a['NN_calls'] for a,t in rows),tail_NN=S.mean(sum(stagecost(a,j) for j in range(2)) for a,t in rows),visits=S.mean(a['visits'] for a,t in rows),exact=sum(full(a) for a,t in rows),
        cpu_ms=S.mean(a['cpu_ns']/1e6 for a,t in rows),wall_ms=S.mean(a['wall_ns']/1e6 for a,t in rows),tail_ms=S.mean(sum(s['wall_us'] for s in a['stages'])/1000 for a,t in rows),
        controller_us=S.mean(sum(s['control_us'] for s in a['stages']) for a,t in rows),paths=dict(C.Counter(str(a['chosen']) for a,t in rows)),
        switched=sum(a['chosen'][0]!=a['chosen'][1] for a,t in rows),peak=dict(peaks),counts=dict(ct),identical_replicas=replica_same,
        mean_lane_overlap_us=S.mean(overlap) if overlap else None)
def pair(aa,bb):
    if not aa:return dict(n=0)
    z=[util(a,u)-util(b,u) for (a,u),(b,_) in zip(aa,bb)];rng=random.Random(SEED+21);bs=sorted(S.mean(rng.choices(z,k=len(z))) for _ in range(3000))
    return dict(n=len(z),gain=S.mean(z),net_gain=S.mean(net(a,u)-net(b,u) for (a,u),(b,_) in zip(aa,bb)),better=sum(x>0 for x in z),worse=sum(x<0 for x in z),ci=[bs[74],bs[2924]])
def variance(aa,bb):
    return dict(utility_sd=math.sqrt(S.mean((util(a,u)-util(b,u))**2/2 for (a,u),(b,_) in zip(aa,bb))),
        NN_sd=math.sqrt(S.mean((a['NN_calls']-b['NN_calls'])**2/2 for (a,u),(b,_) in zip(aa,bb))),
        best_disagree=sum(a['best']!=b['best'] for (a,u),(b,_) in zip(aa,bb)),
        mean_js=S.mean(js(a['result'],b['result']) for (a,u),(b,_) in zip(aa,bb)),
        wall_sd_ms=math.sqrt(S.mean(((a['wall_ns']-b['wall_ns'])/1e6)**2/2 for (a,u),(b,_) in zip(aa,bb))))
def score():
    f=read('FROZEN.json');assert read('TEST_LOCK.json')['freeze']==sha(D/'FROZEN.json')
    rows=list(data('TEST'));assert rows;out={}
    for m in METHODS:
        base=[(t['arms'][m][0],u) for t,u in rows];fulls=[(t['arms'][m][35],u) for t,u in rows]
        live=[(t['extra'][m]['selected0'],u) for t,u in rows];dyn=[(t['extra'][m]['dynamic0'],u) for t,u in rows]
        variants={'base':base,'full':fulls,'selected':live,'dynamic':dyn}
        pairs={};matched={};reps={};cross={};actual={}
        for k,a in variants.items():
            pairs[k]=pair(a,base);mm=[(x,y) for x,y in zip(a,base) if full(x[0]) and full(y[0])];matched[k]=pair([x for x,y in mm],[y for x,y in mm])
            b=[(t['extra'][m][k+'1'],u) for t,u in rows];reps[k]=variance(a,b)
            sr=[(t['arms']['SERIAL'][0 if k=='base' else 35],u) if k in ('base','full') else (t['extra']['SERIAL'][k+'0'],u) for t,u in rows]
            cross[k]=dict(disagree=sum(x[0]['best']!=y[0]['best'] for x,y in zip(a,sr)),JSD=S.mean(js(x[0]['result'],y[0]['result']) for x,y in zip(a,sr)),gain=pair(a,sr))
            actual[k]=aggregate(a)
        paths=[aggregate([(t['arms'][m][i],u) for t,u in rows]) for i in range(36)]
        out[m]=dict(selected=f['methods'][m]['selected'],results=actual,pairs=pairs,matched=matched,repeats=reps,cross=cross,paths=paths,opportunity=sum(util(a,u)<max(u.values()) for a,u in base),prefix_opportunity=sum(u[a['prefix']['best']]<max(u.values()) for a,u in base))
        print(m,'selected',out[m]['selected'],'opportunity',out[m]['opportunity'],'prefix',out[m]['prefix_opportunity'],flush=True)
        for k,x in actual.items():print(k,{j:x[j] for j in ('utility','regret','NN','tail_NN','visits','exact','cpu_ms','wall_ms','tail_ms','switched','identical_replicas','peak')},'pair',pairs[k],'matched',matched[k],'repeat',reps[k],'counts',x['counts'],flush=True)
    save('SUMMARY.json',dict(n=len(rows),methods=out))
def diagnostic():
    rows=[r for r in read('COHORT.json')['roots'] if r['group']=='TEST'][:1];assert rows
    with Server() as srv,gzip.open(D/'DIAGNOSTIC.jsonl.gz','wt',compresslevel=3) as fd:
        for r in rows:
            for m in METHODS:
                for path in [(15,15),(8,8)]:emit(fd,dict(root=r,response=req(srv,r,m,path=path,trace=True)))
        for opt in [dict(path=[16,0]),dict(path=[3,0]),dict(path=[0,0],table={'first':[0]*4,'second':[[0]*8]*16}),dict(path=[0,0],oracle=1)]:
            assert 'error' in srv.request(rows[0]['history'],'SERIAL',**opt)
    save('DIAGNOSTIC_LOCK.json',dict(sha256=sha(D/'DIAGNOSTIC.jsonl.gz'),roots=[r['id'] for r in rows],requests=14,malformed_rejected=4))
    print('DIAGNOSTIC 14',flush=True)
if __name__=='__main__':
    if sys.argv[1] in ('trial','labels'):globals()[sys.argv[1]](sys.argv[2])
    else:globals()[sys.argv[1]]()
