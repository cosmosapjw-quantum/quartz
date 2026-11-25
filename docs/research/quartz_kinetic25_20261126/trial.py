"""Prospective complete factorial over actual MCTS phases and CPU schedules."""
from source_gen import D,ROOT,R,BASE,SEED,PRICE,sha,read,save,raw,emit,ident,k20,p,old,BIN
import os,subprocess as SP,select,time,math,random,statistics as S,collections as C,gzip,sys,json
METHODS=('SERIAL','TREE2','WAVE2','ROOT2')
NAMES=('S','X','V','B')
class Server:
    def __init__(self):
        self.p=SP.Popen(['nice','-n','10',str(BIN),'--server'],stdin=SP.PIPE,stdout=SP.PIPE,stderr=SP.DEVNULL,bufsize=0,env=dict(os.environ,RAYON_NUM_THREADS='2',QUARTZ_DISABLE_NN_CACHE='1',QUARTZ_DISABLE_ASYNC_PIPELINE='1',QUARTZ_DISABLE_QIPC_SHM='1'))
        self.buf=b'';self.cpu=old.process_clock(self.p.pid)
    def request(self,h,method,**opts):
        q=dict(cmd='research_connect4_k25_cpu',history=h,method=method,seed=17,model_path=str(p.MODEL),model_identity=p.MID,**opts)
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
def check(a,r,method,mask=None,table=None,trace=False):
    assert a.get('schema')=='quartz_k25_c4_v1',a
    assert a['history']==r['history'] and a['method']==method and a['legal']==r['legal'] and a['proof']==k20.proof(r['history'])
    assert a['visits']==128 and a['prefix']['visits']==32 and a['result']['visits']==128
    assert a['NN_calls']==a['result']['nn'];assert a['best'] in r['legal']
    n=sorted(a['prefix']['n']);b=2*int(4*n[-1]>=3*32)+int(n[-1]-n[-2]>=4)
    assert b==a['bin'];assert a['profile']==(mask if table is None else table[b]);mask=a['profile']
    cert=k20.cert(a['proof']);assert a['best']==(a['result']['best'] if cert is None else cert)
    lanes=a['lanes'];k=2 if method=='ROOT2' else 1;assert len(lanes)==k
    for l in lanes:
        pre,end,snap=l['prefix'],l['end'],l['snapshot'];ph=l['prefix_hooks'];th=l['tail_hooks'];hp=ph['hooks'];ht=th['hooks']
        assert pre['visits']==32//k and end['visits']==128//k and snap['root_n']==end['visits']
        assert ph['ledger']['iterations']==pre['visits'] and th['ledger']['iterations']==end['visits']
        assert ph['ledger']['transferred']==0 and th['ledger']['transferred']==th['ledger']['receipt_items'] and th['ledger']['repeated_delivery']==0
        assert th['ledger']['eligible']==th['ledger']['novel']+th['ledger']['blocked_receipt']
        assert pre['nn']==1+hp['base_evals']+hp['extra_evals']
        assert end['nn']-pre['nn']==ht['base_evals']+ht['extra_evals']
        assert hp['base_evals']+hp['terminal_leaves']==pre['visits']
        assert ht['base_evals']+ht['terminal_leaves']==end['visits']-pre['visits']
        assert th['peak']<= (2 if method in ('TREE2','WAVE2') else 1)
        for flag,field in [(1,'select_nodes'),(2,'width_nodes'),(4,'refinements')]:
            if not mask&flag:assert ht[field]==0
        if not mask&2:assert th['width_realized']==0
        if not mask&4:assert ht['extra_evals']==0
        if not mask&8:assert th['ledger']['transferred']==th['ledger']['applied']==0
        assert (th['trace'] is not None)==trace
        if trace:assert len(th['trace'])==end['visits']
        assert pre['len']==end['len']==len(r['legal'])
        e={q['move']:q for q in snap['edges']}
        for i,m in enumerate(end['moves'][:end['len']]):
            q=e.get(m,{})
            assert end['n'][i]==q.get('n',0)
            close(end['w'][i],q.get('w',0.));close(end['m2'][i],q.get('m2',0.))
            assert q.get('vl',0)==q.get('vvalue',0)==0
        assert sum(end['n'])==end['visits']
    for tag,t in [('prefix','prefix'),('result','end')]:
        x=a[tag];m=x['moves'];nn=[sum(l[t]['n'][i] for l in lanes) for i in range(len(m))]
        assert nn==x['n'] and x['best']==m[max(range(len(m)),key=lambda i:(nn[i],i))]
        assert sum(l[t]['nn'] for l in lanes)==x['nn']
        assert all(l[t]['moves'][:len(m)]==m for l in lanes)
        for i in range(len(m)):close(sum(l[t]['w'][i] for l in lanes),x['w'][i])
    return a
def trial(part):
    co=read('COHORT.json');assert co['source']==sha(D/'SOURCE.jsonl.gz')
    if part=='TEST':
        f=read('FROZEN.json');assert f['code']==sha(D/'trial.py') and f['cohort']==sha(D/'COHORT.json') and f['identity']==ident()
    path=D/(part+'_ATLAS.jsonl.gz');assert not path.exists()
    cnt=nn=vis=roots=0;t0=time.monotonic()
    with Server() as srv,gzip.open(path,'wt',compresslevel=3) as fd:
        for r in co['roots']:
            if (r['group']=='TEST')!=(part=='TEST'):continue
            order=[(m,c) for m in METHODS for c in range(16)];random.Random(SEED+900000+r['id']).shuffle(order)
            arms={m:[None]*16 for m in METHODS}
            for m,c in order:
                a=check(srv.request(r['history'],m,profile=c),r,m,mask=c);arms[m][c]=a
                cnt+=1;nn+=a['NN_calls'];vis+=a['visits']
            for m in METHODS:assert all(a['prefix']==arms[m][0]['prefix'] and a['bin']==arms[m][0]['bin'] for a in arms[m])
            assert all(arms[m][0]['prefix']==arms['SERIAL'][0]['prefix'] for m in METHODS if m!='ROOT2')
            out=dict(root=r,order=order,arms=arms)
            if part=='TEST':
                live={};repeat={}
                tests=[(m,x) for m in METHODS for x in ('live','base2','full2','live2')]
                random.Random(SEED+800000+r['id']).shuffle(tests)
                for m,x in tests:
                    if x.startswith('live'):
                        tab=f['methods'][m]['table'];a=check(srv.request(r['history'],m,table=tab),r,m,table=tab)
                    else:
                        c=0 if x=='base2' else 15;a=check(srv.request(r['history'],m,profile=c),r,m,mask=c)
                    if x=='live':live[m]=a
                    else:repeat.setdefault(m,{})[x]=a
                    if m in ('SERIAL','ROOT2'):
                        b=arms[m][a['profile']];assert a['result']==b['result'] and a['best']==b['best']
                    cnt+=1;nn+=a['NN_calls'];vis+=a['visits']
                out.update(live=live,repeat=repeat,repeat_order=tests)
            emit(fd,out);roots+=1
            if roots%5==0:print('PROGRESS',part,roots,cnt,round(time.monotonic()-t0,1),flush=True)
    save(part+'_LOCK.json',dict(cohort=sha(D/'COHORT.json'),sha256=sha(path),identity=ident(),requests=cnt,visits=vis,NN=nn,roots=roots,freeze=None if part!='TEST' else sha(D/'FROZEN.json')))
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
def reward(a,u):return u[a['best']]-PRICE*a['NN_calls']
def fit():
    assert not (D/'TEST_ATLAS.jsonl.gz').exists() and not (D/'TEST_LABELS.json').exists()
    tr=[];cal=[]
    for t,u in data('TRAIN_CAL'):
        row=dict(id=t['root']['id'],group=t['root']['group'],b={m:t['arms'][m][0]['bin'] for m in METHODS},y={m:[reward(a,u) for a in t['arms'][m]] for m in METHODS})
        (tr if row['group']=='TRAIN' else cal).append(row)
    assert tr and cal
    models={}
    for m in METHODS:
        pooled=[S.mean(r['y'][m][c]-r['y'][m][0] for r in tr) for c in range(16)]
        gm=max(range(16),key=pooled.__getitem__)
        stats=[];tab=[]
        for b in range(4):
            rr=[r for r in tr if r['b'][m]==b];g=[(sum(r['y'][m][c]-r['y'][m][0] for r in rr)+8*pooled[c])/(len(rr)+8) for c in range(16)]
            stats.append(dict(n=len(rr),gains=g));tab.append(max(range(16),key=g.__getitem__))
        tabs={'base':[0]*4,'global':[gm]*4,'conditional':tab}
        calnet={k:S.mean(r['y'][m][t[r['b'][m]]] for r in cal) for k,t in tabs.items()}
        choice=max(tabs,key=calnet.__getitem__)
        models[m]=dict(pooled=pooled,bins=stats,tables=tabs,calnet=calnet,selected=choice,table=tabs[choice])
    save('FROZEN.json',dict(cohort=sha(D/'COHORT.json'),trace=sha(D/'TRAIN_CAL_ATLAS.jsonl.gz'),labels=sha(D/'TRAIN_CAL_LABELS.json'),code=sha(D/'trial.py'),identity=ident(),train_ids=[r['id'] for r in tr],cal_ids=[r['id'] for r in cal],methods=models))
    print('FIT',len(tr),len(cal),{m:{k:x[k] for k in ('selected','table','calnet')} for m,x in models.items()},flush=True)
def js(a,b):
    x={m:n/a['visits'] for m,n in zip(a['moves'],a['n'])};y={m:n/b['visits'] for m,n in zip(b['moves'],b['n'])}
    z=0.
    for m in x.keys()|y.keys():
        p=x.get(m,0);q=y.get(m,0);r=(p+q)/2
        if p:z+=p*math.log(p/r)/2
        if q:z+=q*math.log(q/r)/2
    return z
def agg(aa,uu):
    n=len(aa);u=[v[a['best']] for a,v in zip(aa,uu)];full=[max(v.values()) for v in uu]
    ct=C.Counter();workers=C.Counter();peaks=[];overlap=[];dup=[];work=[0]*4
    for a in aa:
        for l in a['lanes']:
            th=l['tail_hooks'];h=th['hooks'];q=th['ledger'];ct.update({k:h[k] for k in ('select_nodes','select_changed','width_nodes','width_mass_added','base_evals','extra_evals','refinements','eval_changed','terminal_leaves')})
            ct.update({k:q[k] for k in ('eligible','novel','blocked_receipt','transferred','applied','changed','repeated_delivery')})
            ct['width_realized']+=th['width_realized'];peaks.append(th['peak']);workers[str(h['worker_mask'])]+=1
            for j,z in enumerate(th['work_ns']):work[j]+=z
        if len(a['lanes'])==2:
            l,r=a['lanes'];s,t=l['tail_interval_us'];v,w=r['tail_interval_us'];overlap.append(max(0,min(t,w)-max(s,v)))
            dup.append(l['end']==r['end'])
    return dict(n=n,utility=S.mean(u),regret=S.mean(v-x for v,x in zip(full,u)),net=S.mean(reward(a,v) for a,v in zip(aa,uu)),
        NN=S.mean(a['NN_calls'] for a in aa),CPU_ms=S.mean(a['cpu_ns']/1e6 for a in aa),wall_ms=S.mean(a['wall_ns']/1e6 for a in aa),tail_ms=S.mean(a['tail_us']/1e3 for a in aa),
        peak2=sum(x==2 for x in peaks),peak_count=len(peaks),worker_masks=dict(workers),root_overlap_us=S.mean(overlap) if overlap else None,identical_replica_roots=sum(dup),counts=dict(ct),work_ns_per_root=[v/n for v in work])
def paired(aa,bb,uu):
    z=[v[a['best']]-v[b['best']] for a,b,v in zip(aa,bb,uu)];rng=random.Random(SEED+21)
    bs=sorted(S.mean(rng.choices(z,k=len(z))) for _ in range(3000))
    return dict(gain=S.mean(z),net=S.mean(reward(a,v)-reward(b,v) for a,b,v in zip(aa,bb,uu)),better=sum(x>0 for x in z),worse=sum(x<0 for x in z),ci=[bs[74],bs[2924]])
def variance(aa,bb,uu):
    d=[v[a['best']]-v[b['best']] for a,b,v in zip(aa,bb,uu)]
    c=[reward(a,v)-reward(b,v) for a,b,v in zip(aa,bb,uu)]
    return dict(utility_sd=math.sqrt(S.mean(x*x/2 for x in d)),net_sd=math.sqrt(S.mean(x*x/2 for x in c)),
      best_disagree=sum(a['best']!=b['best'] for a,b in zip(aa,bb)),mean_js=S.mean(js(a['result'],b['result']) for a,b in zip(aa,bb)),
      NN_sd=math.sqrt(S.mean((a['NN_calls']-b['NN_calls'])**2/2 for a,b in zip(aa,bb))),wall_sd_ms=math.sqrt(S.mean(((a['wall_ns']-b['wall_ns'])/1e6)**2/2 for a,b in zip(aa,bb))))
def factorial(vals):
    beta=[sum(v*((-1)**(k.bit_count()-(k&m).bit_count())) for m,v in enumerate(vals))/16 for k in range(16)]
    for m,v in enumerate(vals):close(v,sum(x*((-1)**(k.bit_count()-(k&m).bit_count())) for k,x in enumerate(beta)))
    return dict(beta=beta,main={name:2*beta[1<<i] for i,name in enumerate(NAMES)},
      pair={NAMES[i]+NAMES[j]:4*beta[(1<<i)|(1<<j)] for i in range(4) for j in range(i+1,4)},
      leaveout={name:vals[15]-vals[15^(1<<i)] for i,name in enumerate(NAMES)},full_minus_base=vals[15]-vals[0])
def score():
    f=read('FROZEN.json');assert read('TEST_LOCK.json')['freeze']==sha(D/'FROZEN.json')
    tt=list(data('TEST'));assert tt;rows=[t for t,u in tt];uu=[u for t,u in tt];out={}
    for m in METHODS:
        arms=[[t['arms'][m][c] for t in rows] for c in range(16)]
        live=[t['live'][m] for t in rows]
        ag=[agg(a,uu) for a in arms];liv=agg(live,uu)
        out[m]=dict(arms=ag,live=liv,live_profiles=dict(C.Counter(a['profile'] for a in live)),full_vs_base=paired(arms[15],arms[0],uu),live_vs_base=paired(live,arms[0],uu),
            factorial_U=factorial([a['utility'] for a in ag]),factorial_net=factorial([a['net'] for a in ag]),
            repeats={k:variance(a,[t['repeat'][m][q] for t in rows],uu) for k,a,q in [('base',arms[0],'base2'),('full',arms[15],'full2'),('live',live,'live2')]},
            matched_full_base=sum(a['NN_calls']==b['NN_calls'] for a,b in zip(arms[15],arms[0])))
    for m in METHODS:
        a=out[m];a['schedule_interaction_full_U']=a['full_vs_base']['gain']-out['SERIAL']['full_vs_base']['gain'];a['schedule_interaction_full_net']=a['full_vs_base']['net']-out['SERIAL']['full_vs_base']['net']
        a['schedule_base_U_vs_serial']=a['arms'][0]['utility']-out['SERIAL']['arms'][0]['utility'];a['schedule_full_U_vs_serial']=a['arms'][15]['utility']-out['SERIAL']['arms'][15]['utility']
    save('SUMMARY.json',dict(n=len(rows),methods=out))
    for m,a in out.items():
        print(m, 'base/full/live', [{k:v[k] for k in ('utility','regret','net','NN','CPU_ms','wall_ms','tail_ms','peak2','peak_count','root_overlap_us','identical_replica_roots')} for v in [a['arms'][0],a['arms'][15],a['live']]],'full',a['full_vs_base'],'live',a['live_vs_base'],'repeats',a['repeats'],'U',a['factorial_U'],'net',a['factorial_net'],'full_counts',a['arms'][15]['counts'],'profiles',a['live_profiles'],flush=True)
def diagnostic():
    co=read('COHORT.json');rows=[r for r in co['roots'] if r['group']=='TEST'][:2];path=D/'DIAGNOSTIC.jsonl.gz';assert not path.exists()
    with Server() as srv,gzip.open(path,'wt',compresslevel=3) as fd:
        for r in rows:
            for m in METHODS:
                for c in (0,15):
                    a=check(srv.request(r['history'],m,profile=c,trace=True),r,m,mask=c,trace=True);emit(fd,dict(root=r,response=a))
        h=rows[0]['history']
        for opts in [dict(profile=16),dict(profile=0,table=[0]*4),dict(table=[0,0,0,16]),dict(profile=0,oracle=1)]:
            assert 'error' in srv.request(h,'SERIAL',**opts)
    save('DIAGNOSTIC_LOCK.json',dict(sha256=sha(path),roots=[r['id'] for r in rows],requests=len(rows)*8,malformed_rejected=4))
    print('DIAGNOSTIC',len(rows)*8,flush=True)
if __name__=='__main__':
    if sys.argv[1] in ('trial','labels'):globals()[sys.argv[1]](sys.argv[2])
    else:globals()[sys.argv[1]]()
