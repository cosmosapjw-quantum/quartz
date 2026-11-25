"""Split-fit signed gain tables and prospective actual native K20 search."""
import pathlib,sys,json,gzip,hashlib,subprocess as SP,random,itertools,collections as C,statistics as S,math,time,os,select
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2]
P19=ROOT.parent/'quartz_kinetic19_20261126/docs/research/quartz_kinetic19_20261126'
sys.path.insert(0,str(P19));import run_k19 as k19
p=k19.p;old=p.old;P18=k19.P18;P17=k19.r18.P17
sys.path.insert(0,str(D))
BIN=ROOT.parent/'quartz_kinetic20_target_20261126/mcts_demo'
BASE='b0979f88611b5bf71ba1c06851b1bc1e1be25723';SEED=2026112620
CH=tuple(f'{a}:{n}' for n in (32,64) for a in ('K16N','K16BAL','K17EXP'))
PR=(.002,0.,.001,.004)
def sha(p0):return hashlib.file_digest(p0.open('rb'),'sha256').hexdigest()
def save(n,x):
    f=D/n;assert not f.exists(),n;f.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n')
def read(n):return json.loads((D/n).read_text())
def raw(path):
    with gzip.open(path,'rt') as f:
        for l in f:yield json.loads(l)
def emit(f,x):f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n');f.flush()
def lock(path):
    rel=str(path.relative_to(ROOT))
    b=SP.check_output(['git','-C',str(ROOT),'show',BASE+':'+rel])
    assert hashlib.sha256(b).hexdigest()==sha(path)
    return dict(commit=BASE,path=rel,sha256=sha(path))
def identity():
    a=read('RUST_BUILD_IDENTITY.json');assert sha(BIN)==a['binary_sha256']
    return dict(native=sha(BIN),source=a['source_head'],previous=k19.r18.identity())
def proof(h):
    b,z=old.c4replay(h);side=len(h)%2;rows=[]
    for c in range(7):
        if z[c]==6:continue
        bit=1<<(7*c+z[c]);bb=b.copy();zz=z.copy();bb[side]|=bit;zz[c]+=1
        if old.won(bb[side]):l=u=1.
        elif sum(zz)==42:l=u=.5
        elif old.canwin(bb,zz,1-side):l=u=0.
        else:l,u=0.,1.
        rows.append([c,l,u])
    return rows
def cert(proofs):
    u=max(r[2] for r in proofs)
    return next((r[0] for r in proofs if r[1]>=u),None)
def scores(snap,panel):
    ca,n,w=p.pr.moments(snap)
    return [w[ca.index(m)]/(n[ca.index(m)]+4) for m in panel]
def prefix_bin(snap,panel):
    ca,n,w=p.pr.moments(snap);s=scores(snap,panel)
    return 2*int(n[ca.index(panel[0])]>=6)+int(s[0]-max(s[1:])>=.15)
def response(snap,panel):
    s=scores(snap,panel);j=max(range(3),key=s.__getitem__)
    return j,int(s[j]-s[0]>=.15)
def sources():
    return [lock(ROOT/'docs/research/quartz_kinetic18_20261126'/n) for n in ('COHORT.json','ATLAS.jsonl.gz','ATLAS_LOCK.json','TRAIN_CAL_LABELS.json')]
def rows18():
    src=ROOT/'docs/research/quartz_kinetic18_20261126';ll={z['root']['id']:z for z in json.loads((src/'TRAIN_CAL_LABELS.json').read_text())['rows'] if z['status']=='EXACT'};out={}
    for t in raw(src/'ATLAS.jsonl.gz'):
        r=t['root'];i=r['id']
        if r['group']=='TEST':continue
        assert i in ll and r==ll[i]['root'] and ll[i]['agreement']
        u={int(a):(v+1)/2 for a,v in ll[i]['independent'].items()}
        a=t['response'];pre=a['evidence']['prefix'];panel=t['panels']['common'];b=prefix_bin(pre,panel);pr=proof(r['history'])
        assert panel[0]==pre['best'] and all(l<=u[m]<=v for m,l,v in pr)
        c=CH.index(f"{t['arm']}:{t['cap']}");j,h=response(a['result'],panel)
        if i not in out:out[i]=dict(id=i,group=r['group'],family=r['family'],bin=b,panel=panel,utility=u,proof=pr,obs={})
        z=out[i];assert z['bin']==b and z['panel']==panel and c not in z['obs']
        z['obs'][c]=dict(j=j,high=h,native=a['best'],cost=a['NN_calls']-pre['evaluator_calls'])
    assert set(out)==set(ll) and all(len(r['obs'])==6 for r in out.values())
    return sorted(out.values(),key=lambda x:x['id'])
def readout(A):
    table=[[[False]*2 for _ in range(3)] for _ in CH];stats={}
    for c in range(6):
        if c%3==0:continue
        for j in (1,2):
            rr=[r for r in A if r['obs'][c]['j']==j]
            pooled=S.mean(r['utility'][r['panel'][j]]-r['utility'][r['panel'][0]] for r in rr) if rr else 0.
            for h in (0,1):
                g=[r['utility'][r['panel'][j]]-r['utility'][r['panel'][0]] for r in rr if r['obs'][c]['high']==h]
                mu=(sum(g)+4*pooled)/(len(g)+4)
                table[c][j][h]=bool(g and mu>0)
                stats[f'{c}:{j}:{h}']=dict(n=len(g),positive=sum(x>0 for x in g),negative=sum(x<0 for x in g),mu=mu,pooled=pooled,switch=table[c][j][h])
    return table,stats
def decision(r,c,table):
    a=r['panel'][0];o=r['obs'][c];m=cert(r['proof'])
    if m is not None:return m
    if c%3==0:return o['native']
    j=o['j'];m=r['panel'][j];pr={m:(l,u) for m,l,u in r['proof']}
    return m if j!=0 and table[c][j][o['high']] and pr[m][1]>pr[a][0] else a
def targets(r,c,table):return r['utility'][decision(r,c,table)]-r['utility'][r['panel'][0]],r['obs'][c]['cost']
def fit_tables(B,table,stratified):
    gains=[];costs=[];hist=[]
    for b in range(4):
        gg=[];cc=[];hh=[]
        for c in range(6):
            z=[targets(r,c,table) for r in B]
            cnt=[sum(x==v/2 for x,t in z) for v in range(-2,3)]
            prob=[v/len(B) for v in cnt];cost=S.mean(t for x,t in z)
            bb=[targets(r,c,table) for r in B if r['bin']==b]
            if stratified:
                p0=[(sum(x==v/2 for x,t in bb)+8*prob[v+2])/(len(bb)+8) for v in range(-2,3)]
                cost=(sum(t for x,t in bb)+8*cost)/(len(bb)+8)
            else:p0=prob
            mu=sum(v/2*p0[v+2] for v in range(-2,3));gg.append(mu);cc.append(cost)
            hh.append(dict(prob=p0,n=len(bb) if stratified else len(B),positive=p0[3]+p0[4],negative=p0[0]+p0[1],positive_mass=p0[3]/2+p0[4],negative_mass=p0[1]/2+p0[0],mu=mu))
        gains.append(gg);costs.append(cc);hist.append(hh)
    return dict(gains=gains,costs=costs,switch=table),hist
def choose(m,b,price):
    best=None;v=0.
    for c in range(6):
        x=m['gains'][b][c]-price*m['costs'][b][c]
        if x>v+1e-15:best=c;v=x
    return best,v
def fit():
    assert not (D/'SOURCE.jsonl.gz').exists()
    sl=sources();rr=rows18();tr=[r for r in rr if r['group']=='TRAIN'];cal=[r for r in rr if r['group']=='CAL'];assert len(tr)==110 and len(cal)==74
    A=tr[::2];B=tr[1::2];table,stats=readout(A);fits={};metrics={}
    for mode in ('global','stratified'):
        m,h=fit_tables(B,table,mode=='stratified')
        mse=S.mean((targets(r,c,table)[0]-m['gains'][r['bin']][c])**2 for r in cal for c in range(6))
        fits[mode]=dict(controller=m,hist=h);metrics[mode]=mse
    selected=min(fits,key=metrics.__getitem__);m=fits[selected]['controller']
    support=[]
    for c in range(6):
        g=[targets(r,c,table)[0] for r in B];support.append(dict(channel=CH[c],positive=sum(x>0 for x in g),negative=sum(x<0 for x in g),zero=sum(x==0 for x in g)))
    save('FROZEN.json',dict(identity=identity(),sources=sl,code=sha(D/'acquire.py'),split_A=[r['id'] for r in A],split_B=[r['id'] for r in B],cal=[r['id'] for r in cal],readout_stats=stats,fit=fits,selected=selected,cal_mse=metrics,controller=m,support=support,price=.002))
    print('FIT',selected,metrics,support,'switch',table,'gains',m['gains'],'costs',m['costs'],flush=True)
def historical():
    _,boards,locks=old.c4historical();families=set();histlocks=[]
    dirs=ROOT/'docs/research'
    paths=[dirs/'quartz_kinetic16_20261126/C06_COHORT.json',dirs/'quartz_kinetic17_20261126/COHORT.json',dirs/'quartz_kinetic17_20261126/quiet/COHORT.json',dirs/'quartz_kinetic18_20261126/COHORT.json',dirs/'quartz_kinetic19_20261126/COHORT.json']
    for path in paths:
        histlocks.append(lock(path));co=json.loads(path.read_text())
        for r in co['roots']:
            if r.get('domain','C4')!='C4':continue
            h=r['history'];boards.add(old.boardkey(old.c4flat(old.c4replay(h)[0])))
            if len(h)>=8:families.add(old.mirror(h[:8]))
        for j in co.get('jobs',co.get('c4_attempts',[])):
            h=j['history']
            if len(h)>=8:families.add(old.mirror(h[:8]))
    return families,boards,locks,histlocks
def plan():
    assert not (D/'SOURCE.jsonl.gz').exists()
    f=read('FROZEN.json');assert f['code']==sha(D/'acquire.py')
    fam,boards,locks,hl=historical();n=len(fam);rng=random.Random(SEED);jobs=[]
    for i in range(360):
        while True:
            h='';b=[0,0];z=[0]*7
            for j in range(8):
                opts=[c for c in range(7) if z[c]<6 and not old.won(b[j%2]|(1<<(7*c+z[c])))]
                c=rng.choice(opts);b[j%2]|=1<<(7*c+z[c]);z[c]+=1;h+=str(c+1)
            k=old.mirror(h)
            if k not in fam:break
        fam.add(k);jobs.append(dict(id=i,opening=h,family=k,seed=rng.randrange(2**63)))
    save('PLAN.json',dict(identity=identity(),frozen=sha(D/'FROZEN.json'),seed=SEED,historical_families=n,historical_boards=len(boards),source_locks=locks,history_locks=hl,jobs=jobs))
    print('PLAN',len(jobs),'historic',n,len(boards),flush=True)
def source():
    pl=read('PLAN.json');assert pl['identity']==identity() and pl['frozen']==sha(D/'FROZEN.json')
    fam,boards,locks,hl=historical();assert hl==pl['history_locks'] and locks==pl['source_locks']
    path=D/'SOURCE.jsonl.gz';assert not path.exists();roots=[];nn=queries=plies=0
    with p.Server(p.BIN) as srv,gzip.open(path,'wt',compresslevel=3) as fd:
        for j in pl['jobs']:
            assert j['family'] not in fam;fam.add(j['family'])
            h=j['opening'];rng=random.Random(j['seed']);eligible=[];steps=[]
            while True:
                b,z=old.c4replay(h);side=len(h)%2
                if any(old.won(v) for v in b) or len(h)==42:break
                legal=[c for c in range(7) if z[c]<6]
                own=[c for c in legal if old.won(b[side]|(1<<(7*c+z[c])))]
                opp=[c for c in legal if old.won(b[1-side]|(1<<(7*c+z[c])))]
                key=old.boardkey(old.c4flat(b))
                if len(h) in (24,26,28,30) and len(legal)>=3 and not own and not opp and key not in boards:eligible.append((h,legal,key))
                if len(h)>=30:break
                a=None
                if own:mv=own[0];kind='WIN'
                elif len(opp)==1:mv=opp[0];kind='BLOCK'
                elif len(legal)==1:mv=legal[0];kind='ONLY'
                elif rng.random()<.75:
                    a=srv.request(h,'K16N',32);assert 'error' not in a and a['best'] in legal
                    mv=a['best'];kind='N32';queries+=1;nn+=a['NN_calls']
                else:mv=rng.choice(legal);kind='UNIFORM'
                steps.append(dict(move=mv,kind=kind,NN=0 if a is None else a['NN_calls'],native_best=None if a is None else a['best']));plies+=1;h+=str(mv+1)
            root=None
            if eligible:
                hh,legal,key=eligible[-1];boards.add(key);root=dict(id=j['id'],group='TEST',family=j['family'],history=hh,legal=legal,depth=len(hh),tactical=False);roots.append(root)
            emit(fd,dict(job=j,history=h,steps=steps,selected=root))
            if j['id']%40==0:print('SOURCE',j['id'],'admitted',len(roots),'queries',queries,flush=True)
    save('COHORT.json',dict(plan=sha(D/'PLAN.json'),source=sha(path),roots=roots,source_queries=queries,source_NN=nn,source_plies=plies,attempts=len(pl['jobs'])))
    print('COHORT',len(roots),'queries',queries,'NN',nn,flush=True)
class Server:
    def __init__(self,bin):
        self.p=SP.Popen(['nice','-n','10',str(bin),'--server'],stdin=SP.PIPE,stdout=SP.PIPE,stderr=SP.DEVNULL,bufsize=0,env=dict(os.environ,RAYON_NUM_THREADS='1',QUARTZ_DISABLE_NN_CACHE='1',QUARTZ_DISABLE_ASYNC_PIPELINE='1',QUARTZ_DISABLE_QIPC_SHM='1'))
        self.buf=b'';self.cpu=old.process_clock(self.p.pid)
    def request(self,h,arm,cap=64,**opts):
        q=dict(cmd='research_connect4_evidence_cpu',history=h,arm=arm,cap=cap,seed=17,model_path=str(p.MODEL),model_identity=p.MID,**opts)
        t=time.perf_counter_ns();cpu=time.clock_gettime_ns(self.cpu)
        self.p.stdin.write((json.dumps(q,separators=(',',':'))+'\n').encode());self.p.stdin.flush()
        deadline=time.monotonic()+30
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
    def __exit__(self,t,*args):
        self.p.stdin.close()
        try:self.p.wait(timeout=5)
        except SP.TimeoutExpired:self.p.kill();self.p.wait()
        if t is None:assert self.p.returncode==0
def validate_live(a,r,m):
    assert 'error' not in a,a
    assert a['history']==r['history'] and a['legal']==r['legal'] and a['best'] in r['legal'] and a['result']['root_n']==a['visits'] and a['result']['evaluator_calls']==a['NN_calls']
    d=a['decision'];panel=d['panel'];pre=a['prefix']
    assert pre['root_n']==8 and panel[0]==pre['best'] and d['proof']==proof(r['history'])
    b=prefix_bin(pre,panel);assert b==d['bin'];pr=cert(d['proof'])
    if pr is not None:assert d['channel']=='CERTIFIED' and a['best']==pr
    else:
        c,v=choose(m,b,.002);assert d['channel']==('STOP' if c is None else CH[c])
        if c is None:assert a['visits']==8 and a['best']==panel[0]
        else:
            assert a['visits']==(32 if c<3 else 64) and abs(d['pred_gain']-m['gains'][b][c])<1e-12 and abs(d['pred_cost']-m['costs'][b][c])<1e-12
            if c%3==0:assert a['best']==a['result']['best']
            else:
                j,high=response(a['result'],panel);bounds={m:(l,u) for m,l,u in d['proof']}
                expected=panel[j] if j and m['switch'][c][j][high] and bounds[panel[j]][1]>bounds[panel[0]][0] else panel[0]
                assert a['best']==expected
def atlas():
    f=read('FROZEN.json');co=read('COHORT.json');assert f['code']==sha(D/'acquire.py') and f['identity']==identity()
    assert co['source']==sha(D/'SOURCE.jsonl.gz');m=f['controller']
    path=D/'ATLAS.jsonl.gz';assert not path.exists();req=nn=vis=0;matches=0;errors=[]
    with Server(BIN) as srv,gzip.open(path,'wt',compresslevel=3) as fd:
        for r in co['roots']:
            a=srv.request(r['history'],'K20LIVE',controller=m);validate_live(a,r,m)
            b=srv.request(r['history'],'K20N64');c=srv.request(r['history'],'K20MATCH',nn_budget=a['NN_calls']);t=srv.request(r['history'],'K20TIME',tail_budget_us=a['decision']['tail_us'])
            for z in (b,c,t):
                assert 'error' not in z,z
                assert z['prefix']==a['prefix'] and z['decision']['proof']==a['decision']['proof'] and z['best'] in r['legal'] and z['result']['root_n']==z['visits']
            matches+=c['NN_calls']==a['NN_calls']
            # paired fixed-channel atlas, including exact baseline parity
            fixed=[]
            for ch in CH:
                arm,cap=ch.split(':');v=srv.request(r['history'],arm,int(cap));o=p.check(v,r,arm,int(cap))
                assert v['evidence']['prefix']==a['prefix'] and o['panel']==a['decision']['panel']
                fixed.append(v)
            if cert(a['decision']['proof']) is None:assert b['result']==fixed[3]['result'] and b['best']==fixed[3]['best']
            if a['decision']['channel'] in CH:assert a['result']==fixed[CH.index(a['decision']['channel'])]['result']
            emit(fd,dict(root=r,live=a,normal=b,match=c,time=t,fixed=fixed))
            for z in (a,b,c,t,*fixed):req+=1;nn+=z['NN_calls'];vis+=z['visits']
            if r['id']%25==0:print('ATLAS',r['id'],req,'matches',matches,flush=True)
    save('ATLAS_LOCK.json',dict(identity=identity(),cohort=sha(D/'COHORT.json'),frozen=sha(D/'FROZEN.json'),sha256=sha(path),requests=req,NN=nn,visits=vis,nn_matches=matches))
    print('ATLAS',req,nn,vis,'matches',matches,flush=True)
def labels():
    co=read('COHORT.json');lk=read('ATLAS_LOCK.json');assert lk['sha256']==sha(D/'ATLAS.jsonl.gz') and lk['frozen']==sha(D/'FROZEN.json')
    out=[];ct=C.Counter()
    for r in co['roots']:
        q=old.oracle(r);item=dict(root=r,status=q['status'])
        if q['status']=='EXACT':
            try:
                v,n=p.ref.exact_wdl(r['history']);truth={a:(q['scores'][a]>0)-(q['scores'][a]<0) for a in r['legal']}
                item.update(scores=q['scores'],independent=v,nodes=n,agreement=v==truth)
                if v!=truth:item['status']='MISMATCH'
            except RuntimeError as e:item.update(status='INDEPENDENT_CAP',reason=str(e),scores=q['scores'])
        else:item['oracle_result']=q
        ct[item['status']]+=1;out.append(item)
    save('TEST_LABELS.json',dict(atlas=lk['sha256'],frozen=lk['frozen'],counts=dict(ct),rows=out))
    print('LABELS',dict(ct),flush=True)
def score():
    f=read('FROZEN.json');lk=read('ATLAS_LOCK.json');ll=read('TEST_LABELS.json');assert ll['atlas']==lk['sha256']==sha(D/'ATLAS.jsonl.gz') and ll['frozen']==sha(D/'FROZEN.json')
    m=f['controller'];labels={r['root']['id']:r for r in ll['rows'] if r['status']=='EXACT'};rows=[]
    for a in raw(D/'ATLAS.jsonl.gz'):
        r=a['root'];i=r['id']
        if i not in labels:continue
        u={int(k):(v+1)/2 for k,v in labels[i]['independent'].items()};assert all(l<=u[m0]<=v for m0,l,v in a['live']['decision']['proof'])
        live=a['live'];panel=live['decision']['panel'];anchor=panel[0];b=live['decision']['bin']
        rr=dict(id=i,family=r['family'],bin=b,panel=panel,anchor=anchor,anchor_u=u[anchor],full=max(u.values()),channel=live['decision']['channel'],pred_gain=live['decision']['pred_gain'],pred_cost=live['decision']['pred_cost'],arms={},fixed={},sensitivity={})
        for name in ('live','normal','match','time'):
            z=a[name];rr['arms'][name]=dict(best=z['best'],u=u[z['best']],regret=rr['full']-u[z['best']],NN=z['NN_calls'],visits=z['visits'],wall_ms=z['wall_ns']/1e6,cpu_ms=z['cpu_ns']/1e6,tail_us=z['decision']['tail_us'],whole_us=z['whole_us'])
        for c,z in enumerate(a['fixed']):
            j,h=response(z['result'],panel);obs={c:dict(j=j,high=h,native=z['best'],cost=z['NN_calls']-z['evidence']['prefix']['evaluator_calls'])}
            r0=dict(panel=panel,proof=live['decision']['proof'],obs=obs)
            d=decision(r0,c,m['switch']);g=u[d]-u[anchor]
            rr['fixed'][CH[c]]=dict(best=d,gain=g,pred_gain=m['gains'][b][c],u=u[d],regret=rr['full']-u[d],native_regret=rr['full']-u[z['best']],NN=z['NN_calls'])
        for price in PR:
            c,v=choose(m,b,price);cert0=cert(live['decision']['proof'])
            if cert0 is not None:d=cert0;ch='CERTIFIED'
            elif c is None:d=anchor;ch='STOP'
            else:ch=CH[c];d=rr['fixed'][ch]['best']
            rr['sensitivity'][str(price)]=dict(channel=ch,regret=rr['full']-u[d],u=u[d])
        assert rr['sensitivity']['0.002']['u']==rr['arms']['live']['u']
        rows.append(rr)
    assert rows
    with gzip.open(D/'SCORED.jsonl.gz','wt',compresslevel=3) as fd:
        for r in rows:emit(fd,r)
    avg=S.mean
    arms={}
    for name in ('live','normal','match','time'):
        z=[r['arms'][name] for r in rows]
        arms[name]={k:avg(x[k] for x in z) for k in ('u','regret','NN','visits','wall_ms','cpu_ms','tail_us','whole_us')}
    compare={}
    for name in ('normal','match','time'):
        z=[r['arms']['live']['u']-r['arms'][name]['u'] for r in rows];rng=random.Random(SEED);boot=sorted(avg(rng.choices(z,k=len(z))) for _ in range(4000))
        compare[name]=dict(gain=avg(z),positive=sum(x>0 for x in z),negative=sum(x<0 for x in z),ci=[boot[99],boot[3899]],NN_equal=sum(r['arms']['live']['NN']==r['arms'][name]['NN'] for r in rows),mean_abs_tail_us=avg(abs(r['arms']['live']['tail_us']-r['arms'][name]['tail_us']) for r in rows))
    gain=[r['arms']['live']['u']-r['anchor_u'] for r in rows];fixed={}
    for c in CH:
        z=[r['fixed'][c] for r in rows]
        fixed[c]={k:avg(x[k] for x in z) for k in ('gain','pred_gain','regret','native_regret','NN')}
        fixed[c].update(positive=sum(x['gain']>0 for x in z),negative=sum(x['gain']<0 for x in z))
    sensitivity={str(price):dict(regret=avg(r['sensitivity'][str(price)]['regret'] for r in rows),counts=dict(C.Counter(r['sensitivity'][str(price)]['channel'] for r in rows))) for price in PR}
    s=dict(n=len(rows),all_roots=len(ll['rows']),exact_actions=sum(len(z['independent']) for z in labels.values()),nodes=sum(z['nodes'] for z in labels.values()),anchor_regret=avg(r['full']-r['anchor_u'] for r in rows),channels=dict(C.Counter(r['channel'] for r in rows)),actual_gain=avg(gain),pred_gain=avg(r['pred_gain'] for r in rows),positive=sum(x>0 for x in gain),negative=sum(x<0 for x in gain),arms=arms,compare=compare,fixed=fixed,sensitivity=sensitivity,scored=sha(D/'SCORED.jsonl.gz'))
    save('SUMMARY.json',s);print('SUMMARY',json.dumps(s),flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
