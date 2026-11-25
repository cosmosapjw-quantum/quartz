"""Fresh native sequential stopping; historical data used only for exclusions/diagnostics."""
import pathlib,sys,json,gzip,hashlib,subprocess as SP,random,statistics as S,collections as C,math
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2];R=ROOT/'docs/research'
P20=ROOT.parent/'quartz_kinetic20_20261126/docs/research/quartz_kinetic20_20261126'
sys.path.insert(0,str(P20));import acquire as k20
sys.path.insert(0,str(D))
from geometry import phi,kl,close
p=k20.p;old=k20.old;BIN=ROOT.parent/'quartz_kinetic21_target_20261126/mcts_demo'
BASE='a1586354b78f78c3eaf6d51e5e88b0e8fb70f079';SEED=2026112621;PRICE=.002
def sha(path):return hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
def read(n):return json.loads((D/n).read_text())
def save(n,x):
    f=D/n;assert not f.exists(),n;f.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n')
def raw(path):
    with gzip.open(path,'rt') as f:
        for l in f:yield json.loads(l)
def emit(f,x):f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n');f.flush()
def lock(path):
    rel=str(path.relative_to(ROOT));b=SP.check_output(['git','-C',str(ROOT),'show',BASE+':'+rel])
    assert hashlib.sha256(b).hexdigest()==sha(path);return dict(commit=BASE,path=rel,sha256=sha(path))
def ident():
    b=read('RUST_BUILD_IDENTITY.json');assert sha(BIN)==b['binary_sha256']
    return dict(binary=sha(BIN),source=b['source_head'],parent=k20.identity())
def historical():
    f,b,l,h=k20.historical();locks=[]
    for n in ('PLAN.json','COHORT.json','SOURCE.jsonl.gz'):
        locks.append(lock(R/'quartz_kinetic20_20261126'/n))
    co=json.loads((R/'quartz_kinetic20_20261126/COHORT.json').read_text())
    for r in co['roots']:
        b.add(old.boardkey(old.c4flat(old.c4replay(r['history'])[0])))
    for j in json.loads((R/'quartz_kinetic20_20261126/PLAN.json').read_text())['jobs']:f.add(j['family'])
    return f,b,locks,l,h
def plan():
    f,b,l,oldl,h=historical();n=len(f);rng=random.Random(SEED);jobs=[]
    for i in range(480):
        while True:
            seq='';board=[0,0];ht=[0]*7
            for j in range(8):
                opts=[c for c in range(7) if ht[c]<6 and not old.won(board[j%2]|(1<<(7*c+ht[c])))]
                c=rng.choice(opts);board[j%2]|=1<<(7*c+ht[c]);ht[c]+=1;seq+=str(c+1)
            k=old.mirror(seq)
            if k not in f:break
        f.add(k);jobs.append(dict(id=i,group='TRAIN' if i<240 else 'CAL' if i<360 else 'TEST',opening=seq,family=k,seed=rng.randrange(2**63)))
    save('PLAN.json',dict(identity=ident(),seed=SEED,historical_families=n,historical_boards=len(b),K20_locks=l,prior_locks=oldl,history_locks=h,jobs=jobs))
    print('PLAN',len(jobs),n,len(b),flush=True)
def source():
    pl=read('PLAN.json');assert pl['identity']==ident();f,b,l,oldl,h=historical();assert l==pl['K20_locks'] and oldl==pl['prior_locks'] and h==pl['history_locks']
    path=D/'SOURCE.jsonl.gz';assert not path.exists();roots=[];queries=nn=plies=0
    with p.Server(p.BIN) as srv,gzip.open(path,'wt',compresslevel=3) as fd:
        for j in pl['jobs']:
            assert j['family'] not in f;f.add(j['family'])
            hh=j['opening'];rng=random.Random(j['seed']);eligible=[];steps=[]
            while True:
                board,z=old.c4replay(hh);side=len(hh)%2
                if any(old.won(x) for x in board) or len(hh)==42:break
                legal=[c for c in range(7) if z[c]<6]
                own=[c for c in legal if old.won(board[side]|(1<<(7*c+z[c])))]
                opp=[c for c in legal if old.won(board[1-side]|(1<<(7*c+z[c])))]
                key=old.boardkey(old.c4flat(board))
                if len(hh) in (24,26,28,30) and len(legal)>=3 and not own and not opp and key not in b:eligible.append((hh,legal,key))
                if len(hh)>=30:break
                a=None
                if own:mv=own[0];kind='WIN'
                elif len(opp)==1:mv=opp[0];kind='BLOCK'
                elif len(legal)==1:mv=legal[0];kind='ONLY'
                elif rng.random()<.75:
                    a=srv.request(hh,'K16N',32);assert 'error' not in a and a['best'] in legal
                    mv=a['best'];kind='N32';queries+=1;nn+=a['NN_calls']
                else:mv=rng.choice(legal);kind='UNIFORM'
                steps.append(dict(move=mv,kind=kind,NN=0 if a is None else a['NN_calls']));plies+=1;hh+=str(mv+1)
            root=None
            if eligible:
                h0,legal,key=eligible[-1];b.add(key);root=dict(id=j['id'],group=j['group'],family=j['family'],history=h0,legal=legal,depth=len(h0),tactical=False);roots.append(root)
            emit(fd,dict(job=j,history=hh,steps=steps,selected=root))
            if j['id']%60==0:print('SOURCE',j['id'],len(roots),flush=True)
    save('COHORT.json',dict(plan=sha(D/'PLAN.json'),source=sha(path),roots=roots,queries=queries,NN=nn,plies=plies,counts=dict(C.Counter(x['group'] for x in roots))))
    print('COHORT',len(roots),C.Counter(x['group'] for x in roots),'NN',nn,'queries',queries,flush=True)
def check(a,r):
    assert 'error' not in a,a
    assert a['history']==r['history'] and a['legal']==r['legal'] and a['best'] in r['legal'] and a['proof']==k20.proof(r['history'])
    assert a['states'][-1]['visits']==a['visits']==a['result']['root_n'] and a['states'][-1]['nn']==a['NN_calls']==a['result']['evaluator_calls']
    prev=None
    for s in a['states']:
        k=s['len'];n=s['n'][:k];w=s['w'][:k];assert k==len(r['legal']) and set(s['moves'][:k])==set(r['legal']) and sum(n)==s['visits']
        assert all(abs(x)<=y+1e-6 for x,y in zip(w,n))
        x=[(ww/(nn+4)+1)/2 for nn,ww in zip(n,w)];f1,q=phi(x)
        if prev is None:xx=[.5]*k;f0=.5;pp=[1/k]*k
        else:
            assert prev['moves']==s['moves'] and all(x>=y for x,y in zip(s['n'],prev['n']))
            xx=[(ww/(nn+4)+1)/2 for nn,ww in zip(prev['n'][:k],prev['w'][:k])];f0,pp=phi(xx)
        kld=kl(pp,q);dr=sum(v*(a-b) for v,a,b in zip(pp,x,xx))
        close(s['phi'],f1);close(s['kl'],kld);close(s['drift'],dr);close(s['bregman'],f1-f0-dr);close(s['bregman'],.25*kld)
        cnt=sorted(n,reverse=True);b=4*int(cnt[0]/s['visits']>=.75)+2*int(cnt[0]-cnt[1]>=2)+int(kld>=.02)
        assert s['bin']==b
        prev=s
    r0=a['result'];s=a['states'][-1];assert s['best']==r0['best']
    e={x['move']:x for x in r0['edges']}
    assert s['moves'][:s['len']]==[m for m,v in r0['candidates']]
    for i,m in enumerate(s['moves'][:s['len']]):
        x=e.get(m,{});assert s['n'][i]==x.get('n',0);close(s['w'][i],x.get('w',0.));close(s['m2'][i],x.get('m2',0.))
    cert=k20.cert(a['proof'])
    assert a['best']==(s['best'] if cert is None else cert)
    if a['arm']=='K21TRACE' and cert is None:assert [s['visits'] for s in a['states']]==list(range(8,65,8))
def trace_train():
    co=read('COHORT.json');assert co['source']==sha(D/'SOURCE.jsonl.gz')
    path=D/'TRAIN_CAL_TRACE.jsonl.gz';assert not path.exists()
    nn=vis=cnt=0
    with k20.Server(BIN) as srv,gzip.open(path,'wt',compresslevel=3) as fd:
        for r in co['roots']:
            if r['group']=='TEST':continue
            a=srv.request(r['history'],'K21TRACE');check(a,r)
            # source selected quiet roots, no exact certificate expected; if certificate, exclude from learning but preserve.
            emit(fd,dict(root=r,response=a));nn+=a['NN_calls'];vis+=a['visits'];cnt+=1
    save('TRAIN_TRACE_LOCK.json',dict(cohort=sha(D/'COHORT.json'),sha256=sha(path),requests=cnt,NN=nn,visits=vis,identity=ident()))
    print('TRAIN_CAL TRACE',cnt,nn,vis,flush=True)
def labels(part):
    co=read('COHORT.json');assert co['source']==sha(D/'SOURCE.jsonl.gz')
    if part=='TEST':
        f=read('FROZEN.json');lk=read('TEST_LOCK.json');assert lk['frozen']==sha(D/'FROZEN.json') and lk['sha256']==sha(D/'TEST_ATLAS.jsonl.gz')
    else:lk=read('TRAIN_TRACE_LOCK.json');assert lk['sha256']==sha(D/'TRAIN_CAL_TRACE.jsonl.gz')
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
def fit():
    assert not (D/'TEST_ATLAS.jsonl.gz').exists() and not (D/'TEST_LABELS.json').exists()
    lab=read('TRAIN_CAL_LABELS.json');assert lab['atlas']==sha(D/'TRAIN_CAL_TRACE.jsonl.gz')
    ll={r['root']['id']:r for r in lab['rows'] if r['status']=='EXACT'}
    rows=[]
    for t in raw(D/'TRAIN_CAL_TRACE.jsonl.gz'):
        r=t['root'];a=t['response']
        if r['id'] not in ll or a['cause']=='CERTIFIED':continue
        u={int(k):(v+1)/2 for k,v in ll[r['id']]['independent'].items()}
        s=a['states'];rows.append(dict(id=r['id'],group=r['group'],s=[dict(b=z['bin'],u=u[z['best']],nn=z['nn']) for z in s]))
    tr=[r for r in rows if r['group']=='TRAIN'];cal=[r for r in rows if r['group']=='CAL'];assert len(tr)>10 and len(cal)>5
    fits={}
    def stop(row,table):
        return next((i for i,s in enumerate(row['s'][:-1]) if table[i][s['b']]<=0),7)
    for mode in ('basic','geometry'):
        table=[[0.]*8 for _ in range(7)];stats=[None]*7
        for t in range(6,-1,-1):
            targets=[]
            for r in tr:
                s=r['s'];j=next((j for j in range(t+1,7) if table[j][s[j]['b']]<=0),7)
                y=s[j]['u']-s[t]['u']-PRICE*(s[j]['nn']-s[t]['nn']);targets.append((s[t]['b'],y))
            g=S.mean(y for b,y in targets);st=[]
            for b in range(8):
                group=[y for x,y in targets if x==b or mode=='basic' and x//2==b//2]
                m=(sum(group)+8*g)/(len(group)+8);table[t][b]=m;st.append(dict(n=len(group),positive=sum(x>0 for x in group),negative=sum(x<0 for x in group),mean=m,pooled=g))
            stats[t]=st
        cal0=[(r,stop(r,table)) for r in cal]
        net=S.mean(r['s'][j]['u']-PRICE*r['s'][j]['nn'] for r,j in cal0)
        fits[mode]=dict(table=table,stats=stats,cal_net=net,cal_mean_NN=S.mean(r['s'][j]['nn'] for r,j in cal0),cal_mean_u=S.mean(r['s'][j]['u'] for r,j in cal0),cal_stops=sum(j<7 for r,j in cal0))
    selected=max(fits,key=lambda k:fits[k]['cal_net'])
    save('FROZEN.json',dict(cohort=sha(D/'COHORT.json'),trace=sha(D/'TRAIN_CAL_TRACE.jsonl.gz'),labels=sha(D/'TRAIN_CAL_LABELS.json'),code=sha(D/'sequential.py'),identity=ident(),price=PRICE,train_ids=[r['id'] for r in tr],cal_ids=[r['id'] for r in cal],fits=fits,selected=selected))
    print('FIT',len(tr),len(cal),selected,{k:{a:v[a] for a in ('cal_net','cal_mean_NN','cal_mean_u','cal_stops')} for k,v in fits.items()},flush=True)
def live():
    f=read('FROZEN.json');assert f['code']==sha(D/'sequential.py') and f['identity']==ident()
    co=read('COHORT.json');assert f['cohort']==sha(D/'COHORT.json')
    path=D/'TEST_ATLAS.jsonl.gz';assert not path.exists()
    nn=vis=cnt=0
    with k20.Server(BIN) as srv,gzip.open(path,'wt',compresslevel=3) as fd:
        # 12 previously known roots, not part of the new scientific test, verify parent parity.
        oldco=json.loads((R/'quartz_kinetic20_20261126/COHORT.json').read_text())
        parity=0
        for r in oldco['roots'][:12]:
            a=srv.request(r['history'],'K21TRACE');b=srv.request(r['history'],'K20N64')
            check(a,r);assert a['result']==b['result'] and a['best']==b['best'];parity+=1
        for r in co['roots']:
            if r['group']!='TEST':continue
            reps=[]
            for rep in range(2):
                aa={};order=['normal','basic','geometry'] if (r['id']+rep)%2==0 else ['geometry','basic','normal']
                for key in order:
                    a=srv.request(r['history'],'K21TRACE') if key=='normal' else srv.request(r['history'],'K21LIVE',seq_table=f['fits'][key]['table'])
                    check(a,r);aa[key]=a
                std=aa['normal']
                for key in ('basic','geometry'):
                    a=aa[key];assert all({k:v for k,v in x.items() if k!='net'}=={k:v for k,v in y.items() if k!='net'} for x,y in zip(a['states'],std['states']))
                    if a['cause']!='CERTIFIED':
                        table=f['fits'][key]['table'];j=next((i for i,x in enumerate(std['states'][:-1]) if table[i][x['bin']]<=0),7)
                        assert a['visits']==std['states'][j]['visits'] and a['best']==std['states'][j]['best'] and a['NN_calls']==std['states'][j]['nn']
                a=aa[f['selected']]
                b=srv.request(r['history'],'K21MATCH',nn_budget=a['NN_calls'])
                t=srv.request(r['history'],'K21TIME',tail_budget_us=a['core_us'])
                for x in (b,t):check(x,r)
                assert b['NN_calls']<=a['NN_calls'] and (b['NN_calls']==a['NN_calls'] or b['visits']==256)
                aa['match']=b;aa['time']=t
                for x in aa.values():nn+=x['NN_calls'];vis+=x['visits'];cnt+=1
                reps.append(aa)
            for key in ('normal','basic','geometry','match'):
                assert reps[0][key]['best']==reps[1][key]['best'] and reps[0][key]['NN_calls']==reps[1][key]['NN_calls']
            emit(fd,dict(root=r,reps=reps))
    save('TEST_LOCK.json',dict(cohort=sha(D/'COHORT.json'),frozen=sha(D/'FROZEN.json'),sha256=sha(path),requests=cnt,NN=nn,visits=vis,prior_parity=parity,identity=ident()))
    print('TEST ATLAS',cnt,nn,vis,'prior parity',parity,flush=True)
def score():
    f=read('FROZEN.json');lk=read('TEST_LOCK.json');lab=read('TEST_LABELS.json');assert lab['atlas']==lk['sha256']==sha(D/'TEST_ATLAS.jsonl.gz') and lk['frozen']==sha(D/'FROZEN.json')
    ll={z['root']['id']:z for z in lab['rows'] if z['status']=='EXACT'};rows=[]
    for t in raw(D/'TEST_ATLAS.jsonl.gz'):
        r=t['root'];i=r['id']
        if i not in ll:continue
        u={int(k):(v+1)/2 for k,v in ll[i]['independent'].items()};assert all(l<=u[a]<=h for a,l,h in t['reps'][0]['normal']['proof'])
        z=dict(id=i,family=r['family'],full=max(u.values()),arms={})
        for key in ('normal','basic','geometry','match','time'):
            aa=[x[key] for x in t['reps']]
            z['arms'][key]=dict(u=S.mean(u[a['best']] for a in aa),regret=S.mean(max(u.values())-u[a['best']] for a in aa),NN=S.mean(a['NN_calls'] for a in aa),visits=S.mean(a['visits'] for a in aa),cpu_ms=S.mean(a['cpu_ns']/1e6 for a in aa),wall_ms=S.mean(a['wall_ns']/1e6 for a in aa),core_us=S.mean(a['core_us'] for a in aa),meta_us=S.mean(a['meta_ns']/1e3 for a in aa),stop=int(aa[0]['cause']=='STOP'),rep=[dict(u=u[a['best']],NN=a['NN_calls'],cpu=a['cpu_ns'],core=a['core_us']) for a in aa])
        rows.append(z)
    selected=f['selected'];arms={}
    for key in rows[0]['arms']:
        a=[r['arms'][key] for r in rows]
        arms[key]={k:S.mean(x[k] for x in a) for k in ('u','regret','NN','visits','cpu_ms','wall_ms','core_us','meta_us')};arms[key]['stops']=sum(x['stop'] for x in a);arms[key]['net']=arms[key]['u']-PRICE*arms[key]['NN']
    def pair(a,b,subset):
        v=[r['arms'][a]['u']-r['arms'][b]['u'] for r in subset]
        if not v:return dict(n=0)
        rng=random.Random(SEED);bs=sorted(S.mean(rng.choices(v,k=len(v))) for _ in range(4000))
        return dict(n=len(v),gain=S.mean(v),positive=sum(x>0 for x in v),negative=sum(x<0 for x in v),ci=[bs[99],bs[3899]],net_gain=S.mean(r['arms'][a]['u']-r['arms'][b]['u']-PRICE*(r['arms'][a]['NN']-r['arms'][b]['NN']) for r in subset))
    exact=[r for r in rows if all(a['NN']==b['NN'] for a,b in zip(r['arms'][selected]['rep'],r['arms']['match']['rep']))]
    near=[r for r in rows if all(abs(a['cpu']-b['cpu'])<=.1*a['cpu'] for a,b in zip(r['arms'][selected]['rep'],r['arms']['time']['rep']))]
    out=dict(n=len(rows),all_roots=len(lab['rows']),selected=selected,actions=sum(len(v['independent']) for v in ll.values()),nodes=sum(v['nodes'] for v in ll.values()),arms=arms,geometry_vs_basic=pair('geometry','basic',rows),selected_vs_normal=pair(selected,'normal',rows),selected_vs_match=pair(selected,'match',exact),selected_vs_time_all=pair(selected,'time',rows),selected_vs_time_cpu10=pair(selected,'time',near),rows=rows)
    save('SUMMARY.json',out);print('SCORE',json.dumps({k:v for k,v in out.items() if k!='rows'}),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='labels':labels(sys.argv[2])
    else:globals()[sys.argv[1]]()
