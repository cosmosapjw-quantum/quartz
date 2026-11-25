"""Prospective replay: shared prior, fixed response channels, fresh C4 test."""
import pathlib,sys,json,gzip,hashlib,subprocess as SP,random,itertools,collections as C,statistics as S,math,time
D=pathlib.Path(__file__).resolve().parent
ROOT=D.parents[2]
P18=ROOT.parent/'quartz_kinetic18_20261126/docs/research/quartz_kinetic18_20261126'
sys.path.insert(0,str(P18))
import runner as r18
sys.path.insert(0,str(D))
from transition import STATES,CHANNELS,HVALUES,PRICES,estimate,choose,joint_log
p=r18.p;old=p.old
SEED=2026112619
BASE='8d39424f6adaeccfd2e39c18ed7461190dbd13f5'
def sha(p0):return hashlib.file_digest(p0.open('rb'),'sha256').hexdigest()
def save(n,x):
    f=D/n;assert not f.exists(),n;f.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n')
def read(n):return json.loads((D/n).read_text())
def raw(path):
    with gzip.open(path,'rt') as f:
        for l in f:yield json.loads(l)
def emit(f,x):f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n');f.flush()
def lock(path):
    rel=path.relative_to(ROOT.parent/'quartz_kinetic18_20261126')
    b=SP.check_output(['git','-C',str(ROOT),'show',BASE+':'+str(rel)])
    assert hashlib.sha256(b).hexdigest()==sha(path)
    return dict(commit=BASE,path=str(rel),sha256=sha(path))
def sources():
    return [lock(P18/n) for n in ('COHORT.json','ATLAS.jsonl.gz','ATLAS_LOCK.json','TRAIN_CAL_LABELS.json','runner.py')]
def feature(pre,panel):
    ca,n,w=p.pr.moments(pre);j=ca.index(panel[0]);pi=dict(pre['candidates']);x=[]
    for m in panel[1:]:
        i=ca.index(m);x.extend([(w[i]/(n[i]+4)-w[j]/(n[j]+4))/2,(n[i]-n[j])/8,pi[m]-pi[panel[0]]])
    return x
def query_row(t):
    a=t['response'];e=a['evidence'];r=t['root']
    panel=t['panels']['common'] if 'panels' in t else t['panel']
    assert panel[0]==e['prefix']['best'] and len(set(panel))==3
    c=f"{t['arm']}:{t['cap']}"
    ca,n,w=p.pr.moments(a['result'])
    z=max(range(3),key=lambda j:w[ca.index(panel[j])]/(n[ca.index(panel[j])]+4))
    assert z in range(3)
    return c,dict(z=z,cost=sum(x['step_us'] for x in e['phases'])/1000,phase_visits=t['cap']-8,phase_NN=a['NN_calls']-e['prefix']['evaluator_calls'],native=a['best'],cpu_ms=a['cpu_ns']/1e6,wall_ms=a['wall_ns']/1e6,prefix_us=e['prefix_us'],total_NN=a['NN_calls'])
def scenarios(rows,labels):
    ll={r['root']['id']:r for r in labels if r['status']=='EXACT'};out={}
    for t in rows:
        r=t['root'];i=r['id']
        if i not in ll:continue
        l=ll[i];assert r==l['root']
        u={int(a):(v+1)/2 for a,v in l['independent'].items()}
        assert set(u)==set(r['legal'])
        panel=t['panels']['common'] if 'panels' in t else t['panel']
        pre=t['response']['evidence']['prefix'];x=feature(pre,panel);c,v=query_row(t)
        y=[int(2*(u[m]-u[panel[0]])) for m in panel[1:]]
        assert tuple(y) in STATES
        if i not in out:out[i]=dict(id=i,group=r['group'],family=r['family'],panel=panel,x=x,y=y,utility=[u[m] for m in panel],full=max(u.values()),all_u=u,obs={},prefix_NN=pre['evaluator_calls'])
        s=out[i];assert s['x']==x and s['y']==y and s['panel']==panel and c not in s['obs'];s['obs'][c]=v
    assert set(out)==set(ll) and all(set(v['obs'])==set(CHANNELS) for v in out.values())
    return list(out.values())
def plan():
    f,b,locks=old.c4historical()
    oldco=old.read('C06_COHORT.json');f.update(j['family'] for j in old.read('C06_PLAN.json')['c4'])
    for r in oldco['roots']:
        if r['domain']=='C4':b.add(old.boardkey(old.c4flat(old.c4replay(r['history'])[0])))
    for path in (r18.P17/'COHORT.json',r18.P17/'quiet/COHORT.json',P18/'COHORT.json'):
        co=json.loads(path.read_text());f.update(j['family'] for j in co['jobs'])
        for r in co['roots']:b.add(old.boardkey(old.c4flat(old.c4replay(r['history'])[0])))
    available=sorted({old.mirror(''.join(t)) for t in itertools.product('1234567',repeat=5)}-f)
    rng=random.Random(SEED);rng.shuffle(available);assert len(available)==106
    jobs=[];roots=[]
    for i,k in enumerate(available[:100]):
        h=k if rng.randrange(2)==0 else k.translate(str.maketrans('1234567','7654321'))
        seed=rng.randrange(2**63);g=random.Random(seed);op=h;depth=(24,26,28)[i%3]
        board,z=old.c4replay(h)
        while len(h)<depth:
            side=len(h)%2;opts=[]
            for c in range(7):
                if z[c]==6:continue
                bit=1<<(7*c+z[c]);board[side]|=bit;z[c]+=1
                if not old.won(board[side]) and not any(old.canwin(board,z,j) for j in (0,1)):opts.append(c)
                board[side]^=bit;z[c]-=1
            if not opts:break
            c=g.choice(opts);board[side]|=1<<(7*c+z[c]);z[c]+=1;h+=str(c+1)
        legal=[c for c in range(7) if z[c]<6];key=old.boardkey(old.c4flat(board))
        ok=len(h)==depth and len(legal)>=3 and key not in b
        jobs.append(dict(id=i,group='TEST',family=k,opening=op,seed=seed,depth=depth,history=h,accepted=ok))
        if ok:
            assert not any(old.canwin(board,z,j) for j in (0,1))
            b.add(key);roots.append(dict(id=i,group='TEST',family=k,history=h,legal=legal,depth=depth,tactical=False))
    save('COHORT.json',dict(identity=r18.identity(),seed=SEED,available=len(available),source_locks=locks,previous=sources(),jobs=jobs,roots=roots))
    print('C4 COHORT',len(jobs),len(roots),flush=True)
def fit():
    assert not (D/'ATLAS.jsonl.gz').exists() and not (D/'TEST_LABELS.json').exists()
    sl=sources();co=read('COHORT.json');assert co['previous']==sl
    l=json.loads((P18/'TRAIN_CAL_LABELS.json').read_text())
    rows=scenarios((t for t in raw(P18/'ATLAS.jsonl.gz') if t['root']['group'] in ('TRAIN','CAL')),l['rows'])
    tr=[r for r in rows if r['group']=='TRAIN'];cal=[r for r in rows if r['group']=='CAL']
    assert len(tr)==110 and len(cal)==74
    training=[{k:r[k] for k in ('id','x','y','obs')} for r in tr]
    scores={}
    for h in HVALUES:
        ll=[]
        for r in cal:
            q=estimate(training,r['x'],h)
            ll.extend(joint_log(q,r['y'],c,r['obs'][c]['z']) for c in CHANNELS)
        scores[str(h)]=S.mean(ll)
    selected=min(HVALUES,key=lambda h:scores[str(h)])
    support=[dict(positive=sum(r['y'][i]>0 for r in tr),negative=sum(r['y'][i]<0 for r in tr),zero=sum(r['y'][i]==0 for r in tr)) for i in range(2)]
    save('FROZEN.json',dict(sources=sl,cohort=sha(D/'COHORT.json'),fit_code=sha(D/'transition.py'),runner_code=sha(D/'run_k19.py'),training=training,cal_n=len(cal),scores=scores,selected=selected,support=support,prices=PRICES,selection='CAL pooled joint log score only; no K18 TEST labels'))
    print('FIT',selected,scores,support,flush=True)
def atlas():
    co=read('COHORT.json');f=read('FROZEN.json');assert f['cohort']==sha(D/'COHORT.json') and f['fit_code']==sha(D/'transition.py') and f['runner_code']==sha(D/'run_k19.py')
    path=D/'ATLAS.jsonl.gz';assert not path.exists()
    nn=vis=cnt=0
    with p.Server(p.BIN) as s,gzip.open(path,'wt',compresslevel=3) as fd:
        for r in co['roots']:
            pre=None;common=None
            for cap in p.CAPS:
                for arm in p.ARMS[r['id']%3:]+p.ARMS[:r['id']%3]:
                    a=s.request(r['history'],arm,cap);o=p.check(a,r,arm,cap)
                    if pre is None:pre=a['evidence']['prefix'];common=o['panel']
                    assert pre==a['evidence']['prefix'] and common==o['panel']
                    emit(fd,dict(root=r,cap=cap,arm=arm,response=a,panel=common))
                    nn+=a['NN_calls'];vis+=cap;cnt+=1
    save('ATLAS_LOCK.json',dict(cohort=sha(D/'COHORT.json'),frozen=sha(D/'FROZEN.json'),sha256=sha(path),requests=cnt,visits=vis,NN=nn,identity=r18.identity()))
    print('C4 ATLAS',cnt,vis,nn,flush=True)
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
    print('C4 LABELS',dict(ct),flush=True)
def evaluate():
    f=read('FROZEN.json');assert f['fit_code']==sha(D/'transition.py') and f['runner_code']==sha(D/'run_k19.py')
    labels0=read('TEST_LABELS.json');assert labels0['frozen']==sha(D/'FROZEN.json')
    rows=scenarios(raw(D/'ATLAS.jsonl.gz'),labels0['rows']);out=[]
    for r in rows:
        v=r['utility'];u=r['all_u'];tick=time.process_time_ns();q=estimate(f['training'],r['x'],f['selected']);chosen={str(price):choose(q,price) for price in PRICES};overhead=(time.process_time_ns()-tick)/1e6
        i=q['stop'];p0=v[i];channels={}
        for c in CHANNELS:
            a=q['channels'][c];o=r['obs'][c];z=o['z'];d=a['choices'][z]
            channels[c]=dict(z=z,decision=d,utility=v[d],native_u=u[o['native']],gain=v[d]-p0,pred_gain=a['delta'],pred_cost=a['cost'],pred_risk=a['risk'],cost=o['cost'],visits=o['phase_visits'],NN=o['phase_NN'],log=joint_log(q,r['y'],c,z),pi=a['pi'],J=a['J'],wall_ms=o['wall_ms'],cpu_ms=o['cpu_ms'])
        pol={}
        for price in PRICES:
            c,net=chosen[str(price)]
            a=channels[c] if c!='STOP' else dict(utility=p0,gain=0.,pred_gain=0.,cost=0.,pred_cost=0.,visits=0,NN=0)
            pol[str(price)]=dict(channel=c,pred_net=net,**{k:a[k] for k in ('utility','gain','pred_gain','cost','pred_cost','visits','NN')},actual_net=a['gain']-price*a['cost'],regret=r['full']-a['utility'])
        out.append(dict(id=r['id'],family=r['family'],x=r['x'],y=r['y'],panel=r['panel'],utility=v,full=r['full'],omission=r['full']-max(v),prefix_native=v[0],stop=p0,stop_decision=i,stop_risk=q['risk'],p=q['p'],mu=q['mu'],ess=q['ess'],overhead_ms=overhead,prefix_NN=r['prefix_NN'],channels=channels,policy=pol))
    with gzip.open(D/'PREDICTIONS.jsonl.gz','wt',compresslevel=3) as fd:
        for x in out:emit(fd,x)
    def avg(a):return S.mean(a)
    fixed={}
    for c in CHANNELS:
        a=[r['channels'][c] for r in out];g=[r['gain'] for r in a]
        fixed[c]=dict(regret=avg(r['full']-x['utility'] for r,x in zip(out,a)),native_regret=avg(r['full']-x['native_u'] for r,x in zip(out,a)),mean_gain=avg(g),positive=sum(x>0 for x in g),negative=sum(x<0 for x in g),pred_gain=avg(x['pred_gain'] for x in a),cost_ms=avg(x['cost'] for x in a),pred_cost_ms=avg(x['pred_cost'] for x in a),phase_NN=avg(x['NN'] for x in a),mean_log=avg(x['log'] for x in a),wall_ms=avg(x['wall_ms'] for x in a),cpu_ms=avg(x['cpu_ms'] for x in a))
    policies={}
    for price in PRICES:
        a=[r['policy'][str(price)] for r in out];g=[x['gain'] for x in a];v=[x['utility']-r['channels']['K16N:64']['native_u'] for r,x in zip(out,a)]
        rng=random.Random(SEED);boot=sorted(avg(rng.choices(v,k=len(v))) for _ in range(4000))
        policies[str(price)]=dict(n=len(a),choice_counts=dict(C.Counter(x['channel'] for x in a)),regret=avg(x['regret'] for x in a),gain=avg(g),positive=sum(x>0 for x in g),negative=sum(x<0 for x in g),pred_gain=avg(x['pred_gain'] for x in a),mean_net=avg(x['actual_net'] for x in a),pred_net=avg(x['pred_net'] for x in a),cost_ms=avg(x['cost'] for x in a),pred_cost_ms=avg(x['pred_cost'] for x in a),visits=avg(x['visits']+8 for x in a),NN=avg(x['NN']+r['prefix_NN'] for r,x in zip(out,a)),gain_vs_N64=avg(v),positive_vs_N64=sum(x>0 for x in v),negative_vs_N64=sum(x<0 for x in v),gain_vs_N64_ci=[boot[99],boot[3899]])
    s=dict(n=len(out),frozen=sha(D/'FROZEN.json'),test=sha(D/'TEST_LABELS.json'),predictions=sha(D/'PREDICTIONS.jsonl.gz'),prefix_native_regret=avg(r['full']-r['prefix_native'] for r in out),model_stop_regret=avg(r['full']-r['stop'] for r in out),mean_omission=avg(r['omission'] for r in out),model_stop_risk=avg(r['stop_risk'] for r in out),actual_stop_panel_risk=avg(max(r['utility'])-r['stop'] for r in out),model_overhead_ms=avg(r['overhead_ms'] for r in out),fixed=fixed,policies=policies)
    save('SUMMARY.json',s)
    print('SUMMARY',json.dumps(s),flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
