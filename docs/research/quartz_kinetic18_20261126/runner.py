"""Fresh native Connect4 atlas, frozen train/CAL model, separately held-out TEST.
All paths refer to this worktree and immutable K16/K17 sources."""
import pathlib,sys,json,gzip,hashlib,subprocess as SP,random,itertools,collections as C,statistics as S,math,time
D=pathlib.Path(__file__).resolve().parent
ROOT=D.parents[2]
P17=ROOT.parent/'quartz_kinetic17_20261126/docs/research/quartz_kinetic17_20261126'
sys.path.insert(0,str(P17))
import pilot as p
sys.path.insert(0,str(D))
from joint import STATES, predict, decide, metrics, HVALUES
# p inserts K16 paths; our unique joint module is unaffected.
OLD=p.old
ARMS=p.ARMS;CAPS=p.CAPS;MODES=('common','native')
SEED=2026112618;ATTEMPTS=300
def sha(path):return hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
def save(name,x):
    path=D/name;assert not path.exists(),name
    path.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n')
def read(name):return json.loads((D/name).read_text())
def gread(name):
    with gzip.open(D/name,'rt') as f:return [json.loads(x) for x in f]
def gwrite(name,rows):
    path=D/name;assert not path.exists()
    with gzip.open(path,'wt',compresslevel=3) as f:
        for x in rows:f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n')
def git(*a):return SP.check_output(['git','-C',str(ROOT),*a],text=True).strip()
def locked(path,commit):
    path=path.resolve();base=ROOT.parent/'quartz_kinetic17_20261126'
    rel=path.relative_to(base)
    raw=SP.check_output(['git','-C',str(base),'show',commit+':'+str(rel)])
    assert hashlib.sha256(raw).hexdigest()==sha(path)
    return dict(commit=commit,path=str(rel),sha256=sha(path))
def identity():
    r=p.identity()
    assert r['binary']=='c238b84ca99695e8415974d122b17d576f6b8a30d135d8a5ff92fa88a18bf64b'
    return r
def plan():
    f,b,locks=OLD.c4historical()
    oldco=OLD.read('C06_COHORT.json')
    f.update(j['family'] for j in OLD.read('C06_PLAN.json')['c4'])
    for r in oldco['roots']:
        if r['domain']=='C4':b.add(OLD.boardkey(OLD.c4flat(OLD.c4replay(r['history'])[0])))
    k17=[]
    for path in [P17/'COHORT.json',P17/'quiet/COHORT.json']:
        k17.append(locked(path,'751e36fd76d8d1e945f3c2a0419dd9dde9ccc976'))
        co=json.loads(path.read_text());f.update(j['family'] for j in co['jobs'])
        for r in co['roots']:b.add(OLD.boardkey(OLD.c4flat(OLD.c4replay(r['history'])[0])))
    allfamilies={OLD.mirror(''.join(t)) for t in itertools.product('1234567',repeat=5)}
    unused=sorted(allfamilies-f);rng=random.Random(SEED);rng.shuffle(unused)
    assert len(unused)>=ATTEMPTS
    jobs=[];roots=[];nb=len(b);nold=len(f)
    for i,k in enumerate(unused[:ATTEMPTS]):
        h=k if rng.randrange(2)==0 else k.translate(str.maketrans('1234567','7654321'))
        assert OLD.mirror(h)==k
        op=h;seed=rng.randrange(2**63);r=random.Random(seed);depth=(24,26,28)[i%3]
        group='TRAIN' if i<120 else 'CAL' if i<200 else 'TEST'
        board,z=OLD.c4replay(h)
        while len(h)<depth:
            side=len(h)%2;opts=[]
            for c in range(7):
                if z[c]==6:continue
                bit=1<<(7*c+z[c]);board[side]|=bit;z[c]+=1
                if not OLD.won(board[side]) and not any(OLD.canwin(board,z,x) for x in (0,1)):opts.append(c)
                board[side]^=bit;z[c]-=1
            if not opts:break
            c=r.choice(opts);board[side]|=1<<(7*c+z[c]);z[c]+=1;h+=str(c+1)
        legal=[c for c in range(7) if z[c]<6];key=OLD.boardkey(OLD.c4flat(board))
        ok=len(h)==depth and len(legal)>=3 and key not in b
        jobs.append(dict(id=i,group=group,family=k,opening=op,seed=seed,depth=depth,history=h,accepted=ok))
        if ok:
            assert not any(OLD.canwin(board,z,x) for x in (0,1))
            b.add(key);roots.append(dict(id=i,group=group,family=k,history=h,legal=legal,depth=depth,tactical=False))
    save('COHORT.json',dict(identity=identity(),seed=SEED,old_families=nold,old_boards=nb,available=len(unused),source_locks=locks,k17_locks=k17,jobs=jobs,roots=roots))
    print('COHORT',len(roots),dict(C.Counter(x['group'] for x in roots)),flush=True)
def features(a,panel):
    e=a['evidence'];ca,n0,w0=p.pr.moments(e['prefix']);cb,n,w=p.pr.moments(a['result']);assert ca==cb
    anchor=ca.index(panel[0]);v0=n0[anchor];v=n[anchor];prior=dict(e['prefix']['candidates'])
    out=[]
    for m in panel[1:]:
        i=ca.index(m)
        out += [(w[i]/(n[i]+4)-w[anchor]/(v+4))/2,(w0[i]/(n0[i]+4)-w0[anchor]/(v0+4))/2,(n[i]-v)/a['cap'],(n0[i]-v0)/8,prior[m]-prior[panel[0]]]
    assert len(out)==10 and all(math.isfinite(t) and abs(t)<=1.000001 for t in out)
    return out
def panels(a,o):
    common=o['panel'];anchor=a['best'];alt=[m for m in common if m!=anchor]
    return dict(common=common,native=[anchor]+alt[:2])
def atlas():
    co=read('COHORT.json');assert co['identity']==identity()
    path=D/'ATLAS.jsonl.gz';assert not path.exists();n=0;nn=0;vis=0
    with p.Server(p.BIN) as s,gzip.open(path,'wt',compresslevel=3) as f:
        for r in co['roots']:
            pre=None
            for cap in CAPS:
                for arm in ARMS[r['id']%3:]+ARMS[:r['id']%3]:
                    a=s.request(r['history'],arm,cap);o=p.check(a,r,arm,cap)
                    if pre is None:pre=a['evidence']['prefix']
                    assert pre==a['evidence']['prefix']
                    pn=panels(a,o)
                    assert all(len(x)==len(set(x))==3 and set(x)<=set(r['legal']) for x in pn.values())
                    assert pn['native'][0]==a['best']
                    row=dict(root=r,cap=cap,arm=arm,response=a,readout=o,panels=pn,features={m:features(a,x) for m,x in pn.items()})
                    f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');f.flush();n+=1;nn+=a['NN_calls'];vis+=cap
            if r['id']%20==0:print('ATLAS',r['id'],n,flush=True)
    save('ATLAS_LOCK.json',dict(identity=identity(),cohort=sha(D/'COHORT.json'),sha256=sha(path),requests=n,NN=nn,visits=vis))
    print('ATLAS',n,nn,vis,flush=True)
def label(part):
    co=read('COHORT.json');lock=read('ATLAS_LOCK.json')
    assert sha(D/'ATLAS.jsonl.gz')==lock['sha256'] and sha(D/'COHORT.json')==lock['cohort']
    if part=='TEST':
        assert (D/'FROZEN.json').is_file()
        frozen=read('FROZEN.json')
        assert frozen['atlas']==lock['sha256'] and frozen['train']==sha(D/'TRAIN_CAL_LABELS.json')
        assert frozen['fit_code']==sha(D/'joint.py') and frozen['runner_code']==sha(D/'runner.py')
    rows=[];count=C.Counter()
    for r in co['roots']:
        if (r['group']=='TEST')!=(part=='TEST'):continue
        q=OLD.oracle(r);item=dict(root=r,status=q['status'])
        if q['status']=='EXACT':
            try:
                z,n=p.ref.exact_wdl(r['history'])
                truth={a:(1 if q['scores'][a]>0 else -1 if q['scores'][a]<0 else 0) for a in r['legal']}
                item.update(scores=q['scores'],independent=z,nodes=n,agreement=z==truth)
                if z!=truth:item['status']='MISMATCH'
            except RuntimeError as e:item.update(status='INDEPENDENT_CAP',reason=str(e),scores=q['scores'])
        else:item['oracle_result']=q
        count[item['status']]+=1;rows.append(item)
    save(part+'_LABELS.json',dict(cohort=lock['cohort'],atlas=lock['sha256'],counts=dict(count),rows=rows))
    print('LABELS',part,dict(count),flush=True)
def data(part):
    labels=read(part+'_LABELS.json');lock=read('ATLAS_LOCK.json')
    assert labels['atlas']==sha(D/'ATLAS.jsonl.gz')==lock['sha256']
    ll={r['root']['id']:r for r in labels['rows'] if r['status']=='EXACT'}
    rows=[]
    for t in gread('ATLAS.jsonl.gz'):
        r=t['root'];i=r['id']
        if i not in ll:continue
        z=ll[i];assert z['root']==r
        u={int(a):(v+1)/2 for a,v in z['independent'].items()}
        assert set(u)==set(r['legal'])
        for mode in MODES:
            panel=t['panels'][mode];a=t['response'];obs=t['features'][mode]
            assert obs==features(a,panel)
            y=tuple(int(2*(u[m]-u[panel[0]])) for m in panel[1:])
            assert y in STATES
            ca,n,w=p.pr.moments(a['result']);raw=max(panel,key=lambda m:w[ca.index(m)]/(n[ca.index(m)]+4))
            c0,pre_n,_=p.pr.moments(a['evidence']['prefix'])
            row=dict(id=i,group=r['group'],cap=t['cap'],arm=t['arm'],mode=mode,panel=panel,x=obs,y=list(y),utility=[u[m] for m in panel],full=max(u.values()),native_u=u[a['best']],raw_u=u[raw],native_move=a['best'],NN=a['NN_calls'],cpu_ms=a['cpu_ns']/1e6,wall_ms=a['wall_ns']/1e6,third_visited=pre_n[c0.index(t['readout']['panel'][2])]>0,native_in_common=a['best'] in t['readout']['panel'])
            rows.append(row)
    return rows
def key(row):return f"{row['mode']}:{row['arm']}:{row['cap']}"
def fit():
    assert not (D/'TEST_LABELS.json').exists()
    rr=data('TRAIN_CAL');models={}
    for k in sorted({key(r) for r in rr}):
        tr=[r for r in rr if key(r)==k and r['group']=='TRAIN'];cal=[r for r in rr if key(r)==k and r['group']=='CAL']
        assert tr and cal and len({r['id'] for r in tr})==len(tr)
        training=[dict(id=r['id'],x=r['x'],y=r['y']) for r in tr]
        scores={}
        for h in HVALUES:
            v=[metrics(predict(training,r['x'],h)[0],r['y']) for r in cal]
            scores[str(h)]=dict(log=S.mean(x['log'] for x in v),brier=S.mean(x['brier'] for x in v))
        # candidate order favors infinity then larger bandwidth on exact tie
        selected=min(HVALUES,key=lambda h:scores[str(h)]['log'])
        support={str(j):dict(positive=sum(r['y'][j]>0 for r in tr),negative=sum(r['y'][j]<0 for r in tr),zero=sum(r['y'][j]==0 for r in tr)) for j in range(2)}
        gate=all(min(v['positive'],v['negative'])>=4 for v in support.values())
        models[k]=dict(training=training,cal_n=len(cal),selected=selected,cal_scores=scores,support=support,exposure_gate=gate)
    lock=read('ATLAS_LOCK.json')
    save('FROZEN.json',dict(identity=identity(),atlas=lock['sha256'],train=sha(D/'TRAIN_CAL_LABELS.json'),fit_code=sha(D/'joint.py'),runner_code=sha(D/'runner.py'),models=models,selection='CAL mean joint log loss; no decision-score tuning; infinity first on ties'))
    print('FROZEN',[(k,m['selected'],m['exposure_gate'],m['support']) for k,m in models.items()],flush=True)
def evaluate():
    f=read('FROZEN.json');assert f['fit_code']==sha(D/'joint.py') and f['runner_code']==sha(D/'runner.py') and f['train']==sha(D/'TRAIN_CAL_LABELS.json')
    rows=data('TEST');out=[]
    for r in rows:
        m=f['models'][key(r)]
        tick=time.process_time_ns();q,ess=predict(m['training'],r['x'],m['selected']);i=decide(q);inference_us=(time.process_time_ns()-tick)/1000
        g,_=predict(m['training'],r['x'],None)
        i=decide(q);j=decide(g);v=r['utility'];d=metrics(q,r['y']);base=metrics(g,r['y'])
        mu=d['mean'];predrisk=d['risk']
        row=dict(r,key=key(r),selected=m['selected'],p=q,global_p=g,ESS=ess,inference_us=inference_us,choice=i,global_choice=j,chosen_move=r['panel'][i],gain=v[i]-r['native_u'],global_gain=v[j]-r['native_u'],regret=r['full']-v[i],global_regret=r['full']-v[j],native_regret=r['full']-r['native_u'],raw_regret=r['full']-r['raw_u'],omission=r['full']-max(v),rank=max(v)-v[i],anchor_gain=v[i]-v[0],predicted_anchor_gain=mu[i],predicted_panel_risk=predrisk,real_panel_risk=max(v)-v[i],log=d['log'],brier=d['brier'],gap_mse=d['gap_mse'],global_log=base['log'],global_brier=base['brier'],global_gap_mse=base['gap_mse'])
        assert row['regret']==row['omission']+row['rank']
        out.append(row)
    gwrite('TEST_PREDICTIONS.jsonl.gz',out)
    sums={}
    fields=('ESS','inference_us','gain','global_gain','regret','global_regret','native_regret','raw_regret','omission','rank','anchor_gain','predicted_anchor_gain','predicted_panel_risk','real_panel_risk','log','brier','gap_mse','global_log','global_brier','global_gap_mse','NN','cpu_ms','wall_ms')
    for k in sorted({r['key'] for r in out}):
        a=[r for r in out if r['key']==k];g=[r['gain'] for r in a];rng=random.Random(SEED)
        boot=sorted(S.mean(rng.choices(g,k=len(g))) for _ in range(4000))
        x=dict(n=len(a),selected=f['models'][k]['selected'],exposure_gate=f['models'][k]['exposure_gate'],positive=sum(z>0 for z in g),negative=sum(z<0 for z in g),switches=sum(r['chosen_move']!=r['native_move'] for r in a),anchor_switches=sum(r['choice']!=0 for r in a),gain_ci=[boot[99],boot[3899]],third_prefix_visited=sum(r['third_visited'] for r in a),native_in_common=sum(r['native_in_common'] for r in a),**{k0:S.mean(r[k0] for r in a) for k0 in fields})
        x['descriptive_admission']=x['exposure_gate'] and x['log']<x['global_log'] and x['gain_ci'][0]>0
        sums[k]=x
    # Factorial allocation comparison for common panel (identical action set across arms).
    contrast={}
    for cap in CAPS:
        for arm in ARMS[1:]:
            n={r['id']:r for r in out if r['key']==f'common:K16N:{cap}'}
            a={r['id']:r for r in out if r['key']==f'common:{arm}:{cap}'}
            assert n.keys()==a.keys() and all(n[i]['panel']==a[i]['panel'] for i in n)
            contrast[f'{arm}:{cap}']=dict(n=len(a),common_omission=S.mean(a[i]['omission'] for i in a),native_gain_vs_N=S.mean(n[i]['native_regret']-a[i]['native_regret'] for i in a),raw_gain_vs_N=S.mean(n[i]['raw_regret']-a[i]['raw_regret'] for i in a),fitted_gain_vs_N=S.mean(n[i]['regret']-a[i]['regret'] for i in a),cpu_delta_ms=S.mean(a[i]['cpu_ms']-n[i]['cpu_ms'] for i in a),NN_delta=S.mean(a[i]['NN']-n[i]['NN'] for i in a))
    save('SUMMARY.json',dict(frozen=sha(D/'FROZEN.json'),test=sha(D/'TEST_LABELS.json'),predictions=sha(D/'TEST_PREDICTIONS.jsonl.gz'),groups=sums,common_contrasts=contrast,primary='native:K16N:64'))
    for k,v in sums.items():print('TEST',k, {x:v[x] for x in ['n','selected','regret','native_regret','raw_regret','gain','positive','negative','log','global_log','brier','global_brier','gain_ci','exposure_gate','descriptive_admission']},flush=True)
    print('COMMON_CONTRAST',contrast,flush=True)
if __name__=='__main__':
    if sys.argv[1]=='label':label(sys.argv[2])
    else:globals()[sys.argv[1]]()
