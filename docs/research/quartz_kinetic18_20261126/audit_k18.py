"""Independent readback: no runner or joint estimator imports."""
import json,gzip,hashlib,pathlib,subprocess as SP,statistics as S,random,math,itertools,collections as C,importlib.util
D=pathlib.Path(__file__).resolve().parent
ROOT=D.parents[2]
P17=ROOT/'docs/research/quartz_kinetic17_20261126'
spec=importlib.util.spec_from_file_location('grid17',P17/'audit_k17.py')
grid=importlib.util.module_from_spec(spec);spec.loader.exec_module(grid)
ST=tuple((a,b) for a,b in itertools.product(range(-2,3),repeat=2) if max(0,a,b)-min(0,a,b)<=2)
assert len(ST)==19
def read(n):return json.loads((D/n).read_text())
def raw(n):
    with gzip.open(D/n,'rt') as f:
        for l in f:yield json.loads(l)
def sha(n):return hashlib.file_digest((D/n).open('rb'),'sha256').hexdigest()
def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12),(a,b)
def vec(a,b):
    assert len(a)==len(b)
    for x,y in zip(a,b):close(x,y)
def pred(tr,x,h):
    weights=[1 if h is None else math.exp(-sum((a-b)**2 for a,b in zip(x,r['x']))/(2*h*h)) for r in tr]
    v=[(1/19+sum(w for r,w in zip(tr,weights) if tuple(r['y'])==s))/(1+sum(weights)) for s in ST]
    return v,sum(weights)**2/sum(w*w for w in weights)
def scored(q,y):
    i=ST.index(tuple(y));m=[0]+[sum(p*s[j]/2 for p,s in zip(q,ST)) for j in range(2)]
    decision=max(range(3),key=lambda i:m[i])
    return dict(choice=decision,means=m,log=-math.log(q[i]),brier=sum((p-int(k==i))**2 for k,p in enumerate(q)),gap_mse=sum((m[k+1]-y[k]/2)**2 for k in range(2))/2,risk=sum(p*max(0,s[0]/2,s[1]/2) for p,s in zip(q,ST))-m[decision])
def features(x,y,panel,cap):
    ca=[m for m,_ in x['candidates']];prior=dict(x['candidates']);a={m:(0,0.) for m in ca};b=dict(a)
    for r in x['edges']:a[r['move']]=(r['n'],r['w'])
    for r in y['edges']:b[r['move']]=(r['n'],r['w'])
    a0=panel[0];v=[]
    for j in panel[1:]:
        n,w=b[j];n0,w0=b[a0];i,z=a[j];i0,z0=a[a0]
        v.extend([(w/(n+4)-w0/(n0+4))/2,(z/(i+4)-z0/(i0+4))/2,(n-n0)/cap,(i-i0)/8,prior[j]-prior[a0]])
    return v
def main():
    co=read('COHORT.json');lock=read('ATLAS_LOCK.json');fr=read('FROZEN.json');su=read('SUMMARY.json')
    assert sha('COHORT.json')==lock['cohort']
    assert sha('ATLAS.jsonl.gz')==lock['sha256']==fr['atlas']
    assert sha('TRAIN_CAL_LABELS.json')==fr['train']
    assert sha('FROZEN.json')==su['frozen'] and sha('TEST_LABELS.json')==su['test'] and sha('TEST_PREDICTIONS.jsonl.gz')==su['predictions']
    assert sha('joint.py')==fr['fit_code'] and sha('runner.py')==fr['runner_code']
    frozen_commit='2c2638ac14154c15fb2cc9f4ee3f520540d36818'
    def git(*args):return SP.check_output(['git','-C',str(ROOT),*args])
    rel=str(D.relative_to(ROOT))
    assert hashlib.sha256(git('show',f'{frozen_commit}:{rel}/FROZEN.json')).hexdigest()==sha('FROZEN.json')
    assert SP.run(['git','-C',str(ROOT),'cat-file','-e',f'{frozen_commit}:{rel}/TEST_LABELS.json'],capture_output=True).returncode!=0
    roots={r['id']:r for r in co['roots']};fam=set();boards=set()
    for p in [P17/'COHORT.json',P17/'quiet/COHORT.json']:
        old=json.loads(p.read_text());fam.update(j['family'] for j in old['jobs'])
        boards.update(grid.boardkey(grid.replay(r['history'])) for r in old['roots'])
    assert len(co['jobs'])==300
    for j in co['jobs']:
        i=j['id'];h=j['opening'];k=grid.canonical(h)
        assert k==j['family'] and k not in fam;fam.add(k)
        assert j['group']==('TRAIN' if i<120 else 'CAL' if i<200 else 'TEST')
        assert j['depth']==(24,26,28)[i%3]
        b=grid.replay(h);rng=random.Random(j['seed'])
        while len(h)<j['depth']:
            side=len(h)%2+1;opts=[]
            for c,col in enumerate(b):
                if len(col)==6:continue
                col.append(side)
                ok=not grid.won(b,side) and not any(grid.canwin(b,k) for k in (1,2))
                col.pop()
                if ok:opts.append(c)
            if not opts:break
            c=rng.choice(opts);b[c].append(side);h+=str(c+1)
        assert h==j['history'] and j['accepted']==(i in roots)
        if i in roots:
            r=roots[i];key=grid.boardkey(b)
            assert key not in boards and len(h)==j['depth'] and len(r['legal'])>=3
            boards.add(key)
            assert r['history']==h and r['group']==j['group'] and r['family']==k
            assert r['legal']==[c for c in range(7) if len(b[c])<6] and not any(grid.canwin(b,k) for k in (1,2))
    labels={}
    for part in ['TRAIN_CAL','TEST']:
        f=read(part+'_LABELS.json')
        assert f['cohort']==lock['cohort'] and f['atlas']==lock['sha256']
        for z in f['rows']:
            i=z['root']['id'];assert i not in labels and z['root']==roots[i] and z['status']=='EXACT' and z['agreement']
            assert (roots[i]['group']=='TEST')==(part=='TEST')
            u={int(a):v for a,v in z['independent'].items()}
            assert set(u)==set(roots[i]['legal']) and all(v==((z['scores'][a]>0)-(z['scores'][a]<0)) for a,v in u.items())
            labels[i]=z
    assert labels.keys()==roots.keys()
    cases={};prefix={};nn=vis=0;seen=set()
    for t in raw('ATLAS.jsonl.gz'):
        r=t['root'];i=r['id'];cap=t['cap'];arm=t['arm'];a=t['response'];e=a['evidence'];x=e['prefix'];y=a['result']
        key=(i,cap,arm);assert key not in seen;seen.add(key)
        assert r==roots[i] and a['history']==r['history'] and a['legal']==r['legal'] and a['visits']==cap
        if i in prefix:assert x==prefix[i]
        else:prefix[i]=x
        ca,n,w,edges=grid.moments(x);ca2,n2,w2,_=grid.moments(y)
        assert ca==ca2 and set(ca)==set(r['legal'])
        assert x['root_n']==8 and y['root_n']==cap and y['best']==a['best']
        pair=e['pair'];third=next(m for m in ca if m not in pair);common=pair+[third];native=[a['best']]+[m for m in common if m!=a['best']][:2]
        assert t['panels']=={'common':common,'native':native}
        target=[None,None,None] if arm=='K16N' else [pair[0],pair[1] if arm=='K16BAL' else third,None]
        oldn=n;oldw=w;step=8;q=(cap-16)//2
        for phase,delta,target_move in zip(e['phases'],(q,q,8),target):
            rr=phase['result'];cc,nr,wr,_=grid.moments(rr);step+=delta
            ds=[nr[m]-oldn[m] for m in ca]
            assert cc==ca and phase['delta']==ds and min(ds)>=0 and sum(ds)==delta and rr['root_n']==step and rr['root_hash']==x['root_hash']
            assert phase['target']==target_move and (target_move is None or ds[ca.index(target_move)]==delta)
            assert all(abs(wr[m]-oldw[m])<=ds[j]+1e-6 for j,m in enumerate(ca))
            oldn,oldw=nr,wr
        assert rr==y and y['evaluator_calls']==a['NN_calls']
        u={int(m):(v+1)/2 for m,v in labels[i]['independent'].items()}
        nn+=a['NN_calls'];vis+=cap
        for mode,panel in t['panels'].items():
            obs=features(x,y,panel,cap);vec(obs,t['features'][mode])
            v=[u[m] for m in panel];yy=[int(2*(t-v[0])) for t in v[1:]]
            k=f'{mode}:{arm}:{cap}'
            cases[(i,k)]=dict(id=i,group=r['group'],x=obs,y=yy,utility=v,full=max(u.values()),native_u=u[a['best']],raw_u=u[max(panel,key=lambda m:w2[m]/(n2[m]+4))],panel=panel,native_move=a['best'],visited=n[third]>0)
    assert len(seen)==len(roots)*6==lock['requests'] and nn==lock['NN'] and vis==lock['visits']
    calchecks=0
    for k,m in fr['models'].items():
        tr=[x for (i,kk),x in cases.items() if kk==k and x['group']=='TRAIN']
        cal=[x for (i,kk),x in cases.items() if kk==k and x['group']=='CAL']
        assert len(tr)==len(m['training']) and len(cal)==m['cal_n']
        for a,b in zip(tr,m['training']):
            assert a['id']==b['id'] and a['y']==b['y'];vec(a['x'],b['x'])
        for h in [None,1.,.5,.25]:
            v=[scored(pred(tr,r['x'],h)[0],r['y']) for r in cal]
            for f0 in ['log','brier']:close(S.mean(z[f0] for z in v),m['cal_scores'][str(h)][f0])
            calchecks+=len(cal)
        assert m['selected']==min([None,1.,.5,.25],key=lambda h:m['cal_scores'][str(h)]['log'])
        for j in range(2):
            x=dict(positive=sum(r['y'][j]>0 for r in tr),negative=sum(r['y'][j]<0 for r in tr),zero=sum(r['y'][j]==0 for r in tr))
            assert x==m['support'][str(j)]
        assert m['exposure_gate']==all(min(v['positive'],v['negative'])>=4 for v in m['support'].values())
    out=[];test_seen=set()
    for r in raw('TEST_PREDICTIONS.jsonl.gz'):
        i,k=r['id'],r['key'];assert (i,k) not in test_seen;test_seen.add((i,k));c=cases[i,k];assert c['group']=='TEST'
        assert r['y']==c['y'] and r['utility']==c['utility'] and r['panel']==c['panel']
        vec(r['x'],c['x'])
        m=fr['models'][k];q,ess=pred(m['training'],c['x'],m['selected']);g,_=pred(m['training'],c['x'],None)
        vec(q,r['p']);vec(g,r['global_p']);close(ess,r['ESS'])
        sc=scored(q,c['y']);sg=scored(g,c['y']);v=c['utility'];d=sc['choice'];j=sg['choice']
        assert r['choice']==d and r['global_choice']==j and r['chosen_move']==c['panel'][d]
        for f0 in ['log','brier','gap_mse']:close(r[f0],sc[f0]);close(r['global_'+f0],sg[f0])
        for key,z in dict(regret=c['full']-v[d],native_regret=c['full']-c['native_u'],raw_regret=c['full']-c['raw_u'],global_regret=c['full']-v[j],gain=v[d]-c['native_u'],global_gain=v[j]-c['native_u'],omission=c['full']-max(v),rank=max(v)-v[d],predicted_panel_risk=sc['risk'],real_panel_risk=max(v)-v[d],predicted_anchor_gain=sc['means'][d],anchor_gain=v[d]-v[0]).items():close(r[key],z)
        out.append(r)
    assert test_seen=={(i,k) for (i,k),x in cases.items() if x['group']=='TEST'}
    diag={}
    for k,v in su['groups'].items():
        rr=[r for r in out if r['key']==k]
        assert len(rr)==v['n']
        for z in ['ESS','inference_us','gain','global_gain','regret','global_regret','native_regret','raw_regret','omission','rank','anchor_gain','predicted_anchor_gain','predicted_panel_risk','real_panel_risk','log','brier','gap_mse','global_log','global_brier','global_gap_mse','NN','cpu_ms','wall_ms']:close(S.mean(r[z] for r in rr),v[z])
        assert v['positive']==sum(r['gain']>0 for r in rr) and v['negative']==sum(r['gain']<0 for r in rr)
        switched=[r for r in rr if r['chosen_move']!=r['native_move']]
        assert len(switched)==v['switches']
        diag[k]=dict(potential_panel_improvements=sum(max(r['utility'])>r['native_u'] for r in rr),potential_panel_gain=S.mean(max(0,max(r['utility'])-r['native_u']) for r in rr),switches=len(switched),switch_improved=sum(r['gain']>0 for r in switched),switch_harmed=sum(r['gain']<0 for r in switched),switch_equal=sum(r['gain']==0 for r in switched),visited_strata={str(b):dict(n=len(a),mean_gain=S.mean(r['gain'] for r in a) if a else None) for b in (True,False) for a in [[r for r in rr if r['third_visited']==b]]})
    return dict(roots=len(roots),attempts=len(co['jobs']),requests=len(seen),exact_actions=sum(len(z['independent']) for z in labels.values()),nodes=sum(z['nodes'] for z in labels.values()),train_cal_model_checks=calchecks,test_predictions=len(out),models=len(fr['models']),aggregate_groups=len(su['groups']),freeze_before_test=True,diagnostics=diag,scope='independent grid replay and K17 family/board exclusions, WDL sign readback, all native moments/forced phase counts, features, kernel weights, CAL selection, TRAIN support, heldout predictions and aggregate scores; no re-solve or full internal MCTS tree trace')
if __name__=='__main__':print(json.dumps(main(),indent=2,sort_keys=True))
