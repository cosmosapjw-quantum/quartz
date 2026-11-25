"""Independent readback: no transition or runner imports; independent grid rules."""
import pathlib,sys,json,gzip,hashlib,subprocess as SP,statistics as S,random,math,itertools,collections as C,importlib.util
D=pathlib.Path(__file__).resolve().parent
ROOT=D.parents[2]
P18=ROOT/'docs/research/quartz_kinetic18_20261126'
P17=ROOT/'docs/research/quartz_kinetic17_20261126'
def module(path,name):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
grid=module(P17/'audit_k17.py','grid17')
gref=module(ROOT/'docs/research/quartz_fouraxis06_20261125/rules.py','gref')
ST=tuple((a,b) for a,b in itertools.product(range(-2,3),repeat=2) if max(0,a,b)-min(0,a,b)<=2)
VV=[(0,a/2,b/2) for a,b in ST]
CH=tuple(f'{a}:{n}' for n in (32,64) for a in ('K16N','K16BAL','K17EXP'))
H=(None,1.,.5,.25)
PR=(.01,0.,.002,.02)
def read(p):return json.loads(p.read_text())
def raw(p):
    with gzip.open(p,'rt') as f:
        for l in f:yield json.loads(l)
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12),(a,b)
def vec(a,b):
    assert len(a)==len(b)
    for x,y in zip(a,b):close(x,y)
def git(*a):return SP.check_output(['git','-C',str(ROOT),*a])
def feature(x,panel):
    ca,n,w,_=grid.moments(x);p=dict(x['candidates']);a=panel[0];out=[]
    for b in panel[1:]:
        out.extend([(w[b]/(n[b]+4)-w[a]/(n[a]+4))/2,(n[b]-n[a])/8,p[b]-p[a]])
    return out
def extract(rows,labels,group=None,verify_native=False):
    ll={z['root']['id']:z for z in labels if z['status']=='EXACT'}
    out={};count=vis=nn=0
    for t in rows:
        r=t['root'];i=r['id']
        if group is not None and r['group'] not in group:continue
        if i not in ll:continue
        l=ll[i];assert l['root']==r and l['agreement']
        u={int(a):(k+1)/2 for a,k in l['independent'].items()};assert set(u)==set(r['legal'])
        assert all(v==((l['scores'][a]>0)-(l['scores'][a]<0)+1)/2 for a,v in u.items())
        a=t['response'];e=a['evidence'];x=e['prefix'];y=a['result'];c=f"{t['arm']}:{t['cap']}"
        panel=t['panels']['common'] if 'panels' in t else t['panel']
        ca,n,w,ee=grid.moments(x);cb,nf,wf,_=grid.moments(y)
        assert ca==cb and x['root_n']==8 and y['root_n']==t['cap'] and set(ca)==set(r['legal'])
        assert panel[0]==x['best'] and len(set(panel))==3
        if verify_native:
            q=(t['cap']-16)//2;target=[None,None,None] if t['arm']=='K16N' else [panel[0],panel[1] if t['arm']=='K16BAL' else panel[2],None]
            prev=n;pw=w;step=8
            for phase,k,m in zip(e['phases'],(q,q,8),target):
                rr=phase['result'];cc,nr,wr,_=grid.moments(rr);step+=k;d=[nr[z]-prev[z] for z in ca]
                assert cc==ca and phase['delta']==d and sum(d)==k and min(d)>=0 and rr['root_n']==step and phase['target']==m
                assert m is None or d[ca.index(m)]==k
                assert all(abs(wr[z]-pw[z])<=d[j]+1e-6 for j,z in enumerate(ca))
                prev,pw=nr,wr
            assert rr==y and y['best']==a['best'] and y['evaluator_calls']==a['NN_calls']
        zz=max(range(3),key=lambda j:wf[panel[j]]/(nf[panel[j]]+4));xx=feature(x,panel);vv=[u[m] for m in panel];dy=[int(2*(v-vv[0])) for v in vv[1:]]
        if i not in out:out[i]=dict(id=i,group=r['group'],x=xx,y=dy,panel=panel,utility=vv,full=max(u.values()),all_u=u,obs={},pre=x,prefix_NN=x['evaluator_calls'])
        z=out[i];assert z['pre']==x and z['panel']==panel and z['x']==xx and z['y']==dy and c not in z['obs']
        z['obs'][c]=dict(z=zz,cost=sum(ph['step_us'] for ph in e['phases'])/1000,phase_visits=t['cap']-8,phase_NN=a['NN_calls']-x['evaluator_calls'],native=a['best'])
        count+=1;vis+=t['cap'];nn+=a['NN_calls']
    assert set(out)==set(ll) and all(set(z['obs'])==set(CH) for z in out.values())
    return out,count,vis,nn
def estimate_independent(tr,x,h):
    weights=[1 if h is None else math.exp(-sum((a-b)**2 for a,b in zip(x,z['x']))/(2*h*h)) for z in tr]
    total=sum(weights);den=1+total
    p=[(1/19+sum(w for w,z in zip(weights,tr) if tuple(z['y'])==s))/den for s in ST]
    mu=[sum(w*v[i] for w,v in zip(p,VV)) for i in range(3)]
    V=max(mu);H0=sum(w*max(v) for w,v in zip(p,VV));q={}
    for c in CH:
        J=[[(1/57+sum(w for w,r in zip(weights,tr) if tuple(r['y'])==s and r['obs'][c]['z']==k))/den for k in range(3)] for s in ST]
        vec([sum(r) for r in J],p)
        M=[[sum(r[k]*v[i] for r,v in zip(J,VV)) for i in range(3)] for k in range(3)]
        vec([sum(r[i] for r in M) for i in range(3)],mu)
        delta=sum(max(r) for r in M)-V
        assert -1e-12<=delta<=H0-V+1e-12
        q[c]=dict(J=J,pi=[sum(r[k] for r in J) for k in range(3)],decisions=[max(range(3),key=r.__getitem__) for r in M],gain=max(0,delta),risk=H0-V-delta,cost=sum(w*z['obs'][c]['cost'] for w,z in zip(weights,tr))/total)
    return p,mu,max(range(3),key=mu.__getitem__),H0-V,q
def c4():
    co=read(D/'COHORT.json');fr=read(D/'FROZEN.json');lk=read(D/'ATLAS_LOCK.json');la=read(D/'TEST_LABELS.json');su=read(D/'SUMMARY.json')
    assert sha(D/'COHORT.json')==fr['cohort']==lk['cohort']
    assert sha(D/'FROZEN.json')==lk['frozen']==la['frozen']==su['frozen']
    assert sha(D/'ATLAS.jsonl.gz')==lk['sha256']==la['atlas']
    assert sha(D/'TEST_LABELS.json')==su['test'] and sha(D/'PREDICTIONS.jsonl.gz')==su['predictions']
    assert sha(D/'transition.py')==fr['fit_code'] and sha(D/'run_k19.py')==fr['runner_code']
    fc='03cfb18e6705b534d1c34c402a88ee89ae1f00ee';rel=str(D.relative_to(ROOT))
    assert hashlib.sha256(git('show',fc+':'+rel+'/FROZEN.json')).hexdigest()==sha(D/'FROZEN.json')
    for n in ('ATLAS.jsonl.gz','TEST_LABELS.json'):
        assert SP.run(['git','-C',str(ROOT),'cat-file','-e',fc+':'+rel+'/'+n],capture_output=True).returncode!=0
    for x in fr['sources']:
        assert sha(ROOT/x['path'])==x['sha256']==hashlib.sha256(git('show',x['commit']+':'+x['path'])).hexdigest()
    fam=set();boards=set()
    for path in [P17/'COHORT.json',P17/'quiet/COHORT.json',P18/'COHORT.json']:
        a=read(path);fam.update(z['family'] for z in a['jobs']);boards.update(grid.boardkey(grid.replay(r['history'])) for r in a['roots'])
    roots={r['id']:r for r in co['roots']}
    for j in co['jobs']:
        h=j['opening'];k=grid.canonical(h);assert k==j['family'] and k not in fam;fam.add(k)
        b=grid.replay(h);rng=random.Random(j['seed'])
        while len(h)<j['depth']:
            side=len(h)%2+1;opts=[]
            for c,col in enumerate(b):
                if len(col)==6:continue
                col.append(side);ok=not grid.won(b,side) and not any(grid.canwin(b,t) for t in (1,2));col.pop()
                if ok:opts.append(c)
            if not opts:break
            c=rng.choice(opts);b[c].append(side);h+=str(c+1)
        assert h==j['history'] and j['accepted']==(j['id'] in roots)
        if j['accepted']:
            r=roots[j['id']];key=grid.boardkey(b);assert key not in boards;boards.add(key)
            assert r['history']==h and r['family']==k and len(h)==j['depth'] and r['legal']==[c for c in range(7) if len(b[c])<6] and not any(grid.canwin(b,t) for t in (1,2))
    old,_,_,_=extract(raw(P18/'ATLAS.jsonl.gz'),read(P18/'TRAIN_CAL_LABELS.json')['rows'],('TRAIN','CAL'))
    tr=[x for x in old.values() if x['group']=='TRAIN'];cal=[x for x in old.values() if x['group']=='CAL']
    assert len(tr)==110 and len(cal)==74 and len(fr['training'])==len(tr)
    for a,b in zip(tr,fr['training']):
        assert a['id']==b['id'] and a['y']==b['y'];vec(a['x'],b['x'])
        for c in CH:
            for k,v in a['obs'][c].items():close(v,b['obs'][c][k])
    scores={}
    for h in H:
        ll=[]
        for r in cal:
            *_,q=estimate_independent(fr['training'],r['x'],h)
            ll.extend(-math.log(q[c]['J'][ST.index(tuple(r['y']))][r['obs'][c]['z']]) for c in CH)
        scores[str(h)]=S.mean(ll);close(scores[str(h)],fr['scores'][str(h)])
    assert fr['selected']==min(H,key=lambda h:scores[str(h)])
    test,n,vis,nn=extract(raw(D/'ATLAS.jsonl.gz'),la['rows'],verify_native=True)
    assert n==lk['requests'] and vis==lk['visits'] and nn==lk['NN'] and len(test)==len(roots)
    preds=list(raw(D/'PREDICTIONS.jsonl.gz'));assert len(preds)==len(test);ss=set()
    for r in preds:
        i=r['id'];assert i not in ss;ss.add(i);z=test[i]
        assert r['y']==z['y'] and r['panel']==z['panel'] and r['utility']==z['utility']
        vec(r['x'],z['x'])
        p,mu,d,R,q=estimate_independent(fr['training'],z['x'],fr['selected']);vec(p,r['p']);vec(mu,r['mu']);close(R,r['stop_risk'])
        assert d==r['stop_decision'] and r['stop']==z['utility'][d]
        for c in CH:
            a=r['channels'][c];b=q[c];o=z['obs'][c]
            for x,y in zip(a['J'],b['J']):vec(x,y)
            vec(a['pi'],b['pi'])
            close(a['pred_gain'],b['gain']);close(a['pred_cost'],b['cost']);close(a['pred_risk'],b['risk'])
            assert a['z']==o['z'] and a['decision']==b['decisions'][o['z']]
            close(a['utility'],z['utility'][a['decision']]);close(a['native_u'],z['all_u'][o['native']])
            close(a['gain'],a['utility']-r['stop']);close(a['cost'],o['cost'])
            close(a['log'],-math.log(b['J'][ST.index(tuple(z['y']))][o['z']]))
        for price in PR:
            a=r['policy'][str(price)];ch='STOP';net=0
            for c in CH:
                v=q[c]['gain']-price*q[c]['cost']
                if v>net+1e-15:ch=c;net=v
            assert a['channel']==ch;close(a['pred_net'],net)
            u0=r['stop'] if ch=='STOP' else r['channels'][ch]['utility'];cost=0 if ch=='STOP' else z['obs'][ch]['cost']
            close(a['utility'],u0);close(a['actual_net'],u0-r['stop']-price*cost);close(a['regret'],z['full']-u0)
    assert ss==set(test)
    for c in CH:
        q=su['fixed'][c];aa=[r['channels'][c] for r in preds]
        close(q['regret'],S.mean(r['full']-a['utility'] for r,a in zip(preds,aa)))
        close(q['native_regret'],S.mean(r['full']-a['native_u'] for r,a in zip(preds,aa)))
        for k,v in [('mean_gain','gain'),('pred_gain','pred_gain'),('cost_ms','cost'),('pred_cost_ms','pred_cost')]:close(q[k],S.mean(a[v] for a in aa))
    for price in PR:
        q=su['policies'][str(price)];aa=[r['policy'][str(price)] for r in preds]
        assert q['choice_counts']==dict(C.Counter(a['channel'] for a in aa))
        close(q['regret'],S.mean(a['regret'] for a in aa));close(q['gain_vs_N64'],S.mean(a['utility']-r['channels']['K16N:64']['native_u'] for r,a in zip(preds,aa)))
        close(q['mean_net'],S.mean(a['actual_net'] for a in aa))
    # Secondary fixed BAL comparison: descriptive paired result only.
    v=[r['channels']['K16BAL:64']['utility']-r['channels']['K16N:64']['native_u'] for r in preds]
    rng=random.Random(2026112619);bb=sorted(S.mean(rng.choices(v,k=len(v))) for _ in range(4000))
    return dict(attempts=len(co['jobs']),roots=len(test),exact_actions=sum(len(r['independent']) for r in la['rows']),independent_nodes=sum(r['nodes'] for r in la['rows']),native_queries=n,visits=vis,NN=nn,training=len(tr),cal=len(cal),cal_joint_checks=len(cal)*len(H)*len(CH),test_joint_checks=len(test)*len(CH),policy_checks=len(test)*len(PR),freeze_before_atlas_and_labels=True,secondary_BAL64_gain_vs_N64=S.mean(v),secondary_BAL64_boot=[bb[99],bb[3899]],secondary_BAL64_positive=sum(x>0 for x in v),secondary_BAL64_negative=sum(x<0 for x in v))
def gomoku():
    plan=read(D/'G_PLAN.json');co=read(D/'G_COHORT.json');pan=read(D/'G_PANEL.json');lk=read(D/'G_LABELS_LOCK.json');su=read(D/'G_SUMMARY.json')
    assert sha(D/'G_PLAN.json')==co['plan'] and sha(D/'G_SOURCE.jsonl.gz')==co['source']
    assert sha(D/'G_COHORT.json')==pan['cohort'] and sha(D/'G_ATLAS.jsonl.gz')==pan['atlas']
    assert sha(D/'G_PANEL.json')==lk['panel'] and sha(D/'G_LABELS.jsonl.gz')==lk['sha256']
    roots={r['id']:r for r in co['roots']};fam=set();src_plies=src_nn=0
    for z in raw(D/'G_SOURCE.jsonl.gz'):
        j=z['job'];b,side,w=gref.replay(j['opening'],'freestyle');assert w is None
        k=hashlib.sha256(gref.orbit_board(b,side,'freestyle')).hexdigest();assert k==j['family'] and k not in fam;fam.add(k)
        h=list(j['opening'])
        for row in z['rows']:
            b,side,w=gref.replay(h,'freestyle');assert w is None and not b[row['move']]
            h.append(row['move']);src_plies+=1;src_nn+=row['NN']
        assert h==z['history']
        if z['selected']:
            r=z['selected'];assert roots[r['id']]==r and h[:len(r['history'])]==r['history']
            b,s,w=gref.replay(r['history'],'freestyle');proof=gref.proof_reference(b,s,'freestyle');assert w is None and s==r['player'] and not proof['own'] and not proof['opponent']
    assert src_plies==co['source_plies'] and src_nn==co['source_NN']
    atlas={};hidden=0;query=0
    for z in raw(D/'G_ATLAS.jsonl.gz'):
        i=z['root']['id'];a=z['response'];arm=z['arm'];assert z['root']==roots[i]
        b,s,w=gref.replay(roots[i]['history'],'freestyle');assert w is None and a['board']==b and a['player']==s and a['proof']==gref.proof_reference(b,s,'freestyle')
        ph=a['decision']['phases'];assert [x['visits'] for x in ph]==[8,12,16,24,32]
        pre=ph[0]['result'];ca,n,w0,edges=grid.moments(pre);cc,n1,w1,e1=grid.moments(ph[1]['result']);assert ca==cc
        t=a['decision']['target']
        if arm=='DSX' and t:
            m=t['move'];assert m not in edges and m in e1 and n[m]==0 and n1[m]==4 and t['materialized'] is False;hidden+=1
        if arm=='DSF' and t:
            m=t['move'];assert m in edges and n1[m]-n[m]==4 and t['materialized'] is True
        for t0 in ph:grid.moments(t0['result'])
        assert ph[-1]['result']==a['result'] and a['result']['best']==a['best']
        atlas[i,arm]=a;query+=1
    assert hidden==pan['hidden_materialized'] and query==pan['requests']
    U={};n=nn=plies=0
    for z in raw(D/'G_LABELS.jsonl.gz'):
        r=roots[z['id']];h=r['history']+[z['action']]
        for row in z['rows']:
            b,s,w=gref.replay(h,'freestyle');assert w is None and not b[row['move']]
            h.append(row['move']);plies+=1;nn+=row['NN']
        b,s,w=gref.replay(h,'freestyle');assert h==z['history'] and w is not None and w==z['winner']
        close(z['utility'],(1+r['player']*w)/2)
        key=(z['id'],z['action']);d=U.setdefault(key,{})
        assert z['arm'] not in d;d[z['arm']]=z['utility'];n+=1
    assert n==lk['games'] and nn==lk['NN'] and plies==lk['plies']
    assert all(set(v)=={'STANDARD_PW','STANDARD_FULL'} for v in U.values())
    V={k:S.mean(v.values()) for k,v in U.items()}
    assert len(V)==su['action_pairs']
    for r in pan['panels']:
        i=r['root']['id'];p0=atlas[i,'DS0']['decision']['phases'][0]['result']
        assert all(atlas[i,c]['decision']['phases'][0]['result']==p0 for c in ('DS0','DSF','DSX'))
        for c in ('DS0','DSF','DSX'):
            a=atlas[i,c];ca,nf,wf,_=grid.moments(a['result'])
            assert r['native'][c]==a['best'] and r['qbest'][c]==max(r['prefix_panel'],key=lambda m:wf[m]/(nf[m]+4))
        assert all((i,m) in V for m in r['moves'])
    for c in ('DS0','DSF','DSX'):
        for kind,field in [('native','native'),('qbest','q')]:
            val=[V[r['root']['id'],r[kind][c]] for r in pan['panels']]
            bas=[V[r['root']['id'],r['native']['DS0']] for r in pan['panels']]
            regret=[max(V[r['root']['id'],m] for m in r['moves'])-v for r,v in zip(pan['panels'],val)]
            close(S.mean(val),su[field][c]['utility']);close(S.mean(regret),su[field][c]['panel_regret']);close(S.mean(v-b for v,b in zip(val,bas)),su[field][c]['gain_vs_DS0'])
    hd=[V[r['root']['id'],r['targets']['DSX']]-V[r['root']['id'],r['anchor']] for r in pan['panels'] if r['targets']['DSX'] is not None]
    assert su['hidden_positive']==sum(x>0 for x in hd) and su['hidden_negative']==sum(x<0 for x in hd) and su['hidden_equal']==sum(x==0 for x in hd)
    close(S.mean(hd),su['hidden_mean_gain'])
    return dict(source_families=len(fam),roots=len(roots),source_plies=src_plies,source_NN=src_nn,native_queries=query,hidden_materialized=hidden,continuations=n,continuation_plies=plies,continuation_NN=nn,action_pairs=len(V),policy_disagreements=sum(len(set(x.values()))>1 for x in U.values()))
if __name__=='__main__':print(json.dumps(dict(c4=c4(),gomoku=gomoku(),scope='independent C4/Gomoku rule replay, source split/hash checks, native phase counts, independent joint fit/selection/readout, compact continuation replay; no repeat exact oracle or full internal MCTS simulation'),sort_keys=True,indent=2))
