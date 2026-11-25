"""Independent grid arithmetic, fit, decision and resource audit (no acquire import)."""
import pathlib,json,gzip,hashlib,subprocess as SP,statistics as S,math,random,itertools,collections as C,importlib.util
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2];R=ROOT/'docs/research'
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
grid=module(R/'quartz_kinetic17_20261126/audit_k17.py','grid17')
gref=module(R/'quartz_fouraxis06_20261125/rules.py','gref')
CH=tuple(f'{a}:{n}' for n in (32,64) for a in ('K16N','K16BAL','K17EXP'))
PR=(.002,0.,.001,.004)
def read(n):return json.loads((D/n).read_text())
def raw(p):
    with gzip.open(p,'rt') as f:
        for l in f:yield json.loads(l)
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def git(*a):return SP.check_output(['git','-C',str(ROOT),*a])
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12),(a,b)
def lock(x):
    assert hashlib.sha256(git('show',x['commit']+':'+x['path'])).hexdigest()==x['sha256']
    if (ROOT/x['path']).exists():assert sha(ROOT/x['path'])==x['sha256']
def interval(h):
    b=grid.replay(h);side=len(h)%2+1;rows=[]
    for c in range(7):
        if len(b[c])==6:continue
        b[c].append(side)
        if grid.won(b,side):l=u=1.
        elif sum(map(len,b))==42:l=u=.5
        elif grid.canwin(b,3-side):l=u=0.
        else:l,u=0.,1.
        b[c].pop();rows.append([c,l,u])
    return rows
def certified(pr):
    return next((a for a,l,u in pr if l>=max(v for _,_,v in pr)),None)
def sc(s,panel):
    _,n,w,_=grid.moments(s);return [w[a]/(n[a]+4) for a in panel]
def bin0(s,panel):
    _,n,w,_=grid.moments(s);a,b,c=panel
    return 2*int(n[a]>=6)+int(w[a]/(n[a]+4)-max(w[b]/(n[b]+4),w[c]/(n[c]+4))>=.15)
def obs(s,panel):
    x=sc(s,panel);j=max(range(3),key=lambda k:x[k]);return j,int(x[j]-x[0]>=.15)
def panel0(s):
    ca,n,w,e=grid.moments(s);a=max(ca,key=lambda m:(n[m],ca.index(m)));b=max((m for m in ca if m!=a),key=lambda m:(n[m],ca.index(m))) if any(n[m]>0 for m in ca if m!=a) else next(m for m in ca if m!=a)
    c=next(m for m in ca if m!=a and m!=b)
    return [a,b,c]
def pick(m,b,price):
    v=[g-price*c for g,c in zip(m['gains'][b],m['costs'][b])]
    j=max(range(6),key=lambda j:v[j])
    return j if v[j]>1e-15 else None
def readout(panel,pr,c,final,table):
    a=certified(pr)
    if a is not None:return a
    if c%3==0:return final['best']
    j,h=obs(final,panel);p={a:(l,u) for a,l,u in pr}
    return panel[j] if j and table[c][j][h] and p[panel[j]][1]>p[panel[0]][0] else panel[0]
def fit_audit(f):
    p=R/'quartz_kinetic18_20261126'
    labs={r['root']['id']:r for r in json.loads((p/'TRAIN_CAL_LABELS.json').read_text())['rows']}
    rr={}
    for t in raw(p/'ATLAS.jsonl.gz'):
        r=t['root'];i=r['id']
        if r['group']=='TEST':continue
        a=t['response'];s=a['evidence']['prefix'];pn=panel0(s);assert pn==t['panels']['common']
        u={int(k):(v+1)/2 for k,v in labs[i]['independent'].items()}
        if i not in rr:rr[i]=dict(id=i,group=r['group'],panel=pn,u=u,pr=interval(r['history']),bin=bin0(s,pn),arms={})
        row=rr[i];c=CH.index(t['arm']+':'+str(t['cap']));assert row['panel']==pn and row['bin']==bin0(s,pn) and c not in row['arms']
        row['arms'][c]=(a['result'],a['NN_calls']-s['evaluator_calls'])
    tr=[r for i,r in sorted(rr.items()) if r['group']=='TRAIN'];cal=[r for i,r in sorted(rr.items()) if r['group']=='CAL']
    A=tr[::2];B=tr[1::2];assert [x['id'] for x in A]==f['split_A'] and [x['id'] for x in B]==f['split_B'] and [x['id'] for x in cal]==f['cal']
    table=[[[False]*2 for _ in range(3)] for _ in CH]
    for c in range(6):
        if c%3==0:continue
        for j in (1,2):
            lst=[(r['u'][r['panel'][j]]-r['u'][r['panel'][0]],obs(r['arms'][c][0],r['panel'])[1]) for r in A if obs(r['arms'][c][0],r['panel'])[0]==j]
            for b in range(2):
                g=[x for x,h in lst if h==b];mu=(sum(g)+4*(S.mean(x for x,h in lst) if lst else 0))/(len(g)+4)
                table[c][j][b]=bool(g and mu>0)
    assert table==f['controller']['switch']
    def target(r,c):
        s,n=r['arms'][c];a=readout(r['panel'],r['pr'],c,s,table)
        return r['u'][a]-r['u'][r['panel'][0]],n
    for mode in ('global','stratified'):
        m=f['fit'][mode]['controller'];h=f['fit'][mode]['hist']
        for b in range(4):
            for c in range(6):
                v=[target(r,c) for r in B];z=[target(r,c) for r in B if r['bin']==b]
                p=[sum(g==k/2 for g,n in v)/len(v) for k in range(-2,3)]
                cost=S.mean(n for g,n in v)
                if mode=='stratified':
                    p=[(sum(g==k/2 for g,n in z)+8*p[k+2])/(len(z)+8) for k in range(-2,3)]
                    cost=(sum(n for g,n in z)+8*cost)/(len(z)+8)
                for x,y in zip(p,h[b][c]['prob']):close(x,y)
                close(sum((k/2)*p[k+2] for k in range(-2,3)),m['gains'][b][c]);close(cost,m['costs'][b][c])
        mse=S.mean((target(r,c)[0]-m['gains'][r['bin']][c])**2 for r in cal for c in range(6))
        close(mse,f['cal_mse'][mode])
    assert f['selected']==min(('global','stratified'),key=lambda k:f['cal_mse'][k])
    return len(A),len(B),len(cal)*6*2
def source_audit(plan,co):
    for x in list(plan['source_locks'].values())+plan['history_locks']:lock(x)
    fam=set();boards=set()
    for x in plan['history_locks']:
        j=json.loads(git('show',x['commit']+':'+x['path']))
        for r in j['roots']:
            if r.get('domain','C4')!='C4':continue
            h=r['history'];boards.add(grid.boardkey(grid.replay(h)));fam.add(grid.canonical(h[:8]))
        for z in j.get('jobs',j.get('c4_attempts',[])):
            if len(z['history'])>=8:fam.add(grid.canonical(z['history'][:8]))
    assert len(fam)==plan['historical_families']
    roots={r['id']:r for r in co['roots']};q=nn=plies=0;seen=set();late_discrepancy=0
    for z in raw(D/'SOURCE.jsonl.gz'):
        j=z['job'];i=j['id'];assert j==plan['jobs'][i] and i not in seen;seen.add(i)
        h=j['opening'];b=grid.replay(h);k=grid.canonical(h);assert k==j['family'] and k not in fam;fam.add(k);rng=random.Random(j['seed'])
        eligible=[]
        for a in z['steps']:
            side=len(h)%2+1;assert not grid.won(b,1) and not grid.won(b,2)
            legal=[m for m in range(7) if len(b[m])<6]
            def wins(s):
                out=[]
                for m in legal:
                    b[m].append(s)
                    if grid.won(b,s):out.append(m)
                    b[m].pop()
                return out
            own,opp=wins(side),wins(3-side)
            if len(h) in (24,26,28,30) and len(legal)>=3 and not own and not opp and grid.boardkey(b) not in boards:eligible.append(h)
            if own:assert a['kind']=='WIN' and a['move']==own[0]
            elif len(opp)==1:assert a['kind']=='BLOCK' and a['move']==opp[0]
            elif len(legal)==1:assert a['kind']=='ONLY' and a['move']==legal[0]
            elif rng.random()<.75:
                assert a['kind']=='N32' and a['move']==a['native_best'] and a['NN']>=1;q+=1
            else:assert a['kind']=='UNIFORM' and a['move']==rng.choice(legal)
            m=a['move'];assert m in legal;b[m].append(side);h+=str(m+1);plies+=1;nn+=a['NN']
        assert h==z['history'] and (len(h)==30 or grid.won(b,1) or grid.won(b,2))
        if len(h)==30 and not grid.won(b,1) and not grid.won(b,2):
            legal=[m for m in range(7) if len(b[m])<6]
            if len(legal)>=3 and not grid.canwin(b,1) and not grid.canwin(b,2) and grid.boardkey(b) not in boards:eligible.append(h)
        if z['selected']:
            r=z['selected'];assert roots[i]==r and r['history'] in eligible
            if r['history']!=eligible[-1]:late_discrepancy+=1
            key=grid.boardkey(grid.replay(r['history']));assert key not in boards;boards.add(key)
        else:assert i not in roots
    assert len(seen)==len(plan['jobs'])==360 and q==co['source_queries'] and nn==co['source_NN'] and plies==co['source_plies']
    return dict(attempts=len(seen),plies=plies,queries=q,NN=nn,latest_within_limited_inventory_discrepancies=late_discrepancy)
def c4():
    f=read('FROZEN.json');plan=read('PLAN.json');co=read('COHORT.json');lk=read('ATLAS_LOCK.json');la=read('TEST_LABELS.json');su=read('SUMMARY.json');ext=read('EXT_SUMMARY.json')
    assert sha(D/'FROZEN.json')==plan['frozen']==lk['frozen']==la['frozen']==ext['frozen']
    assert sha(D/'PLAN.json')==co['plan'] and sha(D/'SOURCE.jsonl.gz')==co['source'] and sha(D/'COHORT.json')==lk['cohort']
    assert sha(D/'ATLAS.jsonl.gz')==lk['sha256']==la['atlas']==ext['atlas'];assert sha(D/'EXT_MATCH.jsonl.gz')==ext['sha256']
    assert sha(D/'SCORED.jsonl.gz')==su['scored'] and sha(D/'acquire.py')==f['code']
    for x in f['sources']:lock(x)
    frozen='3c4dfe23f0427d981922999c2511bd29872fdfb4';rel=str(D.relative_to(ROOT))
    assert hashlib.sha256(git('show',frozen+':'+rel+'/FROZEN.json')).hexdigest()==sha(D/'FROZEN.json')
    assert all(SP.run(['git','-C',str(ROOT),'cat-file','-e',frozen+':'+rel+'/'+n],capture_output=True).returncode!=0 for n in ('SOURCE.jsonl.gz','ATLAS.jsonl.gz','TEST_LABELS.json'))
    fit=fit_audit(f);source=source_audit(plan,co)
    labs={x['root']['id']:x for x in la['rows']};scored={x['id']:x for x in raw(D/'SCORED.jsonl.gz')};ex={x['root']['id']:x for x in raw(D/'EXT_MATCH.jsonl.gz')}
    assert set(labs)==set(scored)==set(ex)=={r['id'] for r in co['roots']}
    model=f['controller'];nn=vis=req=nnmatch=extmatch=0;extg=[];proofcount=0
    armvals={k:[] for k in ('live','normal','match','time')};time10=cpu10=0
    for r in raw(D/'ATLAS.jsonl.gz'):
        root=r['root'];i=root['id'];z=scored[i];l=labs[i];x=ex[i]
        assert l['status']=='EXACT' and l['agreement'] and l['root']==root
        u={int(k):(v+1)/2 for k,v in l['independent'].items()}
        assert all(u[m]==((l['scores'][m]>0)-(l['scores'][m]<0)+1)/2 for m in u)
        pr=interval(root['history']);assert all(a<=u[m]<=b for m,a,b in pr);proofcount+=len(pr)
        live=r['live'];pre=live['prefix'];panel=panel0(pre);b=bin0(pre,panel);assert z['panel']==panel and z['bin']==b and live['decision']['proof']==pr and live['decision']['panel']==panel
        c=pick(model,b,.002);cc=certified(pr)
        channel='CERTIFIED' if cc is not None else 'STOP' if c is None else CH[c]
        assert channel==live['decision']['channel']==z['channel']
        decision=cc if cc is not None else panel[0] if c is None else readout(panel,pr,c,live['result'],model['switch'])
        assert live['best']==decision
        for name in armvals:
            a=r[name];grid.moments(a['result']);assert a['prefix']==pre and a['decision']['proof']==pr and a['result']['root_n']==a['visits']
            v=u[a['best']];armvals[name].append(v)
            close(z['arms'][name]['u'],v);close(z['arms'][name]['regret'],max(u.values())-v)
        for j,a in enumerate(r['fixed']):
            assert a['evidence']['prefix']==pre;prev=pre;cap=a['cap'];targets=[None,None,None] if j%3==0 else [panel[0],panel[j%3],None]
            for ph,k,target in zip(a['evidence']['phases'],((cap-16)//2,(cap-16)//2,8),targets):
                ca,n,w,e=grid.moments(prev);cb,n1,w1,e1=grid.moments(ph['result']);assert ca==cb
                delta=[n1[m]-n[m] for m in ca];assert min(delta)>=0 and sum(delta)==k and ph['delta']==delta and ph['target']==target
                if target is not None:assert delta[ca.index(target)]==k
                prev=ph['result']
            assert prev==a['result']
            d=readout(panel,pr,j,a['result'],model['switch'])
            close(z['fixed'][CH[j]]['gain'],u[d]-u[panel[0]]);assert z['fixed'][CH[j]]['best']==d
        for price in PR:
            c1=pick(model,b,price);d=cc if cc is not None else panel[0] if c1 is None else readout(panel,pr,c1,r['fixed'][c1]['result'],model['switch'])
            close(z['sensitivity'][str(price)]['u'],u[d])
        for a in (live,r['normal'],r['match'],r['time'],*r['fixed']):nn+=a['NN_calls'];vis+=a['visits'];req+=1
        nnmatch+=live['NN_calls']==r['match']['NN_calls']
        time10+=abs(live['decision']['tail_us']-r['time']['decision']['tail_us'])<=.1*max(1,live['decision']['tail_us'])
        cpu10+=abs(live['cpu_ns']-r['time']['cpu_ns'])<=.1*live['cpu_ns']
        for name in ('live','normal'):assert x[name]['prefix']==r[name]['prefix'] and x[name]['result']==r[name]['result'] and x[name]['best']==r[name]['best']
        a=x['matchx'];grid.moments(a['result']);assert a['prefix']==pre and a['NN_calls']<=live['NN_calls'] and (a['NN_calls']==live['NN_calls'] or a['visits']==256)
        if a['NN_calls']==live['NN_calls']:extmatch+=1;extg.append(u[live['best']]-u[a['best']])
    assert (req,nn,vis,nnmatch)==(lk['requests'],lk['NN'],lk['visits'],lk['nn_matches']) and extmatch==ext['matched']['n']
    close(S.mean(extg),ext['matched']['gain'])
    for k,v in armvals.items():close(S.mean(v),su['arms'][k]['u'])
    rng=random.Random(2026112620);boot=sorted(S.mean(rng.choices(extg,k=len(extg))) for _ in range(4000))
    return dict(fit_A=fit[0],fit_B=fit[1],cal_checks=fit[2],source=source,roots=len(labs),actions=proofcount,independent_nodes=sum(x['nodes'] for x in labs.values()),native_requests=req,visits=vis,NN=nn,initial_exact_NN=nnmatch,extended_exact_NN=extmatch,ext_gain=S.mean(extg),ext_descriptive_boot=[boot[99],boot[3899]],time_within_10pct=time10,cpu_within_10pct=cpu10)
def gomoku():
    pl=read('G_PLAN.json');co=read('G_COHORT.json');pan=read('G_PANEL.json');lk=read('G_LABELS_LOCK.json');su=read('G_SUMMARY.json')
    assert sha(D/'G_PLAN.json')==co['plan'] and sha(D/'G_SOURCE.jsonl.gz')==co['source'] and sha(D/'G_COHORT.json')==pan['cohort']
    assert sha(D/'G_ATLAS.jsonl.gz')==pan['atlas'] and sha(D/'G_PANEL.json')==lk['panel'] and sha(D/'G_LABELS.jsonl.gz')==lk['sha256']
    for x in pl['source_locks']:
        if 'commit' in x:lock(x)
        else:assert sha(ROOT/x['path'])==x['sha256']
    fam=set();roots={r['id']:r for r in co['roots']};plies=nn=0
    for z in raw(D/'G_SOURCE.jsonl.gz'):
        j=z['job'];b,s,w=gref.replay(j['opening'],'freestyle')
        k=hashlib.sha256(gref.orbit_board(b,s,'freestyle')).hexdigest();assert k==j['family'] and k not in fam;fam.add(k)
        h=list(j['opening'])
        for a in z['rows']:
            b,s,w=gref.replay(h,'freestyle');assert w is None and b[a['move']]==0;h.append(a['move']);plies+=1;nn+=a['NN']
        assert h==z['history']
        if z['selected']:
            r=z['selected'];assert roots[r['id']]==r and h[:len(r['history'])]==r['history']
            b,s,w=gref.replay(r['history'],'freestyle');p=gref.proof_reference(b,s,'freestyle')
            assert w is None and not p['own'] and not p['opponent']
    assert plies==co['source_plies'] and nn==co['source_NN']
    atlas={};hidden=0
    for z in raw(D/'G_ATLAS.jsonl.gz'):
        r=z['root'];i=r['id'];a=z['response'];arm=z['arm'];b,s,w=gref.replay(r['history'],'freestyle')
        assert a['board']==b and a['proof']==gref.proof_reference(b,s,'freestyle')
        ph=a['decision']['phases'];pre=ph[0]['result'];ca,n,w,e=grid.moments(pre);cb,n1,w1,e1=grid.moments(ph[1]['result']);assert ca==cb
        t=a['decision']['target']
        if arm in ('DSX','DSX2') and t:
            m=t['move'];assert m not in e and m in e1 and n[m]==0 and n1[m]==4 and not t['materialized'];hidden+=1
            if arm=='DSX2':assert next(x[2] for x in a['proof']['actions'] if x[0]==m)>-1
        for ph0 in ph:grid.moments(ph0['result'])
        assert a['result']==ph[-1]['result'];atlas[i,arm]=a
    assert hidden==pan['hidden_materialized']
    U={};games=plies2=nn2=0
    for z in raw(D/'G_LABELS.jsonl.gz'):
        r=roots[z['id']];h=r['history']+[z['action']]
        for a in z['rows']:
            b,s,w=gref.replay(h,'freestyle');assert w is None and b[a['move']]==0;h.append(a['move']);plies2+=1;nn2+=a['NN']
        b,s,w=gref.replay(h,'freestyle');assert h==z['history'] and w is not None and w==z['winner']
        close(z['utility'],(1+r['player']*w)/2)
        d=U.setdefault((z['id'],z['action']),{});assert z['arm'] not in d;d[z['arm']]=z['utility'];games+=1
    assert (games,plies2,nn2)==(lk['games'],lk['plies'],lk['NN']) and all(set(x)=={'STANDARD_PW','STANDARD_FULL'} for x in U.values())
    V={k:S.mean(v.values()) for k,v in U.items()}
    for r in pan['panels']:
        i=r['root']['id'];p0=atlas[i,'DS0']['decision']['phases'][0]['result']
        assert all(atlas[i,c]['decision']['phases'][0]['result']==p0 for c in ('DS0','DSX','DSX2'))
        assert r['targets']['DSX']!=r['targets']['DSX2']
    for c in ('DS0','DSX','DSX2'):
        for kind,field in [('native','native'),('qbest','q')]:
            val=[V[r['root']['id'],r[kind][c]] for r in pan['panels']];reg=[max(V[r['root']['id'],m] for m in r['moves'])-v for r,v in zip(pan['panels'],val)]
            close(S.mean(val),su[field][c]['utility']);close(S.mean(reg),su[field][c]['panel_regret'])
    for c,field in [('DSX','hidden'),('DSX2','hidden2')]:
        v=[V[r['root']['id'],r['targets'][c]]-V[r['root']['id'],r['anchor']] for r in pan['panels'] if r['targets'][c] is not None]
        close(S.mean(v),su[field+'_mean_gain']);assert sum(x>0 for x in v)==su[field+'_positive'] and sum(x<0 for x in v)==su[field+'_negative']
    return dict(families=len(fam),roots=len(roots),queries=len(atlas),materialized=hidden,games=games,label_plies=plies2,label_NN=nn2,action_pairs=len(V),policy_disagreements=sum(len(set(v.values()))>1 for v in U.values()))
if __name__=='__main__':print(json.dumps(dict(c4=c4(),gomoku=gomoku(),scope='independent fixed-grid source replay, split readout/gain-table reconstruction, all exact utility/proof checks and native phase/decision/resource checks; source native moves verified from recorded decisions not independently re-searched, older historical manifests hash-verified and K16-19 opening/board exclusions independently checked; no full MCTS reimplementation'),indent=2,sort_keys=True))
