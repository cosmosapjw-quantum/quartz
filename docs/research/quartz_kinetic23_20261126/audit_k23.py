"""Independent readback: grid legality, raw path snapshots, fixed-design exact OPE."""
import pathlib,subprocess as SP,hashlib,json,gzip,random,collections as C,math,itertools,importlib.util,statistics as S
from fractions import Fraction as F
import numpy as np
D=pathlib.Path(__file__).resolve().parent;R=D.parents[2];P=D.parent
def read(n):return json.loads((D/n).read_text())
def lines(n):
    with gzip.open(D/n,'rt') as f:return [json.loads(x) for x in f]
def sha(x):return hashlib.sha256(x).hexdigest()
def fs(n):return sha((D/n).read_bytes())
def git(*a):return SP.check_output(['git','-C',str(R),*a])
def near(a,b):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-11)
spec=importlib.util.spec_from_file_location('grid17',P/'quartz_kinetic17_20261126/audit_k17.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
plan=read('PLAN.json');co=read('COHORT.json');f=read('FROZEN.json');su=read('SUMMARY.json')
locks=plan['K20_K21_K22_locks']+plan['history_locks']+list(plan['prior_locks'].values())
for l in locks:assert sha(git('show',l['commit']+':'+l['path']))==l['sha256']
assert fs('PLAN.json')==co['plan'] and fs('SOURCE.jsonl.gz')==co['source']
assert fs('COHORT.json')==f['cohort'] and fs('TRAIN_CAL_ATLAS.jsonl.gz')==f['trace'] and fs('TRAIN_CAL_LABELS.json')==f['labels'] and fs('h2.py')==f['code']
freeze='d2ba7153b2ffc2bfd5cc6619da1d00733cc6579c';prefix=str(D.relative_to(R))+'/'
assert sha(git('show',freeze+':'+prefix+'FROZEN.json'))==fs('FROZEN.json')
for n in ('TEST_ATLAS.jsonl.gz','TEST_LABELS.json'):
    assert SP.run(['git','-C',str(R),'cat-file','-e',freeze+':'+prefix+n],capture_output=True).returncode!=0
assert read('TEST_LOCK.json')['freeze']==fs('FROZEN.json')
roots={r['id']:r for r in co['roots']};families=set();boards=set()
for l in locks:
    if l['path'].endswith('COHORT.json'):
        obj=json.loads(git('show',l['commit']+':'+l['path']))
        for r in obj.get('roots',[]):
            h=r.get('history','')
            if isinstance(h,str) and h and all(c in '1234567' for c in h):boards.add(g.boardkey(g.replay(h)));families.add(g.canonical(h[:8]))
    if l['path'].endswith('PLAN.json'):
        obj=json.loads(git('show',l['commit']+':'+l['path']))
        for j in obj.get('jobs',[]):families.add(j['family'])
src=lines('SOURCE.jsonl.gz');assert len(src)==720;seen=set();counts=C.Counter();nn=plies=0
for j,t in zip(plan['jobs'],src):
    assert t['job']==j and j['family']==g.canonical(j['opening']) and j['family'] not in seen and j['family'] not in families
    seen.add(j['family']);h=j['opening'];board=g.replay(h);rng=random.Random(j['seed'])
    for step in t['steps']:
        side=len(h)%2+1;assert not g.won(board,1) and not g.won(board,2)
        legal=[i for i,c in enumerate(board) if len(c)<6]
        def wins(p):
            out=[]
            for i in legal:
                board[i].append(p)
                if g.won(board,p):out.append(i)
                board[i].pop()
            return out
        own,opp=wins(side),wins(3-side);m=step['move'];assert m in legal
        if own:assert (m,step['kind'])==(own[0],'WIN')
        elif len(opp)==1:assert (m,step['kind'])==(opp[0],'BLOCK')
        elif len(legal)==1:assert (m,step['kind'])==(legal[0],'ONLY')
        elif rng.random()<.75:assert step['kind']=='N32'
        else:assert (m,step['kind'])==(rng.choice(legal),'UNIFORM')
        counts[step['kind']]+=1;nn+=step['NN'];plies+=1;h+=str(m+1);board[m].append(side)
    assert h==t['history'] and (len(h)==30 or g.won(board,1) or g.won(board,2))
    r=t['selected'];assert (r is not None)==(j['id'] in roots)
    if r:
        assert r==roots[j['id']] and h.startswith(r['history'])
        b=g.replay(r['history']);key=g.boardkey(b);assert key not in boards and not any(g.canwin(b,s) for s in (1,2)) and len(r['history']) in (24,26,28,30)
        assert r['legal']==[i for i,c in enumerate(b) if len(c)<6] and len(r['legal'])>=3;boards.add(key)
assert (nn,plies,counts['N32'])==(co['NN'],co['plies'],co['queries'])
lab={};actions=nodes=0
for part in ('TRAIN_CAL','TEST'):
    l=read(part+'_LABELS.json');k=read(part+'_LOCK.json');assert l['atlas']==k['sha256']==fs(part+'_ATLAS.jsonl.gz')
    for z in l['rows']:
        r=z['root'];assert roots[r['id']]==r and z['agreement'] and z['status']=='EXACT'
        w={int(k):v for k,v in z['independent'].items()};assert set(w)==set(r['legal']) and all(v==np.sign(z['scores'][a]) for a,v in w.items())
        lab[r['id']]={a:F(v+1,2) for a,v in w.items()};actions+=len(w);nodes+=z['nodes']
assert lab.keys()==roots.keys()
paths=list(itertools.product(range(3),repeat=2));records=[];req=states=pulses=0;caps=C.Counter()
def score(s):return [.5+.5*w/(n+4) for w,n in zip(s['w'][:s['len']],s['n'][:s['len']])]
def pan(s):
    moves=s['moves'][:s['len']];a=moves.index(s['best']);b=sorted([i for i in range(len(moves)) if i!=a],key=lambda i:(-s['n'][i],i))[0];c=next(i for i in range(len(moves)) if i not in (a,b));return [moves[i] for i in (a,b,c)]
def choice(r,t):
    a=t['first'][r['b']];return a*3+t['second'][a][r['mid'][a]]
for part in ('TRAIN_CAL','TEST'):
    for t in lines(part+'_ATLAS.jsonl.gz'):
        r=t['root'];u=lab[r['id']];seq=list(range(9));random.Random(2026112623+900000+r['id']).shuffle(seq)
        assert seq==t['order'] and seq[0]==t['logged'] and t['e']==[1/3]*3 and t['f']==[[1/3]*3]*3
        pre=t['arms'][0]['states'][0];panel=pan(pre);inds=[pre['moves'].index(m) for m in panel]
        us=[];cost=[];mid=[];z1=[];z2=[]
        def z(s):a=score(s);return 2*max(range(3),key=lambda i:a[inds[i]])+int(s['kl']>=.02)
        for i,x in enumerate(t['arms']+([t['live']] if part=='TEST' else [])):
            req+=1;st,md,en=x['states'];assert st==pre and x['panels']==[pan(st),pan(md)]
            a,b=x['chosen'];assert md==t['arms'][3*a]['states'][1]
            if i<9:assert (a,b)==paths[i]
            else:
                policy=f['tables'][f['selected']];assert a==policy['first'][st['bin']] and b==policy['second'][a][md['bin']]
                assert all(x[k]==t['arms'][3*a+b][k] for k in ('states','pulses','result','best','incremental_nn'))
            assert x['best']==en['best']==x['result']['best'] and en['nn']==x['NN_calls']
            assert st['visits']==8 and md['visits']<=64 and en['visits']<=128 and en['visits']==x['visits']
            for j,(s,p,e) in enumerate(zip(x['states'],x['pulses'],x['states'][1:])):
                pulses+=1;assert sum(p['n'][:p['len']])==p['visits']
                assert s['nn']<=p['nn']<=e['nn'] and s['visits']<=p['visits']<=min(s['visits']+4,e['visits'])
                assert e['nn']-s['nn']==x['incremental_nn'][j] and 0<=x['incremental_nn'][j]<=6
                assert all(a<=b<=c for a,b,c in zip(s['n'],p['n'],e['n']))
                cc=x['chosen'][j]
                if cc:
                    m=pan(s)[cc];ii=s['moves'].index(m);assert x['forced'][j]==m and p['n'][ii]-s['n'][ii]==p['visits']-s['visits']
                else:assert x['forced'][j] is None
                if i<9:caps[(part,x['causes'][j])]+=1
            for j,s in enumerate(x['states']):
                states+=1;n=s['n'][:s['len']];assert sum(n)==s['visits'] and set(s['moves'][:s['len']])==set(r['legal'])
                v=score(s);w=[.5]*len(v) if j==0 else score(x['states'][j-1])
                def soft(y):
                    z=[math.exp(4*a) for a in y];return math.log(sum(z)/len(z))/4,[a/sum(z) for a in z]
                f0,p0=soft(w);f1,p1=soft(v);k=sum(a*math.log(a/b) for a,b in zip(p0,p1));dr=sum(a*(b-c) for a,b,c in zip(p0,v,w))
                near([f1,k,dr,f1-f0-dr],[s['phi'],s['kl'],s['drift'],s['bregman']])
                n=sorted(n);assert s['bin']==4*int(n[-1]/sum(n)>=.75)+2*int(n[-1]-n[-2]>=2)+int(k>=.02)
            assert all(lo<=u[m]<=hi for m,lo,hi in x['proof'])
            if i<9:
                us.append(u[x['best']]);cost.append(x['incremental_nn']);z2.append(z(en))
                if i%3==0:mid.append(md['bin']);z1.append(z(md))
        records.append(dict(id=r['id'],group=r['group'],b=pre['bin'],mid=mid,u=tuple(u[m] for m in panel),us=us,n=cost,y=[v-F(sum(c),500) for v,c in zip(us,cost)],pre=u[pre['best']],proxy=score(pre)[inds[0]],full=max(u.values()),z1=z1,z2=z2,logged=t['logged']))
tr=[r for r in records if r['group']=='TRAIN'];cal=[r for r in records if r['group']=='CAL'];te=[r for r in records if r['group']=='TEST']
assert [r['id'] for r in tr]==f['train_ids'] and [r['id'] for r in cal]==f['cal_ids']
assert len(records)==254 and req==2357 and states==7071 and pulses==4714
# Independently recompute training means, adaptive tables, anchors and CAL selection.
def pool(x,z):return (sum(x)+8*z)/(len(x)+8)
q2=[]
for a in range(3):
    q2.append([[float(pool([r['y'][a*3+c] for r in tr if r['mid'][a]==b],S.mean(r['y'][a*3+c] for r in tr))) for c in range(3)] for b in range(8)])
near(q2,f['q2'])
for a in range(3):
    assert [int(np.argmax(q)) for q in q2[a]]==f['tables']['adaptive']['second'][a]
for k,t in f['tables'].items():
    q1=[]
    for b in range(8):
        q1.append([float(pool([r['y'][a*3+t['second'][a][r['mid'][a]]] for r in tr if r['b']==b],S.mean(r['y'][a*3+t['second'][a][r['mid'][a]]] for r in tr))) for a in range(3)])
    near(q1,f['q1'][k]);near(float(S.mean(r['y'][choice(r,t)] for r in cal)),f['calnet'][k])
    if k=='adaptive':assert [int(np.argmax(q)) for q in q1]==t['first']
    diff=[r for r in tr if choice(r,t)!=0];m=[(r['y'][choice(r,t)]+r['y'][0])/2 for r in diff]
    near([float(pool([(r['y'][choice(r,t)]+r['y'][0])/2 for r in diff if r['b']==b],S.mean(m) if m else F(1,2))) for b in range(8)],f['anchors'][k])
assert f['selected']==max(('NN','XN','XX','best_static','adaptive'),key=f['calnet'].__getitem__)
# Exact rational assignment expectations using frozen float nuisances as exact rational values.
for k,t in f['tables'].items():
    acc=C.defaultdict(list);vv=C.defaultdict(list);between=[];within=[]
    for r in te:
        c=choice(r,t);tar=r['y'][c]-r['y'][0];opt=(r['y'][c]+r['y'][0])/2
        def drscore(key,i):
            p=f['tables'][key];a,b=paths[i];a0=p['first'][r['b']];b0=p['second'][a][r['mid'][a]]
            q1=list(map(F,f['q1'][key][r['b']]));q2=list(map(F,f['q2'][a][r['mid'][a]]))
            return q1[a0]+3*int(a==a0)*(q2[b0]-q1[a])+9*int(a==a0 and b==b0)*(r['y'][i]-q2[b])
        def ip(m):return [9*(int(i==c)-int(i==0))*(y-m) for i,y in enumerate(r['y'])]
        scores={n:ip(F(m)) for n,m in [('raw',0),('proxy',r['proxy']),('learned',f['anchors'][k][r['b']]),('prefix_offline',r['pre']),('oracle_offline',opt)]}
        scores['sequential_DR']=[drscore(k,i)-drscore('NN',i) for i in range(9)]
        for name,x in scores.items():
            assert S.mean(x)==tar;acc[name].append(float(x[r['logged']]));vv[name].append(float(S.mean((v-tar)**2 for v in x)))
        x=scores['sequential_DR'];av=[S.mean(x[a*3:a*3+3]) for a in range(3)]
        between.append(float(S.mean((v-tar)**2 for v in av)));within.append(float(S.mean((x[i]-av[i//3])**2 for i in range(9))))
    n=len(te);z=su['ope'][k]
    near(float(S.mean(r['y'][choice(r,t)]-r['y'][0] for r in te)),z['full'])
    near(float(S.mean(r['us'][choice(r,t)] for r in te)),su['full'][k]['utility'])
    for name in acc:near([S.mean(acc[name]),math.sqrt(sum(vv[name]))/n],[z['actual'][name],z['sd'][name]])
    near([sum(between)/n**2,sum(within)/n**2],[z['DR_between'],z['DR_within']])
# Reconstruct channel likelihood independently.
U=list(itertools.product([F(0),F(1,2),F(1)],repeat=3))
pr0=[(sum(r['u']==u for r in tr)+.5)/(len(tr)+13.5) for u in U]
for b in range(8):
    rr=[r for r in tr if r['b']==b];m=f['models'][b]
    pr=[(sum(r['u']==u for r in rr)+8*p)/(len(rr)+8) for u,p in zip(U,pr0)];near(pr,m['pr'])
    for a in range(3):
        for i,u in enumerate(U):
            h=[sum(r['u']==u and r['z1'][a]==z for r in tr)+.5 for z in range(6)];h=[x/sum(h) for x in h]
            v=[sum(r['u']==u and r['z1'][a]==z for r in rr)+8*h[z] for z in range(6)];v=[x/sum(v) for x in v];near(v,m['L'][a][i])
        near([sum(pr[i]*m['L'][a][i][z] for i in range(27)) for z in range(6)],m['marg'][a])
for a in range(3):
    vals=[]
    for alpha in [0,.25,.5,.75,1.]:
        vals.append(S.mean(-math.log((1-alpha)*f['models'][r['b']]['marg'][a][r['z1'][a]]+alpha*f['models'][r['b']]['L'][a][U.index(r['u'])][r['z1'][a]]) for r in cal))
    near(vals,f['cal_channel_loss'][a]);assert f['alpha'][a]==[0,.25,.5,.75,1.][int(np.argmin(vals))]
    def losses(alpha):
        return S.mean(-math.log((1-alpha)*f['models'][r['b']]['marg'][a][r['z1'][a]]+alpha*f['models'][r['b']]['L'][a][U.index(r['u'])][r['z1'][a]]) for r in te)
    near([losses(0),losses(1),losses(f['alpha'][a])],[su['channel'][a][x] for x in ('marginal_log','raw_conditional_log','shrink_log')])
matched=[r for r in te if all(v==[6,6] for v in r['n'])];assert len(matched)==su['matched_n']==70
for i,(a,b) in enumerate(paths):
    v=[r['us'][i]-r['us'][3*a]-r['us'][b]+r['us'][0] for r in matched];name='NFX'[a]+'NFX'[b];q=su['interaction_matched'][name]
    near(float(S.mean(v)),q['mean']);assert sum(x>0 for x in v)==q['positive'] and sum(x<0 for x in v)==q['negative']
out=dict(roots=len(records),source_attempts=len(src),source_plies=plies,source_NN=nn,source_counts=dict(counts),actions=actions,oracle_nodes=nodes,requests=req,typed_states=states,pulse_states=pulses,all_matched=len(matched),exact_assignment_policy_unit_checks=len(te)*len(f['tables'])*6,source_locks=len(locks),scope='independent raw readback and exact rational OPE, training/channel reconstruction; source neural choices and both solver executions not rerun')
(D/'C09_AUDIT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(out)
