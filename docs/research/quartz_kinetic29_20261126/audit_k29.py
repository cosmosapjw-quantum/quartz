"""Independent exact readback of locked K27 data and policy, not another CNN/solver execution."""
import pathlib,hashlib,subprocess as SP,importlib.util,json,gzip,random,math,statistics as S,collections as C
from fractions import Fraction as F
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2];P=D.parent
SEED=2026112629
A=[(0,1,0,False,0),(0,4,0,False,0),(0,8,0,False,0),(0,4,4,False,0),(0,8,8,False,0),(0,8,8,True,0),(15,8,0,False,4),(15,8,8,True,4)]
N=8;STOP=8;KINDS=('short','serial','tree4','prior','continue','priced')
def read(n):return json.loads((D/n).read_text())
def raw(p):
    with gzip.open(p,'rt') as f:
        for l in f:yield json.loads(l)
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def git(*a):return SP.check_output(['git','-C',str(ROOT),*a])
def near(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(a,b)
def gridkey(b):return min(tuple(tuple(x) for x in b),tuple(tuple(x) for x in b[::-1]))
def flatkey(b):
    return gridkey([[1 if b[7*y+c]==1 else 2 for y in range(6) if b[7*y+c]!=0] for c in range(7)])
sp=importlib.util.spec_from_file_location('grid17',P/'quartz_kinetic17_20261126/audit_k17.py');g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)
plan=read('PLAN.json');co=read('COHORT.json');model=read('MODEL.json');frozen=read('FROZEN.json');summary=read('SUMMARY.json')
assert sha(D/'PLAN.json')==co['plan'] and sha(D/'SOURCE.jsonl.gz')==co['source']
ml=read('MODEL_LOCK.json');cal_scoring=read('CAL_SCORING.json')
assert sha(D/'MODEL.json')==ml['model']==frozen['model'] and sha(D/'MODEL_LOCK.json')==frozen['model_lock']
assert sha(D/'CAL_LIVE_LOCK.json')==frozen['cal_live'] and sha(D/'CAL_SCORING.json')==ml['cal_scoring'] and sha(D/'HIST_PRIOR.json')==ml['historical']
assert sha(D/'TRAIN_CAL_ATLAS_LOCK.json')==ml['atlas'] and sha(D/'TRAIN_CAL_LABELS.json')==ml['labels']
assert sha(D/'trial.py')==ml['code']==frozen['code']
locks=plan['history_locks']+list(plan['prior_locks'].values())+plan['assets']['helpers']
for lk in locks:assert hashlib.sha256(git('show',lk['commit']+':'+lk['path'])).hexdigest()==lk['sha256']
prefix=str(D.relative_to(ROOT))+'/'
for name,absent in [('MODEL.json',['CAL_LIVE_LOCK.json','FROZEN.json','TEST_ATLAS_LOCK.json','TEST_LIVE_LOCK.json']),('FROZEN.json',['TEST_ATLAS_LOCK.json','TEST_LIVE_LOCK.json','TEST_LABELS.json'])]:
    c=git('log','-1','--format=%H','--',prefix+name).decode().strip()
    assert hashlib.sha256(git('show',c+':'+prefix+name)).hexdigest()==sha(D/name)
    for z in absent:assert SP.run(['git','-C',str(ROOT),'cat-file','-e',c+':'+prefix+z],capture_output=True).returncode!=0
families=set();boards=set()
for name,lk in plan['prior_locks'].items():
    if name.endswith('.gz') or name=='FA05/attempts':continue
    obj=json.loads(git('show',lk['commit']+':'+lk['path']))
    rows=obj['queries'] if name.endswith('C07_B_QUERIES.json') else (obj.get('rows') or obj.get('connect4'))
    for r in rows:
        if r.get('board') is not None:boards.add(flatkey(r['board']))
        else:boards.add(gridkey(g.replay(r.get('history',r.get('sequence')))))
obj=json.loads(git('show','9b8a4718f7f92d83db2fb00e5806c6c4bd8cbfc2:docs/research/quartz_fouraxis05_20261125/C07_COHORT.json'))
for r in obj['rows']:boards.add(flatkey(r['board']))
for lk in plan['history_locks']:
    if lk['path'].endswith('SOURCE.jsonl.gz'):continue
    obj=json.loads(git('show',lk['commit']+':'+lk['path']))
    for r in obj.get('roots',[]):
        if r.get('domain','C4')!='C4':continue
        h=r['history'];boards.add(gridkey(g.replay(h)))
        if len(h)>=8:families.add(g.canonical(h[:8]))
    for j in obj.get('jobs',obj.get('c4_attempts',[])):
        if 'family' in j and len(j['family'])==8:families.add(j['family'])
        elif len(j.get('history',''))>=8:families.add(g.canonical(j['history'][:8]))
assert len(boards)==plan['historical_boards'],(len(boards),plan['historical_boards'])
assert len(families)==plan['historical_families'],(len(families),plan['historical_families'])
roots={r['id']:r for r in co['roots']};src=0;source_NN=source_plies=queries=0
for j,t in zip(plan['jobs'],raw(D/'SOURCE.jsonl.gz'),strict=True):
    src+=1;assert t['job']==j and g.canonical(j['opening'])==j['family'] and j['family'] not in families
    families.add(j['family']);b=g.replay(j['opening']);h=j['opening'];rng=random.Random(j['seed']);eligible=[]
    def wins(side):
        out=[]
        for k,col in enumerate(b):
            if len(col)==6:continue
            col.append(side)
            if g.won(b,side):out.append(k)
            col.pop()
        return out
    def cand():
        if len(h) in (24,26,28,30) and not g.won(b,1) and not g.won(b,2) and len([v for v in b if len(v)<6])>=3 and not wins(1) and not wins(2):
            eligible.append(dict(history=h,legal=[i for i,c in enumerate(b) if len(c)<6]))
    for st in t['steps']:
        cand();side=len(h)%2+1;legal=[i for i,c in enumerate(b) if len(c)<6];own=wins(side);opp=wins(3-side);m=st['move'];assert m in legal and not any(g.won(b,k) for k in (1,2))
        if own:assert st['kind']=='WIN' and m==own[0]
        elif len(opp)==1:assert st['kind']=='BLOCK' and m==opp[0]
        elif len(legal)==1:assert st['kind']=='ONLY' and m==legal[0]
        elif rng.random()<.75:assert st['kind']=='N32'
        else:assert st['kind']=='UNIFORM' and m==rng.choice(legal)
        if st['kind']=='N32':queries+=1;assert st['NN']>0
        else:assert st['NN']==0
        source_NN+=st['NN'];source_plies+=1;b[m].append(side);h+=str(m+1)
    cand();assert t['history']==h and t['eligible']==eligible,(j['id'],t['eligible'],eligible)
    selected=None
    for a in eligible:
        key=gridkey(g.replay(a['history']))
        if key not in boards:
            boards.add(key);selected=dict(id=j['id'],group=j['group'],family=j['family'],history=a['history'],legal=a['legal'],depth=len(a['history']),tactical=False);break
    assert t['selected']==selected==roots.get(j['id'])
assert src==3600 and (source_NN,source_plies,queries)==(co['NN'],co['plies'],co['queries'])
lab={};statuses=C.Counter();actions=nodes=0
for part in ('TRAIN_CAL','TEST'):
    x=read(part+'_LABELS.json');assert x['atlas']==sha(D/(part+'_ATLAS_LOCK.json'))
    for z in x['rows']:
        r=z['root'];assert roots[r['id']]==r;statuses[z['status']]+=1
        if z['status']!='EXACT':continue
        assert z['agreement'];u={int(k):v for k,v in z['independent'].items()}
        assert set(u)==set(r['legal']) and all(v==(z['scores'][a]>0)-(z['scores'][a]<0) for a,v in u.items())
        lab[r['id']]={a:F(v+1,2) for a,v in u.items()};actions+=len(u);nodes+=z['nodes']
# Independent grid/response and budget readback, no imported collector/fitter.
def proof_grid(r):
    b=g.replay(r['history']);side=len(r['history'])%2+1;pr=[]
    for mv in r['legal']:
        b[mv].append(side)
        if g.won(b,side):lo=hi=1.
        elif sum(map(len,b))==42:lo=hi=.5
        elif g.canwin(b,3-side):lo=hi=0.
        else:lo,hi=0.,1.
        pr.append([mv,lo,hi]);b[mv].pop()
    return pr
def certificate(pr):
    ub=max(x[2] for x in pr);return next((m for m,l,u in pr if l>=ub),None)
def feature_grid(r,s,pr):
    c=certificate(pr);c=s['best'] if c is None else c
    ix=sorted(range(s['len']),key=lambda i:(s['moves'][i]!=c,-s['n'][i],i))
    heights=list(map(len,g.replay(r['history'])));ps={m:(l,u) for m,l,u in pr};rank=[];allow=[];f=[len(r['history'])/42]
    for i in ix:
        m=s['moves'][i];n=s['n'][i];l,u=ps[m];rank.append(m);allow.append([int(2*l),int(2*u)])
        f.extend([math.sqrt(n/s['visits']),math.sqrt(max(0,s['prior'][i])),(1+s['w'][i]/(n+4))/2,math.sqrt(max(0,s['m2'][i])/(n+1))/2,heights[m]/6,abs(m-3)/3,l,u])
    return rank,f,allow
def native(a,r,arm,policy,tot,by,trace=False):
    assert a['schema']=='quartz_k29_c4_v1' and a['history']==r['history'] and a['legal']==r['legal'] and a['arm']==arm and a['policy']==policy
    pr=proof_grid(r);assert pr==a['proof'];cc=certificate(pr)
    if r['id'] in lab:assert all(lo<=lab[r['id']][m]<=hi for m,lo,hi in pr)
    pre,mid,end=[a[k] for k in ('prefix','mid','end')]
    ph,mh,eh=[a[k] for k in ('prefix_hooks','mid_hooks','end_hooks')]
    assert pre['visits']==8 and pre['nn']==1+ph['hooks']['base_evals']
    assert ph['ledger']['iterations']==8 and ph['ledger']['transferred']==0 and ph['hooks']['extra_evals']==0
    rank,f,allowed=feature_grid(r,mid,pr);assert rank==a['rank'] and allowed==a['allowed'] and len(f)==len(a['features'])
    for x,y in zip(f,a['features']):near(x,y)
    assert a['mid_best']==rank[0] and a['native_best']==(end['best'] if cc is None else cc)
    assert a['z']==rank.index(a['native_best'])
    c=C.Counter(requests=1,NN=a['NN_calls'],physical=a['physical_calls'],visits=a['visits'],stopped=int(arm==8))
    def stage(b,prev,s,h,old,mask,w,reserve):
        q=h['ledger'];hh=h['hooks'];assert b['target']==32 and b['cap']==128 and b['reserve']==reserve
        assert b['used_nn']==s['nn']-prev['nn']==hh['base_evals']+hh['extra_evals'] and b['used_nn']+b['unused']==32
        assert b['visits']==s['visits']-prev['visits']==hh['base_evals']+hh['terminal_leaves']
        assert 0<b['visits']<=128 and 0<=b['unused']<=32 and (b['unused']==0 or b['visits']==128)
        assert q['iterations']==s['visits'] and q['eligible']==q['novel']+q['blocked_receipt']
        assert q['transferred']==q['receipt_items']>=old['transferred'] and q['repeated_delivery']==0
        assert 1<=h['peak']<=w
        if not mask:assert hh['select_nodes']==hh['width_nodes']==hh['refinements']==hh['extra_evals']==0 and q['transferred']==old['transferred']
        if reserve:
            assert b['protected_extra']+b['released']==reserve and b['protected_extra']+b['borrowed']==b['pre_extra']
            assert b['protected_base']+b['borrowed']<=32-reserve and b['pre_extra']<=hh['extra_evals']
        c.update(stages=1,target=int(b['unused']==0),cap=int(b['unused']>0),unused=b['unused'],transferred=q['transferred']-old['transferred'],blocked=q['blocked_receipt']-old['blocked_receipt'],refinements=hh['refinements'])
    stage(a['first'],pre,mid,mh,ph['ledger'],0,1,0)
    q=a['queue']
    if arm==8:
        assert end==mid and a['tail'] is None and q is None
    else:
        mask,w,width,dedup,res=A[arm];stage(a['tail'],mid,end,eh,mh['ledger'],mask,w,res)
        if width:
            assert q['width']==width and q['dedup']==dedup and q['logical']==end['nn']-mid['nn']
            assert q['logical']==q['physical']+q['coalesced']==q['distinct']+q['repeat_keys']
            assert q['physical']>=q['distinct'] and q['coalesced']<=q['repeat_keys']
            lh,hh=q['logical_hist'],q['physical_hist']
            assert len(lh)==len(hh)==9 and lh[0]==hh[0]==0
            assert sum(lh)==sum(hh)==q['batches'] and sum(j*x for j,x in enumerate(lh))==q['logical'] and sum(j*x for j,x in enumerate(hh))==q['physical']
            assert not any(lh[width+1:]) and not any(hh[width+1:]) and 0<=q['peak_pending']<=w
            assert 0<=q['wait_ns']<=q['latency_ns'] and q['compute_ns']>=0
            if not dedup:assert q['coalesced']==0 and lh==hh
            assert q['parity_checks']==(q['physical'] if trace else 0) and (q['trace'] is not None)==trace
            c.update({k:q[k] for k in ('logical','coalesced','repeat_keys','distinct','batches','wait_ns','compute_ns','latency_ns','parity_checks')})
            c['batch_physical']+=q['physical'];c['padded_lanes']+=8*q['batches'];c['multi_batches']+=sum(lh[2:])
            for j,z in enumerate(lh):c['LH'+str(j)]+=z
            for j,z in enumerate(hh):c['PH'+str(j)]+=z
        else:assert q is None
    assert a['NN_calls']==end['nn'] and a['visits']==end['visits'] and a['physical_calls']==a['NN_calls']-(q['coalesced'] if q else 0)
    assert (eh['trace'] is not None)==trace and eh['ledger']['repeated_delivery']==0
    for s in (pre,mid,end):assert sum(s['n'][:s['len']])==s['visits'] and set(s['moves'][:s['len']])==set(rank)
    snap=a['snapshot'];assert snap['root_n']==end['visits'] and snap['evaluator_calls']==end['nn'] and snap['best']==end['best']
    for j,m in enumerate(end['moves'][:end['len']]):
        e=next((e for e in snap['edges'] if e['move']==m),{})
        assert e.get('n',0)==end['n'][j] and e.get('vl',0)==e.get('vvalue',0)==0
        near(e.get('w',0),end['w'][j]);near(e.get('m2',0),end['m2'][j])
    if policy is None:assert a['best']==a['native_best'] and a['prediction'] is None
    else:assert a['prediction'] is not None and a['belief_identity']==ml['model']
    tot.update(c);by[arm].update(c)
def rows(phase):
    lk=read(phase+'_LOCK.json');assert lk['cohort']==sha(D/'COHORT.json')
    for fi in lk['files']:
        p=D/fi['file'];assert sha(p)==fi['sha256'];yield from raw(p)
def datum(t):
    r=t['root'];z=t['results'];a=z['0:0'];u=lab[r['id']]
    rank,f,allow=feature_grid(r,a['mid'],proof_grid(r))
    return dict(f=f,u=[int(2*u[m]) for m in rank],allowed=allow,z=[[z[str(c)+':'+str(j)]['z'] for j in range(2)] for c in range(8)],
        cost=[[z[str(c)+':'+str(j)]['tail']['used_nn'] for j in range(2)] for c in range(8)])
def neighbors(donors,a):
    L=len(a['u']);return sorted([(i,math.fsum((x-y)*(x-y) for x,y in zip(a['f'],d['f']))/len(a['f'])) for i,d in enumerate(donors) if len(d['u'])==L and all(l<=v<=h for v,(l,h) in zip(d['u'],a['allowed']))],key=lambda z:(z[1],z[0]))
def weight(ds,k,eps):
    z=ds[:k] if k else ds
    if not z:return [],1.
    w=[1/(.01+d) if k else 1. for i,d in z];s=math.fsum(w)
    return [(i,v*(1-eps)/s) for (i,d),v in zip(z,w)],eps
def am(v):
    i=0
    for j in range(1,len(v)):
        if v[j]>v[i]+1e-12:i=j
    return i
def law(donors,a,k,eps,ds=None,prior_only=False):
    if ds is None:ds=neighbors(donors,a)
    ww,ep=weight(ds,k,eps);L=len(a['u']);u=tuple(a['u']);al=a['allowed'];base=[(l+h)/4 for l,h in al];uniform=ep/math.prod(h-l+1 for l,h in al)
    pu=C.defaultdict(float);joint=[C.defaultdict(float) for _ in range(8)]
    for i,w in ww:
        d=donors[i];v=tuple(d['u']);pu[v]+=w
        if not prior_only:
            for c,z in enumerate(d['z']):
                for zz in z:joint[c][v,zz]+=w/2
    p=pu[u]+uniform
    m=[ep*b+math.fsum(w*v[j]/2 for v,w in pu.items()) for j,b in enumerate(base)];rank=am(m) if ww else 0
    brier=0
    for j,(l,h) in enumerate(al):
        for v in range(3):
            x=(ep/(h-l+1) if l<=v<=h else 0)+math.fsum(w for uu,w in pu.items() if uu[j]==v)
            brier+=(x-(u[j]==v))**2/L
    out=dict(prior=m,prior_rank=rank,prior_nll=-math.log(p),prior_regret=(max(u)-u[rank])/2,prior_native_gain=(u[rank]-u[0])/2,brier=brier,utility_mse=math.fsum((v/2-x)**2 for v,x in zip(u,m))/L,ess=(1-ep)**2/math.fsum(w*w for i,w in ww) if ww else 0,donors=len(ww))
    if prior_only:return out
    ar=[]
    for c in range(8):
        z=a['z'][c]
        mass=[ep/L+math.fsum(w for (v,j),w in joint[c].items() if j==i) for i in range(L)]
        jj=[[ep*b/L+math.fsum(w*v[j]/2 for (v,r),w in joint[c].items() if r==i) for j,b in enumerate(base)] for i in range(L)]
        for i,x in enumerate(m):near(sum(row[i] for row in jj),x)
        ju=[uniform/L+joint[c][u,i] for i in range(L)]
        post=[am(row) if ww else 0 for row in jj];G=max(0,sum(row[am(row)] for row in jj)-m[am(m)]);dn=sum(row[i] for i,row in enumerate(jj))-m[0]
        cost=ep*32+math.fsum(w*sum(donors[i]['cost'][c])/2 for i,w in ww)
        ar.append(dict(mass=mass,joint_moments=jj,post_rank=post,G=G,D=dn,cost=cost,net=G-.002*cost,
            conditional_nll=S.mean(-math.log(ju[j]/p) for j in z),marginal_nll=S.mean(-math.log(mass[j]) for j in z),
            posterior_nll=S.mean(-math.log(ju[j]/mass[j]) for j in z),actual_native=S.mean((u[j]-u[0])/2 for j in z),
            actual_post=S.mean((u[post[j]]-u[rank])/2 for j in z),actual_post_regret=S.mean((max(u)-u[post[j]])/2 for j in z)))
    out['arms']=ar;out['continue_arm']=am([v['G'] for v in ar]) if ww else 0;z=am([v['net'] for v in ar]);out['priced_arm']=z if ww and ar[z]['net']>1e-12 else 8
    out['cond_nll']=S.mean(v['conditional_nll'] for v in ar);out['marg_nll']=S.mean(v['marginal_nll'] for v in ar);out['joint_nll']=out['prior_nll']+out['cond_nll']
    return out
def scores(donors,a,ds,k,eps):
    ww,ep=weight(ds,k,eps);L=len(a['u']);u=a['u'];q=ep/math.prod(h-l+1 for l,h in a['allowed'])
    good=[(i,w) for i,w in ww if donors[i]['u']==u];p=q+math.fsum(w for i,w in good)
    v=[]
    for c in range(8):
        for z in a['z'][c]:
            w=q/L+math.fsum(w*donors[i]['z'][c].count(z)/2 for i,w in good)
            v.append(-math.log(w/p))
    return -math.log(p),S.mean(v)
def check_prediction(a,p):
    q=a['prediction']
    for k in ('prior_rank','donors','continue_arm','priced_arm'):assert q[k]==p[k],(k,q[k],p[k])
    near(q['ess'],p['ess'])
    for x,y in zip(q['prior'],p['prior']):near(x,y)
    for x,y in zip(q['arms'],p['arms']):
        assert x['post_rank']==y['post_rank']
        for k in ('G','D','cost','net'):near(x[k],y[k])
        for u,v in zip(x['mass'],y['mass']):near(u,v)
        for u,v in zip(x['joint_moments'],y['joint_moments']):
            for b,c in zip(u,v):near(b,c)
    arm=8 if a['policy']=='prior' else p['continue_arm'] if a['policy']=='continue' else p['priced_arm'];assert a['arm']==arm
    cc=certificate(a['proof']);best=a['rank'][p['prior_rank'] if arm==8 else p['arms'][arm]['post_rank'][a['z']]]
    assert a['best']==(best if cc is None else cc)
def small(a,u):
    e=a['end'];return dict(u=u[a['best']],native=u[a['native_best']],opt=max(u.values()),net=u[a['best']]-F(a['NN_calls'],500),
        NN=a['NN_calls'],physical=a['physical_calls'],tail_NN=a['NN_calls']-a['mid']['nn'],visits=a['visits'],best=a['best'],arm=a['arm'],n=e['n'][:e['len']],moves=e['moves'][:e['len']],q=a['queue'])
tot=C.Counter();by=C.defaultdict(C.Counter);pc={};dat={};train=[];tids=[];calids=[];testids=[];fixed={};live={};cal=C.defaultdict(list);actual_model_checks=0;fp={}
for phase in ('TRAIN_CAL_ATLAS','CAL_LIVE','TEST_ATLAS','TEST_LIVE'):
    start=tot.copy();seen=set();lk=read(phase+'_LOCK.json')
    if phase=='CAL_LIVE':assert lk['freeze']==sha(D/'MODEL_LOCK.json')
    if phase.startswith('TEST'):assert lk['freeze']==sha(D/'FROZEN.json')
    for t in rows(phase):
        r=t['root'];i=r['id'];assert roots[i]==r and i not in seen;seen.add(i);u=lab.get(i)
        z=t['results'];oo=[(c,j) for c in range(8) for j in range(2)] if phase.endswith('ATLAS') else [(k,j) for k in KINDS for j in range(2)]
        random.Random(SEED+(900000 if phase.endswith('ATLAS') else 800000)+i).shuffle(oo);assert t['order']==[list(x) for x in oo]
        a0=next(iter(z.values()))
        if phase.endswith('ATLAS') and u:
            d=datum(t);dat[i]=d
            if r['group']=='TRAIN':train.append({k:d[k] for k in ('f','u','z','cost')});tids.append(i)
            elif r['group']=='CAL':calids.append(i)
            else:testids.append(i);fixed[i]={str(c):[small(z[str(c)+':'+str(j)],u) for j in range(2)] for c in range(8)}
        if not phase.endswith('ATLAS') and u:
            fp[i]=law(model['donors'],dat[i],model['k'],model['eps'])
            v={k:[small(z[k+':'+str(j)],u) for j in range(2)] for k in KINDS}
            if phase=='CAL_LIVE':
                for k in KINDS:cal[k].append(S.mean(a['net'] for a in v[k]))
            else:live[i]=v
        for k,j in oo:
            a=z[str(k)+':'+str(j)];assert a['prefix']==a0['prefix'] and a['mid']==a0['mid'] and a['rank']==a0['rank'] and a['features']==a0['features']
            pol=None if phase.endswith('ATLAS') or k in KINDS[:3] else k
            arm=k if phase.endswith('ATLAS') else {'short':8,'serial':0,'tree4':1}.get(k,a['arm'])
            native(a,r,arm,pol,tot,by)
            if pol and u:check_prediction(a,fp[i]);actual_model_checks+=1
    assert len(seen)==lk['roots']
    assert all(tot[k]-start[k]==lk[k] for k in ('requests','NN','physical','visits'));pc[phase]={k:lk[k] for k in ('roots','requests','NN','physical','visits')}
# Compare historical priors and exact model training identity.
assert tids==ml['train_ids'] and calids==ml['cal_ids'] and len(train)==len(model['donors'])
for a,b in zip(train,model['donors']):
    for k in ('u','z','cost'):assert a[k]==b[k]
    for x,y in zip(a['f'],b['f']):near(x,y)
historical=read('HIST_PRIOR.json')
for lk in historical['locks']:assert hashlib.sha256(git('show',lk['commit']+':'+lk['path'])).hexdigest()==lk['sha256']
hp=P/'quartz_kinetic28_20261126'
hl={z['root']['id']:{int(k):int(v+1) for k,v in z['independent'].items()} for z in json.loads((hp/'TRAIN_CAL_LABELS.json').read_text())['rows'] if z['status']=='EXACT' and z['root']['group']=='TRAIN'}
hs=[]
for fi in json.loads((hp/'TRAIN_CAL_ATLAS_LOCK.json').read_text())['files']:
    if fi['budget']!=32:continue
    assert sha(hp/fi['file'])==fi['sha256']
    for t in raw(hp/fi['file']):
        r=t['root']
        if r['id'] not in hl:continue
        a=t['paths'][0];rank,f,allow=feature_grid(r,a['stages'][0]['lite'],a['proof']);hs.append(dict(f=f,u=[hl[r['id']][m] for m in rank]))
assert len(hs)==len(historical['donors'])
for a,b in zip(hs,historical['donors']):
    assert a['u']==b['u']
    for x,y in zip(a['f'],b['f']):near(x,y)
v=C.defaultdict(list)
for i in calids:
    a=dat[i];ds=neighbors(model['donors'],a)
    for p in cal_scoring['params']:v[p['k'],p['eps']].append(scores(model['donors'],a,ds,p['k'],p['eps']))
for p in cal_scoring['params']:
    rr=v[p['k'],p['eps']];near(S.mean(x for x,y in rr),p['prior']);near(S.mean(y for x,y in rr),p['cond']);near(S.mean(x+y for x,y in rr),p['joint'])
best=min(cal_scoring['params'],key=lambda p:p['joint']);pool=min([p for p in cal_scoring['params'] if p['k']==0],key=lambda p:p['joint'])
assert (best['k'],best['eps'])==(model['k'],model['eps']) and pool==ml['pooled']
cv={k:S.mean(cal[k]) for k in KINDS}
for k,x in cv.items():near(x,frozen['calnet'][k])
assert frozen['fixed']==max(('serial','tree4','continue'),key=cv.__getitem__) and frozen['priced']==max(KINDS,key=cv.__getitem__)
# Actual means, repeated deviations, root-paired contrasts.
ids=sorted(testids);assert len(ids)==summary['n'] and set(ids)==set(fixed)==set(live)
def stats(v,z):
    aa=[a for w in v for a in w]
    for k in ('u','native','net','NN','physical','visits','tail_NN'):near(S.mean(a[k] for a in aa),z[k])
    near(S.mean(a['opt']-a['u'] for a in aa),z['regret']);assert sum(a['arm']==8 for a in aa)==z['stopped']
    q=C.Counter()
    for a in aa:
        if a['q']:
            t=a['q'];q.update({k:t[k] for k in ('logical','physical','coalesced','distinct','repeat_keys','batches','wait_ns','compute_ns','latency_ns')})
            q['padded_lanes']+=8*t['batches'];q['multi_batches']+=sum(t['logical_hist'][2:])
            for j,x in enumerate(t['logical_hist']):q['LH'+str(j)]+=x
            for j,x in enumerate(t['physical_hist']):q['PH'+str(j)]+=x
    for k,x in q.items():assert z['counts'][k]==x
def pair(a,b,z):
    assert len(a)==len(b)==z['n']
    if not a:return
    d=[S.mean(x['u']-y['u'] for x,y in zip(v,w)) for v,w in zip(a,b)]
    near(S.mean(d),z['gain']);near(S.mean(x['net']-y['net'] for v,w in zip(a,b) for x,y in zip(v,w)),z['net_gain'])
    assert sum(x>0 for x in d)==z['better'] and sum(x<0 for x in d)==z['worse']
    near(S.mean(x['NN']-y['NN'] for v,w in zip(a,b) for x,y in zip(v,w)),z['NN_delta'])
    near(S.mean(x['physical']-y['physical'] for v,w in zip(a,b) for x,y in zip(v,w)),z['physical_delta'])
def compare(a,b,z):
    pair(a,b,z['all'])
    for key,name in [('NN','equal_logical'),('physical','equal_physical')]:
        m=[(v,w) for v,w in zip(a,b) if all(x[key]==y[key] for x,y in zip(v,w))]
        pair([v for v,w in m],[w for v,w in m],z[name])
def js(a,b):
    x={m:n/sum(a['n']) for m,n in zip(a['moves'],a['n'])};y={m:n/sum(b['n']) for m,n in zip(b['moves'],b['n'])}
    return sum((p*math.log(2*p/(p+q)) if p else 0.)/2+(q*math.log(2*q/(p+q)) if q else 0.)/2 for m in x.keys()|y.keys() for p,q in [(x.get(m,0),y.get(m,0))])
fd={str(c):[fixed[i][str(c)] for i in ids] for c in range(8)};lv={k:[live[i][k] for i in ids] for k in KINDS}
for d,s0 in [(fd,'fixed'),(lv,'live')]:
    for k,v in d.items():
        stats(v,summary[s0][k]);z=summary[s0+'_repeats'][k]
        near(math.sqrt(S.mean((a['u']-b['u'])**2/2 for a,b in v)),z['utility_sd']);near(math.sqrt(S.mean((a['NN']-b['NN'])**2/2 for a,b in v)),z['logical_sd'])
        near(math.sqrt(S.mean((a['physical']-b['physical'])**2/2 for a,b in v)),z['physical_sd']);near(S.mean(js(a,b) for a,b in v),z['JSD'])
        assert sum(a['best']!=b['best'] for a,b in v)==z['disagree']
    for k,z in summary[s0+'_contrasts'].items():
        a,b=k.split(':');compare(d[a],d[b],z)
def forecast(v,z):
    for k in ('prior_nll','brier','utility_mse','prior_regret','prior_native_gain','ess'):near(S.mean(x[k] for x in v),z[k])
    if 'arms' not in z:return
    for k in ('joint_nll','cond_nll','marg_nll'):near(S.mean(x[k] for x in v),z[k])
    assert sum(x['donors']==0 for x in v)==z['no_donor']
    for c,a in enumerate(z['arms']):
        q=[x['arms'][c] for x in v]
        for k in ('G','D','cost','conditional_nll','marginal_nll','posterior_nll','actual_native','actual_post','actual_post_regret'):near(S.mean(x[k] for x in q),a[k])
        x=[p['D'] for p in q];y=[p['actual_native'] for p in q];near(math.sqrt(S.mean((b-c)**2 for b,c in zip(x,y))),a['native_D_rmse'])
        if a['native_D_corr'] is not None:near(S.correlation(x,y),a['native_D_corr'])
forecast([fp[i] for i in ids],summary['forecast'])
forecast([law(model['donors'],dat[i],0,pool['eps']) for i in ids],summary['pooled_forecast'])
forecast([law(historical['donors'],dat[i],model['k'],model['eps'],prior_only=True) for i in ids],summary['old_prior'])
assert actual_model_checks==(len(calids)+len(ids))*6
print('MAIN_AUDITED',dict(tot),flush=True)
# Batch trace actual order and exact keys; independent receipt chronological replay.
import struct
def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
def exactboard(b):return tuple(tuple(c) for c in b)
def statekey(b,side):
    k=1<<84 if side==1 else 0
    for x in range(7):
        for y in range(6):
            digit=1 if y>=len(b[x]) else 2 if b[x][y]==1 else 0
            k|=digit<<(2*(7*y+x))
    return format(k,'022x')
def check_trace(r,a):
    q=a['queue'];records=q['trace'];assert len(records)==q['batches'];basekeys=C.Counter();queued=C.Counter();lh=C.Counter();ph=C.Counter()
    for z in records:
        ks=z['keys'];route=z['route'];assert len(ks)==len(route)==len(z['wait_ns'])
        u=[];m={}
        for key in ks:
            if q['dedup']:
                if key not in m:m[key]=len(m)
                u.append(m[key])
            else:u.append(len(u))
        assert u==route and z['physical']==len(set(route))
        queued.update(ks);lh[len(ks)]+=1;ph[z['physical']]+=1
    assert sum(queued.values())==q['logical'] and len(queued)==q['distinct']
    assert [lh[i] for i in range(9)]==q['logical_hist'] and [ph[i] for i in range(9)]==q['physical_hist']
    assert sum(sum(z['wait_ns']) for z in records)==q['wait_ns'] and sum(z['compute_ns'] for z in records)==q['compute_ns']
    root=g.replay(r['history']);ledger={};payloads={};samples=C.defaultdict(list);count=C.Counter();trace=a['end_hooks']['trace']
    bounds={a['prefix']['visits']:(a['prefix'],a['prefix_hooks']),a['mid']['visits']:(a['mid'],a['mid_hooks']),a['end']['visits']:(a['end'],a['end_hooks'])}
    assert len(trace)==a['visits']
    def ls(i):
        return dict(iterations=i,edges=count['edges'],children=len(ledger),contexts=sum(len(x) for x in ledger.values()),shared=sum(len(x)>1 for x in ledger.values()),unique_base_leaf=len(payloads),eligible=count['eligible'],novel=count['novel'],blocked_receipt=count['blocked_receipt'],transferred=count['transferred'],receipt_items=sum(len(z[1]) for x in ledger.values() for z in x.values()),applied=count['applied'],changed=count['changed'],repeated_delivery=0)
    for i,t in enumerate(trace):
        b=[c[:] for c in root];side=len(r['history'])%2+1;keys=[];mask=0 if i<a['mid']['visits'] else A[a['arm']][0]
        assert t['mask']==mask
        for mv in t['path']:
            assert not g.won(b,1) and not g.won(b,2) and len(b[mv])<6
            p0=exactboard(b);b[mv].append(side);side=3-side;keys.append((exactboard(b),(p0,mv)))
        terminal=g.won(b,1) or g.won(b,2) or sum(map(len,b))==42
        assert t['terminal']==terminal
        if i>=a['mid']['visits'] and not terminal:basekeys[statekey(b,side)]+=1
        leaf=(exactboard(b),terminal);raw0=f32(t['raw']);y=f32(t['leaf'])
        if terminal:assert raw0==y==(-1 if g.won(b,3-side) else 0)
        if not mask&4:assert raw0==y
        if leaf in payloads:assert payloads[leaf]==raw0
        else:payloads[leaf]=raw0
        for j in reversed(range(len(keys))):
            count['edges']+=1;child,ctx=keys[j];cc=ledger.setdefault(child,{})
            own,received=cc.setdefault(ctx,({},{}));other={}
            for p,(dd,rr) in cc.items():
                if p==ctx:continue
                for k,v in dd.items():
                    if k in other:assert other[k]==v
                    other[k]=v
            other.pop(leaf,None);elig={k:v for k,v in other.items() if k not in own};new={k:v for k,v in elig.items() if k not in received}
            s=t['steps'][j];assert (s['eligible'],s['novel'],s['blocked'])==(len(elig),len(new),len(elig)-len(new))
            count['eligible']+=len(elig);count['novel']+=len(new);count['blocked_receipt']+=len(elig)-len(new)
            mean=S.mean(new.values()) if new else None
            if mean is None:assert s['mean'] is None
            else:near(mean,s['mean'])
            assert f32(s['incoming'])==y
            if leaf in own:assert own[leaf]==raw0
            if leaf in received:assert received[leaf]==raw0
            own[leaf]=raw0
            if not terminal and mask&8 and new:
                assert not received.keys()&new.keys();received.update(new);count['transferred']+=len(new);count['applied']+=1
                old=y;alpha=min(.25,len(new)/(len(new)+4));y=f32(max(-1.,min(1.,(1-alpha)*y+alpha*mean)));count['changed']+=abs(y-old)>1e-7
            raw0=-raw0;y=-y;assert f32(s['backup'])==y
        samples[t['path'][0]].append(y)
        if i+1 in bounds:
            s,h=bounds[i+1];assert ls(i+1)==h['ledger']
            for j,m in enumerate(s['moves'][:s['len']]):
                v=samples[m];assert len(v)==s['n'][j];mu=S.mean(v) if v else 0.
                near(sum(v),s['w'][j]);near(sum((x-mu)**2 for x in v),s['m2'][j])
    assert all(queued[k]>=n for k,n in basekeys.items())
    assert sum(basekeys.values())==a['end_hooks']['hooks']['base_evals']
    assert sum((queued-basekeys).values())==a['end_hooks']['hooks']['extra_evals']
    return dict(iterations=len(trace),edges=count['edges'],receipt=count['transferred'],blocked=count['blocked_receipt'],queue_base=sum(basekeys.values()),queue_extra=sum((queued-basekeys).values()),parity=q['parity_checks'])
extra={};timing=C.defaultdict(list);tr=C.Counter()
for phase in ('DIAGNOSTIC','TIMING'):
    lk=read(phase+'_LOCK.json');p=D/(phase+'.jsonl.gz');assert sha(p)==lk['sha256'];t0=C.Counter();b0=C.defaultdict(C.Counter);seen=[]
    for t in raw(p):
        r=t['root'];seen.append(r['id'])
        for k,a in t['results'].items():
            c=int(k.split(':')[0]);native(a,r,c,None,t0,b0,phase=='DIAGNOSTIC')
            if phase=='DIAGNOSTIC':tr.update(check_trace(r,a))
            else:timing[str(c)].append((a['cpu_ns']/1e6,a['wall_ns']/1e6,a['tail_us']/1000,(a['queue_setup_us']+a['queue_join_us'])/1000,a['NN_calls'],a['physical_calls']))
    assert seen==lk['roots'] and t0['requests']==lk['requests'];extra[phase]=dict(t0)
out=dict(source_attempts=src,source_plies=source_plies,source_NN=source_NN,source_queries=queries,source_locks=len(locks),roots=len(roots),statuses=dict(statuses),exact_actions=actions,oracle_nodes=nodes,
    main=dict(tot),by_arm={str(k):dict(v) for k,v in by.items()},phase_counts=pc,model_native_checks=actual_model_checks,extra=extra,trace=dict(tr),
    timing={k:dict(n=len(v),**{a:S.mean(x[i] for x in v) for i,a in enumerate(('cpu_ms','wall_ms','tail_ms','setup_join_ms','NN','physical'))}) for k,v in timing.items()},
    scope='Independent source grids, exact stored WDL, full raw accounting, sparse joint law and native forecasts, actual policy/repeat readback, exact batch routes and receipt traces; no independent CNN/solver rerun except native batch-scalar trace parity.')
(D/'C09_AUDIT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print('AUDIT',json.dumps({k:v for k,v in out.items() if k not in ('by_arm','timing')},sort_keys=True),flush=True)
