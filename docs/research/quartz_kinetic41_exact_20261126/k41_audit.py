"""Raw-grid audit and independent compiled exhaustive minimax. No experiment module import."""
import collections as C, fractions, gzip, importlib.util,json,math,pathlib,hashlib,subprocess,statistics as S,time
from k41_plan import D,ROOT,SEEDS,FIT,H,BUDGET,family,load,save,sha
F=fractions.Fraction
T=ROOT.parent/'quartz_kinetic41_exact_target_20261126'
sp=importlib.util.spec_from_file_location('_grid41',D.parent/'quartz_kinetic33_20261126/audit_k33.py')
grid=importlib.util.module_from_spec(sp);sp.loader.exec_module(grid)
CACHE={};CT=C.Counter();queries={};expected={};query_tags=C.defaultdict(set)
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def rows(mode):
    lk=load(mode+'_LOCK.json')
    for k,n in [('plan','PLAN.json'),('solver','exact41.py'),('engine','k41_engine.py'),('experiment','k41_experiment.py')]:assert sha(D/n)==lk[k]
    for fi in lk['files']:
        assert sha(D/fi['file'])==fi['sha256'];rr=list(raw(D/fi['file']))
        if not mode.startswith('EXACT'):
            assert sum(r['nn'] for r in rr)==fi['counts']['nn']
            assert all(r['seed']==fi['seed'] for r in rr)
        else:
            assert sum(r['exact']['nodes'] for r in rr)==fi['counts']['solver_nodes']
        yield from rr
def gate(b,p):
    k=tuple(b),p
    if k not in CACHE:CACHE[k]=grid.cert(b,p)
    return CACHE[k]
def register(h,values,tag):
    b,w=grid.replay(h);assert not w;p=1-2*(len(h)%2)
    k=tuple(sum(1<<i for i,v in enumerate(b) if v==s) for s in (p,-p))
    i=queries.setdefault(k,len(queries));query_tags[i].add(tag)
    if values is not None:
        assert set(map(int,values))=={i for i,v in enumerate(b) if v==0}
        assert all(v in (-1,0,1) for v in values.values())
        if i in expected:assert expected[i]==values
        expected[i]=values
    return i
def val(h,r,cap,lock,q=False):
    b,w=grid.replay(h);p=1-2*(len(h)%2);assert not w and b[r['move']]==0
    assert r['cap']==cap and r['lock']==lock
    g=gate(b,p);CT['rule_checks']+=1
    if q:
        should=len(h)>=37 and g['move'] is None
        assert r['exact_attempt']==should and 0<=r['exact_nodes']<=BUDGET and r['exact_cpu_ns']>=0
        if not should:assert not r['exact_complete'] and not r['exact_nodes']
        elif not r['exact_complete']:
            assert r['exact_nodes']==BUDGET and 'exact_values' not in r
            register(h,None,'arena_censored');CT['arena_censored']+=1
    if r['kind']=='CERTIFICATE':
        assert g['move']==r['move'] and g['kind']==r['proof'] and r['nn']==r['visits']==0 and not r['votes'];CT['certificates']+=1
    elif r['kind']=='EXACT41':
        assert q and g['move'] is None and r['exact_complete'] and r['exact_attempt']
        assert r['nn']==r['visits']==0 and not r['votes']
        z=r['exact_values'];assert r['exact_value']==max(z.values()) and r['move']==min(int(m) for m,v in z.items() if v==r['exact_value'])
        register(h,z,'arena_complete');CT['arena_exact']+=1
    else:
        assert g['move'] is None and r['proof']=='UNRESOLVED'
        legal=[i for i,v in enumerate(b) if not v];n=dict(r['votes'])
        assert list(n)==legal and all(type(v) is int and v>=0 for v in n.values())
        a=min(legal,key=lambda i:(-n[i],i));d=min((n[a]-n[i]+int(i>a) for i in legal if i!=a),default=10**9)
        assert r['move']==a and r['distance']==d and r['prior_best'] in legal
        assert sum(n.values())==r['visits']<=cap and r['nn']==1+r['visits']-r['terminal_leaves']
        assert 0<=r['terminal_leaves']<=r['visits']
        if r['kind']=='VOTE_LOCK':
            assert lock and d>cap-r['visits'] and r['visits']<cap;CT['locks']+=1
        else:assert r['kind']=='CAP' and r['visits']==cap
        CT['legal_edges']+=len(legal)
    assert r['wall_ns']>=r['rule_ns']>=0 and r['cpu_ns']>=0
def scalar(ev,h,cap,lock):
    b,w=grid.replay(h);p=1-2*(len(h)%2);assert not w and gate(b,p)['move'] is None
    tree={};start=ev.rows;terminal=0;rv=None;root_prior=None
    def expand(board,p,path):
        nonlocal rv,root_prior
        legal=[i for i,v in enumerate(board) if not v]
        a=sum(1<<i for i,v in enumerate(board) if v==1);c=sum(1<<i for i,v in enumerate(board) if v==-1)
        pr,v=ev(a,c,p,legal);tree[path]={m:[0,0.,float(q)] for m,q in zip(legal,pr)}
        if not path:rv=v;root_prior=[float(q) for q in pr]
        return v
    def step(board,p,path):
        nonlocal terminal
        if path not in tree:
            if path and grid.won(board,path[-1],-p):terminal+=1;return -1.
            if not any(v==0 for v in board):terminal+=1;return 0.
            return expand(board,p,path)
        e=tree[path];s=math.sqrt(max(1,sum(v[0] for v in e.values())))
        m=max(e,key=lambda m:((e[m][1]/e[m][0] if e[m][0] else 0.)+2*e[m][2]*s/(1+e[m][0]),-m))
        board[m]=p;v=-step(board,-p,path+(m,));board[m]=0;e[m][0]+=1;e[m][1]+=v;return v
    expand(b,p,());t=0;e=tree[()]
    while t<cap:
        a=max(e,key=lambda m:(e[m][0],-m))
        if lock and all(e[a][0]>e[i][0]+cap-t or (e[a][0]==e[i][0]+cap-t and a<i) for i in e if i!=a):break
        step(b,p,());t+=1
    a=max(e,key=lambda m:(e[m][0],-m));d=min((e[a][0]-e[i][0]+int(i>a) for i in e if i!=a),default=10**9)
    prior=sorted(root_prior,reverse=True);f=[1+sum(q*math.log(q) for q in root_prior if q>0)/math.log(len(root_prior)) if len(root_prior)>1 else 1.,abs(sum(v[1] for v in e.values())/max(1,t)-rv),d/max(1,t),math.log(prior[0]/prior[1]) if len(prior)>1 and prior[1]>0 else 0.]
    return dict(move=a,votes=[[i,v[0]] for i,v in e.items()],visits=t,nn=ev.rows-start,terminal_leaves=terminal,prior_best=max(e,key=lambda i:(e[i][2],-i))),f
def run():
    pl=load('PLAN.json');jobs={j['id']:j for j in pl['jobs']};assert len(jobs)==368 and len({family(j['opening']) for j in jobs.values()})==368
    for p in pl['prior_locks']:
        f=ROOT/p['path'];assert sha(f)==p['sha256'];old=json.loads(f.read_text())
        assert not ({family(j['opening']) for j in old['jobs']}&{family(j['opening']) for j in jobs.values()})
    for j in jobs.values():
        b,w=grid.replay(j['opening']);assert not w and j['empties']==b.count(0)
        if j['split']!='EARLY':assert 8<=b.count(0)<=12 and gate(b,1-2*(len(j['opening'])%2))['move'] is None
    ex={};atlas={};sample=C.defaultdict(list)
    for split in ('TRAIN','CAL','TEST'):
        for r in rows('EXACT_'+split):
            j=r['job'];assert j==jobs[j['id']] and j['split']==split and j['id'] not in ex
            z=r['exact'];assert 0<z['nodes']<=BUDGET
            if z['complete']:
                assert z['best_value']==max(z['values'].values()) and z['best']==min(int(m) for m,v in z['values'].items() if v==z['best_value'])
                q=register(j['opening'],z['values'],'corpus_complete')
            else:
                assert z['nodes']==BUDGET and 'values' not in z;q=register(j['opening'],None,'corpus_censored')
            ex[j['id']]=(r,q)
        for r in rows('ATLAS_'+split):
            j=r['job'];s=r['seed'];assert j==jobs[j['id']] and j['split']==split and (j['id'],s) not in atlas
            if split=='TRAIN':assert s in FIT
            z=ex[j['id']][0]['exact'];assert z['complete']
            a=r['atlas'];h=j['opening']
            for k in ('B32',*(f'L{h}' for h in H)):
                val(h,a[k],int(k[1:]),k[0]=='L');assert r['values'][k]==z['values'][str(a[k]['move'])]
            assert r['best_value']==z['best_value'] and a['B32']['move']==a['L32']['move']
            assert a['L20']['nn']<=a['L26']['nn']<=a['L32']['nn']<=a['L64']['nn']
            assert a['actual_nn']==r['nn']==a['B32']['nn']+a['L64']['nn']
            atlas[j['id'],s]=r
            if len(sample[s])<4:sample[s].append(r)
    assert len(ex)==240
    assert set(atlas)=={(i,s) for i,(r,q) in ex.items() if r['exact']['complete'] for s in (FIT if r['job']['split']=='TRAIN' else SEEDS)}
    fr=load('FROZEN.json');tr=[r for r in atlas.values() if r['job']['split']=='TRAIN']
    cuts=[S.median(r['atlas']['features'][i] for r in tr) for i in range(3)];assert cuts==fr['cuts']
    pol=fr['policy'];pts={}
    for r in tr:
        z=str(sum(1<<i for i,(v,t) in enumerate(zip(r['atlas']['features'][:3],cuts)) if v>t))
        p=pts.setdefault(z,{str(h):[0,0] for h in (20,26,32)})
        for h in p:
            p[h][0]+=r['atlas']['L'+h]['nn']-r['atlas']['L20']['nn'];p[h][1]+=2*(r['values']['L'+h]-r['values']['L20'])
    assert pts==fr['train_points']
    B=F(7,10)*sum(r['atlas']['B32']['nn'] for r in tr)-sum(r['atlas']['L20']['nn'] for r in tr);assert B==F(fr['budget']) and fr['feasible']
    cost=sum(F(q)*pts[z][h][0] for z,hh in pol.items() for h,q in hh.items());gain=sum(F(q)*pts[z][h][1] for z,hh in pol.items() for h,q in hh.items());lam=4*F(fr['dual_score_per_row'])
    assert all(sum(map(F,hh.values()))==1 for hh in pol.values())
    assert cost==F(fr['extra'])<=B and gain==F(fr['gain4'])==F(fr['dual_gain4'])==lam*B+sum(max(F(g)-lam*k for k,g in hh.values()) for hh in pts.values()) and lam*(B-cost)==0
    summ=load('SUMMARY.json')
    for split in ('TRAIN','CAL','TEST'):
        for subgroup,seeds in [('all',SEEDS),('fit',FIT)]+[(str(s),(s,)) for s in SEEDS if s!=102 or split!='TRAIN']:
            rr=[r for r in atlas.values() if r['job']['split']==split and r['seed'] in seeds]
            n=len(rr);x=summ['atlas'][split]['by_seed'][subgroup] if subgroup.isdigit() else summ['atlas'][split][subgroup]
            assert n==x['rows']
            for k in ('B32',*(f'L{h}' for h in H)):
                assert x['nn'][k]==sum(r['atlas'][k]['nn'] for r in rr)
                assert x['bad'][k]==sum(r['values'][k]!=r['best_value'] for r in rr)
                assert abs(x['mean_regret'][k]-float(F(sum(r['best_value']-r['values'][k] for r in rr),2*n)))<1e-14
            g=c=F(0);lo=b=0
            for r in rr:
                z=str(sum(1<<i for i,(v,t) in enumerate(zip(r['atlas']['features'][:3],cuts)) if v>t));a=r['atlas'];vv=r['values'];lo+=a['L20']['nn'];b+=a['B32']['nn']
                for h,q in pol.get(z,{'20':'1'}).items():
                    g+=F(q)*F(vv['L'+h]-vv['L20'],2);c+=F(q)*(a['L'+h]['nn']-a['L20']['nn'])
            assert abs(float(g/n)-x['expected_gain'])<1e-14 and abs(float(1-(lo+c)/b)-x['expected_reduction'])<1e-14
    pairs={};costs=C.defaultdict(C.Counter)
    for g in rows('ARENA'):
        j=g['job'];s=g['seed'];color=g['color'];assert j==jobs[j['id']] and j['split'] in ('EARLY','LATE')
        k=j['id'],s,color;assert k not in pairs;pairs[k]=g
        h=j['opening'].copy();ct=costs[j['split']];nn=0
        for r in g['rows']:
            p=1-2*(len(h)%2);q=p==color
            assert r['player']==p and r['ply']==len(h) and r['arm']==('Q41' if q else 'B32')
            val(h,r,32,q,q);h.append(r['move']);nn+=r['nn']
            tag='candidate' if q else 'baseline';ct[tag+'_moves']+=1;ct[tag+'_nn']+=r['nn'];ct[tag+'_cpu']+=r['cpu_ns'];ct[tag+'_wall']+=r['wall_ns']
        b,w=grid.replay(h);assert h==g['history'] and (w or all(b)) and w==g['winner'] and g['score2']==1+color*w and nn==g['nn']
        ct['games']+=1;ct['score2']+=g['score2'];ct['wins']+=g['score2']==2;ct['draws']+=g['score2']==1
    assert len(pairs)==1024
    for split,ct in costs.items():
        x=summ['arena'][split];assert x['games']==ct['games'] and x['wins']==ct['wins'] and x['draws']==ct['draws'] and F(ct['score2'],2*ct['games'])==F(x['score']).limit_denominator(2048)
        for tag in ('candidate','baseline'):
            for k in ('moves','nn'):assert x[tag+'_'+k]==ct[tag+'_'+k]
        for k in ('nn','cpu','wall'):
            if ct['baseline_'+k]:
                red=1-F(ct['candidate_'+k]*ct['baseline_moves'],ct['baseline_'+k]*ct['candidate_moves'])
                assert abs(float(red)-x[k+'_reduction'])<1e-14
    nref=checks=0
    from k41_engine import load_eval
    for s,rs in sorted(sample.items()):
        ev=load_eval(s)
        for r in rs:
            for k in ('B32',*(f'L{h}' for h in H)):
                a,f=scalar(ev,r['job']['opening'],int(k[1:]),k[0]=='L')
                assert all(a[x]==r['atlas'][k][x] for x in a),(s,r['job']['id'],k)
                if k=='L20':assert all(abs(a-b)<1e-12 for a,b in zip(f,r['atlas']['features']))
                checks+=1
        nref+=ev.rows
    # Separate compiled algorithm, with direct coordinate win test, no tactical pruning.
    inp=''.join(f'{i} {a} {b}\n' for (a,b),i in sorted(queries.items(),key=lambda x:x[1]))
    t=time.monotonic();proc=subprocess.run([str(T/'independent')],input=inp,text=True,capture_output=True,check=True,timeout=180)
    result=list(map(json.loads,proc.stdout.splitlines()));assert len(result)==len(queries) and {r['id'] for r in result}==set(queries.values())
    cc=C.Counter();nodes=0;byid={r['id']:r for r in result}
    for r in result:
        i=r['id'];nodes+=r['nodes']
        assert r['complete'],(i,r)
        if i in expected:assert r['values']==expected[i],(i,r,expected[i]);cc['complete_value_vectors_matched']+=1
        for tag in query_tags[i]:cc[tag]+=1
    censor=C.Counter()
    for j,(r,i) in ex.items():
        if not r['exact']['complete']:
            v=byid[i]['values'];censor[max(v.values())]+=1
    # Record independent solutions for censored roots without refitting or reclassifying frozen dataset.
    path=D/'INDEPENDENT.jsonl.gz';assert not path.exists()
    with gzip.open(path,'wt',compresslevel=3) as fd:
        for (a,b),i in sorted(queries.items(),key=lambda x:x[1]):
            fd.write(json.dumps(dict(a=a,b=b,tags=sorted(query_tags[i]),**byid[i]),separators=(',',':'))+'\n')
    native=load_native()
    out=dict(counts=dict(CT),unique_rule_states=len(CACHE),exact_corpus=len(ex),atlas=len(atlas),arena_games=len(pairs),scalar_comparisons=checks,scalar_NN=nref,exact_dual=str(gain),independent=dict(counts=dict(cc),queries=len(queries),nodes=nodes,elapsed_s=time.monotonic()-t,artifact_sha256=sha(path),binary_sha256=sha(T/'independent'),corpus_censored_best_values=dict(censor)),native_identity=native,scope='all raw histories and votes/aggregates, independent scalar PUCT shared evaluator, independent compiled exhaustive minimax all exact vectors and censored boards; no CI recomputation')
    save('AUDIT.json',out);print(json.dumps(out,indent=2),flush=True)
def load_native():
    n=json.loads((D.parent/'quartz_kinetic40_budget_20261126/NATIVE_IDENTITY.json').read_text())
    b=pathlib.Path.home()/'Dropbox/personal_projects/quartz_kinetic40_budget_backups/mcts_demo';assert sha(b)==n['binary_sha256']
    lock=json.loads((D.parent/'quartz_kinetic37_20261126/RAPFI_LOCK.json').read_text())
    for a in lock['files']:assert sha(pathlib.Path(a['path']))==a['sha256']
    return dict(binary_sha256=sha(b),rapfi_assets=len(lock['files']),native_execution='NOT_RUN_K41')
if __name__=='__main__':run()
