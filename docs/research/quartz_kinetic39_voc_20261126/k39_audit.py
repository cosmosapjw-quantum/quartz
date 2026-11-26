"""Independent grid and integer audit, exact aggregate reconstruction and scalar replay."""
import collections as C, fractions, gzip, importlib.util, json, math, random
from k39_plan import D,ROOT,SEEDS,SPLITS,ARMS,SEED,family,load,save,sha
from k39_engine import load_eval
F=fractions.Fraction
spec=importlib.util.spec_from_file_location('grid_audit',D.parent/'quartz_kinetic33_20261126/audit_k33.py')
grid=importlib.util.module_from_spec(spec);spec.loader.exec_module(grid)
def allrows(mode):
    lk=load(mode+'_LOCK.json')
    for name,expected in [('PLAN.json',lk['plan_sha256']),('k39_engine.py',lk['engine_sha256']),('k39_experiment.py',lk['experiment_sha256'])]:
        assert sha(D/name)==expected
    for fi in lk['files']:
        assert sha(D/fi['file'])==fi['sha256']
        with gzip.open(D/fi['file'],'rt') as f:
            yield from map(json.loads,f)
CACHE={};CT=C.Counter()
def gate(b,p):
    key=(tuple(b),p)
    if key not in CACHE:CACHE[key]=grid.cert(b,p)
    return CACHE[key]
def validate(h,r,arm=None,mask=None):
    b,w=grid.replay(h);p=1-2*(len(h)%2)
    assert not w and b[r['move']]==0
    if arm is not None:assert r['player']==p and r['ply']==len(h) and r['arm']==arm
    z=gate(b,p);CT['rule_checks']+=1
    if r['kind']=='CERTIFICATE':
        assert z['move']==r['move'] and r['proof']==z['kind'] and r['nn']==r['visits']==0 and not r['votes'] and r['distance'] is None
        CT['certificates']+=1
    else:
        assert z['move'] is None and r['proof']=='UNRESOLVED'
        legal=[i for i,v in enumerate(b) if not v];n=dict(r['votes'])
        assert [a for a,v in r['votes']]==legal and all(type(v) is int and v>=0 for v in n.values())
        leader=min(legal,key=lambda a:(-n[a],a))
        d=min((n[leader]-n[a]+int(a>leader) for a in legal if a!=leader),default=10**9)
        assert r['move']==leader and r['distance']==d and r['prior_best'] in legal
        assert sum(n.values())==r['visits']<=r['cap'] and r['nn']==1+r['visits']-r['terminal_leaves']
        assert 0<=r['terminal_leaves']<=r['visits']
        if r['kind']=='VOTE_LOCK':
            assert r['lock'] and d>r['cap']-r['visits'] and r['visits']<r['cap'];CT['locks']+=1
        else:assert r['kind']=='CAP' and r['visits']==r['cap']
        CT['legal_edges']+=len(legal)
    assert r['wall_ns']>=r['rule_ns']>=0 and r['cpu_ns']>=0
    if arm=='A':
        assert r['mask']==mask and r['lock']
        if r['kind']=='CERTIFICATE':assert r['bin'] is None and not r['extended'] and r['low_nn']==0
        else:
            x=int(r['low_distance']<=3)+2*int(r['low_move']!=r['low_prior'])
            assert r['bin']==x and r['extended']==bool(mask>>x&1) and r['cap']==(32 if r['extended'] else 20)
            assert r['low_nn']<=r['nn'] and r['low_visits']<=r['visits']
            assert r['low_visits']<=20 and (r['low_visits']==20 or r['low_distance']>20-r['low_visits'])
            if not r['extended']:assert r['move']==r['low_move'] and r['nn']==r['low_nn'] and r['distance']==r['low_distance']
    elif arm is not None:assert r['cap']==int(arm[1:]) and r['lock']==(arm[0]=='L')
def game(g,arm,base,mask=0):
    h=g['start'].copy();player=g['player'];nn=0
    for r in g['rows']:
        p=1-2*(len(h)%2);who=arm if p==player else base
        validate(h,r,who,mask);h.append(r['move']);nn+=r['nn']
    b,w=grid.replay(h)
    assert h==g['history'] and (w or all(b)) and g['winner']==w and g['score2']==1+player*w and nn==g['nn']
    return nn
def scalar(ev,h,H,lock):
    b,w=grid.replay(h);p=1-2*(len(h)%2);assert not w and gate(b,p)['move'] is None
    tree={};start=ev.rows;terminal=0
    def expand(board,p,path):
        legal=[i for i,v in enumerate(board) if not v]
        a=sum(1<<i for i,v in enumerate(board) if v==1);c=sum(1<<i for i,v in enumerate(board) if v==-1)
        pr,v=ev(a,c,p,legal);tree[path]={m:[0,0.,float(q)] for m,q in zip(legal,pr)}
        return v
    def step(board,p,path):
        nonlocal terminal
        if path not in tree:
            if path and grid.won(board,path[-1],-p):terminal+=1;return -1.
            if not any(v==0 for v in board):terminal+=1;return 0.
            return expand(board,p,path)
        e=tree[path];s=math.sqrt(max(1,sum(v[0] for v in e.values())))
        def key(m):
            n,w,prior=e[m];return ((w/n if n else 0.)+2*prior*s/(n+1),-m)
        m=max(e,key=key);board[m]=p
        v=-step(board,-p,path+(m,));board[m]=0;e[m][0]+=1;e[m][1]+=v
        return v
    expand(b,p,());t=0;e=tree[()]
    while t<H:
        a=max(e,key=lambda m:(e[m][0],-m))
        if lock and all(e[a][0]>e[i][0]+H-t or (e[a][0]==e[i][0]+H-t and a<i) for i in e if i!=a):break
        step(b,p,());t+=1
    a=max(e,key=lambda m:(e[m][0],-m))
    return dict(move=a,votes=[[i,v[0]] for i,v in e.items()],visits=t,nn=ev.rows-start,terminal_leaves=terminal,prior_best=max(e,key=lambda i:(e[i][2],-i)))
def run():
    pl=load('PLAN.json');jobs={j['id']:j for j in pl['jobs']};assert len(jobs)==832
    assert len({family(j['opening']) for j in jobs.values()})==len(jobs)
    for p in pl['prior_locks']:
        f=ROOT/p['path'];assert sha(f)==p['sha256']
        old=load_prior(f);assert not ({family(j['opening']) for j in old['jobs']}&{family(j['opening']) for j in jobs.values()})
    sc=sorted(allrows('SCREEN'),key=lambda r:r['job']['id']);assert len(sc)==640
    screen={};admitted=[];duplicates=[];seen=set();screen_nn=0;sample=C.defaultdict(list)
    for r in sc:
        j=r['job'];assert j==jobs[j['id']] and j['split']!='ARENA'
        h=j['opening'].copy();n=0
        for x in r['source']:
            p=1-2*(len(h)%2);validate(h,x,'L32' if p==1 else 'B32');h.append(x['move']);n+=x['nn']
        b,w=grid.replay(h);assert r['history']==h
        if w or all(b):assert r['status']=='TERMINAL'
        else:
            assert len(h)==j['target']
            z=gate(b,1-2*(len(h)%2));assert (r['status']=='ROOT')==(z['move'] is None)
        if r['status']=='ROOT':
            a=r['atlas'];assert a is not None
            for k in ('B20','L20','B32','L32'):
                validate(h,a[k]);assert a[k]['cap']==int(k[1:]) and a[k]['lock']==(k[0]=='L')
            lo,hi=a['L20'],a['L32']
            assert a['B20']['move']==lo['move'] and a['B32']['move']==hi['move']
            assert lo['nn']<=hi['nn']<=a['B32']['nn'] and lo['nn']<=a['B20']['nn']
            assert a['bin']==int(lo['distance']<=3)+2*int(lo['move']!=lo['prior_best'])
            assert a['actual_nn']==a['B20']['nn']+a['B32']['nn']+hi['nn']
            n+=a['actual_nn'];f=family(h)
            if f in seen:duplicates.append(j['id'])
            else:
                seen.add(f);admitted.append(j['id']);screen[j['id']]=r
                if len(sample[j['seed']])<6:sample[j['seed']].append(r)
        else:assert r['atlas'] is None
        assert n==r['actual_nn'];screen_nn+=n
    fr=load('FRAME.json');assert fr['screen']==sha(D/'SCREEN_LOCK.json') and fr['ids']==admitted and fr['duplicates']==duplicates
    assert screen_nn==load('SCREEN_LOCK.json')['counts']['nn']
    all_labels=[];label_nn={}
    for mode in ('TRAIN_CAL','TEST'):
        ls=list(allrows(mode));label_nn[mode]=0
        expected={i for i in admitted if jobs[i]['split'] in ({'TRAIN','CAL'} if mode=='TRAIN_CAL' else {'TEST'})}
        assert len(ls)==len(expected) and {r['id'] for r in ls}==expected
        for r in ls:
            x=screen[r['id']];a=x['atlas'];h=x['history'];lo=a['L20']['move'];hi=a['L32']['move']
            assert r['seed']==x['job']['seed'] and r['split']==x['job']['split'] and r['bin']==a['bin'] and r['flip']==(lo!=hi)
            assert r['cost_delta']==a['L32']['nn']-a['L20']['nn']>=0
            if lo==hi:assert not r['labels'] and r['delta4']==0 and r['actual_nn']==0
            else:
                assert {(g['action'],g['refcap']) for g in r['labels']}=={(m,c) for m in (lo,hi) for c in (32,64)} and len(r['labels'])==4
                n=0;u=C.Counter()
                for g in r['labels']:
                    c=g['refcap'];assert g['start']==h+[g['action']] and g['player']==1-2*(len(h)%2)
                    n+=game(g,f'L{c}',f'B{c}');u[g['action']]+=g['score2'];CT['reference_games']+=1
                assert n==r['actual_nn'] and r['delta4']==u[hi]-u[lo]
                label_nn[mode]+=n
        assert label_nn[mode]==load(mode+'_LOCK.json')['counts']['nn']
        all_labels+=ls
    frozen=load('FROZEN.json');mask=frozen['mask'];assert frozen['train_cal_lock']==sha(D/'TRAIN_CAL_LOCK.json') and frozen['frame_lock']==sha(D/'FRAME.json')
    tr=[r for r in all_labels if r['split']=='TRAIN']
    scores=[]
    for m in range(16):
        z=[r for r in tr if m>>r['bin']&1];k=sum(r['cost_delta'] for r in z)
        net=sum(F(r['delta4'],4)-F(r['cost_delta'],500) for r in z)
        scores.append((net,-k,-m))
        assert frozen['train_table'][m]['net_units']==net*500
    assert mask==-max(scores)[2]
    pairs={};counts=C.defaultdict(C.Counter);n_arena=0
    for g in allrows('ARENA'):
        j=g['job'];assert j==jobs[j['id']] and j['split']=='ARENA' and g['start']==j['opening'] and g['match'] in ARMS
        k=j['id'],g['match'],g['player'];assert k not in pairs;pairs[k]=g
        arm,base=g['match'].split('_');n_arena+=game(g,arm,base,mask)
        c=counts[g['match']];c['score2']+=g['score2'];c['games']+=1
        for r in g['rows']:
            tag='candidate' if r['player']==g['player'] else 'baseline'
            c[tag+'_moves']+=1;c[tag+'_NN']+=r['nn'];c[tag+'_cpu_ns']+=r['cpu_ns'];c[tag+'_wall_ns']+=r['wall_ns']
    assert len(pairs)==1920 and n_arena==load('ARENA_LOCK.json')['counts']['nn']
    for j in (x for x in jobs.values() if x['split']=='ARENA'):
        for H in (20,32):
            m=f'L{H}_B{H}';a,b=[pairs[j['id'],m,p] for p in (1,-1)]
            assert a['history']==b['history'] and a['score2']+b['score2']==2;CT['same_horizon_pairs']+=1
            for x,y in zip(a['rows'],b['rows']):
                l,k=(x,y) if x['arm'].startswith('L') else (y,x);assert l['move']==k['move'] and l['nn']<=k['nn'];CT['same_horizon_matched_moves']+=1
        for p in (1,-1):
            a=pairs[j['id'],'L20_B32',p];b=pairs[j['id'],'L32_B20',-p]
            assert a['history']==b['history'] and a['score2']+b['score2']==2;CT['reverse_cross_horizon_games']+=1
            if mask in (0,15):
                q=pairs[j['id'], 'L20_B32' if mask==0 else 'L32_B32',p]
                aa=pairs[j['id'],'A_B32',p];assert aa['history']==q['history'] and aa['score2']==q['score2']
    summ=load('SUMMARY.json')
    for m,c in counts.items():
        a=summ['arena'][m];assert a['games']==c['games'] and F(c['score2'],2*c['games'])==F(a['score']).limit_denominator(768)
        for tag in ('candidate','baseline'):
            for k in ('moves','NN'):assert a[tag+'_'+k]==c[tag+'_'+k]
        for k,v in [('NN','NN_reduction'),('cpu_ns','CPU_reduction'),('wall_ns','wall_reduction')]:
            r=1-F(c['candidate_'+k]*c['baseline_moves'],c['baseline_'+k]*c['candidate_moves'])
            assert abs(float(r)-a[v])<1e-14
    nref=checks=0
    for s,rs in sorted(sample.items()):
        ev=load_eval(s)
        for r in rs:
            for k in ('B20','L20','B32','L32'):
                q=scalar(ev,r['history'],int(k[1:]),k[0]=='L')
                assert all(q[f]==r['atlas'][k][f] for f in q),(s,r['job']['id'],k,q,r['atlas'][k])
                checks+=1
        nref+=ev.rows
    out=dict(counts=dict(CT),unique_rule_states=len(CACHE),screen_nn=screen_nn,label_nn=label_nn,arena_nn=n_arena,scalar_comparisons=checks,scalar_NN=nref,scalar_roots=checks//4,mask=mask,exact_summary_reconstructed=True,scope='all archived game histories/rule gates/vote snapshots, exact costs and signed labels, exact table enumeration and bounded independent scalar PUCT using same trained evaluator; no independent CI or generalization claim')
    save('AUDIT.json',out);print(json.dumps(out,indent=2),flush=True)
def load_prior(f):return json.loads(f.read_text())
if __name__=='__main__':run()
