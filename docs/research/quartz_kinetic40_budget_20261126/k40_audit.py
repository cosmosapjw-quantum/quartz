"""Independent raw-grid and exact integer/rational audit, no experiment-code imports."""
import collections as C, fractions, gzip, importlib.util,json,math,pathlib,hashlib
from k40_plan import D,ROOT,SEEDS,FIT,H,family,load,save,sha
from k40_engine import load_eval
F=fractions.Fraction
spec=importlib.util.spec_from_file_location('grid_audit',D.parent/'quartz_kinetic33_20261126/audit_k33.py')
grid=importlib.util.module_from_spec(spec);spec.loader.exec_module(grid)
CACHE={};CT=C.Counter()
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def rows(mode):
    lk=load(mode+'_LOCK.json')
    for key,n in [('plan','PLAN.json'),('engine','k40_engine.py'),('experiment','k40_experiment.py')]:assert sha(D/n)==lk[key]
    for fi in lk['files']:
        assert sha(D/fi['file'])==fi['sha256'];rr=list(raw(D/fi['file']))
        assert sum(r['nn'] for r in rr)==fi['counts']['nn']
        if mode in ('ATLAS','FIT_LABEL','TEST_LABEL','ARENA'):assert all(r['seed']==fi['seed'] for r in rr)
        yield from rr
def gate(b,p):
    key=(tuple(b),p)
    if key not in CACHE:CACHE[key]=grid.cert(b,p)
    return CACHE[key]
def draw(h,p,z):
    a,b=0,0
    for i,m in enumerate(h):
        if i%2:b|=1<<m
        else:a|=1<<m
    n=int(hashlib.sha256(f'K40-2026112640:{a}:{b}:{len(h)%2}'.encode()).hexdigest(),16)
    u=F(n,1<<256);q=p.get(str(z),{'20':'1'});s=F(0)
    for cap,r in sorted(q.items(),key=lambda x:int(x[0])):
        s+=F(r)
        if u<s:return int(cap)
    raise AssertionError
def val(h,r,cap=None,lock=None):
    b,w=grid.replay(h);p=1-2*(len(h)%2);assert not w and b[r['move']]==0
    if cap is not None:assert r['cap']==cap and r['lock']==lock
    g=gate(b,p);CT['rule_checks']+=1
    if r['kind']=='CERTIFICATE':
        assert g['move']==r['move'] and g['kind']==r['proof'] and r['nn']==r['visits']==0 and not r['votes'];CT['certificates']+=1
    else:
        assert g['move'] is None and r['proof']=='UNRESOLVED'
        legal=[i for i,v in enumerate(b) if not v];n=dict(r['votes'])
        assert list(n)==legal and all(type(v)==int and v>=0 for v in n.values())
        a=min(legal,key=lambda i:(-n[i],i));d=min((n[a]-n[i]+int(i>a) for i in legal if i!=a),default=10**9)
        assert r['move']==a and r['distance']==d and r['prior_best'] in legal
        assert sum(n.values())==r['visits']<=r['cap'] and r['nn']==1+r['visits']-r['terminal_leaves']
        assert 0<=r['terminal_leaves']<=r['visits']
        if r['kind']=='VOTE_LOCK':
            assert r['lock'] and d>r['cap']-r['visits'] and r['visits']<r['cap'];CT['locks']+=1
        else:assert r['kind']=='CAP' and r['visits']==r['cap']
        CT['legal_edges']+=len(legal)
    assert r['wall_ns']>=r['rule_ns']>=0 and r['cpu_ns']>=0
def play(g,arm,base,p):
    h=g['start'].copy();nn=0
    for r in g['rows']:
        side=1-2*(len(h)%2);who=arm if side==g['player'] else base
        assert r['player']==side and r['ply']==len(h) and r['arm']==who
        if who=='A':
            if r['kind']=='CERTIFICATE':assert r['bin'] is None and r['low_nn']==0;cap=20
            else:
                z=int(r['entropy']>=.8)+2*int(r['low_move']!=r['low_prior'])+4*int(abs(r['root_value'])>=.5)
                assert r['bin']==z;cap=draw(h,p,z);assert r['low_nn']<=r['nn'] and r['low_visits']<=20
                if cap==20:assert r['nn']==r['low_nn'] and r['move']==r['low_move']
            assert r['selected_cap']==cap;val(h,r,cap,True)
        else:val(h,r,int(who[1:]),who[0]=='L')
        h.append(r['move']);nn+=r['nn']
    b,w=grid.replay(h);assert h==g['history'] and (w or all(b)) and w==g['winner'] and g['score2']==1+g['player']*w and nn==g['nn']
    return nn
def scalar(ev,h,H,lock):
    b,w=grid.replay(h);p=1-2*(len(h)%2);assert not w and gate(b,p)['move'] is None
    tree={};start=ev.rows;terminal=0;rv=None;ent=None
    def expand(board,p,path):
        nonlocal rv,ent
        legal=[i for i,v in enumerate(board) if not v]
        a=sum(1<<i for i,v in enumerate(board) if v==1);c=sum(1<<i for i,v in enumerate(board) if v==-1)
        pr,v=ev(a,c,p,legal);tree[path]={m:[0,0.,float(q)] for m,q in zip(legal,pr)}
        if not path:
            rv=v;ent=-sum(float(q)*math.log(float(q)) for q in pr if q>0)/math.log(len(pr)) if len(pr)>1 else 0
        return v
    def step(board,p,path):
        nonlocal terminal
        if path not in tree:
            if path and grid.won(board,path[-1],-p):terminal+=1;return -1.
            if not any(v==0 for v in board):terminal+=1;return 0.
            return expand(board,p,path)
        e=tree[path];s=math.sqrt(max(1,sum(v[0] for v in e.values())))
        def key(m):
            n,w,pr=e[m];return ((w/n if n else 0.)+2*pr*s/(n+1),-m)
        m=max(e,key=key);board[m]=p
        v=-step(board,-p,path+(m,));board[m]=0;e[m][0]+=1;e[m][1]+=v
        return v
    expand(b,p,());t=0;e=tree[()]
    while t<H:
        a=max(e,key=lambda m:(e[m][0],-m))
        if lock and all(e[a][0]>e[i][0]+H-t or (e[a][0]==e[i][0]+H-t and a<i) for i in e if i!=a):break
        step(b,p,());t+=1
    a=max(e,key=lambda m:(e[m][0],-m))
    return dict(move=a,votes=[[i,v[0]] for i,v in e.items()],visits=t,nn=ev.rows-start,terminal_leaves=terminal,prior_best=max(e,key=lambda i:(e[i][2],-i))),rv,ent
def run():
    pl=load('PLAN.json');jobs={j['id']:j for j in pl['jobs']};assert len(jobs)==304 and len({family(j['opening']) for j in jobs.values()})==304
    for p in pl['prior_locks']:
        f=ROOT/p['path'];assert sha(f)==p['sha256'];old=json.loads(f.read_text())
        assert not ({family(j['opening']) for j in old['jobs']}&{family(j['opening']) for j in jobs.values()})
    src={};ad=[];seen=set();dups=[]
    for r in sorted(rows('SOURCE'),key=lambda x:x['job']['id']):
        j=r['job'];assert j==jobs[j['id']] and j['split']!='ARENA' and (j['split']=='TEST' or j['source'] in FIT)
        h=j['opening'].copy();nn=0
        for x in r['source']:
            p=1-2*(len(h)%2);assert x['player']==p and x['ply']==len(h)
            val(h,x,32,p==1);nn+=x['nn'];h.append(x['move'])
        b,w=grid.replay(h);assert r['history']==h and nn==r['nn']
        if w or all(b):assert r['status']=='TERMINAL'
        else:
            assert len(h)==j['target'];g=gate(b,1-2*(len(h)%2))
            assert r['status']==('ROOT' if g['move'] is None else 'GATED')
        src[j['id']]=r
        if r['status']=='ROOT':
            f=family(h)
            if f in seen:dups.append(j['id'])
            else:seen.add(f);ad.append(j['id'])
    fr=load('FRAME.json');assert fr['ids']==ad and fr['duplicates']==dups and fr['source_lock']==sha(D/'SOURCE_LOCK.json')
    assert len(src)==240
    atlas={};sample=C.defaultdict(list)
    for r in sorted(rows('ATLAS'),key=lambda x:(x['seed'],x['id'])):
        i,s=r['id'],r['seed'];assert (i,s) not in atlas and i in ad and r['history']==src[i]['history'] and r['split']==jobs[i]['split']
        a=r['atlas'];h=r['history']
        for k in ('B32','L20','L26','L32'):val(h,a[k],int(k[1:]),k[0]=='L')
        lo,mid,hi=a['L20'],a['L26'],a['L32']
        assert lo['nn']<=mid['nn']<=hi['nn']<=a['B32']['nn'] and hi['move']==a['B32']['move'] and a['actual_nn']==r['nn']==a['B32']['nn']+hi['nn']
        assert a['bin']==int(a['entropy']>=.8)+2*int(lo['move']!=lo['prior_best'])+4*int(abs(a['root_value'])>=.5)
        atlas[i,s]=r
        if len(sample[s])<4:sample[s].append(r)
    assert set(atlas)=={(i,s) for i in ad for s in SEEDS}
    ls={};refgames=0
    for mode in ('FIT_LABEL','TEST_LABEL'):
        for r in rows(mode):
            i,s=r['id'],r['seed'];assert (i,s) in atlas and (i,s) not in ls
            assert (mode=='FIT_LABEL')==(r['split'] in ('TRAIN','CAL') and s in FIT)
            x=atlas[i,s];a=x['atlas'];h=r['history'];assert x['history']==h and x['split']==r['split']
            assert r['bin']==a['bin'] and r['entropy']==a['entropy'] and r['root_value']==a['root_value']
            acts={str(cap):a[f'L{cap}']['move'] for cap in H};assert r['actions']==acts
            assert r['low_nn']==a['L20']['nn'] and r['base_nn']==a['B32']['nn']
            assert r['cost_delta']=={str(cap):a[f'L{cap}']['nn']-a['L20']['nn'] for cap in H}
            if len(set(acts.values()))==1:assert not r['labels'] and not any(r['delta4'].values()) and r['nn']==0
            else:
                u=C.Counter();n=0
                assert {(g['action'],g['refcap']) for g in r['labels']}=={(m,c) for m in acts.values() for c in (32,64)} and len(r['labels'])==2*len(set(acts.values()))
                for g in r['labels']:
                    c=g['refcap'];assert g['start']==h+[g['action']] and g['player']==1-2*(len(h)%2)
                    n+=play(g,f'L{c}',f'B{c}',{});u[g['action']]+=g['score2'];refgames+=1
                assert n==r['nn'] and r['delta4']=={cap:u[m]-u[acts['20']] for cap,m in acts.items()}
            ls[i,s]=r
    assert set(ls)==set(atlas)
    frozen=load('FROZEN.json');policy=frozen['policy'];tr=[r for r in ls.values() if r['split']=='TRAIN' and r['seed'] in FIT]
    B=F(7,10)*sum(r['base_nn'] for r in tr)-sum(r['low_nn'] for r in tr);assert B==F(frozen['budget'])
    p={}
    for r in tr:
        z=str(r['bin']);v=p.setdefault(z,{str(h):[0,0] for h in H})
        for cap in v:v[cap][0]+=r['cost_delta'][cap];v[cap][1]+=r['delta4'][cap]
    assert p==frozen['train_points']
    assert B>=0 and frozen['feasible']
    gain=sum(F(q)*p[z][h][1] for z,hh in policy.items() for h,q in hh.items())
    cost=sum(F(q)*p[z][h][0] for z,hh in policy.items() for h,q in hh.items())
    lam=4*F(frozen['dual_score_per_row'])
    dual=lam*B+sum(max(F(g)-lam*k for k,g in hh.values()) for hh in p.values())
    assert gain==F(frozen['gain4'])==F(frozen['dual_gain4'])==dual and cost==F(frozen['extra'])<=B
    assert all(sum(map(F,hh.values()))==1 for hh in policy.values())
    assert lam*(B-cost)==0
    pairs={};counts=C.defaultdict(C.Counter)
    for g in rows('ARENA'):
        j=g['job'];assert j==jobs[j['id']] and j['split']=='ARENA' and g['start']==j['opening']
        k=(j['id'],g['seed'],g['match'],g['player']);assert k not in pairs;pairs[k]=g
        arm=g['match'].split('_')[0];play(g,arm,'B32',policy)
        c=counts[g['match']];c['games']+=1;c['score2']+=g['score2'];c['wins']+=g['score2']==2;c['losses']+=g['score2']==0
        for r in g['rows']:
            tag='candidate' if r['player']==g['player'] else 'baseline'
            c[tag+'_moves']+=1;c[tag+'_NN']+=r['nn'];c[tag+'_cpu']+=r['cpu_ns']
    assert len(pairs)==1024
    alias=all(hh=={'20':'1'} for hh in policy.values())
    if alias:
        for (i,s,m,p),g in pairs.items():
            if m=='A_B32':
                z=pairs[i,s,'L20_B32',p];assert g['history']==z['history'] and g['score2']==z['score2'] and g['nn']==z['nn'];CT['alias_games']+=1
    summ=load('SUMMARY.json')
    for m,c in counts.items():
        a=summ['arena'][m];assert a['games']==c['games'] and a['wins']==c['wins'] and a['losses']==c['losses']
        assert F(c['score2'],2*c['games'])==F(a['score']).limit_denominator(1024)
        for tag in ('candidate','baseline'):
            for k in ('moves','NN'):assert a[tag+'_'+k]==c[tag+'_'+k]
        for k,v in [('NN','NN_reduction'),('cpu','CPU_reduction')]:
            q=1-F(c['candidate_'+k]*c['baseline_moves'],c['baseline_'+k]*c['candidate_moves'])
            assert abs(float(q)-a[v])<1e-14
    checks=nref=0
    for s,rs in sorted(sample.items()):
        ev=load_eval(s)
        for r in rs:
            for k in ('B32','L20','L26','L32'):
                q,v,e=scalar(ev,r['history'],int(k[1:]),k[0]=='L')
                assert all(q[f]==r['atlas'][k][f] for f in q),(s,r['id'],k,q,r['atlas'][k])
                assert abs(v-r['atlas']['root_value'])<1e-12 and abs(e-r['atlas']['entropy'])<1e-12
                checks+=1
        nref+=ev.rows
    # Native independent terminal histories and count aggregation (without running old endpoint).
    rp=load('RAPFI_PLAN.json');rs=load('RAPFI_RESULT.json')
    assert sha(D/'RAPFI_PLAN.json')==rs['plan_sha256']
    prior=json.loads((D.parent/'quartz_kinetic37_20261126/PLAN.json').read_text())
    assert not ({j['family'] for j in rp['jobs']}&{j['family'] for j in prior['jobs']})
    rg=C.defaultdict(list);ct=C.Counter()
    def win15(h):
        b=[0]*225;win=0
        for i,m in enumerate(h):
            assert 0<=m<225 and not b[m] and not win
            p=1-2*(i%2);b[m]=p;r,c=divmod(m,15)
            for dr,dc in ((1,0),(0,1),(1,1),(1,-1)):
                n=1
                for sg in (-1,1):
                    x,y=r+sg*dr,c+sg*dc
                    while 0<=x<15 and 0<=y<15 and b[x*15+y]==p:
                        n+=1;x+=sg*dr;y+=sg*dc
                if n>=5:win=p
        return win
    for fi in rs['files']:
        assert sha(D/fi['file'])==fi['sha256']
        for g in raw(D/fi['file']):
            h=g['job']['opening'].copy()
            for a in g['rows']:
                assert a['history']==h and a['best'] not in h and a['player']==1-2*(len(h)%2) and a['locked']==(a['player']==g['color'])
                assert a['NN_calls']<=128 and a['visits']<=128
                h.append(a['best']);ct['nn']+=a['NN_calls'];ct['moves']+=1;ct['checks']+=a['certificate_checks']
            assert h==g['history'] and win15(h)==g['winner'] and g['score']==(1+g['color']*g['winner'])/2
            rg[g['job']['id']].append(g)
    assert len(rg)==64 and all(len(v)==2 and v[0]['history']==v[1]['history'] for v in rg.values())
    assert ct['nn']==rs['counts']['nn'] and ct['moves']==rs['counts']['moves'] and ct['checks']==rs['counts']['certificate_checks']
    out=dict(counts=dict(CT),unique_rule_states=len(CACHE),source_families=len(src),admitted_families=len(ad),atlas_records=len(atlas),reference_games=refgames,arena_games=len(pairs),scalar_comparisons=checks,scalar_NN=nref,exact_dual=str(dual),native=dict(ct),native_pairs=len(rg),scope='independent rule replay, scalar PUCT and entropy/value replay with shared evaluator, exact rational dual and game counts, native independent terminal replay; bootstrap CIs not independently recomputed')
    save('AUDIT.json',out);print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':run()
