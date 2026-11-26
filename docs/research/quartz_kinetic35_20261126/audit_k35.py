"""Independent exact proof, resource, paired-game and chronological receipt audit."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter,defaultdict
from functools import lru_cache
import hashlib,json,gzip,random,statistics as S,math,sys,struct
D=Path(__file__).resolve().parent;ROOT=D.parents[2];R=D.parent
sys.path.insert(0,str(R/'quartz_tactical08_20261125'))
from rules import replay,win_at,orbit_board,LINES
def read(n):return json.loads((D/n).read_text())
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rows(p):
    with gzip.open(p,'rt') as f:
        for l in f:yield json.loads(l)
def near(a,b):assert math.isclose(a,b,abs_tol=1e-7,rel_tol=1e-9),(a,b)
def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
def bits(x):
    while x:
        a=x&-x;yield a.bit_length()-1;x^=a
ALL=(1<<225)-1
MASKS=[sum(1<<i for i in l[k:k+5]) for l in LINES for k in range(len(l)-4)]
assert len(MASKS)==572 and len(set(MASKS))==572
@lru_cache(maxsize=2048)
def proof(b,p):
    x=1 if p==1 else 2
    a=sum(1<<i for i,v in enumerate(b) if v==x);c=sum(1<<i for i,v in enumerate(b) if v==3-x);empty=ALL^(a|c)
    own=opp=0;adj=defaultdict(int);npair=0
    for w in MASKS:
        e=w&empty
        if e.bit_count()==1:
            if not w&c:own|=e
            if not w&a:opp|=e
        if e.bit_count()==2 and not w&c:
            u,v=list(bits(e));adj[u]|=1<<v;adj[v]|=1<<u;npair+=1
    av=list(bits(own));cv=list(bits(opp));legal=list(bits(empty))
    assert av==[i for i in legal if win_at(b,i,x,'freestyle')]
    assert cv==[i for i in legal if win_at(b,i,3-x,'freestyle')]
    bounds=[];wins=[];replies=0
    for m in legal:
        rep=None;ts=[]
        if own>>m&1:l=u=1;k='WIN1'
        elif len(legal)==1:l=u=0;k='DRAW'
        elif opp&~(1<<m):l=u=-1;k='LOSE2';rep=next(bits(opp&~(1<<m)))
        else:
            ts=list(bits((own&~(1<<m))|adj[m]))
            if len(ts)>=2:
                l=u=1;k='WIN3';wins.append(m)
                # Independently check resulting state and the entire opponent-reply set.
                bb=list(b);bb[m]=x
                assert not any(win_at(bb,j,3-x,'freestyle') for j in legal if j!=m)
                assert all(win_at(bb,j,x,'freestyle') for j in ts)
                for r in legal:
                    if r==m:continue
                    bb[r]=3-x;t=next(j for j in ts if j!=r)
                    assert win_at(bb,t,x,'freestyle')
                    bb[r]=0;replies+=1
            else:l,u=-1,1;k='UNKNOWN'
        bounds.append([m,l,u,k,rep,ts])
    old=av[0] if av else cv[0] if len(cv)==1 else legal[0] if len(legal)==1 else None
    gate=av[0] if av else wins[0] if wins else old
    kind='WIN1' if av else 'WIN3' if wins else 'SOLE' if len(legal)==1 else 'BLOCK' if old is not None else 'UNRESOLVED'
    return dict(own=av,opponent=cv,win3=wins,actions=bounds,gate=gate,kind=kind,old_guard=old,windows=572,pair_windows=npair),replies
def root_dist(r):
    n={e['move']:e['n'] for e in r['edges']};ca=[m for m,v in r['candidates']]
    assert [e['move'] for e in r['edges']]==ca[:len(n)] and sum(n.values())==r['root_n']
    d=[n.get(m,0) for m in ca];L=max(range(len(ca)),key=lambda i:(d[i],i))
    assert ca[L]==r['best']
    return min((d[L]-d[i]+int(i<L) for i in range(len(d)) if i!=L),default=None)
def budget(t,cap,target):
    assert t['cap']==cap and t['target']==target and t['used_nn']+t['unused']==target
    assert 0<t['visits']<=cap and 0<=t['used_nn']<=target
    assert t['cause']==('NN_TARGET' if t['unused']==0 else 'VISIT_CAP')
    if t['unused']:assert t['visits']==cap
def audit_native(a,h,ct,trace=False):
    b,p,w=replay(h,'freestyle');assert w is None and a['history']==h and a['player']==p
    assert b[a['best']]==0
    z,replies=proof(tuple(b),p);assert a['proof']==z,(h,a['proof']['gate'],z['gate'])
    ct['native_requests']+=1;ct['bounds']+=len(z['actions']);ct['win3_actions']+=len(z['win3']);ct['win3_replies_logical']+=replies
    ct['windows']+=572;ct['NN']+=a['NN_calls'];ct['visits']+=a['visits'];ct[a['cause']]+=1
    if z['gate'] is not None:
        assert a['cause']==z['kind'] and a['best']==z['gate'] and a['NN_calls']==a['visits']==0 and a['hooks'] is None
        ct['zero_gate']+=1;ct['new_gate']+=int(z['old_guard'] is None);return
    cap=a['cap'];assert cap in (80,128) and a['threads'] in (1,4)
    ini=a['initial'];mid=a['prefix']['result'];end=a['result'];pre=a['prefix']['hooks'];eh=a['hooks']
    legal=[i for i,v in enumerate(b) if not v];ca=ini['candidates']
    assert set(m for m,v in ca)==set(legal) and len(ca)==len(legal)
    assert ini['root_n']==0 and ini['evaluator_calls']==1 and ca==mid['candidates']==end['candidates']
    ev=a['root_eval'];assert ev['board']==''.join(map(str,b)) and ev['player']==p
    assert abs(ev['win']+ev['draw']+ev['loss']-1)<1e-6;near(ev['win']-ev['loss'],ev['value']);assert abs(sum(ev['policy'])-1)<1e-5
    assert all(abs(ev['policy'][m]-v)<1e-7 for m,v in ca) and all(ev['policy'][m]==0 for m in range(225) if b[m])
    t=a['prefix']['budget'];budget(t,16,15);assert t['visits']==mid['root_n'] and t['used_nn']==mid['evaluator_calls']-1
    assert pre['hooks']['base_evals']==t['used_nn'] and pre['ledger']['iterations']==mid['root_n']
    n=mid['root_n'];nn=mid['evaluator_calls'];d=root_dist(mid)
    for v in a['waves']:
        assert (v['before_n'],v['before_nn'],v['before_d'])==(n,nn,d)
        if a['locked']:assert d is not None and d<=cap-n
        t=v['budget'];budget(t,min(cap-n,8),cap-nn);n+=t['visits'];nn+=t['used_nn'];d=v['after_d']
        assert (n,nn)==(v['after_n'],v['after_nn']);ct['waves']+=1
    assert (n,nn)==(end['root_n'],end['evaluator_calls'])==(a['visits'],a['NN_calls'])
    assert n<=cap and nn<=cap and a['best']==end['best'] and root_dist(end)==d
    if a['cause']=='VOTE_LOCK':assert a['locked'] and (d is None or d>cap-n)
    elif a['cause']=='NN_TARGET':assert nn==cap
    else:assert a['cause']=='VISIT_CAP' and n==cap
    l=eh['ledger'];h0=eh['hooks'];assert l['iterations']==n and l['eligible']==l['novel']+l['blocked_receipt'] and l['transferred']==l['receipt_items'] and l['repeated_delivery']==0
    assert h0['base_evals']+h0['extra_evals']==nn-mid['evaluator_calls'] and h0['base_evals']+h0['terminal_leaves']==n-mid['root_n']
    assert eh['peak']<=a['threads'];ct['peak4']+=int(eh['peak']==4)
    assert all(v['vl']==0 and v['vvalue']==0 for v in end['edges'])
    mask=a['selected_mask']
    assert a['bin']==int(abs(ev['value'])>=.5)+2*int(root_dist(mid) is None or root_dist(mid)>=4)
    assert mask==(a['request_mask'] if a['table'] is None else a['table'][a['bin']])
    assert (a['request_mask'] is None)!=(a['table'] is None)
    for bit,key in ((1,'select_nodes'),(2,'width_nodes'),(4,'extra_evals')):
        if not mask&bit:assert h0[key]==0
    if not mask&8:assert l['transferred']==0
    assert a['bridge_wait_ns']>=0 and min(a['cpu_ns'])>=0 and len(a['cpu_ns'])==2
    for k in ('transferred','blocked_receipt','eligible','novel','edges'):ct[k]+=l[k]
    for k in ('select_changed','refinements','eval_changed','extra_evals'):ct[k]+=h0[k]
    ct['width_realized']+=eh['width_realized'];ct['all4']+=int(h0['select_changed']>0 and eh['width_realized']>0 and h0['eval_changed']>0 and l['transferred']>0)
    if trace:audit_trace(a,b,p,ct)
def audit_trace(a,root,p,ct):
    ts=a['hooks']['trace'];assert len(ts)==a['visits']
    led={};payload={};samples=defaultdict(list);n0=a['prefix']['result']['root_n'];cnt=Counter()
    def ck(b,side):return tuple(b),side
    def snapshot(i,rep):
        s=dict(iterations=i,edges=cnt['edges'],children=len(led),contexts=sum(len(x) for x in led.values()),shared=sum(len(x)>1 for x in led.values()),unique_base_leaf=len(payload),
          eligible=cnt['eligible'],novel=cnt['novel'],blocked_receipt=cnt['blocked'],transferred=cnt['transferred'],receipt_items=sum(len(rr) for x in led.values() for dd,rr in x.values()),applied=cnt['applied'],changed=cnt['changed'],repeated_delivery=0)
        assert rep==s,(a['selected_mask'],rep,s)
    for i,t in enumerate(ts):
        b=list(root);side=p;path=t['path'];edges=[];term=False
        for mv in path:
            assert not term and b[mv]==0;old=ck(b,side);x=1 if side==1 else 2;b[mv]=x;term=win_at(b,mv,x,'freestyle') or all(b);side=-side
            edges.append((ck(b,side),(old,mv)))
        assert term==t['terminal'];leaf=(ck(b,side),term)
        v=f32(t['raw']);y=f32(t['leaf']);assert t['mask']==(0 if i<n0 else a['selected_mask'])
        if term:assert v==y==(-1 if win_at(b,path[-1],1 if side==-1 else 2,'freestyle') else 0)
        if not t['mask']&4:assert v==y
        if leaf in payload:assert payload[leaf]==v
        else:payload[leaf]=v
        raw=v
        assert len(t['steps'])==len(edges)
        for j in reversed(range(len(edges))):
            child,ctx=edges[j];c=led.setdefault(child,{});own,received=c.setdefault(ctx,({},{}));other={}
            for k,(d,r) in c.items():
                if k==ctx:continue
                for l,z in d.items():
                    if l in other:assert other[l]==z
                    other[l]=z
            other.pop(leaf,None);eligible={k:z for k,z in other.items() if k not in own};novel={k:z for k,z in eligible.items() if k not in received}
            z=t['steps'][j];assert (len(eligible),len(novel),len(eligible)-len(novel))==(z['eligible'],z['novel'],z['blocked'])
            cnt['edges']+=1;cnt['eligible']+=len(eligible);cnt['novel']+=len(novel);cnt['blocked']+=len(eligible)-len(novel)
            if novel:near(S.mean(novel.values()),z['mean'])
            else:assert z['mean'] is None
            assert f32(z['incoming'])==y
            if leaf in own:assert own[leaf]==raw
            if leaf in received:assert received[leaf]==raw
            own[leaf]=raw
            if not term and t['mask']&8 and novel:
                assert not received.keys()&novel.keys();received.update(novel);cnt['transferred']+=len(novel);cnt['applied']+=1
                x=y;alpha=min(.25,len(novel)/(len(novel)+4));y=f32(max(-1.,min(1.,(1-alpha)*y+alpha*S.mean(novel.values()))));cnt['changed']+=abs(y-x)>1e-7
            raw=-raw;y=-y;assert f32(z['backup'])==y
        samples[path[0]].append(y)
        if i+1==n0:snapshot(i+1,a['prefix']['hooks']['ledger'])
    snapshot(len(ts),a['hooks']['ledger'])
    for e in a['result']['edges']:
        v=samples[e['move']];assert len(v)==e['n'];near(sum(v),e['w'])
        mu=S.mean(v) if v else 0.;near(sum((x-mu)**2 for x in v),e['m2'])
    ct['trace_iters']+=len(ts);ct['trace_edges']+=cnt['edges'];ct['trace_transferred']+=cnt['transferred']
def locked_rows(name):
    lk=read(name+'_LOCK.json')
    for f in lk['files']:
        p=D/f['file'];assert sha(p)==f['sha256'];yield from rows(p)
def expected_seed(j,n):return (j['seed']+n*0x9e3779b97f4a7c15)&((1<<64)-1)
def gate_small(a,h,j,c,cap=128,mask=0,locked=False,table=None):
    b,p,w=replay(h,'freestyle');assert w is None and a['history']==h and a['player']==p and b[a['best']]==0
    assert a['seed']==expected_seed(j,len(h)) and a['cap']==cap and a['threads']==1 and a['locked']==locked
    assert a['request_mask']==(mask if table is None else None) and a['table']==table
    z,r=proof(tuple(b),p)
    c.update(requests=1,NN=a['NN_calls'],visits=a['visits'])
    c[a['cause']]+=1
    assert 0<=a['NN_calls']<=cap and 0<=a['visits']<=cap and min(a['cpu_ns'])>=0 and a['bridge_wait_ns']>=0
    if z['gate'] is not None:
        assert a['best']==z['gate'] and a['cause']==z['kind'] and a['NN_calls']==a['visits']==0 and a['selected_mask'] is None
    else:
        assert a['NN_calls']>0 and a['visits']>0 and a['selected_mask']==(mask if table is None else table[a['bin']])
        assert a['cause'] in ('NN_TARGET','VISIT_CAP','VOTE_LOCK')
        if a['cause']=='NN_TARGET':assert a['NN_calls']==cap
        elif a['cause']=='VISIT_CAP':assert a['visits']==cap
        else:assert locked and a['visits']<cap and a['NN_calls']<cap
    return z
def score(a,z):return F(z['labels'][str(a['best'])]['utility'])
def net(a,z):return score(a,z)-F(a['NN_calls'],500)
def frac_mean(x):return sum(x,F())/len(x)
def boot(v,seed=2026112635+111):
    rng=random.Random(seed);q=sorted(S.mean(rng.choices(v,k=len(v))) for _ in range(4000))
    return [float(q[99]),float(q[3899])]
def main():
    pl=read('PLAN.json');co=read('COHORT.json');jobs={j['id']:j for j in pl['jobs']};assert len(jobs)==256 and len(set(j['family'] for j in jobs.values()))==256
    for j in jobs.values():
        b,p,w=replay(j['opening'],'freestyle');assert w is None and hashlib.sha256(orbit_board(b,p,'freestyle')).hexdigest()==j['family']
    for lk in pl['prior_locks']:assert sha(ROOT/lk['path'])==lk['sha256']
    assert sha(ROOT.parent/'quartz_kinetic35_target_20261126/mcts_demo')==pl['identity']['binary']
    assert sha(ROOT.parent/'rapfi_250615_quartz_bridge_build_20261125/pbrain-rapfi')==pl['identity']['bridge']
    for x in read('RAPFI_LOCK.json')['files']:assert sha(Path(x['path']))==x['sha256']
    assert co['plan']==sha(D/'PLAN.json')
    source=Counter();roots={};seen=set()
    for f in co['files']:
        p=D/f['file'];assert sha(p)==f['sha256'];c=Counter()
        for z in rows(p):
            j=z['job'];i=j['id'];assert j==jobs[i] and i<192 and i not in seen;seen.add(i);h=list(j['opening'])
            for a in z['rows']:gate_small(a,h,j,c);h.append(a['best'])
            b,p,w=replay(h,'freestyle');reason='TERMINAL' if w is not None else 'GATED' if proof(tuple(b),p)[0]['gate'] is not None else 'ADMITTED'
            assert z['reason']==reason and h==z['history'] and z['side']==p and (len(h)==j['target'] or w is not None)
            c[reason]+=1
            if reason=='ADMITTED':roots[i]=dict(job=j,history=h,side=p)
        for k,v in f.items():
            if k not in ('file','sha256'):assert c[k]==v
        source.update(c)
    assert len(seen)==192 and list(roots[i] for i in sorted(roots))==co['roots']
    for k,v in co['counts'].items():assert source[k]==v
    model=read('MODEL.json');fr=read('FROZEN.json');train=[];bins=defaultdict(list);allc={};live_ref={};stats={}
    for mode in ('TRAIN_CAL','TEST'):
        lk=read(mode+'_LOCK.json');assert lk['cohort']==sha(D/'COHORT.json') and lk['code']==sha(D/'experiment.py')
        if mode=='TEST':assert lk['freeze']==sha(D/'FROZEN.json')
        c=Counter();lc=Counter();seen=set();table=[];fixed=[]
        for z in locked_rows(mode):
            r=z['root'];j=r['job'];i=j['id'];h=r['history'];assert roots[i]==r and i not in seen;seen.add(i)
            order=[(m,w,t) for m in range(16) for w in (1,4) for t in (0,1)];random.Random(pl['seed']+i).shuffle(order)
            assert z['order']==[list(x) for x in order] and len(z['results'])==64
            a0=z['results']['0:1:0'];assert a0['proof']['gate'] is None
            for m,w,t in order:
                a=z['results'][f'{m}:{w}:{t}'];assert a['request_mask']==m and a['table'] is None and a['threads']==w and a['cap']==80 and not a['locked'] and a['seed']==expected_seed(j,len(h))
                audit_native(a,h,c);assert a['prefix']['result']==a0['prefix']['result'] and a['root_eval']==a0['root_eval']
            assert set(z['labels'])=={str(a['best']) for a in z['results'].values()}
            for m,v in z['labels'].items():
                hh=h+[int(m)];assert v['action']==int(m)
                for a in v['rows']:gate_small(a,hh,j,lc);hh.append(a['best'])
                b,p,w=replay(hh,'freestyle');assert w is not None and w==v['winner'] and hh==v['history'] and F(v['utility'])==F(1+r['side']*w,2)
                c['labels']+=1
            c['roots']+=1
            # exact rational root-wise deltas independent of production fitter
            d=[frac_mean([net(z['results'][f'{m}:1:{t}'],z) for t in (0,1)]) for m in range(16)]
            d=[x-d[0] for x in d]
            if j['group']=='TRAIN':train.append(d);bins[a0['bin']].append(d)
            else:
                # store only sufficient outcomes and snapshots for CAL/TEST live parity
                live_ref[i]=dict(root=r,results={k:v for k,v in z['results'].items() if k.endswith(':1:0')},labels=z['labels'])
            if mode=='TEST':
                fixed.append(dict(root=i,arms={str(w):[[dict(u=float(score(a,z)),NN=a['NN_calls'],visits=a['visits'],net=float(net(a,z)),wall_ms=a['transport_ns']/1e6,cpu_ms=sum(a['cpu_ns'])/1e6,wait_ms=a['bridge_wait_ns']/1e6,ipc_ms=a['eval_ipc_ns']/1e6,best=a['best'],S=a['hooks']['hooks']['select_changed'],X=a['hooks']['width_realized'],V=a['hooks']['hooks']['eval_changed'],B=a['hooks']['ledger']['transferred'],blocked=a['hooks']['ledger']['blocked_receipt'],shared=a['hooks']['ledger']['shared'],extra=a['hooks']['hooks']['extra_evals'],peak4=int(a['hooks']['peak']==4)) for a in [z['results'][f'{m}:{w}:{t}'] for t in (0,1)]] for m in range(16)] for w in (1,4)},distinct=len(z['labels']),nonconstant=len({v['utility'] for v in z['labels'].values()})>1))
        assert seen=={i for i,r in roots.items() if r['job']['group'] in (('TRAIN','CAL') if mode=='TRAIN_CAL' else ('TEST',))}
        for k,v in lk['counts'].items():
            a={'requests':c['native_requests'],'NN':c['NN'],'visits':c['visits'],'label_requests':lc['requests'],'label_NN':lc['NN'],'label_visits':lc['visits'],'roots':c['roots'],'labels':c['labels']}[k];assert a==v,(k,a,v)
        allc[mode]=dict(c,continuation=dict(lc))
        print('AUDITED',mode,dict(c),'labels',dict(lc),flush=True)
        if mode=='TRAIN_CAL':
            pooled=[frac_mean([v[m] for v in train]) for m in range(16)]
            tab=[max(range(16),key=lambda m:((sum(v[m] for v in bins[b])+8*pooled[m])/(len(bins[b])+8),-m)) for b in range(4)]
            static=max(range(16),key=lambda m:(pooled[m],-m))
            assert model['atlas']==sha(D/'TRAIN_CAL_LOCK.json') and model['train']==len(train) and model['bin_n']==[len(bins[b]) for b in range(4)]
            # Float rounding can only change exact tie selection, so require actual selected utility equals exact maximizer.
            for x,y in zip(model['pooled'],pooled):near(x,y)
            assert pooled[model['static']]==pooled[static]
            for b,m in enumerate(model['table']):assert sum(v[m] for v in bins[b])+8*pooled[m]==sum(v[tab[b]] for v in bins[b])+8*pooled[tab[b]]
    # Heldout live policies and CAL selection
    for mode in ('CAL','TEST'):
        lk=read(mode+'_LIVE_LOCK.json');p=D/f'{mode}_LIVE.jsonl.gz';assert sha(p)==lk['sha256'] and lk['model']==sha(D/'MODEL.json')
        rr=list(rows(p));assert len(rr)==lk['n']==sum(r['job']['group']==mode for r in roots.values());c=Counter();vv=defaultdict(list);uu=defaultdict(list)
        for x in rr:
            r=x['root'];i=r['job']['id'];z=live_ref[i];assert z['root']==r
            table={'base':[0]*4,'static':[model['static']]*4,'table':model['table']}
            if mode=='TEST':table['selected']=fr['table']
            order=list(table);random.Random(pl['seed']+81+i).shuffle(order);assert x['order']==order
            for k,v in x['policies'].items():
                a=v['response'];assert a['table']==table[k] and a['request_mask'] is None and not a['locked'] and a['cap']==80 and a['threads']==1
                audit_native(a,r['history'],c)
                ref=z['results'][f"{a['selected_mask']}:1:0"];assert a['result']==ref['result'] and a['NN_calls']==ref['NN_calls']
                u=score(a,z);y=net(a,z);near(v['u'],u);near(v['net'],y);uu[k].append(u);vv[k].append(y)
        s={}
        for k,v in vv.items():
            d=[x-y for x,y in zip(v,vv['base'])];lo=float(frac_mean(d))-2*S.stdev(d)/math.sqrt(len(d))
            a=lk['stats'][k];near(a['net'],frac_mean(v));near(a['u'],frac_mean(uu[k]));near(a['diff'],frac_mean(d));near(a['lower'],lo);s[k]=(float(frac_mean(v)),lo)
        if mode=='CAL':
            assert fr['model']==sha(D/'MODEL.json') and fr['cal']==sha(p.with_name('CAL_LIVE_LOCK.json'))
            selected=max(['base']+[k for k in ('static','table') if s[k][1]>0],key=lambda k:s[k][0])
            assert selected==fr['selected'] and fr['table']=={'base':[0]*4,'static':[model['static']]*4,'table':model['table']}[selected]
        for k,v in lk['counts'].items():assert v==c['native_requests' if k=='requests' else k]
        allc[mode+'_LIVE']=dict(c)
    # Independent factorial decomposition and matched actual NN subset.
    ss=read('ATLAS_SUMMARY.json');assert ss['n']==len(fixed)
    extra=dict(n=len(fixed),disagree_roots=sum(x['distinct']>1 for x in fixed),nonconstant_label_roots=sum(x['nonconstant'] for x in fixed),workers={})
    for w in ('1','4'):
        out=ss['workers'][w];contrasts={};effects={}
        for m in range(16):
            aa=[r['arms'][w][m][t] for r in fixed for t in (0,1)]
            for k in ('u','NN','visits','net','wall_ms','cpu_ms','wait_ms','ipc_ms'):near(S.mean(a[k] for a in aa),out['arms'][str(m)][k])
            for k in ('S','X','V','B','blocked','shared','extra','peak4'):assert sum(a[k] for a in aa)==out['arms'][str(m)][k]
            delta=[S.mean(r['arms'][w][m][t]['u']-r['arms'][w][0][t]['u'] for t in (0,1)) for r in fixed]
            dn=[S.mean(r['arms'][w][m][t]['net']-r['arms'][w][0][t]['net'] for t in (0,1)) for r in fixed]
            match=[r['arms'][w][m][t]['u']-r['arms'][w][0][t]['u'] for r in fixed for t in (0,1) if r['arms'][w][m][t]['NN']==r['arms'][w][0][t]['NN']]
            contrasts[str(m)]=dict(utility=S.mean(delta),utility_ci=boot(delta),net=S.mean(dn),net_ci=boot(dn),matched_NN=len(match),matched_utility=S.mean(match) if match else None)
        for k in ('u','NN','net','wall_ms','cpu_ms','wait_ms'):
            betas=[]
            for i in range(16):
                v=[sum((-1)**((i&~m).bit_count())*S.mean(r['arms'][w][m][t][k] for t in (0,1)) for m in range(16))/16 for r in fixed]
                beta=S.mean(v);near(beta,out['effects'][k]['beta'][i]);betas.append(beta)
                if i in (1,2,4,8):effects.setdefault(k,{})[str(i)]=dict(effect=2*beta,ci=boot([2*x for x in v]))
            near(2*sum(betas[i] for i in range(16) if i.bit_count()%2),out['effects'][k]['full_base'])
        extra['workers'][w]=dict(contrasts=contrasts,main_effects=effects)
    # independent arena replay and paired bootstrap
    lk=read('ARENA_LOCK.json');assert lk['plan']==sha(D/'PLAN.json') and lk['freeze']==sha(D/'FROZEN.json')
    ss=read('ARENA_SUMMARY.json');groups=defaultdict(lambda:defaultdict(list));seen=set();games=Counter();histories={}
    pols={'B128':(128,0,False,None),'B80':(80,0,False,None),'L80':(80,0,True,None),'F80L':(80,15,True,None),'SELECT80L':(80,0,True,fr['table'])}
    for f in lk['files']:
        p=D/f['file'];assert sha(p)==f['sha256'];c=Counter()
        for g in rows(p):
            j=g['job'];arm=g['arm'];color=g['color'];i=j['id'];assert jobs[i]==j and j['group']=='ARENA' and f['arm']==arm
            k=(arm,i,color);assert k not in seen;seen.add(k);h=list(j['opening']);cost=Counter()
            for a in g['rows']:
                b,p,w=replay(h,'freestyle');who=arm if p==color else 'B128';assert a['arm']==who
                gate_small(a,h,j,c,*pols[who])
                tag='candidate' if p==color else 'base';cost[tag+'_moves']+=1
                for key,val in [('NN',a['NN_calls']),('visits',a['visits']),('wall_ns',a['transport_ns']),('cpu_ns',sum(a['cpu_ns'])),('wait_ns',a['bridge_wait_ns'])]:cost[tag+'_'+key]+=val
                cost[tag+'_'+a['cause']]+=1;h.append(a['best'])
            b,p,w=replay(h,'freestyle');assert w is not None and w==g['winner'] and h==g['history'] and F(g['score'])==F(1+color*w,2)
            groups[arm][i].append((F(g['score']),color,cost));histories[k]=(h,cost);c['games']+=1
        for k in ('games','requests','NN','visits'):assert c[k]==f[k]
        games.update(c)
    for k,v in lk['counts'].items():assert games[k]==v
    for arm,gg in groups.items():
        v=ss['arms'][arm];assert len(gg)==64 and all(len(g)==2 and {x[1] for x in g}=={-1,1} for g in gg.values())
        d={i:sum(x[0] for x in g)/2 for i,g in gg.items()};cs={}
        for i,g in gg.items():
            c=Counter()
            for _,_,z in g:c.update(z)
            cs[i]=c
        cc=Counter()
        for c in cs.values():cc.update(c)
        for k,z in cc.items():assert z==v['counts'][k]
        near(frac_mean(list(d.values())),v['score']);assert sum(x[0]==1 for g in gg.values() for x in g)==v['wins'];assert sum(x[0]==0 for g in gg.values() for x in g)==v['losses']
        def red(c):return 1-F(c['candidate_NN']*c['base_moves'],c['candidate_moves']*c['base_NN'])
        rng=random.Random(pl['seed']+42);ids=sorted(gg);bs=[];bn=[]
        for _ in range(4000):
            ix=rng.choices(ids,k=len(ids));bs.append(frac_mean([d[i] for i in ix]));c=Counter()
            for i in ix:c.update(cs[i])
            bn.append(red(c))
        bs.sort();bn.sort();near(bs[199],v['NI_lower']);near(bs[99],v['score_ci'][0]);near(bs[3899],v['score_ci'][1]);near(red(cc),v['NN_reduction']);near(bn[99],v['NN_ci'][0]);near(bn[3899],v['NN_ci'][1])
        assert v['joint']==(v['NI_lower']>=.45 and v['NN_reduction']>=.3)
    same=0;saved=0;total=0;stops=0
    for i in range(192,256):
        for p in (-1,1):
            a=histories['B80',i,p];b=histories['L80',i,p];assert a[0]==b[0];same+=1
            assert b[1]['candidate_NN']<=a[1]['candidate_NN'] and b[1]['base_NN']==a[1]['base_NN']
            saved+=a[1]['candidate_NN']-b[1]['candidate_NN'];total+=a[1]['candidate_NN'];stops+=b[1]['candidate_VOTE_LOCK']
    extra['lock_parity']=dict(games=same,NN_saved=saved,NN_before=total,reduction=saved/total,stops=stops)
    allc['ARENA']=dict(games);print('AUDITED ARENA',dict(games),extra['lock_parity'],flush=True)
    c=Counter();lk=read('DIAGNOSTIC_LOCK.json');p=D/'DIAGNOSTIC.jsonl.gz';assert sha(p)==lk['sha256'];nr=0
    for z in rows(p):
        r=z['root'];assert r==roots[r['job']['id']];nr+=1
        for k,a in z['results'].items():audit_native(a,r['history'],c,True)
    assert nr==lk['roots']
    for k,v in lk['counts'].items():assert v==c['native_requests' if k=='requests' else k]
    allc['DIAGNOSTIC']=dict(c);allc['SOURCE']=dict(source)
    (D/'C09_AUDIT.json').write_text(json.dumps(allc,indent=2,sort_keys=True)+'\n')
    (D/'FACTORIAL_AUDIT.json').write_text(json.dumps(extra,indent=2,sort_keys=True)+'\n')
    print('AUDIT_OK',json.dumps(allc,sort_keys=True),flush=True)
if __name__=='__main__':main()
