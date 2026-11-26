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
H=(80,96,112,128,256)
def vote(v,legal):
    assert len(v)==len(legal) and {m for m,n in v}==set(legal) and all(isinstance(n,int) and n>=0 for m,n in v)
    n=[x[1] for x in v];L=max(range(len(n)),key=lambda j:(n[j],j))
    return sum(n),v[L][0],min((n[L]-n[j]+int(j<L) for j in range(len(n)) if j!=L),default=None)
def vector(r):
    es={e['move']:e['n'] for e in r['edges']}
    return [[m,es.get(m,0)] for m,p in r['candidates']]
def compact(a):
    z={k:a[k] for k in ('history','best','player','cap','threads','locked','requested_cap','horizons','selected_mask','bin','NN_calls','visits','cause','proof_us','bridge_wait_ns','eval_ipc_ns','transport_ns','cpu_ns','seed','probe_wave')}
    z['votes']=vector(a['result']) if a['result'] else None
    z['probe_votes']=vector(a['probe']) if a['probe'] else None
    z['prefix_votes']=vector(a['prefix']['result']) if a['prefix'] else None
    z['prefix_nn']=a['prefix']['result']['evaluator_calls'] if a['prefix'] else None
    z['probe_nn']=a['probe']['evaluator_calls'] if a['probe'] else None
    z['waves']=[[w['before_n'],w['before_nn'],w['before_d'],w['after_n'],w['after_nn'],w['after_d'],w['budget']['target'],w['budget']['cap']] for w in a['waves']]
    return z
def gate_small(a,h,j,c,cap=128,locked=False,table=None):
    b,p,w=replay(h,'freestyle');assert w is None and a['history']==h and a['player']==p and b[a['best']]==0
    assert a['seed']==expected_seed(j,len(h)) and a['threads']==1 and a['locked']==locked
    assert a['requested_cap']==(cap if table is None else None) and a['horizons']==table
    assert len(a['cpu_ns'])==2 and min(a['cpu_ns'])>=0 and a['bridge_wait_ns']>=0
    z,replies=proof(tuple(b),p)
    c.update(requests=1,NN=a['NN_calls'],visits=a['visits'],bounds=len(z['actions']),win3_actions=len(z['win3']),win3_replies_logical=replies)
    c[a['cause']]+=1
    if z['gate'] is not None:
        assert a['best']==z['gate'] and a['cause']==z['kind'] and a['NN_calls']==a['visits']==0 and a['selected_mask'] is None
        assert a['votes'] is None and a['prefix_votes'] is None and a['probe_votes'] is None and a['waves']==[] and a['cap']==(cap if table is None else 128)
        return z
    legal=[i for i,x in enumerate(b) if x==0];v=a['votes'];pv=a['probe_votes'];iv=a['prefix_votes']
    assert [m for m,n in v]==[m for m,n in pv]==[m for m,n in iv]
    n,leader,d=vote(iv,legal);nn=a['prefix_nn'];pn,pl,pd=vote(pv,legal);fn,fl,fd=vote(v,legal)
    assert n<=16 and nn<=16 and 1<=nn<=n+1 and (pn>=32 or a['probe_nn']>=32) and pn<=39
    bn=int(pv[0][0]!=pl)+2*int(leader!=pl)+4*int(pd is None or pd>=8)
    assert a['bin']==bn and a['selected_mask']==0 and a['cap']==(cap if table is None else table[bn])
    hcap=a['cap'];k=a['probe_wave'];assert 0<k<=len(a['waves'])
    for i,(bn0,bnn,bd,an,ann,ad,target,quota) in enumerate(a['waves']):
        assert (bn0,bnn,bd)==(n,nn,d)
        active=(cap if table is None else 256) if i<k else hcap
        assert target==active-nn and quota==min(8,active-n) and 0<an-n<=quota and 0<=ann-nn<=min(target,an-n)
        if ann-nn<target:assert an-n==quota
        if i<k:assert n<32 and nn<32
        if locked:assert d is not None and d<=active-n
        n,nn,d=an,ann,ad
        if i+1==k:assert (n,nn,d)==(pn,a['probe_nn'],pd)
        c['waves']+=1
    assert n==fn==a['visits'] and nn==a['NN_calls'] and fl==a['best'] and d==fd
    assert 0<n<=hcap and 0<nn<=hcap
    if a['cause']=='NN_TARGET':assert nn==hcap
    elif a['cause']=='VISIT_CAP':assert n==hcap
    else:
        assert a['cause']=='VOTE_LOCK' and locked and n<hcap and nn<hcap and (d is None or d>hcap-n)
        c['lock_checks']+=1
    return z
def audit_native(a,h,j,c,cap=128,locked=False,table=None,trace=False):
    z=gate_small(compact(a),h,j,c,cap,locked,table);assert a['schema']=='quartz_rapfi_k36_v1' and a['proof']==z
    c['detailed']+=1
    if z['gate'] is not None:return
    ini=a['initial'];mid=a['prefix']['result'];end=a['result'];pre=a['prefix']['hooks'];eh=a['hooks'];probe=a['probe'];ev=a['root_eval']
    b,p,w=replay(h,'freestyle');legal=[i for i,x in enumerate(b) if not x];ca=ini['candidates']
    assert len(ca)==len(legal) and set(m for m,x in ca)==set(legal)
    assert ini['root_n']==0 and ini['evaluator_calls']==1 and ca==mid['candidates']==probe['candidates']==end['candidates']
    assert ev['board']==''.join(map(str,b)) and ev['player']==p
    assert abs(ev['win']+ev['draw']+ev['loss']-1)<1e-6 and abs(sum(ev['policy'])-1)<1e-5
    near(ev['win']-ev['loss'],ev['value']);assert all(abs(ev['policy'][m]-v)<1e-7 for m,v in ca)
    t=a['prefix']['budget'];budget(t,16,15);assert t['visits']==mid['root_n'] and t['used_nn']==mid['evaluator_calls']-1
    assert pre['hooks']['base_evals']==t['used_nn'] and pre['ledger']['iterations']==mid['root_n']
    for w in a['waves']:budget(w['budget'],w['budget']['cap'],w['budget']['target'])
    l=eh['ledger'];h0=eh['hooks'];assert l['iterations']==end['root_n'] and l['eligible']==l['novel']+l['blocked_receipt'] and l['transferred']==l['receipt_items']==l['repeated_delivery']==0
    assert h0['base_evals']==end['evaluator_calls']-mid['evaluator_calls'] and h0['base_evals']+h0['terminal_leaves']==end['root_n']-mid['root_n']
    assert h0['select_nodes']==h0['width_nodes']==h0['extra_evals']==0 and eh['peak']<=1
    assert all(v['vl']==0 and v['vvalue']==0 for v in end['edges'])
    for s in (mid,probe,end):assert root_dist(s)==vote(vector(s),legal)[2]
    c['ledger_edges']+=l['edges'];assert (eh['trace'] is not None)==trace
    if trace:audit_trace(a,b,p,c)
def U(a,z,ref=None):
    q=z['labels'][str(a['best'])]
    return F(q[str(ref)]['utility']) if ref else (F(q['128']['utility'])+F(q['256']['utility']))/2
def Y(a,z):return U(a,z)-F(a['NN_calls'],500)
def mean(v):return sum(v,F())/len(v)
def tie(h):return (h==128,-h)
def boot(v,sd=2026112636+211):
    rng=random.Random(sd);a=sorted(S.mean(rng.choices(v,k=len(v))) for _ in range(4000));return [a[99],a[3899]]
def audit_atlas_summary(rr,out,main):
    for h in H:
        for lk in ('B','L'):
            k=lk+str(h);s=out[k];aa=[(x,x['results'][k]) for x in rr]
            for a,v in [('u',mean([U(a,x) for x,a in aa])),('u128',mean([U(a,x,128) for x,a in aa])),('u256',mean([U(a,x,256) for x,a in aa])),('net',mean([Y(a,x) for x,a in aa])),('NN',S.mean(a['NN_calls'] for x,a in aa)),('visits',S.mean(a['visits'] for x,a in aa)),('cpu_ms',S.mean(sum(a['cpu_ns'])/1e6 for x,a in aa)),('wall_ms',S.mean(a['transport_ns']/1e6 for x,a in aa)),('lock_stops',sum(a['cause']=='VOTE_LOCK' for x,a in aa)),('disagree',sum(a['best']!=x['results']['B128']['best'] for x,a in aa))]:near(s[a],v)
            d=[U(a,x)-U(x['results']['L128'],x) for x,a in aa];dn=[Y(a,x)-Y(x['results']['L128'],x) for x,a in aa];near(s['gain'],mean(d));near(s['net_gain'],mean(dn))
            if main:
                for key,v in [('gain_ci',d),('net_ci',dn)]:
                    for a,b in zip(s[key],boot(v)):near(a,b)
def label_stats(rr):
    return dict(actions=sum(len(x['labels']) for x in rr),label_disagree=sum(v['128']['utility']!=v['256']['utility'] for x in rr for v in x['labels'].values()),nonconstant_avg=sum(len({(v['128']['utility']+v['256']['utility'])/2 for v in x['labels'].values()})>1 for x in rr),
      nonconstant128=sum(len({v['128']['utility'] for v in x['labels'].values()})>1 for x in rr),nonconstant256=sum(len({v['256']['utility'] for v in x['labels'].values()})>1 for x in rr),disagree_roots=sum(x['stratum']['disagree'] for x in rr),ref_disagree=sum(x['stratum']['ref_disagree'] for x in rr))
def main():
    pl=read('PLAN.json');co=read('COHORT.json');jobs={j['id']:j for j in pl['jobs']};assert len(jobs)==416 and len({j['family'] for j in jobs.values()})==416
    for j in jobs.values():
        b,p,w=replay(j['opening'],'freestyle');assert w is None and hashlib.sha256(orbit_board(b,p,'freestyle')).hexdigest()==j['family']
    for x in pl['prior_locks']:assert sha(ROOT/x['path'])==x['sha256']
    assert sha(ROOT.parent/'quartz_kinetic36_target_20261126/mcts_demo')==pl['identity']['binary']
    assert sha(ROOT.parent/'rapfi_250615_quartz_bridge_build_20261125/pbrain-rapfi')==pl['identity']['bridge']
    for x in read('RAPFI_LOCK.json')['files']:assert sha(Path(x['path']))==x['sha256']
    assert co['plan']==sha(D/'PLAN.json')
    c=Counter();roots={};seen=set()
    for fi in co['files']:
        p=D/fi['file'];assert sha(p)==fi['sha256'];q=Counter()
        for z in rows(p):
            j=z['job'];i=j['id'];assert j==jobs[i] and i<320 and i not in seen;seen.add(i);h=list(j['opening'])
            for a in z['rows']:gate_small(a,h,j,q);h.append(a['best'])
            b,p,w=replay(h,'freestyle');reason='TERMINAL' if w is not None else 'GATED' if proof(tuple(b),p)[0]['gate'] is not None else 'ADMITTED'
            assert z['reason']==reason and z['side']==p and h==z['history'] and (len(h)==j['target'] or w is not None)
            q[reason]+=1
            if reason=='ADMITTED':roots[i]=dict(job=j,history=h,side=p)
        for k,v in fi.items():
            if k not in ('file','sha256'):assert q[k]==v
        c.update(q)
    assert len(seen)==320 and [roots[i] for i in sorted(roots)]==co['roots']
    for k,v in co['counts'].items():assert c[k]==v
    out={'SOURCE':dict(c)};print('AUDIT SOURCE',dict(c),flush=True)
    m=read('MODEL.json');fr=read('FROZEN.json');tr=[];bins=defaultdict(list);refs={};test=[];labels_by={}
    for mode in ('TRAIN_CAL','TEST'):
        lk=read(mode+'_LOCK.json');assert lk['cohort']==sha(D/'COHORT.json') and lk['code']==sha(D/'experiment.py')
        if mode=='TEST':assert lk['freeze']==sha(D/'FROZEN.json')
        c=Counter();lc=Counter();seen=set();rr=[]
        for z in locked_rows(mode):
            r=z['root'];j=r['job'];i=j['id'];h=r['history'];assert roots[i]==r and i not in seen;seen.add(i)
            order=[(h,lk) for h in H for lk in (False,True)];random.Random(pl['seed']+i).shuffle(order);assert z['order']==[list(x) for x in order]
            a0=z['results']['B128'];assert a0['proof']['gate'] is None
            for cap,lock in order:
                a=z['results'][('L' if lock else 'B')+str(cap)];audit_native(a,h,j,c,cap,lock)
                assert a['probe']==a0['probe'] and a['prefix']['result']==a0['prefix']['result'] and a['root_eval']==a0['root_eval']
            for cap in H:
                a,b=z['results']['B'+str(cap)],z['results']['L'+str(cap)]
                assert a['best']==b['best'] and b['NN_calls']<=a['NN_calls'] and b['visits']<=a['visits'];c['parity']+=1
            assert set(z['labels'])=={str(a['best']) for a in z['results'].values()}
            for action,v in z['labels'].items():
                assert set(v)=={'128','256'}
                for cap in (128,256):
                    q=v[str(cap)];hh=h+[int(action)];assert q['cap']==cap and q['action']==int(action)
                    for a in q['rows']:gate_small(a,hh,j,lc,cap);hh.append(a['best'])
                    b,p,w=replay(hh,'freestyle');assert w is not None and hh==q['history'] and w==q['winner'] and F(q['utility'])==F(1+r['side']*w,2)
                    c['labels']+=1
                c['actions']+=1
            assert z['stratum']==dict(disagree=len(z['labels'])>1,ref_disagree=a0['best']!=z['results']['B256']['best'],prior_conflict=bool(a0['bin']&1),side=r['side'],ply=len(h))
            c['roots']+=1;rr.append(z)
            if j['group']=='TRAIN':
                d={h:Y(z['results']['L'+str(h)],z)-Y(z['results']['L128'],z) for h in H};tr.append(d);bins[a0['bin']].append(d)
            else:refs[i]=z
        assert seen=={i for i,r in roots.items() if r['job']['group'] in (('TRAIN','CAL') if mode=='TRAIN_CAL' else ('TEST',))}
        for k,v in lk['counts'].items():
            a={'label_requests':lc['requests'],'label_NN':lc['NN'],'label_visits':lc['visits']}.get(k,c[k]);assert a==v,(k,a,v)
        out[mode]=dict(c,continuation=dict(lc))
        labels_by[mode]=label_stats(rr);print('AUDIT',mode,dict(c),'labels',dict(lc),labels_by[mode],flush=True)
        if mode=='TRAIN_CAL':
            pool={h:mean([v[h] for v in tr]) for h in H}
            stat=max(H,key=lambda h:(pool[h],*tie(h)))
            tab=[max(H,key=lambda h:((sum(v[h] for v in bins[b])+8*pool[h])/(len(bins[b])+8),*tie(h))) for b in range(8)]
            assert m['atlas']==sha(D/'TRAIN_CAL_LOCK.json') and m['train']==len(tr) and m['bin_n']==[len(bins[b]) for b in range(8)] and m['static']==stat and m['table']==tab
            for h,x in pool.items():near(m['pooled'][str(h)],x)
        else:test=rr
    # actual live table parity and frozen selection
    for mode in ('CAL','TEST'):
        p=D/f'{mode}_LIVE.jsonl.gz';lk=read(mode+'_LIVE_LOCK.json');assert sha(p)==lk['sha256'] and lk['model']==sha(D/'MODEL.json')
        rr=list(rows(p));assert len(rr)==lk['n']==sum(r['job']['group']==mode for r in roots.values());c=Counter();u=defaultdict(list);y=defaultdict(list);nn=defaultdict(list);hs=defaultdict(Counter)
        for z in rr:
            r=z['root'];j=r['job'];i=j['id'];ref=refs[i];assert ref['root']==r
            tabs={'base':[128]*8,'static':[m['static']]*8,'table':m['table']}
            if mode=='TEST':tabs['selected']=fr['table']
            order=list(tabs);random.Random(pl['seed']+81+i).shuffle(order);assert z['order']==order
            for k,v in z['policies'].items():
                a=v['response'];audit_native(a,r['history'],j,c,locked=True,table=tabs[k])
                x=ref['results']['L'+str(a['cap'])];assert a['result']==x['result'] and a['NN_calls']==x['NN_calls'] and a['probe']==x['probe']
                near(v['u'],U(a,ref));near(v['u128'],U(a,ref,128));near(v['u256'],U(a,ref,256));near(v['net'],Y(a,ref))
                u[k].append(U(a,ref));y[k].append(Y(a,ref));nn[k].append(a['NN_calls']);hs[k][str(a['cap'])]+=1
        def low(v):return float(mean(v))-2*S.stdev(v)/math.sqrt(len(v))
        stats={}
        for k,v in y.items():
            d=[a-b for a,b in zip(v,y['base'])];du=[a-b for a,b in zip(u[k],u['base'])]
            q=lk['stats'][k];near(q['net'],mean(v));near(q['u'],mean(u[k]));near(q['diff'],mean(d));near(q['lower'],low(d));near(q['u_diff'],mean(du));near(q['u_lower'],low(du));near(q['NN'],S.mean(nn[k]));assert q['horizons']==dict(hs[k])
            stats[k]=(mean(v),low(d),low(du))
        if mode=='CAL':
            assert fr['model']==sha(D/'MODEL.json') and fr['cal']==sha(D/'CAL_LIVE_LOCK.json')
            selected=max(['base']+[k for k in ('static','table') if stats[k][1]>0 and stats[k][2]>=-.05],key=lambda k:stats[k][0])
            assert fr['selected']==selected and fr['table']=={'base':[128]*8,'static':[m['static']]*8,'table':m['table']}[selected]
        for k,v in lk['counts'].items():assert c[k]==v
        out[mode+'_LIVE']=dict(c)
    ss=read('ATLAS_SUMMARY.json');assert ss['n']==len(test) and ss['labels']==label_stats(test);audit_atlas_summary(test,ss['arms'],True)
    strata=defaultdict(list)
    for x in test:
        for k,v in x['stratum'].items():strata[k+':'+str(v)].append(x)
    assert set(strata)==set(ss['strata'])
    for k,rr in strata.items():assert len(rr)==ss['strata'][k]['n'];audit_atlas_summary(rr,ss['strata'][k]['arms'],False)
    # actual games, independently reconstructed per-move certificate and paired bootstrap
    lk=read('ARENA_LOCK.json');assert lk['plan']==sha(D/'PLAN.json') and lk['freeze']==sha(D/'FROZEN.json')
    ss=read('ARENA_SUMMARY.json');assert ss['alias']==lk['alias'];groups=defaultdict(lambda:defaultdict(list));seen=set();games=Counter();histories={}
    pols={'B128':(128,False,None),'L128':(128,True,None),'L96':(96,True,None),'L112':(112,True,None),'ADAPT':(128,True,fr['table'])}
    for fi in lk['files']:
        p=D/fi['file'];assert sha(p)==fi['sha256'];c=Counter()
        for g in rows(p):
            j=g['job'];arm=g['arm'];color=g['color'];i=j['id'];assert jobs[i]==j and j['group']=='ARENA' and fi['arm']==arm
            k=(arm,i,color);assert k not in seen;seen.add(k);h=list(j['opening']);cost=Counter()
            for a in g['rows']:
                b,p,w=replay(h,'freestyle');who=arm if p==color else 'B128';assert a['arm']==who
                gate_small(a,h,j,c,*pols[who])
                tag='candidate' if p==color else 'base';cost[tag+'_moves']+=1
                for key,val in [('NN',a['NN_calls']),('visits',a['visits']),('wall_ns',a['transport_ns']),('cpu_ns',sum(a['cpu_ns'])),('wait_ns',a['bridge_wait_ns'])]:cost[tag+'_'+key]+=val
                cost[tag+'_'+a['cause']]+=1;h.append(a['best'])
            b,p,w=replay(h,'freestyle');assert w is not None and w==g['winner'] and h==g['history'] and F(g['score'])==F(1+color*w,2)
            groups[arm][i].append((F(g['score']),color,cost));histories[k]=g;c['games']+=1
        for k in ('games','requests','NN','visits'):assert c[k]==fi[k]
        games.update(c)
    for k,v in lk['counts'].items():assert games[k]==v
    for arm,gg in groups.items():
        v=ss['arms'][arm];assert len(gg)==96 and all(len(g)==2 and {x[1] for x in g}=={-1,1} for g in gg.values())
        d={i:sum(x[0] for x in g)/2 for i,g in gg.items()};cs={}
        for i,g in gg.items():
            c=Counter()
            for _,_,z in g:c.update(z)
            cs[i]=c
        cc=Counter()
        for c in cs.values():cc.update(c)
        for k,z in cc.items():assert z==v['counts'][k]
        near(mean(list(d.values())),v['score']);assert sum(x[0]==1 for g in gg.values() for x in g)==v['wins'];assert sum(x[0]==0 for g in gg.values() for x in g)==v['losses']
        def red(c):return 1-F(c['candidate_NN']*c['base_moves'],c['candidate_moves']*c['base_NN'])
        rng=random.Random(pl['seed']+42);ids=sorted(gg);bs=[];bn=[]
        for _ in range(4000):
            ix=rng.choices(ids,k=len(ids));bs.append(mean([d[i] for i in ix]));c=Counter()
            for i in ix:c.update(cs[i])
            bn.append(red(c))
        bs.sort();bn.sort();near(bs[199],v['NI_lower']);near(bs[99],v['score_ci'][0]);near(bs[3899],v['score_ci'][1]);near(red(cc),v['NN_reduction']);near(bn[99],v['NN_ci'][0]);near(bn[3899],v['NN_ci'][1]);assert v['joint']==(v['NI_lower']>=.45 and v['NN_reduction']>=.3)
    # The two color-swapped L128 games directly supply baseline and lock at identical positions.
    parity=Counter()
    for i in range(320,416):
        a,b=histories['L128',i,1],histories['L128',i,-1];assert a['history']==b['history'] and a['winner']==b['winner']
        parity['pairs']+=1;parity['games']+=2
        for x,y in zip(a['rows'],b['rows']):
            assert x['history']==y['history'] and x['best']==y['best'] and x['player']==y['player']
            if x['arm']=='B128':x,y=y,x
            assert x['arm']=='L128' and y['arm']=='B128' and x['NN_calls']<=y['NN_calls'] and x['visits']<=y['visits']
            parity.update(moves=1,NN_before=y['NN_calls'],NN_after=x['NN_calls'],visits_before=y['visits'],visits_after=x['visits'])
            parity['lock_stops']+=x['cause']=='VOTE_LOCK'
    parity['NN_saved']=parity['NN_before']-parity['NN_after']
    out['ARENA']=dict(games);print('AUDIT ARENA',dict(games),dict(parity),flush=True)
    c=Counter();lk=read('DIAGNOSTIC_LOCK.json');p=D/'DIAGNOSTIC.jsonl.gz';assert sha(p)==lk['sha256'];nr=0;timing=defaultdict(list)
    for z in rows(p):
        r=z['root'];j=r['job'];assert r==roots[j['id']]
        for k,a in z['results'].items():
            audit_native(a,r['history'],j,c,int(k[1:]),k[0]=='L',trace=nr<3);timing[k].append(a)
        nr+=1
    assert nr==lk['roots']==12
    for k,v in lk['counts'].items():assert c[k]==v
    out['DIAGNOSTIC']=dict(c)
    tm={k:dict(n=len(v),NN=S.mean(a['NN_calls'] for a in v),visits=S.mean(a['visits'] for a in v),wall_ms=S.mean(a['transport_ns']/1e6 for a in v),cpu_ms=S.mean(sum(a['cpu_ns'])/1e6 for a in v),lock=sum(a['cause']=='VOTE_LOCK' for a in v)) for k,v in timing.items()}
    (D/'C09_AUDIT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    (D/'HORIZON_AUDIT.json').write_text(json.dumps(dict(labels=labels_by,same128=dict(parity),same128_reduction=parity['NN_saved']/parity['NN_before'],timing=tm,scope='actual paired deterministic paths, full atlas/compact wave and vote reconstruction; timing includes traces and host contention'),indent=2,sort_keys=True)+'\n')
    print('AUDIT_OK',json.dumps(out,sort_keys=True),flush=True)
if __name__=='__main__':main()
