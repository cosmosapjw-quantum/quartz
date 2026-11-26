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
    cap=128 if a['arm'].endswith('128') else 80;assert a['cap']==cap and a['threads']==(4 if a['arm']=='F80L4' else 1)
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
        if a['arm'] in ('F80L','F80L4'):assert d is not None and d<=cap-n
        t=v['budget'];budget(t,min(cap-n,8),cap-nn);n+=t['visits'];nn+=t['used_nn'];d=v['after_d']
        assert (n,nn)==(v['after_n'],v['after_nn']);ct['waves']+=1
    assert (n,nn)==(end['root_n'],end['evaluator_calls'])==(a['visits'],a['NN_calls'])
    assert n<=cap and nn<=cap and a['best']==end['best'] and root_dist(end)==d
    if a['cause']=='VOTE_LOCK':assert d is None or d>cap-n
    elif a['cause']=='NN_TARGET':assert nn==cap
    else:assert a['cause']=='VISIT_CAP' and n==cap
    l=eh['ledger'];h0=eh['hooks'];assert l['iterations']==n and l['eligible']==l['novel']+l['blocked_receipt'] and l['transferred']==l['receipt_items'] and l['repeated_delivery']==0
    assert h0['base_evals']+h0['extra_evals']==nn-mid['evaluator_calls'] and h0['base_evals']+h0['terminal_leaves']==n-mid['root_n']
    assert eh['peak']<=a['threads'];ct['peak4']+=int(eh['peak']==4)
    assert all(v['vl']==0 and v['vvalue']==0 for v in end['edges'])
    if a['arm'].startswith('B'):assert l['transferred']==h0['extra_evals']==h0['select_nodes']==h0['width_nodes']==0
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
        assert rep==s,(a['arm'],rep,s)
    for i,t in enumerate(ts):
        b=list(root);side=p;path=t['path'];edges=[];term=False
        for mv in path:
            assert not term and b[mv]==0;old=ck(b,side);x=1 if side==1 else 2;b[mv]=x;term=win_at(b,mv,x,'freestyle') or all(b);side=-side
            edges.append((ck(b,side),(old,mv)))
        assert term==t['terminal'];leaf=(ck(b,side),term)
        v=f32(t['raw']);y=f32(t['leaf']);assert t['mask']==(0 if i<n0 or a['arm'].startswith('B') else 15)
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
def main():
    pl=read('PLAN.json');jobs={j['id']:j for j in pl['jobs']};assert len(jobs)==42 and len(set(j['family'] for j in jobs.values()))==42
    for j in jobs.values():
        b,p,w=replay(j['opening'],'freestyle');assert w is None
        assert hashlib.sha256(orbit_board(b,p,'freestyle')).hexdigest()==j['family']
    for lk in pl['prior_locks']:assert sha(ROOT/lk['path'])==lk['sha256']
    assert sha(ROOT.parent/'quartz_kinetic34_target_20261126/mcts_demo')==pl['identity']['binary']
    assert sha(ROOT.parent/'rapfi_250615_quartz_bridge_build_20261125/pbrain-rapfi')==pl['identity']['bridge']
    for z in read('RAPFI_LOCK.json')['files']:assert sha(Path(z['path']))==z['sha256']
    modes={}
    for mode in ('MAIN','EXT'):
        lk=read(mode+'_LOCK.json');assert lk['plan']==sha(D/'PLAN.json') and lk['code']==sha(D/'arena.py')
        summary=read(mode+'_SUMMARY.json');ct=Counter();seen=set();by=defaultdict(lambda:defaultdict(list))
        for fi in lk['files']:
            p=D/fi['file'];assert sha(p)==fi['sha256'];local=Counter()
            for g in rows(p):
                j=g['job'];a=g['arm'];color=g['color'];assert jobs[j['id']]==j and (mode=='EXT')==g['external']
                k=(j['id'],a,color);assert k not in seen;seen.add(k);h=list(j['opening']);cost=Counter()
                for r in g['rows']:
                    b,side,w=replay(h,'freestyle');assert w is None and r['player']==side and r['history']==h and b[r['best']]==0
                    who=a if side==color else 'RAPFI100' if mode=='EXT' else 'B128';assert r['arm']==who
                    assert r['seed']==(j['seed']+len(h)*0x9e3779b97f4a7c15)&((1<<64)-1)
                    tag='candidate' if side==color else 'base';cost[tag+'_moves']+=1;cost[tag+'_wall_ns']+=r['transport_ns']
                    if who=='RAPFI100':
                        assert r['NN_calls'] is None and r['status']=='OK' and r['compatible_weight_reported'] and r['saw_loaded_nnue'] and r['wall_budget_requested_ms']==100 and r['best']==r['flat_action']
                        local['external_moves']+=1
                    else:
                        audit_native(r,h,local);cost[tag+'_NN']+=r['NN_calls'];cost[tag+'_visits']+=r['visits']
                    h.append(r['best']);local['moves']+=1
                b,side,w=replay(h,'freestyle');assert w is not None and w==g['winner'] and h==g['history'] and F(g['score'])==F(1+color*w,2)
                by[a][j['id']].append((F(g['score']),color,cost));local['games']+=1
            for k in ('games','moves','NN','external_moves'):assert local[k]==fi[k],(k,local[k],fi[k])
            ct.update(local)
        for k in ('games','moves','NN','external_moves'):assert ct[k]==lk[k]
        assert len(seen)==(360 if mode=='MAIN' else 24)
        for arm,groups in by.items():
            v=summary['arms'][arm];assert all(len(g)==2 and {x[1] for x in g}=={-1,1} for g in groups.values())
            ds={i:sum(x[0] for x in g)/2 for i,g in groups.items()};cs={}
            for i,g in groups.items():
                c=Counter()
                for _,_,z in g:c.update(z)
                cs[i]=c
            cc=Counter()
            for c in cs.values():cc.update(c)
            near(S.mean(ds.values()),v['score']);assert sum(x[0]==1 for g in groups.values() for x in g)==v['wins'];assert sum(x[0]==0 for g in groups.values() for x in g)==v['losses']
            for k,z in cc.items():assert z==v['counts'][k]
            def red(c):return 1-F(c['candidate_NN']*c['base_moves'],c['candidate_moves']*c['base_NN'])
            rng=random.Random(pl['seed']+41);ss=[];rs=[];ids=sorted(groups)
            for _ in range(4000):
                sel=rng.choices(ids,k=len(ids));ss.append(sum(ds[i] for i in sel)/len(ids))
                if mode=='MAIN':
                    c=Counter()
                    for i in sel:c.update(cs[i])
                    rs.append(red(c))
            ss.sort();near(ss[99],v['score_ci'][0]);near(ss[3899],v['score_ci'][1]);near(ss[199],v['NI_lower'])
            if mode=='MAIN':
                rs.sort();near(red(cc),v['NN_reduction']);near(rs[99],v['NN_ci'][0]);near(rs[3899],v['NN_ci'][1]);assert v['point_gate']==(v['NI_lower']>=.45 and v['NN_reduction']>=.3)
        modes[mode]=dict(ct)
        print('AUDITED',mode,dict(ct),flush=True)
    ct=Counter();lk=read('DIAGNOSTIC_LOCK.json');p=D/'DIAGNOSTIC.jsonl.gz';assert sha(p)==lk['sha256'];n=0
    for z in rows(p):
        j=z['job'];assert j==jobs[n];g=z['results']
        for a,r in g.items():audit_native(r,j['opening'],ct,True)
        assert g['F80']['best']==g['F80L']['best']
        assert all(r['proof']==g['B128']['proof'] for r in g.values())
        if g['B128']['root_eval'] is not None:assert all(r['root_eval']==g['B128']['root_eval'] and r['prefix']['result']==g['B128']['prefix']['result'] for r in g.values())
        n+=1
    assert n==lk['n']==12;modes['DIAGNOSTIC']=dict(ct)
    modes['proof_cache']=str(proof.cache_info())
    (D/'C09_AUDIT.json').write_text(json.dumps(modes,indent=2,sort_keys=True)+'\n')
    print('AUDIT',json.dumps(modes,sort_keys=True),flush=True)
if __name__=='__main__':main()
