"""Independent Python integer geometry, forcing proof and raw chronological cost audit.
The collector v0 retained a mutable history reference in atlas metadata.
Reconstruct exact probe histories from raw ply/order and verify every receipt there.
"""
import collections as C,gzip,json,hashlib,pathlib,functools,random,math
from fractions import Fraction as F
from k43_plan import D,ROOT,T,SEEDS,TIERS,load,save,sha,family
CT=C.Counter()
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def rows(mode):
    lk=load(mode+'_LOCK.json')
    assert lk['plan']==sha(D/'PLAN.json') and lk['source']==sha(D/'partial43.cpp') and lk['lib']==sha(T/'partial43.so')
    assert lk['engine']==sha(D/'k43_engine.py') and lk['collector']==sha(D/'k43_experiment.py')
    for fi in lk['files']:
        p=D/fi['file'];assert sha(p)==fi['sha256'];rr=list(raw(p))
        assert sum(g['nn'] for g in rr)==fi['counts']['nn']
        assert sum(len(g['rows']) for g in rr)==fi['counts']['moves']
        assert len(rr)==fi['counts']['games']
        if mode.endswith('SOURCE'):
            assert sum(len(g['atlas']) for g in rr)==fi['counts']['atlas_roots']
            assert sum(a['L' if r['arm']=='B' else 'B']['nn'] for g in rr for r,a in zip([r for r in g['rows'] if r['kind']!='CERTIFICATE'],g['atlas']))==fi['counts']['atlas_extra_nn']
        yield from rr
def bits(e):
    while e:
        x=e&-e;yield x.bit_length()-1;e^=x
def setup(n):
    k=4 if n==7 else 5;ls=[]
    for r in range(n):
        for c in range(n):
            for dr,dc in ((1,0),(0,1),(1,1),(1,-1)):
                if 0<=r+(k-1)*dr<n and 0<=c+(k-1)*dc<n:
                    ls.append(sum(1<<((r+i*dr)*n+c+i*dc) for i in range(k)))
    assert len(ls)==(88 if n==7 else 572),(n,len(ls))
    return k,(1<<(n*n))-1,ls
# 7x7 length-four has 88? Horizontal28, vertical28, diagonals16+16=88.
G={n:setup(n) for n in (7,15)}
def coordwin(a,m,n):
    k=G[n][0];r,c=divmod(m,n)
    for dr,dc in ((1,0),(0,1),(1,1),(1,-1)):
        cnt=1
        for sg in (-1,1):
            x,y=r+sg*dr,c+sg*dc
            while 0<=x<n and 0<=y<n and (a>>(n*x+y))&1:
                cnt+=1;x+=sg*dr;y+=sg*dc
        if cnt>=k:return True
    return False
def replay(h,n):
    a=b=0;w=0
    for i,m in enumerate(h):
        assert not w and 0<=m<n*n and not (a|b)>>m&1
        if i%2==0:
            a|=1<<m
            if coordwin(a,m,n):w=1
        else:
            b|=1<<m
            if coordwin(b,m,n):w=-1
    return a,b,w
@functools.lru_cache(maxsize=300000)
def im(a,b,n):
    k,full,ls=G[n];e=full^(a|b);out=0
    for l in ls:
        if l&b==0 and (l&e).bit_count()==1:out|=l&e
    return out
def orient(h,n):
    a,b,w=replay(h,n);assert not w and (a|b)!=G[n][1]
    return (a,b) if len(h)%2==0 else (b,a)
@functools.lru_cache(maxsize=150000)
def forcing(a,b,n,d):
    # Independent implementation: generate candidate squares by scanning integer line masks;
    # recompute completion cells after actual candidate placement, not C++ adjacency.
    if d<=0:return False
    full=G[n][1]
    if (a|b)==full or any(b&l==l for l in G[n][2]):return False
    if im(a,b,n):return True
    if d<3:return False
    opp=im(b,a,n)
    if opp.bit_count()>1:return False
    e=full^(a|b);can=0
    for l in G[n][2]:
        if l&b==0 and (l&e).bit_count()==2:can|=l&e
    if opp:can&=opp
    return any(after(a,b,m,n,d) for m in bits(can))
@functools.lru_cache(maxsize=200000)
def after(a,b,m,n,d):
    x=a|1<<m
    if d>=1 and coordwin(x,m,n):return True
    if d<3 or im(b,x,n):return False
    t=im(x,b,n)
    if t.bit_count()>=2:
        # All legal defensive replies leave a distinct completion cell.
        assert all(t&~(1<<q) for q in bits(G[n][1]^(x|b)))
        return True
    return bool(t) and forcing(x,b|t,n,d-2)
@functools.lru_cache(maxsize=20000)
def base(a,b,n):
    k,full,ls=G[n];e=full^(a|b);own=im(a,b,n);opp=im(b,a,n);pair=C.defaultdict(int);E=list(bits(e))
    for l in ls:
        x=l&e
        if not l&b and x.bit_count()==2:
            i,j=list(bits(x));pair[i]|=1<<j;pair[j]|=1<<i
    bd={}
    for m in E:
        x=1<<m
        if own&x:v=(1,1)
        elif len(E)==1:v=(0,0)
        elif opp&~x:v=(-1,-1)
        elif ((own|pair[m])&~x).bit_count()>=2:v=(1,1)
        else:v=(-1,1)
        bd[m]=v
    if n==7:
        can=[m for m,(l,u) in bd.items() if all(l>=v[1] for j,v in bd.items() if j!=m)]
        g=min(can,key=lambda m:(-bd[m][0],not bool(own>>m&1),abs(m//7-3)+abs(m%7-3),m)) if can else None
    else:
        wins=[m for m in E if bd[m][0]==1]
        g=next(bits(own)) if own else wins[0] if wins else next(bits(opp)) if opp.bit_count()==1 else E[0] if len(E)==1 else None
    return bd,g
def inspect(h,n,z):
    a,b=orient(h,n);bd,g=base(a,b,n);assert z['gate']==g
    q={m:(l,u) for m,l,u in z['bounds']};assert list(q)==list(bd)
    assert (z['depth'],z['budget']) in TIERS and 0<=z['queries']<=z['budget'] and z['scans']>=len(G[n][2]) and z['checks']>=0
    if z['exhausted']:assert z['queries']==z['budget']
    CT['partial_receipts']+=1;CT['bound_cells']+=len(q);CT['queries']+=z['queries']
    for m,(l,u) in q.items():
        l0,u0=bd[m];assert l0<=l<=u<=u0
        if l>l0:
            assert l==u==1 and after(a,b,m,n,z['depth']),(n,h,m,z)
            CT['new_win_facts']+=1
        if u<u0:
            assert l==u==-1 and forcing(b,a|1<<m,n,z['depth']-1),(n,h,m,z)
            CT['new_loss_facts']+=1
    cert=z['cert'];allowed=[m for m,(l,u) in q.items() if all(l>=v[1] for j,v in q.items() if j!=m)]
    assert cert==(g if g is not None else min(allowed) if allowed else None)
    CT['partial_cert']+=cert is not None
    return q
def vote(h,n,r,lock,pre=None):
    a,b=orient(h,n);bd,g=base(a,b,n);E=list(bd)
    assert r['nn']>=0 and r['visits']>=0 and r['cpu_ns']>=0 and r['wall_ns']>=0
    if g is not None:
        assert r['kind']=='CERTIFICATE' and r['move']==g and r['nn']==r['visits']==0 and not r['votes']
        if n==15:assert r['gate']==g and r['native']['history']==h
        CT['common']+=1;return
    if r['kind']=='PARTIAL43':
        assert lock and r['nn']==r['visits']==0 and not r['votes']
        return
    v=r['votes'];assert set(m for m,_ in v)==set(E) and len(v)==len(E)
    assert all(type(x) is int and x>=0 for m,x in v)
    key=lambda i:(v[i][1],i if n==15 else -v[i][0])
    lead=max(range(len(v)),key=key);move=v[lead][0]
    assert (r['move'] if pre is None else pre)==move
    cap=32 if n==7 else 128
    assert sum(x for m,x in v)==r['visits']<=cap
    if n==7:
        assert [m for m,x in v]==E and r['nn']==1+r['visits']-r['terminal_leaves']
        d=min((v[lead][1]-x+int(m>move) for m,x in v if m!=move),default=10**9)
        assert r['distance']==d and r['lock']==lock and r['cap']==32
        if r['kind']=='VOTE_LOCK':assert lock and d>cap-r['visits'] and r['visits']<cap
        else:assert r['kind']=='CAP' and r['visits']==cap
    else:
        x=r['native'];assert x['history']==h and x['best']==move and x['NN_calls']==r['nn'] and x['visits']==r['visits'] and x['cap']==128 and x['locked']==lock
        assert r['nn']<=cap and r['certificate_checks']>0
        d=min((v[lead][1]-x+int(i<lead) for i,(m,x) in enumerate(v) if i!=lead),default=10**9)
        if r['kind']=='VOTE_LOCK':assert lock and d>cap-r['visits'] and r['visits']<cap and r['nn']<cap
        elif r['kind']=='VISIT_CAP':assert r['visits']==cap
        else:assert r['kind']=='NN_TARGET' and r['nn']==cap
        for bn,bnn,bd0,an,ann,ad,tar,cv in x['waves']:
            assert bn<=an<=cap and bnn<=ann<=cap and tar==cap-bnn and cv==min(8,cap-bn)
    CT['vote_rows']+=1;CT['legal_edges']+=len(v)
def changed(move,votes,q,n):
    if q[move][1]!=-1:return move
    can=[(x,i if n==15 else -m,m) for i,(m,x) in enumerate(votes) if q[m][1]!=-1]
    return max(can)[2] if can else move
def run():
    pl=load('PLAN.json');s=load('SUMMARY.json')
    jobs={j['id']:j for j in pl['jobs']};rjobs={j['id']:j for j in pl['rapfi_jobs']}
    assert len(jobs)==160 and len(rjobs)==96 and len({family(j['opening']) for j in jobs.values()})==160
    for x in pl['prior_locks']:
        p=ROOT/x['path'];assert sha(p)==x['sha256'];o=json.loads(p.read_text())
        assert not ({family(j['opening']) for j in o['jobs']}&{family(j['opening']) for j in jobs.values()})
    # verify Rapfi archived family lock and all historical contributing files; canonical D4 first-six keys.
    import sys
    sys.path.insert(0,str(D.parent/'quartz_kinetic37_20261126'))
    import experiment as e
    f0=e.read('PLAN.json');assert sha(e.D/'PLAN.json')==pl['rapfi_k37_plan_sha256']
    assert not {j['family'] for j in f0['jobs']}&{j['family'] for j in rjobs.values()}
    assert len({e.family(j['opening']) for j in rjobs.values()})==96
    for j in rjobs.values():assert e.family(j['opening'])==j['family']
    for x in pl['rapfi_prior_locks']:assert sha(ROOT/x['path'])==x['sha256']
    corrections={};arenas={};firsts={};raw_nn=0
    for mode in ('NN_SOURCE','R_SOURCE','NN_ARENA','R_ARENA'):
        n=7 if mode.startswith('NN') else 15;source=mode.endswith('SOURCE');jj=jobs if n==7 else rjobs
        seen=set();cnt=C.Counter();tiers=[C.Counter() for _ in TIERS];stages=C.defaultdict(C.Counter);pairs=C.defaultdict(list)
        for g in rows(mode):
            j=g['job'];assert j==jj[j['id']] and j['split']==mode.split('_')[1]
            key=j['id'],g['seed'],g['color'];assert key not in seen;seen.add(key)
            h=j['opening'].copy();rr=g['rows'];aa=iter(g['atlas']);count=0;nn=0
            for r in rr:
                p=1-2*(len(h)%2);q=p==g['color'];assert r['player']==p and r['ply']==len(h) and r['arm']==('Q' if q else 'B')
                a,b=orient(h,n);assert not (a|b)>>r['move']&1
                bd,gate=base(a,b,n)
                if source:
                    vote(h,n,r,q)
                    if gate is None:
                        z=next(aa);B,L=z['B'],z['L'];assert r==(L if q else B)
                        assert z['history']==g['history'],(mode,j['id'],g['seed'])
                        if n==15:assert B['native']['history']==L['native']['history']==h
                        if q:vote(h,n,B,False)
                        else:vote(h,n,L,True)
                        assert B['move']==L['move'] and L['nn']<=B['nn']
                        cnt['unresolved']+=1;cnt['B_nn']+=B['nn'];cnt['L_nn']+=L['nn'];cnt['B_cpu_ns']+=B['cpu_ns'];cnt['L_cpu_ns']+=L['cpu_ns']
                        cnt['corrected_atlas_histories']+=1
                        st=stages[str(len(h)//4*4)];st['roots']+=1
                        for i,x in enumerate(z['tiers']):
                            pz=x['receipt'];assert (pz['depth'],pz['budget'])==TIERS[i];bounds=inspect(h,n,pz)
                            cert=pz['cert'] is not None;mv=pz['cert'] if cert else changed(L['move'],L['votes'],bounds,n)
                            rep=not cert and mv!=L['move']
                            assert mv==x['move'] and rep==x['replacement'] and x['nn']==(0 if cert else L['nn'])
                            assert x['total_cpu_ns']==x['cpu_ns']+(0 if cert else L['cpu_ns'])
                            c=tiers[i];c['roots']+=1;c['cert']+=cert;c['replacement']+=rep;c['changed']+=mv!=B['move'];c['exhausted']+=pz['exhausted'];c['nn']+=x['nn'];c['cpu_ns']+=x['total_cpu_ns'];c['proof_cpu_ns']+=x['cpu_ns'];c['queries']+=pz['queries'];c['scans']+=pz['scans'];c['checks']+=pz['checks']
                            c['known_loss']+=sum(u==-1 for l,u in bounds.values());c['unknown']+=sum(l<u for l,u in bounds.values())
                            c['cert_unknown_value']+=cert and bounds[mv][0]<bounds[mv][1]
                            c['cert_saved_B_nn']+=B['nn'] if cert else 0;c['lock_saved_nn']+=B['nn']-L['nn'] if not cert else 0
                            if i==2:st['cert']+=cert;st['replacement']+=rep
                        count+=1
                else:
                    z=r.get('partial')
                    if q and gate is None:
                        assert z is not None and (z['depth'],z['budget'])==TIERS[-1]
                        bounds=inspect(h,n,z);cnt['partial_attempts']+=1;cnt['queries']+=z['queries']
                        if z['cert'] is not None:
                            assert r['kind']=='PARTIAL43' and r['move']==z['cert'] and not r['replacement'] and r['original_move'] is None
                            cnt['partial_cert']+=1;vote(h,n,r,True)
                        else:
                            vote(h,n,r,True,r['original_move']);mv=changed(r['original_move'],r['votes'],bounds,n)
                            assert mv==r['move'] and r['replacement']==(mv!=r['original_move'])
                            cnt['replacement']+=r['replacement']
                    else:
                        assert not z;vote(h,n,r,q)
                    tag='candidate' if q else 'baseline'
                    for k in ('nn','cpu_ns','wall_ns'):cnt[tag+'_'+k]+=r[k]
                    cnt[tag+'_moves']+=1
                nn+=r['nn'];cnt['common']+=gate is not None;cnt['moves']+=1;h.append(r['move'])
            assert next(aa,None) is None
            a,b,w=replay(h,n);assert h==g['history'] and (w or (a|b)==G[n][1]) and w==g['winner'] and g['score2']==1+g['color']*w and nn==g['nn']
            cnt['nn']+=nn;cnt['games']+=1;cnt['score2']+=g['score2'];cnt['wins']+=g['score2']==2;cnt['draws']+=g['score2']==1;pairs[j['id'],g['seed']].append(g)
        assert len(seen)==(256 if mode=='NN_SOURCE' else 32 if mode=='R_SOURCE' else 768 if mode=='NN_ARENA' else 128)
        assert cnt['nn']==s['counts'][mode]['nn'] and cnt['moves']==s['counts'][mode]['moves'] and cnt['games']==s['counts'][mode]['games']
        raw_nn+=cnt['nn']+s['counts'][mode].get('atlas_extra_nn',0)
        if source:
            x=s['source'][mode];assert cnt['moves']==cnt['common']+cnt['unresolved']
            for k,v in x['counts'].items():assert cnt[k]==v,(mode,k,cnt[k],v)
            for (d,cap),c in zip(TIERS,tiers):
                for k,v in c.items():assert x['tiers'][str(d)][k]==v,(mode,k,v,x['tiers'][str(d)][k])
            corrections[mode]=dict(corrected_atlas_histories=cnt['corrected_atlas_histories'],ply={k:dict(v) for k,v in sorted(stages.items(),key=lambda x:int(x[0]))})
        else:
            x=s['arena'][mode]
            for k in ('games','wins','draws','partial_attempts','partial_cert','replacement','queries','candidate_moves','candidate_nn','baseline_moves','baseline_nn'):assert x[k]==cnt[k],(mode,k,x[k],cnt[k])
            assert x['losses']==cnt['games']-cnt['wins']-cnt['draws']
            assert abs(x['score']-cnt['score2']/(2*cnt['games']))<1e-15
            for k,v in [('nn','NN_reduction'),('cpu_ns','CPU_reduction'),('wall_ns','wall_reduction')]:
                z=1-F(cnt['candidate_'+k]*cnt['baseline_moves'],cnt['baseline_'+k]*cnt['candidate_moves']);assert abs(float(z)-x[v])<1e-14
            assert all(len(v)==2 for v in pairs.values())
            idpairs=sum(v[0]['history']==v[1]['history'] for v in pairs.values());assert idpairs==x['identical_pairs']
            arenas[mode]=dict(cnt,identical_pairs=idpairs)
        print('AUDITED',mode,dict(cnt),flush=True)
    save('CORRECTED_SOURCE_SUMMARY.json',dict(corrections=corrections,original_summary_sha256=sha(D/'SUMMARY.json'),scope='Canonical ply occupancy correction only: collector v0 atlas history alias replaced by actual raw game prefix at recorded ply; all receipt bounds and source aggregate totals independently audited. Raw immutable. Source hypothetical CPU omits duplicated common K33 gate when cert skips MCTS, not actual arena CPU.'))
    out=dict(counts=dict(CT),raw_NN=raw_nn,corrected_source=corrections,arena=arenas,proof_cache=forcing.cache_info()._asdict(),scope='independent coordinate replay, Python integer completion sets and sufficient forcing proof verification of all new W/L facts, interval certificate and weak replacement checks, same-horizon vote parity and actual cost aggregates; independent bootstrap not repeated')
    save('AUDIT.json',out);print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':run()
