"""Prospectively stratified early interventions, exact stop and two actual continuations."""
from arena import Server,identity,check,write,read,D,ROOT,BIN,sha,replay,proof_reference,orbit_board,fixtures
import collections as C,gzip,json,hashlib,random,statistics as S,math,time,sys,os
ARMS=('DS0','DSF','DSX','DSL','DSV','DSB','DSSTOP')
def rows(name):
    with gzip.open(D/name,'rt') as f:
        for line in f:yield json.loads(line)
def emit(f,x):
    f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n');f.flush()
def host():
    z={}
    for line in open('/proc/vmstat'):
        k,v=line.split()
        if k in ('pswpin','pswpout'):z[k]=int(v)
    return dict(mono=time.monotonic(),load=os.getloadavg(),swap=z)
def host_end(start):
    end=host();dt=end['mono']-start['mono']
    return dict(start=start,end=end,seconds=dt,swap_pages_s=sum(end['swap'][k]-start['swap'][k] for k in end['swap'])/dt)
def reach(r):
    ca=r['candidates'];es=r['edges'];assert [e['move'] for e in es]==[m for m,v in ca[:len(es)]]
    n=[e['n'] for e in es]+[0]*(len(ca)-len(es))
    L=max(range(len(n)),key=lambda i:(n[i],i));assert ca[L][0]==r['best'] and sum(n)==r['root_n']>0
    d=[None if i==L else n[L]-n[i]+int(i<L) for i in range(len(n))]
    return L,n,d,min((v for v in d if v is not None),default=None)
def dscheck(a,h,arm,p):
    check(a,h,arm,32,17)
    if a['cause']=='GUARDED':assert a['decision'] is None;return
    x=a['decision'];assert x['prefix']==p and x['exact_merge']==(arm=='DSB')
    assert a['cause']==('DS_LOCKED' if x['stopped'] else 'DS_FIXED')
    ps=x['phases'];ends=sorted(set([p,p+4,p+8,p+16,32]))
    assert [z['visits'] for z in ps]==ends[:len(ps)]
    if x['stopped']:assert arm=='DSSTOP' and a['visits']<32 and ps[-1]['locked']
    else:assert a['visits']==32
    prev=a['initial'];last=None;prefix=ps[0]['result'];L,n,d,md=reach(prefix);ca=prefix['candidates'];es=prefix['edges']
    reachable=lambda i:d[i] is not None and d[i]<=32-p
    cand=[i for i in range(len(es)) if n[i]>0 and reachable(i)]
    idx=max(cand,key=lambda i:(n[i],i),default=None) if arm=='DSF' else next((i for i in range(len(es),len(ca)) if reachable(i)),None) if arm=='DSX' else L if arm=='DSL' else None
    expected=None if idx is None else dict(rank=idx,move=ca[idx][0],d=d[idx],n=n[idx],materialized=idx<len(es),leader=ca[L][0])
    assert x['target']==expected
    for z in ps:
        r=z['result'];c=z['counters'];v=z['visits'];_,counts,dd,min_d=reach(r)
        assert r['root_n']==v and r['root_hash']==prev['root_hash'] and r['candidates']==prev['candidates']
        assert z['min_d']==min_d and z['locked']==(min_d is None or min_d>32-v)
        assert r['evaluator_calls']==1+c['base_evals']+c['extra_evals']
        assert abs(sum(e['w'] for e in r['edges'])-r['root_w'])<1e-6
        old={e['move']:e['n'] for e in prev['edges']}
        for e in r['edges']:
            N=e['n'];W=e['w'];M=e['m2'];assert type(N) is int and N>=old.get(e['move'],0)
            assert e['vl']==0 and abs(e['vvalue'])<1e-6 and abs(W)<=N+1e-6 and -1e-7<=M<=(N-W*W/N if N else 0)+1e-5
        if v==p+4 and idx is not None:
            assert counts[idx]-n[idx]==4 and all(counts[i]==n[i] for i in range(len(n)) if i!=idx)
        if v==p:assert c['extra_evals']==c['proof_scans']==c['merge_blends']==0
        if v>p+8 and last is not None:assert all(c[k]==last[k] for k in ('extra_evals','proof_scans','merge_blends','ledger_observations'))
        if arm=='DSSTOP' and v!=p+4 and v<32 and z['locked']:assert z is ps[-1]
        prev=r;last=c
    assert ps[-1]['result']==a['result']
    tr=x['b_trace'];assert len(tr)==(p+8 if arm=='DSB' else 0)
    if tr:
        assert [t['mode'] for t in tr]==[5]*p+[3]*8
        assert all(t['blends']==0 for t in tr[:p])
        assert sum(t['blends'] for t in tr)==last['merge_blends']
def family(h):
    b,p,w=replay(h[:6],'freestyle');assert w is None
    return hashlib.sha256(orbit_board(b,p,'freestyle')).hexdigest()
def historical():
    fam=set();boards=set()
    def record(h):
        if len(h)>=6:fam.add(family(h))
        b,p,w=replay(h,'freestyle');boards.add(orbit_board(b,p,'freestyle'))
    for h in fixtures().values():record(h)
    c=json.loads((ROOT/'docs/research/quartz_tactical08_20261125/C07_COHORT.json').read_text())
    for r in c['CAL']+c['TEST']:record(r['history'])
    for fp in (ROOT/'docs/research/quartz_preflight07_20261125/P2_CORPUS').glob('shard_*/*.json.gz'):
        with gzip.open(fp,'rt') as f:g=json.load(f)
        h=list(g['job']['opening'])
        for row in g['moves']:record(h);h.append(row['played_move'])
        record(h)
    for fp in (ROOT/'docs/research/quartz_rapfi09_20261125').glob('PAIR_GAME_*.json.gz'):
        with gzip.open(fp,'rt') as f:g=json.load(f)
        for row in g['rows']:record(row['response']['history'])
        record(g['final_history'])
    for n in ('C06_SOURCE_GAMES.jsonl.gz','C08_CONTINUATIONS.jsonl.gz'):
        with gzip.open(ROOT/'docs/research/quartz_rapfi10_20261125'/n,'rt') as f:
            for line in f:
                g=json.loads(line)
                for row in g['rows']:record(row['response']['history'])
                if 'root' in g:record(g['root']['history'])
                record(g['final_history'])
    return fam,boards
def smoke():
    f=fixtures();out=[];reject=[]
    with Server() as s:
        for name in ('trans_a','long','immediate'):
            h=f[name]
            for p in (8,16):
                aa=[s.search(h,arm,32,17,prefix=p) for arm in ARMS]
                for a in aa:dscheck(a,h,a['arm'],p);out.append(dict(fixture=name,response=a))
                if aa[0]['decision'] is not None:
                    assert all(a['decision']['phases'][0]['result']==aa[0]['decision']['phases'][0]['result'] and a['root_eval']==aa[0]['root_eval'] for a in aa)
                    standard=s.search(h,'STANDARD_PW',32,17)
                    assert standard['result']==aa[0]['result'] and aa[-1]['best']==standard['best']
                else:assert all(a['best']==aa[0]['best'] and a['cause']=='GUARDED' for a in aa)
                print('SMOKE',name,p,[(a['arm'],a['visits'],a['best']) for a in aa],flush=True)
        for q in [
            dict(cmd='research_rapfi_shared_search',rule='freestyle',history=f['trans_a'],arm='DSF',visits=32,seed=17),
            dict(cmd='research_rapfi_shared_search',rule='freestyle',history=f['trans_a'],arm='DSF',visits=32,seed=17,prefix=12),
            dict(cmd='research_rapfi_shared_search',rule='freestyle',history=f['trans_a'],arm='STANDARD_PW',visits=32,seed=17,prefix=8)
        ]:
            a=s.request(q);assert 'error' in a;reject.append(a)
    write('C06_SMOKE.json',dict(identity=identity(),rows=out,rejected=reject,standard_parity=4))
def plan():
    old,boards=historical();nold=len(old);rng=random.Random(2026112511);cells=[15*i+j for i in range(4,11) for j in range(4,11)];jobs=[]
    for i in range(24):
        for _ in range(1000):
            h=rng.sample(cells,6);k=family(h)
            if k not in old:break
        else:raise RuntimeError('opening exhaustion')
        old.add(k);jobs.append(dict(id=f'G{i:02d}',group='TRAIN' if i<6 else 'CAL' if i<12 else 'TEST',stratum='UNFILTERED' if i%2==0 else 'SENSITIVE',family=k,opening=h))
    write('C06_SOURCE_PLAN.json',dict(identity=identity(),jobs=jobs,excluded_families=nold,excluded_boards=len(boards),cap=64,seed=17,rule='freestyle'))
def play(s,h,cap):
    h=list(h);rr=[]
    while True:
        b,p,w=replay(h,'freestyle')
        if w is not None:return h,w,rr
        a=s.search(h,'STANDARD_PW',cap,17)
        rr.append(dict(ply=len(h),player=p,move=a['best'],response=a))
        h.append(a['best'])
def source():
    p=read('C06_SOURCE_PLAN.json');assert p['identity']==identity();_,boards=historical();co=[];start=host()
    n=D/'C06_SOURCE_GAMES.jsonl.gz';assert not n.exists()
    with Server() as s,gzip.open(n,'wt',compresslevel=3) as f:
        for j in p['jobs']:
            h,w,rr=play(s,j['opening'],64)
            candidates=[]
            for row in rr:
                a=row['response'];b,pl,_=replay(a['history'],'freestyle');k=orbit_board(b,pl,'freestyle')
                if not a['proof']['own'] and not a['proof']['opponent'] and k not in boards:
                    candidates.append((a,k))
            candidates=sorted(candidates,key=lambda x:(abs(len(x[0]['history'])-18),len(x[0]['history'])))[:3]
            probes=[];selected=None
            for a,k in candidates:
                h0=a['history'];z=s.search(h0,'STANDARD_PW',8,17);L,counts,d,md=reach(z['result'])
                probes.append(dict(history=h0,min_d=md,response=z))
                if j['stratum']=='UNFILTERED' or md is not None and md<=4:
                    selected=dict(id=j['id'],group=j['group'],stratum=j['stratum'],family=j['family'],history=h0,player=z['player'],min_d8=md);co.append(selected);boards.add(k);break
            emit(f,dict(job=j,final_history=h,winner_black=w,rows=rr,probes=probes,selected=selected))
            print('SOURCE',j['id'],j['group'],j['stratum'],len(h),'selected',None if selected is None else (len(selected['history']),selected['min_d8']),flush=True)
    write('C06_COHORT.json',dict(identity=identity(),roots=co,missing=[j['id'] for j in p['jobs'] if j['id'] not in {r['id'] for r in co}],source_sha256=sha(n),host=host_end(start)))
def intervention():
    co=read('C06_COHORT.json')['roots'];name=D/'C07_RAW.jsonl.gz';assert not name.exists();start=host()
    with Server() as s,gzip.open(name,'wt',compresslevel=3) as f:
        for i,r in enumerate(co):
            for p in (8,16):
                arms=ARMS[(i+p)%7:]+ARMS[:(i+p)%7];aa={}
                for arm in arms:
                    a=s.search(r['history'],arm,32,17,prefix=p);dscheck(a,r['history'],arm,p);aa[arm]=a
                    emit(f,dict(root=r,prefix=p,response=a))
                z=aa['DS0']
                assert all(a['decision']['phases'][0]['result']==z['decision']['phases'][0]['result'] and a['root_eval']==z['root_eval'] for a in aa.values())
                assert aa['DSSTOP']['best']==z['best']
                assert all(x['result']==y['result'] for x,y in zip(aa['DSSTOP']['decision']['phases'],z['decision']['phases']))
                if p==8:assert z['decision']['phases'][0]['min_d']==r['min_d8']
            print('INTERVENTION',r['id'],r['stratum'],flush=True)
    write('C07_RAW_LOCK.json',dict(identity=identity(),requests=len(co)*14,raw_sha256=sha(name),host=host_end(start)))
def recommend(a,p):
    ps=a['decision']['phases'];z=ps[0]['result']['best']
    mid=next((x['result']['best'] for x in ps if x['visits']==p+8),ps[-1]['result']['best'])
    return (z,mid,a['best'])
def continuation():
    raw=list(rows('C07_RAW.jsonl.gz'));assert sha(D/'C07_RAW.jsonl.gz')==read('C07_RAW_LOCK.json')['raw_sha256']
    req={}
    for x in raw:
        r=x['root'];a=x['response']
        for m in recommend(a,x['prefix']):req[r['id'],m]=r
    assert len(req)<=24*29
    name=D/'C08_CONTINUATIONS.jsonl.gz';assert not name.exists();start=host()
    with Server() as s,gzip.open(name,'wt',compresslevel=3) as f:
        for (i,m),r in sorted(req.items()):
            for cap in (64,256):
                h,w,rr=play(s,r['history']+[m],cap)
                emit(f,dict(root=r,action=m,cap=cap,final_history=h,winner_black=w,utility=(1+r['player']*w)/2,rows=rr))
            print('CONT',i,m,flush=True)
    write('C08_CONT_LOCK.json',dict(identity=identity(),unique_actions=len(req),count=2*len(req),raw_sha256=sha(name),host=host_end(start)))
if __name__=='__main__':{'smoke':smoke,'plan':plan,'source':source,'intervention':intervention,'continuation':continuation}[sys.argv[1]]()
