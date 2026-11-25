"""Actual bounded natural roots, common prefix, and policy continuation outcomes."""
from arena import Server,identity,check,write,read,D,ROOT,BIN,sha,replay,proof_reference,orbit_board,fixtures
import collections as C, gzip, json, hashlib, random, statistics as S, math,time,sys,os
ARMS=('AX0','AXS','AXX','AXV','AXB','AXALL')
MODES=dict(AX0=(0,0,0),AXS=(1,0,0),AXX=(2,0,0),AXV=(0,3,0),AXB=(0,0,3),AXALL=(3,3,3))
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
def axischeck(a,h,arm):
    check(a,h,arm,96,17)
    if a['cause']=='GUARDED':assert a['axis'] is None;return
    ax=a['axis'];m,v,b=MODES[arm]
    assert (ax['mask'],ax['vmode'],ax['bmode'])==(m,v,b) and ax['exact_merge']==(b==3)
    assert a['cause']=='AXIS_FIXED' and len(ax['phases'])==3
    prev=a['initial'];last={}
    for i,z in enumerate(ax['phases']):
        r=z['result'];c=z['counters']
        assert r['root_n']==z['visits']==32*(i+1) and r['candidates']==prev['candidates']
        assert r['root_hash']==prev['root_hash'] and r['evaluator_calls']==1+c['base_evals']+c['extra_evals']
        assert sum(e['n'] for e in r['edges'])==r['root_n'] and abs(sum(e['w'] for e in r['edges'])-r['root_w'])<1e-6
        assert [e['move'] for e in r['edges']]==[m for m,p in r['candidates'][:len(r['edges'])]]
        assert r['best'] in a['legal']
        pe={e['move']:e for e in prev['edges']}
        for e in r['edges']:
            n=e['n'];w=e['w'];M=e['m2']
            assert type(n) is int and n>=0 and math.isfinite(w) and math.isfinite(M)
            assert e['vl']==0 and abs(e['vvalue'])<1e-6 and abs(w)<=n+1e-6 and -1e-7<=M<=(n-w*w/n if n else 0)+1e-5
            if e['move'] in pe:assert n>=pe[e['move']]['n']
        assert all(c[k]>=last.get(k,0) for k in ('base_evals','extra_evals','select_nodes','width_nodes','merge_blends','ledger_observations'))
        if i==0:assert c['select_nodes']==c['width_nodes']==c['extra_evals']==c['merge_blends']==0
        if i==2:
            assert all(c[k]==last[k] for k in ('select_nodes','width_nodes','extra_evals','merge_blends','ledger_observations','proof_scans'))
        prev=r;last=c
    assert prev==a['result']
    tr=ax['b_trace']
    assert len(tr)==(64 if b==3 else 0)
    if b==3:
        assert [x['mode'] for x in tr]==[5]*32+[3]*32 and sum(x['blends'] for x in tr)==last['merge_blends']
        assert all(x['blends']==0 for x in tr[:32])
    else:assert last['ledger_observations']==last['merge_blends']==0
def smoke():
    f=fixtures();out=[]
    with Server() as s:
        for nm in ('trans_a','long','immediate'):
            h=f[nm];a=[s.search(h,arm,96,17) for arm in ARMS]
            for x in a:axischeck(x,h,x['arm']);x['fixture']=nm;out.append(x)
            if a[0]['axis'] is not None:
                assert all(x['axis']['phases'][0]['result']==a[0]['axis']['phases'][0]['result'] and x['root_eval']==a[0]['root_eval'] for x in a)
                old=s.search(h,'STANDARD_PW',96,17)
                assert old['result']==a[0]['result'],nm
            else:assert all(x['cause']=='GUARDED' and x['best']==a[0]['best'] for x in a)
            print('SMOKE',nm,[(x['arm'],x['visits'],x['NN_calls'],x['best']) for x in a],flush=True)
        q=dict(cmd='research_rapfi_shared_search',rule='freestyle',history=f['trans_a'],arm='AXALL',visits=64,seed=17)
        rejected=s.request(q);assert 'error' in rejected
    write('C06_SMOKE.json',dict(identity=identity(),rows=out,reject=rejected,standard_parity=2))
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
        h=list(g['opening'])
        for row in g['rows']:record(h);h.append(row['played'])
        record(h)
    return fam,boards
def plan():
    old,boards=historical();nold=len(old);r=random.Random(2026112510);cells=[15*i+j for i in range(4,11) for j in range(4,11)]
    jobs=[]
    for i in range(16):
        for attempt in range(1000):
            h=r.sample(cells,6);key=family(h)
            if key not in old:break
        else:raise RuntimeError('opening exhaustion')
        old.add(key);jobs.append(dict(id=f'G{i:02d}',group='TRAIN' if i<4 else 'CAL' if i<8 else 'TEST',family=key,opening=h,desired_player=1 if i%2==0 else -1))
    write('C06_SOURCE_PLAN.json',dict(identity=identity(),jobs=jobs,excluded_families=nold,excluded_boards=len(boards),cap=64,seed=17,rule='freestyle',max_games=16,policy='STANDARD_PW'))
def play(s,h,cap=64):
    h=list(h);rr=[]
    while True:
        b,p,w=replay(h,'freestyle')
        if w is not None:return h,w,rr
        a=s.search(h,'STANDARD_PW',cap,17)
        rr.append(dict(ply=len(h),player=p,move=a['best'],response=a))
        h.append(a['best'])
def source():
    p=read('C06_SOURCE_PLAN.json');assert p['identity']==identity();old,boards=historical()
    co=[];start=host();name=D/'C06_SOURCE_GAMES.jsonl.gz';assert not name.exists()
    with Server() as s,gzip.open(name,'wt',compresslevel=3) as f:
        for j in p['jobs']:
            h,w,r=play(s,j['opening'],64)
            emit(f,dict(job=j,final_history=h,winner_black=w,rows=r))
            elig=[]
            for row in r:
                a=row['response'];b,pl,win=replay(a['history'],'freestyle');key=orbit_board(b,pl,'freestyle')
                if not a['proof']['own'] and not a['proof']['opponent'] and key not in boards:
                    elig.append((a,key))
            desired=[(a,k) for a,k in elig if a['player']==j['desired_player']]
            elig=desired or elig
            if elig:
                a,key=min(elig,key=lambda x:(abs(len(x[0]['history'])-18),len(x[0]['history'])))
                boards.add(key);co.append(dict(id=j['id'],group=j['group'],family=j['family'],history=a['history'],player=a['player'],source_ply=len(a['history']),rule='freestyle'))
            print('SOURCE',j['id'],j['group'],len(h),w,'root',co[-1]['source_ply'] if co and co[-1]['id']==j['id'] else None,flush=True)
    write('C06_COHORT.json',dict(roots=co,source_sha256=sha(name),identity=identity(),missing=[j['id'] for j in p['jobs'] if j['id'] not in {r['id'] for r in co}],host=host_end(start)))
def intervention():
    co=read('C06_COHORT.json')['roots'];name=D/'C07_RAW.jsonl.gz';assert not name.exists();start=host()
    with Server() as s,gzip.open(name,'wt',compresslevel=3) as f:
        for i,r in enumerate(co):
            arms=ARMS[i%len(ARMS):]+ARMS[:i%len(ARMS)];prefix=None;ev=None
            for arm in arms:
                a=s.search(r['history'],arm,96,17);axischeck(a,r['history'],arm)
                assert a['axis'] is not None
                q=a['axis']['phases'][0]['result']
                if prefix is None:prefix=q;ev=a['root_eval']
                else:assert prefix==q and ev==a['root_eval']
                emit(f,dict(root=r,response=a))
            print('INTERVENTION',r['id'],r['group'],'prefix',prefix['best'],flush=True)
    write('C07_RAW_LOCK.json',dict(identity=identity(),requests=len(co)*6,raw_sha256=sha(name),host=host_end(start)))
def continuation():
    raw=list(rows('C07_RAW.jsonl.gz'));assert sha(D/'C07_RAW.jsonl.gz')==read('C07_RAW_LOCK.json')['raw_sha256']
    req={}
    for v in raw:
        r=v['root'];a=v['response']
        for p in a['axis']['phases']:
            m=p['result']['best'];req[(r['id'],m)]=r
    assert len(req)<=16*13
    name=D/'C08_CONTINUATIONS.jsonl.gz';assert not name.exists();start=host()
    with Server() as s,gzip.open(name,'wt',compresslevel=3) as f:
        for (id,m),r in sorted(req.items()):
            h,w,rr=play(s,r['history']+[m],64)
            emit(f,dict(root=r,action=m,final_history=h,winner_black=w,utility=(1+r['player']*w)/2,rows=rr))
            print('CONT',id,m,len(h),w,flush=True)
    write('C08_CONT_LOCK.json',dict(identity=identity(),count=len(req),raw_sha256=sha(name),host=host_end(start)))
if __name__=='__main__':{'smoke':smoke,'plan':plan,'source':source,'intervention':intervention,'continuation':continuation}[sys.argv[1]]()
