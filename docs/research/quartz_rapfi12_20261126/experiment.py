"""P5 cost-only calibration and independent paired mature-evaluator games."""
from arena import Server,identity,check,write,read,D,ROOT,BIN,sha,replay,proof_reference,orbit_board,fixtures
from pathlib import Path
from fractions import Fraction
import collections as C,gzip,json,hashlib,random,statistics as S,math,time,sys,os,importlib.util
ARMS=('P5PW','P5FULL','P5Q','P5ALLOC')
GRID=(32,48,64,96,128,160,192,256)
def rows(name):
    with gzip.open(D/name,'rt') as f:
        for line in f:yield json.loads(line)
def emit(f,x):
    f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n');f.flush()
def host():
    v={}
    for l in open('/proc/vmstat'):
        k,x=l.split()
        if k in ('pswpin','pswpout'):v[k]=int(x)
    return dict(mono=time.monotonic(),load=os.getloadavg(),swap=v)
def host_end(start,peak):
    e=host();dt=e['mono']-start['mono'];sw=sum(e['swap'][k]-start['swap'][k] for k in e['swap'])/dt
    return dict(start=start,end=e,seconds=dt,swap_pages_s=sw,max_load=peak,timing_gate=sw<=512 and peak<=12)
def ticks(pid):
    a=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return int(a[11])+int(a[12])
class Meter:
    def __init__(self,s):
        self.s=s;self.start=host();self.peak=self.start['load'][0]
        ch=Path(f'/proc/{s.p.pid}/task/{s.p.pid}/children').read_text().split();assert len(ch)==1,ch
        self.child=int(ch[0]);self.hz=os.sysconf('SC_CLK_TCK')
    def search(self,h,arm,cap):
        r0=ticks(self.s.p.pid);c0=ticks(self.child);p0=time.process_time_ns()
        a=self.s.search(h,arm,cap,17)
        py=time.process_time_ns()-p0
        rt=ticks(self.s.p.pid)-r0;ct=ticks(self.child)-c0
        a['measure']=dict(rust_ticks=rt,rapfi_ticks=ct,tick_hz=self.hz,python_cpu_ns=py)
        assert rt>=0 and ct>=0
        self.peak=max(self.peak,os.getloadavg()[0])
        if arm in ARMS:p5check(a,h,arm,cap)
        return a
    def end(self):return host_end(self.start,self.peak)
def reach(r):
    ca=r['candidates'];es=r['edges'];assert [e['move'] for e in es]==[m for m,v in ca[:len(es)]]
    n=[e['n'] for e in es]+[0]*(len(ca)-len(es))
    L=max(range(len(n)),key=lambda i:(n[i],i));assert ca[L][0]==r['best'] and sum(n)==r['root_n']>0
    return min((n[L]-n[i]+int(i<L) for i in range(len(n)) if i!=L),default=None)
def p5check(a,h,arm,cap):
    check(a,h,arm,cap,17)
    if a['cause']=='GUARDED':assert a['resource'] is None;return
    r=a['resource'];n=a['visits'];d=reach(a['result'])
    assert r['final_d']==d
    selected=r['selected_cap'];assert n<=selected
    if arm=='P5ALLOC':
        assert r['hard']==(r['prefix_d'] is not None and r['prefix_d']<=4)
        assert selected==(cap*2 if r['hard'] else cap//2)
    else:assert selected==cap and r['hard'] is None
    if arm!='P5Q':
        assert r['prefix_best'] in a['legal'] and r['checks']>=0
        assert n>=16 and n%8==0
        assert a['cause']==('P5_LOCKED' if r['locked'] else 'P5_FIXED')
        if not r['locked']:assert n==selected
    if r['locked']:assert a['cause']=='P5_LOCKED' and n<selected and n>=16 and n%8==0 and (d is None or d>selected-n)
    else:assert a['cause']!='P5_LOCKED'
def family(h):
    b,p,w=replay(h[:6],'freestyle');assert w is None
    return hashlib.sha256(orbit_board(b,p,'freestyle')).hexdigest()
def historical():
    p=ROOT/'docs/research/quartz_rapfi11_20261125/experiment.py'
    spec=importlib.util.spec_from_file_location('previous_sources',p);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    fam,boards=old.historical()
    for n in ('C06_SOURCE_GAMES.jsonl.gz','C08_CONTINUATIONS.jsonl.gz'):
        with gzip.open(p.parent/n,'rt') as f:
            for line in f:
                g=json.loads(line)
                for row in g['rows']:
                    h=row['response']['history'];fam.add(family(h))
                    b,p0,w=replay(h,'freestyle');boards.add(orbit_board(b,p0,'freestyle'))
                h=g['final_history'];fam.add(family(h))
                b,p0,w=replay(h,'freestyle');boards.add(orbit_board(b,p0,'freestyle'))
    return fam,boards
def smoke():
    f=fixtures();out=[];reject=[]
    with Server() as s:
        for nm in ('trans_a','long','immediate'):
            h=f[nm]
            for cap in (32,128,256):
                a=[s.search(h,m,cap,17) for m in ARMS]
                for x in a:p5check(x,h,x['arm'],cap)
                assert all(x['root_eval']==a[0]['root_eval'] for x in a)
                parity=[]
                for x in (a[0],a[1],a[3]):
                    c=x['resource']['selected_cap'] if x['resource'] else cap
                    std=s.search(h,'STANDARD_FULL' if x['arm']=='P5FULL' else 'STANDARD_PW',c,17)
                    assert x['best']==std['best'];parity.append(dict(arm=x['arm'],selected_cap=c,standard=std))
                    if x['resource'] and not x['resource']['locked']:assert x['result']==std['result']
                if a[0]['resource']:
                    pre=s.search(h,'STANDARD_PW',16,17)
                    for x in (a[0],a[3]):assert x['resource']['prefix_d']==reach(pre['result']) and x['resource']['prefix_best']==pre['best']
                else:pre=None
                out.append(dict(fixture=nm,cap=cap,responses=a,parity=parity,prefix_reference=pre))
                print('SMOKE',nm,cap,[(x['arm'],x['visits'],x['best'],x['cause']) for x in a],flush=True)
        for q in [dict(cmd='research_rapfi_shared_search',rule='freestyle',history=[],arm='P5ALLOC',visits=16,seed=17),dict(cmd='research_rapfi_shared_search',rule='freestyle',history=[],arm='P5PW',visits=64,seed=17,prefix=8),dict(cmd='research_rapfi_shared_search',rule='renju',history=[],arm='P5Q',visits=128,seed=17)]:
            z=s.request(q);assert 'error' in z;reject.append(z)
    write('C06_SMOKE.json',dict(identity=identity(),rows=out,rejected=reject))
def plan():
    old,boards=historical();n0=len(old);rng=random.Random(2026112612);cells=[15*i+j for i in range(4,11) for j in range(4,11)];jobs=[]
    for i in range(16):
        for _ in range(1000):
            h=rng.sample(cells,6);k=family(h)
            if k not in old:break
        else:raise RuntimeError('opening exhaustion')
        old.add(k);jobs.append(dict(id=f'G{i:02d}',group='TRAIN' if i<4 else 'CAL' if i<8 else 'TEST',family=k,opening=h))
    write('C06_SOURCE_PLAN.json',dict(identity=identity(),jobs=jobs,excluded_families=n0,excluded_boards=len(boards),grid=list(GRID),reference_cap=128,seed=17,rule='freestyle'))
def play(m,h,policy):
    h=list(h);rr=[]
    while True:
        b,p,w=replay(h,'freestyle')
        if w is not None:return h,w,rr
        arm,cap=policy(h,p);a=m.search(h,arm,cap)
        rr.append(dict(ply=len(h),player=p,move=a['best'],response=a))
        h.append(a['best'])
def source():
    p=read('C06_SOURCE_PLAN.json');assert p['identity']==identity()
    n=D/'C06_SOURCE_GAMES.jsonl.gz';assert not n.exists()
    with Server() as s,gzip.open(n,'wt',compresslevel=3) as f:
        m=Meter(s)
        for j in p['jobs']:
            if j['group']=='TEST':continue
            h,w,rr=play(m,j['opening'],lambda h,p:('STANDARD_PW',64))
            emit(f,dict(job=j,final_history=h,winner_black=w,rows=rr))
            print('SOURCE',j['id'],j['group'],len(h),w,flush=True)
        end=m.end()
    write('C06_SOURCE_LOCK.json',dict(identity=identity(),raw_sha256=sha(n),host=end))
def source_roots(group):
    assert sha(D/'C06_SOURCE_GAMES.jsonl.gz')==read('C06_SOURCE_LOCK.json')['raw_sha256']
    out=[]
    for g in rows('C06_SOURCE_GAMES.jsonl.gz'):
        j=g['job']
        if j['group']!=group:continue
        for row in g['rows']:
            out.append(dict(id=j['id'],family=j['family'],group=group,history=row['response']['history'],ply=row['ply'],player=row['player']))
    return out
def grid():
    co=source_roots('TRAIN');arms=[('P5PW',128)]+[(a,b) for a in ARMS[1:] for b in GRID]
    n=D/'C07_GRID_RAW.jsonl.gz';assert not n.exists()
    with Server() as s,gzip.open(n,'wt',compresslevel=3) as f:
        m=Meter(s)
        for i,r in enumerate(co):
            off=i%len(arms);aa={}
            for arm,cap in arms[off:]+arms[:off]:
                a=m.search(r['history'],arm,cap);aa[arm,cap]=a
                emit(f,dict(root=r,response=a))
            z=aa['P5PW',128]
            for (arm,cap),a in aa.items():
                assert z['root_eval']==a['root_eval']
                if arm=='P5ALLOC' and a['resource']:assert a['resource']['prefix_d']==z['resource']['prefix_d'] and a['resource']['prefix_best']==z['resource']['prefix_best']
            if i%20==0:print('GRID_ROOT',i+1,len(co),flush=True)
        end=m.end()
    write('C07_GRID_LOCK.json',dict(identity=identity(),raw_sha256=sha(n),roots=len(co),requests=len(co)*len(arms),host=end))
def mixfit():
    lock=read('C07_GRID_LOCK.json');assert sha(D/'C07_GRID_RAW.jsonl.gz')==lock['raw_sha256']
    vals=C.defaultdict(list)
    for r in rows('C07_GRID_RAW.jsonl.gz'):
        a=r['response'];vals[a['arm'],a['cap']].append(a['transport_ns'])
    n=lock['roots'];assert all(len(v)==n for v in vals.values())
    sums={k:sum(v) for k,v in vals.items()};target=sums['P5PW',128]
    fit={'P5PW':dict(low=128,high=128,num=0,den=1,bracket=True,train_low_sum=target,train_high_sum=target)}
    for arm in ARMS[1:]:
        cand=sorted((sums[arm,c],c) for c in GRID)
        lower=[x for x in cand if x[0]<=target];upper=[x for x in cand if x[0]>=target]
        if lower and upper:
            l,lo=max(lower);h,hi=min(upper)
            q=Fraction(target-l,h-l) if h>l else Fraction(0)
            fit[arm]=dict(low=lo,high=hi,num=q.numerator,den=q.denominator,bracket=True,train_low_sum=l,train_high_sum=h)
            assert (1-q)*l+q*h==target
        else:
            x,c=min(cand,key=lambda a:(abs(a[0]-target),a[1]))
            fit[arm]=dict(low=c,high=c,num=0,den=1,bracket=False,train_low_sum=x,train_high_sum=x)
    write('C07_FROZEN_MIX.json',dict(identity=identity(),target_wall_ns=target/n,n=n,grid_sum_ns={f'{a}/{c}':v for (a,c),v in sums.items()},fit=fit,grid_sha256=lock['raw_sha256'],mix_seed='P5-20261126-v1',selection='wall-only measured bracket, no wins used'))
    print('FROZEN_MIX',json.dumps(fit),flush=True)
def choose(fit,fam,ply,arm):
    x=fit['fit'][arm];b=hashlib.sha256(f"{fit['mix_seed']}|{fam}|{ply}|{arm}".encode()).digest()
    u=int.from_bytes(b[:8],'big')
    return x['high'] if u*x['den']<x['num']*2**64 else x['low']
def cal():
    fit=read('C07_FROZEN_MIX.json');co=source_roots('CAL');n=D/'C07_CAL_RAW.jsonl.gz';assert not n.exists()
    with Server() as s,gzip.open(n,'wt',compresslevel=3) as f:
        m=Meter(s)
        for i,r in enumerate(co):
            aa={};order=ARMS[i%4:]+ARMS[:i%4]
            for arm in order:
                cap=choose(fit,r['family'],r['ply'],arm);a=m.search(r['history'],arm,cap);aa[arm]=a
                emit(f,dict(root=r,response=a))
            assert all(x['root_eval']==aa['P5PW']['root_eval'] for x in aa.values())
            if aa['P5ALLOC']['resource']:assert aa['P5ALLOC']['resource']['prefix_d']==aa['P5PW']['resource']['prefix_d']
        end=m.end()
    write('C07_CAL_LOCK.json',dict(identity=identity(),raw_sha256=sha(n),fit_sha256=sha(D/'C07_FROZEN_MIX.json'),roots=len(co),requests=len(co)*4,host=end))
def arena():
    fit=read('C07_FROZEN_MIX.json');p=read('C06_SOURCE_PLAN.json');assert fit['identity']==p['identity']==identity()
    n=D/'C08_ARENA_RAW.jsonl.gz';assert not n.exists()
    with Server() as s,gzip.open(n,'wt',compresslevel=3) as f:
        m=Meter(s)
        for i,j in enumerate(x for x in p['jobs'] if x['group']=='TEST'):
            arms=ARMS[1:];off=i%3
            for candidate in arms[off:]+arms[:off]:
                for color in (1,-1) if i%2==0 else (-1,1):
                    def policy(h,p):
                        arm=candidate if p==color else 'P5PW';return arm,choose(fit,j['family'],len(h),arm)
                    h,w,rr=play(m,j['opening'],policy)
                    emit(f,dict(job=j,candidate=candidate,candidate_color=color,winner_black=w,score=(1+color*w)/2,final_history=h,rows=rr))
                    print('GAME',j['id'],candidate,color,len(h),w,flush=True)
        end=m.end()
    write('C08_ARENA_LOCK.json',dict(identity=identity(),raw_sha256=sha(n),fit_sha256=sha(D/'C07_FROZEN_MIX.json'),games=48,host=end))
if __name__=='__main__':{'smoke':smoke,'plan':plan,'source':source,'grid':grid,'fit':mixfit,'cal':cal,'arena':arena}[sys.argv[1]]()
