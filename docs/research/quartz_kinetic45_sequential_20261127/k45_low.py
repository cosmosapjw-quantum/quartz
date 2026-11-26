import collections as C,gzip,json,random,sys,time,pathlib,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from k45_plan import D,ROOT,T,SEEDS,TIERS,SEED,WORKERS,load,save,sha
from seq45 import staged,Session as ProofSession
from hybrid45 import bundle,replace,compare,exact
OLD=D.parent/'quartz_kinetic44_incremental_20261127'
K42=D.parent/'quartz_kinetic42_threshold_20261127'
K41=D.parent/'quartz_kinetic41_exact_20261126'
H=(20,26,32)
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def emit(f,x):f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n')
def locked(d,mode):
    lk=json.loads((d/(mode+'_LOCK.json')).read_text())
    for x in lk['files']:
        p=d/x['file'];assert sha(p)==x['sha256'];yield from raw(p)
def rows(mode):yield from locked(D,mode)
def cold(h,n):
    sys.path.insert(0,str(OLD))
    from inc44 import probe
    t=time.perf_counter_ns();c=time.process_time_ns();z=probe(h,n,7,4096)
    return dict(receipt=z,cpu_ns=time.process_time_ns()-c,wall_ns=time.perf_counter_ns()-t)
def bench():
    p=D/'BENCHMARK.jsonl.gz';assert not p.exists();ct=C.Counter()
    with gzip.open(p,'wt',compresslevel=3) as f:
        for lane,n in [('NN',7),('R',15)]:
            for split in ('CAL','TEST'):
                for g in locked(OLD,lane+'_'+split):
                    for a in g['atlas']:
                        h=a['history'];i=sum(ct.values());ops=['cold','staged'] if i%2==0 else ['staged','cold']
                        rr={k:(cold(h,n) if k=='cold' else staged(h,n)) for k in ops}
                        assert rr['cold']['receipt']==a['tiers'][-1]['receipt']
                        s=rr['staged'];assert s['receipt']['gate'] is None
                        prev=None
                        for r in s['stages']:
                            assert r['queries']<=r['budget']
                            if prev:
                                assert all(x[0]==y[0] and x[1]<=y[1]<=y[2]<=x[2] for x,y in zip(prev['bounds'],r['bounds']))
                                for k in ('queries','scans','checks','skipped','hits','entries','positive','negative'):assert r[k]>=prev[k]
                            prev=r
                        # Archived proof bundle for historical NN TRAIN. Exact fallback separate.
                        e=exact(h) if n==7 and len(h)>=33 and s['receipt']['cert'] is None else None
                        bd={m:(l,u) for m,l,u in s['receipt']['bounds']};cert=s['receipt']['cert']
                        if e and e['complete']:
                            cert=e['best']
                            for m,l,u in e['bounds']:
                                x,y=bd[m];bd[m]=(max(l,x),min(u,y));assert bd[m][0]<=bd[m][1]
                        z=dict(seq=s,exact=e,cert=cert,bounds=[[m,l,u] for m,(l,u) in bd.items()],cpu_ns=s['cpu_ns']+(e['cpu_ns'] if e else 0),wall_ns=s['wall_ns']+(e['wall_ns'] if e else 0))
                        emit(f,dict(lane=lane,split=split,job=g['job']['id'],seed=g['seed'],history=h,B=a['B'],L=a['L'],order=ops,cold=rr['cold'],proof=z))
                        ct[lane]+=1
    assert ct=={'NN':1709,'R':772},ct
    save('BENCHMARK_LOCK.json',dict(file=p.name,sha256=sha(p),counts=dict(ct),old_source_locks={m:sha(OLD/(m+'_LOCK.json')) for m in ('NN_CAL','NN_TEST','R_CAL','R_TEST')},source=sha(D/'seq45.cpp'),geometry=sha(OLD/'inc44.cpp'),library=sha(T/'seq45.so'),scope='cold serial alternating with full staged lifecycle; K44 historical TRAIN, exact fallback only NN'))
    print('BENCH',dict(ct),flush=True)
def benchrows():
    x=load('BENCHMARK_LOCK.json');p=D/x['file'];assert sha(p)==x['sha256'];yield from raw(p)
def late():
    p=D/'LATE.jsonl.gz';assert not p.exists()
    prior={(r['a'],r['b']):r['values'] for r in raw(K41/'INDEPENDENT.jsonl.gz') if r.get('complete') and ('corpus_complete' in r['tags'] or 'corpus_censored' in r['tags'])}
    new={(r['a'],r['b']):r['values'] for r in raw(K42/'INDEPENDENT.jsonl.gz') if r.get('complete') and 'fresh' in r['tags']}
    ct=C.Counter()
    with gzip.open(p,'wt',compresslevel=3) as f:
        for r in raw(K42/'BENCHMARK.jsonl.gz'):
            h=r['job']['opening'];key=r['a'],r['b'];truth=prior.get(key) if r['job']['frame']=='K41' else new.get(key)
            z=bundle(h,7);q=z['seq']['receipt'];ct['roots']+=1;ct['truth']+=truth is not None;ct['partial']+=q['cert'] is not None;ct['hybrid']+=z['cert'] is not None
            if z['exact']:ct['exact_attempt']+=1;ct['exact_complete']+=z['exact']['complete'];ct['exact_nodes']+=z['exact']['nodes']
            if truth is not None:
                for m,l,u in z['bounds']:assert l<=truth[str(m)]<=u
                if z['cert'] is not None:assert truth[str(z['cert'])]==max(truth.values());ct['verified_hybrid']+=1
            emit(f,dict(job=r['job'],a=r['a'],b=r['b'],truth=truth,proof=z))
    assert ct['roots']==384 and ct['truth']==377,ct
    save('LATE_LOCK.json',dict(file=p.name,sha256=sha(p),counts=dict(ct),truth_sources={str(d.relative_to(ROOT)):sha(d) for d in (K41/'INDEPENDENT.jsonl.gz',K42/'INDEPENDENT.jsonl.gz',K42/'BENCHMARK.jsonl.gz')},scope='all 384 historical late roots, 377 independent full action labels, no TRAIN fitting'))
    print('LATE',dict(ct),flush=True)
def nested(ev,h,B=None):
    from k45_engine import Session
    ss=Session(ev.ev,h)
    ls={str(k):ss.advance(k,True) for k in H}
    assert ls['32']['kind']!='CERTIFICATE'
    if B is None:B=ev.search(h,32,False)
    assert B['move']==ls['32']['move'] and ls['20']['nn']<=ls['26']['nn']<=ls['32']['nn']<=B['nn']
    return B,ls
def options(h,B,ls,z):
    opts={k:compare(z,B,L,7) for k,L in ls.items()}
    assert opts['32']['gain2'][0]>=0
    return dict(history=h.copy(),B=B,L=ls,proof=z,options=opts)
def train_worker(arg):
    seed,i,rr=arg
    from k45_engine import Puct,load_eval
    ev=Puct(load_eval(seed));p=D/f'TRAIN_{seed}_{i}.jsonl.gz';assert not p.exists();ct=C.Counter()
    with gzip.open(p,'wt',compresslevel=3) as f:
        for r in rr:
            B,L=nested(ev,r['history'],r['B']);assert L['32']['nn']==r['L']['nn'] and L['32']['move']==r['L']['move']
            a=options(r['history'],B,L,r['proof'])
            emit(f,dict(seed=seed,job=r['job'],split=r['split'],**a));ct['roots']+=1;ct['nn']+=L['32']['nn']
    assert ev.ev.rows==ct['nn']
    print('TRAIN_SHARD',seed,i,dict(ct),flush=True)
    return dict(file=p.name,sha256=sha(p),counts=dict(ct))
def train():
    rr=[r for r in benchrows() if r['lane']=='NN'];tasks=[(s,i,[r for r in rr if r['seed']==s][i::4]) for s in SEEDS for i in range(4)]
    with ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(train_worker,tasks))
    c=sum((C.Counter(x['counts']) for x in fs),C.Counter());assert c['roots']==1709
    save('TRAIN_LOCK.json',dict(files=fs,counts=dict(c),benchmark=sha(D/'BENCHMARK_LOCK.json'),source=sha(D/'k45_low.py')))
    print('TRAIN',dict(c),flush=True)
def aggregate(rr):
    c=C.Counter();v={str(h):C.Counter() for h in H}
    for a in rr:
        c['roots']+=1;c['B_nn']+=a['B']['nn'];c['B_cpu_ns']+=a['B']['cpu_ns']
        for k,x in a['options'].items():
            y=v[k];lo,hi=x['gain2'];y['nn']+=x['nn'];y['gain2_lo']+=lo;y['gain2_hi']+=hi;y['cert']+=x['cert'];y['replacement']+=x['replacement'];y['changes']+=x['move']!=a['B']['move'];y['cost_cpu_ns']+=x['cost_cpu_ns'];y['exact_gain']+=lo==hi;y['strict_positive']+=lo>0;y['strict_negative']+=hi<0
    return dict(counts=dict(c),options={k:dict(y) for k,y in v.items()})
def fit():
    x=aggregate(rows('TRAIN'));v=x['options'];h=next(h for h in H if v[str(h)]['gain2_lo']>=0 and v[str(h)]['nn']<=v['32']['nn'])
    save('FROZEN.json',dict(cap=h,train=x,train_sha256=sha(D/'TRAIN_LOCK.json'),scope='earliest fixed cap with nonnegative aggregate certified lower gain and NN<=Q32; empirical finite sample, not posterior VOC'))
    print('FIT',h,json.dumps(x,separators=(',',':')),flush=True)
def nn_worker(arg):
    mode,s,i,js=arg
    from k45_engine import Puct,load_eval,from_history,won,FULL
    ev=Puct(load_eval(s));ct=C.Counter();p=D/f'{mode}_{s}_{i}.jsonl.gz';assert not p.exists();source=mode in ('NN_CAL','NN_TEST')
    cap=20 if mode=='NN_LOW' else 32
    def end(h):
        a,b=from_history(h);return 1 if won(a) else -1 if won(b) else 0 if a|b==FULL else None
    with gzip.open(p,'wt',compresslevel=3) as f:
        for j in js:
            for color in (j['color_order'][:1] if source else j['color_order']):
                h=j['opening'].copy();rr=[];aa=[]
                while end(h) is None:
                    side=1-2*(len(h)%2);q=side==color
                    if source:
                        r=ev.search(h,32,False) if not q else ev.search(h,32,True)
                        if r['kind']!='CERTIFICATE':
                            # One extra nested locked session, B32 search only if not already executed.
                            B,L=nested(ev,h,r if not q else None)
                            if q:assert r['move']==L['32']['move'] and r['nn']==L['32']['nn']
                            z=bundle(h,7);a=options(h,B,L,z);aa.append(a);ct['atlas_roots']+=1
                            ct['atlas_extra_nn']+=L['32']['nn']+(B['nn'] if q else 0)
                    else:r=ev.quartz45(h,cap) if q else ev.search(h,32,False)
                    r.update(arm='Q' if q else 'B',player=side,ply=len(h));rr.append(r);h.append(r['move'])
                    ct['nn']+=r['nn'];ct['moves']+=1;ct['common']+=r['kind']=='CERTIFICATE'
                w=end(h);emit(f,dict(job=j,seed=s,color=color,history=h,winner=w,score2=1+color*w,rows=rr,atlas=aa,nn=sum(r['nn'] for r in rr)));ct['games']+=1
    assert ev.ev.rows==ct['nn']+ct['atlas_extra_nn']
    print('SHARD',mode,s,i,dict(ct),flush=True)
    return dict(file=p.name,sha256=sha(p),counts=dict(ct))
def rapfi_worker(arg):
    mode,s,i,js=arg
    sys.path.insert(0,str(D.parent/'quartz_kinetic37_20261126'))
    import experiment as e
    e.BIN=T/'mcts_demo';assert sha(e.BIN)==load('PLAN.json')['rapfi_identity']['actual_binary_sha256']
    ct=C.Counter();p=D/f'{mode}_{s}_{i}.jsonl.gz';assert not p.exists()
    with e.Server(e.BIN) as server,gzip.open(p,'wt',compresslevel=3) as f:
        for j in js:
            for color in j['color_order']:
                h=j['opening'].copy();rr=[]
                while True:
                    b,side,w=e.replay(h,'freestyle')
                    if w is not None:break
                    q=side==color;t=time.perf_counter_ns();c=time.process_time_ns()
                    z=bundle(h,15) if q else None;g=z['seq']['receipt']['gate'] if z else None
                    if z and g is None and z['cert'] is not None:
                        r=dict(move=z['cert'],nn=0,visits=0,kind='PARTIAL45',votes=[],original_move=None,replacement=False,gate=None,certificate_checks=0,native=None,partial=z,proof_cpu_ns=z['cpu_ns'])
                        native_cpu=0
                    else:
                        a=server.search(h,e.seed(j,len(h)),128,q);native_cpu=sum(a['cpu_ns']);gate=a['proof']['gate'];v=e.votes(a['result']) or []
                        r=dict(move=a['best'],nn=a['NN_calls'],visits=a['visits'],kind='CERTIFICATE' if gate is not None else a['cause'],votes=v,gate=gate,certificate_checks=a.get('certificate_checks',0),native=e.small(a))
                        if z:
                            assert g==gate
                            o=r['move'];r['move']=replace(o,v,z['bounds'],15) if gate is None else o
                            r.update(original_move=o,replacement=r['move']!=o,partial=z if gate is None else None,proof_cpu_ns=z['cpu_ns'])
                    r.update(cpu_ns=time.process_time_ns()-c+native_cpu,wall_ns=time.perf_counter_ns()-t,arm='Q' if q else 'B',player=side,ply=len(h));rr.append(r);h.append(r['move'])
                    ct['nn']+=r['nn'];ct['moves']+=1;ct['common']+=r['kind']=='CERTIFICATE'
                emit(f,dict(job=j,seed=0,color=color,history=h,winner=w,score2=1+color*w,rows=rr,atlas=[],nn=sum(r['nn'] for r in rr)));ct['games']+=1
    print('SHARD',mode,s,i,dict(ct),flush=True)
    return dict(file=p.name,sha256=sha(p),counts=dict(ct))
def run(mode):
    pl=load('PLAN.json');nn=mode.startswith('NN');split=mode.split('_')[1];split='ARENA' if split in ('FIXED','ADMIT','ARENA','LOW') else split
    js=[j for j in pl['jobs' if nn else 'rapfi_jobs'] if j['split']==split]
    seeds=SEEDS if nn else (0,);tasks=[(mode,s,i,js[i::4]) for s in seeds for i in range(4)]
    random.Random(SEED+len(mode)).shuffle(tasks)
    with ProcessPoolExecutor(max_workers=WORKERS if nn else 4,mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(nn_worker if nn else rapfi_worker,tasks))
    ct=sum((C.Counter(x['counts']) for x in fs),C.Counter())
    save(mode+'_LOCK.json',dict(files=fs,counts=dict(ct),plan=sha(D/'PLAN.json'),source=sha(D/'k45_low.py'),engine=sha(D/'k45_engine.py'),hybrid=sha(D/'hybrid45.py'),library=sha(T/'seq45.so')))
    print(mode,dict(ct),flush=True)
def source(mode):return aggregate(a for g in rows(mode) for a in g['atlas'])
def admit():
    h=load('FROZEN.json')['cap'];x=source('NN_CAL');v=x['options'];ok=v[str(h)]['gain2_lo']>=0 and v[str(h)]['nn']<=v['32']['nn']
    save('ADMISSION.json',dict(cap=h if ok else 32,selected=h,admitted=ok,cal=x,frozen_sha256=sha(D/'FROZEN.json')))
    print('ADMISSION',h,ok,json.dumps(x,separators=(',',':')),flush=True)
def arena(mode):
    import numpy as np
    gg=list(rows(mode));families=C.defaultdict(list)
    for g in gg:families[g['job']['id']].append(g)
    assert len(families)==48
    def stat(gs):
        x=[0]*20
        for g in gs:
            x[0]+=g['score2'];x[1]+=1;x[2]+=g['score2']==2;x[3]+=g['score2']==1;x[4]+=g['score2']==0
            for r in g['rows']:
                i=5 if r['arm']=='Q' else 9;x[i]+=1;x[i+1]+=r['nn'];x[i+2]+=r['cpu_ns'];x[i+3]+=r['wall_ns']
                z=r.get('partial')
                if z:
                    x[13]+=1;x[14]+=z['cert'] is not None;x[15]+=r['replacement'];x[16]+=z['seq']['receipt']['queries'];x[17]+=z['seq']['receipt']['hits'];x[18]+=z['exact'] is not None;x[19]+=bool(z['exact'] and z['exact']['complete'])
        return x
    def rec(v):
        red=lambda k:1-(v[5+k]/v[5])/(v[9+k]/v[9])
        return dict(games=int(v[1]),wins=int(v[2]),draws=int(v[3]),losses=int(v[4]),score=float(v[0]/(2*v[1])),candidate_moves=int(v[5]),candidate_nn=int(v[6]),baseline_moves=int(v[9]),baseline_nn=int(v[10]),NN_reduction=red(1),CPU_reduction=red(2),wall_reduction=red(3),attempts=int(v[13]),cert=int(v[14]),replacement=int(v[15]),queries=int(v[16]),hits=int(v[17]),exact_attempt=int(v[18]),exact_complete=int(v[19]))
    a=np.asarray([stat(v) for _,v in sorted(families.items())],dtype=float);rng=np.random.default_rng(SEED+45);bs=a[rng.integers(0,len(a),(4000,len(a)))].sum(axis=1)
    sc=np.sort(bs[:,0]/(2*bs[:,1]));re=np.sort(1-(bs[:,6]/bs[:,5])/(bs[:,10]/bs[:,9]))
    x=rec(stat(gg));x.update(NI_lower=float(sc[199]),score_CI=[float(sc[99]),float(sc[3899])],NN_CI=[float(re[99]),float(re[3899])],by_seed={str(s):rec(stat([g for g in gg if g['seed']==s])) for s in sorted({g['seed'] for g in gg})})
    return x
def summarize():
    b={}
    for a in benchrows():
        lane=a['lane'];c=b.setdefault(lane,C.Counter());o=a['cold'];s=a['proof']['seq'];q=s['receipt'];r=o['receipt']
        c['roots']+=1;c['cold_cpu_ns']+=o['cpu_ns'];c['seq_cpu_ns']+=s['cpu_ns'];c['cold_wall_ns']+=o['wall_ns'];c['seq_wall_ns']+=s['wall_ns']
        for k,z in [('cold',r),('seq',q)]:
            for field in ('queries','scans','checks','skipped','hits','entries','positive','negative'):c[k+'_'+field]+=z.get(field,0)
            c[k+'_cert']+=z['cert'] is not None;c[k+'_exhausted']+=z['exhausted']
        c['different_bounds']+=r['bounds']!=q['bounds'];c['different_cert']+=r['cert']!=q['cert'];c['seq_only_cert']+=q['cert'] is not None and r['cert'] is None;c['cold_only_cert']+=r['cert'] is not None and q['cert'] is None;c['stages']+=len(s['stages'])
    modes=('NN_FIXED','NN_ADMIT','R_ARENA') if (D/'NN_ADMIT_LOCK.json').exists() else ('NN_FIXED','R_ARENA')
    out=dict(benchmark={k:dict(v) for k,v in b.items()},late=load('LATE_LOCK.json')['counts'],source={m:source(m) for m in ('NN_CAL','NN_TEST')},train=load('FROZEN.json'),admission=load('ADMISSION.json'),arena={m:arena(m) for m in modes},counts={m:load(m+'_LOCK.json')['counts'] for m in ('TRAIN','NN_CAL','NN_TEST',*modes)})
    save('SUMMARY.json',out);print('SUMMARY',json.dumps(out,separators=(',',':')),flush=True)
if __name__=='__main__':
    m=sys.argv[1]
    if m in ('bench','late','train','fit','admit','summarize'):globals()[m]()
    else:run(m)
