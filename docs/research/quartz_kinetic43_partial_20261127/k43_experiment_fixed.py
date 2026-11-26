"""K43 same-evaluator natural trajectory census and disjoint arenas."""
import collections as C,gzip,json,multiprocessing as mp,random,sys,time,pathlib
from concurrent.futures import ProcessPoolExecutor
from k43_plan import D,ROOT,T,SEEDS,TIERS,WORKERS,SEED,load,save,sha
from partial43 import probe
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def emit(f,x):f.write(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n')
def probe_timed(h,n,d,cap):
    t=time.perf_counter_ns();c=time.process_time_ns();z=probe(h,n,d,cap)
    return dict(receipt=z,cpu_ns=time.process_time_ns()-c,wall_ns=time.perf_counter_ns()-t)
def change(move,votes,z,native=False):
    bad={m for m,l,u in z['bounds'] if u==-1}
    if move not in bad:return move
    pool=[(v,i if native else -m,m) for i,(m,v) in enumerate(votes) if m not in bad]
    return max(pool)[2] if pool else move
def atlas(h,n,B,L):
    assert B['move']==L['move'] and L['nn']<=B['nn']
    z=[]
    for d,b in TIERS:
        x=probe_timed(h,n,d,b);r=x['receipt'];assert r['gate'] is None
        a=r['cert'] if r['cert'] is not None else change(L['move'],L['votes'],r,n==15)
        x.update(move=a,nn=0 if r['cert'] is not None else L['nn'],replacement=r['cert'] is None and a!=L['move'],total_cpu_ns=x['cpu_ns']+(0 if r['cert'] is not None else L['cpu_ns']))
        z.append(x)
    return dict(history=h.copy(),B=B,L=L,tiers=z)
def nn_worker(arg):
    mode,s,i,js=arg
    from k43_engine import Puct,load_eval,from_history,won,FULL
    ev=Puct(load_eval(s));ct=C.Counter();t=time.monotonic()
    path=D/f'{mode}_{s}_{i}.jsonl.gz';assert not path.exists()
    def end(h):
        a,b=from_history(h);return 1 if won(a) else -1 if won(b) else 0 if a|b==FULL else None
    with gzip.open(path,'wt',compresslevel=3) as f:
        for j in js:
            for color in (j['color_order'][:1] if mode=='NN_SOURCE' else j['color_order']):
                h=j['opening'].copy();rr=[];aa=[]
                while end(h) is None:
                    p=1-2*(len(h)%2);q=p==color
                    if mode=='NN_SOURCE':
                        r=ev.search(h,32,q)
                        if r['kind']!='CERTIFICATE':
                            other=ev.search(h,32,not q)
                            B,L=(other,r) if q else (r,other)
                            a=atlas(h,7,B,L);aa.append(a);ct['atlas_roots']+=1;ct['atlas_extra_nn']+=other['nn']
                    else:r=ev.quartz43(h) if q else ev.search(h,32,False)
                    r.update(arm='Q' if q else 'B',player=p,ply=len(h));rr.append(r);h.append(r['move'])
                    ct['nn']+=r['nn'];ct['moves']+=1;ct['cert']+=r['kind']=='CERTIFICATE'
                w=end(h);g=dict(job=j,seed=s,color=color,history=h,winner=w,score2=1+color*w,rows=rr,atlas=aa,nn=sum(r['nn'] for r in rr))
                emit(f,g);ct['games']+=1
            ct['items']+=1
    assert ev.ev.rows==ct['nn']+ct['atlas_extra_nn']
    print('SHARD',mode,s,i,dict(ct),flush=True)
    return dict(file=path.name,sha256=sha(path),counts=dict(ct),elapsed_s=time.monotonic()-t)
def rapfi_worker(arg):
    mode,s,i,js=arg
    sys.path.insert(0,str(D.parent/'quartz_kinetic37_20261126'))
    import experiment as e
    e.BIN=T/'mcts_demo'
    assert sha(e.BIN)==load('PLAN.json')['rapfi_identity']['actual_binary_sha256']
    ct=C.Counter();t=time.monotonic();path=D/f'{mode}_{s}_{i}.jsonl.gz';assert not path.exists()
    def search(server,h,j,lock,partial=False):
        start=time.perf_counter_ns();cpu=time.process_time_ns()
        z=probe_timed(h,15,7,4096) if partial else None
        receipt=z['receipt'] if z else None
        if receipt and receipt['gate'] is None and receipt['cert'] is not None:
            r=dict(move=receipt['cert'],nn=0,visits=0,kind='PARTIAL43',votes=[],original_move=None,replacement=False,gate=None,certificate_checks=0,native=None,partial=receipt,proof_cpu_ns=z['cpu_ns'])
            native_cpu=0
        else:
            a=server.search(h,e.seed(j,len(h)),128,lock)
            native_cpu=sum(a['cpu_ns']);gate=a['proof']['gate'];v=e.votes(a['result']) or []
            r=dict(move=a['best'],nn=a['NN_calls'],visits=a['visits'],kind='CERTIFICATE' if gate is not None else a['cause'],votes=v,gate=gate,certificate_checks=a.get('certificate_checks',0),native=e.small(a))
            if z:
                assert receipt['gate']==gate,(h,receipt['gate'],gate)
                o=r['move'];r['move']=change(o,v,receipt,True) if gate is None else o
                r.update(original_move=o,replacement=r['move']!=o,partial=receipt if gate is None else None,proof_cpu_ns=z['cpu_ns'])
        r.update(cpu_ns=time.process_time_ns()-cpu+native_cpu,wall_ns=time.perf_counter_ns()-start)
        return r
    with e.Server(e.BIN) as server,gzip.open(path,'wt',compresslevel=3) as f:
        for j in js:
            for color in (j['color_order'][:1] if mode=='R_SOURCE' else j['color_order']):
                h=j['opening'].copy();rr=[];aa=[]
                while True:
                    b,p,w=e.replay(h,'freestyle')
                    if w is not None:break
                    q=p==color
                    r=search(server,h,j,q,partial=mode=='R_ARENA' and q)
                    if mode=='R_SOURCE' and r['kind']!='CERTIFICATE':
                        other=search(server,h,j,not q);B,L=(other,r) if q else (r,other)
                        a=atlas(h,15,B,L);aa.append(a);ct['atlas_roots']+=1;ct['atlas_extra_nn']+=other['nn']
                    r.update(arm='Q' if q else 'B',player=p,ply=len(h));rr.append(r);h.append(r['move'])
                    ct['nn']+=r['nn'];ct['moves']+=1;ct['cert']+=r['kind']=='CERTIFICATE'
                emit(f,dict(job=j,seed=0,color=color,history=h,winner=w,score2=1+color*w,rows=rr,atlas=aa,nn=sum(r['nn'] for r in rr)))
                ct['games']+=1
            ct['items']+=1
    print('SHARD',mode,s,i,dict(ct),flush=True)
    return dict(file=path.name,sha256=sha(path),counts=dict(ct),elapsed_s=time.monotonic()-t)
def execute(mode):
    pl=load('PLAN.json');nn=mode.startswith('NN');split=mode.split('_')[1]
    js=[j for j in pl['jobs' if nn else 'rapfi_jobs'] if j['split']==split]
    tasks=[];seeds=SEEDS if nn else (0,);n=4
    for s in seeds:
        for i in range(n):tasks.append((mode,s,i,js[i::n]))
    random.Random(SEED+len(mode)).shuffle(tasks);t=time.monotonic()
    with ProcessPoolExecutor(max_workers=WORKERS if nn else 4,mp_context=mp.get_context('spawn')) as ex:fs=list(ex.map(nn_worker if nn else rapfi_worker,tasks))
    ct=C.Counter()
    for x in fs:ct.update(x['counts'])
    save(mode+'_LOCK.json',dict(plan=sha(D/'PLAN.json'),source=sha(D/'partial43.cpp'),lib=sha(T/'partial43.so'),engine=sha(D/'k43_engine.py'),collector=sha(pathlib.Path(__file__).resolve()),files=fs,counts=dict(ct),elapsed_s=time.monotonic()-t))
    print(mode,dict(ct),flush=True)
def rows(mode):
    for fi in load(mode+'_LOCK.json')['files']:
        p=D/fi['file'];assert sha(p)==fi['sha256'];yield from raw(p)
def sum_source(mode):
    c=C.Counter();ts=[C.Counter() for _ in TIERS];family=C.defaultdict(list);stages=C.defaultdict(C.Counter)
    for g in rows(mode):
        c['games']+=1;c['moves']+=len(g['rows']);c['common']+=sum(r['kind']=='CERTIFICATE' for r in g['rows'])
        for a in g['atlas']:
            B,L=a['B'],a['L'];h=a['history'];c['unresolved']+=1;c['B_nn']+=B['nn'];c['L_nn']+=L['nn'];c['B_cpu_ns']+=B['cpu_ns'];c['L_cpu_ns']+=L['cpu_ns']
            stage=stages[str(len(h)//4*4)];stage['roots']+=1
            for i,x in enumerate(a['tiers']):
                q=x['receipt'];z=ts[i];cert=q['cert'] is not None
                z['roots']+=1;z['cert']+=cert;z['replacement']+=x['replacement'];z['changed']+=x['move']!=B['move'];z['exhausted']+=q['exhausted'];z['nn']+=x['nn'];z['cpu_ns']+=x['total_cpu_ns'];z['proof_cpu_ns']+=x['cpu_ns'];z['queries']+=q['queries'];z['scans']+=q['scans'];z['checks']+=q['checks']
                z['known_loss']+=sum(u==-1 for m,l,u in q['bounds']);z['unknown']+=sum(l<u for m,l,u in q['bounds'])
                z['cert_unknown_value']+=cert and next(l<u for m,l,u in q['bounds'] if m==q['cert'])
                z['cert_saved_B_nn']+=B['nn'] if cert else 0
                z['lock_saved_nn']+=B['nn']-L['nn'] if not cert else 0
                if i==2:stage['cert']+=cert;stage['replacement']+=x['replacement']
    assert c['moves']==c['common']+c['unresolved']
    out=dict(counts=dict(c),tiers={str(d):dict(z,NN_reduction=1-z['nn']/c['B_nn'],CPU_reduction=1-z['cpu_ns']/c['B_cpu_ns'],coverage=z['cert']/c['unresolved']) for (d,b),z in zip(TIERS,ts)},ply={k:dict(v) for k,v in sorted(stages.items(),key=lambda x:int(x[0]))},scope='actual natural policy trajectories; unresolved roots census; physical extra queries accounted in locks; CPU estimate proof+locked path, omits separate Python K33 gate when NN certificate bypasses search; not arena performance')
    return out
def sum_arena(mode):
    import numpy as np
    gs=list(rows(mode));pairs=C.defaultdict(list);ct=C.Counter()
    for g in gs:pairs[g['job']['id']].append(g)
    assert len(pairs)==(96 if mode=='NN_ARENA' else 64) and all(len(v)==(8 if mode=='NN_ARENA' else 2) for v in pairs.values())
    def stat(gg):
        v=[0]*17
        for g in gg:
            v[0]+=g['score2'];v[1]+=1;v[2]+=g['score2']==2;v[3]+=g['score2']==1;v[4]+=g['score2']==0
            for r in g['rows']:
                i=5 if r['arm']=='Q' else 9
                v[i]+=1;v[i+1]+=r['nn'];v[i+2]+=r['cpu_ns'];v[i+3]+=r['wall_ns']
                z=r.get('partial')
                if z:
                    v[13]+=1;v[14]+=z['cert'] is not None;v[15]+=r['replacement'];v[16]+=z['queries']
        return v
    def rec(v):
        red=lambda k:1-(v[5+k]/v[5])/(v[9+k]/v[9])
        return dict(games=int(v[1]),wins=int(v[2]),draws=int(v[3]),losses=int(v[4]),score=float(v[0]/(2*v[1])),candidate_moves=int(v[5]),candidate_nn=int(v[6]),baseline_moves=int(v[9]),baseline_nn=int(v[10]),NN_reduction=red(1),CPU_reduction=red(2),wall_reduction=red(3),partial_attempts=int(v[13]),partial_cert=int(v[14]),replacement=int(v[15]),queries=int(v[16]))
    a=np.asarray([stat(v) for _,v in sorted(pairs.items())],dtype=float);rng=np.random.default_rng(SEED+43);b=a[rng.integers(0,len(a),(4000,len(a)))].sum(axis=1)
    sc=np.sort(b[:,0]/(2*b[:,1]));red=np.sort(1-(b[:,6]/b[:,5])/(b[:,10]/b[:,9]))
    z=rec(stat(gs));z.update(NI_lower=float(sc[199]),score_CI=[float(sc[99]),float(sc[3899])],NN_CI=[float(red[99]),float(red[3899])])
    z['by_seed']={str(s):rec(stat([g for g in gs if g['seed']==s])) for s in sorted({g['seed'] for g in gs})}
    z['identical_pairs']=sum(v[0]['history']==v[1]['history'] for vv in pairs.values() for s in {g['seed'] for g in vv} for v in [[g for g in vv if g['seed']==s]])
    return z
def summarize():
    x=dict(source={m:sum_source(m) for m in ('NN_SOURCE','R_SOURCE')},arena={m:sum_arena(m) for m in ('NN_ARENA','R_ARENA')},counts={m:load(m+'_LOCK.json')['counts'] for m in ('NN_SOURCE','R_SOURCE','NN_ARENA','R_ARENA')})
    save('SUMMARY.json',x);print('SUMMARY',json.dumps(x,separators=(',',':')),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='summarize':summarize()
    else:execute(sys.argv[1])
