import gzip,json,collections as C,random,sys,time
from k47_plan import D,T,ROOT,load,save,sha,SEEDS
from native47 import NNServer,rapfi_module,rsearch,normalize,BIN,OLD,REF,probe,bundle,replace,exact
K45=D.parent/'quartz_kinetic45_sequential_20261127';K42=D.parent/'quartz_kinetic42_threshold_20261127'
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def rows(d,m):
    l=json.loads((d/(m+'_LOCK.json')).read_text())
    for f in l['files']:
        p=d/f['file'];assert sha(p)==f['sha256'];yield from raw(p)
def put(f,z):f.write(json.dumps(z,separators=(',',':'),allow_nan=False)+'\n')
def proof_parity():
    ct=C.Counter();path=D/'PROOF_PARITY.jsonl.gz';assert not path.exists()
    lk=json.loads((K45/'BENCHMARK_LOCK.json').read_text());p=K45/lk['file'];assert sha(p)==lk['sha256']
    late=K42/'BENCHMARK.jsonl.gz'
    from native47 import NNServer
    with NNServer() as srv,gzip.open(path,'wt',compresslevel=3) as out:
        for r in raw(p):
            h=r['history'];n=7 if r['lane']=='NN' else 15
            ref=probe(h,n,7,4096);assert ref==r['cold']['receipt']
            z=srv.request(dict(cmd='research_k47_proof',history=h,n=n,depth=7,cap=4096,exact=False))
            sem={k:v for k,v in z.items() if k not in ('cpu_ns','wall_ns')}
            assert sem==ref,(h,n,sem,ref)
            ct['NN' if n==7 else 'Rapfi']+=1;ct['queries']+=ref['queries'];ct['cert']+=ref['cert'] is not None
            put(out,dict(lane=r['lane'],history=h,receipt=sem,cpu_ns=z['cpu_ns'],wall_ns=z['wall_ns']))
        for r in raw(late):
            h=r['job']['opening'];ref=probe(h,7,7,4096)
            z=srv.request(dict(cmd='research_k47_proof',history=h,n=7,depth=7,cap=4096,exact=False));sem={k:v for k,v in z.items() if k not in ('cpu_ns','wall_ns')};assert sem==ref
            e=exact(h);zz=srv.request(dict(cmd='research_k47_proof',history=h,n=7,depth=7,cap=25000,exact=True))
            a={k:v for k,v in zz.items() if k not in ('cpu_ns','wall_ns')};b={k:v for k,v in e.items() if k not in ('cpu_ns','wall_ns')};assert a==b,(h,a,b)
            truth=r.get('truth')
            if truth is not None and a['complete']:
                assert all(l<=truth[str(m)]<=u for m,l,u in a['bounds'])
                assert truth[str(a['best'])]==max(truth.values());ct['late_truth_complete']+=1
            ct['late']+=1;ct['late_complete']+=a['complete'];ct['late_nodes']+=a['nodes']
            put(out,dict(lane='late',history=h,receipt=sem,exact=a,truth=truth,cpu_ns=zz['cpu_ns'],wall_ns=zz['wall_ns']))
        # Fixed early and resolved cases with diverse common gate paths.
        fixtures=[(7,[0,48,1,47,2,42]),(7,[0,48,7,47]),(7,[10,0,11,1,12,2]),(7,[24,0,25,1,27,2]),(15,[112,0,113,1,114,2,115,3]),(15,[0,224,1,223]),(15,[112,111,97,96,128,127]),(15,[112,0,113,1,114,2,116,3])]
        for n,h in fixtures:
            r=probe(h,n,7,4096);z=srv.request(dict(cmd='research_k47_proof',history=h,n=n,depth=7,cap=4096,exact=False));a={k:v for k,v in z.items() if k not in ('cpu_ns','wall_ns')};assert a==r
            ct['fixture']+=1;ct['fixture_gate']+=r['gate'] is not None
            put(out,dict(lane='fixture',n=n,history=h,receipt=a))
    assert ct['NN']==1709 and ct['Rapfi']==772 and ct['late']==384 and ct['fixture']==8
    save('PROOF_PARITY_LOCK.json',dict(counts=dict(ct),file=path.name,sha256=sha(path),source_bench=sha(p),source_late=sha(late),binary=sha(BIN)))
    print('PROOF_PARITY',dict(ct),flush=True)
def nn_parity():
    from k46_engine import Puct,load_eval
    ct=C.Counter();path=D/'NN_PARITY.jsonl.gz';assert not path.exists()
    # Eight archived roots per seed, fixed independently of parity outcome:
    # four earliest roots and four unresolved threat roots from K46 trajectories.
    with gzip.open(path,'wt',compresslevel=3) as out:
        for seed in SEEDS:
            gs=[g for g in rows(REF,'NN_LOW') if g['seed']==seed]
            early=[g['job']['opening'] for g in gs[:4]]
            other=[];seen={tuple(h) for h in early}
            for g in gs:
                for r in g['rows']:
                    if r.get('partial') and len(g['history'][:r['ply']])>=7:
                        h=g['history'][:r['ply']]
                        if tuple(h) not in seen:other.append(h);seen.add(tuple(h))
                        if len(other)==4:break
                if len(other)==4:break
            assert len(early+other)==8
            ev=load_eval(seed);ref=Puct(ev)
            trace=[]
            original=ev.__class__.__call__
            # Python evaluator uses special method lookup; trace via subclass wrapper.
            class Trace:
                @property
                def rows(self):return ev.rows
                def __call__(self,a,b,p,legal):
                    pr,v=ev(a,b,p,legal);trace.append(dict(a=a,b=b,p=p,legal=list(legal),prior=pr.tolist(),value=v));return pr,v
            wrap=Trace();ref=Puct(wrap)
            with NNServer(ev) as srv:
                for h in early+other:
                    for q in (False,True):
                        trace.clear();r=ref.quartz46(h,20) if q else ref.search(h,32,False);reftrace=list(trace)
                        a=srv.search(h,q,trace=True)
                        fields=('move','proof','cap','lock','nn','visits','terminal_leaves','votes','distance','prior_best')
                        for k in fields:assert a[k]==r[k],(seed,h,q,k,a[k],r[k])
                        assert a['kind']==('PARTIAL47' if r['kind']=='PARTIAL46' else r['kind'])
                        if q:assert a['original_move']==r['original_move'] and a['replacement']==r['replacement'] and normalize(a['partial'])==normalize(r['partial'])
                        tr=a['eval_trace'];assert len(tr)==len(reftrace)==a['nn']
                        err=0.
                        for x,y in zip(tr,reftrace):
                            assert all(x[k]==y[k] for k in ('a','b','p','legal'))
                            err=max(err,abs(x['value']-y['value']),max((abs(u-v) for u,v in zip(x['prior'],y['prior'])),default=0.))
                        assert err<=2e-15,(seed,h,err)
                        ct['roots']+=1;ct['nn_native']+=a['nn'];ct['nn_ref']+=r['nn'];ct['cert']+=a['kind'] in ('CERTIFICATE','PARTIAL47')
                        put(out,dict(seed=seed,history=h,quartz=q,native=a,reference=r,max_eval_error=err))
    assert ct['roots']==112
    save('NN_PARITY_LOCK.json',dict(counts=dict(ct),file=path.name,sha256=sha(path),binary=sha(BIN),reference=sha(REF/'k46_engine.py')))
    print('NN_PARITY',dict(ct),flush=True)
def rapfi_parity():
    e=rapfi_module();gs=list(rows(REF,'R_ARENA'));hs=[];seen=set()
    # 24 first roots, 24 later roots including proof and search; no game replay.
    for g in gs:
        h=g['job']['opening']
        if tuple(h) not in seen:hs.append((h,g['job']));seen.add(tuple(h))
        if len(hs)==24:break
    for g in gs:
        for r in g['rows']:
            h=g['history'][:r['ply']]
            if r.get('partial') and len(h)>=8 and tuple(h) not in seen:hs.append((h,g['job']));seen.add(tuple(h))
            if len(hs)==48:break
        if len(hs)==48:break
    assert len(hs)==48
    path=D/'R_PARITY.jsonl.gz';assert not path.exists();ct=C.Counter()
    with e.Server(OLD) as old,e.Server(BIN) as new,gzip.open(path,'wt',compresslevel=3) as out:
        for h,j in hs:
            seed=e.seed(j,len(h))
            for q in (False,True):
                part=bundle(h,15) if q else None
                if part and part['cold']['receipt']['gate'] is None and part['cert'] is not None:
                    r=dict(move=part['cert'],nn=0,visits=0,kind='PARTIAL47',votes=[],original_move=None,replacement=False,partial=part)
                else:
                    z=old.search(h,seed,128,q);g=z['proof']['gate'];vs=e.votes(z['result']) or []
                    r=dict(move=z['best'],nn=z['NN_calls'],visits=z['visits'],kind='CERTIFICATE' if g is not None else z['cause'],votes=vs,original_move=z['best'] if q and g is None else None,replacement=False,partial=part if q and g is None else None)
                    if q and g is None:r['move']=replace(r['move'],vs,part['bounds'],15);r['replacement']=r['move']!=r['original_move']
                a=rsearch(e,new,h,seed,q)
                for k in ('move','nn','visits','kind','votes','original_move','replacement'):assert a[k]==r[k],(h,q,k,a[k],r[k])
                assert normalize(a['partial'])==normalize(r['partial'])
                ct['roots']+=1;ct['native_nn']+=a['nn'];ct['reference_nn']+=r['nn'];ct['cert']+=a['kind'] in ('CERTIFICATE','PARTIAL47')
                put(out,dict(history=h,seed=seed,quartz=q,native=a,reference=r))
    assert ct['roots']==96
    save('R_PARITY_LOCK.json',dict(counts=dict(ct),file=path.name,sha256=sha(path),binary=sha(BIN),old_binary=sha(OLD)))
    print('R_PARITY',dict(ct),flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
