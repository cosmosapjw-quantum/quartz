"""Independent grid replay, exact cost reconstruction, and bounded C++ oracle."""
import collections as C, fractions, gzip, importlib.util,json,pathlib,hashlib,subprocess,time,random
from concurrent.futures import ThreadPoolExecutor
from k42_plan import D,ROOT,SEEDS,BUDGET,family,load,save,sha
F=fractions.Fraction
OLD=D.parent/'quartz_kinetic41_exact_20261126'
B=pathlib.Path.home()/'Dropbox/personal_projects'
T=ROOT.parent/'quartz_kinetic42_threshold_target_20261127'
BIN=B/'quartz_kinetic41_exact_backups'/'independent'
spec=importlib.util.spec_from_file_location('_grid42',D.parent/'quartz_kinetic33_20261126/audit_k33.py')
grid=importlib.util.module_from_spec(spec);spec.loader.exec_module(grid)
CT=C.Counter();CACHE={}
def raw(p):
    with gzip.open(p,'rt') as f:yield from map(json.loads,f)
def rows(mode):
    lk=load(mode+'_LOCK.json')
    assert lk['plan']==sha(D/'PLAN.json') and lk['source']==sha(D/'threshold42.cpp') and lk['binary']==sha(T/'threshold42.so')
    for fi in lk['files']:
        p=D/fi['file'];assert sha(p)==fi['sha256'];r=list(raw(p))
        if mode=='ARENA':
            assert sum(g['nn'] for g in r)==fi['counts']['nn'] and all(g['seed']==fi['seed'] for g in r)
        else:assert sum(g['result']['nodes'] for g in r)==fi['counts']['nodes']
        yield from r
def orient(h):
    b,w=grid.replay(h);assert not w and any(x==0 for x in b);p=1-2*(len(h)%2)
    return tuple(sum(1<<i for i,x in enumerate(b) if x==s) for s in (p,-p))
def gate(b,p):
    k=tuple(b),p
    if k not in CACHE:CACHE[k]=grid.cert(b,p)
    return CACHE[k]
def check_bounds(z,legal,truth=None,census=False):
    assert set(map(int,z['bounds']))==set(legal)
    m=str(z['best']);v=z['best_value'];bd=z['bounds'];assert m in bd and v in (-1,0,1) and bd[m]==[v,v]
    for k,(l,u) in bd.items():
        assert -1<=l<=u<=v<=1
        if int(k)<int(m):assert u<v
        if truth is not None:assert l<=truth[k]<=u
    if truth is not None:assert v==max(truth.values()) and int(m)==min(int(k) for k,x in truth.items() if x==v)
    if census:assert all(l==u for l,u in bd.values()) and z['values']=={k:x[0] for k,x in bd.items()}
def check_receipt(z,legal,cap,truth=None,census=False):
    assert 0<=z['nodes']<=cap and 0<=z['states']<=z['nodes'] and z['hits']>=0
    if z['complete']:check_bounds(z,legal,truth,census)
    else:assert z['nodes']==cap and 'best' not in z and 'bounds' not in z
def val(h,r,q):
    b,w=grid.replay(h);p=1-2*(len(h)%2);legal=[i for i,v in enumerate(b) if not v]
    assert not w and r['move'] in legal and r['cap']==32 and r['lock']==q
    g=gate(b,p);CT['rule_checks']+=1
    if q:
        should=len(h)>=33 and g['move'] is None
        assert r['exact_attempt']==should and 0<=r['exact_nodes']<=BUDGET and r['exact_cpu_ns']>=0
        if should:
            CT['proof_attempts']+=1
            if not r['exact_complete']:assert r['exact_nodes']==BUDGET and 'exact_bounds' not in r;CT['fallback']+=1
        else:assert not r['exact_complete'] and not r['exact_nodes']
    if r['kind']=='CERTIFICATE':
        assert g['move']==r['move'] and g['kind']==r['proof'] and r['nn']==r['visits']==0 and not r['votes'];CT['certificates']+=1
    elif r['kind']=='EXACT42':
        assert q and should and r['exact_complete'] and r['nn']==r['visits']==0 and not r['votes']
        check_bounds(dict(best=r['move'],best_value=r['exact_value'],bounds=r['exact_bounds']),legal);CT['proof_complete']+=1
    else:
        assert g['move'] is None and r['proof']=='UNRESOLVED'
        n=dict(r['votes']);assert list(n)==legal and all(type(v) is int and v>=0 for v in n.values())
        a=min(legal,key=lambda i:(-n[i],i));d=min((n[a]-n[i]+int(i>a) for i in legal if i!=a),default=10**9)
        assert r['move']==a and r['distance']==d and r['prior_best'] in legal and sum(n.values())==r['visits']<=32
        assert r['nn']==1+r['visits']-r['terminal_leaves'] and 0<=r['terminal_leaves']<=r['visits']
        if r['kind']=='VOTE_LOCK':assert q and d>32-r['visits'] and r['visits']<32;CT['locks']+=1
        else:assert r['kind']=='CAP' and r['visits']==32
        CT['legal_edges']+=len(legal)
    assert r['wall_ns']>=r['rule_ns']>=0 and r['cpu_ns']>=0
def run():
    pl=load('PLAN.json');jobs={j['id']:j for j in pl['jobs']};assert len(jobs)==304 and len({family(j['opening']) for j in jobs.values()})==304
    for p in pl['prior_locks']:
        f=ROOT/p['path'];assert sha(f)==p['sha256'];old=json.loads(f.read_text())
        assert not ({family(j['opening']) for j in old['jobs']}&{family(j['opening']) for j in jobs.values()})
    for j in jobs.values():
        b,w=grid.replay(j['opening']);assert not w and b.count(0)==j['empties']
        if j['split']!='EARLY':assert gate(b,1-2*(len(j['opening'])%2))['move'] is None
    oracle={r['job']['id']:r['result'] for r in rows('ORACLE')};assert len(oracle)==144
    prior={(r['a'],r['b']):r['values'] for r in raw(OLD/'INDEPENDENT.jsonl.gz') if 'corpus_complete' in r['tags'] or 'corpus_censored' in r['tags']}
    bk=load('BENCHMARK_LOCK.json');p=D/bk['file'];assert sha(p)==bk['sha256']
    bench=list(raw(p));assert len(bench)==384
    fresh={};qmap={};native_truth={};queries={};tags=C.defaultdict(set)
    def register(h,tag):
        k=orient(h);i=queries.setdefault(k,len(queries));tags[i].add(tag);return k
    for r in bench:
        j=r['job'];h=j['opening'];k=orient(h);assert k==(r['a'],r['b'])
        legal=[i for i,v in enumerate(grid.replay(h)[0]) if v==0]
        if j['frame']=='K41':truth=prior[k];assert r['truth']==truth
        else:
            assert j['id'] in oracle;z=oracle[j['id']];check_receipt(z,legal,500000,census=True)
            truth=z.get('values');fresh[k]=r;register(h,'fresh');native_truth[k]=truth
        for name,x in r['runs'].items():
            z=x['result'];assert x['cpu_ns']>=0 and x['wall_ns']>=0
            if name in ('action','census'):check_receipt(z,legal,BUDGET,truth,name=='census')
            elif z['complete'] and truth is not None:assert z['values']==truth
        if r['runs']['action']['result']['complete']:qmap[k]=r['runs']['action']['result']
    assert len(fresh)==144
    pairs={};cc=C.defaultdict(C.Counter);first=C.defaultdict(dict);pool={}
    for g in rows('ARENA'):
        j=g['job'];s=g['seed'];color=g['color'];assert j==jobs[j['id']] and j['split']!='ATLAS'
        k=j['id'],s,color;assert k not in pairs;pairs[k]=g
        h=j['opening'].copy();nn=0;c=cc[j['split']]
        for i,r in enumerate(g['rows']):
            p=1-2*(len(h)%2);q=p==color;assert r['player']==p and r['ply']==len(h) and r['arm']==('Q42' if q else 'B32')
            val(h,r,q);tag='candidate' if q else 'baseline'
            c[tag+'_moves']+=1;c[tag+'_nn']+=r['nn'];c[tag+'_cpu']+=r['cpu_ns'];c[tag+'_wall']+=r['wall_ns']
            if i==0 and j['split']!='EARLY':
                k0=register(h,'arena_first');first[j['id'],s][tag]=(k0,r)
            if q and r['exact_complete']:
                k0=orient(h)
                z=dict(best=r['move'],best_value=r['exact_value'],bounds=r['exact_bounds'])
                if k0 in pool:assert pool[k0][1]==z
                else:pool[k0]=(h.copy(),z)
            h.append(r['move']);nn+=r['nn']
        b,w=grid.replay(h);assert h==g['history'] and (w or all(b)) and w==g['winner'] and g['score2']==1+color*w and nn==g['nn']
        c['games']+=1;c['score2']+=g['score2'];c['wins']+=g['score2']==2;c['draws']+=g['score2']==1
    assert len(pairs)==1280 and len(first)==384
    summ=load('SUMMARY.json')
    for sp,c in cc.items():
        x=summ['arena'][sp];assert x['games']==c['games'] and x['wins']==c['wins'] and x['draws']==c['draws'] and F(c['score2'],2*c['games'])==F(x['score']).limit_denominator(2560)
        for tag in ('candidate','baseline'):
            for k in ('moves','nn'):assert x[tag+'_'+k]==c[tag+'_'+k]
        for k in ('nn','cpu','wall'):
            if c['baseline_'+k]:
                red=1-F(c['candidate_'+k]*c['baseline_moves'],c['baseline_'+k]*c['candidate_moves'])
                assert abs(float(red)-x[k+'_reduction'])<1e-14
    # Predetermined bounded independent audit: all fresh, all late/mid first roots,
    # all <=10-empty exact arena roots, plus 64 random distinct deeper exact arena roots.
    for k,(h,z) in pool.items():
        if 49-len(h)<=10:register(h,'arena_small')
    others=sorted(k for k in pool if k not in queries)
    random.Random(202611274209).shuffle(others)
    for k in others[:64]:register(pool[k][0],'arena_deep_sample')
    assert sha(BIN)=='a1e943f250eae145e10a12482fc575d11af08960dfd7ddf804c1f0c0ff14c7af'
    tasks=sorted(queries.items(),key=lambda x:x[1])
    def batch(xs):
        inp=''.join(f'{i} {a} {b}\n' for (a,b),i in xs)
        p=subprocess.run([str(BIN)],input=inp,text=True,capture_output=True,check=True,timeout=180)
        rs=list(map(json.loads,p.stdout.splitlines()));assert len(rs)==len(xs);return rs
    t=time.monotonic()
    with ThreadPoolExecutor(max_workers=6) as ex:rs=sum(ex.map(batch,[tasks[i::6] for i in range(6)]),[])
    assert {r['id'] for r in rs}==set(queries.values())
    rev={i:k for k,i in queries.items()};outp=D/'INDEPENDENT.jsonl.gz';assert not outp.exists()
    cov=C.defaultdict(C.Counter);truths={};count=C.Counter()
    with gzip.open(outp,'wt',compresslevel=3) as fd:
        for r in sorted(rs,key=lambda x:x['id']):
            i=r['id'];k=rev[i];count['queries']+=1;count['nodes']+=r['nodes'];count['complete']+=r['complete']
            for tag in tags[i]:cov[tag]['queries']+=1;cov[tag]['complete']+=r['complete']
            if r['complete']:
                v=r['values'];truths[k]=v
                if k in native_truth:assert v==native_truth[k];count['fresh_full_match']+=1
                if k in qmap:check_bounds(qmap[k],list(map(int,v)),v);count['bench_certificate_match']+=1
                if k in pool:check_bounds(pool[k][1],list(map(int,v)),v);count['arena_certificate_match']+=1
            fd.write(json.dumps(dict(a=k[0],b=k[1],tags=sorted(tags[i]),**r),separators=(',',':'))+'\n')
    first_stats=C.defaultdict(C.Counter)
    for (i,s),p in first.items():
        assert set(p)=={'candidate','baseline'}
        k,q=p['candidate'];k2,b=p['baseline'];assert k==k2
        c=first_stats[jobs[i]['split']];c['pairs']+=1
        if k in truths:
            v=truths[k];a=v[str(q['move'])];d=v[str(b['move'])];best=max(v.values())
            c['independent_pairs']+=1;c['improved']+=a>d;c['worse']+=a<d;c['same']+=a==d;c['q_regret2']+=best-a;c['b_regret2']+=best-d;c['q_bad']+=a!=best;c['b_bad']+=d!=best
    native=json.loads((D.parent/'quartz_kinetic40_budget_20261126/NATIVE_IDENTITY.json').read_text())
    assert sha(B/'quartz_kinetic40_budget_backups'/'mcts_demo')==native['binary_sha256']
    lock=json.loads((D.parent/'quartz_kinetic37_20261126/RAPFI_LOCK.json').read_text())
    for a in lock['files']:assert sha(pathlib.Path(a['path']))==a['sha256']
    out=dict(counts=dict(CT),unique_rule_states=len(CACHE),bench_rows=len(bench),arena_games=len(pairs),arena_proof_unique=len(pool),independent=dict(count),coverage={k:dict(v) for k,v in cov.items()},independent_elapsed_s=time.monotonic()-t,independent_artifact_sha256=sha(outp),independent_binary_sha256=sha(BIN),first_action={k:dict(v) for k,v in first_stats.items()},native_rapfi_identity=native['binary_sha256'],native_execution='NOT_RUN_K42',scope='all raw rule/vote/bound and cost checks; independent exact truth for specified query subset with explicit censoring; CI not independently recomputed')
    save('AUDIT.json',out);print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':run()
