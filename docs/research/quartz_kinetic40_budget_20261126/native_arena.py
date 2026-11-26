import sys,pathlib,json,hashlib,random,gzip,collections,statistics as S,time
from concurrent.futures import ProcessPoolExecutor
T=pathlib.Path.home()/'quantum/research/quartz_kinetic40_budget_target_20261126'
W=pathlib.Path.home()/'quantum/research/quartz_kinetic40_budget_20261126'
sys.path.insert(0,str(W/'docs/research/quartz_kinetic37_20261126'))
import experiment as e
NEW=T/'mcts_demo';OLD=e.BIN;SEED=2026112640;N=64
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def save(n,x):
    p=T/n;assert not p.exists();p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def read(n):return json.loads((T/n).read_text())
def plan():
    identity=e.identity()
    parity=read('NATIVE_PARITY.json');assert sha(NEW)==parity['new_binary_sha256']
    fam,locks=e.historical();n=len(fam);rng=random.Random(SEED);jobs=[]
    for i in range(N):
        cells=[15*r+c for r in range(4,11) for c in range(4,11)] if i%2==0 else [15*r+c for r in range(2,13) for c in range(2,13)]
        while True:
            h=rng.sample(cells,6);f=e.family(h)
            if f not in fam:break
        fam.add(f);jobs.append(dict(id=i,opening=h,family=f,seed=rng.randrange(2**63),color_order=rng.sample([1,-1],2)))
    save('RAPFI_PLAN.json',dict(jobs=jobs,seed=SEED,prior_families=n,prior_locks=locks,old_identity=identity,new_binary_sha256=sha(NEW),scope='fresh paired same frozen Rapfi250615 evaluator, identical common gate and cap128, native Quartz lock vs baseline PUCT; no alpha-beta/cross-evaluator games'))
    print('RAPFI_PLAN',n,N,flush=True)
def worker(arg):
    i,rr=arg;e.BIN=NEW;p=T/f'RAPFI_GAMES_{i}.jsonl.gz';assert not p.exists();ct=collections.Counter()
    with e.Server(NEW) as s,gzip.open(p,'wt',compresslevel=3) as fd:
        for j in rr:
            pair=[]
            for c in j['color_order']:
                h=list(j['opening']);rows=[]
                while True:
                    b,side,w=e.replay(h,'freestyle')
                    if w is not None:break
                    lock=side==c;a=s.search(h,e.seed(j,len(h)),128,lock)
                    if a['proof']['gate'] is None:
                        assert a['certificate_checks']>0;ct['certificate_checks']+=a['certificate_checks']
                    z=e.small(a);z.update(arm='L128' if lock else 'B128',certificate_checks=a.get('certificate_checks',0));rows.append(z);h.append(a['best'])
                    ct.update(nn=a['NN_calls'],visits=a['visits'],moves=1)
                g=dict(job=j,color=c,history=h,winner=w,score=(1+c*w)/2,rows=rows)
                e.emit(fd,g);pair.append(g);ct['games']+=1
            assert pair[0]['history']==pair[1]['history'] and pair[0]['score']+pair[1]['score']==1
            for a,b in zip(pair[0]['rows'],pair[1]['rows']):
                l,k=(a,b) if a['locked'] else (b,a)
                assert l['best']==k['best'] and l['NN_calls']<=k['NN_calls'] and l['visits']<=k['visits']
                ct['matched_moves']+=1
    print('RAPFI_SHARD',i,dict(ct),flush=True)
    return dict(file=p.name,sha256=sha(p),counts=dict(ct))
def run():
    pl=read('RAPFI_PLAN.json');assert sha(NEW)==pl['new_binary_sha256']
    jobs=pl['jobs']
    with ProcessPoolExecutor(max_workers=4) as ex:fs=list(ex.map(worker,[(i,jobs[i::4]) for i in range(4)]))
    ct=collections.Counter();pairs=collections.defaultdict(list)
    for f in fs:
        ct.update(f['counts'])
        assert sha(T/f['file'])==f['sha256']
        for g in e.raw(T/f['file']):pairs[g['job']['id']].append(g)
    assert len(pairs)==N and ct['games']==2*N
    def cost(gg):
        z=collections.Counter()
        for g in gg:
            for a in g['rows']:
                k='L' if a['locked'] else 'B'
                z[k+'moves']+=1;z[k+'nn']+=a['NN_calls'];z[k+'cpu']+=sum(a['cpu_ns']);z[k+'wall']+=a['transport_ns']
        return z
    pc={i:cost(v) for i,v in pairs.items()};tot=sum(pc.values(),collections.Counter());rng=random.Random(SEED+1)
    def reduction(z,k):return 1-(z['L'+k]/z['Lmoves'])/(z['B'+k]/z['Bmoves'])
    bs=[]
    for _ in range(4000):
        x=sum((pc[i] for i in rng.choices(sorted(pc),k=N)),collections.Counter());bs.append(reduction(x,'nn'))
    bs.sort()
    out=dict(plan_sha256=sha(T/'RAPFI_PLAN.json'),files=fs,counts=dict(ct),pair_counts=dict(tot),score=.5,NN_reduction=reduction(tot,'nn'),NN_CI=[bs[99],bs[3899]],CPU_reduction=reduction(tot,'cpu'),wall_reduction=reduction(tot,'wall'),scope='same-cap exact path replication only; cost goal .30 separate; fixed 64 independent opening families')
    save('RAPFI_RESULT.json',out);print('RAPFI_RESULT',json.dumps(out,separators=(',',':')),flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
