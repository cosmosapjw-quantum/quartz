"""Native K17 prospective diagnostic; exact oracle strictly outside request."""
import collections as C, gzip, hashlib, json, math, os, pathlib, random, select, statistics as S, subprocess as SP, sys, time
D=pathlib.Path(__file__).resolve().parent
ROOT=D.parents[2]
OLD=ROOT.parent/'quartz_kinetic16_20261126/docs/research/quartz_kinetic16_20261126'
sys.path.insert(0,str(OLD))
import pair_run as old
import pair_ref as pr
import importlib.util
spec=importlib.util.spec_from_file_location("k16_independent_audit",OLD/"audit.py")
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
from fractions import Fraction as F
BIN=ROOT.parent/'quartz_kinetic17_target_20261126/mcts_demo'
OBIN=ROOT.parent/'quartz_kinetic16_target_20261126/mcts_demo'
MODEL=ROOT/'docs/research/cognitive_metareasoning_20261120/connect4_generalization_07/evidence/mlp_conv_train_20261121/conv_model.json'
MID=old.MID
ARMS=('K16N','K16BAL','K17EXP')
CAPS=(32,64)
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def save(n,x):
    p=D/n;assert not p.exists()
    p.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n')
def read(n):return json.loads((D/n).read_text())
def raw(n):
    with gzip.open(D/n,'rt') as f:return [json.loads(l) for l in f]
def identity():
    a=read('RUST_BUILD_IDENTITY.json')
    assert sha(BIN)==a['binary_sha256']
    assert sha(OBIN)=='dbd544df9b7288f5c54364a6e4818b0484d3c3be22aab072982a9ec7c4955ea7'
    assert sha(MODEL)==MID
    assert sha(old.ORACLE)=='fca2b665b1b3dbd88f2f50a7e0ba86fd8ae253f2861da1ad51cf110c69f741ed'
    assert sha(old.BOOK)=='38f9834317c37d9516e45a21da598569a5d1556595686593d14c2e63f59c1f38'
    return dict(binary=sha(BIN),parent_binary=sha(OBIN),source=a['source_head'],model=MID,oracle=sha(old.ORACLE),book=sha(old.BOOK))
def plan():
    fam,boards,locks=old.c4historical()
    prior=old.read('C06_PLAN.json')
    fam.update(j['family'] for j in prior['c4'])
    for r in old.read('C06_COHORT.json')['roots']:
        if r['domain']=='C4':boards.add(old.boardkey(old.c4flat(old.c4replay(r['history'])[0])))
    nf,nb=len(fam),len(boards)
    rng=random.Random(2026112617);jobs=[];roots=[]
    for i in range(96):
        while True:
            h=''.join(str(rng.randrange(1,8)) for _ in range(5));k=old.mirror(h)
            if k not in fam:break
        fam.add(k);seed=rng.randrange(2**63);r=random.Random(seed);op=h
        depth=(26,28,30)[i%3];group=('TRAIN','CAL','TEST')[i//32]
        b,z=old.c4replay(h)
        while len(h)<depth:
            side=len(h)%2
            opts=[c for c in range(7) if z[c]<6 and not old.won(b[side]|(1<<(7*c+z[c])))]
            if not opts:break
            c=r.choice(opts);b[side]|=1<<(7*c+z[c]);z[c]+=1;h+=str(c+1)
        key=old.boardkey(old.c4flat(b));legal=[c for c in range(7) if z[c]<6]
        ok=len(h)==depth and len(legal)>=3 and key not in boards
        job=dict(id=i,group=group,family=k,opening=op,seed=seed,depth=depth,history=h,accepted=ok)
        jobs.append(job)
        if ok:
            boards.add(key)
            roots.append(dict(id=i,group=group,family=k,history=h,legal=legal,depth=depth,tactical=any(old.canwin(b,z,p) for p in (0,1))))
    save('COHORT.json',dict(identity=identity(),source_locks=locks,old_families=nf,old_boards=nb,seed=2026112617,jobs=jobs,roots=roots))
    print('COHORT',len(roots),dict(C.Counter((r['group'],r['tactical']) for r in roots)),flush=True)
class Server:
    def __init__(self,bin):
        self.p=SP.Popen(['nice','-n','10',str(bin),'--server'],stdin=SP.PIPE,stdout=SP.PIPE,stderr=SP.DEVNULL,bufsize=0,env=dict(os.environ,RAYON_NUM_THREADS='1',QUARTZ_DISABLE_NN_CACHE='1',QUARTZ_DISABLE_ASYNC_PIPELINE='1',QUARTZ_DISABLE_QIPC_SHM='1'))
        self.buf=b'';self.cpu=old.process_clock(self.p.pid)
    def request(self,h,arm,cap):
        q=dict(cmd='research_connect4_evidence_cpu',history=h,arm=arm,cap=cap,seed=17,model_path=str(MODEL),model_identity=MID)
        t=time.perf_counter_ns();cpu=time.clock_gettime_ns(self.cpu)
        self.p.stdin.write((json.dumps(q,separators=(',',':'))+'\n').encode());self.p.stdin.flush()
        end=time.monotonic()+30
        while b'\n' not in self.buf:
            wait=end-time.monotonic()
            if wait<=0 or not select.select([self.p.stdout],[],[],wait)[0]:raise RuntimeError('native timeout')
            x=os.read(self.p.stdout.fileno(),65536)
            if not x:raise RuntimeError('native EOF')
            self.buf+=x
            if len(self.buf)>4*1024*1024:raise RuntimeError('oversized reply')
        line,self.buf=self.buf.split(b'\n',1);a=json.loads(line)
        a['cpu_ns']=time.clock_gettime_ns(self.cpu)-cpu;a['wall_ns']=time.perf_counter_ns()-t
        return a
    def __enter__(self):return self
    def __exit__(self,t,*args):
        self.p.stdin.close()
        try:self.p.wait(timeout=5)
        except SP.TimeoutExpired:self.p.kill();self.p.wait()
        if t is None:assert self.p.returncode==0
def check(a,r,arm,cap):
    assert 'error' not in a,a
    assert a['history']==r['history'] and a['arm']==arm and a['cap']==cap and a['seed']==17 and a['model_identity']==MID and a['legal']==r['legal']
    assert a['visits']==cap and a['best'] in r['legal'] and a['cpu_ns']>=0 and a['wall_ns']>0
    e=a['evidence'];pre=e['prefix'];ca,n,w=pr.moments(pre);i,j=pr.pair(pre)
    pair=[ca[i],ca[j]];c=next(k for k in ca if k not in pair);panel=pair+[c]
    assert e['pair']==pair and set(ca)==set(r['legal']) and pre['root_n']==8
    q=(cap-16)//2;targets=[None,None,None] if arm=='K16N' else [pair[0],pair[1] if arm=='K16BAL' else c,None]
    if arm=='K17EXP':assert e['reservoir']==panel
    else:
        assert 'reservoir' not in e
        pr.verify(a)
    oldn=n;oldw=w
    for ix,(phase,k,t) in enumerate(zip(e['phases'],(q,q,8),targets)):
        nxt=phase['result'];ca1,nn,ww=pr.moments(nxt)
        assert ca1==ca and nxt['candidates']==pre['candidates'] and nxt['root_hash']==pre['root_hash']
        delta=[v-u for u,v in zip(oldn,nn)]
        assert delta==phase['delta'] and sum(delta)==k and min(delta)>=0
        assert phase['target']==t and (t is None or delta[ca.index(t)]==k)
        assert all(abs(v-u)<=d+1e-6 for u,v,d in zip(oldw,ww,delta))
        if ix==1:
            pulse={m:(ww[ca.index(m)]-w[ca.index(m)]+4*w[ca.index(m)]/(n[ca.index(m)]+4))/(nn[ca.index(m)]-n[ca.index(m)]+4) for m in ca}
        oldn,oldw=nn,ww
    assert nxt==a['result'] and nxt['root_n']==cap and nxt['evaluator_calls']==a['NN_calls']
    final={m:ww[k]/(nn[k]+4) for k,m in enumerate(ca)}
    for s in [pulse,final]:assert all(abs(v)<=1+1e-6 for v in s.values())
    out=dict(pair=pair,panel=panel,native=a['best'])
    for tag,s in [('pulse',pulse),('final',final)]:
        for name,p in [('pair',pair),('triple',panel)]:
            out[tag+'_'+name]=max(p,key=s.get)
    assert out['pulse_pair']==e['pulse_best'] and out['final_pair']==e['final_best']
    return out
def atlas():
    co=read('COHORT.json');assert co['identity']==identity()
    path=D/'ATLAS.jsonl.gz';assert not path.exists();count=0;parity=0
    # Rotate order, same budget and deterministic common prefix. Baseline replay old binary.
    with Server(BIN) as s,Server(OBIN) as p,gzip.open(path,'wt',compresslevel=3) as f:
        for r in co['roots']:
            prefix=None
            for cap in CAPS:
                order=ARMS[r['id']%3:]+ARMS[:r['id']%3]
                for arm in order:
                    a=s.request(r['history'],arm,cap);o=check(a,r,arm,cap)
                    if prefix is None:prefix=a['evidence']['prefix']
                    assert a['evidence']['prefix']==prefix
                    if arm=='K16N':
                        b=p.request(r['history'],arm,cap);check(b,r,arm,cap)
                        assert all(a[k]==b[k] for k in ('result','best','NN_calls','visits'))
                        for k in ('pair','prefix','prefix_score','pulse_score','final_score','native_best'):
                            assert a['evidence'][k]==b['evidence'][k],k
                        parity+=1
                    f.write(json.dumps(dict(root=r,cap=cap,arm=arm,response=a,readout=o),allow_nan=False,separators=(',',':'))+'\n');f.flush();count+=1
            print('ATLAS',r['id'],r['group'],r['depth'],flush=True)
    save('ATLAS_LOCK.json',dict(cohort=sha(D/'COHORT.json'),sha256=sha(path),requests=count,baseline_parity=parity,identity=identity()))
def labels():
    co=read('COHORT.json');assert co['identity']==identity()
    assert sha(D/'ATLAS.jsonl.gz')==read('ATLAS_LOCK.json')['sha256']
    lab=[];cnt=C.Counter()
    for r in co['roots']:
        q=old.oracle(r);item=dict(root=r,status=q['status'])
        if q['status']=='EXACT':
            try:
                v,n=ref.exact_wdl(r['history'])
                truth={int(k):(1 if q['scores'][k]>0 else -1 if q['scores'][k]<0 else 0) for k in r['legal']}
                item.update(scores=q['scores'],independent=v,nodes=n,agreement=v==truth)
                if v!=truth:item['status']='MISMATCH'
            except RuntimeError as e:item.update(status='INDEPENDENT_CAP',reason=str(e),scores=q['scores'])
        else:item['oracle_result']=q
        cnt[item['status']]+=1;lab.append(item)
    save('LABELS.json',dict(cohort=sha(D/'COHORT.json'),atlas=sha(D/'ATLAS.jsonl.gz'),counts=dict(cnt),rows=lab))
    print('LABELS',dict(cnt),flush=True)
def score():
    co=read('COHORT.json');al=read('ATLAS_LOCK.json');la=read('LABELS.json')
    assert al['cohort']==la['cohort']==sha(D/'COHORT.json') and la['atlas']==al['sha256']==sha(D/'ATLAS.jsonl.gz')
    labels0={x['root']['id']:x for x in la['rows']}
    vals=[];strata={}
    for t in raw('ATLAS.jsonl.gz'):
        r=t['root'];a=t['response'];o=check(a,r,t['arm'],t['cap']);assert o==t['readout']
        z=labels0[r['id']]
        if z['status']!='EXACT':continue
        u={int(k):(v+1)/2 for k,v in z['independent'].items()}
        best=max(u.values());mp=max(u[m] for m in o['pair']);mt=max(u[m] for m in o['panel'])
        label=('omission' if best>mp else 'covered')+('_gap' if len({u[m] for m in o['pair']})>1 else '_tie')
        strata[r['id']]=dict(group=r['group'],stratum=label,tactical=r['tactical'],omission_pair=best-mp,omission_triple=best-mt)
        row=dict(id=r['id'],group=r['group'],tactical=r['tactical'],stratum=label,depth=r['depth'],cap=t['cap'],arm=t['arm'],cpu_ms=a['cpu_ns']/1e6,wall_ms=a['wall_ns']/1e6,NN=a['NN_calls'],omission_pair=best-mp,omission_triple=best-mt,oracle_access=mt-mp)
        for k in ['native','pulse_pair','pulse_triple','final_pair','final_triple']:
            v=u[o[k]];row['regret_'+k]=best-v
            if k!='native':
                pm=mp if k.endswith('pair') else mt
                row['rank_'+k]=pm-v
                assert row['regret_'+k]==row['rank_'+k]+(best-pm)
        vals.append(row)
    assert len(vals)==6*len(strata)
    metrics=['cpu_ms','wall_ms','NN','omission_pair','omission_triple','oracle_access']+['regret_'+k for k in ['native','pulse_pair','pulse_triple','final_pair','final_triple']]+['rank_'+k for k in ['pulse_pair','pulse_triple','final_pair','final_triple']]
    summary=[]
    for group in ['TRAIN','CAL','TEST','ALL']:
        for cap in CAPS:
            for arm in ARMS:
                rr=[v for v in vals if (group=='ALL' or v['group']==group) and v['cap']==cap and v['arm']==arm]
                if rr:summary.append(dict(group=group,cap=cap,arm=arm,n=len(rr),**{k:S.mean(v[k] for v in rr) for k in metrics}))
    strata_counts=dict(C.Counter(s['group']+'_'+s['stratum'] for s in strata.values()))
    contrasts=[]
    for group in ['TRAIN','CAL','TEST']:
        for cap in CAPS:
            for policy,key in [('K16BAL','pulse_pair'),('K17EXP','pulse_triple'),('K17EXP','final_triple')]:
                mm={v['id']:v for v in vals if v['group']==group and v['cap']==cap and v['arm']==policy}
                nn={v['id']:v for v in vals if v['group']==group and v['cap']==cap and v['arm']=='K16N'}
                if not mm:continue
                assert mm.keys()==nn.keys()
                ds=[nn[i]['regret_native']-mm[i]['regret_'+key] for i in mm]
                costs=[mm[i]['cpu_ms']-nn[i]['cpu_ms'] for i in mm]
                rnd=random.Random(2026112617);boot=sorted(S.mean(rnd.choices(ds,k=len(ds))) for _ in range(2000))
                contrasts.append(dict(group=group,cap=cap,arm=policy,readout=key,n=len(ds),gain=S.mean(ds),cpu_delta_ms=S.mean(costs),positive=sum(x>0 for x in ds),negative=sum(x<0 for x in ds),descriptive_bootstrap=[boot[49],boot[1949]]))
    save('SUMMARY.json',dict(cohort=sha(D/'COHORT.json'),atlas=al['sha256'],labels=sha(D/'LABELS.json'),counts=la['counts'],strata_counts=strata_counts,strata=strata,summary=summary,contrasts=contrasts,rows=vals))
    print('STRATA',strata_counts,flush=True)
    for t in summary:
        if t['group']=='TEST':print('TEST',t,flush=True)
    for t in contrasts:
        if t['group']=='TEST':print('TEST_CONTRAST',t,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
