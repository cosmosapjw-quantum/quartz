"""Mechanism-resolved actual native factorial subset; no exact labels passed online."""
import collections as C,gzip,json,math,os,random,statistics as S,subprocess as SP,sys,time
from pathlib import Path
import native as n
from native import D,ROOT,write,read,sha
from population import exclusions
BIN=ROOT.parent/'quartz_fouraxis02_target_20261125/mcts_demo'
ST=('TACTICAL','SELFPLAY');SEEDS=(17,43)
ARMS=('BASE','SX','WRITER','TEMPORAL','MERGE','COMBINED','LEGACY_V','PROOF_V','TOP2_V','FULL_V','JOINT')
MODES=('VISITS','NN','TIME')
def gzwrite(name,obj):
    p=D/name;assert not p.exists()
    with gzip.open(p,'wt',compresslevel=5) as f:json.dump(obj,f,separators=(',',':'),allow_nan=False)
def gzread(name):
    with gzip.open(D/name,'rt') as f:return json.load(f)
def lock():
    n.prior_lock();m=read('C06_BUILD_IDENTITY.json');assert sha(BIN)==m['binary_sha256']
def snap():
    a=n.host();a['mono']=time.monotonic();return a
def interval(a,b):
    dt=b['mono']-a['mono'];swap=sum(b['swap_pages'][k]-a['swap_pages'][k] for k in a['swap_pages'])/max(dt,1e-9)
    return dict(seconds=dt,swap_pages_s=swap,max_endpoint_load=max(a['load'][0],b['load'][0]),ok=swap<=512 and max(a['load'][0],b['load'][0])<=12)
class Server:
    def __init__(self):
        lock();self.p=SP.Popen([str(BIN),'--server'],stdin=SP.PIPE,stdout=SP.PIPE,stderr=SP.DEVNULL,text=True,bufsize=1,env=dict(os.environ,RAYON_NUM_THREADS='2'))
    def call(self,h,arm='BASE',mode='VISITS',seed=17):
        q=dict(cmd='research_connect4_fouraxis_cpu',history=h,arm=arm,mode=mode,seed=seed,model_path=str(n.MODEL),model_identity=n.IDENT)
        t=time.perf_counter_ns();self.p.stdin.write(json.dumps(q)+'\n');self.p.stdin.flush();line=self.p.stdout.readline();assert line,('EOF',self.p.poll())
        a=json.loads(line);a['transport_us']=(time.perf_counter_ns()-t)/1000
        assert 'error' not in a,a
        assert a['history']==h and a['model_identity']==n.IDENT
        b,z=n.replay(h);legal={i for i in range(7) if z[i]<6}
        if mode=='DIAG':
            assert {x['move'] for x in a['rows']}==legal and a['nn_calls']==1+sum(not x['terminal'] for x in a['rows'])
            assert -1<=a['base']<=1 and all(-1<=x['h']<=1 for x in a['rows'])
            return a
        assert a['arm']==arm and a['mode']==mode and a['seed']==seed
        p=a['prefix'];r=a['result'];hs=a['hooks'];ph=a['prefix_hooks'];vis=8+8*a['blocks']
        assert p['root_n']==8 and r['root_n']==vis<=512 and p['root_hash']==r['root_hash']
        for x in (p,r):
            assert {v[0] for v in x['candidates']}==legal and x['best'] in legal
            assert [y['move'] for y in x['edges']]==[y[0] for y in x['candidates'][:len(x['edges'])]]
            assert sum(y['n'] for y in x['edges'])==x['root_n'] and abs(sum(y['w'] for y in x['edges'])-x['root_w'])<1e-6
            assert all(y['vl']==0 and abs(y['vvalue'])<1e-6 and abs(y['w'])<=y['n']+1e-6 and y['m2']>=-1e-7 for y in x['edges'])
        if mode!='STANDARD':assert p['evaluator_calls']==1+ph['base_evals'] and ph['base_evals']+ph['terminal_leaves']==8
        assert hs['base_evals']+hs['terminal_leaves']==vis-8
        assert r['evaluator_calls']==p['evaluator_calls']+hs['base_evals']+hs['extra_evals']
        if mode=='VISITS':assert vis==64
        if mode=='DEEP':assert vis==512
        if mode=='NN':assert r['evaluator_calls']>=48 or (vis==512 and a['cause']=='CAP')
        if mode=='TIME':assert a['before_drop_us']>=2500 or (vis==512 and a['cause']=='CAP')
        if arm in ('BASE','SX','WRITER','TEMPORAL','MERGE','COMBINED','PROOF_V'):assert hs['extra_evals']==0
        if arm in ('WRITER','BASE','SX','LEGACY_V','PROOF_V','TOP2_V','FULL_V'):assert hs['temporal_blends']==hs['merge_blends']==0
        if arm=='WRITER':assert hs['ledger_observations']==0
        if arm in ('TEMPORAL','MERGE','COMBINED','JOINT'):assert hs['ledger_observations']==hs['backup_edges']
        if arm=='TEMPORAL':assert hs['merge_blends']==0
        if arm in ('MERGE','JOINT'):assert hs['temporal_blends']==0
        if arm not in ('PROOF_V','TOP2_V','FULL_V','JOINT'):assert hs['proofs']==hs['proof_scans']==0
        else:assert hs['proof_scans']==hs['base_evals'] and hs['proofs']<=hs['proof_scans']
        assert hs['backup_changed']<=hs['temporal_blends']+hs['merge_blends']<=hs['backup_edges']-hs['terminal_edges']
        return a
    def close(self):
        self.p.stdin.close();assert self.p.wait(timeout=30)==0
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
def smoke_diag():
    old=ROOT.parent/'quartz_nn_multiquanta_20261124/docs/research/quartz_nn_multiquanta_20261124/C07_COHORT.json'
    cohort=json.loads(old.read_text())['rows'];roots=[r for st in ST for r in [x for x in cohort if x['group']=='TRAIN' and x['stratum']==st][:6]]
    oldp,oldb,meta=exclusions();used=set(oldb);par=[];deep=[];tests=[];traces=[];rows=[]
    with Server() as srv:
        for root in roots[:4]:
            a=srv.call(root['history'],mode='STANDARD');b=srv.call(root['history'],mode='TRACE_PREFIX')
            assert a['result']==b['result'];par.append(dict(standard=a,research=b))
        for arm in ARMS:tests.append(srv.call(roots[0]['history'],arm))
        for root in roots[:2]:
            for arm in ('WRITER','TEMPORAL','MERGE','COMBINED','JOINT'):deep.append(srv.call(root['history'],arm,'DEEP'))
        for root in roots:
            tr=srv.call(root['history'],mode='TRACE_PREFIX');traces.append(dict(source=root['id'],trace=tr))
            for path in tr['prefix_hooks']['leaf_paths']:
                h=root['history']+''.join(str(x+1) for x in path);b,z=n.replay(h);board=n.flat(b);c=n.cb(board)
                if c in used or any(n.won(x) for x in b) or all(x==6 for x in z):continue
                used.add(c);a=srv.call(h,mode='DIAG')
                rows.append(dict(id='LEAF_%03d'%len(rows),source=root['id'],stratum=root['stratum'],history=h,board=board,legal=[i for i,x in enumerate(z) if x<6],diag=a))
        q=dict(cmd='research_connect4_fouraxis_cpu',history=roots[0]['history'],arm='BASE',seed=17,mode='VISITS',model_path=str(n.MODEL),model_identity=n.IDENT,oracle=[1]);srv.p.stdin.write(json.dumps(q)+'\n');srv.p.stdin.flush();bad=json.loads(srv.p.stdout.readline());assert 'error'in bad
    write('C06_SMOKE.json',dict(parity=par,arms=tests,deep=deep,reject=bad))
    write('C06_DIAGNOSTIC_RAW.json',dict(source_identity=sha(old),traces=traces,rows=rows))
    print('SMOKE',len(par),len(tests),len(deep),'deep_other',sum(x['hooks']['merge_blends'] for x in deep),'actual diagnostic leaves',len(rows),flush=True)
def diagnostic_score():
    rows=read('C06_DIAGNOSTIC_RAW.json')['rows'];labs=n.oracle(rows);write('C07_LEAF_LABELS.json',labs);z={r['id']:r['scores'] for r in labs if r['status']=='EXACT'}
    def sign(x):return int(x>0)-int(x<0)
    out=[]
    for r in rows:
        if r['id'] not in z:continue
        a=r['diag'];t=[sign(z[r['id']][x['move']]) for x in a['rows']];h=[x['h'] for x in a['rows']]
        for j,x in enumerate(a['rows']):
            if x['terminal']:assert x['h']==t[j]
        T=max(t);K=min(2,len(t));Tk=max(t[:K]);Hk=max(h[:K]);Ha=max(h);v=a['base'];proof=1. if any(x['terminal'] and x['h']==1 for x in a['rows']) else max(h) if all(x['terminal'] for x in a['rows']) else None
        assert proof is None or proof==T
        gate=abs(v)<=.65 and len(t)>=2
        raw=.5*v+.5*Hk;assert abs(raw-T-(.5*(v-T)+.5*(Tk-T)+.5*(Hk-Tk)))<1e-10
        pred=dict(base=v,legacy=raw if gate else v,proof=proof if proof is not None else v,top2=proof if proof is not None else raw if gate else v,full=proof if proof is not None else .5*v+.5*Ha if gate else v)
        out.append(dict(id=r['id'],source=r['source'],stratum=r['stratum'],T=T,base_error=v-T,omission=Tk-T,max_error=Hk-Tk,full_max_error=Ha-T,gate=gate,proof=proof,pred=pred,errors={k:x-T for k,x in pred.items()}))
    summary={}
    for st in (*ST,'ALL'):
        a=[x for x in out if st=='ALL' or x['stratum']==st]
        if not a:continue
        summary[st]=dict(n=len(a),families=len({x['source'] for x in a}),proofs=sum(x['proof'] is not None for x in a),omitted=sum(x['omission']<0 for x in a),mean_omission=S.mean(x['omission'] for x in a),mean_max_error=S.mean(x['max_error'] for x in a),mean_full_max_error=S.mean(x['full_max_error'] for x in a),gate=sum(x['gate'] for x in a),mae={k:S.mean(abs(x['errors'][k]) for x in a) for k in a[0]['errors']},mse={k:S.mean(x['errors'][k]**2 for x in a) for k in a[0]['errors']})
    write('C07_LEAF_DIAGNOSIS.json',dict(labels=dict(C.Counter(r['status'] for r in labs)),summary=summary,rows=out,raw_sha=sha(D/'C06_DIAGNOSTIC_RAW.json')))
    print('LEAF',json.dumps(summary),flush=True)
def live():
    co=read('C07_COHORT.json');rows=co['rows'];assert len(rows)==48
    start=snap();raw=[]
    with Server() as srv:
        srv.call(rows[0]['history']) # warm cache
        for rep,seed in enumerate(SEEDS):
            for st in ST[::(-1 if rep%2 else 1)]:
                for k,r in enumerate(x for x in rows if x['stratum']==st):
                    for mo in MODES[(k+rep)%3:]+MODES[:(k+rep)%3]:
                        off=(k+rep*5+MODES.index(mo))%len(ARMS)
                        for arm in ARMS[off:]+ARMS[:off]:
                            a=srv.call(r['history'],arm,mo,seed);a.update(id=r['id'],stratum=st,host_load1=os.getloadavg()[0]);raw.append(a)
                print('LIVE',seed,st,len(raw),flush=True)
    end=snap();gzwrite('C08_LIVE_RAW.json.gz',dict(start=start,end=end,interval=interval(start,end),runs=raw))
    print('BLIND',sha(D/'C08_LIVE_RAW.json.gz'),flush=True)
def score():
    roots=read('C07_COHORT.json')['rows'];raw=gzread('C08_LIVE_RAW.json.gz');runs=raw['runs']
    labs=n.oracle(roots);write('C08_LABELS.json',labs);z={r['id']:r['scores'] for r in labs if r['status']=='EXACT'}
    by={};rng=random.Random(202611250607);summary={}
    for a in runs:
        i=a['id'];key=(i,a['seed'],a['mode'],a['arm']);assert key not in by
        if i not in z:continue
        r=a['result'];v=n.utility(z[i][r['best']]);opt=max(n.utility(q) for q in z[i] if q!=-1000)
        by[key]=(opt-v,a)
    for st in ST:
        ids=[x['id'] for x in roots if x['stratum']==st and x['id'] in z];summary[st]={}
        for mo in MODES:
            arms={};paired={}
            for arm in ARMS:
                vals=[by[i,s,mo,arm] for i in ids for s in SEEDS];rr=[x[1] for x in vals]
                d=dict(regret=S.mean(x[0] for x in vals),whole_us=S.mean(x['whole_us'] for x in rr),transport_us=S.mean(x['transport_us'] for x in rr),evals=S.mean(x['result']['evaluator_calls'] for x in rr),visits=S.mean(x['result']['root_n'] for x in rr),caps=sum(x['cause']=='CAP' for x in rr),workers_two=sum(x['hooks']['worker_mask'].bit_count()>=2 for x in rr),causes=dict(C.Counter(x['cause'] for x in rr)),hooks={k:sum(x['hooks'][k] for x in rr) for k in rr[0]['hooks'] if k not in ('leaf_paths','worker_mask')})
                if mo=='NN':d['mean_overshoot']=S.mean(max(0,x['result']['evaluator_calls']-48) for x in rr);d['max_overshoot']=max(max(0,x['result']['evaluator_calls']-48) for x in rr)
                if mo=='TIME':d['mean_overshoot']=S.mean(max(0,x['before_drop_us']-2500) for x in rr);d['max_overshoot']=max(max(0,x['before_drop_us']-2500) for x in rr)
                arms[arm]=d
            for ref in ('BASE','SX'):
                for arm in ARMS:
                    if arm==ref:continue
                    dif=[S.mean(by[i,s,mo,ref][0]-by[i,s,mo,arm][0] for s in SEEDS) for i in ids]
                    boot=sorted(S.mean(rng.choices(dif,k=len(dif))) for _ in range(2000))
                    paired[arm+'/'+ref]=dict(delta=S.mean(dif),ci95=[boot[50],boot[1949]],time_mismatch=abs(arms[arm]['whole_us']/arms[ref]['whole_us']-1),nn_mismatch=abs(arms[arm]['evals']/arms[ref]['evals']-1))
            summary[st][mo]=dict(n=len(ids),arms=arms,paired=paired)
    host=raw['interval']['ok'] and max(x['host_load1'] for x in runs)<=12
    out=dict(groups=summary,labels=dict(C.Counter(x['status'] for x in labs)),runs=len(runs),raw_sha=sha(D/'C08_LIVE_RAW.json.gz'),host_ok=host,host_interval=raw['interval'],max_load1=max(x['host_load1'] for x in runs),scientific_promotion=False)
    write('C08_SUMMARY.json',out)
    for st in ST:
        for mo in MODES:
            g=summary[st][mo];print(st,mo,[(a,round(v['regret'],5),round(v['whole_us'],1),round(v['evals'],2),round(v['visits'],1),v['caps']) for a,v in g['arms'].items()],flush=True)
    print('HOST',host,raw['interval'],'LABELS',out['labels'],flush=True)
if __name__=='__main__':
    lock();{'smoke':smoke_diag,'diagnostic_score':diagnostic_score,'live':live,'score':score}[sys.argv[1]]()
