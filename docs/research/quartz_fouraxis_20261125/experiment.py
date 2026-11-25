"""Actual native four-axis factorial; exact labels only after blind raw checkpoint."""
import collections as C,gzip,itertools,json,math,os,random,statistics as S,subprocess as SP,sys,time
from pathlib import Path
import native as n
from native import D,ROOT,write,read,sha
BIN=ROOT.parent/'quartz_fouraxis_target_20261125/mcts_demo'
ST=('TACTICAL','SELFPLAY')
SEEDS=(17,43)
MASKS=tuple(range(16))
def gzwrite(name,obj):
    p=D/name;assert not p.exists()
    with gzip.open(p,'wt',compresslevel=5) as f:json.dump(obj,f,separators=(',',':'),allow_nan=False)
def gzread(name):
    with gzip.open(D/name,'rt') as f:return json.load(f)
def lock():
    n.prior_lock();m=read('C06_BUILD_IDENTITY.json');assert sha(BIN)==m['binary_sha256']
def snap():
    a=n.host();a['mono']=time.monotonic()
    return a
def interval(a,b):
    dt=b['mono']-a['mono'];swap=sum(b['swap_pages'][k]-a['swap_pages'][k] for k in a['swap_pages'])/max(dt,1e-9)
    return dict(seconds=dt,swap_pages_s=swap,max_endpoint_load=max(a['load'][0],b['load'][0]),ok=swap<=512 and max(a['load'][0],b['load'][0])<=12)
class Server:
    def __init__(self):
        lock();self.p=SP.Popen([str(BIN),'--server'],stdin=SP.PIPE,stdout=SP.PIPE,stderr=SP.DEVNULL,text=True,bufsize=1,env=dict(os.environ,RAYON_NUM_THREADS='2'))
    def call(self,h,mask,seed=17,visits=64,threads=2,standard=False):
        q=dict(cmd='research_connect4_fouraxis_cpu',history=h,mask=mask,seed=seed,visits=visits,threads=threads,standard=standard,model_path=str(n.MODEL),model_identity=n.IDENT)
        t=time.perf_counter_ns();self.p.stdin.write(json.dumps(q)+'\n');self.p.stdin.flush()
        line=self.p.stdout.readline();assert line,('EOF',self.p.poll())
        a=json.loads(line);a['transport_us']=(time.perf_counter_ns()-t)/1000
        assert 'error' not in a,a
        assert a['history']==h and a['mask']==mask and a['seed']==seed and a['threads']==threads and a['visits']==visits and a['standard']==standard and a['model_identity']==n.IDENT
        b,z=n.replay(h);legal={i for i in range(7) if z[i]<6}
        initial=a['initial'];r=a['result'];hs=a['hooks']
        assert initial['root_hash']==r['root_hash'] and initial['root_n']==0 and r['root_n']==visits
        assert initial['evaluator_calls']==1 and r['best'] in legal
        assert {v[0] for v in r['candidates']}==legal and {v[0] for v in initial['candidates']}==legal
        assert 0<len(initial['edges'])<=len(r['edges'])<=len(legal)
        assert len(initial['edges'])==(1 if mask&2 else len(legal))
        assert [x['move'] for x in r['edges']]==[x[0] for x in r['candidates'][:len(r['edges'])]]
        assert sum(e['n'] for e in r['edges'])==visits and abs(sum(e['w'] for e in r['edges'])-r['root_w'])<1e-6
        assert all(e['move'] in legal and e['vl']==0 and abs(e['vvalue'])<1e-6 and abs(e['w'])<=e['n']+1e-6 and e['m2']>=-1e-7 for e in r['edges'])
        assert a['whole_us']+3>=a['setup_us']+a['search_us']+a['drop_us']
        if not standard:
            assert hs['base_evals']+hs['terminal_leaves']==visits
            assert r['evaluator_calls']==1+hs['base_evals']+hs['extra_evals']
            assert hs['extra_evals']+hs['exact_child_evals']==2*hs['refinements']
            if mask&1:assert hs['select_nodes']>=visits
            else:assert hs['select_nodes']==0
            if mask&2:assert hs['width_nodes']>=visits
            else:assert hs['width_nodes']==0
            if not mask&4:assert hs['refinements']==0
            if mask&8:assert hs['backup_edges']==hs['ledger_observations']>=visits and hs['ledger_contexts']>=hs['ledger_nodes']>=hs['ledger_shared_children']
            else:assert hs['backup_edges']==hs['ledger_observations']==0
        return a
    def close(self):
        self.p.stdin.close();assert self.p.wait(timeout=30)==0
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
def smoke():
    old=ROOT.parent/'quartz_nn_multiquanta_20261124/docs/research/quartz_nn_multiquanta_20261124/C07_COHORT.json'
    roots=[x['history'] for x in json.loads(old.read_text())['rows'] if x['group']=='TRAIN' and x['stratum']=='TACTICAL'][:4]
    raw=[];par=[];deep=[]
    with Server() as srv:
        for h in roots:
            a=srv.call(h,0,threads=1,standard=True);b=srv.call(h,0,threads=1)
            assert a['result']==b['result'],(h,a['result'],b['result'])
            par.append(dict(standard=a,research=b))
        for mask in MASKS:raw.append(srv.call(roots[0],mask))
        for h in roots:
            for mask in (8,15):deep.append(srv.call(h,mask,visits=512))
        q=dict(cmd='research_connect4_fouraxis_cpu',history=roots[0],mask=0,seed=17,visits=64,threads=2,model_path=str(n.MODEL),model_identity=n.IDENT,oracle=[1])
        srv.p.stdin.write(json.dumps(q)+'\n');srv.p.stdin.flush();bad=json.loads(srv.p.stdout.readline());assert 'error'in bad
    write('C06_SMOKE.json',dict(parity=par,all_masks=raw,deep=deep,oracle_reject=bad))
    print('SMOKE parity',len(par),'masks',len(raw),'deep',len(deep),'deep merge',sum(x['hooks']['merge_blends'] for x in deep),flush=True)
def live():
    co=read('C07_COHORT.json');rows=co['rows'];assert len(rows)==64
    start=snap();raw=[]
    with Server() as srv:
        srv.call(rows[0]['history'],0,threads=2) # common warmup outside analysis
        for rep,seed in enumerate(SEEDS):
            for st in ST[::(-1 if rep%2 else 1)]:
                for k,r in enumerate(x for x in rows if x['stratum']==st):
                    offset=(k+rep*7)%16
                    for mask in MASKS[offset:]+MASKS[:offset]:
                        a=srv.call(r['history'],mask,seed);a.update(id=r['id'],stratum=st,host_load1=os.getloadavg()[0]);raw.append(a)
                print('LIVE',seed,st,len(raw),flush=True)
    end=snap();gzwrite('C08_LIVE_RAW.json.gz',dict(start=start,end=end,interval=interval(start,end),runs=raw))
    print('BLIND',sha(D/'C08_LIVE_RAW.json.gz'),flush=True)
def mobius(f):
    return [sum((-1)**((t^a).bit_count())*f[a] for a in MASKS if a&t==a) for t in MASKS]
def shapley(m):return [sum(m[t]/t.bit_count() for t in MASKS[1:] if t&(1<<i)) for i in range(4)]
def summarize():
    co=read('C07_COHORT.json');raw=gzread('C08_LIVE_RAW.json.gz');runs=raw['runs'];roots=co['rows']
    labels=n.oracle(roots);write('C08_LABELS.json',labels);z={x['id']:x['scores'] for x in labels if x['status']=='EXACT'}
    by={};rng=random.Random(202611250506);out={}
    for a in runs:
        i=a['id'];mask=a['mask'];assert (i,a['seed'],mask) not in by
        if i not in z:continue
        scores=z[i];r=a['result'];v=n.utility(scores[r['best']]);opt=max(n.utility(q) for q in scores if q!=-1000)
        by[i,a['seed'],mask]=(opt-v,v,a['whole_us'],r['evaluator_calls'],a['search_us'])
    for st in ST:
        ids=[x['id'] for x in roots if x['stratum']==st and x['id'] in z];assert ids
        arms={}
        for mask in MASKS:
            rr=[a for a in runs if a['stratum']==st and a['mask']==mask];v=[by[i,s,mask] for i in ids for s in SEEDS]
            arms[str(mask)]=dict(regret=S.mean(x[0] for x in v),utility=S.mean(x[1] for x in v),whole_us=S.mean(x[2] for x in v),evals=S.mean(x[3] for x in v),search_us=S.mean(x[4] for x in v),transport_us=S.mean(a['transport_us'] for a in rr),hooks={k:sum(a['hooks'][k] for a in rr) for k in rr[0]['hooks']},n=len(v))
        metrics={}
        for name,col in [('utility',1),('whole_us',2),('evals',3)]:
            f=[S.mean(by[i,s,a][col] for i in ids for s in SEEDS) for a in MASKS];m=mobius(f);p=shapley(m)
            assert abs(sum(p)-(f[15]-f[0]))<1e-6
            metrics[name]=dict(values=f,mobius=m,shapley=p,joint_change=f[15]-f[0])
        diffs=[S.mean(by[i,s,0][0]-by[i,s,15][0] for s in SEEDS) for i in ids]
        boot=sorted(S.mean(rng.choices(diffs,k=len(diffs))) for _ in range(4000))
        out[st]=dict(n=len(ids),arms=arms,attribution=metrics,full_vs_baseline=dict(improvement=S.mean(diffs),ci95=[boot[100],boot[3899]],positive=sum(x>0 for x in diffs),negative=sum(x<0 for x in diffs),time_mismatch=abs(arms['15']['whole_us']/arms['0']['whole_us']-1),eval_ratio=arms['15']['evals']/arms['0']['evals']))
    host=raw['interval']['ok'] and max(a['host_load1'] for a in runs)<=12
    result=dict(groups=out,labels=dict(C.Counter(x['status'] for x in labels)),runs=len(runs),host_ok=host,host_interval=raw['interval'],max_load1=max(a['host_load1'] for a in runs),raw_sha=sha(D/'C08_LIVE_RAW.json.gz'),scientific_promotion=False)
    write('C08_SUMMARY.json',result)
    for st,r in out.items():
        print(st,'full',r['full_vs_baseline'],'shapley utility',r['attribution']['utility']['shapley'],'all arms',[(a,round(v['regret'],6),round(v['whole_us'],1),round(v['evals'],2)) for a,v in r['arms'].items()],flush=True)
    print('HOST',host,raw['interval'],'LABELS',result['labels'],flush=True)
if __name__=='__main__':
    lock();{'smoke':smoke,'live':live,'score':summarize}[sys.argv[1]]()
