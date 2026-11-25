"""Actual native certificate-aware research; external exact solver is offline only."""
import collections as C,gzip,json,math,os,random,statistics as S,subprocess as SP,sys,time
from pathlib import Path
import native as n
from native import D,ROOT,write,read,sha
from population import exclusions
BIN=ROOT.parent/'quartz_fouraxis03_target_20261125/mcts_demo'
ST=('TACTICAL','SELFPLAY','HARD');SEEDS=(17,43)
ARMS=('BASE','SX','SAMPLE','ROOT','ACTIVE','ACTIVE_STOP','FULL','ACTIVE_FULL')
MODES=('VISITS','TIME');GRAPH=('ROOT','ACTIVE','ACTIVE_STOP','ACTIVE_FULL');BMODES=('TEMPORAL','MERGE')
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
def key(h):
    b,z=n.replay(h)
    return '%022x'%((int(len(h)%2==0)<<84)|sum((v+1)<<(2*i) for i,v in enumerate(n.flat(b))))
def regret_bound(actions,a):
    iv=dict(actions)
    return max(0,max((v[1] for k,v in iv.items() if k!=a),default=-1)-iv[a][0])/2
class Server:
    def __init__(self):
        lock();self.p=SP.Popen([str(BIN),'--server'],stdin=SP.PIPE,stdout=SP.PIPE,stderr=SP.DEVNULL,text=True,bufsize=1,env=dict(os.environ,RAYON_NUM_THREADS='2'))
    def call(self,h,arm='BASE',mode='VISITS',seed=17,target=0,reserve=0):
        q=dict(cmd='research_connect4_fouraxis_cpu',history=h,arm=arm,mode=mode,seed=seed,model_path=str(n.MODEL),model_identity=n.IDENT,target_us=target,reserve_us=reserve)
        t=time.perf_counter_ns();self.p.stdin.write(json.dumps(q)+'\n');self.p.stdin.flush();line=self.p.stdout.readline()
        assert line,('EOF',self.p.poll(),q)
        a=json.loads(line);a['transport_us']=(time.perf_counter_ns()-t)/1000
        assert 'error' not in a,a
        assert a['history']==h and a['model_identity']==n.IDENT and a['arm']==arm and a['mode']==mode and a['seed']==seed
        b,z=n.replay(h);legal={i for i in range(7) if z[i]<6}
        p=a['prefix'];r=a['result'];hs=a['hooks'];ph=a['prefix_hooks'];vis=8+8*a['blocks']
        assert p['root_n']==8 and r['root_n']==vis<=512 and p['root_hash']==r['root_hash']
        for x in (p,r):
            assert {v[0] for v in x['candidates']}==legal and x['best'] in legal
            assert [y['move'] for y in x['edges']]==[y[0] for y in x['candidates'][:len(x['edges'])]]
            assert sum(y['n'] for y in x['edges'])==x['root_n'] and abs(sum(y['w'] for y in x['edges'])-x['root_w'])<1e-6
            assert all(y['vl']==0 and abs(y['vvalue'])<1e-6 and abs(y['w'])<=y['n']+1e-6 and y['m2']>=-1e-7 for y in x['edges'])
        if mode!='STANDARD':assert p['evaluator_calls']==1+ph['base_evals'] and ph['base_evals']+ph['terminal_leaves']==8
        assert hs['base_evals']+hs['terminal_leaves']+hs['graph_solved_skips']==vis-8
        assert r['evaluator_calls']==p['evaluator_calls']+hs['base_evals']+hs['extra_evals']
        assert a['best'] in legal and a['whole_us']+3>=a['setup_us']+a['prefix_us']+a['continuation_us']+a['drop_us']
        if mode in ('VISITS','GRAPH'):assert vis==128 or (arm=='ACTIVE_STOP' and a['cause']=='CERTIFIED')
        if mode=='BTRACE':assert vis==512 and len(hs['b_events'])<=8
        if mode=='TIME':assert a['before_drop_us']>=target-reserve or vis==512 or (arm=='ACTIVE_STOP' and a['cause']=='CERTIFIED')
        if arm not in ('FULL','ACTIVE_FULL'):assert hs['extra_evals']==0
        if arm in GRAPH:
            g=a['proof'];assert g['key']==key(h)
            d=dict(g['actions']);assert set(d)==legal and all(-1<=lo<=hi<=1 for lo,hi in d.values())
            assert g['interval']==[max(x[0] for x in d.values()),max(x[1] for x in d.values())]
            assert g['allowed']==[m for m,iv in g['actions'] if iv[1]>=g['interval'][0]]
            w=g['witness'];assert (w is None)==all(regret_bound(g['actions'],m)>0 for m in d)
            if w is not None:assert regret_bound(g['actions'],w)==0 and a['best']==w
            if a['cause']=='CERTIFIED':assert arm=='ACTIVE_STOP' and w is not None
            assert a['best'] in g['allowed'] and g['solved']<=g['nodes']
            if arm=='ROOT':assert hs['graph_solved_skips']==hs['graph_plans']==0
        else:assert a['proof'] is None and a['best']==r['best'] and hs['graph_register']==0
        if mode=='BTRACE':
            for ev in hs['b_events']:
                assert ev['other']==(arm=='MERGE') and ev['n']>=1 and -1<=ev['y']<=1 and -1<=ev['m']<=1
                hh=h+''.join(str(x+1) for x in ev['path']);assert key(hh)==ev['child_key'] and key(hh[:-1])==ev['parent_key']
        return a
    def close(self):
        self.p.stdin.close();assert self.p.wait(timeout=30)==0
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
def smoke():
    old=ROOT.parent/'quartz_nn_multiquanta_20261124/docs/research/quartz_nn_multiquanta_20261124/C07_COHORT.json'
    roots=[r['history'] for r in json.loads(old.read_text())['rows'] if r['group']=='TRAIN' and r['stratum']=='TACTICAL'][:4]
    par=[];arms=[];graphs=[]
    with Server() as srv:
        for h in roots:
            a=srv.call(h,mode='STANDARD');b=srv.call(h,mode='PREFIX')
            assert a['result']==b['result'];par.append(dict(standard=a,research=b))
        for arm in ARMS:arms.append(srv.call(roots[0],arm))
        for h in ('121212','112233','1234','3214'):
            graphs.append(srv.call(h,'ACTIVE','GRAPH'))
        a=srv.call('121212','ACTIVE_STOP')
        assert a['cause']=='CERTIFIED' and a['result']['root_n']==8 and a['best']==0
        q=dict(cmd='research_connect4_fouraxis_cpu',history=roots[0],arm='BASE',seed=17,mode='VISITS',model_path=str(n.MODEL),model_identity=n.IDENT,oracle=[1])
        srv.p.stdin.write(json.dumps(q)+'\n');srv.p.stdin.flush();bad=json.loads(srv.p.stdout.readline());assert 'error' in bad
    gzwrite('C06_SMOKE.json.gz',dict(parity=par,arms=arms,graphs=graphs,stop=a,reject=bad))
    print('SMOKE',len(par),len(arms),len(graphs),'root immediate STOP',a['result']['root_n'],'graph nodes',[x['proof']['nodes'] for x in graphs],flush=True)
def diagnostics():
    co=read('C07_COHORT.json');rr=co['rows'];train=[x for x in rr if x['group']=='TRAIN'];cal=[x for x in rr if x['group']=='CAL']
    assert train and cal
    before=snap();timing=[];graphs=[];br=[]
    with Server() as srv:
        srv.call(train[0]['history'])
        for seed in SEEDS:
            for st in ST:
                for r in [x for x in train if x['stratum']==st]:
                    a=srv.call(r['history'],seed=seed);a.update(id=r['id'],stratum=st);timing.append(a)
        end=snap()
        for st in ST:
            for r in [x for x in train if x['stratum']==st][:2]:
                a=srv.call(r['history'],'ACTIVE','GRAPH');a.update(id=r['id'],stratum=st);graphs.append(a)
        for group in (train,cal):
            for r in group:
                for arm in BMODES:
                    a=srv.call(r['history'],arm,'BTRACE');a.update(id=r['id'],group=r['group'],stratum=r['stratum']);br.append(a)
            print('BTRACE',group[0]['group'],len(br),'events',sum(len(a['hooks']['b_events']) for a in br),flush=True)
    cost={}
    for st in ST:
        a=[x for x in timing if x['stratum']==st];assert a
        med=S.median(x['whole_us'] for x in a);target=min(10000,max(2000,math.ceil(med)))
        reserve=min(target-1,math.ceil(S.median(x['drop_us'] for x in a))+100)
        cost[st]=dict(n=len(a),median_whole_us=med,median_drop_us=S.median(x['drop_us'] for x in a),target=target,reserve=reserve)
    gzwrite('C07_DIAG_RAW.json.gz',dict(timing=timing,graphs=graphs,btrace=br,timing_host=interval(before,end),start=before,end=end))
    write('C07_COST_LOCK.json',dict(cost=cost,diagnostic_raw_sha=sha(D/'C07_DIAG_RAW.json.gz'),cohort_sha=sha(D/'C07_COHORT.json'),binary_sha=sha(BIN)))
    print('COST',json.dumps(cost),flush=True)
def fit_b():
    co=read('C07_COHORT.json');raw=gzread('C07_DIAG_RAW.json.gz');oldp,oldb,meta=exclusions()
    # No overlap with old labelled boards or any current root, and no shared
    # diagnostic state between TRAIN and CAL. Within a split/mode keep first
    # occurrence per family/canonical child; label each unique actual state once.
    banned=set(oldb)|{n.cb(r['board']) for r in co['rows']}
    train_boards=set();rows=[];reject=C.Counter();seen=set();queries={}
    for group in ('TRAIN','CAL'):
        for a in raw['btrace']:
            if a['group']!=group:continue
            for ev in a['hooks']['b_events']:
                h=a['history']+''.join(str(x+1) for x in ev['path'])
                b,z=n.replay(h);assert not any(n.won(x) for x in b) and not all(x==6 for x in z)
                c=n.cb(n.flat(b));k=(a['id'],a['arm'],c)
                if c in banned:reject['old_or_root']+=1;continue
                if group=='CAL' and c in train_boards:reject['cross_split']+=1;continue
                if k in seen:reject['same_family_mode_board']+=1;continue
                seen.add(k)
                if group=='TRAIN':train_boards.add(c)
                qkey=key(h)
                if qkey not in queries:queries[qkey]=dict(id=qkey,history=h,legal=[i for i,v in enumerate(z) if v<6])
                rows.append(dict(source=a['id'],group=group,stratum=a['stratum'],arm=a['arm'],history=h,query=qkey,event=ev))
    write('C07_B_QUERIES.json',dict(rows=rows,queries=list(queries.values()),excluded=dict(reject)))
    print('B_QUERIES',len(queries),'rows',len(rows),'excluded',dict(reject),flush=True)
    labs=n.oracle(list(queries.values()));write('C07_B_LABELS.json',labs)
    z={x['id']:x['scores'] for x in labs if x['status']=='EXACT'}
    by=C.defaultdict(list);sign=lambda x:int(x>0)-int(x<0)
    for r in rows:
        if r['query'] not in z:continue
        v=max(sign(x) for x in z[r['query']] if x!=-1000)
        y=r['event']['y']-v;m=r['event']['m']-v
        by[r['arm'],r['group'],r['source']].append((y*y,m*m,y*m,r['event']['n']))
    out={}
    for arm in BMODES:
        groups={}
        for group in ('TRAIN','CAL'):
            fam=[v for (a,g,_),v in by.items() if a==arm and g==group]
            if not fam:groups[group]=dict(families=0,events=0);continue
            s,t,r=[S.mean(S.mean(x[j] for x in f) for f in fam) for j in range(3)]
            groups[group]=dict(families=len(fam),events=sum(len(f) for f in fam),s=s,t=t,r=r)
        tr=groups['TRAIN'];ca=groups['CAL']
        eta=None
        if tr['families']>=5 and ca['families']>=5:
            d=tr['s']+tr['t']-2*tr['r'];eta=min(.25,max(0,(tr['s']-tr['r'])/d)) if d>1e-15 else 0.
            for g in (tr,ca):
                for name,a in [('fixed025',.25),('fit',eta)]:
                    g['mse_'+name]=g['s']+2*a*(g['r']-g['s'])+a*a*(g['s']+g['t']-2*g['r'])
        out[arm]=dict(groups=groups,eta=eta,support_ok=eta is not None)
    write('C07_B_DIAGNOSIS.json',dict(results=out,labels=dict(C.Counter(x['status'] for x in labs)),queries=len(queries),events=len(rows),raw_sha=sha(D/'C07_DIAG_RAW.json.gz'),query_sha=sha(D/'C07_B_QUERIES.json')))
    print('B_DIAG',json.dumps(out),flush=True)
def live():
    co=read('C07_COHORT.json');rows=[r for r in co['rows'] if r['group']=='TEST'];m=read('C07_COST_LOCK.json')
    assert m['cohort_sha']==sha(D/'C07_COHORT.json') and m['diagnostic_raw_sha']==sha(D/'C07_DIAG_RAW.json.gz') and m['binary_sha']==sha(BIN)
    before=snap();raw=[]
    with Server() as srv:
        srv.call(rows[0]['history'])
        for rep,seed in enumerate(SEEDS):
            for st in ST[::(-1 if rep%2 else 1)]:
                for k,r in enumerate(x for x in rows if x['stratum']==st):
                    for mo in MODES[(k+rep)%2:]+MODES[:(k+rep)%2]:
                        off=(k+rep*3+MODES.index(mo))%len(ARMS)
                        for arm in ARMS[off:]+ARMS[:off]:
                            cfg=m['cost'][st];a=srv.call(r['history'],arm,mo,seed,cfg['target'] if mo=='TIME' else 0,cfg['reserve'] if mo=='TIME' else 0)
                            a.update(id=r['id'],stratum=st,host_load1=os.getloadavg()[0]);raw.append(a)
                print('LIVE',seed,st,len(raw),flush=True)
    end=snap();gzwrite('C08_LIVE_RAW.json.gz',dict(start=before,end=end,interval=interval(before,end),runs=raw))
    print('BLIND',sha(D/'C08_LIVE_RAW.json.gz'),flush=True)
def score():
    roots=[r for r in read('C07_COHORT.json')['rows'] if r['group']=='TEST'];raw=gzread('C08_LIVE_RAW.json.gz');runs=raw['runs']
    labs=n.oracle(roots);write('C08_LABELS.json',labs);z={r['id']:r['scores'] for r in labs if r['status']=='EXACT'}
    by={};rng=random.Random(202611250708);summary={};checked=0;certs=0
    for a in runs:
        i=a['id'];k=(i,a['seed'],a['mode'],a['arm']);assert k not in by
        if i not in z:continue
        scores=z[i];opt=max(n.utility(q) for q in scores if q!=-1000);v=n.utility(scores[a['best']]);g=a['proof'];reg=opt-v
        if g is not None:
            sign=lambda x:int(x>0)-int(x<0)
            for j,(lo,hi) in g['actions']:
                assert lo<=sign(scores[j])<=hi;checked+=1
            if g['witness'] is not None:assert reg==0;certs+=1
            assert reg<=regret_bound(g['actions'],a['best'])
        by[k]=(reg,a)
    for st in ST:
        ids=[x['id'] for x in roots if x['stratum']==st and x['id'] in z];summary[st]={}
        if not ids:continue
        for mo in MODES:
            arms={};paired={}
            for arm in ARMS:
                vals=[by[i,s,mo,arm] for i in ids for s in SEEDS];rr=[x[1] for x in vals];gg=[a for a in rr if a['proof'] is not None]
                d=dict(regret=S.mean(x[0] for x in vals),whole_us=S.mean(x['whole_us'] for x in rr),transport_us=S.mean(x['transport_us'] for x in rr),evals=S.mean(x['result']['evaluator_calls'] for x in rr),visits=S.mean(x['result']['root_n'] for x in rr),caps=sum(x['cause']=='CAP' for x in rr),stops=sum(x['cause']=='CERTIFIED' for x in rr),workers_two=sum(x['hooks']['worker_mask'].bit_count()>=2 for x in rr),certificates=sum(x['proof']['witness'] is not None for x in gg),root_solved=sum(x['proof']['interval'][0]==x['proof']['interval'][1] for x in gg),root_override=sum(x['best']!=x['result']['best'] for x in rr),mean_bound=S.mean(regret_bound(x['proof']['actions'],x['best']) for x in gg) if gg else None,mean_best_bound=S.mean(min(regret_bound(x['proof']['actions'],j) for j,_ in x['proof']['actions']) for x in gg) if gg else None,graph_changes=sum(x['proof']['changes'] for x in gg),graph_shared_notices=sum(x['proof']['shared_notices'] for x in gg),hooks={k:sum(x['hooks'][k] for x in rr) for k in rr[0]['hooks'] if k not in ('leaf_paths','worker_mask','b_events')})
                arms[arm]=d
            for ref in ('BASE','SX','SAMPLE','ACTIVE'):
                for arm in ARMS:
                    if arm==ref:continue
                    dif=[S.mean(by[i,s,mo,ref][0]-by[i,s,mo,arm][0] for s in SEEDS) for i in ids]
                    boot=sorted(S.mean(rng.choices(dif,k=len(dif))) for _ in range(2000))
                    paired[arm+'/'+ref]=dict(delta=S.mean(dif),ci95=[boot[50],boot[1949]],time_mismatch=abs(arms[arm]['whole_us']/arms[ref]['whole_us']-1),nn_mismatch=abs(arms[arm]['evals']/arms[ref]['evals']-1))
            summary[st][mo]=dict(n=len(ids),arms=arms,paired=paired)
    host=raw['interval']['ok'] and max(a['host_load1'] for a in runs)<=12
    out=dict(groups=summary,labels=dict(C.Counter(x['status'] for x in labs)),runs=len(runs),raw_sha=sha(D/'C08_LIVE_RAW.json.gz'),host_ok=host,host_interval=raw['interval'],max_load1=max(x['host_load1'] for x in runs),checked_root_action_intervals=checked,certificates_valid=certs,scientific_promotion=False)
    write('C08_SUMMARY.json',out)
    for st in ST:
        for mo,g in summary[st].items():
            print(st,mo,[(a,round(v['regret'],5),round(v['whole_us']),round(v['evals'],1),round(v['visits'],1),v['certificates'],v['stops']) for a,v in g['arms'].items()],flush=True)
    print('HOST',host,raw['interval'],'LABELS',out['labels'],'ACTION_INTERVALS',checked,'CERT',certs,flush=True)
if __name__=='__main__':
    lock();{'smoke':smoke,'diagnostics':diagnostics,'fit_b':fit_b,'live':live,'score':score}[sys.argv[1]]()
