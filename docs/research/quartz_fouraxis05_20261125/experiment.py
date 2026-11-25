"""Native local interventions, stream evidence before offline exact labels."""
import collections as C,gzip,json,math,os,random,statistics as S,subprocess as SP,sys,time
from pathlib import Path
import native as n
from native import D,ROOT,write,read,sha
from population import exclusions
BIN=ROOT.parent/'quartz_fouraxis05_target_20261125/mcts_demo'
ST=('TACTICAL','SELFPLAY','HARD');SEEDS=(17,43)
ARMS=('BASE','SAFE','ACTIVE','FOCUS','EXPAND','FRONTIER','EVAL','WRITER','MERGE')
MODES=('WORK','NN','TIME')
GRID=(.05,.10,.20)
def lock():
    n.prior_lock();m=read('C06_BUILD_IDENTITY.json');assert sha(BIN)==m['binary_sha256']
def snap():
    a=n.host();a['mono']=time.monotonic();return a
def interval(a,b):
    dt=b['mono']-a['mono'];swap={k:b['swap_pages'][k]-a['swap_pages'][k] for k in a['swap_pages']}
    return dict(seconds=dt,swap_pages_s=sum(swap.values())/max(dt,1e-9),swap_delta=swap,max_endpoint_load=max(a['load'][0],b['load'][0]),ok=sum(swap.values())/dt<=512 and max(a['load'][0],b['load'][0])<=12)
def key(h):
    b,z=n.replay(h)
    return '%022x'%((int(len(h)%2==0)<<84)|sum((v+1)<<(2*i) for i,v in enumerate(n.flat(b))))
def risk(a):
    return min(max(0,max((u for j,(_,u) in a if j!=m),default=-1)-l) for m,(l,u) in a)
def width(a):return sum(u-l for _,(l,u) in a)
def rb(actions,a):
    iv=dict(actions);return max(0,max((v[1] for k,v in iv.items() if k!=a),default=-1)-iv[a][0])/2
def last(a):return a['steps'][-1]['post'] if a['steps'] else a['prefix']
def records(name):
    with gzip.open(D/name,'rt') as f:
        for line in f:yield json.loads(line)
def write_gz(name,obj):
    p=D/name;assert not p.exists()
    with gzip.open(p,'wt',compresslevel=3) as f:json.dump(obj,f,separators=(',',':'),allow_nan=False)
def read_gz(name):
    with gzip.open(D/name,'rt') as f:return json.load(f)
def emit(f,a):f.write(json.dumps(a,separators=(',',':'),allow_nan=False)+'\n')
def delta(pre,post,k):return post['hooks'][k]-pre['hooks'][k]
class Server:
    def __init__(self):
        lock();self.err=open(BIN.parent/'server_stderr.log','a+')
        self.p=SP.Popen(['nice','-n','10',str(BIN),'--server'],stdin=SP.PIPE,stdout=SP.PIPE,stderr=self.err,text=True,bufsize=1,env=dict(os.environ,RAYON_NUM_THREADS='2'))
    def send(self,q):
        t=time.perf_counter_ns();self.p.stdin.write(json.dumps(q)+'\n');self.p.stdin.flush();line=self.p.stdout.readline()
        if not line:self.err.flush();self.err.seek(0);raise AssertionError(('EOF',self.p.poll(),q,self.err.read()[-4000:]))
        a=json.loads(line);a['transport_us']=(time.perf_counter_ns()-t)/1000;return a
    def call(self,r,arm='BASE',mode='WORK',seed=17,target=0,reserve=0,retention=0.):
        q=dict(cmd='research_connect4_fouraxis_local_cpu',history=r['history'],arm=arm,mode=mode,seed=seed,model_path=str(n.MODEL),model_identity=n.IDENT,target_us=target,reserve_us=reserve,retention=retention)
        a=self.send(q);assert 'error' not in a,a
        assert a['arm']==arm and a['mode']==mode and a['seed']==seed and a['threads']==(1 if mode=='DIAG' else 2) and a['model_identity']==n.IDENT
        assert a['history']==r['history'] and a['retention']==retention and a['target_us']==target and a['reserve_us']==reserve
        b,z=n.replay(a['history']);legal={k for k in range(7) if z[k]<6}
        p=a['prefix'];ss=[p]+[s['post'] for s in a['steps']];end=ss[-1];h=end['hooks'];ph=p['hooks'];vis=32+8*len(a['steps'])
        assert p['result']['root_n']==32 and vis<=256 and end['result']['root_n']==vis and a['best']==end['best']
        for j,x in enumerate(ss):
            v=x['result'];hs=x['hooks']
            assert v['root_n']==32+8*j and v['root_hash']==p['result']['root_hash']
            assert {m for m,_ in v['candidates']}==legal and v['best'] in legal and x['best'] in legal
            assert [y['move'] for y in v['edges']]==[y[0] for y in v['candidates'][:len(v['edges'])]]
            assert sum(y['n'] for y in v['edges'])==v['root_n'] and abs(sum(y['w'] for y in v['edges'])-v['root_w'])<1e-6
            assert all(y['vl']==0 and abs(y['vvalue'])<1e-6 and abs(y['w'])<=y['n']+1e-6 and y['m2']>=-1e-7 for y in v['edges'])
            assert hs['base_evals']+hs['terminal_leaves']+hs['graph_solved_skips']==v['root_n']
            assert v['evaluator_calls']==1+hs['base_evals']+hs['extra_evals']
            assert hs['graph_repeat_skips']<=hs['graph_solved_skips']
            g=x['proof']
            if arm=='BASE':assert g is None and x['best']==v['best']
            else:
                assert g['key']==key(a['history']);d=dict(g['actions']);assert set(d)==legal and all(-1<=l<=u<=1 for l,u in d.values())
                assert g['interval']==[max(v[0] for v in d.values()),max(v[1] for v in d.values())]
                assert g['allowed']==[m for m,iv in g['actions'] if iv[1]>=g['interval'][0]]
                w=g['witness'];assert (w is None)==(risk(g['actions'])>0)
                if w is not None:assert rb(g['actions'],w)==0 and x['best']==w
                assert x['best'] in g['allowed']
                if j:
                    prev=ss[j-1]['proof'];assert prev['witness'] is None and all(dict(prev['actions'])[m][0]<=l<=u<=dict(prev['actions'])[m][1] for m,(l,u) in g['actions'])
            if j:
                prev=ss[j-1];assert all(hs[k]>=prev['hooks'][k] for k,v0 in hs.items() if isinstance(v0,int) and k not in ('worker_mask','ledger_nodes','ledger_contexts','ledger_shared_children','ledger_observations'))
                plan=a['steps'][j-1]['plan']
                if arm in ('FOCUS','EXPAND','FRONTIER'):
                    assert plan['mode']==arm and delta(prev,x,'local_claims')==2
                    if arm=='FRONTIER':assert 0<=plan['scored']<=64 and delta(prev,x,'local_scored')==plan['scored']
                else:assert plan is None and delta(prev,x,'local_claims')==0
                if arm=='FOCUS':assert delta(prev,x,'local_focus_edges')<=2
                if arm=='EXPAND':assert delta(prev,x,'local_expand_edges')<=2
        assert a['whole_us']>=a['before_drop_us']>=a['prefix_us'] and a['drop_us']>=0
        assert a['guard_best'] in legal
        if a['cause']=='CERTIFIED':assert arm!='BASE' and end['proof']['witness'] is not None
        elif mode in ('DIAG','WORK'):assert a['cause']=='VISITS' and vis==(64 if mode=='DIAG' else 128)
        elif mode=='NN':assert a['cause']=='NN' and end['result']['evaluator_calls']>=64 or a['cause']=='CAP' and vis==256
        else:assert a['cause']=='TIME' and a['before_drop_us']>=target-reserve or a['cause']=='CAP' and vis==256
        if arm!='EVAL':assert h['extra_evals']==0
        if arm=='WRITER':assert h['ledger_observations']==h['temporal_blends']==h['merge_blends']==0
        elif arm=='MERGE':assert h['ledger_observations']==h['backup_edges'] and h['temporal_blends']==0
        else:assert h['backup_edges']==h['ledger_observations']==h['merge_blends']==h['temporal_blends']==0
        if arm in ('WRITER','MERGE'):assert h['backup_edges']>=vis-32
        assert h['backup_changed']<=h['merge_blends']<=h['backup_edges']-h['terminal_edges']
        assert ph['backup_edges']==ph['local_claims']==ph['extra_evals']==0
        if a['retention_margin'] is None or retention==0 or a['retention_margin']>retention:assert a['guard_best']==a['best']
        else:assert a['guard_best']==p['best']!=a['best']
        return a
    def close(self):
        self.p.stdin.close();rc=self.p.wait(timeout=30);self.err.close();assert rc==0,rc
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
def smoke():
    rr=[dict(history=h) for h in ('121212','1234','3214','112233')]
    tests=[];bad=[];par=[]
    with Server() as srv:
        for arm in ARMS:tests.append(srv.call(rr[1],arm,'DIAG'))
        for r in rr:tests.append(srv.call(r,'FRONTIER','DIAG'))
        assert tests[9]['cause']=='CERTIFIED' and not tests[9]['steps']
        g0=tests[1]['prefix'];assert all(a['prefix']==g0 for a in tests[2:9])
        # Prior standard 8-visit parity preserved on actual existing endpoint.
        for r in rr:
            q=dict(cmd='research_connect4_fouraxis_cpu',history=r['history'],arm='BASE',mode='STANDARD',seed=17,model_path=str(n.MODEL),model_identity=n.IDENT)
            a=srv.send(q);q['mode']='PREFIX';b=srv.send(q);assert 'error' not in a and 'error' not in b and a['result']==b['result'];par.append(dict(standard=a,research=b))
        q=dict(cmd='research_connect4_fouraxis_local_cpu',history=rr[1]['history'],arm='SAFE',mode='DIAG',seed=17,model_path=str(n.MODEL),model_identity=n.IDENT,oracle=[1])
        a=srv.send(q);assert 'error' in a;bad.append(a)
    write_gz('C06_SMOKE.json.gz',dict(tests=tests,parity=par,reject=bad))
    print('SMOKE',[(a['history'],a['arm'],a['cause'],len(a['steps']),last(a)['hooks']['local_frontier_edges'],last(a)['hooks']['local_expand_edges'],last(a)['hooks']['local_focus_edges']) for a in tests],flush=True)
def calibration():
    rows=read('C07_COHORT.json')['rows'];train=[r for r in rows if r['group']=='TRAIN'];cal=[r for r in rows if r['group']=='CAL']
    assert len(train)==36 and len(cal)==24
    before=snap();timing=[]
    with Server() as srv:
        srv.call(train[0])
        for seed in SEEDS:
            for r in train:
                a=srv.call(r,'BASE','WORK',seed);a.update(id=r['id'],stratum=r['stratum'],group='TRAIN');timing.append(a)
        end=snap()
        with gzip.open(D/'C07_DIAG_RAW.jsonl.gz','wt',compresslevel=3) as f:
            for group in (train,cal):
                for i,r in enumerate(group):
                    off=i%len(ARMS)
                    for arm in ARMS[off:]+ARMS[:off]:
                        a=srv.call(r,arm,'DIAG',17);a.update(id=r['id'],stratum=r['stratum'],group=r['group']);emit(f,a)
                print('DIAG',group[0]['group'],len(group)*len(ARMS),flush=True)
    cost={}
    for st in ST:
        x=[a for a in timing if a['stratum']==st];med=S.median(a['whole_us'] for a in x);drop=S.median(a['drop_us'] for a in x);target=min(20000,max(3000,math.ceil(med)));reserve=min(target-1,math.ceil(drop)+150)
        cost[st]=dict(n=len(x),median_whole_us=med,median_drop_us=drop,target=target,reserve=reserve)
    write_gz('C07_TIMING_RAW.json.gz',dict(rows=timing,host=interval(before,end),start=before,end=end))
    write('C07_COST_LOCK.json',dict(cost=cost,timing_sha=sha(D/'C07_TIMING_RAW.json.gz'),diag_sha=sha(D/'C07_DIAG_RAW.jsonl.gz'),cohort_sha=sha(D/'C07_COHORT.json'),binary_sha=sha(BIN)))
    print('COST',cost,'CAL_HOST',interval(before,end),flush=True)
def fit():
    rr=[r for r in read('C07_COHORT.json')['rows'] if r['group']!='TEST'];labs=n.oracle(rr);write('C07_DIAG_LABELS.json',labs)
    z={x['id']:x['scores'] for x in labs if x['status']=='EXACT'};assert len(z)==60
    by=C.defaultdict(list);all_fam=C.defaultdict(set)
    for a in records('C07_DIAG_RAW.jsonl.gz'):
        if a['arm']=='BASE':continue
        k=(a['group'],a['id']);all_fam[a['group']].add(a['id']);v=z[a['id']]
        d=n.utility(v[a['prefix']['best']])-n.utility(v[a['best']])
        by[k].append((a['retention_margin'],d,a['arm']))
    result={}
    for t in GRID:
        gs={}
        for group in ('TRAIN','CAL'):
            fam=[v for (g,_),v in by.items() if g==group];ex=[f for f in fam if any(m is not None and m<=t for m,_,_ in f)]
            change=[d for f in fam for m,d,_ in f if m is not None and m<=t]
            gs[group]=dict(families=len(fam),exposed_families=len(ex),exposed_pairs=len(change),mean=S.mean(S.mean(d if m is not None and m<=t else 0 for m,d,_ in f) for f in fam),better=sum(x>0 for x in change),worse=sum(x<0 for x in change))
        result[str(t)]=gs
    ok=[t for t in GRID if result[str(t)]['TRAIN']['exposed_families']>=6 and result[str(t)]['TRAIN']['mean']>0]
    cand=max(ok,key=lambda t:(result[str(t)]['TRAIN']['mean'],-t)) if ok else 0.
    threshold=cand if cand and result[str(cand)]['CAL']['exposed_families']>=4 and result[str(cand)]['CAL']['mean']>=0 else 0.
    write('C07_RETENTION_LOCK.json',dict(grid=result,train_candidate=cand,threshold=threshold,labels_sha=sha(D/'C07_DIAG_LABELS.json'),diag_sha=sha(D/'C07_DIAG_RAW.jsonl.gz'),reason='admitted conditional output heuristic' if threshold else 'insufficient positive train/cal support'))
    print('RETENTION',json.dumps(dict(grid=result,candidate=cand,threshold=threshold)),flush=True)
def live():
    rows=[r for r in read('C07_COHORT.json')['rows'] if r['group']=='TEST'];assert len(rows)==72
    m=read('C07_COST_LOCK.json');rlock=read('C07_RETENTION_LOCK.json')
    assert m['cohort_sha']==sha(D/'C07_COHORT.json') and m['diag_sha']==sha(D/'C07_DIAG_RAW.jsonl.gz')==rlock['diag_sha'] and m['binary_sha']==sha(BIN)
    path=D/'C08_LIVE_RAW.jsonl.gz';assert not path.exists();count=0;max_load=0;before=snap()
    with Server() as srv,gzip.open(path,'wt',compresslevel=3) as f:
        srv.call(rows[0])
        for rep,seed in enumerate(SEEDS):
            for st in ST[::(-1 if rep%2 else 1)]:
                for k,r in enumerate(x for x in rows if x['stratum']==st):
                    for mo in MODES[(k+rep)%3:]+MODES[:(k+rep)%3]:
                        off=(k+rep*5+MODES.index(mo))%len(ARMS)
                        for arm in ARMS[off:]+ARMS[:off]:
                            cfg=m['cost'][st];b0=snap();a=srv.call(r,arm,mo,seed,cfg['target'] if mo=='TIME' else 0,cfg['reserve'] if mo=='TIME' else 0,rlock['threshold']);b1=snap()
                            a.update(id=r['id'],stratum=st,host_interval=interval(b0,b1));max_load=max(max_load,*[b0['load'][0],b1['load'][0]])
                            emit(f,a);count+=1
                print('LIVE',seed,st,count,flush=True)
    end=snap();write('C08_LIVE_HOST.json',dict(start=before,end=end,interval=interval(before,end),max_load1=max_load,runs=count,raw_sha=sha(path)))
    print('BLIND',count,sha(path),flush=True)
def summarize_records(seq,z):
    out=[];checked=cert=0
    for a in seq:
        v=z[a['id']];opt=max(n.utility(x) for x in v if x!=-1000)
        ss=[a['prefix']]+[s['post'] for s in a['steps']]
        u=[n.utility(v[x['best']]) for x in ss];gains=[y-x for x,y in zip(u,u[1:])];flips=[x['best']!=y['best'] for x,y in zip(ss,ss[1:])]
        assert abs(sum(gains)-(u[-1]-u[0]))<1e-10
        for x,uv in zip(ss,u):
            g=x['proof']
            if g is not None:
                for j,(l,h) in g['actions']:
                    assert l<=(v[j]>0)-(v[j]<0)<=h;checked+=1
                assert opt-uv<=rb(g['actions'],x['best'])
                if g['witness'] is not None:assert opt==uv;cert+=1
        p=ss[0];f=ss[-1];d0=None if p['proof'] is None else risk(p['proof']['actions']);w0=None if p['proof'] is None else width(p['proof']['actions'])
        d1=None if f['proof'] is None else risk(f['proof']['actions']);w1=None if f['proof'] is None else width(f['proof']['actions'])
        if d0 is not None:assert d1<=d0 and w1<=w0
        if a['cause']=='CERTIFIED':assert u[-1]==opt
        hc={k:f['hooks'][k]-p['hooks'][k] for k,v0 in f['hooks'].items() if isinstance(v0,int) and k!='worker_mask'}
        plan=[s['plan'] for s in a['steps'] if s['plan'] is not None]
        gs=n.utility(v[a['guard_best']])
        out.append(dict(id=a['id'],stratum=a['stratum'],seed=a['seed'],mode=a['mode'],arm=a['arm'],regret=opt-u[-1],prefix_regret=opt-u[0],guard_regret=opt-gs,gain=u[-1]-u[0],guard_gain=gs-u[-1],flips=sum(flips),better=sum(x>0 for x in gains),worse=sum(x<0 for x in gains),neutral_flip=sum(f and g==0 for f,g in zip(flips,gains)),final_flip=int(p['best']!=f['best']),final_better=int(u[-1]>u[0]),final_worse=int(u[-1]<u[0]),whole_us=a['whole_us'],transport_us=a['transport_us'],plan_us=sum(s['plan_us'] for s in a['steps']),evals=f['result']['evaluator_calls'],optional_evals=f['result']['evaluator_calls']-p['result']['evaluator_calls'],visits=f['result']['root_n'],blocks=len(a['steps']),risk_reduction=None if d0 is None else (d0-d1)/2,width_reduction=None if w0 is None else w0-w1,stop=int(a['cause']=='CERTIFIED'),cap=int(a['cause']=='CAP'),cert_unsolved=int(a['cause']=='CERTIFIED' and f['proof']['interval'][0]!=f['proof']['interval'][1]),exposed=int(bool(hc['local_focus_edges']+hc['local_expand_edges']+hc['local_frontier_edges']+hc['extra_evals']+hc['merge_blends'])),planned=sum(x.get('target') is not None for x in plan if x['mode']!='EXPAND'),frontier_positive_risk=sum(x.get('target') is not None and x['target']['score'][0]>0 for x in plan if x['mode']=='FRONTIER'),frontier_width_only=sum(x.get('target') is not None and x['target']['score'][0]==0 for x in plan if x['mode']=='FRONTIER'),hooks=hc))
    return out,checked,cert
def score():
    roots=[r for r in read('C07_COHORT.json')['rows'] if r['group']=='TEST'];labels=n.oracle(roots);write('C08_LABELS.json',labels);z={x['id']:x['scores'] for x in labels if x['status']=='EXACT'};assert len(z)==72
    out,checks,cert=summarize_records(records('C08_LIVE_RAW.jsonl.gz'),z)
    assert len(out)==72*2*9*3
    by={(x['id'],x['seed'],x['mode'],x['arm']):x for x in out};assert len(by)==len(out)
    rng=random.Random(202611250911);summary={}
    sumkeys=('flips','better','worse','neutral_flip','final_flip','final_better','final_worse','stop','cap','cert_unsolved','exposed','planned','frontier_positive_risk','frontier_width_only')
    meankeys=('regret','prefix_regret','guard_regret','gain','guard_gain','whole_us','transport_us','plan_us','evals','optional_evals','visits','blocks','risk_reduction','width_reduction')
    for st in (*ST,'ALL'):
        ids=[r['id'] for r in roots if st=='ALL' or r['stratum']==st];summary[st]={}
        for mo in MODES:
            arms={};paired={}
            for arm in ARMS:
                a=[by[i,s,mo,arm] for i in ids for s in SEEDS]
                m={k:sum(x[k] for x in a) for k in sumkeys};m.update({k:S.mean(x[k] for x in a if x[k] is not None) if any(x[k] is not None for x in a) else None for k in meankeys})
                m['hooks']={k:sum(x['hooks'][k] for x in a) for k in a[0]['hooks']};m['n']=len(a);arms[arm]=m
            for ref in ('BASE','SAFE','WRITER'):
                for arm in ARMS:
                    if arm==ref:continue
                    dif={i:S.mean(by[i,s,mo,ref]['regret']-by[i,s,mo,arm]['regret'] for s in SEEDS) for i in ids}
                    strata=[[i for i in ids if i.startswith(t+'_')] for t in ST] if st=='ALL' else [ids]
                    boot=sorted(S.mean(dif[i] for group in strata for i in rng.choices(group,k=len(group))) for _ in range(2000))
                    paired[arm+'/'+ref]=dict(delta=S.mean(dif.values()),ci95=[boot[50],boot[1949]],time_mismatch=abs(arms[arm]['whole_us']/arms[ref]['whole_us']-1),nn_mismatch=abs(arms[arm]['evals']/arms[ref]['evals']-1))
            summary[st][mo]=dict(arms=arms,paired=paired)
    host=read('C08_LIVE_HOST.json');s=dict(groups=summary,labels=dict(C.Counter(x['status'] for x in labels)),runs=len(out),checked_action_intervals=checks,certificate_snapshots=cert,raw_sha=host['raw_sha'],host_ok=host['interval']['ok'] and host['max_load1']<=12,host=host,scientific_promotion=False)
    write('C08_SUMMARY.json',s);write_gz('C08_METRICS.json.gz',out)
    for st in (*ST,'ALL'):
        for mo,g in summary[st].items():print(st,mo,[(a,round(v['regret'],5),round(v['gain'],5),round(v['evals'],1),round(v['visits'],1),v['final_better'],v['final_worse'],v['stop'],v['exposed']) for a,v in g['arms'].items()],flush=True)
    print('HOST',s['host_ok'],host['interval'],'LABELS',s['labels'],'INTERVALS',checks,'CERT',cert,flush=True)
if __name__=='__main__':
    lock();{'smoke':smoke,'calibration':calibration,'fit':fit,'live':live,'score':score}[sys.argv[1]]()
