"""Native resident pool experiment; no exact label at runtime."""
import collections as C,gzip,json,math,os,random,statistics as S,subprocess as SP,sys,time
from pathlib import Path
import native as n
from native import D,ROOT,write,read,sha
from population import exclusions
BIN=ROOT.parent/'quartz_fouraxis04_target_20261125/mcts_demo'
ST=('TACTICAL','SELFPLAY','HARD');SEEDS=(17,43)
ARMS=('BASE','ACTIVE_FIXED','STOP_FIXED','RR','BOUND','RR_SAFE','RR_WRITER','RR_TEMPORAL','RR_MERGE')
MODES=('WORK','TIME');FIXED=ARMS[:3];STOP=ARMS[2:]
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
def rb(actions,a):
    iv=dict(actions);return max(0,max((v[1] for k,v in iv.items() if k!=a),default=-1)-iv[a][0])/2
def records(name):
    with gzip.open(D/name,'rt') as f:
        for line in f:yield json.loads(line)
def write_gz(name,obj):
    p=D/name;assert not p.exists()
    with gzip.open(p,'wt',compresslevel=3) as f:json.dump(obj,f,separators=(',',':'),allow_nan=False)
def read_gz(name):
    with gzip.open(D/name,'rt') as f:return json.load(f)
class Server:
    def __init__(self):
        lock();self.p=SP.Popen(['nice','-n','10',str(BIN),'--server'],stdin=SP.PIPE,stdout=SP.PIPE,stderr=SP.DEVNULL,text=True,bufsize=1,env=dict(os.environ,RAYON_NUM_THREADS='2'))
    def send(self,q):
        t=time.perf_counter_ns();self.p.stdin.write(json.dumps(q)+'\n');self.p.stdin.flush();line=self.p.stdout.readline();assert line,('EOF',self.p.poll(),q)
        a=json.loads(line);a['transport_us']=(time.perf_counter_ns()-t)/1000;return a
    def call(self,rows,arm='BASE',mode='WORK',seed=17,target=0,reserve=0):
        q=dict(cmd='research_connect4_fouraxis_pool_cpu',jobs=[dict(id=x['id'],history=x['history']) for x in rows],arm=arm,mode=mode,seed=seed,model_path=str(n.MODEL),model_identity=n.IDENT,target_us=target,reserve_us=reserve)
        a=self.send(q);assert 'error' not in a,a
        assert a['arm']==arm and a['mode']==mode and a['seed']==seed and a['threads']==2 and a['model_identity']==n.IDENT
        assert len(a['rows'])==4 and a['total_blocks']==len(a['trace'])==sum(r['blocks'] for r in a['rows'])
        assert a['drop_total_us']==sum(r['drop_us'] for r in a['rows'])
        assert a['whole_us']>=a['before_final_us']>=a['mandatory_us']
        for i,(r,j) in enumerate(zip(a['rows'],rows)):
            assert r['history']==j['history'] and r['id']==j['id']
            b,z=n.replay(r['history']);legal={k for k in range(7) if z[k]<6}
            p=r['prefix'];v=r['result'];hs=r['hooks'];ph=r['prefix_hooks'];vis=8+8*r['blocks']
            assert p['root_n']==8 and v['root_n']==vis<=512 and p['root_hash']==v['root_hash']
            for x in (p,v):
                assert {m for m,_ in x['candidates']}==legal and x['best'] in legal
                assert [y['move'] for y in x['edges']]==[y[0] for y in x['candidates'][:len(x['edges'])]]
                assert sum(y['n'] for y in x['edges'])==x['root_n'] and abs(sum(y['w'] for y in x['edges'])-x['root_w'])<1e-6
                assert all(y['vl']==0 and abs(y['vvalue'])<1e-6 and abs(y['w'])<=y['n']+1e-6 and y['m2']>=-1e-7 for y in x['edges'])
            assert ph['base_evals']+ph['terminal_leaves']==8 and p['evaluator_calls']==1+ph['base_evals']
            assert hs['base_evals']+hs['terminal_leaves']+hs['graph_solved_skips']==vis-8
            assert v['evaluator_calls']==p['evaluator_calls']+hs['base_evals']+hs['extra_evals'] and hs['extra_evals']==0
            assert r['best'] in legal
            g=r['proof']
            if arm=='BASE':assert g is None and r['best']==v['best']
            else:
                assert g['key']==key(r['history']);d=dict(g['actions']);assert set(d)==legal and all(-1<=lo<=hi<=1 for lo,hi in d.values())
                assert g['interval']==[max(x[0] for x in d.values()),max(x[1] for x in d.values())]
                assert g['allowed']==[m for m,iv in g['actions'] if iv[1]>=g['interval'][0]]
                w=g['witness'];assert (w is None)==all(rb(g['actions'],m)>0 for m in d)
                if w is not None:assert rb(g['actions'],w)==0 and r['best']==w
                if r['reason']=='CERTIFIED':assert arm in STOP and w is not None
                assert r['best'] in g['allowed']
            if arm in ('RR_WRITER','RR_TEMPORAL','RR_MERGE'):assert hs['backup_edges']>=vis-8
            else:assert hs['backup_edges']==hs['ledger_observations']==0
            if arm=='RR_WRITER':assert hs['ledger_observations']==0 and hs['temporal_blends']==hs['merge_blends']==0
            if arm=='RR_TEMPORAL':assert hs['ledger_observations']==hs['backup_edges'] and hs['merge_blends']==0
            if arm=='RR_MERGE':assert hs['ledger_observations']==hs['backup_edges'] and hs['temporal_blends']==0
            assert hs['backup_changed']<=hs['temporal_blends']+hs['merge_blends']<=hs['backup_edges']-hs['terminal_edges']
        if mode=='WORK':
            v=[r['blocks'] for r in a['rows']];C0=sum(max(0,15-r['blocks']) for r in a['rows'] if r['reason']=='CERTIFIED');E=sum(max(0,b-15) for b in v);S0=sum(max(0,15-b) for b in v)
            assert a['account']==dict(credit=C0,extra=E,shortfall=S0,unused=60-sum(v))
            assert S0-E==60-sum(v)>=0
            if arm in FIXED:assert E==0
            if arm in ('BASE','ACTIVE_FIXED'):assert v==[15]*4
            if arm not in FIXED:assert C0==S0
        else:assert a['account'] is None and a['target_us']==target and a['reserve_us']==reserve
        return a
    def close(self):
        self.p.stdin.close();assert self.p.wait(timeout=30)==0
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
def smoke():
    rr=[dict(id='SMOKE'+str(i),history=h) for i,h in enumerate(('121212','112233','1234','1235'))]
    tests=[];par=[];bad=[]
    with Server() as srv:
        for arm in ARMS:tests.append(srv.call(rr,arm))
        base=tests[0]
        for row in base['rows']:
            q=dict(cmd='research_connect4_fouraxis_cpu',history=row['history'],arm='BASE',mode='STANDARD',seed=17,model_path=str(n.MODEL),model_identity=n.IDENT)
            a=srv.send(q);assert 'error' not in a and a['result']==row['prefix'];par.append(a)
        for q in [
            dict(cmd='research_connect4_fouraxis_pool_cpu',jobs=rr,arm='RR',mode='WORK',seed=17,model_path=str(n.MODEL),model_identity=n.IDENT,oracle=[1]),
            dict(cmd='research_connect4_fouraxis_pool_cpu',jobs=[rr[0],rr[0],rr[2],rr[3]],arm='RR',mode='WORK',seed=17,model_path=str(n.MODEL),model_identity=n.IDENT)]:
            a=srv.send(q);assert 'error' in a;bad.append(a)
    stop=next(a for a in tests if a['arm']=='STOP_FIXED');rr0=next(a for a in tests if a['arm']=='RR')
    assert stop['rows'][0]['reason']==stop['rows'][1]['reason']=='CERTIFIED' and stop['rows'][0]['blocks']==stop['rows'][1]['blocks']==0
    assert rr0['account']['extra']>0 and rr0['account']['credit']>=30
    write_gz('C06_SMOKE.json.gz',dict(tests=tests,parity=par,rejected=bad))
    print('SMOKE',[(x['arm'],x['total_blocks'],x['account'],[r['blocks'] for r in x['rows']]) for x in tests],flush=True)
def pools():
    co=read('C07_COHORT.json');rng=random.Random(202611250809);p=[]
    for st in ST:
        for group,quota in [('TRAIN',12),('CAL',8),('TEST',24)]:
            rr=[x for x in co['rows'] if x['stratum']==st and x['group']==group];assert len(rr)==quota,(st,group,len(rr))
            rng.shuffle(rr)
            for i in range(0,len(rr),4):
                p.append(dict(id=f'{st}_{group}_POOL{i//4:02d}',stratum=st,group=group,rows=rr[i:i+4]))
    write('C07_POOLS.json',dict(pools=p,cohort_sha=sha(D/'C07_COHORT.json')))
    print('POOLS',dict(C.Counter(p0['group'] for p0 in p)),flush=True)
def calibration():
    ps=read('C07_POOLS.json')['pools'];train=[p for p in ps if p['group']=='TRAIN'];cal=[p for p in ps if p['group']=='CAL']
    timing=[];checks=[]
    before=snap()
    with Server() as srv:
        srv.call(train[0]['rows'])
        for seed in SEEDS:
            for p in train:
                a=srv.call(p['rows'],seed=seed);a.update(pool=p['id'],stratum=p['stratum']);timing.append(a)
        end=snap()
        for p in cal:
            for arm in ARMS:
                a=srv.call(p['rows'],arm);a.update(pool=p['id'],stratum=p['stratum']);checks.append(a)
    cost={}
    for st in ST:
        x=[a for a in timing if a['stratum']==st];med=S.median(a['whole_us'] for a in x);drop=S.median(a['drop_total_us'] for a in x)
        target=min(40000,max(6000,math.ceil(med)));reserve=min(target-1,math.ceil(drop)+500)
        cost[st]=dict(n=len(x),median_whole_us=med,median_drop_us=drop,target=target,reserve=reserve)
    write_gz('C07_CAL_RAW.json.gz',dict(timing=timing,checks=checks,host=interval(before,end),start=before,end=end))
    write('C07_COST_LOCK.json',dict(cost=cost,raw_sha=sha(D/'C07_CAL_RAW.json.gz'),pools_sha=sha(D/'C07_POOLS.json'),binary_sha=sha(BIN)))
    print('COST',cost,'CAL_POOL_CHECKS',len(checks),flush=True)
def live():
    ps=[p for p in read('C07_POOLS.json')['pools'] if p['group']=='TEST'];m=read('C07_COST_LOCK.json');assert m['pools_sha']==sha(D/'C07_POOLS.json') and m['raw_sha']==sha(D/'C07_CAL_RAW.json.gz') and m['binary_sha']==sha(BIN)
    fpath=D/'C08_LIVE_RAW.jsonl.gz';assert not fpath.exists();count=0;max_load=0
    before=snap()
    with Server() as srv,gzip.open(fpath,'wt',compresslevel=3) as f:
        srv.call(ps[0]['rows'])
        for rep,seed in enumerate(SEEDS):
            for st in ST[::(-1 if rep%2 else 1)]:
                for k,p in enumerate(x for x in ps if x['stratum']==st):
                    for mo in MODES[(k+rep)%2:]+MODES[:(k+rep)%2]:
                        off=(k+rep*3+MODES.index(mo))%len(ARMS)
                        for arm in ARMS[off:]+ARMS[:off]:
                            cfg=m['cost'][st];a=srv.call(p['rows'],arm,mo,seed,cfg['target'] if mo=='TIME' else 0,cfg['reserve'] if mo=='TIME' else 0)
                            a.update(pool=p['id'],stratum=st,host_load1=os.getloadavg()[0]);max_load=max(max_load,a['host_load1'])
                            f.write(json.dumps(a,separators=(',',':'),allow_nan=False)+'\n');count+=1
                print('LIVE',seed,st,count,flush=True)
    end=snap();write('C08_LIVE_HOST.json',dict(start=before,end=end,interval=interval(before,end),max_load1=max_load,runs=count,raw_sha=sha(fpath)))
    print('BLIND',count,sha(fpath),flush=True)
def score():
    ps=[p for p in read('C07_POOLS.json')['pools'] if p['group']=='TEST'];roots=[r for p in ps for r in p['rows']]
    labels=n.oracle(roots);write('C08_LABELS.json',labels);z={x['id']:x['scores'] for x in labels if x['status']=='EXACT'};assert set(z)=={r['id'] for r in roots}
    by={};checks=0;cert=0;stop=0
    for a in records('C08_LIVE_RAW.jsonl.gz'):
        rr=[]
        for r in a['rows']:
            scores=z[r['id']];opt=max(n.utility(q) for q in scores if q!=-1000);v=n.utility(scores[r['best']]);reg=opt-v;g=r['proof']
            if g is not None:
                for j,(l,u) in g['actions']:
                    assert l<=(scores[j]>0)-(scores[j]<0)<=u;checks+=1
                assert reg<=rb(g['actions'],r['best'])
                if g['witness'] is not None:assert reg==0;cert+=1
                if r['reason']=='CERTIFIED':assert reg==0;stop+=1
            rr.append(dict(id=r['id'],regret=reg,blocks=r['blocks'],evals=r['result']['evaluator_calls'],stop=r['reason']=='CERTIFIED',cert=g is not None and g['witness'] is not None,unsolved_cert=g is not None and g['witness'] is not None and g['interval'][0]!=g['interval'][1],reason=r['reason'],hooks=r['hooks']))
        k=(a['pool'],a['seed'],a['mode'],a['arm']);assert k not in by
        by[k]=dict(regret=S.mean(r['regret'] for r in rr),whole_us=a['whole_us'],transport_us=a['transport_us'],blocks=a['total_blocks'],evals=sum(r['evals'] for r in rr),credit=(a['account'] or {}).get('credit',0),extra=(a['account'] or {}).get('extra',0),unused=(a['account'] or {}).get('unused',0),stops=sum(r['stop'] for r in rr),certs=sum(r['cert'] for r in rr),unsolved_cert=sum(r['unsolved_cert'] for r in rr),caps=sum(r['reason']=='CAP' for r in rr),roots=rr)
    assert len(by)==18*2*2*9
    rng=random.Random(202611250810);summary={}
    for st in (*ST,'ALL'):
        ids=[p['id'] for p in ps if st=='ALL' or p['stratum']==st];summary[st]={}
        for mo in MODES:
            arms={};paired={}
            for arm in ARMS:
                rr=[by[i,s,mo,arm] for i in ids for s in SEEDS];roots=[r for x in rr for r in x['roots']]
                m=dict(n_pools=len(ids),n_runs=len(rr),regret=S.mean(x['regret'] for x in rr),whole_us=S.mean(x['whole_us'] for x in rr),transport_us=S.mean(x['transport_us'] for x in rr),mean_blocks=S.mean(x['blocks'] for x in rr),mean_evals=S.mean(x['evals'] for x in rr),credit=sum(x['credit'] for x in rr),extra=sum(x['extra'] for x in rr),unused=sum(x['unused'] for x in rr),stops=sum(x['stops'] for x in rr),certs=sum(x['certs'] for x in rr),unsolved_cert=sum(x['unsolved_cert'] for x in rr),caps=sum(x['caps'] for x in rr),pools_reused=sum(x['extra']>0 for x in rr),roots_extra=sum(r['blocks']>15 for r in roots),hooks={k:sum(r['hooks'][k] for r in roots) for k in roots[0]['hooks'] if k not in ('leaf_paths','worker_mask','b_events')})
                arms[arm]=m
            for ref in ('BASE','ACTIVE_FIXED','STOP_FIXED','RR','RR_WRITER'):
                for arm in ARMS:
                    if arm==ref:continue
                    dif={i:S.mean(by[i,s,mo,ref]['regret']-by[i,s,mo,arm]['regret'] for s in SEEDS) for i in ids}
                    strata=[[i for i in ids if i.startswith(t+'_')] for t in ST] if st=='ALL' else [ids]
                    boot=sorted(S.mean(dif[i] for group in strata for i in rng.choices(group,k=len(group))) for _ in range(2000))
                    paired[arm+'/'+ref]=dict(delta=S.mean(dif.values()),ci95=[boot[50],boot[1949]],time_mismatch=abs(arms[arm]['whole_us']/arms[ref]['whole_us']-1),nn_mismatch=abs(arms[arm]['mean_evals']/arms[ref]['mean_evals']-1))
            summary[st][mo]=dict(arms=arms,paired=paired)
    host=read('C08_LIVE_HOST.json');out=dict(groups=summary,labels=dict(C.Counter(x['status'] for x in labels)),runs=len(by),root_decisions=4*len(by),root_intervals=checks,valid_certificates=cert,certified_stops=stop,raw_sha=host['raw_sha'],host_ok=host['interval']['ok'] and host['max_load1']<=12,host=host,scientific_promotion=False)
    write('C08_SUMMARY.json',out)
    for st in (*ST,'ALL'):
        for mo,g in summary[st].items():print(st,mo,[(a,round(v['regret'],5),round(v['whole_us']),round(v['mean_evals'],1),round(v['mean_blocks'],1),v['credit'],v['extra'],v['stops']) for a,v in g['arms'].items()],flush=True)
    print('HOST',out['host_ok'],host['interval'],'LABELS',out['labels'],'INTERVALS',checks,'CERT',cert,'STOPS',stop,flush=True)
if __name__=='__main__':
    lock();{'smoke':smoke,'pools':pools,'calibration':calibration,'live':live,'score':score}[sys.argv[1]]()
